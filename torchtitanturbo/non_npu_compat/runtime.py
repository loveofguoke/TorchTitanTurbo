# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Non-NPU TorchTitan runtime compatibility patches.

These patches are intentionally isolated from the NPU adaptation entry point:

* PP pre-split microbatches bridge TorchTitan's newer calling convention to
  PyTorch builds that predate explicit ``arg_mbs``/``kwarg_mbs`` support.
* Empty-gradient norm placement is a generic TorchTitan device-consistency
  workaround, even though it was first observed during NPU validation.

Revisit and remove or upstream these patches after distributed validation is
complete.
"""

import math

import torch
from torch import distributed as dist
from torch.distributed.pipelining.schedules import PipelineScheduleMulti
from torch.distributed.tensor import DTensor

from torchtitan.tools.logging import logger


def _step_pre_split_microbatches(schedule, **kwargs):
    """Run microbatches that TorchTitan has already split and processed."""
    loss_kwargs = kwargs.pop("loss_kwargs", None)
    if isinstance(schedule, PipelineScheduleMulti):
        if (
            schedule._has_backward
            and schedule._backward_requires_autograd
            and not torch.is_grad_enabled()
        ):
            raise RuntimeError(
                "Pipeline schedule requires gradients for backward computation."
            )
        stages = schedule._stages
    else:
        if schedule._has_backward and not torch.is_grad_enabled():
            raise RuntimeError(
                "Pipeline schedule requires gradients for backward computation."
            )
        stages = [schedule._stage]

    for stage in stages:
        stage.has_backward = schedule._has_backward
        stage.clear_runtime_states()
    original_loss_fn = getattr(schedule, "_loss_fn", None)
    if loss_kwargs and original_loss_fn is not None:
        schedule._loss_fn = lambda output, target: original_loss_fn(
            output,
            target,
            **loss_kwargs,
        )
    device_type = torch.device(stages[0].device).type
    original_fork_rng = torch.random.fork_rng

    def fork_rng_for_stage(*args, **fork_kwargs):
        fork_kwargs.setdefault("device_type", device_type)
        return original_fork_rng(*args, **fork_kwargs)

    torch.random.fork_rng = fork_rng_for_stage
    try:
        schedule._step_microbatches(**kwargs)
    finally:
        torch.random.fork_rng = original_fork_rng
        if loss_kwargs and original_loss_fn is not None:
            schedule._loss_fn = original_loss_fn


def _pp_forward_backward_step(
    self, *, input_dict_mbs, label_mbs, global_valid_tokens
):
    arg_mbs = []
    kwarg_mbs = []
    target_mbs = [] if self.pp_has_last_stage else None
    for input_dict, labels in zip(input_dict_mbs, label_mbs, strict=True):
        inputs, labels, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )
        if self.pp_has_first_stage:
            arg_mbs.append((inputs,))
        kwarg_mbs.append(extra_kwargs)
        if target_mbs is not None:
            target_mbs.append(labels)

    with self.train_context():
        losses = [] if self.pp_has_last_stage else None
        _step_pre_split_microbatches(
            self.pp_schedule,
            arg_mbs=arg_mbs if self.pp_has_first_stage else None,
            kwarg_mbs=kwarg_mbs,
            target_mbs=target_mbs,
            losses=losses,
            loss_kwargs={"global_valid_tokens": global_valid_tokens},
            return_outputs=False,
        )

    if self.pp_has_last_stage:
        assert losses is not None
        return torch.sum(torch.stack(losses)).to(self.device)
    return torch.tensor([-1.0], device=self.device)


def _reduce_pp_norm(total_norm, norm_type, pp_mesh):
    if pp_mesh is None:
        return total_norm
    if math.isinf(norm_type):
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
    else:
        total_norm **= norm_type
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
        total_norm **= 1.0 / norm_type
    return total_norm


@torch.no_grad()
def _clip_grad_norm_with_ep(
    parameters,
    max_norm,
    norm_type,
    error_if_nonfinite,
    foreach,
    pp_mesh,
):
    ep_params, non_ep_params = [], []
    ep_grads, non_ep_grads = [], []
    parameter_device = parameters[0].device if parameters else None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        assert isinstance(parameter, DTensor)
        assert isinstance(parameter.grad, DTensor)
        mesh_axis_names = parameter.device_mesh.mesh_dim_names
        assert mesh_axis_names is not None
        if "ep" in mesh_axis_names:
            ep_params.append(parameter)
            ep_grads.append(parameter.grad)
        else:
            non_ep_params.append(parameter)
            non_ep_grads.append(parameter.grad)

    ep_norm = torch.nn.utils.get_total_norm(
        ep_grads, norm_type, error_if_nonfinite, foreach
    )
    non_ep_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, error_if_nonfinite, foreach
    )
    if isinstance(ep_norm, DTensor):
        ep_norm = ep_norm.full_tensor()
    if isinstance(non_ep_norm, DTensor):
        non_ep_norm = non_ep_norm.full_tensor()

    if ep_grads and not non_ep_grads:
        non_ep_norm = non_ep_norm.to(ep_norm.device)
    elif non_ep_grads and not ep_grads:
        ep_norm = ep_norm.to(non_ep_norm.device)
    elif not ep_grads and not non_ep_grads and parameter_device is not None:
        ep_norm = ep_norm.to(parameter_device)
        non_ep_norm = non_ep_norm.to(parameter_device)

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_norm, non_ep_norm)
    else:
        total_norm = ep_norm**norm_type + non_ep_norm**norm_type
        total_norm **= 1.0 / norm_type
    total_norm = _reduce_pp_norm(total_norm, norm_type, pp_mesh)
    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(
        non_ep_params, max_norm, total_norm, foreach
    )
    return total_norm


@torch.no_grad()
def _clip_grad_norm(
    parameters,
    max_norm,
    norm_type=2.0,
    error_if_nonfinite=False,
    foreach=None,
    pp_mesh=None,
    ep_enabled=False,
):
    parameters = (
        [parameters]
        if isinstance(parameters, torch.Tensor)
        else list(parameters)
    )
    if ep_enabled:
        return _clip_grad_norm_with_ep(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
        )

    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )
    if not grads and parameters:
        total_norm = total_norm.to(parameters[0].device)
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    total_norm = _reduce_pp_norm(total_norm, norm_type, pp_mesh)
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def apply_patch() -> None:
    """Apply temporary non-NPU PP and gradient-norm compatibility patches."""
    import torchtitan.distributed.utils as dist_utils
    from torchtitan.trainer import Trainer

    if getattr(Trainer.pp_forward_backward_step, "_torchtitanturbo_patched", False):
        return
    _pp_forward_backward_step._torchtitanturbo_patched = True
    Trainer.pp_forward_backward_step = _pp_forward_backward_step
    dist_utils.clip_grad_norm_ = _clip_grad_norm
    logger.info("Applied temporary non-NPU TorchTitan runtime compatibility patches")
