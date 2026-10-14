# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Opt-in Ascend graph-mode compatibility patches.

Every patch is gated by a ``TORCHTITAN_*`` environment variable. Importing
TorchTitanTurbo for ordinary eager training therefore preserves its existing
behavior.

The patches address different compiler stages and must not be conflated:

* autotuner guards change how generated Triton candidates are benchmarked;
* safe grouped MM gives Dynamo/AOT a traceable custom-op schema, fake kernel,
  and backward for empty-expert cases;
* DTensor strategy registration teaches distributed propagation about complex
  pointwise values;
* PP metadata P2P replaces object communication with tensor communication;
* NPUGraph policies decide which captured FX graphs are eligible for replay.

Each feature is default-off because a graph workaround can affect compilation
coverage or launch behavior even when its eager mathematics is equivalent.
"""

from __future__ import annotations

import inspect
import logging
import operator
import os
import pickle
from datetime import timedelta
from typing import Callable

import torch

logger = logging.getLogger(__name__)


_COMPLEX_STRATEGY_REGISTERED = False


def _pipeline_stage_precompile_shapes() -> tuple[tuple[int, int], ...]:
    value = os.environ.get("TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE")
    if value in (None, ""):
        return ()
    try:
        shapes = tuple(
            tuple(int(item.strip()) for item in shape.split(","))
            for shape in value.split(";")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE must be "
            "NUM_TOKENS,HIDDEN_DIM[;NUM_TOKENS,HIDDEN_DIM...]"
        ) from error
    if any(len(shape) != 2 or any(dim <= 0 for dim in shape) for shape in shapes):
        raise ValueError(
            "TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE must contain positive "
            "two-dimensional shapes"
        )
    return shapes


def _pipeline_stage_precompile_dtype() -> torch.dtype:
    value = os.environ.get("TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_DTYPE")
    if value in (None, ""):
        raise ValueError(
            "TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_DTYPE is required when "
            "TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE is enabled"
        )
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(
            "TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_DTYPE must name a floating-point "
            "torch dtype"
        )
    return dtype


def _floating_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value] if value.is_floating_point() and value.requires_grad else []
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _floating_tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _floating_tensors(item)]
    return []


def _clone_precompile_tensor(tensor: torch.Tensor) -> torch.Tensor:
    clone = tensor.detach().clone()
    if tensor.is_floating_point() and tensor.requires_grad:
        clone.requires_grad_(True)
    return clone


def _precompile_pipeline_stage(
    stage,
    args: tuple[torch.Tensor, ...] | object | None,
    kwargs: dict[str, object] | None,
    shapes: tuple[tuple[int, int], ...],
    activation_dtype: torch.dtype,
) -> None:
    if getattr(stage, "_torchtitanturbo_stage_precompiled", False):
        return

    if isinstance(args, tuple) and all(
        isinstance(item, torch.Tensor) for item in args
    ):
        root_args = tuple(_clone_precompile_tensor(item) for item in args)
        full_num_tokens = root_args[0].shape[0]
        input_sets = [root_args]
        for num_tokens, _ in shapes:
            if num_tokens == full_num_tokens:
                continue
            input_sets.append(
                tuple(
                    item[:num_tokens].detach().clone()
                    if item.ndim > 0 and item.shape[0] == full_num_tokens
                    else item.detach().clone()
                    for item in root_args
                )
            )
    else:
        input_sets = [
            (
                torch.empty(
                    shape,
                    device=stage.device,
                    dtype=activation_dtype,
                    requires_grad=True,
                ),
            )
            for shape in shapes
        ]

    buffer_backup = [
        (buffer, buffer.detach().clone())
        for buffer in stage.submod.buffers(recurse=True)
    ]
    parameter_grad_backup = [
        (
            parameter,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in stage.submod.parameters()
    ]
    cpu_rng_state = torch.get_rng_state()
    device_module = getattr(torch, stage.device.type, None)
    device_rng_state = (
        device_module.get_rng_state(stage.device)
        if device_module is not None
        and hasattr(device_module, "get_rng_state")
        and stage.device.type != "cpu"
        else None
    )
    try:
        for precompile_args in input_sets:
            with torch.enable_grad():
                outputs = stage.submod(*precompile_args, **(kwargs or {}))
                differentiable = _floating_tensors(outputs)
                if differentiable:
                    differentiable_inputs = [
                        tensor
                        for tensor in precompile_args
                        if tensor.is_floating_point() and tensor.requires_grad
                    ]
                    if differentiable_inputs:
                        torch.autograd.grad(
                            differentiable,
                            differentiable_inputs,
                            grad_outputs=[
                                torch.ones_like(tensor) for tensor in differentiable
                            ],
                            allow_unused=True,
                        )

                    outputs = stage.submod(*precompile_args, **(kwargs or {}))
                    differentiable = _floating_tensors(outputs)
                    torch.autograd.backward(
                        differentiable,
                        [torch.ones_like(tensor) for tensor in differentiable],
                    )
            for parameter, _ in parameter_grad_backup:
                parameter.grad = None
    finally:
        for parameter, saved_grad in parameter_grad_backup:
            parameter.grad = saved_grad
        with torch.no_grad():
            for buffer, saved in buffer_backup:
                buffer.copy_(saved)
        torch.set_rng_state(cpu_rng_state)
        if device_rng_state is not None:
            device_module.set_rng_state(device_rng_state, stage.device)

    stage._torchtitanturbo_stage_precompiled = True
    logger.info(
        "Precompiled pipeline stage %s with captured real inputs before schedule P2P",
        stage.stage_index,
    )


def _wait_for_pipeline_stage_precompile(stage) -> None:
    if stage.num_stages != stage.group_size:
        raise RuntimeError(
            "Pipeline stage precompile currently requires one stage per PP rank"
        )
    from torch.distributed.distributed_c10d import _get_default_store

    store = _get_default_store()
    prefix = "torchtitanturbo/pipeline_stage_precompile"
    store.set(f"{prefix}/{stage.stage_index}", b"ready")
    store.wait(
        [f"{prefix}/{stage_index}" for stage_index in range(stage.num_stages)],
        timedelta(minutes=30),
    )


def _install_first_all_reduce_host_barrier() -> None:
    """Align ranks before each compiled-training all-reduce.

    Ascend Inductor autotuning can finish at different times on different ranks.
    Entering HCCL on a fast rank while another rank is still compiling consumes
    the process-group watchdog budget before the collective can be formed. The
    rendezvous store is control-plane only; tensor transport and reduction remain
    the original PyTorch/HCCL all-reduce.
    """

    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_store

    original = dist.all_reduce
    if getattr(original, "_torchtitanturbo_first_collective_barrier", False):
        return

    group_sequences: dict[tuple[int, ...], int] = {}

    def all_reduce(*args, **kwargs):
        if dist.is_initialized() and dist.get_world_size() > 1:
            group = kwargs.get("group")
            if group is None and len(args) > 2:
                group = args[2]
            group_ranks = tuple(dist.get_process_group_ranks(group))
            sequence = group_sequences.get(group_ranks, 0)
            store = _get_default_store()
            group_key = "-".join(str(peer) for peer in group_ranks)
            prefix = (
                f"torchtitanturbo/all_reduce_ready/{group_key}/{sequence}"
            )
            rank = dist.get_rank()
            store.set(f"{prefix}/{rank}", b"ready")
            store.wait(
                [f"{prefix}/{peer}" for peer in group_ranks],
                timedelta(minutes=30),
            )
            if sequence == 0:
                logger.info(
                    "All process-group ranks %s reached their first training "
                    "all-reduce",
                    group_ranks,
                )
            group_sequences[group_ranks] = sequence + 1
        return original(*args, **kwargs)

    all_reduce._torchtitanturbo_first_collective_barrier = True
    dist.all_reduce = all_reduce
    logger.info("Enabled host-store ordering barriers for training all-reduce calls")


def _enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value not in (None, "0", "1"):
        raise ValueError(f"{name} must be 0 or 1; got {value!r}")
    return value == "1"


def _grouped_mm_weight_grad(
    B_t: torch.Tensor,
    expert_ids: torch.Tensor,
    row_grad_B: torch.Tensor,
) -> torch.Tensor:
    grad_B = torch.zeros_like(B_t)
    grad_B.index_add_(0, expert_ids, row_grad_B)
    return grad_B


def _install_vetted_pointwise_autotune() -> None:
    import torch_npu._inductor.runtime.triton_heuristics as runtime

    cls = runtime.NPUCachingAutotuner
    if getattr(
        cls._bench_with_launch_args,
        "_torchtitanturbo_vetted_pointwise_autotune",
        False,
    ):
        return

    signature = inspect.signature(cls._bench_with_launch_args)
    expected_parameters = ("self", "launcher", "launch_args", "reset_args", "kwargs")
    if tuple(signature.parameters) != expected_parameters:
        raise RuntimeError(
            "Unsupported torch_npu NPUCachingAutotuner._bench_with_launch_args "
            f"signature: {signature}"
        )

    def bench_with_launch_args(self, launcher, launch_args, reset_args, **kwargs):
        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        def kernel_call():
            cloned_args, cloned_kwargs = self.clone_args(*launch_args, **kwargs)
            self.reset_to_zero_args(*reset_args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                stream=stream,
            )

        if self.inductor_meta.get(
            "profile_bandwidth_with_do_bench_using_profiling", False
        ):
            return runtime.do_bench_using_profiling_npu(kernel_call, rep=1)

        is_vetted_pointwise = (
            self.heuristic_type == runtime.HeuristicType.POINTWISE
        )
        return runtime.benchmarker.benchmark_gpu(
            kernel_call,
            rep=1,
            device_type="npu",
            is_vetted_benchmarking=is_vetted_pointwise,
        )

    bench_with_launch_args._torchtitanturbo_vetted_pointwise_autotune = True
    cls._bench_with_launch_args = bench_with_launch_args
    logger.info("Enabled vetted NPU pointwise autotuning")


def _install_zero_numel_triton_guard() -> None:
    import torch_npu._inductor.runtime.triton_heuristics as runtime

    def has_zero_numel(self, args) -> bool:
        for name, value in zip(self.fn.arg_names, args):
            if not name.endswith("numel") or name.startswith("r"):
                continue
            try:
                if operator.index(value) == 0:
                    logger.info(
                        "Skipping zero-numel Triton kernel %s (%s=0)",
                        self.get_fn_name(),
                        name,
                    )
                    return True
            except TypeError:
                continue
        return False

    def patch(cls) -> None:
        if getattr(cls.run, "_torchtitanturbo_zero_numel_guard", False):
            return
        original_run = cls.run

        def run(self, *args, stream, benchmark_run=False, **kwargs):
            if has_zero_numel(self, args):
                return None
            return original_run(
                self,
                *args,
                stream=stream,
                benchmark_run=benchmark_run,
                **kwargs,
            )

        run._torchtitanturbo_zero_numel_guard = True
        cls.run = run

    patch(runtime.NPUCachingAutotuner)
    patch(runtime.NPUSymbolicGroupedAutotuner)
    logger.info("Enabled zero-numel Triton launch guard")


def _install_safe_empty_grouped_mm() -> None:
    """Register a compile-visible grouped MM that tolerates empty experts.

    MoE routing is data dependent: a legal step may send zero tokens to one or
    more experts. Padding each empty segment to one row gives the backend valid
    non-empty offsets; positions for real rows are remembered so padding is
    removed after compute. Fake and autograd registrations let Dynamo/AOT reason
    about the op without executing device code during tracing.
    """
    from torchtitan.models.common.linear import GroupedLinear

    namespace = "torchtitanturbo_graph"
    try:
        safe_grouped_mm = torch.ops.torchtitanturbo_graph.safe_grouped_mm.default
    except (AttributeError, RuntimeError):

        @torch.library.custom_op(
            f"{namespace}::safe_grouped_mm", mutates_args=()
        )
        def safe_grouped_mm(
            A: torch.Tensor,
            B_t: torch.Tensor,
            offs: torch.Tensor,
        ) -> torch.Tensor:
            group_sizes = torch.diff(torch.cat((offs.new_zeros((1,)), offs)))
            if bool(torch.all(group_sizes > 0).item()):
                return torch._grouped_mm(A, B_t, offs=offs)
            if A.shape[0] == 0:
                return A.new_empty((0, B_t.shape[-1]))
            padded_sizes = torch.clamp_min(group_sizes, 1)
            padded_offs = torch.cumsum(padded_sizes, 0, dtype=torch.int32)
            expert_ids = torch.repeat_interleave(
                torch.arange(B_t.shape[0], device=A.device), group_sizes
            )
            shifts = torch.cumsum((group_sizes == 0).to(torch.int64), 0)
            shifts = shifts - (group_sizes == 0).to(torch.int64)
            positions = torch.arange(A.shape[0], device=A.device)
            positions = positions + shifts.index_select(0, expert_ids)
            padded_A = A.new_zeros((padded_offs[-1].item(), A.shape[-1]))
            padded_A.index_copy_(0, positions, A)
            output = torch._grouped_mm(padded_A, B_t, offs=padded_offs)
            return output.index_select(0, positions)

        @safe_grouped_mm.register_fake
        def safe_grouped_mm_fake(
            A: torch.Tensor,
            B_t: torch.Tensor,
            offs: torch.Tensor,
        ) -> torch.Tensor:
            del offs
            return A.new_empty((A.shape[0], B_t.shape[-1]))

        @torch.library.custom_op(
            f"{namespace}::safe_grouped_mm_backward", mutates_args=()
        )
        def backward_op(
            grad_output: torch.Tensor,
            A: torch.Tensor,
            B_t: torch.Tensor,
            offs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if A.shape[0] == 0:
                return torch.zeros_like(A), torch.zeros_like(B_t)
            sizes = torch.diff(torch.cat((offs.new_zeros((1,)), offs)))
            expert_ids = torch.repeat_interleave(
                torch.arange(B_t.shape[0], device=A.device), sizes
            )
            row_weights = B_t.index_select(0, expert_ids)
            grad_A = torch.bmm(
                grad_output.unsqueeze(1), row_weights.transpose(-2, -1)
            ).squeeze(1)
            row_grad_B = A.unsqueeze(-1) * grad_output.unsqueeze(-2)
            grad_B = _grouped_mm_weight_grad(B_t, expert_ids, row_grad_B)
            return grad_A, grad_B

        @backward_op.register_fake
        def backward_op_fake(
            grad_output: torch.Tensor,
            A: torch.Tensor,
            B_t: torch.Tensor,
            offs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del grad_output, offs
            return torch.empty_like(A), torch.empty_like(B_t)

        def setup_context(ctx, inputs, output):
            del output
            ctx.save_for_backward(*inputs)

        def backward(ctx, grad_output):
            A, B_t, offs = ctx.saved_tensors
            grad_A, grad_B = backward_op(grad_output, A, B_t, offs)
            return grad_A, grad_B, None

        safe_grouped_mm.register_autograd(backward, setup_context=setup_context)

    if not getattr(GroupedLinear, "_torchtitanturbo_empty_gmm", False):

        def grouped_mm(self, *, A, B_t, offs):
            del self
            return safe_grouped_mm(A, B_t, offs)

        GroupedLinear._grouped_mm = grouped_mm
        GroupedLinear._torchtitanturbo_empty_gmm = True
    logger.info("Enabled empty-expert grouped-mm compatibility")


def _register_complex_dtensor_strategy() -> None:
    global _COMPLEX_STRATEGY_REGISTERED
    if _COMPLEX_STRATEGY_REGISTERED:
        return
    from torch.distributed.tensor._ops._pointwise_ops import (
        _register_single_dim_pointwise,
    )

    try:
        _register_single_dim_pointwise(torch.ops.aten.complex.default)
    except (AssertionError, RuntimeError) as error:
        if "already" not in str(error).lower():
            raise
    _COMPLEX_STRATEGY_REGISTERED = True
    logger.info("Registered aten.complex DTensor pointwise strategy")


def _install_pipeline_metadata_p2p() -> None:
    import torch.distributed as dist
    from torch.distributed.pipelining.stage import PipelineStage

    if getattr(PipelineStage, "_torchtitanturbo_meta_p2p", False):
        return

    def recv_meta(self, src_stage):
        objects = [None]
        dist.recv_object_list(
            objects,
            src=self._resolve_peer_global_rank(src_stage),
            group=self.group,
            device=self.device,
            use_batch=False,
        )
        return objects[0]

    def send_meta(self, meta, dst_stage):
        dist.send_object_list(
            [meta],
            dst=self._resolve_peer_global_rank(dst_stage),
            group=self.group,
            device=self.device,
            use_batch=False,
        )

    PipelineStage._recv_meta = recv_meta
    PipelineStage._send_meta = send_meta
    PipelineStage._torchtitanturbo_meta_p2p = True
    logger.info("Enabled non-batched PipelineStage metadata P2P")


def _install_pipeline_metadata_force_eager() -> None:
    """Keep PyTorch's dynamic metadata protocol out of compiled regions."""

    from torch.distributed.pipelining.stage import PipelineStage

    if getattr(PipelineStage, "_torchtitanturbo_metadata_force_eager", False):
        return
    original_forward = PipelineStage._forward_metadata_inference
    original_backward = PipelineStage._backward_metadata_inference
    precompile_real_inputs = _enabled(
        "TORCHTITAN_PIPELINE_REAL_INPUT_PRECOMPILE"
    )

    def forward_metadata_inference(self, args, kwargs=None, has_backward=False):
        captured = {}

        def capture_inputs(module, hook_args, hook_kwargs):
            captured["args"] = tuple(
                _clone_precompile_tensor(item)
                if isinstance(item, torch.Tensor)
                else item
                for item in hook_args
            )
            captured["kwargs"] = {
                key: _clone_precompile_tensor(value)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in hook_kwargs.items()
            }

        handle = None
        if precompile_real_inputs:
            handle = self.submod.register_forward_pre_hook(
                capture_inputs, with_kwargs=True
            )
        try:
            with torch.compiler.set_stance("force_eager"):
                result = original_forward(self, args, kwargs, has_backward)
        finally:
            if handle is not None:
                handle.remove()

        if precompile_real_inputs:
            if "args" not in captured:
                raise RuntimeError(
                    "Pipeline metadata inference did not execute the stage; "
                    "real-input precompile cannot continue"
                )
            _precompile_pipeline_stage(
                self,
                captured["args"],
                captured["kwargs"],
                (),
                torch.float32,
            )
            _wait_for_pipeline_stage_precompile(self)
        return result

    def backward_metadata_inference(
        self,
        loss_fn=None,
        target=None,
        received_grad_meta=None,
        loss_kwargs=None,
    ):
        with torch.compiler.set_stance("force_eager"):
            return original_backward(
                self,
                loss_fn=loss_fn,
                target=target,
                received_grad_meta=received_grad_meta,
                loss_kwargs=loss_kwargs,
            )

    forward_metadata_inference._torchtitanturbo_force_eager = True
    backward_metadata_inference._torchtitanturbo_force_eager = True
    PipelineStage._forward_metadata_inference = forward_metadata_inference
    PipelineStage._backward_metadata_inference = backward_metadata_inference
    PipelineStage._torchtitanturbo_metadata_force_eager = True
    logger.info("Enabled eager execution for dynamic pipeline metadata inference")


def _install_pipeline_metadata_store_transport() -> None:
    from torch.distributed.distributed_c10d import _get_default_store
    from torch.distributed.pipelining.stage import PipelineStage

    if getattr(PipelineStage, "_torchtitanturbo_meta_store", False):
        return

    def next_sequence(self, direction: str, peer_stage: int) -> int:
        attribute = f"_torchtitanturbo_meta_{direction}_sequence"
        sequences = getattr(self, attribute, None)
        if sequences is None:
            sequences = {}
            setattr(self, attribute, sequences)
        sequence = sequences.get(peer_stage, 0)
        sequences[peer_stage] = sequence + 1
        return sequence

    def key(self, src_stage: int, dst_stage: int, sequence: int) -> str:
        src_rank = self._resolve_peer_global_rank(src_stage)
        dst_rank = self._resolve_peer_global_rank(dst_stage)
        return (
            "torchtitanturbo/pipeline_metadata/"
            f"{src_rank}/{dst_rank}/{src_stage}/{dst_stage}/{sequence}"
        )

    def recv_meta(self, src_stage):
        sequence = next_sequence(self, "recv", src_stage)
        payload = _get_default_store().get(
            key(self, src_stage, self.stage_index, sequence)
        )
        return pickle.loads(payload)

    def send_meta(self, meta, dst_stage):
        sequence = next_sequence(self, "send", dst_stage)
        _get_default_store().set(
            key(self, self.stage_index, dst_stage, sequence),
            pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL),
        )

    PipelineStage._recv_meta = recv_meta
    PipelineStage._send_meta = send_meta
    PipelineStage._torchtitanturbo_meta_store = True
    logger.info("Enabled host-store PipelineStage metadata transport")


def _install_pipeline_stage_precompile() -> None:
    from torch.distributed.pipelining.stage import PipelineStage

    shapes = _pipeline_stage_precompile_shapes()
    if not shapes:
        return
    activation_dtype = _pipeline_stage_precompile_dtype()
    if getattr(
        PipelineStage._forward_metadata_inference,
        "_torchtitanturbo_stage_precompile",
        False,
    ):
        return

    original = PipelineStage._forward_metadata_inference

    def forward_metadata_inference(self, args, kwargs=None, has_backward=False):
        _precompile_pipeline_stage(self, args, kwargs, shapes, activation_dtype)
        _wait_for_pipeline_stage_precompile(self)
        return original(self, args, kwargs, has_backward)

    forward_metadata_inference._torchtitanturbo_stage_precompile = True
    PipelineStage._forward_metadata_inference = forward_metadata_inference
    logger.info(
        "Enabled concurrent pipeline stage precompile with input shape %s and dtype %s",
        shapes,
        activation_dtype,
    )


def _install_npugraph_skip_policy() -> None:
    skip_all = _enabled("TORCHTITAN_NPUGRAPH_SKIP_ALL")
    unsafe_ops = {
        item.strip()
        for item in os.environ.get("TORCHTITAN_NPUGRAPH_UNSAFE_OPS", "").split(",")
        if item.strip()
    }
    if not skip_all and not unsafe_ops:
        return
    import torch_npu.utils._graph_tree as backend
    from torch._inductor.cudagraph_utils import format_default_skip_message

    if getattr(backend.check_for_skip, "_torchtitanturbo_skip_policy", False):
        return
    original = backend.check_for_skip

    def check_for_skip(graph_module, num_fixed):
        if skip_all:
            logger.info("NPUGraph replay disabled by compatibility profile")
            return format_default_skip_message(
                "NPUGraph replay disabled by compatibility profile"
            )
        targets = {
            str(node.target)
            for module in graph_module.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
            if node.op == "call_function"
        }
        matched = sorted(targets & unsafe_ops)
        if matched:
            logger.info(
                "NPUGraph replay skipped for incompatible ops: %s",
                ",".join(matched),
            )
            return format_default_skip_message(
                f"configured incompatible op ({matched[0]})"
            )
        return original(graph_module, num_fixed)

    check_for_skip._torchtitanturbo_skip_policy = True
    backend.check_for_skip = check_for_skip
    logger.info("Enabled NPUGraph compatibility skip policy")


def _install_npugraph_capture_mode() -> None:
    mode = os.environ.get("TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE")
    if mode in (None, ""):
        return
    if mode not in ("global", "thread_local", "relaxed"):
        raise ValueError(
            "TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE must be global, "
            f"thread_local, or relaxed; got {mode!r}"
        )
    import torch_npu

    if getattr(torch.npu.graph, "_torchtitanturbo_capture_mode", False):
        return
    original = torch.npu.graph

    def graph(*args, **kwargs):
        kwargs["capture_error_mode"] = mode
        return original(*args, **kwargs)

    graph._torchtitanturbo_capture_mode = True
    torch.npu.graph = graph
    torch_npu.npu.graph = graph
    logger.info("Enabled NPUGraph capture error mode %s", mode)


def apply_graph_compat_patches() -> None:
    """Apply only graph compatibility features explicitly requested by env."""

    patches: tuple[tuple[str, Callable[[], None]], ...] = (
        (
            "TORCHTITAN_VETTED_POINTWISE_AUTOTUNE",
            _install_vetted_pointwise_autotune,
        ),
        ("TORCHTITAN_SAFE_ZERO_NUMEL_TRITON", _install_zero_numel_triton_guard),
        ("TORCHTITAN_SAFE_EMPTY_GROUPED_MM", _install_safe_empty_grouped_mm),
        (
            "TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY",
            _register_complex_dtensor_strategy,
        ),
    )
    for variable, patch in patches:
        if _enabled(variable):
            patch()
    if _enabled("TORCHTITAN_PIPELINE_METADATA_FORCE_EAGER"):
        _install_pipeline_metadata_force_eager()
    if _enabled("TORCHTITAN_FIRST_ALL_REDUCE_HOST_BARRIER"):
        _install_first_all_reduce_host_barrier()
    pipeline_batch = os.environ.get("TORCHTITAN_PIPELINE_META_USE_BATCH")
    if pipeline_batch not in (None, "0", "1"):
        raise ValueError(
            "TORCHTITAN_PIPELINE_META_USE_BATCH must be 0 or 1; "
            f"got {pipeline_batch!r}"
        )
    metadata_transport = os.environ.get(
        "TORCHTITAN_PIPELINE_META_TRANSPORT", "hccl"
    )
    if metadata_transport not in {"hccl", "store"}:
        raise ValueError(
            "TORCHTITAN_PIPELINE_META_TRANSPORT must be 'hccl' or 'store'"
        )
    if metadata_transport == "store":
        _install_pipeline_metadata_store_transport()
    elif pipeline_batch == "0":
        _install_pipeline_metadata_p2p()
    _install_pipeline_stage_precompile()
    _install_npugraph_skip_policy()
    _install_npugraph_capture_mode()
