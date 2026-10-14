# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU compatibility patches for the device-neutral GLM-5 model.

The patches fall into three independent groups:

* vocab-parallel cross entropy avoids boolean assignment that lowers to an
  unsupported NPU nonzero path while preserving TP loss mathematics;
* DTensor truncated-normal initialization avoids rank-dependent scalar
  collectives during parameter materialization;
* the GLM flavor initializers avoid an NPU DTensor scalar reduction.

No GLM architecture is copied here. Each patch targets and validates the
current source-installed TorchTitan implementation.
"""

import logging
import math
import warnings
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


_ORIGINAL_TRUNC_NORMAL = torch.nn.init.trunc_normal_


def _local_vocab_labels(
    labels: torch.Tensor,
    *,
    vocab_start: int,
    local_vocab_size: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global labels to one vocab shard without boolean assignment."""
    safe_labels = torch.where(labels != ignore_index, labels, 0)
    out_of_range = (safe_labels < vocab_start) | (
        safe_labels >= vocab_start + local_vocab_size
    )
    local_labels = torch.where(
        out_of_range,
        torch.zeros_like(safe_labels),
        safe_labels - vocab_start,
    )
    return local_labels, out_of_range


def _npu_safe_loss_parallel_forward(
    ctx,
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    global_vocab_size: int,
    reduction: str = "sum",
) -> torch.Tensor:
    """Compute vocab-parallel CE without NPU boolean-index assignment.

    For TP rank ``r``, logits contain only vocabulary interval
    ``[vocab_start, vocab_end)``. A global MAX and SUM implement stable global
    log-softmax. Only the rank owning each label gathers a nonzero log
    probability; the final SUM broadcasts that value to every TP rank.
    """
    from torchtitan.components.loss import IGNORE_INDEX

    logits_dtype = logits.dtype
    logits = logits.float()
    tp_world_size = dist.get_world_size(tp_group)
    tp_rank = dist.get_rank(tp_group)
    chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
    vocab_start = min(global_vocab_size, chunk_size * tp_rank)
    vocab_end = min(global_vocab_size, vocab_start + chunk_size)
    local_vocab_size = max(0, vocab_end - vocab_start)
    if logits.shape[-1] != local_vocab_size:
        raise ValueError(
            "_LossParallelCrossEntropy expected local vocab size "
            f"{local_vocab_size} for global vocab size {global_vocab_size}, "
            f"got {logits.shape[-1]}."
        )
    if local_vocab_size == 0:
        raise ValueError(
            "_LossParallelCrossEntropy does not support empty vocab shards."
        )

    # Global logsumexp over a sharded vocabulary: first agree on the maximum,
    # then sum exp(logit - max) from every vocabulary shard.
    local_max = torch.amax(logits, dim=-1, keepdim=True)
    local_max = funcol.all_reduce(
        local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
    )
    shifted = logits - local_max
    shifted_sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
    shifted_sumexp = funcol.all_reduce(
        shifted_sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
    )
    log_probs = shifted - torch.log(shifted_sumexp)

    local_labels, out_of_range = _local_vocab_labels(
        labels,
        vocab_start=vocab_start,
        local_vocab_size=local_vocab_size,
        ignore_index=IGNORE_INDEX,
    )
    # Non-owner ranks gather safe index zero and mask the result. Summing over
    # TP therefore leaves exactly the owning rank's target log-probability.
    local_result = torch.gather(log_probs, -1, local_labels.unsqueeze(-1))
    local_result = torch.where(
        out_of_range.unsqueeze(-1),
        torch.zeros_like(local_result),
        local_result,
    )
    local_result = funcol.all_reduce(
        local_result, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
    )

    result = -local_result.squeeze(-1)
    result = torch.where(labels != IGNORE_INDEX, result, 0)
    ctx.save_for_backward(log_probs, labels)
    ctx.logits_dtype = logits_dtype
    ctx.vocab_start = vocab_start
    ctx.local_vocab_size = local_vocab_size
    ctx.reduction = reduction
    if reduction == "none":
        return result
    return result.sum()


def _npu_safe_loss_parallel_backward(
    ctx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, None, None, None, None]:
    """Differentiate vocab-parallel CE as ``softmax - one_hot(label)``.

    ``grad_update`` is -1 only on the vocabulary shard that owns the label and
    0 elsewhere. ``scatter_`` expresses the same update without boolean-index
    assignment, and ignored labels zero the complete local gradient row.
    """
    from torchtitan.components.loss import IGNORE_INDEX

    log_probs, labels = ctx.saved_tensors
    local_labels, out_of_range = _local_vocab_labels(
        labels,
        vocab_start=ctx.vocab_start,
        local_vocab_size=ctx.local_vocab_size,
        ignore_index=IGNORE_INDEX,
    )
    grad_update = out_of_range.to(log_probs.dtype) - 1.0
    grad_input = torch.zeros_like(log_probs).scatter_(
        -1,
        local_labels.unsqueeze(-1),
        grad_update.unsqueeze(-1),
    )
    if ctx.reduction == "none":
        grad_output = grad_output.unsqueeze(-1)
    grad_output = torch.where(
        (labels != IGNORE_INDEX).unsqueeze(-1), grad_output, 0
    )
    grad_logits = (grad_input + torch.exp(log_probs)) * grad_output
    return grad_logits.to(ctx.logits_dtype), None, None, None, None


def _patch_vocab_parallel_loss() -> None:
    """Avoid aclnnNonzeroV2 in TorchTitan's TP cross-entropy on NPU."""
    from torchtitan.components import loss as loss_module

    loss_class = loss_module._LossParallelCrossEntropy
    if getattr(loss_class, "_torchtitanturbo_npu_patched", False):
        return
    loss_class.forward = staticmethod(_npu_safe_loss_parallel_forward)
    loss_class.backward = staticmethod(_npu_safe_loss_parallel_backward)
    loss_class._torchtitanturbo_npu_patched = True


def _npu_safe_trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Initialize an NPU DTensor without a distributed scalar reduction.

    PyTorch 2.14 uses rejection sampling in ``trunc_normal_`` and evaluates
    ``mask.any()`` on every iteration. For a sharded DTensor that scalar
    result requires redistribution. On an Ascend multi-axis device mesh, the
    resulting lazy HCCL subgroup initialization can leave ranks entering
    different communicators. The inverse-CDF construction below samples the
    same truncated-normal distribution using only elementwise DTensor ops.
    """
    if not isinstance(tensor, DTensor):
        return _ORIGINAL_TRUNC_NORMAL(
            tensor,
            mean=mean,
            std=std,
            a=a,
            b=b,
            generator=generator,
        )
    if tensor.is_meta:
        return tensor
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    def norm_cdf(value: float) -> float:
        return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0

    lower = norm_cdf((a - mean) / std)
    upper = norm_cdf((b - mean) / std)
    with torch.no_grad():
        tensor.uniform_(2 * lower - 1, 2 * upper - 1, generator=generator)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def _replace_trunc_normal_initializers(param_init: dict) -> dict:
    """Replace only exact TorchTitan trunc-normal partials, preserving others."""
    converted = {}
    for name, initializer in param_init.items():
        if (
            isinstance(initializer, partial)
            and initializer.func is _ORIGINAL_TRUNC_NORMAL
        ):
            converted[name] = partial(
                _npu_safe_trunc_normal_,
                *initializer.args,
                **(initializer.keywords or {}),
            )
        else:
            converted[name] = initializer
    return converted


def _patch_glm5_param_initializers(glm5_module) -> None:
    glm5_module._LINEAR_INIT = _replace_trunc_normal_initializers(
        glm5_module._LINEAR_INIT
    )
    for name in ("_output_linear_init", "_depth_init", "_depth_experts_init"):
        current_factory = getattr(glm5_module, name)
        if getattr(current_factory, "_torchtitanturbo_npu_init_patched", False):
            continue

        def factory(*args, _factory=current_factory, **kwargs):
            return _replace_trunc_normal_initializers(
                _factory(*args, **kwargs)
            )

        factory._torchtitanturbo_npu_init_patched = True
        setattr(glm5_module, name, factory)


def apply_patch() -> None:
    """Install all GLM NPU contracts once, leaving GPU/CPU imports untouched."""
    from torchtitan.models.glm5 import flavors

    _patch_vocab_parallel_loss()
    if getattr(flavors, "_torchtitanturbo_glm5_npu_patched", False):
        return
    _patch_glm5_param_initializers(flavors)
    flavors._torchtitanturbo_glm5_npu_patched = True
    logger.info("Applied GLM-5 NPU compatibility patches")
