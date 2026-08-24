# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU-safe deterministic scatter-add used by GLM-5 MoE dispatch."""

import torch


@torch.library.custom_op(
    "torchtitanturbo::glm5_npu_deterministic_scatter_add",
    mutates_args=(),
)
def npu_deterministic_scatter_add(
    out: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor:
    """Reduce repeated rows without the unstable NPU scatter/index-add path."""
    if index.ndim != 2 or src.ndim != 2 or index.shape != src.shape:
        raise ValueError(
            "GLM-5 NPU deterministic_scatter_add expects matching 2D index/src"
        )
    if src.shape[0] == 0:
        return out.clone()

    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        row_index = index[:, 0]
        order = torch.argsort(row_index, stable=True)
        unique_rows = torch.unique(row_index, sorted=True)
        if src.shape[0] % unique_rows.shape[0] != 0:
            raise ValueError("GLM-5 NPU scatter rows must repeat uniformly")
        repeat_count = src.shape[0] // unique_rows.shape[0]
        reduced = src[order].reshape(
            unique_rows.shape[0], repeat_count, src.shape[1]
        ).sum(dim=1)
        updated_rows = out.index_select(dim=0, index=unique_rows) + reduced
        result = out.clone()
        return result.index_copy_(dim=0, index=unique_rows, source=updated_rows)
    finally:
        torch.use_deterministic_algorithms(
            previous,
            warn_only=previous_warn_only,
        )


@npu_deterministic_scatter_add.register_fake
def _fake_npu_deterministic_scatter_add(out, index, src):
    return torch.empty_like(out)


def _setup_context(ctx, inputs, output):
    del output
    _, index, _ = inputs
    ctx.save_for_backward(index)


def _backward(ctx, grad_output):
    (index,) = ctx.saved_tensors
    grad_src = torch.gather(grad_output, dim=0, index=index)
    return grad_output, None, grad_src


npu_deterministic_scatter_add.register_autograd(
    _backward,
    setup_context=_setup_context,
)


def apply_patch() -> None:
    """Route GLM-5 token combine through the Turbo-owned NPU implementation."""
    import torchtitan.models.common.token_dispatcher as token_dispatcher

    current = token_dispatcher.deterministic_scatter_add
    if getattr(current, "_torchtitanturbo_glm5_npu_patched", False):
        return

    npu_deterministic_scatter_add._torchtitanturbo_glm5_npu_patched = True
    token_dispatcher.deterministic_scatter_add = npu_deterministic_scatter_add
