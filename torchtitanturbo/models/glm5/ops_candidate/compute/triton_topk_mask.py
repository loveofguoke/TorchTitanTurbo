# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.

"""Unregistered Triton-Ascend prototype for an INT32 top-k additive mask."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_mask_kernel(
    output_ptr,
    numel: tl.constexpr,
    negative_value: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    tl.store(output_ptr + offsets, negative_value, mask=offsets < numel)


@triton.jit
def _scatter_selected_mask_kernel(
    indices_ptr,
    source_mask_ptr,
    output_ptr,
    num_entries: tl.constexpr,
    num_keys: tl.constexpr,
    topk: tl.constexpr,
    block_size: tl.constexpr,
):
    entries = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid_entry = entries < num_entries
    query = entries // topk
    key = tl.load(indices_ptr + entries, mask=valid_entry, other=-1)
    valid_key = valid_entry & (key >= 0) & (key < num_keys)
    source_offset = query * num_keys + key
    value = tl.load(source_mask_ptr + source_offset, mask=valid_key, other=0.0)
    tl.store(output_ptr + source_offset, value, mask=valid_key)


@torch.no_grad()
def triton_topk_additive_mask(
    topk_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    negative_value: float = float("-inf"),
) -> torch.Tensor:
    """Build a `[Q, K]` mask without an INT64 CANN scatter.

    This candidate consumes already selected INT32 indices. It does not fuse
    the TopK selection itself and is intentionally not registered as a GLM-5
    override. The NPU probe must verify Triton-Ascend code generation, packed
    mask semantics, duplicate-free indices, and graph capture before use.
    """

    if topk_indices.device.type != "npu" or attention_mask.device.type != "npu":
        raise ValueError("the candidate requires NPU tensors")
    if topk_indices.dtype != torch.int32:
        raise TypeError("topk_indices must use int32")
    if topk_indices.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("topk_indices and attention_mask must be two-dimensional")
    if topk_indices.shape[0] != attention_mask.shape[0]:
        raise ValueError("query dimensions must match")
    if topk_indices.shape[0] == 0 or topk_indices.shape[1] == 0:
        raise ValueError("query count and top-k must be positive")
    if attention_mask.shape[1] == 0:
        raise ValueError("key count must be positive")
    if topk_indices.shape[1] > attention_mask.shape[1]:
        raise ValueError("top-k cannot exceed the key count")
    if not attention_mask.is_floating_point():
        raise TypeError("attention_mask must use a floating-point dtype")
    if attention_mask.requires_grad:
        raise ValueError("attention_mask must not require gradients")
    if not topk_indices.is_contiguous() or not attention_mask.is_contiguous():
        raise ValueError("candidate tensors must be contiguous")

    output = torch.empty_like(attention_mask)
    numel = output.numel()
    fill_block = 256
    _fill_mask_kernel[(triton.cdiv(numel, fill_block),)](
        output,
        numel=numel,
        negative_value=negative_value,
        block_size=fill_block,
    )

    num_entries = topk_indices.numel()
    scatter_block = 256
    _scatter_selected_mask_kernel[(triton.cdiv(num_entries, scatter_block),)](
        topk_indices,
        attention_mask,
        output,
        num_entries=num_entries,
        num_keys=attention_mask.shape[1],
        topk=topk_indices.shape[1],
        block_size=scatter_block,
    )
    return output


__all__ = ["triton_topk_additive_mask"]
