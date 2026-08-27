# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Precision-preserving patches for the default NPU Inductor backend."""

from __future__ import annotations

import torch

from torchtitan.tools.logging import logger


def _patch_bf16_narrowing_codegen() -> bool:
    """Preserve graph-visible FP32-to-BF16 rounding in fused pointwise code."""
    from torch_npu._inductor.codegen.triton import NPUTritonKernelOverrides

    current_to_dtype = NPUTritonKernelOverrides.to_dtype
    if getattr(current_to_dtype, "_torchtitanturbo_bf16_rounding", False):
        return False

    original_to_dtype = current_to_dtype

    def to_dtype(
        x,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ):
        # Inductor represents a graph-visible BF16 result as an FP32-to-BF16
        # conversion.  Mapping that conversion back to the FP32 compute type
        # lets a fused consumer observe unrounded values, unlike eager NPU.
        if (
            dtype is torch.bfloat16
            and src_dtype is torch.float32
            and use_compute_types
        ):
            use_compute_types = False
        return original_to_dtype(x, dtype, src_dtype, use_compute_types)

    to_dtype._torchtitanturbo_bf16_rounding = True
    NPUTritonKernelOverrides.to_dtype = staticmethod(to_dtype)
    return True


def _is_low_precision_npu_addmm(match) -> bool:
    inp = match.kwargs.get("inp")
    value = inp.meta.get("val") if isinstance(inp, torch.fx.Node) else None
    return bool(
        isinstance(value, torch.Tensor)
        and value.device.type == "npu"
        and value.dtype is torch.bfloat16
    )


def _patch_low_precision_addmm_fusion() -> int:
    """Keep the BF16 mm output rounding before a residual add on NPU."""
    from torch._inductor.fx_passes import post_grad

    patched = 0
    for entries in post_grad.pass_patterns[2].patterns.values():
        for entry in entries:
            handler = getattr(entry, "handler", None)
            if not (
                getattr(handler, "__name__", None) == "addmm"
                and getattr(handler, "__module__", None) == post_grad.__name__
            ):
                continue

            original_extra_check = entry.extra_check
            if getattr(
                original_extra_check,
                "_torchtitanturbo_low_precision_npu_addmm",
                False,
            ):
                continue

            def extra_check(match, original_check=original_extra_check):
                if _is_low_precision_npu_addmm(match):
                    return False
                return original_check(match)

            extra_check._torchtitanturbo_low_precision_npu_addmm = True
            entry.extra_check = extra_check
            patched += 1
    return patched


def apply_patch() -> None:
    """Install idempotent precision guards for the default NPU backend."""
    narrowing_patched = _patch_bf16_narrowing_codegen()
    addmm_patterns = _patch_low_precision_addmm_fusion()
    if narrowing_patched or addmm_patterns:
        logger.info(
            "Patched NPU Inductor BF16 rounding and %d addmm fusion patterns",
            addmm_patterns,
        )
