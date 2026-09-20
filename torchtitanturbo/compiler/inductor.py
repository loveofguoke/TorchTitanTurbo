# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Precision-preserving patches for the default NPU Inductor backend."""

from __future__ import annotations

import ast
import functools
import inspect
import os
import textwrap
from collections.abc import Callable
from typing import Any

import torch

from torchtitan.tools.logging import logger


_LEGACY_BLOCK_MASK_ARITY = 13
_DQ_SCHEDULING_FIELDS_START = 10
_DQ_SCHEDULING_FIELDS_END = 14
_COMPLEX_DTENSOR_STRATEGY_REGISTERED = False
_FLEX_ATTENTION_BWD_DKDV_TASKLIST_ENV = (
    "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST"
)
_DETERMINISTIC_NPU_COMPILE_OPTIONS_ENV = (
    "TORCHTITAN_DETERMINISTIC_NPU_COMPILE_OPTIONS"
)


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value not in (None, "0", "1"):
        raise ValueError(f"{name} must be 0 or 1; got {value!r}")
    return value == "1"


def _register_complex_dtensor_strategy() -> bool:
    """Register the missing PyTorch 2.14 strategy for ``aten.complex``."""
    global _COMPLEX_DTENSOR_STRATEGY_REGISTERED
    if _COMPLEX_DTENSOR_STRATEGY_REGISTERED:
        return False

    from torch.distributed.tensor._ops._pointwise_ops import (
        _register_single_dim_pointwise,
    )

    try:
        _register_single_dim_pointwise(torch.ops.aten.complex.default)
    except (AssertionError, RuntimeError) as error:
        # Newer PyTorch releases may already carry the upstream registration.
        if "already" not in str(error).lower():
            raise
    _COMPLEX_DTENSOR_STRATEGY_REGISTERED = True
    return True


def _configure_flex_attention_dkdv_tasklist() -> bool:
    """Apply the compatibility mode even when torch_npu config is preloaded."""
    import torch_npu._inductor.config as npu_config

    flex_attention_config = getattr(npu_config, "flex_attention", None)
    if flex_attention_config is None or not hasattr(
        flex_attention_config, "bwd_dkdv_tasklist"
    ):
        return False

    enabled = _env_flag_enabled(_FLEX_ATTENTION_BWD_DKDV_TASKLIST_ENV)
    if flex_attention_config.bwd_dkdv_tasklist == enabled:
        return False
    flex_attention_config.bwd_dkdv_tasklist = enabled
    return True


def _sorted_npu_compile_options(
    metadata: dict[str, Any], compile_option_keys: Any
) -> dict[str, Any]:
    """Return NPU compile options in a process-independent key order."""
    return {
        key: metadata[key]
        for key in sorted(compile_option_keys)
        if key in metadata
    }


def _patch_npu_compile_option_serialization() -> bool:
    """Make generated NPU kernel metadata stable across rank processes.

    torch_npu currently converts its ordered compile-option keys to a
    ``frozenset`` and iterates that set when rebuilding the metadata mapping.
    Python hash randomization can therefore serialize the same semantic graph
    differently in different ranks, defeating persistent Inductor cache hits.
    """
    import torch_npu._inductor.select_algorithm as select_algorithm

    current = getattr(
        select_algorithm,
        "_add_npu_template_compile_options_to_triton_meta",
        None,
    )
    if current is None:
        return False
    if getattr(current, "_torchtitanturbo_deterministic_order", False):
        return False

    @functools.wraps(current)
    def add_compile_options(triton_meta, meta, compile_option_keys):
        compile_options = _sorted_npu_compile_options(meta, compile_option_keys)
        if compile_options:
            triton_meta.setdefault("npu_compile_options", {}).update(
                compile_options
            )

    add_compile_options._torchtitanturbo_deterministic_order = True
    select_algorithm._add_npu_template_compile_options_to_triton_meta = (
        add_compile_options
    )
    return True


def _block_mask_unpack_arity(handler: Callable[..., Any]) -> int | None:
    """Return the number of fields a lowering unpacks from ``block_mask``."""
    try:
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(handler)))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return None

    arities = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Name)
            and node.value.id == "block_mask"
        ):
            continue
        arities.append(len(node.targets[0].elts))
    return arities[0] if len(arities) == 1 else None


def _legacy_npu_block_mask(block_mask: Any) -> Any:
    """Remove PyTorch 2.14 dQ scheduling metadata for a 13-field lowering.

    PyTorch autograd must retain the full 17-field tuple.  Conversion therefore
    happens only at the NPU Inductor lowering boundary, after autograd has saved
    ``dq_write_order*`` and ``dq_kv_order*`` for its own backward bookkeeping.
    """
    if not isinstance(block_mask, tuple) or len(block_mask) != 17:
        return block_mask
    return (
        block_mask[:_DQ_SCHEDULING_FIELDS_START]
        + block_mask[_DQ_SCHEDULING_FIELDS_END:]
    )


def _legacy_block_mask_lowering_wrapper(
    handler: Callable[..., Any], mask_arg_index: int
) -> Callable[..., Any]:
    """Wrap one legacy torch_npu lowering without changing PyTorch's protocol."""

    @functools.wraps(handler)
    def compatible(*args, **kwargs):
        if len(args) > mask_arg_index:
            normalized = _legacy_npu_block_mask(args[mask_arg_index])
            if normalized is not args[mask_arg_index]:
                args = (
                    *args[:mask_arg_index],
                    normalized,
                    *args[mask_arg_index + 1 :],
                )
        elif "block_mask" in kwargs:
            normalized = _legacy_npu_block_mask(kwargs["block_mask"])
            if normalized is not kwargs["block_mask"]:
                kwargs = {**kwargs, "block_mask": normalized}
        return handler(*args, **kwargs)

    compatible._torchtitanturbo_block_mask_protocol = True
    return compatible


def _patch_flex_attention_block_mask_protocol() -> int:
    """Bridge PyTorch's 17-field BlockMask to legacy 13-field NPU lowerings."""
    # Importing the backend installs its higher-order-op lowerings.  Newer
    # torch_npu builds may already consume 17 fields; source-based feature
    # detection deliberately leaves those handlers untouched.
    import torch_npu._inductor  # noqa: F401
    from torch._inductor.lowering import lowerings

    specifications = (
        (torch.ops.higher_order.flex_attention, 4),
        (torch.ops.higher_order.flex_attention_backward, 9),
    )
    patched = 0
    for operation, mask_arg_index in specifications:
        handler = lowerings.get(operation)
        if handler is None or getattr(
            handler, "_torchtitanturbo_block_mask_protocol", False
        ):
            continue
        if not getattr(handler, "__module__", "").startswith("torch_npu."):
            continue
        if _block_mask_unpack_arity(handler) != _LEGACY_BLOCK_MASK_ARITY:
            continue
        lowerings[operation] = _legacy_block_mask_lowering_wrapper(
            handler, mask_arg_index
        )
        patched += 1
    return patched


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
    complex_strategy_registered = False
    if _env_flag_enabled("TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY"):
        complex_strategy_registered = _register_complex_dtensor_strategy()
    dkdv_tasklist_configured = _configure_flex_attention_dkdv_tasklist()
    compile_options_order_patched = False
    deterministic_compile_options = os.environ.get(
        _DETERMINISTIC_NPU_COMPILE_OPTIONS_ENV, "1"
    )
    if deterministic_compile_options not in ("0", "1"):
        raise ValueError(
            f"{_DETERMINISTIC_NPU_COMPILE_OPTIONS_ENV} must be 0 or 1; "
            f"got {deterministic_compile_options!r}"
        )
    if deterministic_compile_options == "1":
        compile_options_order_patched = (
            _patch_npu_compile_option_serialization()
        )
    flex_attention_lowerings = _patch_flex_attention_block_mask_protocol()
    narrowing_patched = _patch_bf16_narrowing_codegen()
    addmm_patterns = _patch_low_precision_addmm_fusion()
    if complex_strategy_registered:
        logger.info("Registered aten.complex DTensor pointwise strategy")
    if dkdv_tasklist_configured:
        logger.info(
            "Configured NPU FlexAttention backward dK/dV task-list mode: %s",
            os.environ.get(_FLEX_ATTENTION_BWD_DKDV_TASKLIST_ENV, "0"),
        )
    if compile_options_order_patched:
        logger.info(
            "Patched deterministic NPU compile-option serialization for cache reuse"
        )
    if flex_attention_lowerings:
        logger.info(
            "Patched %d legacy NPU FlexAttention BlockMask lowerings",
            flex_attention_lowerings,
        )
    if narrowing_patched or addmm_patterns:
        logger.info(
            "Patched NPU Inductor BF16 rounding and %d addmm fusion patterns",
            addmm_patterns,
        )
