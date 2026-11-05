# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Unified NPU patch entry point for TorchTitan.

Applies all NPU-specific monkey patches at import time.

Device-gated: on a host without an Ascend NPU backend, importing this package
is a no-op, so GPU/CPU runs behave identically to plain torchtitan. The gating
logic (``_npu_available``) was migrated here from the core repo's former
``npu/adapt/patches.py`` (that directory is no longer used).
"""

import logging
import os
from importlib.util import find_spec

import torch

logger = logging.getLogger(__name__)

_CONFIG_FIRST_PATCHES = ("remat", "rope", "glm_init", "glm_loss")
_EXPERIMENTAL_PATCHES = ("inductor_scheduler", "inductor_reduction", "pipeline_cleanup")
_PATCH_STATUS = {}


def _select_config_first_patches(raw: str | None) -> tuple[str, ...]:
    if raw is None or raw == "all":
        return _CONFIG_FIRST_PATCHES
    if raw == "none":
        return ()
    names = raw.split(",")
    valid = _EXPERIMENTAL_PATCHES + _CONFIG_FIRST_PATCHES
    unknown = set(names) - set(valid)
    if unknown:
        raise ValueError(f"Unknown Turbo patch names: {sorted(unknown)}")
    return tuple(name for name in valid if name in names)


def get_patch_status() -> dict:
    """Return the explicitly selected and successfully applied bootstrap groups."""
    return dict(_PATCH_STATUS)


def _npu_available() -> bool:
    """True when running on an Ascend NPU (torch_npu backend present)."""
    try:
        import torch_npu  # noqa: F401  registers the npu backend
    except ImportError:
        return False
    npu_backend = getattr(torch, "npu", None)
    return bool(npu_backend is not None and npu_backend.is_available())


def set_environ_variable():
    """Set process-wide NPU runtime policy before any model is constructed.

    These variables affect allocator growth, host enqueueing, stream memory
    reuse, and CPU affinity. They are runtime policy rather than model math.
    The task-queue mode has an explicit user override because graph debugging
    may need synchronous mode while eager throughput normally prefers queues.
    """
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    requested_task_queue = os.environ.get("TORCHTITAN_TASK_QUEUE_ENABLE")
    if requested_task_queue not in (None, "0", "1", "2"):
        raise ValueError(
            "TORCHTITAN_TASK_QUEUE_ENABLE must be 0, 1, or 2; "
            f"got {requested_task_queue!r}"
        )
    os.environ["TASK_QUEUE_ENABLE"] = requested_task_queue or "2"
    os.environ["MULTI_STREAM_MEMORY_REUSE"] = "2"
    os.environ["CPU_AFFINITY_CONF"] = "2"

    logger.info("Set NPU environment variables:")
    logger.info(
        f"  PYTORCH_NPU_ALLOC_CONF = {os.environ.get('PYTORCH_NPU_ALLOC_CONF')}"
    )
    logger.info(f"  TASK_QUEUE_ENABLE = {os.environ.get('TASK_QUEUE_ENABLE')}")
    logger.info(
        f"  MULTI_STREAM_MEMORY_REUSE = {os.environ.get('MULTI_STREAM_MEMORY_REUSE')}"
    )
    logger.info(f"  CPU_AFFINITY_CONF = {os.environ.get('CPU_AFFINITY_CONF')}")


def apply_all_patches():
    """Apply all NPU patches.

    No-op (with a log line) when no Ascend NPU backend is available, so GPU/CPU
    runs that import torchtitanturbo are not affected by NPU-specific patches.
    """
    selection = os.environ.get("TORCHTITAN_TURBO_PATCHES")
    selected = _select_config_first_patches(selection)
    previous = _PATCH_STATUS.get("applied", ())
    if previous and tuple(previous) != selected:
        raise RuntimeError(
            "Patch A/B requires a fresh process; patches cannot be uninstalled"
        )
    _PATCH_STATUS.update(selection=selection, selected=selected, applied=())
    if not _npu_available():
        _PATCH_STATUS["scope"] = "no_npu"
        logger.info("torchtitanturbo: no Ascend NPU detected, skipping NPU patches")
        return

    set_environ_variable()

    # TorchTitan main removed the former global distributed.compile entry
    # point while migrating compilation and models to config-first APIs. The
    # legacy Turbo patch bundle imports several other removed protocols, so do
    # not partially install it against this API generation. GLM has a fully
    # migrated, independently validated patch and remains safe to install.
    if find_spec("torchtitan.distributed.compile") is None:
        _PATCH_STATUS["scope"] = "config_first"
        from torchtitanturbo.distributed.remat import apply_patch as apply_remat_patch
        from torchtitanturbo.models.common.npu_rope import (
            apply_patch as apply_rope_patch,
        )
        from torchtitanturbo.models.glm5 import apply_glm5_patch

        applied = []
        if "inductor_scheduler" in selected:
            from torchtitanturbo.tools.inductor_scheduler import apply_patch

            if apply_patch():
                applied.append("inductor_scheduler")
            else:
                _PATCH_STATUS["not_required"] = ("inductor_scheduler",)
        if "inductor_reduction" in selected:
            from torchtitanturbo.tools.inductor_reduction import apply_patch

            if apply_patch():
                applied.append("inductor_reduction")
        if "pipeline_cleanup" in selected:
            from torchtitanturbo.distributed.pipeline_cleanup import apply_patch

            if apply_patch():
                applied.append("pipeline_cleanup")
        if "remat" in selected:
            apply_remat_patch()
            applied.append("remat")
        if "rope" in selected:
            apply_rope_patch()
            applied.append("rope")
        if "glm_init" in selected or "glm_loss" in selected:
            apply_glm5_patch(
                initialization="glm_init" in selected,
                loss="glm_loss" in selected,
            )
            applied.extend(
                name for name in ("glm_init", "glm_loss") if name in selected
            )
        _PATCH_STATUS["applied"] = tuple(applied)
        logger.warning(
            "Turbo config-first patch selection: %s; legacy converters are not loaded",
            _PATCH_STATUS,
        )
        return

    if selection is not None:
        if selected:
            raise ValueError(
                "Explicit patch selection supports config-first TorchTitan only"
            )
        _PATCH_STATUS["scope"] = "legacy_disabled"
        return

    from torchtitanturbo.distributed import apply_fsdp_patch, apply_remat_patch
    from torchtitanturbo.models import (
        apply_deepseek_patch,
        apply_glm5_patch,
        apply_qwen3_patch,
    )
    from torchtitanturbo.models.common import apply_rope_patch
    from torchtitanturbo.tools import (
        apply_compile_patch,
        apply_graph_compat_patches,
        apply_profiler_patch,
        apply_utils_patch,
    )

    # Patch order follows dependency direction: metrics/profiler/compile and
    # shared kernels first, distributed compatibility second, then models.
    # Model patches may rely on the shared RoPE/FSDP behavior already installed.
    apply_utils_patch()
    apply_profiler_patch()
    apply_compile_patch()
    apply_graph_compat_patches()
    apply_remat_patch()
    apply_rope_patch()
    apply_fsdp_patch()
    apply_deepseek_patch()
    apply_qwen3_patch()
    apply_glm5_patch()

    logger.info("All NPU patches applied successfully")


apply_all_patches()
