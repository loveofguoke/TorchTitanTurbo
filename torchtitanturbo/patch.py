# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Unified NPU patch entry point for TorchTitan.

Applies all NPU-specific monkey patches at import time.

Device-gated: on a host without an Ascend NPU backend, importing this package
is a no-op, so GPU/CPU runs behave identically to plain torchtitan. The gating
logic (``_npu_available``) was migrated here from the core repo's former
``npu/adapt/patches.py`` (that directory is no longer used).
"""

import os

import torch

from torchtitan.tools.logging import logger


def _npu_available() -> bool:
    """True when running on an Ascend NPU (torch_npu backend present)."""
    try:
        import torch_npu  # noqa: F401  registers the npu backend
    except ImportError:
        return False
    npu_backend = getattr(torch, "npu", None)
    return bool(npu_backend is not None and npu_backend.is_available())


def set_environ_variable():
    """Set NPU-specific environment variables."""
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
    if not _npu_available():
        logger.info(
            "torchtitanturbo: no Ascend NPU detected, skipping NPU patches"
        )
        return

    set_environ_variable()

    from torchtitanturbo.tools import (
        apply_compile_patch,
        apply_graph_compat_patches,
        apply_profiler_patch,
        apply_utils_patch,
    )
    from torchtitanturbo.distributed import apply_fsdp_patch
    from torchtitanturbo.models.common import apply_rope_patch
    from torchtitanturbo.models import (
        apply_deepseek_patch,
        apply_glm5_patch,
        apply_qwen3_patch,
    )

    apply_utils_patch()
    apply_profiler_patch()
    apply_compile_patch()
    apply_graph_compat_patches()
    apply_rope_patch()
    apply_fsdp_patch()
    apply_deepseek_patch()
    apply_qwen3_patch()
    apply_glm5_patch()

    logger.info("All NPU patches applied successfully")


apply_all_patches()
