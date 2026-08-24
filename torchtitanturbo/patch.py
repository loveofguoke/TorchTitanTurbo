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
    # torch_npu 2.14 rejects setting both allocator variables. NPU entry points
    # own the accelerator-specific setting and must discard the generic one.
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ.setdefault(
        "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"
    )
    os.environ["TASK_QUEUE_ENABLE"] = "2"
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

    from torchtitanturbo.tools import apply_utils_patch, apply_profiler_patch
    from torchtitanturbo.distributed import apply_fsdp_patch
    from torchtitanturbo.models.common import apply_rope_patch
    from torchtitanturbo.models import apply_deepseek_patch, apply_qwen3_patch

    apply_utils_patch()
    apply_profiler_patch()
    apply_rope_patch()
    apply_fsdp_patch()
    apply_deepseek_patch()
    apply_qwen3_patch()

    logger.info("All NPU patches applied successfully")


apply_all_patches()
