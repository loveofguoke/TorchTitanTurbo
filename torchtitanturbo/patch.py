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
    requested_task_queue = os.environ.get("TORCHTITAN_TASK_QUEUE_ENABLE")
    if requested_task_queue not in (None, "0", "1", "2"):
        raise ValueError(
            "TORCHTITAN_TASK_QUEUE_ENABLE must be 0, 1, or 2; "
            f"got {requested_task_queue!r}"
        )
    os.environ["TASK_QUEUE_ENABLE"] = requested_task_queue or "2"
    requested_dkdv_tasklist = os.environ.get(
        "TORCHTITAN_FLEX_ATTENTION_BWD_DKDV_TASKLIST"
    )
    if requested_dkdv_tasklist not in (None, "0", "1"):
        raise ValueError(
            "TORCHTITAN_FLEX_ATTENTION_BWD_DKDV_TASKLIST must be 0 or 1; "
            f"got {requested_dkdv_tasklist!r}"
        )
    dkdv_tasklist_env = (
        "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST"
    )
    if requested_dkdv_tasklist is None:
        # The current torch_npu task-list path synchronously copies sparse-mask
        # metadata to the host from the middle of a compiled backward graph.
        # That host synchronization can deadlock interleaved FSDP/TP process
        # groups (reproduced by FSDP4 x TP2).  Keep the legacy device-only
        # dK/dV path as the compatibility default until torch_npu removes the
        # runtime host read.  Respect a vendor-level override for experiments.
        os.environ.setdefault(dkdv_tasklist_env, "0")
    else:
        os.environ[dkdv_tasklist_env] = requested_dkdv_tasklist
    os.environ["MULTI_STREAM_MEMORY_REUSE"] = "2"
    os.environ["CPU_AFFINITY_CONF"] = "2"

    logger.info("Set NPU environment variables:")
    logger.info(
        f"  PYTORCH_NPU_ALLOC_CONF = {os.environ.get('PYTORCH_NPU_ALLOC_CONF')}"
    )
    logger.info(f"  TASK_QUEUE_ENABLE = {os.environ.get('TASK_QUEUE_ENABLE')}")
    logger.info(
        "  TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST = %s",
        os.environ.get(dkdv_tasklist_env),
    )
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

    from torchtitanturbo.compiler import apply_inductor_patch
    from torchtitanturbo.non_npu_compat import apply_all_non_npu_patches
    from torchtitanturbo.tools import apply_utils_patch, apply_profiler_patch
    from torchtitanturbo.distributed import apply_fsdp_patch
    from torchtitanturbo.models.common import apply_rope_patch
    from torchtitanturbo.models import (
        apply_deepseek_patch,
        apply_glm5_patch,
        apply_qwen3_patch,
    )

    apply_all_non_npu_patches()
    apply_inductor_patch()
    apply_utils_patch()
    apply_profiler_patch()
    apply_rope_patch()
    apply_fsdp_patch()
    apply_deepseek_patch()
    apply_glm5_patch()
    apply_qwen3_patch()

    logger.info("All NPU patches applied successfully")


apply_all_patches()
