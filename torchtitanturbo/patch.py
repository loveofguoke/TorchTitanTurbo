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
    if not _npu_available():
        logger.info(
            "torchtitanturbo: no Ascend NPU detected, skipping NPU patches"
        )
        return

    set_environ_variable()

    # TorchTitan main removed the former global distributed.compile entry
    # point while migrating compilation and models to config-first APIs. The
    # legacy Turbo patch bundle imports several other removed protocols, so do
    # not partially install it against this API generation. GLM has a fully
    # migrated, independently validated patch and remains safe to install.
    if find_spec("torchtitan.distributed.compile") is None:
        from torchtitanturbo.models.common.npu_rope import (
            apply_patch as apply_rope_patch,
        )
        from torchtitanturbo.models.glm5 import apply_glm5_patch

        apply_rope_patch()
        apply_glm5_patch()
        logger.warning(
            "Detected config-first TorchTitan main: applied the migrated GLM "
            "patch only; legacy common/distributed converters require a "
            "separate upstream-API migration"
        )
        return

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

    # Patch order follows dependency direction: metrics/profiler/compile and
    # shared kernels first, distributed compatibility second, then models.
    # Model patches may rely on the shared RoPE/FSDP behavior already installed.
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
