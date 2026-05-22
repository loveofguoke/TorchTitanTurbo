# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Unified NPU patch entry point for TorchTitan.

Applies all NPU-specific monkey patches at import time.
"""

import os

from torchtitan.tools.logging import logger


def set_environ_variable():
    """Set NPU-specific environment variables."""
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
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
    """Apply all NPU patches."""
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
