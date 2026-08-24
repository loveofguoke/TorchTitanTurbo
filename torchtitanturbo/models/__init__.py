# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from .common import (
    NpuGroupedExperts,
    NpuGroupedExpertsConverter,
    NpuTokenDispatcher,
    NpuTokenDispatcherConverter,
    NpuRMSNorm,
    NpuRMSNormConverter,
    NpuScaledDotProductAttention,
    NpuSDPAConverter,
    apply_rope_patch,
)


def apply_deepseek_patch():
    """Apply DeepSeek patches if available."""
    try:
        from .deepseek_v3 import apply_patch

        apply_patch()
    except ImportError:
        pass


def apply_qwen3_patch():
    """Apply Qwen3 patches if available."""
    try:
        from .qwen3 import apply_patch

        apply_patch()
    except ImportError:
        pass


def apply_glm5_patch():
    """Apply GLM-5 patches if available."""
    try:
        from .glm5 import apply_glm5_patch as apply_patch

        apply_patch()
    except ImportError:
        pass


__all__ = [
    "NpuGroupedExperts",
    "NpuGroupedExpertsConverter",
    "NpuTokenDispatcher",
    "NpuTokenDispatcherConverter",
    "NpuRMSNorm",
    "NpuRMSNormConverter",
    "NpuScaledDotProductAttention",
    "NpuSDPAConverter",
    "apply_rope_patch",
    "apply_deepseek_patch",
    "apply_qwen3_patch",
    "apply_glm5_patch",
]
