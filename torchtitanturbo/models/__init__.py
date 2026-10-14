# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

_COMMON_EXPORTS = {
    "NpuGroupedExperts",
    "NpuGroupedExpertsConverter",
    "NpuTokenDispatcher",
    "NpuTokenDispatcherConverter",
    "NpuRMSNorm",
    "NpuRMSNormConverter",
    "NpuScaledDotProductAttention",
    "NpuSDPAConverter",
    "apply_rope_patch",
}


def __getattr__(name: str):
    """Load optional common converters only when a caller requests one."""
    if name not in _COMMON_EXPORTS:
        raise AttributeError(name)
    from . import common

    value = getattr(common, name)
    globals()[name] = value
    return value


def apply_deepseek_patch():
    """Apply DeepSeek patches if available."""
    try:
        from .deepseek_v3 import apply_deepseek_patch as patch_fn

        patch_fn()
    except ImportError:
        pass


def apply_qwen3_patch():
    """Apply Qwen3 patches if available."""
    try:
        from .qwen3 import apply_qwen3_patch as patch_fn

        patch_fn()
    except ImportError:
        pass


def apply_glm5_patch():
    """Apply GLM-5 patches if available."""
    try:
        from .glm5 import apply_glm5_patch as patch_fn

        patch_fn()
    except ImportError:
        pass


__all__ = [
    *_COMMON_EXPORTS,
    "apply_deepseek_patch",
    "apply_qwen3_patch",
    "apply_glm5_patch",
]
