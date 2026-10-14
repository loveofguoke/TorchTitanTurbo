# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Lazy exports for optional TorchTitan model converters.

Importing one migrated adapter must not eagerly import legacy converters that
target a different TorchTitan API generation.
"""

from importlib import import_module


_EXPORTS = {
    "NpuGroupedExperts": ("npu_gmm", "NpuGroupedExperts"),
    "NpuGroupedExpertsConverter": ("npu_gmm", "NpuGroupedExpertsConverter"),
    "NpuTokenDispatcher": ("npu_permute", "NpuTokenDispatcher"),
    "NpuTokenDispatcherConverter": ("npu_permute", "NpuTokenDispatcherConverter"),
    "NpuRMSNorm": ("npu_rmsnorm", "NpuRMSNorm"),
    "NpuRMSNormConverter": ("npu_rmsnorm", "NpuRMSNormConverter"),
    "NpuScaledDotProductAttention": ("npu_attention", "NpuScaledDotProductAttention"),
    "NpuSDPAConverter": ("npu_attention", "NpuSDPAConverter"),
    "apply_rope_patch": ("npu_rope", "apply_patch"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value

__all__ = list(_EXPORTS)
