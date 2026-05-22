# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from .npu_gmm import NpuGroupedExperts, NpuGroupedExpertsConverter
from .npu_permute import NpuTokenDispatcher, NpuTokenDispatcherConverter
from .npu_rmsnorm import NpuRMSNorm, NpuRMSNormConverter
from .npu_attention import NpuScaledDotProductAttention, NpuSDPAConverter
from .npu_rope import apply_patch as apply_rope_patch

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
]
