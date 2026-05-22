# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU-optimized Llama4 model extensions."""

from .config_registry import (
    npu_llama4_debugmodel,
    npu_llama4_debugmodel_ep,
    npu_llama4_debugmodel_minimal,
)

__all__ = [
    "npu_llama4_debugmodel",
    "npu_llama4_debugmodel_ep",
    "npu_llama4_debugmodel_minimal",
]
