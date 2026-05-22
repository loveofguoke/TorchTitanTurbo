# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from torchtitanturbo.models.common import npu_gmm, npu_permute, npu_rmsnorm
from torchtitanturbo.patch import apply_all_patches  # triggers auto-patch on import

__all__ = [
    "npu_gmm",
    "npu_permute",
    "npu_rmsnorm",
]
__version__ = "0.1.0"
