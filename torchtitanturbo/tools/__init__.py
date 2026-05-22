# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from .utils import get_peak_flops, apply_patch as apply_utils_patch
from .profiler import apply_patch as apply_profiler_patch
from .patch_utils import replace_functions

__all__ = [
    "get_peak_flops",
    "apply_utils_patch",
    "apply_profiler_patch",
    "replace_functions",
]
