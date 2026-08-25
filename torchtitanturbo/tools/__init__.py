# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from .compile import apply_patch as apply_compile_patch
from .graph_compat import apply_graph_compat_patches
from .patch_utils import replace_functions
from .profiler import apply_patch as apply_profiler_patch
from .utils import apply_patch as apply_utils_patch
from .utils import get_peak_flops

__all__ = [
    "get_peak_flops",
    "apply_utils_patch",
    "apply_profiler_patch",
    "apply_compile_patch",
    "apply_graph_compat_patches",
    "replace_functions",
]
