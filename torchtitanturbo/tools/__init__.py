# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import logging

from .graph_compat import apply_graph_compat_patches
from .patch_utils import replace_functions
from .profiler import apply_patch as apply_profiler_patch
from .utils import apply_patch as apply_utils_patch
from .utils import get_peak_flops

logger = logging.getLogger(__name__)


def apply_compile_patch() -> None:
    """Keep compatibility with callers after upstream removed apply_compile.

    Current TorchTitan owns local compilation through model configs and
    function-scoped registrations. There is no global hook for Turbo to patch;
    Ascend backend validation lives in ``graph_compat`` instead.
    """
    logger.info("TorchTitan has no global apply_compile hook; no patch required")

__all__ = [
    "get_peak_flops",
    "apply_utils_patch",
    "apply_profiler_patch",
    "apply_compile_patch",
    "apply_graph_compat_patches",
    "replace_functions",
]
