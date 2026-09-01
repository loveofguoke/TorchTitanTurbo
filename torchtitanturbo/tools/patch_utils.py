# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Simplified function replacement utilities for NPU patches.

Handles 'from X import Y' local references by finding and replacing
all imported function references across sys.modules.

Python ``from package.module import func`` copies the function object into the
consumer module namespace. Reassigning only ``package.module.func`` therefore
does not affect those consumers. ``replace_functions`` scans already-imported
TorchTitan modules and replaces matching attributes by name; callers must first
validate the upstream signature and behavior to avoid patching an unrelated
symbol with the same name.

"""

import sys
from collections.abc import Callable

import logging

logger = logging.getLogger(__name__)


def replace_functions(
    func_name: str,
    new_func: Callable,
    package: str = "torchtitan",
) -> int:
    """Replace a function in all modules that have imported it.

    Handles both:
    - Direct module attributes: module.func
    - Local references: from module import func

    Args:
        func_name: Name of the function to replace
        new_func: New function implementation
        package: Package prefix to filter modules (default: "torchtitan")

    Returns:
        Number of modules patched

    Example:
        >>> count = replace_functions(
        ...     "apply_rotary_emb_complex",
        ...     npu_apply_rotary_emb_complex,
        ...     package="torchtitan"
        ... )
        >>> logger.info(f"Patched {count} modules")
    """
    count = 0
    module_paths = []

    for path, mod in sys.modules.items():
        if not mod or not path.startswith(package):
            continue

        func = getattr(mod, func_name, None)
        if callable(func) and not isinstance(func, type):
            setattr(mod, func_name, new_func)
            count += 1
            module_paths.append(path)

    if count > 0:
        logger.info(f"Replaced {func_name} in {count} modules")
        if module_paths:
            logger.info(f"  Modules: {', '.join(module_paths[:3])}")
            if len(module_paths) > 3:
                logger.info(f"  ... and {len(module_paths) - 3} more")

    return count
