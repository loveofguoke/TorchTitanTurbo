# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Ascend-specific validation for TorchTitan ``torch.compile`` backends.

TorchTitan remains responsible for choosing components and calling
``torch.compile``. Turbo adds only backend-specific preconditions, then
delegates to the original entry point. Modules that used ``from ... import
apply_compile`` hold their own function reference, so patch installation must
replace those already-imported references as well as the defining module.
"""

import sys
from typing import Any

from torchtitan.distributed import compile as compile_module
from torchtitan.tools.logging import logger


_ORIGINAL_APPLY_COMPILE = compile_module.apply_compile


def validate_npu_compile_config(compile_config: Any) -> None:
    """Validate contracts that are specific to the Ascend compile backend."""

    if not compile_config.enable or "model" not in compile_config.components:
        return
    if compile_config.backend != "npugraphs":
        return
    if "loss" in compile_config.components:
        raise ValueError("npugraphs prototype supports model compilation only")


def npu_apply_compile(model, *, compile_config, parallel_dims) -> None:
    """Validate the NPU backend and delegate compilation to TorchTitan."""

    validate_npu_compile_config(compile_config)
    _ORIGINAL_APPLY_COMPILE(
        model,
        compile_config=compile_config,
        parallel_dims=parallel_dims,
    )


def apply_patch() -> None:
    """Patch already-imported TorchTitan references to the compile entry point."""

    if compile_module.apply_compile is npu_apply_compile:
        return

    patched_modules: list[str] = []
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith("torchtitan"):
            continue
        if getattr(module, "apply_compile", None) is _ORIGINAL_APPLY_COMPILE:
            setattr(module, "apply_compile", npu_apply_compile)
            patched_modules.append(module_name)

    logger.info(
        "Patched TorchTitan compile validation for Ascend in %d modules: %s",
        len(patched_modules),
        ", ".join(patched_modules),
    )
