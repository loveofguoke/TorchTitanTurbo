# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Temporary home for compatibility patches unrelated to NPU behavior."""

from .runtime import apply_patch as apply_runtime_compat_patch


def apply_all_non_npu_patches() -> None:
    """Apply compatibility patches caused by upstream version or generic issues."""
    apply_runtime_compat_patch()


__all__ = ["apply_all_non_npu_patches", "apply_runtime_compat_patch"]
