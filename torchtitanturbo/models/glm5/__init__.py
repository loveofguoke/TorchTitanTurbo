# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

def apply_glm5_patch():
    """Load and apply GLM-5 patches after this package is initialized."""
    import torchtitan.models.glm5.sharding  # noqa: F401

    from .patch import apply_patch

    apply_patch()

__all__ = ["apply_glm5_patch"]
