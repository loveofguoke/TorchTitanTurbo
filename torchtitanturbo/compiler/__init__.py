# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU compiler compatibility patches."""

from .inductor import apply_patch as apply_inductor_patch

__all__ = ["apply_inductor_patch"]
