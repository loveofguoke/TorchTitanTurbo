# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Ascend compatibility for ``torch_remat`` placeholder storage.

``torch_remat`` represents an output skipped during recomputation with a
correctly shaped tensor backed by a declared-size, null-pointer storage.  The
current NPU backend can construct that storage, but destroying it after an
NPU recomputation may segfault in the allocator/runtime.  Keep the upstream
placeholder protocol (including the poisoned data-pointer diagnostic), while
using an owning NPU byte storage until the backend supports null storage for
this lifecycle.
"""

from __future__ import annotations

import inspect
import logging

import torch

logger = logging.getLogger(__name__)

_PATCHED_MARKER = "_torchtitanturbo_npu_remat_storage"


def apply_patch() -> None:
    """Install the NPU storage fallback without changing CPU/CUDA behavior."""

    try:
        from torch_remat import _placeholder
    except ImportError as error:
        raise RuntimeError(
            "TorchTitan requires torch_remat, but its placeholder module "
            "could not be imported"
        ) from error

    current = _placeholder._null_storage
    if getattr(current, _PATCHED_MARKER, False):
        return

    signature = inspect.signature(current)
    if tuple(signature.parameters) != ("nbytes", "device"):
        raise RuntimeError(
            "Unsupported torch_remat._placeholder._null_storage signature: "
            f"{signature}"
        )

    def npu_safe_null_storage(
        nbytes: int, device: torch.device
    ) -> torch.UntypedStorage:
        if device.type != "npu":
            return current(nbytes, device)

        # The returned storage owns its allocation after the temporary tensor
        # is released.  torch_remat subsequently attaches it to the placeholder
        # and poisons data_ptr access exactly as in the upstream implementation.
        return torch.empty(
            nbytes, dtype=torch.uint8, device=device
        ).untyped_storage()

    setattr(npu_safe_null_storage, _PATCHED_MARKER, True)
    _placeholder._null_storage = npu_safe_null_storage
    logger.warning(
        "Applied owning-storage fallback for torch_remat placeholders on NPU"
    )
