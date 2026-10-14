"""Lazy, capability-checked access to the DeepSelect TopK extension.

This module does not patch TorchTitan. Model integrations must select the
provider explicitly after validating that their score dtype and ordering
contract are compatible with the installed DeepSelect backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

import torch


class DeepSelectCapabilityError(ValueError):
    """The requested TopK contract is not supported by DeepSelect."""


@dataclass(frozen=True, slots=True)
class DeepSelectRequest:
    platform: str
    dtype: torch.dtype
    num_rows: int
    row_width: int
    topk: int
    row_stride_bytes: int
    input_stride_requirement: int | None = None


def validate_deepselect_request(request: DeepSelectRequest) -> None:
    """Validate the public DeepSelect support and layout contract."""
    if request.platform not in {"cuda", "npu"}:
        raise DeepSelectCapabilityError(
            "DeepSelect supports CUDA and NPU tensors only; "
            f"got platform={request.platform!r}."
        )
    if request.num_rows <= 0 or request.row_width <= 0:
        raise DeepSelectCapabilityError(
            "DeepSelect requires a non-empty two-dimensional score matrix."
        )
    if not 0 < request.topk <= min(4096, request.row_width):
        raise DeepSelectCapabilityError(
            "DeepSelect requires 0 < topk <= min(4096, row_width); "
            f"got topk={request.topk}, row_width={request.row_width}."
        )
    supported_dtypes = (
        {torch.bfloat16, torch.float32}
        if request.platform == "cuda"
        else {torch.bfloat16}
    )
    if request.dtype not in supported_dtypes:
        supported = ", ".join(sorted(str(dtype) for dtype in supported_dtypes))
        raise DeepSelectCapabilityError(
            f"DeepSelect on {request.platform} does not support "
            f"{request.dtype}; supported dtypes: {supported}."
        )
    if request.input_stride_requirement is not None:
        requirement = request.input_stride_requirement
        if requirement <= 0:
            raise DeepSelectCapabilityError(
                "DeepSelect returned an invalid input stride requirement: "
                f"{requirement}."
            )
        if request.row_stride_bytes % requirement != 0:
            raise DeepSelectCapabilityError(
                "DeepSelect requires an aligned row stride; "
                f"got {request.row_stride_bytes} bytes, required multiple "
                f"of {requirement} bytes. Pad the last dimension instead of "
                "silently copying the score matrix."
            )


class DeepSelectTopKProvider:
    """Invoke DeepSelect without importing it on the reference path."""

    name = "deepselect"

    def __init__(self, library: ModuleType | None = None) -> None:
        self._library = library

    def _load_library(self) -> ModuleType:
        if self._library is None:
            try:
                self._library = import_module("deep_select")
            except ImportError as error:
                raise RuntimeError(
                    "DeepSelect was selected but the deep_select package is "
                    "not installed. Install the pinned source revision for "
                    "the active CUDA or Ascend toolchain."
                ) from error
        return self._library

    def indices(self, scores_RV: torch.Tensor, topk: int) -> torch.Tensor:
        """Return the unordered TopK index set as int32 metadata."""
        if scores_RV.ndim != 2:
            raise DeepSelectCapabilityError(
                "DeepSelect expects a two-dimensional score matrix [R, V]; "
                f"got shape={tuple(scores_RV.shape)}."
            )
        if scores_RV.stride(-1) != 1:
            raise DeepSelectCapabilityError(
                "DeepSelect requires the last score dimension to be contiguous."
            )

        library = self._load_library()
        input_stride_requirement, _ = library.get_stride_requirement()
        validate_deepselect_request(
            DeepSelectRequest(
                platform=scores_RV.device.type,
                dtype=scores_RV.dtype,
                num_rows=scores_RV.shape[0],
                row_width=scores_RV.shape[1],
                topk=topk,
                row_stride_bytes=scores_RV.stride(0)
                * scores_RV.element_size(),
                input_stride_requirement=input_stride_requirement,
            )
        )
        _values, indices_RK = library.topk(
            scores_RV,
            topk,
            sorted=False,
            sorted_index=False,
            indices_type=torch.int32,
            return_value=False,
        )
        return indices_RK

    def select(
        self, scores_RV: torch.Tensor, topk: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select indices in the kernel and gather differentiable scores."""
        indices_RK = self.indices(scores_RV, topk)
        values_RK = scores_RV.gather(-1, indices_RK.to(torch.int64))
        return values_RK, indices_RK

    def provenance(self) -> dict[str, object]:
        library = self._load_library()
        return {
            "provider": self.name,
            "version": getattr(library, "__version__", "unknown"),
            "compile_capability": "unverified",
            "sorted_by_value": False,
            "sorted_by_index": False,
            "indices_dtype": "torch.int32",
            "returns_values": False,
        }
