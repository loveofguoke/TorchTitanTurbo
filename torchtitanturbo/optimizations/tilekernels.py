"""Capability-checked TileKernels MoE routing provider."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

import torch


class TileKernelsCapabilityError(ValueError):
    """The requested routing contract is not supported by TileKernels."""


@dataclass(frozen=True, slots=True)
class TileKernelsTopKRequest:
    platform: str
    dtype: torch.dtype
    num_tokens: int
    num_experts: int
    topk: int


def validate_tilekernels_topk_request(request: TileKernelsTopKRequest) -> None:
    if request.platform not in {"cuda", "npu"}:
        raise TileKernelsCapabilityError(
            "TileKernels MoE TopK supports CUDA and NPU only; "
            f"got {request.platform!r}."
        )
    if request.dtype != torch.float32:
        raise TileKernelsCapabilityError(
            "TileKernels topk_gate requires FP32 scores; "
            f"got {request.dtype}."
        )
    if request.num_tokens < 0 or request.num_experts <= 0:
        raise TileKernelsCapabilityError(
            "num_tokens must be non-negative and num_experts must be positive."
        )
    if not 0 < request.topk <= min(32, request.num_experts):
        raise TileKernelsCapabilityError(
            "TileKernels topk_gate requires "
            "0 < topk <= min(32, num_experts)."
        )


class TileKernelsMoeTopKProvider:
    """Return the TileKernels TopK expert index set without global patching."""

    name = "tilekernels_moe_topk"

    def __init__(self, library: ModuleType | None = None) -> None:
        self._library = library

    def _load_library(self) -> ModuleType:
        if self._library is None:
            try:
                self._library = import_module("tile_kernels.moe")
            except ImportError as error:
                raise RuntimeError(
                    "TileKernels was selected but tile_kernels is not "
                    "installed. Install the pinned package for the active "
                    "CUDA or Ascend toolchain."
                ) from error
        return self._library

    def indices(self, scores_TE: torch.Tensor, topk: int) -> torch.Tensor:
        if scores_TE.ndim != 2 or not scores_TE.is_contiguous():
            raise TileKernelsCapabilityError(
                "TileKernels requires contiguous scores[T,E]."
            )
        validate_tilekernels_topk_request(
            TileKernelsTopKRequest(
                platform=scores_TE.device.type,
                dtype=scores_TE.dtype,
                num_tokens=scores_TE.shape[0],
                num_experts=scores_TE.shape[1],
                topk=topk,
            )
        )
        return self._load_library().topk_gate(scores_TE, topk)

    def select(
        self, scores_TE: torch.Tensor, topk: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts in the kernel and gather differentiable scores."""
        indices_TK = self.indices(scores_TE, topk)
        values_TK = scores_TE.gather(-1, indices_TK.to(torch.int64))
        return values_TK, indices_TK

    def provenance(self) -> dict[str, object]:
        library = self._load_library()
        return {
            "provider": self.name,
            "version": getattr(library, "__version__", "unknown"),
            "compile_capability": "unverified",
            "input_dtype": "torch.float32",
            "max_topk": 32,
            "tie_break": "smaller_index",
        }
