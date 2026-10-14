"""Forward-only FlashMLA Ascend sparse-prefill provider."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

import torch


class FlashMlaCapabilityError(ValueError):
    """The requested sparse-attention contract is unsupported."""


@dataclass(frozen=True, slots=True)
class FlashMlaSparseRequest:
    platform: str
    query_dtype: torch.dtype
    kv_dtype: torch.dtype
    indices_dtype: torch.dtype
    num_query_heads: int
    query_key_dim: int
    value_dim: int
    requires_grad: bool


def validate_flashmla_sparse_request(request: FlashMlaSparseRequest) -> None:
    if request.platform != "npu":
        raise FlashMlaCapabilityError(
            "The FlashMLA Ascend sparse provider requires NPU tensors."
        )
    if request.query_dtype != torch.bfloat16 or request.kv_dtype != torch.bfloat16:
        raise FlashMlaCapabilityError(
            "FlashMLA Ascend sparse prefill requires BF16 query and KV tensors."
        )
    if request.indices_dtype != torch.int32:
        raise FlashMlaCapabilityError(
            "FlashMLA Ascend sparse indices must use torch.int32."
        )
    if request.num_query_heads not in {64, 128}:
        raise FlashMlaCapabilityError(
            "FlashMLA Ascend sparse prefill supports 64 or 128 query heads."
        )
    if request.query_key_dim != 512 or request.value_dim != 512:
        raise FlashMlaCapabilityError(
            "FlashMLA Ascend sparse prefill requires d_qk=512 and d_v=512."
        )
    if request.requires_grad:
        raise FlashMlaCapabilityError(
            "The public Ascend sparse-prefill API is forward-only for this "
            "integration. Training requires a compatible backward kernel."
        )


class FlashMlaAscendSparseProvider:
    """Invoke the official Ascend sparse-prefill API after strict validation."""

    name = "flashmla_ascend_sparse_prefill"

    def __init__(self, library: ModuleType | None = None) -> None:
        self._library = library

    def _load_library(self) -> ModuleType:
        if self._library is None:
            try:
                self._library = import_module("flash_mla")
            except ImportError as error:
                raise RuntimeError(
                    "FlashMLA was selected but flash_mla is not installed. "
                    "Install the pinned Ascend source revision."
                ) from error
        return self._library

    def forward(
        self,
        *,
        q_QHD: torch.Tensor,
        kv_KGD: torch.Tensor,
        indices_QGS: torch.Tensor,
        softmax_scale: float,
        value_dim: int = 512,
        attention_sink_H: torch.Tensor | None = None,
        topk_length_Q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q_QHD.ndim != 3 or kv_KGD.ndim != 3 or indices_QGS.ndim != 3:
            raise FlashMlaCapabilityError(
                "Expected q[Q,H,D], kv[K,G,D], and indices[Q,G,S]."
            )
        if q_QHD.device != kv_KGD.device or q_QHD.device != indices_QGS.device:
            raise FlashMlaCapabilityError(
                "q, kv, and indices must share one NPU device."
            )
        if q_QHD.shape[0] != indices_QGS.shape[0]:
            raise FlashMlaCapabilityError("q and indices query dimensions disagree.")
        if kv_KGD.shape[1] != indices_QGS.shape[1]:
            raise FlashMlaCapabilityError("KV heads and index groups disagree.")
        if q_QHD.shape[2] != kv_KGD.shape[2]:
            raise FlashMlaCapabilityError("q and kv query-key dimensions disagree.")

        requires_grad = q_QHD.requires_grad or kv_KGD.requires_grad
        validate_flashmla_sparse_request(
            FlashMlaSparseRequest(
                platform=q_QHD.device.type,
                query_dtype=q_QHD.dtype,
                kv_dtype=kv_KGD.dtype,
                indices_dtype=indices_QGS.dtype,
                num_query_heads=q_QHD.shape[1],
                query_key_dim=q_QHD.shape[2],
                value_dim=value_dim,
                requires_grad=requires_grad,
            )
        )
        return self._load_library().flash_mla_sparse_fwd(
            q_QHD,
            kv_KGD,
            indices_QGS,
            softmax_scale,
            value_dim,
            attention_sink_H,
            topk_length_Q,
        )

    def provenance(self) -> dict[str, object]:
        library = self._load_library()
        return {
            "provider": self.name,
            "version": getattr(library, "__version__", "unknown"),
            "training_backward": False,
            "compile_capability": "unverified",
            "api": "flash_mla_sparse_fwd",
            "query_key_dim": 512,
            "value_dim": 512,
        }
