"""Autograd-aware DeepGEMM-Ascend grouped-BF16 provider."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

import torch


class DeepGemmCapabilityError(ValueError):
    """The requested grouped GEMM contract is not supported by the provider."""


@dataclass(frozen=True, slots=True)
class DeepGemmGroupedRequest:
    platform: str
    input_dtype: torch.dtype
    weight_dtype: torch.dtype
    offsets_dtype: torch.dtype
    num_tokens: int
    input_dim: int
    output_dim: int
    num_experts: int
    requires_grad: bool


def validate_deepgemm_grouped_request(request: DeepGemmGroupedRequest) -> None:
    """Validate the DeepGEMM Ascend grouped training contract."""
    if request.platform != "npu":
        raise DeepGemmCapabilityError(
            "DeepGEMM-Ascend requires NPU tensors; "
            f"got platform={request.platform!r}."
        )
    if request.input_dtype != torch.bfloat16:
        raise DeepGemmCapabilityError(
            "The DeepGEMM provider supports BF16 activations only; "
            f"got {request.input_dtype}."
        )
    if request.weight_dtype != torch.bfloat16:
        raise DeepGemmCapabilityError(
            "The DeepGEMM provider supports BF16 weights only; "
            f"got {request.weight_dtype}."
        )
    if request.offsets_dtype != torch.int32:
        raise DeepGemmCapabilityError(
            "DeepGEMM grouped layout offsets must use torch.int32; "
            f"got {request.offsets_dtype}."
        )
    if request.num_tokens <= 0:
        raise DeepGemmCapabilityError(
            "The provider requires at least one routed token. "
            "All-empty local expert batches require separate backend validation."
        )
    if min(request.input_dim, request.output_dim, request.num_experts) <= 0:
        raise DeepGemmCapabilityError(
            "input_dim, output_dim, and num_experts must be positive."
        )


def _offsets_to_counts(offsets_E: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Convert TorchTitan cumulative offsets to DeepGEMM group row counts."""
    if offsets_E.numel() == 0:
        raise DeepGemmCapabilityError("Grouped GEMM requires at least one expert.")
    counts_E = torch.diff(offsets_E, prepend=offsets_E.new_zeros(1))
    if bool((counts_E < 0).any().item()):
        raise DeepGemmCapabilityError("Grouped GEMM offsets must be nondecreasing.")
    if int(offsets_E[-1].item()) != num_tokens:
        raise DeepGemmCapabilityError(
            "The final cumulative offset must equal the routed token count; "
            f"got {int(offsets_E[-1].item())} and {num_tokens}."
        )
    return counts_E


class _DeepGemmGroupedBf16(torch.autograd.Function):
    """Compose the official forward, input-gradient, and weight-gradient APIs."""

    @staticmethod
    def forward(ctx, library, A_MK, B_t_EKN, offsets_E):
        B_ENK = B_t_EKN.transpose(-2, -1)
        output_MN = torch.empty(
            (A_MK.shape[0], B_t_EKN.shape[-1]),
            dtype=torch.bfloat16,
            device=A_MK.device,
        )
        library.m_grouped_bf16_gemm_nt_contiguous(
            A_MK, B_ENK, output_MN, offsets_E
        )
        ctx.library = library
        ctx.save_for_backward(A_MK, B_t_EKN, offsets_E)
        return output_MN

    @staticmethod
    def backward(ctx, grad_output_MN):
        A_MK, B_t_EKN, offsets_E = ctx.saved_tensors
        library = ctx.library
        grad_output_MN = grad_output_MN.contiguous().to(torch.bfloat16)

        grad_A_MK = torch.empty_like(A_MK)
        B_ENK = B_t_EKN.transpose(-2, -1)
        library.m_grouped_bf16_gemm_nn_contiguous(
            grad_output_MN,
            B_ENK,
            grad_A_MK,
            offsets_E,
        )

        grad_B_ENK = torch.empty(
            (B_t_EKN.shape[0], B_t_EKN.shape[2], B_t_EKN.shape[1]),
            dtype=B_t_EKN.dtype,
            device=B_t_EKN.device,
        )
        counts_E = _offsets_to_counts(offsets_E, A_MK.shape[0])
        counts_cpu = [int(value) for value in counts_E.cpu().tolist()]
        library.k_grouped_bf16_gemm_tn_contiguous(
            grad_output_MN,
            A_MK,
            grad_B_ENK,
            counts_cpu,
            offsets_E,
        )
        grad_B_t_EKN = grad_B_ENK.transpose(-2, -1)
        return None, grad_A_MK, grad_B_t_EKN, None


class DeepGemmGroupedBf16Provider:
    """Adapt TorchTitan grouped-MM inputs to DeepGEMM training APIs."""

    name = "deepgemm_ascend_grouped_bf16"

    def __init__(self, library: ModuleType | None = None) -> None:
        self._library = library

    def _load_library(self) -> ModuleType:
        if self._library is None:
            try:
                self._library = import_module("deep_gemm")
            except ImportError as error:
                raise RuntimeError(
                    "DeepGEMM-Ascend was selected but the deep_gemm package "
                    "is not installed. Install the pinned Ascend source "
                    "revision in a compatible CANN 9.2 environment."
                ) from error
        return self._library

    def forward(
        self,
        *,
        A_MK: torch.Tensor,
        B_t_EKN: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        """Compute grouped ``A @ B_t`` with explicit dA and dB kernels."""
        if A_MK.ndim != 2 or B_t_EKN.ndim != 3 or offsets_E.ndim != 1:
            raise DeepGemmCapabilityError(
                "Expected A[M,K], B_t[E,K,N], and offsets[E]; got "
                f"{tuple(A_MK.shape)}, {tuple(B_t_EKN.shape)}, "
                f"and {tuple(offsets_E.shape)}."
            )
        num_experts, weight_input_dim, output_dim = B_t_EKN.shape
        if A_MK.shape[1] != weight_input_dim:
            raise DeepGemmCapabilityError(
                "Grouped GEMM input dimensions disagree: "
                f"A.shape[1]={A_MK.shape[1]} and "
                f"B_t.shape[1]={weight_input_dim}."
            )
        if offsets_E.shape[0] != num_experts:
            raise DeepGemmCapabilityError(
                "Grouped GEMM requires one cumulative offset per expert; "
                f"got {offsets_E.shape[0]} offsets for {num_experts} experts."
            )
        devices = {A_MK.device, B_t_EKN.device, offsets_E.device}
        if len(devices) != 1:
            raise DeepGemmCapabilityError(
                "A, B_t, and offsets must be on the same NPU device."
            )

        validate_deepgemm_grouped_request(
            DeepGemmGroupedRequest(
                platform=A_MK.device.type,
                input_dtype=A_MK.dtype,
                weight_dtype=B_t_EKN.dtype,
                offsets_dtype=offsets_E.dtype,
                num_tokens=A_MK.shape[0],
                input_dim=A_MK.shape[1],
                output_dim=output_dim,
                num_experts=num_experts,
                requires_grad=A_MK.requires_grad or B_t_EKN.requires_grad,
            )
        )
        if not A_MK.is_contiguous():
            raise DeepGemmCapabilityError("DeepGEMM requires contiguous A[M,K].")

        # TorchTitan passes B_t[E,K,N] for A @ B_t. DeepGEMM's NT API takes
        # the untransposed storage B[E,N,K] and performs the transpose itself.
        B_ENK = B_t_EKN.transpose(-2, -1)
        if not B_ENK.is_contiguous():
            raise DeepGemmCapabilityError(
                "DeepGEMM requires B[E,N,K] to be contiguous. Keep expert "
                "weights in their native [E,N,K] storage before passing B_t."
            )

        _offsets_to_counts(offsets_E, A_MK.shape[0])
        library = self._load_library()
        return _DeepGemmGroupedBf16.apply(library, A_MK, B_t_EKN, offsets_E)

    def provenance(self) -> dict[str, object]:
        library = self._load_library()
        return {
            "provider": self.name,
            "version": getattr(library, "__version__", "unknown"),
            "training_backward": True,
            "compile_capability": "unverified",
            "input_dtype": "torch.bfloat16",
            "weight_dtype": "torch.bfloat16",
            "offsets_dtype": "torch.int32",
            "api": "m_grouped_bf16_gemm_nt_contiguous",
            "input_gradient_api": "m_grouped_bf16_gemm_nn_contiguous",
            "weight_gradient_api": "k_grouped_bf16_gemm_tn_contiguous",
        }
