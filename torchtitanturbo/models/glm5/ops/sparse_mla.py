# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Opt-in Ascend operator implementation for GLM-5 SparseMLA.

This module is intentionally not imported by ``apply_glm5_patch``. Users must
select the override explicitly, so the existing eager and precision experiment
paths retain their current implementation and numerical behavior.
"""

from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.glm5.model import SparseMLA


def _npu_sparse_mla_forward(
    q_QNH: torch.Tensor,
    kv_K1H: torch.Tensor,
    attention_masks_1QK: torch.Tensor,
    topk_indices_QS: torch.Tensor,
    *,
    scale: float,
    latent_dim: int,
    training: bool,
) -> torch.Tensor:
    """Adapt TorchTitan token-first tensors to the torch_npu BSND API."""
    import torch_npu

    safe_indices_QS = topk_indices_QS.clamp_min(0).long()
    selected_mask_QS = attention_masks_1QK[0].gather(-1, safe_indices_QS)
    valid_QS = (topk_indices_QS >= 0) & (selected_mask_QS == 0)
    sparse_indices_BQ1S = torch.where(
        valid_QS,
        topk_indices_QS,
        torch.full_like(topk_indices_QS, -1),
    ).unsqueeze(0).unsqueeze(2).to(torch.int32).contiguous()

    q_BQNH = q_QNH.unsqueeze(0)
    kv_BK1H = kv_K1H.unsqueeze(0)
    q_nope_BQNC, q_rope_BQNR = torch.split(
        q_BQNH, [latent_dim, q_QNH.shape[-1] - latent_dim], dim=-1
    )
    kv_nope_BK1C, kv_rope_BK1R = torch.split(
        kv_BK1H, [latent_dim, kv_K1H.shape[-1] - latent_dim], dim=-1
    )
    result = torch_npu.npu_sparse_flash_attention(
        q_nope_BQNC.contiguous(),
        kv_nope_BK1C.contiguous(),
        kv_nope_BK1C.contiguous(),
        sparse_indices=sparse_indices_BQ1S,
        block_table=None,
        query_rope=q_rope_BQNR.contiguous(),
        key_rope=kv_rope_BK1R.contiguous(),
        scale_value=scale,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        # The supplied indices already encode causal and packed-document
        # boundaries. Applying the operator's global causal mask again is
        # incorrect for local CP queries and reset packed positions.
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=training,
    )
    output_BQNC = result[0] if isinstance(result, tuple) else result
    return output_BQNC.squeeze(0)


class NpuSparseMLA(SparseMLA):
    """Run absorbed GLM-5 SparseMLA with the torch_npu fused operator."""

    @dataclass(kw_only=True, slots=True)
    class Config(SparseMLA.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.attention_dropout != 0.0:
            raise ValueError("NPU SparseMLA does not support attention dropout.")

    def forward(
        self,
        q_QNH: torch.Tensor,
        kv_K1H: torch.Tensor,
        attention_masks_1QK: torch.Tensor,
        topk_indices_QS: torch.Tensor,
        *,
        scale: float,
        latent_dim: int,
    ) -> torch.Tensor:
        if q_QNH.device.type != "npu" or kv_K1H.device.type != "npu":
            raise RuntimeError("NPU SparseMLA requires q and kv on an NPU device.")
        if q_QNH.dtype != torch.bfloat16 or kv_K1H.dtype != torch.bfloat16:
            raise ValueError("NPU SparseMLA requires BF16 q and kv tensors.")
        if q_QNH.shape[-1] != kv_K1H.shape[-1]:
            raise ValueError(
                "NPU SparseMLA requires matching q and kv head dimensions."
            )

        return _npu_sparse_mla_forward(
            q_QNH,
            kv_K1H,
            attention_masks_1QK,
            topk_indices_QS,
            scale=scale,
            latent_dim=latent_dim,
            training=self.training,
        )


@override(
    target=SparseMLA.Config,
    description="Use torch_npu's fused SparseMLA operator for GLM-5.",
)
def npu_sparse_mla(cfg: SparseMLA.Config) -> NpuSparseMLA.Config:
    """Replace only the explicitly selected GLM-5 SparseMLA configs."""
    return derive(cfg, NpuSparseMLA.Config)
