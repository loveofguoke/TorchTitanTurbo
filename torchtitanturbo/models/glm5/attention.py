# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Explicit NPU Flex adapter for upstream GLM gathered attention configs.

This backend preserves selected-token attention, not gathered-kernel compute
sparsity. It remains separate from Attention Gym's CUDA fused gather backend.
"""

from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config.transform import convert_config_type, ModelConfigTransform
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.glm5.dsa import (
    DsaMetadata,
    filter_dsa_indices,
    Glm5SparseAttention,
)


def build_dsa_block_mask(indices_QS, metadata: DsaMetadata, block_size=128):
    """Build conservative blocks with exact pointwise selected-key membership."""
    indices_QS = filter_dsa_indices(indices_QS, metadata)
    Q, S = indices_QS.shape
    K = metadata.positions_T.numel()
    valid_QS = indices_QS >= 0
    safe_QS = indices_QS.clamp_min(0).long()
    B = block_size
    num_q_blocks, num_k_blocks = (Q + B - 1) // B, (K + B - 1) // B
    block_ids_QS = (
        torch.arange(Q, device=indices_QS.device)[:, None] // B
    ) * num_k_blocks + safe_QS // B
    occupancy = torch.zeros(
        num_q_blocks * num_k_blocks, device=indices_QS.device, dtype=torch.int32
    )
    occupancy.scatter_add_(
        0, block_ids_QS.flatten(), valid_QS.flatten().to(torch.int32)
    )
    occupied = occupancy.view(1, 1, num_q_blocks, num_k_blocks) > 0
    counts = occupied.sum(-1).to(torch.int32)
    blocks = occupied.to(torch.float32).argsort(dim=-1, descending=True, stable=True)
    selected_QS = torch.where(valid_QS, indices_QS, K).sort(dim=-1).values

    def mask_mod(b, h, q, k):
        row = q.clamp(max=Q - 1)
        lower = (q + k) * 0
        upper = lower + S
        for _ in range(S.bit_length()):
            # Non-negative bounds: avoid DVM's non-pointwise floor-div fallback.
            middle = (lower + upper) >> 1
            before = selected_QS[row, middle.clamp(max=S - 1)] < k
            lower = torch.where(before, middle + 1, lower)
            upper = torch.where(before, upper, middle)
        return (
            (q < Q)
            & (k < K)
            & (lower < S)
            & (selected_QS[row, lower.clamp(max=S - 1)] == k)
        )

    return BlockMask.from_kv_blocks(
        counts,
        blocks.to(torch.int32),
        BLOCK_SIZE=B,
        mask_mod=mask_mod,
        seq_lengths=(Q, K),
    )


class NpuGlm5FlexAttention(Glm5SparseAttention):
    """Run GLM selections through the TorchNPU Flex lowering without a fallback."""

    @dataclass(kw_only=True, slots=True)
    class Config(Glm5SparseAttention.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.flex = FlexInnerAttention.Config().build()

    def forward(
        self, q_THK, k_THK, v_THV, *, attention_metadata, topk_indices_QS, scale
    ):
        if q_THK.device.type != "npu":
            raise ValueError("NpuGlm5FlexAttention requires an NPU")
        mask = build_dsa_block_mask(topk_indices_QS, attention_metadata)
        return self.flex(q_THK, k_THK, v_THV, attention_metadata=mask, scale=scale)


@dataclass(kw_only=True, slots=True)
class NpuGlm5FlexTransform(ModelConfigTransform):
    """Select the NPU backend explicitly, without global model monkey patches."""

    def transform(self, model):
        for _, existing, parent, field_name in model.traverse(
            Glm5SparseAttention.Config
        ):
            if type(existing) is not Glm5SparseAttention.Config:
                continue
            assert parent is not None and isinstance(field_name, str)
            setattr(
                parent, field_name, convert_config_type(existing, NpuGlm5FlexAttention)
            )
        return model
