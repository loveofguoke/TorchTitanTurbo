# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Optional NPU fused replacement for common dense SDPA.

This converter targets modules configured as TorchTitan's common
``ScaledDotProductAttention``. It is not the GLM DSA inner-attention path: DSA
has an additional top-k mask contract and requires its own sparse/fused kernel.
Keeping the converter explicit prevents a generic optimization from silently
changing GLM attention semantics.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torch import nn

from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


class NpuScaledDotProductAttention(Module):
    """NPU-optimized attention using torch_npu.npu_fusion_attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # TorchTitan uses [B,S,N,H]; npu_fusion_attention's BNSD layout expects
        # [B,N,S,H]. This is a view transpose rather than a semantic reshaping.
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        _, n_heads, seq_len, head_dim = q.shape

        if scale is None:
            scale = 1.0 / (head_dim**0.5)

        # The fused API uses True for disallowed future positions in sparse
        # causal mode, unlike additive masks that store a large negative value.
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device),
            diagonal=1,
        )

        output = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            n_heads,
            "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=causal_mask,
            gen_mask_parallel=True,
            scale=scale,
            sparse_mode=2,
            sync=True,
        )[0]

        # Transpose back to (bs, seq, heads, dim)
        return output.transpose(1, 2)


class NpuSDPAConverter(ModelConfigConverter):
    """Replace ScaledDotProductAttention with NPU-optimized version."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        self.config = config
        logger.info("NpuSDPAConverter initialized")

    def convert(self, model_config) -> None:
        count = 0
        for fqn, cfg, parent, attr in model_config.traverse(
            ScaledDotProductAttention.Config
        ):
            new_config = NpuScaledDotProductAttention.Config()

            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
            count += 1

        if count > 0:
            logger.info(
                f"Converted {count} ScaledDotProductAttention to NpuScaledDotProductAttention"
            )
