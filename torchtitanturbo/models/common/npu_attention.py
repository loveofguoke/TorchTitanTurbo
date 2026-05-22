# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU-optimized ScaledDotProductAttention using torch_npu.npu_fusion_attention."""

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
        _, n_heads, seq_len, head_dim = q.shape

        if scale is None:
            scale = 1.0 / (head_dim**0.5)

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

        return output


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
