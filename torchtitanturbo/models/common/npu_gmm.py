# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import logging
from dataclasses import dataclass

import torch
import torch_npu
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


def npu_grouped_mm(x, weight, group_list):
    return torch._grouped_mm(x, weight, group_list)


def _run_experts_grouped_mm(
    w13: torch.Tensor,
    w2: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    # pyrefly: ignore [missing-attribute]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = npu_grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offsets)
    if swiglu_limit is not None:
        gate, up = h.chunk(2, -1)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
        h = torch.cat([gate, up], dim=-1)
    h = torch_npu.npu_swiglu(h, dim=-1)
    out = npu_grouped_mm(h, w2.bfloat16().transpose(-2, -1), offsets).type_as(x)

    return out


class NpuGroupedExperts(Module):
    """NPU-optimized GroupedExperts with fused w13 parameter."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        token_dispatcher: GroupedExperts.Config.token_dispatcher
        swiglu_limit: float | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.swiglu_limit = config.swiglu_limit

        num_experts = config.num_experts
        dim = config.dim
        hidden_dim = config.hidden_dim

        self.w13 = nn.Parameter(torch.empty(num_experts, hidden_dim * 2, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))

        self.token_dispatcher = config.token_dispatcher.build()
        logger.info(
            f"NpuGroupedExperts initialized: w13 [{num_experts}, {hidden_dim * 2}, {dim}], "
            f"w2 [{num_experts}, {dim}, {hidden_dim}]"
        )

    def _experts_forward(self, x, num_tokens_per_expert):
        is_tp = False
        if isinstance(self.w2, DTensor):
            w2 = self.w2.to_local()
            w13 = self.w13.to_local()
            from torch.distributed.tensor.placement_types import Shard as _Shard

            for p in self.w2.placements:
                if isinstance(p, _Shard) and p.dim == 2:
                    is_tp = True
                    break
            tp_group = self.w2.device_mesh.get_group() if is_tp else None
        else:
            w2 = self.w2
            w13 = self.w13
            tp_group = None

        out = _run_experts_grouped_mm(
            w13, w2, x, num_tokens_per_expert, self.swiglu_limit
        )

        if is_tp and tp_group is not None:
            import torch.distributed as dist

            dist.all_reduce(out, group=tp_group)

        return out

    def forward(self, x, top_scores, selected_experts_indices):
        bs, slen, dim = x.shape
        top_k = top_scores.size(-1)
        x = x.view(bs * slen, dim)
        top_scores = top_scores.view(bs * slen, top_k)
        selected_experts_indices = selected_experts_indices.view(bs * slen, top_k)
        routed_input, num_tokens_local, metadata = self.token_dispatcher.dispatch(
            x, top_scores, selected_experts_indices
        )
        routed_output = self._experts_forward(routed_input, num_tokens_local)
        return self.token_dispatcher.combine(routed_output, metadata, x)

    def _init_self_parameters(self):
        for name, param in self.named_parameters(recurse=False):
            if name in ("w13", "w2"):
                nn.init.normal_(param, mean=0.0, std=0.02)


class NpuGroupedExpertsConverter(ModelConfigConverter):
    """Replace GroupedExperts with NPU-optimized NpuGroupedExperts."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        swiglu_limit: float | None = None

    def __init__(self, config: Config):
        self.config = config
        self.swiglu_limit = config.swiglu_limit
        logger.info("NpuGroupedExpertsConverter initialized")

    def convert(self, model_config) -> None:
        count = 0
        for fqn, cfg, parent, attr in model_config.traverse(GroupedExperts.Config):
            new_config = NpuGroupedExperts.Config(
                dim=cfg.dim,
                hidden_dim=cfg.hidden_dim,
                num_experts=cfg.num_experts,
                token_dispatcher=cfg.token_dispatcher,
                swiglu_limit=self.swiglu_limit,
            )
            if hasattr(cfg, "param_init"):
                new_config.param_init = cfg.param_init
            if hasattr(cfg, "sharding_config"):
                new_config.sharding_config = cfg.sharding_config

            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
            count += 1

        if count > 0:
            logger.info(f"Converted {count} GroupedExperts to NpuGroupedExperts")
