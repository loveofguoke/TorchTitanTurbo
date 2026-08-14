# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU compatibility patch for the GLM-5 MoE router."""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.tools.logging import logger


def _local_gather(scores: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(scores, dim=-1, index=index)


def _gather_router_scores(
    scores_BLE: torch.Tensor,
    topk_expert_ids_BLK: torch.Tensor,
) -> torch.Tensor:
    """Run gather on rank-local shards while preserving DTensor metadata."""
    if not isinstance(scores_BLE, DTensor):
        return _local_gather(scores_BLE, topk_expert_ids_BLK)

    assert isinstance(topk_expert_ids_BLK, DTensor)
    gather_local = local_map(
        _local_gather,
        out_placements=(scores_BLE.placements,),
        in_placements=(scores_BLE.placements, topk_expert_ids_BLK.placements),
        in_grad_placements=(scores_BLE.placements, None),
        device_mesh=scores_BLE.device_mesh,
    )
    return gather_local(scores_BLE, topk_expert_ids_BLK)


class NpuGlm5TokenChoiceTopKRouter(TokenChoiceTopKRouter):
    """GLM-5 router with a local DTensor gather boundary for NPU."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        pass

    def _debug_force_load_balance_routing(
        self, scores_BLE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs, slen, _ = scores_BLE.shape
        topk_expert_ids_BLK = (
            torch.arange(
                bs * slen * self.top_k,
                device=scores_BLE.device,
                dtype=torch.int64,
            ).reshape(bs, slen, self.top_k)
            % self.num_experts
        )
        topk_scores_BLK = _gather_router_scores(
            scores_BLE, topk_expert_ids_BLK
        )
        return topk_expert_ids_BLK, topk_scores_BLK

    def forward(
        self,
        x_BLD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores_BLE = self.gate(x_BLD, compute_dtype=torch.float32)

        if self.score_func == "sigmoid":
            scores_BLE = torch.sigmoid(scores_BLE)
        elif self.score_func == "softmax":
            scores_BLE = F.softmax(scores_BLE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice_BLE = (
            scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_BLE = self._get_node_limited_routing_scores(
                scores_for_choice_BLE
            )
        _, topk_expert_ids_BLK = torch.topk(
            scores_for_choice_BLE,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        topk_scores_BLK = _gather_router_scores(
            scores_BLE, topk_expert_ids_BLK
        )

        if self._debug_force_load_balance:
            topk_expert_ids_BLK, topk_scores_BLK = (
                self._debug_force_load_balance_routing(scores_BLE)
            )

        if self.route_norm:
            denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_BLK = topk_scores_BLK / denominator
        topk_scores_BLK = topk_scores_BLK * self.route_scale

        return topk_scores_BLK, topk_expert_ids_BLK, scores_BLE


def _convert_router_config(
    config: TokenChoiceTopKRouter.Config,
) -> NpuGlm5TokenChoiceTopKRouter.Config:
    config_kwargs = {
        config_field.name: getattr(config, config_field.name)
        for config_field in fields(config)
        if config_field.init
    }
    return NpuGlm5TokenChoiceTopKRouter.Config(**config_kwargs)


def apply_patch() -> None:
    """Use the NPU-safe router only for subsequently built GLM-5 configs."""
    import torchtitan.models.glm5 as glm5_module

    current_factory = glm5_module.make_router_config
    if getattr(current_factory, "_torchtitanturbo_glm5_npu_patched", False):
        return

    def make_router_config(*args, **kwargs):
        return _convert_router_config(current_factory(*args, **kwargs))

    make_router_config._torchtitanturbo_glm5_npu_patched = True
    glm5_module.make_router_config = make_router_config
    logger.info("Patched GLM-5 router gather for NPU")
