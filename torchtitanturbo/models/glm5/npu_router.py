# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU-safe GLM-5 MoE routing with an explicit FP32 gate."""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.tools.logging import logger


def _local_gather(scores: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(scores, dim=-1, index=index)


def _gather_router_scores(
    scores_TE: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
) -> torch.Tensor:
    """Gather rank-local scores while retaining DTensor placement metadata."""
    if not isinstance(scores_TE, DTensor):
        return _local_gather(scores_TE, topk_expert_ids_TK)

    if not isinstance(topk_expert_ids_TK, DTensor):
        raise TypeError("DTensor router scores require DTensor routing indices")
    gather_local = local_map(
        _local_gather,
        out_placements=(scores_TE.placements,),
        in_placements=(scores_TE.placements, topk_expert_ids_TK.placements),
        in_grad_placements=(
            scores_TE.placements,
            topk_expert_ids_TK.placements,
        ),
        device_mesh=scores_TE.device_mesh,
    )
    return gather_local(scores_TE, topk_expert_ids_TK)


class NpuFp32RouterLinear(Linear):
    """Linear gate whose explicit compute-dtype path cannot be autocast down."""

    supports_compute_dtype = True

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(
        self,
        input: torch.Tensor,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if compute_dtype is None:
            return super().forward(input)
        bias = None if self.bias is None else self.bias.to(dtype=compute_dtype)
        with torch.autocast(device_type=input.device.type, enabled=False):
            return F.linear(
                input.to(dtype=compute_dtype),
                self.weight.to(dtype=compute_dtype),
                bias,
            )


class NpuGlm5TokenChoiceTopKRouter(TokenChoiceTopKRouter):
    """GLM-5 router using a stable FP32 gate and an NPU-safe gather."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        pass

    def _debug_force_load_balance_routing(
        self, scores_TE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_shape = scores_TE.shape[:-1]
        num_tokens = scores_TE.numel() // scores_TE.shape[-1]
        topk_expert_ids_TK = (
            torch.arange(
                num_tokens * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            ).reshape(*token_shape, self.top_k)
            % self.num_experts
        )
        topk_scores_TK = _gather_router_scores(
            scores_TE, topk_expert_ids_TK
        )
        return topk_expert_ids_TK, topk_scores_TK

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores_TE = self.gate(x_TD, compute_dtype=torch.float32)

        if self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_TE = self._get_node_limited_routing_scores(
                scores_for_choice_TE
            )
        _, topk_expert_ids_TK = torch.topk(
            scores_for_choice_TE,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        topk_scores_TK = _gather_router_scores(
            scores_TE, topk_expert_ids_TK
        )

        if self._debug_force_load_balance:
            topk_expert_ids_TK, topk_scores_TK = (
                self._debug_force_load_balance_routing(scores_TE)
            )

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator
        topk_scores_TK = topk_scores_TK * self.route_scale

        return topk_scores_TK, topk_expert_ids_TK, scores_TE


def _convert_router_config(
    config: TokenChoiceTopKRouter.Config,
) -> NpuGlm5TokenChoiceTopKRouter.Config:
    config_kwargs = {
        config_field.name: getattr(config, config_field.name)
        for config_field in fields(config)
        if config_field.init
    }
    gate_kwargs = {
        config_field.name: getattr(config.gate, config_field.name)
        for config_field in fields(config.gate)
        if config_field.init
    }
    config_kwargs["gate"] = NpuFp32RouterLinear.Config(**gate_kwargs)
    return NpuGlm5TokenChoiceTopKRouter.Config(**config_kwargs)


def apply_patch() -> None:
    """Use the NPU-safe router for subsequently constructed GLM-5 configs."""
    import torchtitan.models.glm5 as glm5_module

    current_factory = glm5_module.make_router_config
    if getattr(current_factory, "_torchtitanturbo_glm5_router_patched", False):
        return

    def make_router_config(*args, **kwargs):
        return _convert_router_config(current_factory(*args, **kwargs))

    make_router_config._torchtitanturbo_glm5_router_patched = True
    glm5_module.make_router_config = make_router_config
    logger.info("Patched GLM-5 router FP32 gate and NPU gather")
