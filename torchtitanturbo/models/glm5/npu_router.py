# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU-safe GLM-5 MoE routing with an explicit FP32 gate."""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
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


def _local_routing_map(
    scores: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    return torch.zeros_like(scores, dtype=torch.bool).scatter(
        -1, expert_ids, True
    )


def _routing_map(
    scores: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    """Build the one-hot map locally while preserving token sharding."""
    if not isinstance(scores, DTensor):
        return _local_routing_map(scores, expert_ids)
    if not isinstance(expert_ids, DTensor):
        raise TypeError("DTensor router scores require DTensor routing indices")
    scatter_local = local_map(
        _local_routing_map,
        out_placements=(scores.placements,),
        in_placements=(scores.placements, expert_ids.placements),
        in_grad_placements=None,
        device_mesh=scores.device_mesh,
    )
    return scatter_local(scores, expert_ids)


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

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice_TE: torch.Tensor,
    ) -> torch.Tensor:
        """Apply group limiting without DTensor-incompatible in-place scatter."""
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
        )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(
                f"experts_per_group ({experts_per_group}) must be >= 2"
            )
        scores_grouped = scores_for_choice_TE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores,
            k=self.num_limited_groups,
            dim=-1,
            sorted=False,
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool).scatter(
            -1, group_idx, False
        )
        return scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

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


def _npu_moe_forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
    """MoE forward with a DTensor-safe out-of-place routing scatter.

    TorchTitan's current GLM5 path is token-first ``x_TD`` while older
    revisions use batched ``x_BLD``. Keep both interfaces working so the NPU
    routing fix does not change the model's public tensor contract.
    """
    if x_BLD.ndim == 2:
        x_TD = x_BLD
        topk_scores_TK, topk_expert_ids_TK, scores_TE = self.router(
            x_TD, self.expert_bias_E
        )
        routing_map_TE = _routing_map(scores_TE, topk_expert_ids_TK)
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        if self.shared_experts is not None:
            out_TD = out_TD + self.shared_experts(x_TD)
        return out_TD
    if x_BLD.ndim != 3:
        raise ValueError(
            "GLM5 MoE input must be token-first [T, D] or batched [B, L, D], "
            f"got shape {tuple(x_BLD.shape)}"
        )

    B, L, _ = x_BLD.shape
    sp_size = getattr(self.routed_experts.token_dispatcher, "sp_size", 1)
    if not isinstance(x_BLD, DTensor) and self.seq_dim_tp_sharded:
        seq_pad = 0
        seq_dim_pad_tokens = 0
        num_local_tokens_after_seq_dim_padding = B * L
    else:
        seq_pad = sp_size - L if L < sp_size else 0
        if seq_pad:
            x_BLD = F.pad(x_BLD, (0, 0, 0, seq_pad))
            L += seq_pad
        seq_dim_pad_tokens = (-L) % sp_size
        local_batch_size = (
            x_BLD._local_tensor.shape[0] if isinstance(x_BLD, DTensor) else B
        )
        num_local_tokens_after_seq_dim_padding = (
            local_batch_size * (L + seq_dim_pad_tokens) // sp_size
        )

    topk_scores_BLK, topk_expert_ids_BLK, scores_BLE = self.router(
        x_BLD, self.expert_bias_E
    )
    routing_map_BLE = _routing_map(scores_BLE, topk_expert_ids_BLK)
    num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))
    with torch.no_grad():
        self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

    out_BLD = self.routed_experts(
        x_BLD,
        topk_scores_BLK,
        topk_expert_ids_BLK,
        num_local_tokens_per_expert_E,
        num_local_tokens_after_seq_dim_padding=(
            num_local_tokens_after_seq_dim_padding
        ),
    )
    shared_out_BLD = (
        self.shared_experts(x_BLD) if self.shared_experts is not None else None
    )
    if (
        isinstance(self.routed_experts.token_dispatcher, DeepEPTokenDispatcher)
        and self.routed_experts.token_dispatcher.sp_size == 1
    ):
        from torchtitan.distributed.deepep.deepep import sync_combine

        sync_combine()
    if seq_dim_pad_tokens:
        out_BLD = out_BLD[:, :L, :]
    if shared_out_BLD is not None:
        out_BLD = out_BLD + shared_out_BLD
    if seq_pad:
        out_BLD = out_BLD[:, : L - seq_pad, :]
    return out_BLD


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
    import torchtitan.models.common.moe as moe_module
    import torchtitan.models.glm5 as glm5_module

    current_factory = glm5_module.make_router_config
    if getattr(current_factory, "_torchtitanturbo_glm5_router_patched", False):
        return

    def make_router_config(*args, **kwargs):
        return _convert_router_config(current_factory(*args, **kwargs))

    make_router_config._torchtitanturbo_glm5_router_patched = True
    glm5_module.make_router_config = make_router_config
    if not getattr(moe_module.MoE.forward, "_torchtitanturbo_npu_patched", False):
        _npu_moe_forward._torchtitanturbo_npu_patched = True
        moe_module.MoE.forward = _npu_moe_forward
    logger.info("Patched GLM-5 router FP32 gate and NPU gather")
