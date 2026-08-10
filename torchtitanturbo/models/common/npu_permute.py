# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass

import torch
import torch_npu
from torch.distributed._functional_collectives import all_to_all_single

from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.tools.logging import logger


class NPUMoeTokenUnpermute(torch.autograd.Function):
    """Functional wrapper for npu_moe_token_unpermute with proper backward."""

    @staticmethod
    def forward(ctx, permuted_tokens, sorted_indices, restore_shape):
        if not permuted_tokens.numel():
            return permuted_tokens

        output, _, _, _ = torch_npu._npu_moe_token_unpermute_with_routing_map(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=None,
            routing_map=None,
            drop_and_pad=False,
        )
        ctx.restore_shape = restore_shape
        ctx.sorted_indices = sorted_indices
        return output

    @staticmethod
    def backward(ctx, unpermuted_tokens_grad):
        if not unpermuted_tokens_grad.numel():
            return unpermuted_tokens_grad, None, None

        if ctx.needs_input_grad[0]:
            act_grad, _ = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
                unpermuted_tokens_grad,
                ctx.sorted_indices,
                ctx.sorted_indices,
                routing_map=None,
                permuted_tokens=None,
                probs=None,
                drop_and_pad=False,
                restore_shape=ctx.restore_shape,
            )
            return act_grad, None, None

        return None, None, None


class NpuTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher using NPU fused permute/unpermute operators."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pass

    def _permute(
        self, routed_input, num_tokens_per_expert_group, ep_size, num_local_experts
    ):
        device = routed_input.device

        indices = (
            torch.arange(num_local_experts, dtype=torch.int64, device=device)
            .repeat(ep_size)
            .repeat_interleave(num_tokens_per_expert_group.view(-1))
        )

        routed_input, permuted_indices = torch_npu.npu_moe_token_permute(
            routed_input, indices
        )

        num_tokens_per_expert = num_tokens_per_expert_group.view(ep_size, -1).sum(0)

        return (
            routed_input.shape,
            routed_input,
            permuted_indices,
            num_tokens_per_expert,
        )

    def _unpermute(self, routed_output, input_shape, permuted_indices):
        return NPUMoeTokenUnpermute.apply(routed_output, permuted_indices, input_shape)


class NpuTokenDispatcherConverter(ModelConfigConverter):
    """Replace AllToAllTokenDispatcher with NpuTokenDispatcher."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        self.config = config
        logger.info("NpuTokenDispatcherConverter initialized")

    def convert(self, model_config) -> None:
        count = 0
        for fqn, cfg, parent, attr in model_config.traverse(RoutedExperts.Config):
            if isinstance(cfg.token_dispatcher, AllToAllTokenDispatcher.Config):
                new_dispatcher = NpuTokenDispatcher.Config(
                    num_experts=cfg.token_dispatcher.num_experts,
                    top_k=cfg.token_dispatcher.top_k,
                    score_before_experts=cfg.token_dispatcher.score_before_experts,
                )
                cfg.token_dispatcher = new_dispatcher
                count += 1

        if count > 0:
            logger.info(
                f"Converted {count} AllToAllTokenDispatcher to NpuTokenDispatcher"
            )
