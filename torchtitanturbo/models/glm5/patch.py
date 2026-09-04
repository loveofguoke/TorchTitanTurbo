# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU compatibility patches for GLM-5 distributed sharding."""

from dataclasses import fields

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout
from torchtitan.tools.logging import logger


def _replicated_routing_counts() -> SpmdLayout:
    return SpmdLayout(
        {
            MeshAxisName.DP: spmd.P,
            MeshAxisName.CP: spmd.P,
            MeshAxisName.TP: spmd.R,
        }
    )


def _fix_ep_without_sp(moe_cfg) -> None:
    activation = dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    routing_counts = _replicated_routing_counts()
    moe_cfg.sharding_config.state_shardings["tokens_per_expert_E"] = routing_counts

    gate_state = moe_cfg.router.gate.sharding_config.state_shardings
    moe_cfg.router.gate.sharding_config = ShardingConfig(
        state_shardings=gate_state,
        in_src_shardings={"input": activation},
        in_dst_shardings={"input": activation},
        out_src_shardings=activation,
        out_dst_shardings=activation,
    )

    routed_cfg = moe_cfg.routed_experts.sharding_config
    routed_cfg.in_src_shardings = {
        "x_TD": activation,
        "topk_scores_TK": activation,
        "topk_expert_ids_TK": activation,
        "num_local_tokens_per_expert_E": routing_counts,
    }
    routed_cfg.in_dst_shardings = dict(routed_cfg.in_src_shardings)
    routed_cfg.local_map = LocalMapConfig(
        in_grad_placements=(activation, activation, activation, routing_counts)
    )


def _fix_tp_only_routed_expert_grad_layout(moe_cfg) -> None:
    """Mark TP-local RoutedExperts input gradients as Partial on TP.

    Routed expert weights are tensor-parallel sharded while router inputs and
    scores are replicated. Each TP rank therefore produces only a partial
    contribution to their gradients; local_map must retain that placement so
    the replicated router gate receives the TP sum during backward.
    """

    routed_cfg = moe_cfg.routed_experts.sharding_config
    routed_inputs = routed_cfg.in_dst_shardings or routed_cfg.in_src_shardings
    if routed_inputs is None or "x_BLD" not in routed_inputs:
        return
    routing_counts = routed_inputs["num_local_tokens_per_expert_E"]
    partial_activation = dense_activation_placement(tp=spmd.P)
    routed_cfg.local_map = LocalMapConfig(
        in_grad_placements=(
            partial_activation,
            partial_activation,
            partial_activation,
            routing_counts,
        )
    )


def apply_patch():
    """Patch GLM-5 EP layouts for the supported no-SP configuration."""
    import torchtitan.models.common.moe as moe_module
    import torchtitan.models.glm5.sharding as glm5_sharding

    from .npu_scatter_add import apply_patch as apply_npu_scatter_add_patch
    from .npu_router import apply_patch as apply_npu_router_patch

    apply_npu_scatter_add_patch()
    apply_npu_router_patch()

    if getattr(
        glm5_sharding.set_moe_sharding_config,
        "_torchtitanturbo_patched",
        False,
    ):
        return

    original_set_moe_sharding = glm5_sharding.set_moe_sharding_config
    legacy_seq_dim_tp_sharded = any(
        config_field.name == "seq_dim_tp_sharded"
        for config_field in fields(moe_module.MoE.Config)
    )

    def set_moe_sharding_config(
        moe_cfg, *, enable_ep, enable_sp, expert_param_layout
    ):
        original_set_moe_sharding(
            moe_cfg,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=expert_param_layout,
        )
        if enable_ep and not enable_sp:
            _fix_ep_without_sp(moe_cfg)
        elif not enable_ep:
            _fix_tp_only_routed_expert_grad_layout(moe_cfg)

    set_moe_sharding_config._torchtitanturbo_patched = True
    glm5_sharding.set_moe_sharding_config = set_moe_sharding_config
    if legacy_seq_dim_tp_sharded:
        original_moe_init = moe_module.MoE.__init__
        original_routed_parallelize = moe_module.RoutedExperts.parallelize

        def moe_init(self, config):
            original_moe_init(self, config)
            self.routed_experts.seq_dim_tp_sharded = config.seq_dim_tp_sharded

        def routed_parallelize(self, parallel_dims):
            original_routed_parallelize(self, parallel_dims)
            if not getattr(self, "seq_dim_tp_sharded", False):
                self.token_dispatcher.wire_meshes(
                    ep_mesh=parallel_dims.get_optional_mesh("ep"),
                    tp_mesh=None,
                )

        moe_module.MoE.__init__ = moe_init
        moe_module.RoutedExperts.parallelize = routed_parallelize
    logger.info("Patched GLM-5 EP sharding for NPU")
