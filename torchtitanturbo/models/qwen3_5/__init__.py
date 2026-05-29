# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RoPE
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    Qwen35Attention,
    Qwen35GatedDeltaNet,
    Qwen35MoeExperts,
    Qwen35MoeTopKRouter,
    Qwen35Model,
    Qwen35RMSNorm,
    Qwen35SparseMoeBlock,
    Qwen35TransformerBlock,
)
from .parallelize import parallelize_qwen3_5
from .state_dict_adapter import Qwen35StateDictAdapter

__all__ = [
    "parallelize_qwen3_5",
    "Qwen35Model",
    "qwen3_5_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.zeros_}
_GATED_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EPS = 1e-6


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _qwen35_norm(dim: int) -> Qwen35RMSNorm.Config:
    return Qwen35RMSNorm.Config(
        normalized_shape=dim,
        eps=_EPS,
        param_init=_NORM_INIT,
    )


def _qwen35_attention(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_id: int,
) -> Qwen35Attention.Config:
    return Qwen35Attention.Config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        q_proj=Linear.Config(
            in_features=dim,
            out_features=n_heads * head_dim * 2,
            param_init=_LINEAR_INIT,
        ),
        k_proj=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        v_proj=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        o_proj=Linear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        q_norm=_qwen35_norm(head_dim),
        k_norm=_qwen35_norm(head_dim),
    )


def _qwen35_linear_attention(
    *,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    layer_id: int,
    conv_kernel_size: int = 4,
) -> Qwen35GatedDeltaNet.Config:
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    return Qwen35GatedDeltaNet.Config(
        hidden_size=dim,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        rms_norm_eps=_EPS,
        in_proj_qkv=Linear.Config(
            in_features=dim,
            out_features=key_dim * 2 + value_dim,
            param_init=_LINEAR_INIT,
        ),
        in_proj_z=Linear.Config(
            in_features=dim,
            out_features=value_dim,
            param_init=_LINEAR_INIT,
        ),
        in_proj_b=Linear.Config(
            in_features=dim,
            out_features=num_v_heads,
            param_init=_LINEAR_INIT,
        ),
        in_proj_a=Linear.Config(
            in_features=dim,
            out_features=num_v_heads,
            param_init=_LINEAR_INIT,
        ),
        out_proj=Linear.Config(
            in_features=value_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
    )


def _build_qwen35_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    linear_num_k_heads: int,
    linear_num_v_heads: int,
    linear_head_k_dim: int,
    linear_head_v_dim: int,
    layer_types: list[str] | None = None,
) -> list[Qwen35TransformerBlock.Config]:
    if layer_types is None:
        layer_types = [
            "linear_attention" if bool((i + 1) % 4) else "full_attention"
            for i in range(n_layers)
        ]
    if len(layer_types) != n_layers:
        raise ValueError("Qwen3.5 layer_types length must match n_layers")

    layers = []
    for layer_id, layer_type in enumerate(layer_types):
        attention_cfg = _qwen35_attention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
        )
        layers.append(
            Qwen35TransformerBlock.Config(
                layer_type=layer_type,
                attention_norm=_qwen35_norm(dim),
                ffn_norm=_qwen35_norm(dim),
                linear_attn=(
                    _qwen35_linear_attention(
                        dim=dim,
                        num_k_heads=linear_num_k_heads,
                        num_v_heads=linear_num_v_heads,
                        head_k_dim=linear_head_k_dim,
                        head_v_dim=linear_head_v_dim,
                        layer_id=layer_id,
                    )
                    if layer_type == "linear_attention"
                    else None
                ),
                attention=attention_cfg,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _qwen35_moe(
    *,
    dim: int,
    moe_hidden_dim: int,
    shared_expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
) -> Qwen35SparseMoeBlock.Config:
    return Qwen35SparseMoeBlock.Config(
        hidden_size=dim,
        gate=Qwen35MoeTopKRouter.Config(
            hidden_size=dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init={"weight": _LINEAR_INIT["weight"]},
        ),
        experts=Qwen35MoeExperts.Config(
            hidden_size=dim,
            intermediate_size=moe_hidden_dim,
            num_experts=num_experts,
            param_init={
                "gate_up_proj": _LINEAR_INIT["weight"],
                "down_proj": _LINEAR_INIT["weight"],
            },
        ),
        shared_expert=make_ffn_config(
            dim=dim,
            hidden_dim=shared_expert_hidden_dim,
            w1_param_init=_LINEAR_INIT,
            w2w3_param_init=_LINEAR_INIT,
        ),
        shared_expert_gate=Linear.Config(
            in_features=dim,
            out_features=1,
            param_init=_LINEAR_INIT,
        ),
    )


def _build_qwen35_moe_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    linear_num_k_heads: int,
    linear_num_v_heads: int,
    linear_head_k_dim: int,
    linear_head_v_dim: int,
    moe_hidden_dim: int,
    shared_expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    layer_types: list[str] | None = None,
) -> list[Qwen35TransformerBlock.Config]:
    if layer_types is None:
        layer_types = [
            "linear_attention" if bool((i + 1) % 4) else "full_attention"
            for i in range(n_layers)
        ]
    if len(layer_types) != n_layers:
        raise ValueError("Qwen3.5 MoE layer_types length must match n_layers")

    layers = []
    for layer_id, layer_type in enumerate(layer_types):
        attention_cfg = _qwen35_attention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
        )
        layers.append(
            Qwen35TransformerBlock.Config(
                layer_type=layer_type,
                attention_norm=_qwen35_norm(dim),
                ffn_norm=_qwen35_norm(dim),
                linear_attn=(
                    _qwen35_linear_attention(
                        dim=dim,
                        num_k_heads=linear_num_k_heads,
                        num_v_heads=linear_num_v_heads,
                        head_k_dim=linear_head_k_dim,
                        head_v_dim=linear_head_v_dim,
                        layer_id=layer_id,
                    )
                    if layer_type == "linear_attention"
                    else None
                ),
                attention=attention_cfg,
                moe=_qwen35_moe(
                    dim=dim,
                    moe_hidden_dim=moe_hidden_dim,
                    shared_expert_hidden_dim=shared_expert_hidden_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                ),
            )
        )
    return layers


def _debugmodel() -> Qwen35Model.Config:
    dim = 128
    head_dim = 32
    rotary_dim = 8
    n_layers = 4
    vocab_size = 2048
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen35_norm(dim),
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rotary_dim,
            max_seq_len=1024,
            theta=10000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen35_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            hidden_dim=384,
            linear_num_k_heads=4,
            linear_num_v_heads=4,
            linear_head_k_dim=32,
            linear_head_v_dim=32,
        ),
    )


def _9b_text() -> Qwen35Model.Config:
    dim = 4096
    head_dim = 256
    rotary_dim = 64
    n_layers = 32
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen35_norm(dim),
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rotary_dim,
            max_seq_len=262144,
            theta=10000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen35_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=4,
            head_dim=head_dim,
            hidden_dim=12288,
            linear_num_k_heads=16,
            linear_num_v_heads=32,
            linear_head_k_dim=128,
            linear_head_v_dim=128,
        ),
    )


def _35b_a3b() -> Qwen35Model.Config:
    dim = 2048
    head_dim = 256
    rotary_dim = 64
    n_layers = 40
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen35_norm(dim),
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rotary_dim,
            max_seq_len=262144,
            theta=10000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen35_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=2,
            head_dim=head_dim,
            linear_num_k_heads=16,
            linear_num_v_heads=32,
            linear_head_k_dim=128,
            linear_head_v_dim=128,
            moe_hidden_dim=512,
            shared_expert_hidden_dim=512,
            num_experts=256,
            top_k=8,
        ),
    )


def _35b_a3b_debugmodel() -> Qwen35Model.Config:
    dim = 512
    head_dim = 128
    rotary_dim = 32
    n_layers = 8
    vocab_size = 2048
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen35_norm(dim),
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rotary_dim,
            max_seq_len=1024,
            theta=10000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen35_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=1,
            head_dim=head_dim,
            linear_num_k_heads=4,
            linear_num_v_heads=4,
            linear_head_k_dim=64,
            linear_head_v_dim=128,
            moe_hidden_dim=128,
            shared_expert_hidden_dim=128,
            num_experts=256,
            top_k=8,
        ),
    )


def _35b_a3b_micro_debugmodel() -> Qwen35Model.Config:
    dim = 128
    head_dim = 32
    rotary_dim = 8
    n_layers = 2
    vocab_size = 1024
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen35_norm(dim),
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rotary_dim,
            max_seq_len=256,
            theta=10000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen35_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=4,
            n_kv_heads=2,
            head_dim=head_dim,
            linear_num_k_heads=4,
            linear_num_v_heads=4,
            linear_head_k_dim=32,
            linear_head_v_dim=32,
            moe_hidden_dim=32,
            shared_expert_hidden_dim=32,
            num_experts=8,
            top_k=2,
            layer_types=["linear_attention", "full_attention"],
        ),
    )


qwen3_5_configs = {
    "debugmodel": _debugmodel,
    "9B_text": _9b_text,
    "35B-A3B": _35b_a3b,
    "35B-A3B_debugmodel": _35b_a3b_debugmodel,
    "35B-A3B_micro_debugmodel": _35b_a3b_micro_debugmodel,
}


def model_registry(flavor: str) -> ModelSpec:
    config = qwen3_5_configs[flavor]()
    return ModelSpec(
        name="qwen3_5",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen35StateDictAdapter,
    )
