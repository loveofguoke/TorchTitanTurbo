# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State-dict key adapter for Qwen3.5 text models.

The adapter keeps HF-compatible names for parity tests and checkpoint export.
It covers the dense text path used by ``qwen3_5_debugmodel`` and
``qwen3_5_9b_text`` plus the Qwen3.5 35B-A3B MoE text backbone. Vision weights
are intentionally not mapped here.
"""

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Qwen35Model


class Qwen35StateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: Qwen35Model.Config, hf_assets_path: str | None):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.gate.weight",
            "model.layers.{}.mlp.experts.gate_up_proj": "layers.{}.moe.experts.gate_up_proj",
            "model.layers.{}.mlp.experts.down_proj": "layers.{}.moe.experts.down_proj",
            "model.layers.{}.mlp.shared_expert.gate_proj.weight": "layers.{}.moe.shared_expert.w1.weight",
            "model.layers.{}.mlp.shared_expert.up_proj.weight": "layers.{}.moe.shared_expert.w3.weight",
            "model.layers.{}.mlp.shared_expert.down_proj.weight": "layers.{}.moe.shared_expert.w2.weight",
            "model.layers.{}.mlp.shared_expert_gate.weight": "layers.{}.moe.shared_expert_gate.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.q_proj.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.k_proj.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.v_proj.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.o_proj.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.linear_attn.conv1d.weight": "layers.{}.linear_attn.conv1d.weight",
            "model.layers.{}.linear_attn.dt_bias": "layers.{}.linear_attn.dt_bias",
            "model.layers.{}.linear_attn.A_log": "layers.{}.linear_attn.A_log",
            "model.layers.{}.linear_attn.norm.weight": "layers.{}.linear_attn.norm.weight",
            "model.layers.{}.linear_attn.out_proj.weight": "layers.{}.linear_attn.out_proj.weight",
            "model.layers.{}.linear_attn.in_proj_qkv.weight": "layers.{}.linear_attn.in_proj_qkv.weight",
            "model.layers.{}.linear_attn.in_proj_z.weight": "layers.{}.linear_attn.in_proj_z.weight",
            "model.layers.{}.linear_attn.in_proj_b.weight": "layers.{}.linear_attn.in_proj_b.weight",
            "model.layers.{}.linear_attn.in_proj_a.weight": "layers.{}.linear_attn.in_proj_a.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}
        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)  # pyrefly: ignore[union-attr]
                if abstract_key in to_hf_map:
                    hf_state_dict[to_hf_map[abstract_key].format(layer_num)] = value
            elif key in to_hf_map:
                if self.model_config.enable_weight_tying and key == "lm_head.weight":
                    continue
                hf_state_dict[to_hf_map[key]] = value
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)  # pyrefly: ignore[union-attr]
                titan_key = self.from_hf_map.get(abstract_key)
                if titan_key is not None:
                    state_dict[titan_key.format(layer_num)] = value
            else:
                titan_key = self.from_hf_map.get(key)
                if titan_key is not None:
                    state_dict[titan_key] = value
        return state_dict
