# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import fields

from ._compat import apply_qwen35_graph_trainer_compat

apply_qwen35_graph_trainer_compat()

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec
from torchtitanturbo.models.qwen3_5 import qwen3_5_configs
from torchtitanturbo.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

from .model import GraphTrainerQwen35Model
from .parallelize import parallelize_qwen3_5


def model_registry(flavor: str) -> ModelSpec:
    base = qwen3_5_configs[flavor]()
    config = GraphTrainerQwen35Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="torchtitanturbo/graph_trainer/qwen3_5",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen35StateDictAdapter,
    )
