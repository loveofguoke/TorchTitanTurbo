# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitanturbo.models.qwen3_5.config_registry import (
    qwen3_5_35b_a3b,
    qwen3_5_35b_a3b_debugmodel,
    qwen3_5_35b_a3b_micro_debugmodel,
    qwen3_5_9b_text,
    qwen3_5_debugmodel,
)

from . import model_registry


def _aot_eager_compile_config() -> GraphTrainerCompileConfig:
    return GraphTrainerCompileConfig(enable=True, enable_passes=False)


def graph_trainer_qwen3_5_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_5_debugmodel(), model_registry)
    config.compile = _aot_eager_compile_config()
    return config


def graph_trainer_qwen3_5_9b_text() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_5_9b_text(), model_registry)
    config.compile = _aot_eager_compile_config()
    return config


def graph_trainer_qwen3_5_35b_a3b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_5_35b_a3b(), model_registry)
    config.compile = _aot_eager_compile_config()
    return config


def graph_trainer_qwen3_5_35b_a3b_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(qwen3_5_35b_a3b_debugmodel(), model_registry)
    config.compile = _aot_eager_compile_config()
    return config


def graph_trainer_qwen3_5_35b_a3b_micro_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        qwen3_5_35b_a3b_micro_debugmodel(),
        model_registry,
    )
    config.compile = _aot_eager_compile_config()
    return config
