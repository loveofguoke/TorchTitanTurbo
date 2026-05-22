# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU-optimized Llama4 training configurations for use with TorchTitan.

This module provides Trainer.Config instances with NPU converters enabled.
These configs should be used with torchtitan.trainer.Trainer.

Example:
    from torchtitanturbo.models.llama4.config_registry import npu_llama4_debugmodel
    from torchtitan.trainer import Trainer

    config = npu_llama4_debugmodel()
    trainer = Trainer(config)
    trainer.train()
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.llama4 import model_registry
from torchtitan.trainer import Trainer

from torchtitanturbo.models.common import (
    NpuRMSNormConverter,
    NpuSDPAConverter,
    NpuGroupedExpertsConverter,
    NpuTokenDispatcherConverter,
)


def npu_llama4_debugmodel() -> Trainer.Config:
    """NPU-optimized Llama4 debugmodel with full Turbo converters.

    Returns:
        Trainer.Config with all NPU converters enabled
    """
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry(
            "debugmodel",
            attn_backend="sdpa",
            converters=[
                NpuRMSNormConverter.Config(),
                NpuSDPAConverter.Config(),
                NpuGroupedExpertsConverter.Config(),
                NpuTokenDispatcherConverter.Config(),
            ],
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=OptimizersContainer.Config(lr=4e-3, eps=1e-15),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def npu_llama4_debugmodel_ep() -> Trainer.Config:
    """NPU-optimized Llama4 debugmodel with Expert Parallelism.

    Returns:
        Trainer.Config with EP enabled
    """
    config = npu_llama4_debugmodel()
    config.parallelism.expert_parallel_degree = 2
    return config


def npu_llama4_debugmodel_minimal() -> Trainer.Config:
    """NPU-optimized Llama4 debugmodel with minimal converters.

    Only RMSNorm and SDPA converters (no MoE optimizations).

    Returns:
        Trainer.Config with minimal converters
    """
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry(
            "debugmodel",
            attn_backend="sdpa",
            converters=[
                NpuRMSNormConverter.Config(),
                NpuSDPAConverter.Config(),
            ],
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=OptimizersContainer.Config(lr=4e-3, eps=1e-15),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


__all__ = [
    "npu_llama4_debugmodel",
    "npu_llama4_debugmodel_ep",
    "npu_llama4_debugmodel_minimal",
]
