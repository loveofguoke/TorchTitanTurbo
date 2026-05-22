#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Example: Training Llama4 on NPU with TorchTitan + TorchTitanTurbo.

This script demonstrates how to train a Llama4 debugmodel on NPU
using TorchTitan's Trainer with TorchTitanTurbo's converters.

Usage:
    python examples/train_llama4_npu.py

Prerequisites:
    - torch_npu installed
    - Ascend NPU hardware
    - TorchTitan installed
"""

import torch_npu  # Must import before torchtitan

# Use TorchTitan's Trainer + Turbo's config
from torchtitanturbo.models.llama4.config_registry import npu_llama4_debugmodel
from torchtitan.trainer import Trainer


def main():
    """Main training entry point."""

    # Get NPU-optimized configuration from turbo
    config = npu_llama4_debugmodel()

    print("=" * 60)
    print("TorchTitan + TorchTitanTurbo: Llama4 NPU Training")
    print("=" * 60)
    print(f"Model: Llama4 debugmodel (dim=256, layers=6, experts=8)")
    print(f"Batch size: {config.training.local_batch_size}")
    print(f"Sequence length: {config.training.seq_len}")
    print(f"Training steps: {config.training.steps}")
    print()
    print("NPU Converters enabled:")
    print("  - NpuRMSNormConverter")
    print("  - NpuSDPAConverter")
    print("  - NpuGroupedExpertsConverter")
    print("  - NpuTokenDispatcherConverter")
    print("=" * 60)
    print()

    # Use TorchTitan's Trainer (not turbo's)
    trainer = Trainer(config)

    # Start training
    print("Starting training...")
    trainer.train()

    print()
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
