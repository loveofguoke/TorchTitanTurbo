#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

# Comprehensive parallelism training examples
#
# This script demonstrates all supported parallelism strategies:
#   1. DP (Data Parallel) - DDP/FSDP/HSDP
#   2. TP (Tensor Parallel)
#   3. EP (Expert Parallel)
#   4. PP (Pipeline Parallel)
#   5. Hybrid combinations
#
# Prerequisites:
#   - torch_npu installed
#   - Ascend NPU hardware (8 NPUs per node)
#   - TorchTitan installed
#
# Usage:
#   bash examples/train_parallelism_examples.sh [example_name]
#
# Examples:
#   bash examples/train_parallelism_examples.sh ddp_4npu
#   bash examples/train_parallelism_examples.sh fsdp_8npu
#   bash examples/train_parallelism_examples.sh hsdp_8npu
#   bash examples/train_parallelism_examples.sh tp_fsdp_8npu

set -e

EXAMPLE=${1:-"fsdp_8npu"}

echo "=============================================="
echo "Parallelism Strategy Examples"
echo "=============================================="
echo ""

# ============================================================================
# Data Parallel Strategies
# ============================================================================

case $EXAMPLE in
    # DDP (Distributed Data Parallel) - weights replicated
    ddp_4npu)
        echo "Example: DDP with 4 NPUs"
        echo "  - Weights replicated across 4 NPUs"
        echo "  - Gradients synchronized via all-reduce"
        NGPU=4
        PARALLELISM_ARGS="--parallelism.data_parallel_replicate_degree 4"
        ;;
    
    # FSDP (Fully Sharded Data Parallel) - weights sharded
    fsdp_8npu)
        echo "Example: FSDP with 8 NPUs"
        echo "  - Weights sharded across 8 NPUs"
        echo "  - Memory efficient for large models"
        NGPU=8
        PARALLELISM_ARGS="--parallelism.data_parallel_shard_degree 8"
        ;;
    
    # HSDP (Hybrid Sharded Data Parallel) - sharded + replicated
    hsdp_8npu)
        echo "Example: HSDP with 8 NPUs (shard=4, replicate=2)"
        echo "  - Weights sharded across 4 NPUs within group"
        echo "  - Then replicated across 2 groups"
        echo "  - Total: 4x2=8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="
            --parallelism.data_parallel_shard_degree 4
            --parallelism.data_parallel_replicate_degree 2
        "
        ;;
    
    # ============================================================================
    # Tensor Parallel Strategies
    # ============================================================================
    
    tp_2npu)
        echo "Example: Tensor Parallel with 2 NPUs"
        echo "  - Model weights split across 2 NPUs"
        echo "  - Each NPU computes partial results"
        NGPU=2
        PARALLELISM_ARGS="--parallelism.tensor_parallel_degree 2"
        ;;
    
    tp_4npu)
        echo "Example: Tensor Parallel with 4 NPUs"
        echo "  - Model weights split across 4 NPUs"
        NGPU=4
        PARALLELISM_ARGS="--parallelism.tensor_parallel_degree 4"
        ;;
    
    # ============================================================================
    # Expert Parallel (MoE-specific)
    # ============================================================================
    
    ep_2npu)
        echo "Example: Expert Parallel with 2 NPUs"
        echo "  - Different experts on different NPUs"
        echo "  - Suitable for MoE models"
        NGPU=2
        PARALLELISM_ARGS="--parallelism.expert_parallel_degree 2"
        ;;
    
    ep_4npu)
        echo "Example: Expert Parallel with 4 NPUs"
        echo "  - 4 NPUs hold different expert groups"
        NGPU=4
        PARALLELISM_ARGS="--parallelism.expert_parallel_degree 4"
        ;;
    
    # ============================================================================
    # Pipeline Parallel
    # ============================================================================
    
    pp_2npu)
        echo "Example: Pipeline Parallel with 2 NPUs"
        echo "  - Different layers on different NPUs"
        echo "  - Pipeline schedule: 1F1B"
        NGPU=2
        PARALLELISM_ARGS="
            --parallelism.pipeline_parallel_degree 2
            --parallelism.pipeline_parallel_schedule 1F1B
        "
        ;;
    
    pp_4npu)
        echo "Example: Pipeline Parallel with 4 NPUs"
        echo "  - 4 stages, each on different NPU"
        NGPU=4
        PARALLELISM_ARGS="
            --parallelism.pipeline_parallel_degree 4
            --parallelism.pipeline_parallel_schedule Interleaved1F1B
        "
        ;;
    
    # ============================================================================
    # Hybrid Combinations
    # ============================================================================
    
    # TP + FSDP
    tp_fsdp_8npu)
        echo "Example: TP(4) + FSDP(2) with 8 NPUs"
        echo "  - Tensor parallel: model split across 4 NPUs"
        echo "  - FSDP: each TP group sharded across 2 replicas"
        echo "  - Layout: 4x2=8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="
            --parallelism.tensor_parallel_degree 4
            --parallelism.data_parallel_shard_degree 2
        "
        ;;
    
    # TP + EP (MoE optimized)
    tp_ep_8npu)
        echo "Example: TP(4) + EP(2) with 8 NPUs for MoE"
        echo "  - Tensor parallel: attention/FFN split across 4"
        echo "  - Expert parallel: experts distributed across 2"
        echo "  - Layout: 4x2=8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="
            --parallelism.tensor_parallel_degree 4
            --parallelism.expert_parallel_degree 2
        "
        ;;
    
    # PP + TP
    pp_tp_8npu)
        echo "Example: PP(2) + TP(4) with 8 NPUs"
        echo "  - Pipeline: 2 stages"
        echo "  - Each stage: TP across 4 NPUs"
        echo "  - Layout: 2x4=8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="
            --parallelism.pipeline_parallel_degree 2
            --parallelism.tensor_parallel_degree 4
            --parallelism.pipeline_parallel_schedule Interleaved1F1B
        "
        ;;
    
    # Full hybrid: PP + TP + EP
    pp_tp_ep_8npu)
        echo "Example: PP(2) + TP(2) + EP(2) with 8 NPUs"
        echo "  - Pipeline: 2 stages"
        echo "  - Each stage: TP(2) + EP(2)"
        echo "  - Layout: 2x(2x2)=8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="
            --parallelism.pipeline_parallel_degree 2
            --parallelism.tensor_parallel_degree 2
            --parallelism.expert_parallel_degree 2
            --parallelism.pipeline_parallel_schedule Interleaved1F1B
        "
        ;;
    
    *)
        echo "Error: Unknown example '$EXAMPLE'"
        echo ""
        echo "Available examples:"
        echo ""
        echo "Data Parallel:"
        echo "  ddp_4npu       - DDP with 4 NPUs"
        echo "  fsdp_8npu      - FSDP with 8 NPUs"
        echo "  hsdp_8npu      - HSDP with 8 NPUs (shard=4, replicate=2)"
        echo ""
        echo "Tensor Parallel:"
        echo "  tp_2npu        - TP with 2 NPUs"
        echo "  tp_4npu        - TP with 4 NPUs"
        echo ""
        echo "Expert Parallel:"
        echo "  ep_2npu        - EP with 2 NPUs"
        echo "  ep_4npu        - EP with 4 NPUs"
        echo ""
        echo "Pipeline Parallel:"
        echo "  pp_2npu        - PP with 2 NPUs"
        echo "  pp_4npu        - PP with 4 NPUs"
        echo ""
        echo "Hybrid Combinations:"
        echo "  tp_fsdp_8npu   - TP(4) + FSDP(2) with 8 NPUs"
        echo "  tp_ep_8npu     - TP(4) + EP(2) with 8 NPUs"
        echo "  pp_tp_8npu     - PP(2) + TP(4) with 8 NPUs"
        echo "  pp_tp_ep_8npu  - PP(2) + TP(2) + EP(2) with 8 NPUs"
        exit 1
        ;;
esac

echo "NPUs: $NGPU"
echo "Arguments: $PARALLELISM_ARGS"
echo ""

# NPU environment
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE="2"

# Run training
if [ $NGPU -eq 1 ]; then
    LOCAL_RANK=0 python -m torchtitan.train \
        --module torchtitanturbo.models.llama4 \
        --config npu_llama4_debugmodel \
        --training.steps 5 \
        $PARALLELISM_ARGS
else
    torchrun --nproc_per_node=$NGPU \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        -m torchtitan.train \
        --module torchtitanturbo.models.llama4 \
        --config npu_llama4_debugmodel \
        --training.steps 5 \
        $PARALLELISM_ARGS
fi

echo ""
echo "=============================================="
echo "Example completed!"
echo "=============================================="