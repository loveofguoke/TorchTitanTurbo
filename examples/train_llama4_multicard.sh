#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

# Multi-NPU training examples using TorchTitan + TorchTitanTurbo
#
# This script demonstrates various distributed training configurations:
#   1. Single NPU (baseline)
#   2. Data Parallel (FSDP)
#   3. Tensor Parallel (TP)
#   4. Expert Parallel (EP)
#   5. Hybrid: TP + EP + FSDP
#
# Prerequisites:
#   - torch_npu installed
#   - Ascend NPU hardware (8 NPUs available)
#   - TorchTitan installed
#
# Usage:
#   bash examples/train_llama4_multicard.sh [config_name]
#
# Examples:
#   bash examples/train_llama4_multicard.sh single_npu     # Single NPU baseline
#   bash examples/train_llama4_multicard.sh fsdp_2npu      # FSDP with 2 NPUs
#   bash examples/train_llama4_multicard.sh tp_4npu        # TP with 4 NPUs
#   bash examples/train_llama4_multicard.sh ep_2npu        # EP with 2 NPUs
#   bash examples/train_llama4_multicard.sh hybrid_8npu    # TP+EP+FSDP with 8 NPUs

set -e

# Configuration selector
CONFIG_NAME=${1:-"single_npu"}

echo "=============================================="
echo "TorchTitan + TorchTitanTurbo: Multi-NPU Training"
echo "=============================================="
echo "Configuration: $CONFIG_NAME"
echo ""

case $CONFIG_NAME in
    # Single NPU baseline
    single_npu)
        echo "Mode: Single NPU (baseline)"
        NGPU=1
        PARALLELISM_ARGS=""
        ;;
    
    # FSDP (Fully Sharded Data Parallel)
    fsdp_2npu)
        echo "Mode: FSDP with 2 NPUs"
        NGPU=2
        PARALLELISM_ARGS="--parallelism.data_parallel_shard_degree 2"
        ;;
    
    fsdp_8npu)
        echo "Mode: FSDP with 8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="--parallelism.data_parallel_shard_degree 8"
        ;;
    
    # Tensor Parallel
    tp_2npu)
        echo "Mode: Tensor Parallel with 2 NPUs"
        NGPU=2
        PARALLELISM_ARGS="--parallelism.tensor_parallel_degree 2"
        ;;
    
    tp_4npu)
        echo "Mode: Tensor Parallel with 4 NPUs"
        NGPU=4
        PARALLELISM_ARGS="--parallelism.tensor_parallel_degree 4"
        ;;
    
    tp_8npu)
        echo "Mode: Tensor Parallel with 8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="--parallelism.tensor_parallel_degree 8"
        ;;
    
    # Expert Parallel (for MoE)
    ep_2npu)
        echo "Mode: Expert Parallel with 2 NPUs"
        NGPU=2
        PARALLELISM_ARGS="--parallelism.expert_parallel_degree 2"
        ;;
    
    ep_4npu)
        echo "Mode: Expert Parallel with 4 NPUs"
        NGPU=4
        PARALLELISM_ARGS="--parallelism.expert_parallel_degree 4"
        ;;
    
    # Hybrid: TP + EP + FSDP
    hybrid_8npu)
        echo "Mode: Hybrid TP(4) + EP(2) + FSDP with 8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="
            --parallelism.tensor_parallel_degree 4
            --parallelism.expert_parallel_degree 2
            --parallelism.data_parallel_shard_degree 1
        "
        ;;
    
    # Pipeline Parallel + Tensor Parallel
    pp_tp_8npu)
        echo "Mode: Pipeline Parallel(2) + Tensor Parallel(4) with 8 NPUs"
        NGPU=8
        PARALLELISM_ARGS="
            --parallelism.pipeline_parallel_degree 2
            --parallelism.tensor_parallel_degree 4
            --parallelism.pipeline_parallel_schedule Interleaved1F1B
        "
        ;;
    
    *)
        echo "Error: Unknown configuration '$CONFIG_NAME'"
        echo "Available configurations:"
        echo "  single_npu      - Single NPU baseline"
        echo "  fsdp_2npu       - FSDP with 2 NPUs"
        echo "  fsdp_8npu       - FSDP with 8 NPUs"
        echo "  tp_2npu         - Tensor Parallel with 2 NPUs"
        echo "  tp_4npu         - Tensor Parallel with 4 NPUs"
        echo "  tp_8npu         - Tensor Parallel with 8 NPUs"
        echo "  ep_2npu         - Expert Parallel with 2 NPUs"
        echo "  ep_4npu         - Expert Parallel with 4 NPUs"
        echo "  hybrid_8npu     - TP(4) + EP(2) + FSDP with 8 NPUs"
        echo "  pp_tp_8npu      - PP(2) + TP(4) with 8 NPUs"
        exit 1
        ;;
esac

echo "NPUs: $NGPU"
echo "Parallelism: $PARALLELISM_ARGS"
echo ""

# NPU-specific environment variables
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE="2"
export MULTI_STREAM_MEMORY_REUSE="2"
export CPU_AFFINITY_CONF="2"

# Run training with torchrun
if [ $NGPU -eq 1 ]; then
    # Single NPU: no need for torchrun
    python -m torchtitan.train \
        --module torchtitanturbo.models.llama4 \
        --config npu_llama4_debugmodel \
        --training.steps 10 \
        $PARALLELISM_ARGS
else
    # Multi-NPU: use torchrun
    torchrun --nproc_per_node=$NGPU \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        --local-ranks-filter 0 \
        --role rank \
        --tee 3 \
        -m torchtitan.train \
        --module torchtitanturbo.models.llama4 \
        --config npu_llama4_debugmodel \
        --training.steps 10 \
        $PARALLELISM_ARGS
fi

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="