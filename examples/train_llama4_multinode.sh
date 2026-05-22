#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

# Multi-node NPU training example
#
# This script demonstrates distributed training across multiple nodes.
# Each node has 8 NPUs, total 16 NPUs across 2 nodes.
#
# Prerequisites:
#   - torch_npu installed on all nodes
#   - Ascend NPU hardware on all nodes
#   - Network connectivity between nodes
#   - Same code and data on all nodes
#
# Usage:
#   On node 0 (MASTER):
#     bash examples/train_llama4_multinode.sh master
#
#   On node 1:
#     bash examples/train_llama4_multinode.sh worker
#
# Environment variables (set before running):
#   MASTER_ADDR=10.0.0.1  # IP address of master node
#   MASTER_PORT=29500     # Port for rendezvous
#   NNODES=2              # Total number of nodes
#   NODE_RANK=0           # Rank of this node (0 for master, 1 for worker)

set -e

ROLE=${1:-"master"}

echo "=============================================="
echo "TorchTitan + TorchTitanTurbo: Multi-Node Training"
echo "=============================================="
echo "Node role: $ROLE"
echo ""

# Configuration
export MASTER_ADDR=${MASTER_ADDR:-"10.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export NNODES=${NNODES:-"2"}
export NODE_RANK=${NODE_RANK:-"0"}
export NGPU_PER_NODE=${NGPU_PER_NODE:-"8"}

# Total world size
export WORLD_SIZE=$((NNODES * NGPU_PER_NODE))

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Total nodes: $NNODES"
echo "Node rank: $NODE_RANK"
echo "NPUs per node: $NGPU_PER_NODE"
echo "World size: $WORLD_SIZE"
echo ""

# NPU-specific environment variables
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE="2"
export MULTI_STREAM_MEMORY_REUSE="2"
export CPU_AFFINITY_CONF="2"

# Parallelism configuration for multi-node
# Example: TP=8 across single node, FSDP across 2 nodes
PARALLELISM_ARGS="
    --parallelism.tensor_parallel_degree 8
    --parallelism.data_parallel_shard_degree 2
"

# Run training with torchrun
torchrun --nnodes=$NNODES \
    --nproc_per_node=$NGPU_PER_NODE \
    --rdzv_backend c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --node_rank=$NODE_RANK \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --module torchtitanturbo.models.llama4 \
    --config npu_llama4_debugmodel \
    --training.steps 10 \
    $PARALLELISM_ARGS

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="