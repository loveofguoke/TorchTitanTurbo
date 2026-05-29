#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

# Qwen3.5 training examples using TorchTitan + TorchTitanTurbo.
#
# Usage:
#   bash examples/train_qwen3_5_npu.sh [config_name] [steps] [extra torchtitan args...]
#
# Examples:
#   bash examples/train_qwen3_5_npu.sh
#   bash examples/train_qwen3_5_npu.sh qwen3_5_debugmodel 10
#   bash examples/train_qwen3_5_npu.sh qwen3_5_35b_a3b_debugmodel 2 --training.seq_len 64
#   NGPU=4 bash examples/train_qwen3_5_npu.sh qwen3_5_35b_a3b_debugmodel 2 \
#       --parallelism.data_parallel_shard_degree 4

set -e

CONFIG_NAME=${1:-"qwen3_5_debugmodel"}
STEPS=${2:-${STEPS:-10}}
NGPU=${NGPU:-1}
EXTRA_ARGS=("${@:3}")
MODULE_NAME="torchtitanturbo.models.qwen3_5"
BOOTSTRAP_IMPORT="import torchtitanturbo"

if [[ "${CONFIG_NAME}" == graph_trainer_qwen3_5* ]]; then
    MODULE_NAME="torchtitanturbo.experiments.graph_trainer.qwen3_5"
    BOOTSTRAP_IMPORT="import torchtitanturbo; import torchtitanturbo.experiments.graph_trainer.qwen3_5"
fi

echo "=============================================="
echo "TorchTitan + TorchTitanTurbo: Qwen3.5 NPU Training"
echo "=============================================="
echo "Config: ${CONFIG_NAME}"
echo "Module: ${MODULE_NAME}"
echo "Steps: ${STEPS}"
echo "NPUs: ${NGPU}"
echo ""

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-2}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-2}"

TORCHTITAN_ARGS=(
    --module "${MODULE_NAME}"
    --config "${CONFIG_NAME}"
    --training.steps "${STEPS}"
    "${EXTRA_ARGS[@]}"
)

if [ "${NGPU}" -eq 1 ]; then
    export LOCAL_RANK="${LOCAL_RANK:-0}"
    export RANK="${RANK:-0}"
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29635}"

    python -c "${BOOTSTRAP_IMPORT}; from torchtitan.train import main; main()" \
        "${TORCHTITAN_ARGS[@]}"
else
    torchrun --nproc_per_node="${NGPU}" \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        --local-ranks-filter 0 \
        --role rank \
        --tee 3 \
        --no-python \
        python -c "${BOOTSTRAP_IMPORT}; from torchtitan.train import main; main()" \
        "${TORCHTITAN_ARGS[@]}"
fi

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="
