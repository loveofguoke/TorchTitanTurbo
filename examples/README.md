# TorchTitanTurbo Multi-NPU Training Examples

This directory contains examples for training Llama4 on NPU using various distributed training strategies.

## Quick Start

### Single NPU
```bash
bash examples/train_llama4_multicard.sh single_npu
```

### Multi-NPU Training
```bash
# FSDP with 2 NPUs
bash examples/train_llama4_multicard.sh fsdp_2npu

# Tensor Parallel with 4 NPUs
bash examples/train_llama4_multicard.sh tp_4npu

# Expert Parallel with 2 NPUs (for MoE)
bash examples/train_llama4_multicard.sh ep_2npu

# Hybrid: TP(4) + EP(2) with 8 NPUs
bash examples/train_llama4_multicard.sh hybrid_8npu
```

## Distributed Training Strategies

### 1. Data Parallel (DP)

**DDP (Distributed Data Parallel)**
- Weights replicated across NPUs
- Gradients synchronized via all-reduce
```bash
bash examples/train_parallelism_examples.sh ddp_4npu
```

**FSDP (Fully Sharded Data Parallel)**
- Weights sharded across NPUs
- Memory efficient for large models
```bash
bash examples/train_parallelism_examples.sh fsdp_8npu
```

**HSDP (Hybrid Sharded Data Parallel)**
- Combination of sharding + replication
- Balance between memory and communication
```bash
bash examples/train_parallelism_examples.sh hsdp_8npu
```

### 2. Tensor Parallel (TP)

- Model weights split across NPUs
- Each NPU computes partial results
- Suitable for large dense layers
```bash
bash examples/train_parallelism_examples.sh tp_4npu
```

### 3. Expert Parallel (EP)

- Different experts on different NPUs
- MoE-specific optimization
- Efficient for models like Llama4 (MoE architecture)
```bash
bash examples/train_parallelism_examples.sh ep_4npu
```

### 4. Pipeline Parallel (PP)

- Different layers on different NPUs
- Pipeline schedule: 1F1B, Interleaved1F1B
- Memory efficient for very deep models
```bash
bash examples/train_parallelism_examples.sh pp_4npu
```

### 5. Hybrid Combinations

**TP + FSDP**
```bash
bash examples/train_parallelism_examples.sh tp_fsdp_8npu
```

**TP + EP (MoE optimized)**
```bash
bash examples/train_parallelism_examples.sh tp_ep_8npu
```

**PP + TP**
```bash
bash examples/train_parallelism_examples.sh pp_tp_8npu
```

**PP + TP + EP (Full hybrid)**
```bash
bash examples/train_parallelism_examples.sh pp_tp_ep_8npu
```

## Multi-Node Training

For training across multiple nodes (e.g., 2 nodes with 8 NPUs each):

**On node 0 (MASTER):**
```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export NNODES=2
export NODE_RANK=0

bash examples/train_llama4_multinode.sh master
```

**On node 1:**
```bash
export MASTER_ADDR=10.0.0.1  # Same as master
export MASTER_PORT=29500
export NNODES=2
export NODE_RANK=1

bash examples/train_llama4_multinode.sh worker
```

## Parallelism Configuration Matrix

| Strategy | NPUs | TP | EP | PP | FSDP | Use Case |
|----------|------|----|----|----|------|-----------|
| Single NPU | 1 | - | - | - | - | Baseline/debugging |
| FSDP-2 | 2 | - | - | - | 2 | Small models |
| FSDP-8 | 8 | - | - | - | 8 | Large models |
| TP-4 | 4 | 4 | - | - | - | Dense layers |
| EP-2 | 2 | - | 2 | - | - | MoE models |
| TP+EP | 8 | 4 | 2 | - | - | MoE optimization |
| PP+TP | 8 | 4 | - | 2 | - | Deep models |
| Full Hybrid | 8 | 2 | 2 | 2 | - | Complex models |

## Command Line Override

You can override any parallelism configuration via command line:

```bash
# Override TP degree
torchrun --nproc_per_node=4 -m torchtitan.train \
    --module torchtitanturbo.models.llama4 \
    --config npu_llama4_debugmodel \
    --parallelism.tensor_parallel_degree 4

# Override multiple settings
torchrun --nproc_per_node=8 -m torchtitan.train \
    --module torchtitanturbo.models.llama4 \
    --config npu_llama4_debugmodel \
    --parallelism.tensor_parallel_degree 4 \
    --parallelism.expert_parallel_degree 2 \
    --training.local_batch_size 4
```

## Environment Variables

Key NPU environment variables (set automatically in scripts):

```bash
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE="2"
export MULTI_STREAM_MEMORY_REUSE="2"
export CPU_AFFINITY_CONF="2"
```

## Files

- `train_llama4_npu.sh` - Single NPU baseline
- `train_llama4_multicard.sh` - Single-node multi-NPU with various strategies
- `train_llama4_multinode.sh` - Multi-node distributed training
- `train_parallelism_examples.sh` - Comprehensive parallelism strategy examples
- `train_llama4_npu.py` - Python script example