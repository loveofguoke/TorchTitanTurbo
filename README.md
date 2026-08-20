# TorchTitanTurbo

NPU-optimized converters and patches for TorchTitan. Provides fused operators for Ascend NPU acceleration.

## Overview

TorchTitanTurbo is a **plugin library** for TorchTitan that provides NPU-specific optimizations through:
- **Converters**: Replace module configs with NPU-optimized implementations
- **Patches**: Replace global functions with NPU implementations  
- **Config Registry**: Training configurations with NPU converters enabled

**Design Philosophy:**
- ✅ Reuse TorchTitan's infrastructure (Trainer, model_registry, train.py)
- ✅ Only provide NPU optimizations (converters + patches)
- ✅ Minimal code changes for users
- ✅ Clean separation of concerns

## Installation

```bash
pip install torchtitan
pip install torchtitanturbo
```

## Quick Start

### Auto Patch (Recommended)

Import `torchtitanturbo` to automatically apply all NPU patches:

```python
import torchtitanturbo  # Auto-applies all patches
```

This enables:
- NPU environment variables
- NPU profiler support
- NPU peak flops calculation
- NPU RoPE optimization
- DeepSeek/Qwen3 model compatibility
- GLM-5 NPU-safe DTensor router gather

### Training with NPU Optimizations

**Python API:**
```python
import torchtitanturbo
from torchtitanturbo.models.llama4.config_registry import npu_llama4_debugmodel
from torchtitan.trainer import Trainer

config = npu_llama4_debugmodel()  # NPU-optimized config
trainer = Trainer(config)         # Use TorchTitan's Trainer
trainer.train()
```

**Command Line:**
```bash
python -m torchtitan.train \
    --module torchtitanturbo.models.llama4 \
    --config npu_llama4_debugmodel

# Multi-NPU with torchrun
torchrun --nproc_per_node=4 -m torchtitan.train \
    --module torchtitanturbo.models.llama4 \
    --config npu_llama4_debugmodel \
    --parallelism.tensor_parallel_degree 4
```

**Example Scripts:**
```bash
# Single NPU
bash examples/train_llama4_multicard.sh single_npu

# Multi-NPU strategies
bash examples/train_llama4_multicard.sh tp_4npu      # Tensor Parallel
bash examples/train_llama4_multicard.sh ep_2npu      # Expert Parallel
bash examples/train_llama4_multicard.sh hybrid_8npu  # TP+EP+FSDP

# Multi-node
bash examples/train_llama4_multinode.sh master
```

## Converters (Model-Level)

Converters replace module configs with NPU-optimized implementations:

| Converter | Original Module | NPU Operator | Description |
|-----------|----------------|--------------|-------------|
| `NpuRMSNormConverter` | `RMSNorm` | `npu_rms_norm` | Fused normalization |
| `NpuSDPAConverter` | `ScaledDotProductAttention` | `npu_fusion_attention` | Fused attention |
| `NpuGroupedExpertsConverter` | `GroupedExperts` | `npu_grouped_matmul` + `npu_swiglu` | MoE experts |
| `NpuTokenDispatcherConverter` | `AllToAllTokenDispatcher` | `npu_moe_token_permute` | Token routing |

### Using Converters

```python
from torchtitanturbo.models.common import (
    NpuRMSNormConverter,
    NpuSDPAConverter,
    NpuGroupedExpertsConverter,
    NpuTokenDispatcherConverter,
)

# Use in model config
from torchtitan.models.llama4 import model_registry

model_spec = model_registry(
    "debugmodel",
    converters=[
        NpuRMSNormConverter.Config(),
        NpuSDPAConverter.Config(),
        NpuGroupedExpertsConverter.Config(),
        NpuTokenDispatcherConverter.Config(),
    ],
)
```

## Patches (Global-Level)

Patches replace global functions at import time using a unified replacement mechanism:

### Patch Infrastructure

**Unified Replacement Mechanism** (`tools/patch_utils.py`):
- `replace_functions()` - Replace functions in all modules (handles `from X import Y`)
- `replace_methods()` - Replace class methods
- `find_functions()` - Find all modules with a function reference
- `find_methods()` - Find all classes with a method

This mechanism ensures that both:
- Module attributes: `module.func`
- Local references: `from module import func`
are properly replaced.

### Applied Patches

| Patch | Target | NPU Operator | Description |
|-------|--------|--------------|-------------|
| `get_peak_flops` | `torchtitan.tools.utils` | - | Ascend device flops |
| `build_torch_profiler` | `torchtitan.tools.profiler` | `torch_npu.profiler` | NPU profiler |
| `build_memory_profiler` | `torchtitan.tools.profiler` | - | NPU memory snapshot |
| `apply_rotary_emb_*` | `torchtitan.models.common.rope` | `npu_rotary_mul` | NPU RoPE |
| `_get_gradient_divide_factors` | `torch.distributed.fsdp` | - | NPU gradient handling |
| `update_from_config` | `DeepSeekV3ModelArgs` | - | DeepSeek config |
| `apply_non_moe_tp` | `qwen3.infra.parallelize` | - | Qwen3 TP |
| GLM-5 router config factory | `torchtitan.models.glm5` | rank-local `gather` | Preserve TP/SP placements while avoiding the NPU DTensor gather backward shape bug |

## Training Configurations

NPU-optimized training configs for Llama4:

| Configuration | Converters | Description |
|---------------|-----------|-------------|
| `npu_llama4_debugmodel()` | Full (4 converters) | Complete NPU optimization |
| `npu_llama4_debugmodel_ep()` | Full + EP=2 | With Expert Parallelism |
| `npu_llama4_debugmodel_minimal()` | RMSNorm + SDPA | Minimal (no MoE) |

## Distributed Training

### Parallelism Strategies

TorchTitanTurbo supports all distributed training strategies:

**Data Parallel:**
- DDP (Distributed Data Parallel)
- FSDP (Fully Sharded Data Parallel)
- HSDP (Hybrid Sharded Data Parallel)

**Model Parallel:**
- TP (Tensor Parallel) - Split model weights
- EP (Expert Parallel) - MoE-specific
- PP (Pipeline Parallel) - Split layers

**Hybrid Combinations:**
- TP + FSDP
- TP + EP (MoE optimized)
- PP + TP
- PP + TP + EP (full hybrid)

### Usage Examples

```bash
# FSDP with 8 NPUs
bash examples/train_llama4_multicard.sh fsdp_8npu

# Tensor Parallel with 4 NPUs
bash examples/train_llama4_multicard.sh tp_4npu

# Expert Parallel with 2 NPUs (MoE)
bash examples/train_llama4_multicard.sh ep_2npu

# Hybrid: TP(4) + EP(2) with 8 NPUs
bash examples/train_llama4_multicard.sh hybrid_8npu

# Comprehensive examples
bash examples/train_parallelism_examples.sh tp_fsdp_8npu
bash examples/train_parallelism_examples.sh pp_tp_ep_8npu
```

See `examples/README.md` for complete documentation.

## How It Works

### Converter Application Flow

```
model_registry("debugmodel", converters=[...])
  ↓
Create original Llama4Model.Config
  ↓
Apply converters:
  - RMSNorm.Config → NpuRMSNorm.Config
  - SDPA.Config → NpuSDPA.Config
  - GroupedExperts.Config → NpuGroupedExperts.Config
  - TokenDispatcher.Config → NpuTokenDispatcher.Config
  ↓
Config.build() creates NPU modules
  ↓
Trainer trains with NPU operators
```

### Module Compatibility

All NPU modules maintain the same interface as original modules:

| Module | Original forward() | NPU forward() | Compatibility |
|--------|-------------------|---------------|---------------|
| RMSNorm | forward(x) | forward(x) | ✅ Identical |
| SDPA | forward(q,k,v,scale,...) | forward(q,k,v,scale,...) | ✅ Identical |
| GroupedExperts | forward(x,num_tokens) | forward(x,num_tokens) | ✅ Identical |
| TokenDispatcher | permute/unpermute | permute/unpermute | ✅ Identical |

**Design Principle:**
- Replace implementation, keep interface
- Caller code unchanged
- Automatic optimization

## Directory Structure

Aligned with torchtitan-new:

```
torchtitanturbo/
├── patch.py                    # Unified patch entry point
├── tools/
│   ├── utils.py                # get_peak_flops patch
│   └── profiler.py             # NPU profiler patch
├── distributed/
│   └── fsdp.py                 # FSDP gradient divide patch
├── models/
│   ├── common/
│   │   ├── npu_rope.py         # NPU RoPE optimization (patch)
│   │   ├── npu_attention.py    # NpuSDPA + Converter
│   │   ├── npu_gmm.py          # NpuGroupedExperts + Converter
│   │   ├── npu_permute.py      # NpuTokenDispatcher + Converter
│   │   └── npu_rmsnorm.py      # NpuRMSNorm + Converter
│   ├── llama4/
│   │   ├── __init__.py         # Export NPU configs
│   │   └── config_registry.py  # NPU Llama4 training configs
│   ├── deepseek_v3/
│   │   └── patch.py            # DeepSeek config patch
│   └── qwen3/
│       └── patch.py            # Qwen3 TP patch
├── examples/
│   ├── README.md               # Training examples documentation
│   ├── train_llama4_npu.py     # Python script example
│   ├── train_llama4_multicard.sh # Multi-NPU strategies
│   ├── train_llama4_multinode.sh # Multi-node training
│   └── train_parallelism_examples.sh # Comprehensive examples
└── tests/
    └── unit_tests/
        ├── test_converters.py  # Converter interface tests
        ├── test_npu_rmsnorm.py # NpuRMSNorm tests
        └── test_npu_attention.py # NpuSDPA tests
```

## Testing

### Run Tests

```bash
# All tests
python -m pytest tests/unit_tests/

# Specific test
python -m pytest tests/unit_tests/test_converters.py -v

# With verbose output
python -m pytest tests/unit_tests/ -v --tb=short
```

### Test Coverage

Tests verify:
- ✅ Config build() creates correct module types
- ✅ Converter replaces correct Config types
- ✅ NPU operators called correctly (mocked)
- ✅ Interface consistency across converters
- ✅ Naming conventions

## Requirements

- Python >= 3.7
- torch_npu (for NPU hardware)
- torchtitan >= 0.2.2
- Ascend NPU hardware (optional, for execution)

## Performance Benefits

Expected improvements on NPU:

| Operation | Original | NPU Optimized | Benefit |
|-----------|----------|---------------|---------|
| RMSNorm | Multiple ops | Fused `npu_rms_norm` | Reduced kernel launches |
| Attention | SDPA + multiple ops | Fused `npu_fusion_attention` | Better memory efficiency |
| MoE Experts | grouped_mm | `npu_grouped_matmul` + `npu_swiglu` | Fused activation |
| Token Routing | all_to_all | `npu_moe_token_permute` | Optimized routing |

## Troubleshooting

### Common Issues

**Import Error: No module named 'torch_npu'**
- Ensure torch_npu is installed
- Verify NPU drivers are installed

**NPU device not found**
- Check NPU hardware availability: `torch.npu.is_available()`
- Verify environment variables

**Converter not applied**
- Ensure converter is in converters list
- Check converter order (use `validate_converter_order`)

### Environment Variables

Key NPU environment variables (auto-set by turbo):

```bash
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE="2"
export MULTI_STREAM_MEMORY_REUSE="2"
export CPU_AFFINITY_CONF="2"
```

### Ascend profiler controls

TorchTitan continues to own the common profiler schedule through
`--profiler.*`. TorchTitanTurbo maps that lifecycle to Ascend Profiler and
accepts NPU-only collection controls through environment variables:

```bash
export TORCHTITAN_NPU_PROFILER_LEVEL=level0
export TORCHTITAN_NPU_PROFILER_RANKS=0
export TORCHTITAN_NPU_PROFILER_RECORD_SHAPES=false
export TORCHTITAN_NPU_PROFILER_PROFILE_MEMORY=false
export TORCHTITAN_NPU_PROFILER_WITH_STACK=false
export TORCHTITAN_NPU_PROFILER_WITH_MODULES=false
export TORCHTITAN_NPU_PROFILER_PARSE_MODE=sync
export TORCHTITAN_NPU_PROFILER_AIC_METRICS=none
export TORCHTITAN_NPU_PROFILER_HOST_SYSTEM=none
export TORCHTITAN_NPU_PROFILER_GC_DETECT_THRESHOLD=none
```

Supported levels are `level_none`, `level0`, `level1`, and `level2`. Rank
selection accepts `all` or comma-separated global ranks. Parse mode is `sync`,
`async`, or `offline`. Sync parsing can extend the profiled training step;
async parsing avoids that blocking but its outputs may finish after training.
Offline mode preserves raw `*_ascend_pt` data for later
`torch_npu.profiler.profiler.analyse` processing. The legacy
`TORCHTITAN_NPU_PROFILER_ONLINE_PARSE` switch remains supported.

The parsed DB, CSV, and JSON outputs are intended for MindStudio Insight and
msprof-analyze. The handler name is inherited from the upstream API; it does
not mean TensorBoard is the preferred Ascend trace viewer. Host CPU, memory,
disk, network, OS runtime, and NUMA collection is opt-in through
`TORCHTITAN_NPU_PROFILER_HOST_SYSTEM` because those collectors add overhead
and may require host permissions.

For deep runtime diagnosis, set `GC_DETECT_THRESHOLD=1` to record Python GC
events longer than one millisecond. MSTX collection and optional domain filters
are exposed through `MSTX`, `MSTX_DOMAIN_INCLUDE`, and `MSTX_DOMAIN_EXCLUDE`;
they remain disabled until the application adds MSTX marks.

The runnable GLM performance probes and HTML reports live in the independent
`torchtitan-test/tests/glm5_2_performance` framework.

## Contributing

Contributions welcome! Areas for improvement:
- Additional model support (DeepSeek V3, Qwen3 configs)
- More NPU operators (quantization, CP)
- Performance benchmarks
- Extended test coverage

## License

Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

## Acknowledgments

Built on TorchTitan's excellent framework:
- Model configuration system
- Converter architecture
- Distributed training support
- Trainer infrastructure
