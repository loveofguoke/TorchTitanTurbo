# Optional optimization providers

This package contains explicit, default-off adapters for third-party kernels.
Importing TorchTitanTurbo must not import or activate these dependencies.

`OptimizationProvidersConfig` gives every replaceable seam a typed reference
default and constructs only explicitly selected providers. Provider
construction still does not import the third-party package; the first actual
operation performs the lazy import. `selected_names()` is suitable for
experiment identity and provenance. This config is intentionally Turbo-owned
until the providers pass runtime validation; it does not modify TorchTitan's
public Trainer configuration.

`package_capabilities()` adds a non-importing package-presence check for the
explicitly selected providers. Discoverability is installation evidence only;
it does not claim that dtype, device, shape, backward, distributed, or graph
contracts have passed.

Render provider selection and package evidence without running a kernel:

```bash
python -m torchtitanturbo.optimizations.providers_cli \
  --ep-dispatch-backend deepep_ascend
```

The JSON deliberately contains `runtime_accepted=false`; only the downstream
seven-gate experiment report may promote a provider.

## DeepSelect

`DeepSelectTopKProvider` is the first provider prototype. It lazy-loads the
official `deep_select` package, validates the documented platform, dtype,
TopK, two-dimensional layout, and byte-stride contracts, and records a small
provenance payload. It never falls back to `torch.topk` silently.

It is deliberately not wired into GLM-5 yet. The current GLM-5 DSA indexer
computes FP32 ranking scores to preserve the K/K+1 selection boundary, while
the official Ascend DeepSelect kernel supports BF16 input only. Selecting it
would therefore change numerical semantics. The provider becomes eligible
after either:

1. the official Ascend implementation supports FP32; or
2. the fixed-input accuracy workflow proves that an explicit BF16 indexer
   configuration is acceptable for the intended training contract.

Compile compatibility is also unverified. Before model wiring, test Dynamo
capture, FakeTensor/meta behavior, `torch.compile(fullgraph=True)`, and dynamic
shape recompilation. If the pybind entry point is not traceable, expose it
through a Turbo-owned `torch.library` custom op with a fake implementation.
Until those checks pass, the provider is neither `compile_safe` nor eligible
for silent graph fallback.

Official source and installation instructions:
[deepseek-ai/DeepSelect](https://github.com/deepseek-ai/DeepSelect).

## DeepGEMM-Ascend grouped BF16

`DeepGemmGroupedBf16Provider` adapts TorchTitan's `A[M,K]`, `B_t[E,K,N]`,
and cumulative `offsets[E]` contract to DeepGEMM-Ascend's
`m_grouped_bf16_gemm_nt_contiguous` API. It lazy-loads `deep_gemm`, validates
NPU/BF16/INT32 layout requirements, preserves the native contiguous expert
weight storage, and never falls back silently.

The provider now supplies explicit autograd using the official Ascend grouped
APIs: M-grouped NT for forward, M-grouped NN for the input gradient, and
K-grouped TN for the weight gradient. TorchTitan cumulative expert offsets are
passed as DeepGEMM's prefix-sum group layout; dW additionally derives the
per-expert K lengths required by the K-grouped API. This code path is
training-capable at the API level, but it must not replace
`GroupedExperts._grouped_mm()` by default until real-library dX/dW, fake/meta,
fullgraph, empty-expert, activation-checkpoint, FSDP/EP, and checkpoint tests
pass. Compile capability remains `unverified`.

`DeepGemmGroupedExpertsConverter` is the explicit model-config integration. It
derives a subclass from the configured experts implementation and overrides
only `_grouped_mm`; parameter names, checkpoint layout, expert math, dispatch,
and combine stay owned by TorchTitan. The converter is never applied by the
Turbo import bootstrap.

Official source and installation instructions:
[deepseek-ai/DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend).

## TileKernels MoE TopK

`TileKernelsMoeTopKProvider` wraps the public `tile_kernels.moe.topk_gate`
entry point for contiguous FP32 token-expert scores and `topk <= 32`. Unlike
DeepSelect, this kernel targets MoE routing and is semantically relevant to the
GLM router. Tie behavior, node-limited routing, expert bias, load-balancing
debug mode, graph capture, and fixed-input parity still require validation
before wiring the provider into `NpuGlm5TokenChoiceTopKRouter`.

Official source: [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels).

## FlashMLA Ascend sparse prefill

`FlashMlaAscendSparseProvider` wraps `flash_mla_sparse_fwd` with the public
Ascend constraints: BF16 Q/KV, INT32 indices, 64 or 128 query heads, and
`d_qk=d_v=512`. It is forward-only and rejects tensors requiring gradients.
The current public Ascend sparse API therefore cannot replace TorchTitan's
training attention path. GLM dimensions must also pass the exact shape checks;
API naming alone is not compatibility evidence.

Official source: [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA).

## DeepEP-Ascend EPBuffer

`DeepEpAscendBufferProvider` builds an explicitly destroyed `EPBuffer` with
Ascend's required non-hybrid, multiple-reduction settings. Dispatchers with the
same EP group and capacity specification share it through a reference-counted
pool, so a multi-layer MoE does not allocate one communication workspace per
layer. Its
training methods implement the official dual: dispatch backward calls combine,
and combine backward reuses the forward handle for dispatch.

`AscendDeepEPTokenDispatcher` now implements TorchTitan's
`BaseEPTokenDispatcher` protocol. It returns the expanded expert-major rows,
derives grouped-GEMM segment lengths from
`handle.psum_num_recv_tokens_per_expert`, saves the routing weights and handle
in `EPDispatchMetadata`, applies each row's routing weight before combine, and
exposes `close()` for the required buffer teardown before process-group
destruction. `AscendDeepEPTokenDispatcher.Config` derives from TorchTitan's
`DeepEPTokenDispatcher.Config`, so the existing runtime capacity finalizer also
fills `hidden_dim` and `num_max_tokens_per_rank`.

The default-off `AscendDeepEPTokenDispatcherConverter` replaces only an
explicitly selected TorchTitan `DeepEPTokenDispatcher.Config`; it never changes
the standard AllToAll dispatcher or activates on import. The converter is the
NPU counterpart of the GLM `glm5_debugmodel_deepep` GPU experiment rather than
a device-dependent reinterpretation of the `deepep` name.

This is an API-level integration, not a 910B2 support claim. The official
DeepEP-Ascend release is currently validated on Ascend 950/CANN 9.2, graph
capture is unsupported, and real EP2/EP8 stream, empty-expert, teardown,
checkpoint, numerical, and performance tests remain mandatory before enabling
it in an NPU training config.

Official source:
[deepseek-ai/DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend).

## Training-system strategy prototypes

`SystemOptimizationConfig` covers optimizations that schedule the whole train
step rather than replacing one operator: AutoParallel, Async TP, FSDP AG/RS and
dense-region overlap, EP overlap, activation recompute/offload, FSDP CPU state
offload, and asynchronous checkpointing. It reuses TorchTitan and GraphTrainer
implementations by producing overrides for the already selected recipe; it does
not copy Megatron or DeepSpeed algorithms into Turbo.

The prototype also exposes the current official
`compile.inductor_compilation`, `compile.numerics_changing_optim`, and
`compile.precompile_artifact_dir` fields. Precompile is rejected when PP is
enabled because the current GraphPP implementation explicitly does not support
loading precompile artifacts; this is a capability boundary, not a fallback.

GraphTrainer is a distinct Trainer/config recipe, not a regular Trainer compile
mode. `execution_mode="graph_trainer"` therefore records and validates the
required recipe family but does not generate nonexistent `--compile.enable` or
`--compile.mode` flags. The caller must first select a GraphTrainer recipe and
then apply the returned `--compile.*` overrides. The default strategy returns no
overrides, so merely constructing the prototype cannot change training.
Checkpoint overrides target the current `Trainer.Config.checkpointer` field
(`--checkpointer.async_mode`), not the legacy/nonexistent `--checkpoint.*`
namespace.

```python
from torchtitanturbo.optimizations import SystemOptimizationConfig

strategy = SystemOptimizationConfig(
    execution_mode="graph_trainer",
    parallel_plan="autoparallel",
    tp_schedule="async",
    fsdp_overlap="ag_rs",
    ep_overlap="enabled",
    ep_overlap_module_fqn="layers.*.feed_forward",
    activation_memory="selective_recompute_and_cpu_offload",
    state_offload="cpu",
    checkpoint_io="async_pinned",
    inductor_compilation="full",
    numerics_changing_optim=True,
    precompile_artifact_dir="artifacts/graph-trainer",
)
overrides = strategy.torchtitan_overrides(
    tensor_parallel_degree=2,
    data_parallel_shard_degree=2,
    expert_parallel_degree=2,
)
```

Validation is topology-aware and runs before command construction. Graph-only
features cannot be requested in eager mode, and TP/FSDP/EP overlap requires a
corresponding mesh degree greater than one. FSDP dense-region overlap is
rejected when EP overlap is selected because current GraphTrainer would warn
and ignore the dense-region request; an experiment identity must not claim a
strategy that did not run. GTP rematerialization and NVMe
offload are reserved experiment identities but intentionally fail: the current
TorchTitan checkout has no PyTorch-native executor for them, and Turbo must not
silently add a Megatron/DeepSpeed runtime dependency or substitute CPU offload.

Eager activation checkpointing is selected by a polymorphic
`ActivationCheckpointing.Config` in the recipe, not by a scalar CLI field.
Therefore this command translator accepts activation memory policies only for
GraphTrainer's real `compile.memory_policy` field; it does not emit invented
`activation-checkpoint:*` pseudo-arguments. Eager full/selective/regional AC
must first be integrated as an explicit GLM recipe variant.

These prototypes establish configuration, provenance, and failure contracts.
They are not hardware-performance claims. Each non-reference strategy still
requires fixed-input numerics, target-NPU execution, profiler-off repeated A/B,
memory/communication evidence, checkpoint recovery, and relevant combination
tests before it can be recommended.

Render validated overrides before adding them to an experiment command:

```bash
python -m torchtitanturbo.optimizations.cli \
  --execution-mode graph_trainer \
  --tensor-parallel-degree 2 \
  --tp-schedule async \
  --inductor-compilation full \
  --format shell
```

JSON output is the default and includes recipe family, exact overrides, and
provenance. The renderer validates topology and feature composition but does
not launch training; `torchtitan-test` remains responsible for run lifecycle,
artifacts, and acceptance evidence.
