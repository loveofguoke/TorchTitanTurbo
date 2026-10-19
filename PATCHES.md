# TorchTitanTurbo patch inventory

This document records why each global NPU patch exists, which TorchTitan API it
targets, and how the patch set evolved. The graph and GLM patch targets were
most recently resolved against:

- TorchTitanTurbo `a5306484` (`glm-dev`) plus the documented graph working tree
- TorchTitan `59899ade` (`feat/glm5-model-distributed`)
- torchtitan-test `01f2f3e1` (`master`) plus the documented experiment working tree

TorchTitan changes frequently. These revisions are reference points, not a
permanent compatibility promise.

The GLM patch group is the reviewed scope for these revisions. The current
DeepSeek V3 and Qwen3 patch targets still reference older TorchTitan module
paths and are excluded from the GLM compatibility claim until separately
updated and tested. Legacy Llama4 examples are likewise historical, not a
supported entry point for the current TorchTitan checkout.

## Ownership and activation

TorchTitan owns device-neutral training, model math, parallelism contracts,
checkpointing, and public configuration. TorchTitanTurbo owns Ascend-specific
compatibility and acceleration that should not be added to TorchTitan core.

Importing `torchtitanturbo` calls `apply_all_patches()` from
`torchtitanturbo/patch.py`. Patch application is guarded by
`torch.npu.is_available()`:

- on an Ascend host, the NPU environment and all patches below are applied;
- on a GPU or CPU host, patching is skipped and TorchTitan behavior is unchanged.

The bootstrap currently sets these NPU runtime variables before applying code
patches:

| Variable | Current value | Purpose |
|---|---:|---|
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | Enable expandable allocator segments. |
| `TASK_QUEUE_ENABLE` | `2` | Enable the NPU task-queue execution mode. |
| `MULTI_STREAM_MEMORY_REUSE` | `2` | Enable cross-stream memory reuse. |
| `CPU_AFFINITY_CONF` | `2` | Enable the configured CPU-affinity policy. |

These assignments are part of Turbo startup behavior, but are not monkey
patches of TorchTitan objects.

## Current patch application order

`apply_all_patches()` applies patches in this order. The order is explicit so
that shared infrastructure is installed before model-specific adapters.

| Order | Turbo implementation | Patched object | Purpose |
|---:|---|---|---|
| 1 | `torchtitanturbo.tools.utils` | `torchtitan.tools.utils.get_peak_flops` | Add Ascend 910B peak-FLOPS values used by throughput/MFU reporting. |
| 2 | `torchtitanturbo.tools.profiler` | `torchtitan.tools.profiler.Profiler.build_torch_profiler` and `build_memory_profiler` | Translate TorchTitan's profiler lifecycle to `torch_npu.profiler` and NPU allocator snapshots. |
| 3 | `torchtitanturbo.tools.compile` | references to `torchtitan.distributed.compile.apply_compile` already imported by TorchTitan modules | Validate Ascend graph-backend restrictions, then delegate compilation to TorchTitan. |
| 4 | `torchtitanturbo.models.common.npu_rope` | `torchtitan.models.common.rope._reshape_for_broadcast` references and `ComplexRoPE.apply_rotary_emb` | Avoid unsupported complex gather and use the fused Ascend rotary kernel while preserving token-first RoPE semantics. |
| 5 | `torchtitanturbo.distributed.fsdp` | `torch.distributed.fsdp._fully_shard._fsdp_collectives._get_gradient_divide_factors` | Use SUM plus explicit division where NPU collectives cannot use the upstream AVG path. |
| 6 | `torchtitanturbo.distributed.remat` | `torch_remat._placeholder._null_storage` | Preserve remat placeholder semantics with owning NPU storage because the current backend can segfault while releasing declared-size null-pointer storage. |
| 7 | `torchtitanturbo.models.deepseek_v3.patch` | `DeepSeekV3ModelArgs.update_from_config` and registered DeepSeek configs | Apply the NPU-compatible attention/configuration policy. |
| 7 | `torchtitanturbo.models.qwen3.patch` | `torchtitan.models.qwen3.infra.parallelize.apply_non_moe_tp` | Install the NPU-compatible Qwen3 non-MoE TP plan. |
| 8 | `torchtitanturbo.models.glm5.patch` | GLM flavor parameter initializers and TorchTitan vocab-parallel CE | Fix NPU DTensor initialization and TP-loss paths without replacing current upstream Router semantics. |

## Patch details

### Opt-in system optimization strategy prototypes

`torchtitanturbo.optimizations.system_strategies` provides a default-off typed
experiment surface for AutoParallel, Async TP, FSDP AG/RS and dense-region
overlap, EP overlap, activation memory policies, FSDP CPU state offload, and
asynchronous checkpointing. It validates combinations and translates them to
existing TorchTitan configuration; it does not patch or copy those algorithms.
GraphTrainer remains an explicitly selected Trainer recipe--the prototype never
pretends it can be entered through regular-Trainer `--compile.enable` or
`--compile.mode` flags. Its reference configuration is a true no-op and emits
no override.

GTP rematerialization and NVMe offload are recorded as explicit unsupported
identities until TorchTitan has a PyTorch-native executor. Selecting either
fails before launch instead of importing Megatron/DeepSpeed or silently using
a different offload tier.

Implementation: `torchtitanturbo/optimizations/system_strategies.py`

Validated command/provenance renderer:
`python -m torchtitanturbo.optimizations.cli`. It only renders overrides for an
already selected recipe; training lifecycle remains in `torchtitan-test`.
Provider selection/package preflight is available separately through
`python -m torchtitanturbo.optimizations.providers_cli`; package presence is
never reported as runtime acceptance.

### Peak-FLOPS reporting

Turbo replaces `torchtitan.tools.utils.get_peak_flops` and adds values for
Ascend 910B1/B2/B3/B4. TorchTitan metrics call this function when computing
hardware-normalized throughput such as MFU. The patch does not change model
execution or numerical results.

Implementation: `torchtitanturbo/tools/utils.py`

### Ascend PyTorch Profiler

TorchTitan continues to own profiler enablement, scheduling, step calls, and
output-folder conventions. Turbo replaces only the two device-facing builder
methods on `torchtitan.tools.profiler.Profiler`:

- `build_torch_profiler` creates a `torch_npu.profiler.profile` instance;
- `build_memory_profiler` records Ascend allocator snapshots using TorchTitan's
  memory-profiler lifecycle.

NPU-only collection details are supplied through
`TORCHTITAN_NPU_PROFILER_*` environment variables. They include rank filters,
profiler level, AIC metrics, parsing mode, export types, shapes, memory, stack,
module hierarchy, L2 cache, operator arguments, host-system data, interconnect
data, legacy msprof TX ranges, and MSTX controls. This keeps TorchTitan's
public profiler config device-neutral.

When `WITH_STACK=true`, the optional `EXPORT_STACKS=true` control exports
official CPU/NPU folded-stack files during synchronous trace handling. The test
repository renders and indexes optional flame graph SVGs; Turbo does not change
the report format or add a visualization dependency.

The optional `EXPORT_MEMORY_TIMELINE=true` path exports the official interactive
HTML, categorized JSON series, and raw memory-event stream after a scheduled
capture. It validates the shape, memory, and stack/module prerequisites before
training. `WITH_FLOPS` is also forwarded, but remains raw diagnostic data because
the current official parser does not support that field.

Implementation: `torchtitanturbo/tools/profiler.py`

Detailed boundary, option groups, data flow, and links to the authoritative
experiment reports: `torchtitanturbo/tools/PROFILER.md`

### Graph compile validation

Turbo wraps TorchTitan's `apply_compile` entry point. It does not implement a
separate compiler or replace TorchTitan's compilation flow. The wrapper checks
Ascend-specific restrictions and then calls the original function.

The current NPUGraph prototype permits model compilation only. Requesting the
loss component with backend `npugraphs` raises a clear error before compilation.
Inductor and eager behavior continue through TorchTitan.

Because TorchTitan modules may use `from ... import apply_compile`, the patch
updates already-imported TorchTitan references, not only the defining module.

Implementation: `torchtitanturbo/tools/compile.py`

### Token-first ComplexRoPE

Target TorchTitan contract:

```text
torchtitan.models.common.rope._reshape_for_broadcast(
    rope_cache, query_shape, positions
)
torchtitan.models.common.rope.ComplexRoPE.apply_rotary_emb(
    query, key, rope_cache
)
```

The current TorchTitan interface is token-first: query/key use `[T, N, H]`
style tensors and the selected cache broadcasts as `[T, 1, cache_width]`.
Ascend gather does not accept the complex cache used by the reference
implementation. Turbo gathers the real and imaginary FP32 components
separately, reconstructs the complex values, and applies rotation through
`torch_npu.npu_rotary_mul`.

DTensor metadata is restored after the local fused operation. Before patching,
Turbo validates both TorchTitan signatures and fails clearly if the upstream
contract has changed.

Implementation: `torchtitanturbo/models/common/npu_rope.py`

### FSDP gradient reduction

TorchTitan uses PyTorch composable FSDP, so this patch targets a PyTorch object
rather than a function defined inside TorchTitan:

```text
torch.distributed.fsdp._fully_shard._fsdp_collectives
    ._get_gradient_divide_factors
```

For `device_type == "npu"`, Turbo forces SUM reduction and applies the required
division factors explicitly. The resulting gradient average is mathematically
equivalent to the upstream AVG path. This patch exists for collective/backend
compatibility and does not define TorchTitan parallel topology.

Implementation: `torchtitanturbo/distributed/fsdp.py`

### torch_remat placeholder storage

TorchTitan main now implements activation-checkpoint policies with
`torch_remat`. During replay, an output of a saved region that is not needed is
represented by a correctly shaped placeholder backed by a declared-size,
null-pointer storage. PyTorch owns that protocol and still poisons data-pointer
access so an accidental computation fails diagnostically.

On the current Ascend stack, constructing the placeholder succeeds, but a real
GLM backward can segfault while the NPU allocator/runtime releases that storage.
Turbo changes only the storage provider for `device.type == "npu"`: it returns
an owning byte storage of the same declared size. `torch_remat` continues to
attach it to the placeholder, preserve its shape/stride/dtype, and poison data
access. CPU and CUDA call the untouched upstream function. This is a bounded
compatibility fallback with a temporary memory cost, not a second
activation-checkpoint algorithm; it should be removed after the backend can
safely own null-pointer placeholder storage.

Implementation: `torchtitanturbo/distributed/remat.py`

### DeepSeek V3 configuration

Turbo replaces `DeepSeekV3ModelArgs.update_from_config` and disables FlexAttention
in the registered DeepSeek configurations. The adapter also:

- takes sequence length from current TorchTitan `TrainingConfig`;
- disables grouped MM for the `ZBVZeroBubble` PP schedule;
- rejects the unsupported CP plus FlexAttention combination;
- forwards TorchTitan's debug force-load-balance setting to MoE arguments.

If the installed TorchTitan does not contain DeepSeek V3, this model-specific
patch logs a warning and is skipped.

Implementation: `torchtitanturbo/models/deepseek_v3/patch.py`

### Qwen3 tensor parallelism

Turbo replaces `torchtitan.models.qwen3.infra.parallelize.apply_non_moe_tp`.
The replacement preserves TorchTitan's TP/SP layout contract while selecting
an NPU-compatible plan for embeddings, attention projections, normalization,
output, and dense FFN layers. Float8 tensorwise TP and async TP remain optional
branches of the same plan.

If the installed TorchTitan does not contain Qwen3, this patch logs a warning
and is skipped.

Implementation: `torchtitanturbo/models/qwen3/patch.py`

### GLM-5 compatibility patch group

For the FlexAttention DSA test branch, see
[DSA compatibility audit](torchtitanturbo/models/glm5/DSA.md).
No new runtime patch is installed for DSA; the added tests check that existing
patches preserve sparse selection and shared-index configuration.

The GLM patch is deliberately attached to TorchTitan's existing GLM model. It
does not copy the model into Turbo and does not change the device-neutral GLM
math stored in TorchTitan.

After the 2026-10-13 migration to TorchTitan's config-first model API, the
initializer target is `torchtitan.models.glm5.flavors`. The former Router
factory patch is intentionally removed: upstream now supplies the required
FP32 gate through `HiMidLoLinear`, while its current forward also carries
padding-mask, auxiliary-loss, rematerialization, and grouped-selection
contracts that the old Turbo copy did not implement.

#### Router construction and DTensor gather

Target:

```text
torchtitan.models.glm5.make_router_config
```

Turbo wraps the factory so GLM router configs build
`NpuGlm5TokenChoiceTopKRouter` with `NpuFp32RouterLinear`. The gate has an
explicit FP32 compute path. Selected scores are gathered through a rank-local
`local_map` operation while preserving DTensor placements, including placement
metadata for the integer top-k indices.

The explicit gate dtype is not redundant with TorchTitan's generic router.
Since TorchTitan `ad17686a`, that router uses `torch.autocast(...,
dtype=torch.float32)` and the common `Linear` no longer accepts
`compute_dtype`. CPU autocast disables that unsupported target dtype, while the
Ascend GLM precision contract still requires FP32 routing. Turbo therefore owns
and unit-tests the stronger NPU contract: BF16 inputs invoke the gate exactly
once, compute the linear and bias in FP32, and return FP32 routing scores.

This avoids the NPU DTensor gather backward-shape failure while preserving
GLM's router scoring, top-k selection, normalization, route scaling, and debug
load-balancing behavior.

Because the upstream router does not expose the score-gather operation as an
override hook, the NPU subclass must retain a small forward override. The
Turbo unit suite therefore compares it directly with the current TorchTitan
router across sigmoid/softmax, expert bias, node-limited routing,
normalization, and debug load balancing. An upstream behavior change must
update or remove this override rather than drift silently.

#### DTensor truncated-normal initialization

Targets inside `torchtitan.models.glm5`:

```text
_LINEAR_INIT
_output_linear_init
_depth_init
_depth_experts_init
```

Turbo replaces only initializers that are partial applications of
`torch.nn.init.trunc_normal_`. For ordinary tensors it delegates to the original
PyTorch initializer. For a sharded DTensor it uses the inverse-CDF form of the
same truncated-normal distribution.

The PyTorch rejection-sampling implementation evaluates `mask.any()` inside a
loop. On a sharded NPU DTensor this scalar test can initialize lazy HCCL
subgroups in a different order across ranks. The local elementwise inverse-CDF
path avoids those collectives and the resulting distributed hang.

#### Vocab-parallel cross entropy

Target:

```text
torchtitan.components.loss._LossParallelCrossEntropy.forward
torchtitan.components.loss._LossParallelCrossEntropy.backward
```

TorchTitan's TP loss path used boolean-index assignment, which lowers to
`aclnnNonzeroV2` on NPU and fails in the affected stack. Turbo computes the same
distributed log-softmax and label gradient with `where`, `gather`, `scatter`,
and functional all-reduce operations. Ignore-index handling, reductions,
global-vocabulary sharding, and output dtype are preserved.

Implementation: `torchtitanturbo/models/glm5/patch.py`

### Opt-in graph compatibility

For compiled pipeline parallelism, dynamic `PipelineStage` metadata inference
executes a real forward on every stage in pipeline order. Cold compilation is
therefore serialized inside the metadata P2P chain. The diagnostic precompile
path accepts a representative shape through
`TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE=T,D` and the real dtype through
`TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_DTYPE`; inferring it from BF16 parameters
is incorrect when the pipeline activation contract is FP32 and would create a
second Dynamo specialization. This path and the host-store transport remain
diagnostic only. They do not replace PyTorch's metadata algorithm and are not
the PP8 fix. The production direction is PyTorch's public static PipelineStage
contract (`input_args`/`output_args`, plus grad metadata for DTensors) or the
unchanged dynamic metadata protocol.

The supported compatibility path is
`TORCHTITAN_PIPELINE_METADATA_FORCE_EAGER=1`. It calls the unchanged PyTorch
dynamic metadata implementation inside the public
`torch.compiler.set_stance("force_eager")` context. Metadata still comes from
the real upstream stage and follows the original transport. The installed
PyTorch build does not yet expose `PipelineStage.register_forward_context`, so
Turbo provides only this narrow integration bridge; it does not manufacture
input/output or gradient metadata.

`TORCHTITAN_PIPELINE_REAL_INPUT_PRECOMPILE=1` complements that bridge for cold
compiled PP. A temporary forward pre-hook records the real args/kwargs received
by each stage during PyTorch's unchanged dynamic metadata probe. Each rank then
builds its local forward/backward cache before schedule P2P, restores CPU/device
RNG state, module buffers, and parameter gradients, and waits on the torchrun
rendezvous store. The store is only a compilation-completion barrier; metadata,
activations, activation gradients, and training collectives retain their
original PyTorch/torch_npu transports.

Target modules include the torch_npu Triton autotuners, TorchTitan
`GroupedExperts`, PyTorch DTensor pointwise registration, PipelineStage
metadata exchange, and the torch_npu graph-tree skip policy. These patches are
implemented in `torchtitanturbo/tools/graph_compat.py` and are disabled unless
their documented `TORCHTITAN_*` environment variable is selected. See
`torchtitanturbo/tools/GRAPH_MODE.md` for the exact switches, limitations and
ownership boundary.

The experiment repository still owns CANN activation, HCCL ports/timeouts,
compiler caches and fallback A/B settings. TorchTitan remains device-neutral.

The deterministic precision profile additionally opts into a narrowly scoped
override of `NPUCachingAutotuner._bench_with_launch_args`. Current torch_npu
does not pass PyTorch's `is_vetted_benchmarking` flag, so even pointwise
autotuning is rejected when deterministic algorithms are enabled. Turbo marks
only `HeuristicType.POINTWISE` as vetted; reduction heuristics remain subject
to the upstream deterministic ban. The override validates the private method
signature before installation and has behavioral tests for both branches.

## Historical addition sequence

The current files contain the accumulated result. These commits explain when
and why each patch family entered the branch:

| Commit | Change |
|---|---|
| `98c2de1` | Introduced Turbo's initial NPU converters and global patches: peak FLOPS, profiler, RoPE, FSDP, DeepSeek V3, and Qwen3. |
| `612aad7` | Aligned the common integrations with the TorchTitan GLM branch and added NPU availability gating so GPU/CPU imports remain untouched. |
| `f6940a9` | Added the GLM router factory and rank-local DTensor gather patch. |
| `203a7e6` | Completed router gather placement metadata required by the current `local_map` contract. |
| `856e300` | Expanded Ascend Profiler support and added the graph compile validation patch. |
| `0ecf9c2` | Adapted NPU integrations to the newer TorchTitan configuration and module interfaces. |
| `86c9b5e` | Reworked RoPE for TorchTitan's token-first `ComplexRoPE` API and added explicit API validation. |
| `f5c7e59` | Added the GLM DTensor truncated-normal initializer patch to avoid inconsistent HCCL subgroup initialization. |
| `36cbba6` | Added the NPU-safe TP vocab-parallel loss implementation to avoid `aclnnNonzeroV2`. |

## Patches versus converters

Default-off third-party kernel providers live under
`torchtitanturbo/optimizations/`. They are neither global patches nor automatic
converters. Importing Turbo does not load their external dependencies. The
first prototype, `DeepSelectTopKProvider`, validates the official DeepSelect
platform/dtype/layout contract but is not wired into GLM-5 because the current
Ascend kernel accepts BF16 while GLM-5 intentionally ranks DSA scores in FP32.
`DeepGemmGroupedBf16Provider` is likewise default-off. It adapts TorchTitan
grouped-MM layouts and implements dX/dW with the official Ascend M-grouped NN
and K-grouped TN APIs. It remains ineligible for default model wiring until the
real-library, graph, distributed, activation-checkpoint, and checkpoint
contracts are validated.
The same package also contains default-off capability adapters for TileKernels
MoE TopK, FlashMLA Ascend sparse prefill, and DeepEP-Ascend EPBuffer. None is
installed by `apply_all_patches()`: TileKernels still needs router/fullgraph
parity and FlashMLA is forward-only under its public Ascend API.

DeepEP-Ascend additionally has a formal `BaseEPTokenDispatcher` implementation
and an explicit `ModelConfigConverter`. The converter targets only a configured
TorchTitan `DeepEPTokenDispatcher`; standard AllToAll remains unchanged. The
dispatcher preserves the official expanded-layout contract: expert counts come
from the handle prefix sums, routing weights are applied once before combine,
and dispatch/combine retain their autograd dual. It remains default-off pending
supported-hardware EP2/EP8 validation, and rejects `torch.compile` because the
official Ascend backend does not currently support graph capture.

The following common NPU implementations are converters, not automatic global
patches:

| Converter | TorchTitan configuration replaced | NPU implementation |
|---|---|---|
| `NpuRMSNormConverter` | `RMSNorm.Config` | `torch_npu.npu_rms_norm` |
| `NpuSDPAConverter` | `ScaledDotProductAttention.Config` | `torch_npu.npu_fusion_attention` |
| `NpuGroupedExpertsConverter` | `GroupedExperts.Config` | grouped matmul and fused SwiGLU |
| `NpuTokenDispatcherConverter` | `AllToAllTokenDispatcher.Config` | fused NPU permute/unpermute |

Converters run only when an NPU model configuration explicitly includes them.
Importing Turbo registers their classes but does not replace every matching
TorchTitan model globally.

## Maintenance rules for TorchTitan updates

When updating the source-installed TorchTitan revision:

1. Compare every target listed in the application-order table with the new
   TorchTitan implementation and signature.
2. Run Turbo unit tests for compile, profiler, RoPE, converters, and GLM patches.
3. Run `torchtitan-test` eager smoke tests before graph, precision, performance,
   checkpoint, or stability experiments.
4. Verify single-card and distributed NPU execution for every affected patch;
   router/loss/init changes require TP and combined-topology coverage.
5. Remove a patch when the compatible NPU behavior is implemented upstream.
   Do not keep a second implementation merely because it existed historically.
6. Keep device-neutral GLM model math in TorchTitan. New NPU-only workarounds,
   fused operators, profiler integration, and backend restrictions belong here.

Patch failures should be explicit. A changed upstream API must not silently
fall back to an unverified implementation.
