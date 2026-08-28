# Ascend graph compatibility patches

The authoritative experiment entry and current solved/unresolved issue status
live in the
[torchtitan-test graph documentation](https://github.com/loveofguoke/torchtitan-test/blob/master/tests/glm5_2_graph/README.md)
and its
[NPU debug report](https://github.com/loveofguoke/torchtitan-test/blob/master/tests/glm5_2_graph/NPU_GRAPH_DEBUG_REPORT.md).
Lower-layer source locations, ticket boundaries, and proposed upstream patch
points are maintained in the
[backend issue handoff](https://github.com/loveofguoke/torchtitan-test/blob/master/tests/glm5_2_graph/LOWER_LAYER_ISSUE_HANDOFF.md).
This file documents only the Turbo implementation boundary. It must not be
used by itself to claim that a backend or topology passed smoke, precision, or
performance acceptance.

The source-installed three-repository stack was validated on 2026-08-26 at
TorchTitan `33270583`, Turbo `7343c9be` plus the documented working-tree fixes,
and test `77f4e2eb`: Inductor passed 15/15 smoke topologies and NPUGraphs passed
15/15 only in AOT-compatible, replay-disabled mode. The current heads
TorchTitan `59899ade`, Turbo `a5306484`, and test `01f2f3e1` retain compatible
patch target signatures, but each new working-tree change still requires the
focused validation recorded in the linked test reports. These are compatibility
results, not claims that the lower backend defects are fixed.

The deterministic pointwise-autotune addition was focused-validated on the
current heads on 2026-08-28: a cold-cache deterministic single-card Inductor
run and an independent smoke-runner single-card run both completed 10/10
steps. The maintained 5000-step all-topology precision matrix remains
incomplete and must not be inferred from these focused runs.

`graph_compat.py` owns opt-in compatibility patches required by TorchTitan
`torch.compile` experiments on Ascend. It is imported by the normal
TorchTitanTurbo bootstrap, but every behavior change is explicitly gated by a
`TORCHTITAN_*` environment variable. Eager training is unchanged.

| Variable | Patch | Current reason |
|---|---|---|
| `TORCHTITAN_VETTED_POINTWISE_AUTOTUNE=1` | Mark only pointwise Triton autotuning as a vetted deterministic benchmark. | torch_npu omits PyTorch's vetted flag, so formal deterministic graph precision fails before step 1. |
| `TORCHTITAN_SAFE_ZERO_NUMEL_TRITON=1` | Skip non-reduction Triton kernels whose dynamic `*numel` is zero. | CANN rejects a zero-core launch. |
| `TORCHTITAN_SAFE_EMPTY_GROUPED_MM=1` | Pad partially empty expert groups, bypass an all-empty input, and use an explicit backward. | EP can route zero tokens to a rank/expert. |
| `TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY=1` | Register the existing broadcast pointwise strategy for `aten.complex`. | TP RoPE reaches an operator missing from the current DTensor table. |
| `TORCHTITAN_PIPELINE_META_USE_BATCH=0` | Use ordinary object P2P for one-time pipeline metadata. | The tested HCCL batched metadata path corrupts the serialized size. |
| `TORCHTITAN_NPUGRAPH_SKIP_ALL=1` | Preserve Dynamo/AOT execution but skip NPUGraph replay. | Compatibility profile only; this is not native replay. |
| `TORCHTITAN_NPUGRAPH_UNSAFE_OPS=...` | Skip replay only for an FX graph containing a named incompatible op. | Diagnostic path for future backend retesting. |
| `TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE=...` | Override graph capture error mode. | Diagnostic only; relaxing capture does not fix synchronized grouped-mm offsets. |

`TORCHTITAN_TASK_QUEUE_ENABLE=0|1|2` is handled by the package bootstrap and
therefore is no longer reset to `2` after the graph launcher selected a mode.

The experiment launcher, CANN activation, HCCL ports/timeouts, Inductor
fallback list and cache directories remain in `torchtitan-test`. No NPU patch
is added to the device-neutral TorchTitan repository.

## Implementation index

| File | Responsibility |
|---|---|
| `torchtitanturbo/tools/graph_compat.py` | Validates opt-in variables and installs the selected runtime patches. |
| `torchtitanturbo/tools/compile.py` | Validates Ascend compile-component/backend combinations before delegating to TorchTitan. |
| `torchtitanturbo/patch.py` | Preserves an explicitly selected task-queue mode and invokes the normal Turbo patch bootstrap. |
| `tests/unit_tests/test_graph_compat.py` | CPU-side activation, validation, and patch-isolation coverage. |
| `PATCHES.md` | Repository-wide patch order and compatibility maintenance rules. |

These functions are deliberate compatibility sites, not claims of an upstream
root fix:

| Turbo function | Patched runtime object | Upstream root-fix target |
|---|---|---|
| `_install_vetted_pointwise_autotune` | `torch_npu._inductor.runtime.triton_heuristics.NPUCachingAutotuner._bench_with_launch_args` | torch_npu must pass PyTorch's `is_vetted_benchmarking=True` only for pointwise heuristics; reductions remain protected in deterministic mode. |
| `_install_zero_numel_triton_guard` | `torch_npu._inductor.runtime.triton_heuristics.NPUCachingAutotuner.run` and `NPUSymbolicGroupedAutotuner.run` | torch_npu Inductor grid/autotuner zero-work policy. |
| `_install_safe_empty_grouped_mm` | `torchtitan.models.common.moe.GroupedExperts._grouped_mm` | op-plugin grouped-mm empty-group contract and CANN `aclnnGroupedMatmulV4/V5`. |
| `_register_complex_dtensor_strategy` | PyTorch DTensor pointwise strategy table | PyTorch `torch.distributed.tensor._ops._pointwise_ops`. |
| `_install_pipeline_metadata_p2p` | `PipelineStage._recv_meta/_send_meta` | PyTorch PipelineStage object P2P and torch_npu ProcessGroupHCCL batched P2P boundary. |
| `_install_npugraph_skip_policy` | `torch_npu.utils._graph_tree.check_for_skip` | torch_npu NPUGraph input contract and capture-safe grouped-mm. |

Two local contract fixes are also intentional. An empty
`TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE` means "unset" because the clean test
launcher exports all fields. The safe grouped-mm backward performs in-place
`index_add_` into `zeros_like(B_t)` so the real output preserves the transposed
stride declared by its fake kernel. Both have behavioral regression tests.

TorchTitan continues to own `torchtitan/distributed/compile.py`. The test
repository continues to own launch policy, fallback A/B settings, compiler
diagnostics, raw logs, reports, and acceptance decisions.
