# Ascend graph compatibility patches

The authoritative experiment entry and current solved/unresolved issue status
live in the
[torchtitan-test graph documentation](https://github.com/loveofguoke/torchtitan-test/blob/master/tests/glm5_2_graph/README.md)
and its
[NPU debug report](https://github.com/loveofguoke/torchtitan-test/blob/master/tests/glm5_2_graph/NPU_GRAPH_DEBUG_REPORT.md).
This file documents only the Turbo implementation boundary. It must not be
used by itself to claim that a backend or topology passed smoke, precision, or
performance acceptance.

The compatibility behavior was exercised by the earlier test-local prototype.
The same logic is now implemented on `glm-dev`, but the source-installed
three-repository version still requires server regression. Until the linked
test report records that regression, these patches are "implemented, awaiting
revalidation", not final backend fixes.

`graph_compat.py` owns opt-in compatibility patches required by TorchTitan
`torch.compile` experiments on Ascend. It is imported by the normal
TorchTitanTurbo bootstrap, but every behavior change is explicitly gated by a
`TORCHTITAN_*` environment variable. Eager training is unchanged.

| Variable | Patch | Current reason |
|---|---|---|
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

TorchTitan continues to own `torchtitan/distributed/compile.py`. The test
repository continues to own launch policy, fallback A/B settings, compiler
diagnostics, raw logs, reports, and acceptance decisions.
