# Ascend profiler integration

The authoritative user commands, captured measurements, topology reports, and
optimization conclusions live in the
[torchtitan-test performance documentation](https://github.com/loveofguoke/torchtitan-test/blob/master/tests/glm5_2_performance/README.md).
The current cross-topology evidence is indexed by the
[exploration reports](https://github.com/loveofguoke/torchtitan-test/tree/master/tests/glm5_2_performance/explorations/reports).
This document describes only TorchTitanTurbo's Ascend implementation boundary.

## Three-repository data flow

```text
torchtitan-test
  selects topology, schedule, ranks, preset, parsing and output roots
        |
        v
TorchTitan torchtitan/tools/profiler.py
  owns enablement, schedule lifecycle, Profiler.step() and folder contract
        |
        v
TorchTitanTurbo torchtitanturbo/tools/profiler.py
  builds torch_npu.profiler and Ascend allocator snapshot implementations
        |
        v
torchtitan-test
  parses, runs msprof-analyze, records MindStudio inputs and builds reports
```

Turbo does not define experiment topology, performance thresholds, report
format, or optimization acceptance. TorchTitan does not contain an Ascend
special case. Importing Turbo installs the NPU builder methods, but no profile
is collected unless the normal TorchTitan profiler configuration enables it.

## Patched interface

`torchtitanturbo/tools/profiler.py::apply_patch` replaces two device-facing
methods on `torchtitan.tools.profiler.Profiler`:

- `build_torch_profiler` constructs `torch_npu.profiler.profile`, preserving
  TorchTitan's wait/warmup/active/repeat and initial-skip schedule;
- `build_memory_profiler` records Ascend allocator snapshots through the same
  lifecycle used by TorchTitan's memory profiler.

Collection includes CPU and NPU activities. The trace handler writes the
official Ascend PyTorch Profiler result layout, which remains suitable for
offline parsing, `msprof-analyze`, and MindStudio Insight.

## NPU-only controls

NPU collection options use the `TORCHTITAN_NPU_PROFILER_` prefix so the
TorchTitan public configuration remains device independent.

| Group | Variables |
|---|---|
| Scope and parsing | `RANKS`, `PARSE_MODE`, compatibility alias `ONLINE_PARSE`, `EXPORT_TYPES` |
| Detail level | `LEVEL`, `AIC_METRICS`, `RECORD_SHAPES`, `WITH_STACK`, `WITH_MODULES`, `WITH_FLOPS`, `RECORD_OP_ARGS`, `EXPORT_STACKS` |
| Memory and cache | `PROFILE_MEMORY`, `EXPORT_MEMORY_TIMELINE`, `L2_CACHE`, `DATA_SIMPLIFICATION`, `GC_DETECT_THRESHOLD` |
| Host and fabric | `HOST_SYSTEM`, `SYSTEM_IO`, `SYSTEM_INTERCONNECTION` |
| User ranges | `MSPROF_TX`, `MSTX`, `MSTX_DOMAIN_INCLUDE`, `MSTX_DOMAIN_EXCLUDE` |

The exact defaults and validation sets are defined by
`NpuProfilerOptions` in `torchtitanturbo/tools/profiler.py`. Experiment presets
in `torchtitan-test` translate a user-facing goal such as `overview`,
`distributed`, or `kernel` into these variables. Users should normally select
a preset instead of setting every variable manually.

`MSPROF_TX` exposes the legacy TX marker collector; `MSTX` exposes the current
MSTX collector and optional domain filters. Both default to false and are kept
separate so experiments do not silently change the annotation mechanism.

`EXPORT_STACKS=true` is an opt-in post-processing action for synchronous
profiles captured with `WITH_STACK=true`. At the trace callback, Turbo asks the
official `torch_npu.profiler` object to export both
`self_npu_time_total` and `self_cpu_time_total` folded-stack files. It does not
vendor a flame graph renderer. `torchtitan-test --preset flamegraph` owns the
user-facing switch, optional `flamegraph.pl` SVG conversion, artifact links,
and report text. Timeline visualization remains a MindStudio Insight concern,
while TensorBoard remains a TorchTitan scalar-metrics concern.

`EXPORT_MEMORY_TIMELINE=true` is another scheduled-callback export. It requires
`RECORD_SHAPES=true`, `PROFILE_MEMORY=true`, and at least one of
`WITH_STACK=true` or `WITH_MODULES=true`, matching the official Ascend API
contract. Turbo writes an interactive HTML, a categorized JSON time series,
and a raw event JSON stream for the current NPU device. The HTML renderer needs
`matplotlib`; the test repository owns that optional dependency and links all
three files from the performance report.

`WITH_FLOPS` is exposed because it exists in the current torch_npu API, but the
official documentation states that profiler parsing does not currently support
that field. It must not be confused with TorchTitan's model-formula TFLOPS/MFU
metrics or used as a parsed acceptance result.

## Ownership and acceptance

| Concern | Owner |
|---|---|
| Profiler enablement, schedule, and per-step lifecycle | TorchTitan |
| Ascend activities, experimental config, parsing mode, and allocator snapshot | TorchTitanTurbo |
| CANN environment, rank selection, topology launch, storage policy, offline parser, advisor, cluster analysis, visualization handoff, and HTML report | torchtitan-test |
| Performance conclusion and optimization promotion | torchtitan-test evidence plus an explicit A/B acceptance decision |

Profiler-off steady-state runs are throughput evidence. Profiler-active runs
are attribution evidence and can carry substantial overhead. Neither this
patch nor a successful profile capture proves that an optimization is faster
or numerically acceptable.

## Implementation and validation index

- implementation: `torchtitanturbo/tools/profiler.py`;
- patch order and TorchTitan compatibility notes: `PATCHES.md`;
- Turbo unit coverage: `tests/unit_tests/test_profiler.py`;
- experiment commands and report schema: the linked `torchtitan-test` README;
- measured 1/2/4/8-card evidence: the linked exploration report tree.

When TorchTitan changes its `Profiler` method signatures or schedule fields,
update Turbo against the new device-neutral contract, run the Turbo unit tests,
then run a small `overview` probe before any distributed or deep preset.
