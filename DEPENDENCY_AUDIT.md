# TorchTitanTurbo dependency and regression contract

TorchTitanTurbo is a downstream, NPU-specific extension of source-installed
TorchTitan. Its patches are coupled to exact TorchTitan symbols even when the
Python package dependency has no version bound.

## Repository relationship

- Upstream model and trainer contracts: `../torchtitan`.
- NPU patches and optimized replacements: this repository.
- Launch, numerical, graph, performance, and lifecycle validation:
  `../torchtitan-test`.

For GLM-5, also read
`D:/yyb/huawei/glm/weeks/weekx/98_TorchTitan提交核验/TorchTitan三仓依赖审计.md`
and `../torchtitan-test/DEPENDENCY_AUDIT.md`.

## Patch dependency map

| Turbo surface | Upstream dependency | Downstream validation |
|---|---|---|
| `models/glm5/patch.py` initialization and loss | `models/glm5/flavors.py` initializer factories and `components.loss._LossParallelCrossEntropy` | GLM patch test, TP/EP smoke/parity/precision |
| GLM parameter initialization patch | GLM config factories and parameter-init dictionaries | GLM patch unit test, multi-axis TP/EP smoke |
| optional DeepSelect provider | DeepSelect public TopK API and GLM `DSAIndexerTopK` score contract | capability unit tests; fixed-input DSA accuracy; Dynamo/fake/fullgraph and recompile checks before model wiring |
| optional DeepGEMM provider | DeepGEMM-Ascend M-grouped BF16 NT/NN and K-grouped TN APIs plus `GroupedExperts._grouped_mm` layout contract | capability/autograd contract tests; official microbenchmarks; real-library dX/dW, fake/fullgraph, empty-expert, EP and checkpoint checks before default wiring |
| optional TileKernels provider | TileKernels FP32 MoE `topk_gate` and GLM router selection contract | capability/tie tests; router contract; fixed-input parity; fullgraph before wiring |
| optional FlashMLA provider | FlashMLA Ascend `flash_mla_sparse_fwd` and GLM sparse-attention layout | shape capability tests; forward oracle; reject training without backward; graph checks |
| optional DeepEP dispatcher | DeepEP-Ascend `EPBuffer` lifecycle, expanded-layout handle metadata, and TorchTitan `BaseEPTokenDispatcher`/runtime-capacity protocol | mocked converter/wiring/count/weight/autograd/teardown contracts; EP2/EP8 backward, handles/streams, empty experts, checkpoint, numerics and graph rejection tests on supported NPU |
| system optimization strategy prototypes | GraphTrainer AutoParallel, Async TP, FSDP/EP overlap and activation offload flags; Trainer FSDP CPU offload; DCP async modes | pure config translation and rejection tests; each selected strategy still requires its existing TorchTitan integration/numerics test and an Ascend profiler-off A/B |
| NPU vocab-parallel loss | `components.loss._LossParallelCrossEntropy` | TP smoke, TP precision and backward comparison |
| `graph_compat.py` and compile patches | TorchTitan model/parallel call graph, `PipelineStage._forward_metadata_inference`, plus torch_npu private APIs | graph debug, PP cold-start, graph smoke, eager-vs-graph precision/performance |
| NPU remat placeholder storage | `torch_remat._placeholder._null_storage` and PyTorch storage data-pointer poisoning | selective-AC unit contract; GLM eager single/all smoke; memory comparison before formal runs |
| profiler integration | Trainer profiler hooks and torch_npu profiler API, including stack and memory-timeline export contracts | performance probe, flamegraph/memory/all presets, offline analysis, and visualization report |
| import-time patch bootstrap | all patch modules above | patch status/idempotency tests and one NPU import smoke test |

TorchTitan `ad17686a` removed the repository-local
`Linear(..., compute_dtype=...)` extension while retaining an autocast-based
FP32 intent in the generic router. CPU autocast cannot exercise that intent,
and the Ascend implementation must not depend on it. Consequently the exact
BF16-input/FP32-gate contract is owned and behaviorally tested here through
`NpuFp32RouterLinear` and `NpuGlm5TokenChoiceTopKRouter`; downstream CPU parity
tests validate only TorchTitan's generic module-call contract.

The 2026-10-13 upstream sync removes Turbo's GLM Router replacement. Current
TorchTitan uses `HiMidLoLinear` for the FP32 gate and its Router now owns
padding-mask, auxiliary-loss, rematerialization, and group-limited selection
semantics. Turbo must not replace that forward with the older copied algorithm.

The graph private-symbol audit was last repeated against TorchTitan
`59899ade`, torch `2.14.0.dev20260805+cpu`, and torch_npu `2.14.0`. It resolved
`GroupedExperts._grouped_mm`, PipelineStage metadata P2P methods, the DTensor
pointwise registrar, both NPU Triton autotuner `run` contracts, graph-tree
`check_for_skip`, and `NPUCachingAutotuner._bench_with_launch_args`. Formal
deterministic precision exposed the missing vetted pointwise benchmark flag;
the failure and upstream handoff are recorded as G020 in torchtitan-test.

## Mandatory audit after changes

GLM DSA test-branch audit against TorchTitan `8ac999e8`: sparse attention and
index sharing remain upstream-owned. See `torchtitanturbo/models/glm5/DSA.md`
and `tests/unit_tests/test_glm5_dsa_contract.py`. NPU execution is not yet verified.

1. Resolve every patched/imported upstream symbol against the installed
   TorchTitan checkout. Do not interpret an API-break `ImportError` as “model
   unavailable.”
2. When copying or overriding a forward method, compare it with the current
   upstream implementation and maintain a behavioral equivalence test for all
   unchanged semantics.
3. Keep NPU-specific code here; never edit TorchTitan trainer/framework code to
   make a Turbo workaround pass.
4. Keep every patch optional, idempotent, observable, and default-off when it
   changes graph or numerical behavior.
5. Audit torchtitan-test smoke, parity, precision, graph, performance,
   checkpoint, stability, and combination call sites affected by the patch.
6. Update `PATCHES.md`, this document, and downstream experiment documentation
   whenever a patch target, fallback, or requirement changes.

Other model forks are independent review scopes. A GLM change must not
opportunistically rewrite Qwen, DeepSeek, or Llama patches.
