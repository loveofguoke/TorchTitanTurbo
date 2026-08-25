# GLM-5 Ascend performance extensions

TorchTitanTurbo owns Ascend-only operators and compatibility code. Performance
replacements are opt-in; importing Turbo or applying normal GLM-5 compatibility
patches must not silently change Full DSA semantics.

## Implemented paths

`ops/triton.py` registers the TorchTitan GLM-specific Triton kernels for NPU.
The same Triton language source computes index scores and SparseMLA
forward/backward on CUDA or Ascend; Triton-Ascend supplies the NPU backend.

```bash
--override.imports \
torchtitanturbo.models.glm5.ops.triton.npu_triton_dsa_indexer,torchtitanturbo.models.glm5.ops.triton.npu_triton_sparse_mla
```

The indexer is complete for the current frozen-indexer training contract.
SparseMLA supplies Triton forward and backward; target NPU releases must still
validate FP32 atomic support, gradients, graph capture, and distributed shapes.

`ops/sparse_mla.py` implements the `SparseMLA.Config` boundary with
`torch_npu.npu_sparse_flash_attention`. It receives the same query,
compressed-KV, mask, top-k indices, scale, and latent dimension as the PyTorch
reference. Invalid selected positions are mapped to the operator sentinel.

Enable it explicitly:

```bash
--override.imports \
torchtitanturbo.models.glm5.ops.sparse_mla.npu_sparse_mla
```

This is a compute/memory optimization: selected attention is fused inside one
backend operator and avoids Python-side gather and multiple elementwise
launches. It does not fuse the GLM-5 indexer. The public lightning-indexer
interface currently has incompatible head geometry, so the PyTorch indexer
remains active.

## Existing Turbo building blocks

Turbo also contains NPU RMSNorm, grouped expert, token permute/unpermute, and
attention implementations. Their existence does not make them valid GLM-5
Full DSA replacements. Each converter must demonstrate matching config,
parameter/state-dict layout, TP/EP sharding, backward behavior, and precision.

A fused primitive is not yet a model optimization. The integration boundary
must be correct and the end-to-end trace must show a benefit.

## Communication optimization candidate

For EP, the target is dispatch/GMM/combine overlap rather than only changing a
collective algorithm:

```python
# Pseudocode for a future Turbo path.
for token_chunk in chunk(tokens):
    dispatched = hccl_all_to_all_async(token_chunk, routing)
    if previous is not None:
        expert_output = npu_grouped_matmul(previous.wait())
        combine_futures.append(
            hccl_all_to_all_async(expert_output, inverse_map)
        )
    previous = dispatched
return concatenate(future.wait() for future in combine_futures)
```

Correctness requires stable token order, capacity metadata, stream/event
ownership, and gradients. Performance acceptance requires lower exposed HCCL
time, not merely a higher overlap ratio.

For CP Full DSA, a production kernel should exchange index metadata and only
selected remote compressed KV where possible. The current all-gather path is
deliberately correctness-first.

## Compute optimization candidates

A GLM-specific Triton indexer now fuses query/key dot products, ReLU-weighted
head reduction, and mask addition before PyTorch top-k. It supports GLM-5's
actual index-head geometry and must return the same selected set as the
reference within the BF16 boundary policy. A future second-stage Triton top-k
can remove the remaining framework operation after profiling proves it is a
bottleneck.

```python
# Target boundary; never adapt an incompatible lightning-indexer shape silently.
scores, indices = npu_glm5_indexer(
    q_index, k_index, head_weights, causal_mask, topk=index_topk
)
```

NPU RMSNorm and grouped expert GMM are additional candidates, but each must be
a separate override so an A/B experiment can attribute the improvement.

## Memory and graph considerations

Fused operators should avoid per-step host synchronization and unbounded
temporary allocation. A reusable workspace is acceptable only with:

- shape/dtype/device identity;
- stream-safe reuse;
- a bounded cache and explicit release policy;
- compatibility with `torch.compile` and NPUGraph capture;
- profiler evidence showing fewer allocations or lower peak memory.

Dynamic shapes can cause recompilation or graph breaks even when eager mode is
correct. Validate eager single-card, distributed eager, compiled single-card,
and finally compiled distributed execution.

## Required evidence

Every Turbo optimization must report:

1. operator forward/backward comparison with the TorchTitan reference;
2. single-card and applicable distributed smoke results;
3. formal eager precision results;
4. graph-mode precision when graph support is claimed;
5. throughput, peak memory, kernel inventory, and HCCL exposure;
6. exact CANN, torch_npu, and hardware versions.

Confirmed operators and unregistered NPU-only ideas are documented separately
in `ops/README.md` and `ops_candidate/README.md`.
