# GLM-5 Full DSA on Ascend

## Repository boundary

TorchTitan owns the readable device-independent DSA model. TorchTitanTurbo
owns only Ascend-specific operator implementations and compatibility patches.
The default `apply_glm5_patch()` path does not enable Full DSA operators, so
existing precision, checkpoint, stability, and performance experiments retain
their original behavior.

## Component mapping

| DSA stage | TorchTitan reference | Ascend implementation |
|---|---|---|
| Index projections and exact top-k | `Glm5DsaIndexer` and `DSAIndexerTopK` | PyTorch reference retained |
| Absorbed sparse attention | `SparseMLA` | `ops/sparse_mla.py::NpuSparseMLA` |
| Sparse KV gather | PyTorch advanced indexing | Performed inside `npu_sparse_flash_attention` from supplied indices |
| Cross-layer index reuse | Decoder tensor carrier | Same model path; no NPU patch |
| TP/CP/PP/EP semantics | TorchTitan sharding and wrappers | Uses the same outer model contracts |

`NpuSparseMLA` converts TorchTitan token-first tensors to the operator's BSND
layout, splits latent and RoPE dimensions, converts invalid or masked indices
to `-1`, and calls `torch_npu.npu_sparse_flash_attention`. The fused operator
returns latent-space output; the stock model still owns `W_V` and `W_O`.

Enable it explicitly:

```bash
./run_train.sh \
  --override.imports \
  torchtitanturbo.models.glm5.ops.sparse_mla.npu_sparse_mla
```

## GPU, NPU, and reference correspondence

All three paths implement the same `SparseMLA.Config` interface:

- reference: explicit top-k KV gather and PyTorch attention;
- GPU: local TileLang SparseMLA forward/backward;
- NPU: `npu_sparse_flash_attention`.

The model projections, top-k tensor, masks, state dict, and post-attention
`W_V/W_O` are shared. This boundary permits reference-vs-NPU capture and
comparison without changing the GLM model definition.

The implementation was checked against the public GLM-5 Slime flow for
absorbed MLA and index sharing, and against the Ascend DSA operator integration
used by the NPU DeepSeek implementation. Turbo does not import Slime.

## Current NPU limits

- The public fused NPU lightning-indexer geometry does not match GLM-5's
  index-head layout, so the PyTorch indexer remains active.
- `npu_sparse_flash_attention` requires a compatible CANN/torch_npu release,
  BF16 tensors, and supported production dimensions.
- Single-card forward/backward must pass before TP, CP, PP, EP, graph mode, or
  combined topology claims are made.
- Graph and distributed behavior are test obligations, not implied by the
  operator adapter alone.
