# Temporary non-NPU compatibility patches

This directory isolates patches that are not caused directly by Ascend NPU or
`torch_npu` behavior. `torchtitanturbo.patch` applies them only after confirming
that an NPU is available, because they bridge the PyTorch/TorchTitan versions
shipped in the current NPU runtime; GPU and CPU imports remain unchanged.

Current patches:

- `runtime.py::_pp_forward_backward_step`: compatibility for PyTorch builds
  before upstream commit `8ed829041aff7b0ce83a66a5cf1b79a864b0954c`, which
  added explicit pre-split pipeline microbatch inputs.
- `runtime.py::_clip_grad_norm`: generic empty-gradient norm device placement.
  The issue was observed on NPU, but the missing device placement is not an
  NPU-specific behavior.

Reassess, remove, or upstream these patches when the corresponding runtime
versions provide the same behavior directly.
