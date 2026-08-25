# GLM-5 昇腾已确认优化算子

本目录只放可通过 override 显式接入 GLM-5、且属于 Ascend 后端职责的算子适配。
普通 `apply_glm5_patch()` 不会自动启用这些算子。

## Triton-Ascend 路径

`triton.py::AscendTritonDSAIndexerTopK` 和 `AscendTritonSparseMLA` 从 TorchTitan 的
`models/glm5/ops/triton.py::TritonDSAIndexerTopK`、`TritonSparseMLA` 复用同一份 Triton 数学
kernel，只改变组件要求的设备类型并注册 NPU override：

```bash
--override.imports \
torchtitanturbo.models.glm5.ops.triton.npu_triton_dsa_indexer,torchtitanturbo.models.glm5.ops.triton.npu_triton_sparse_mla
```

- `AscendTritonDSAIndexerTopK`：Lightning Indexer score forward；
- `AscendTritonSparseMLA`：selected compressed-KV forward、dQ 和 dKV backward；
- 编译和运行依赖 Triton-Ascend，TorchTitan 本身不 import `torch_npu`。

共享源代码只说明 GPU/NPU 使用同一数学实现，不说明 Triton-Ascend 已支持所有 kernel
primitive。尤其是 dKV FP32 atomic、large top-k softmax、动态 shape 和图模式必须在目标
CANN/torch_npu/triton-ascend 组合上实测。

## Ascend native SparseMLA

`sparse_mla.py::NpuSparseMLA` 把 token-first tensor 适配到
`torch_npu.npu_sparse_flash_attention` 的 BSND 接口：

```bash
--override.imports \
torchtitanturbo.models.glm5.ops.sparse_mla.npu_sparse_mla
```

它保留 TorchTitan 外层的 Q/KV projection、跨层 index sharing、`W_V` 和 `W_O`，只替换
`SparseMLA.Config`。输入 top-k 已经编码 causal/packed-document 边界，因此 native op
不能再次套用错误的全局 causal mask。

## 分布式关系

算子不自行创建 TP/EP/CP/PP group。它消费 TorchTitan sharding/parallelize 已形成的
local tensor 契约：TP-local head、dense-region attention、CP local-query/global-key。
所以 TP+EP 在架构上兼容；实际支持仍需 smoke、backward、精度和 profiler 验收。
