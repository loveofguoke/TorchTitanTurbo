# GLM-5 昇腾优化原型

本目录只放尚未接入 GLM5、不会注册 override、不会被 patch 导入的 Ascend 原型。
通用计算/通信/内存原型在 TorchTitan 的 `models/glm5/ops_candidate/`；这里仅记录
HCCL、CANN、NPUGraph 等 NPU 特有问题。

| 维度 | 原型 | 参考思想 | GLM-5 落点 |
| --- | --- | --- | --- |
| 计算 | `compute/triton_topk_mask.py` | `torchtitan/models/glm5/model.py::DSAIndexerTopK.forward`；`slime_plugins/models/glm5/glm5.py` 的 `fused_select_topk` 局部函数 | INT32 top-k -> additive mask，规避 CANN INT64 scatter 候选 |
| 通信 | `communication/hccl_overlap_plan.py` | TorchTitan `experiments/graph_trainer/ep_eager_chunk.py::maybe_apply_ep_overlap_eager_chunking`；Megatron `transformer/moe/token_dispatcher.py::MoEAlltoAllTokenDispatcher` | EP dispatch/GMM/combine 的双缓冲 |
| 内存/图 | `memory/graph_workspace_plan.py` | TorchTitan `distributed/linear.py::AllGatherLinear`、`LinearReduceScatter`；NPUGraph 静态地址约束 | index/SparseMLA 临时空间 |

引用用 `仓库路径::符号`，不固定容易随上游提交漂移的行号。

## Triton-Ascend INT32 top-k mask

`triton_topk_additive_mask` 是未注册候选：先用 Triton fill kernel 写 `-inf`，再用
INT32 top-k index 将原 additive mask 的合法位置写回。它不调用 `torch.scatter`，也不把
index 转为 INT64；但它还没有融合 TopK 本身，而且两个 kernel 是否优于 CANN 路径必须
由目标 CANN/Triton-Ascend 版本的 profiler 决定。

```python
# Pseudocode only. Candidate is not an override.
topk_indices = indexer(...).to(torch.int32)
sparse_mask = triton_topk_additive_mask(topk_indices, attention_mask)
```

验收必须覆盖 packed-document mask、dynamic Q/K/top-k、重复/越界保护、eager/graph、
single/CP/TP 和端到端 loss/grad。只有 AiCPU ArgSort/Scatter 或 cast launch 消失且 DSA
区间和 profiler-off step 都改善，才考虑移入 `ops/`。

## HCCL 双缓冲计划

原型只生成 `(clock, chunk, phase, buffer_bank, buffer_slot)`，不调用 HCCL。input
和 output 各自双缓冲，避免同一 clock 的新 dispatch 覆盖旧 combine。未来接入伪代码：

```python
# Pseudocode only.
plan = plan_hccl_double_buffer(num_chunks=2)
for stage in plan:
    if stage.phase == "dispatch":
        handles[stage.chunk_id] = hccl_all_to_all_async(
            chunks[stage.chunk_id], stream=comm_stream, buffer=stage.buffer_slot
        )
    elif stage.phase == "compute":
        expert_out[stage.chunk_id] = npu_grouped_matmul(
            handles[stage.chunk_id].wait()
        )
    else:
        combine[stage.chunk_id] = hccl_all_to_all_async(
            expert_out[stage.chunk_id], stream=comm_stream
        )
```

对应 GLM MoE，而不是 attention 内部。实际参考符号还包括 Megatron
`megatron/core/transformer/moe/moe_utils.py::permute`、`unpermute`。正式实现必须保证 rank 发起顺序相同、动态 token
count 不触发 host sync、buffer 在 combine 完成前不复用，并覆盖 backward。验收看
Ascend timeline 上的 exposed HCCL 时间和 step critical path。

## graph-stable workspace

原型只规划固定对齐 offset，不分配 NPU tensor。未来由 Turbo 在 graph capture 前一次性
申请，再把 slice 传给 Triton-Ascend/CANN op：

```python
# Pseudocode only.
layout = plan_graph_workspace((
    ("index_score", score_bytes),
    ("sparse_probability", probability_bytes),
    ("dkv_fp32", dkv_bytes),
))
workspace = torch.empty(total_bytes(layout), dtype=torch.uint8, device="npu")
compile_and_capture(model, workspace_views=bind(layout, workspace))
```

它不能直接复用 eager pool：NPUGraph 需要稳定地址和稳定 shape，异步多 stream 还需要
明确 region 生命周期。验收看 allocation events、reserved memory、recompile/graph
break、replay 正确性和 step time。

## 晋级规则

候选进入 `ops/` 前必须补齐真实 HCCL/CANN/Triton-Ascend 接入、autograd、异常清理、
多 rank 单测、精度对齐、graph 测试和 Ascend Profiler 证据。仅有调度计划不算训练
优化，也不能出现在 torchtitan-test 的用户选项中。
