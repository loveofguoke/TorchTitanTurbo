# GLM DSA 测试分支适配

配套 TorchTitan 当前 `daily/glm5-upstream-sync-20261013` 分支。Turbo 不复制
GLM 架构、router、`Glm5FlexAttention`、indexer 或跨层索引来源；仅保留经
NPU 运行证据证明仍有必要的初始化与词表并行 loss 兼容 patch。

## 边界

- 稀疏 BlockMask 和 MLA 数学由 TorchTitan 实现，不复制到 Turbo。
- 不以普通 SDPA/dense attention 静默替代 sparse DSA。
- 不移入旧 full-dsa 分支的未验证 kernels，也不修改其他模型。
- NPU 即使不编译整个模型，FlexAttention 仍依赖可用的 Inductor/NPU 编译后端。
- CANN 9.1.0 是当前实验环境要求，不意味着 FlexAttention 已通过验证。
- TP/CP/EP/FSDP 以及 PP stage 内共享沿用 TorchTitan 接入。CP DSA 使用
  TorchTitan 官方 `KVAllGatherCPFlexInnerAttention` 通信范式：Q 保持本地，
  K/V all-gather，反向使用其官方 reduce-scatter 语义。

## 验证

```bash
python -m pytest tests/unit_tests/test_glm5_patch.py tests/unit_tests/test_glm5_dsa_contract.py -q > glm5-dsa-tests.log 2>&1
```

测试检查 patch 重复应用后仍保留 sparse 配置与 `index_sources`，且不会把
TorchTitan 原生 router 替换成 Turbo 私有实现。Q 与 K 长度不同的单测只验证
CP 内核边界形状，不替代真实 collective 测试。测试容差也不是 HF 或 BF16
训练验收阈值；仍需 BF16、完整模型和多卡 smoke/精度验证。
如果 NPU 编译失败，应保存完整日志，针对实际失败的 lowering/kernel 再增加最小 patch，
而不是提前假定支持或引入未经验证的回退实现。

## 训练 kernel 原型（默认关闭）

当前分支新增
`torchtitanturbo.models.glm5.ops.triton_dsa.triton_dsa_training`。它针对当前
TorchTitan 展开的 `q[Q,N,H] / k[K,N,H] / v[K,N,V]` 合同实现 selected
attention，而不是复用旧分支已经失效的 compressed-KV `SparseMLA` 接口。

这组 Triton kernel 同时包含：

- forward：selected QK、FP32 softmax、PV；
- backward：softmax backward、`dQ`、`dK`、`dV`；
- 对多个 query 命中同一 key 的 `dK/dV` 使用 FP32 atomic accumulation；
- `spmd.register_local_autograd_function` 注册，与 TorchTitan KDA/GDN wrapper
  使用相同的本地 autograd 边界；
- 通过 TorchTitan `@override` 显式注册，默认 recipe 和 Turbo bootstrap 不导入；
- 不支持的设备、dtype、shape 直接报错，不回退到 dense 或 FlexAttention。

显式实验入口：

```bash
./run_train.sh \
  --override.imports \
  torchtitanturbo.models.glm5.ops.triton_dsa.triton_dsa_training
```

首阶段只替换非 CP 的 `Glm5FlexAttention.Config`。CP 使用独立的
`Glm5AllGatherKVFlexAttention.Config`，因此不会被误替换；等单卡前反向数值、
有限性和性能通过后，再为 CP 保留官方 all-gather/reduce-scatter 外层语义做
单独适配。该代码目前是实验原型，不能在完成 CUDA/NPU oracle、BF16 长序列、
dynamic shape、fullgraph、激活重计算和多卡验收前写成“已支持”。

FlashMLA Ascend 的公开 sparse API 仍只作为推理候选：它缺少该训练合同所需的
backward，且 `d_qk=d_v=512` 与当前 GLM 展开维度不相同，不能接入这个 override。

## CANN/AscendC 原生 kernel 现状

Kimi K3 KDA 的 `torchtitan-ascend` 适配并不是在 PR 中用 Triton 重写底层
kernel。它通过 `NpuChunkKDAFunction` 封装 `fla_npu` 已有的 AscendC/CANN
forward/backward，并补齐布局、autograd、SPMD 和 recipe override。因此 KDA
已有一组与其数学和训练合同匹配的原生内核，而当前 GLM DSA 尚未找到同等级、
可直接使用的完整 CANN 训练实现。

需要严格区分以下能力：

| 实现 | 覆盖范围 | 当前 GLM DSA 训练结论 |
|---|---|---|
| `fla_npu` KDA | KDA forward/backward | 数学不同，只能参考集成模式 |
| `torch_npu.npu_sparse_flash_attention` | CANN 稀疏注意力候选 | 旧实验适配针对 compressed KV；当前展开 Q/K/V、backward、shape 和 CP 合同未证明 |
| FlashMLA Ascend sparse | 高性能 sparse forward | 无匹配公开 backward，维度合同也不匹配 |
| DeepSelect Ascend | TopK selection | 不计算 sparse attention，不提供 dQ/dK/dV；NPU 仅 BF16 也不等价于当前 FP32 indexer |
| Turbo Triton DSA 原型 | 当前展开 Q/K/V forward 和 dQ/dK/dV | 接口完整但仍需 CUDA/NPU oracle、编译和训练验收 |

所以“CANN 有相关稀疏算子”不能写成“当前 GLM DSA 已有可用 CANN 训练
kernel”。后续必须先检查目标 torch_npu/CANN 版本的真实 schema、autograd
注册与支持矩阵；若没有匹配 backward，再以已通过数值 oracle 的 Triton 实现为
数学基线开发 AscendC provider。两个 provider 应共享同一模型接口，由显式
override 选择，禁止 forward-only 或 shape 不匹配时静默回退。
