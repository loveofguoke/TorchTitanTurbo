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
