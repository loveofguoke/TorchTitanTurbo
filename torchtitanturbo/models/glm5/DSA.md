# GLM DSA 测试分支适配

配套 TorchTitan `8ac999e8`（`codex/glm5-dsa-test`）。本次未新增运行时 patch：
现有 router、初始化、TP loss、公共 ComplexRoPE patch 的目标仍存在；
它们不替换 `Glm5FlexAttention`、indexer 或跨层索引来源。

## 边界

- 稀疏 BlockMask 和 MLA 数学由 TorchTitan 实现，不复制到 Turbo。
- 不以普通 SDPA/dense attention 静默替代 sparse DSA。
- 不移入旧 full-dsa 分支的未验证 kernels，也不修改其他模型。
- NPU 即使不编译整个模型，FlexAttention 仍依赖可用的 Inductor/NPU 编译后端。
- CANN 9.1.0 是当前实验环境要求，不意味着 FlexAttention 已通过验证。
- TP/CP/EP/FSDP 以及 PP stage 内共享沿用 TorchTitan 接入；跨 stage 索引共享仍不支持。

## 验证

```bash
python -m pytest tests/unit_tests/test_glm5_patch.py tests/unit_tests/test_glm5_dsa_contract.py -q > glm5-dsa-tests.log 2>&1
```

新增测试检查 patch 重复应用后保留 sparse 配置与 index_sources，并在 NPU 上对照
稀疏 attention 和显式 dense masked reference 的 FP32 前向及 Q/K/V 梯度。
Q 与 K 长度不同仅验证 CP 内核边界形状，不替代真实 CP collective 测试。
测试容差不是 HF 或 BF16 训练验收阈值。随后仍需 BF16、完整模型和多卡 smoke/精度验证。

本地没有 torch/torch_npu，当前仅完成语法和 diff 检查；测试尚未执行。
如果 NPU 编译失败，应保存完整日志，针对实际失败的 lowering/kernel 再增加最小 patch，
而不是提前假定支持或引入未经验证的回退实现。
