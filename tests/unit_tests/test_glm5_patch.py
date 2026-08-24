# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from functools import partial

import torch

from torchtitanturbo.models.glm5.patch import (
    _gather_router_scores,
    _local_vocab_labels,
    _npu_safe_trunc_normal_,
    apply_patch,
    NpuGlm5TokenChoiceTopKRouter,
)


class TestGlm5RouterPatch(unittest.TestCase):
    def test_local_vocab_labels_match_boolean_indexing(self):
        labels = torch.tensor([-100, 0, 3, 4, 7, 8, 15])
        actual, actual_out_of_range = _local_vocab_labels(
            labels,
            vocab_start=4,
            local_vocab_size=4,
            ignore_index=-100,
        )

        safe_labels = torch.where(labels != -100, labels, 0)
        expected_out_of_range = (safe_labels < 4) | (safe_labels >= 8)
        expected = safe_labels - 4
        expected[expected_out_of_range] = 0
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual_out_of_range, expected_out_of_range
        )

    def test_plain_tensor_gather_matches_torch_forward_and_backward(self):
        index_TK = torch.tensor([[0, 2], [1, 3], [2, 0]])
        actual_TE = torch.randn(3, 4, requires_grad=True)
        expected_TE = actual_TE.detach().clone().requires_grad_(True)

        actual_TK = _gather_router_scores(actual_TE, index_TK)
        expected_TK = torch.gather(expected_TE, dim=-1, index=index_TK)
        torch.testing.assert_close(actual_TK, expected_TK)

        grad_TK = torch.randn_like(actual_TK)
        actual_TK.backward(grad_TK)
        expected_TK.backward(grad_TK)
        torch.testing.assert_close(actual_TE.grad, expected_TE.grad)

    def test_patch_changes_only_glm5_router_configs(self):
        import torchtitan.models.glm5 as glm5
        from torchtitan.components import loss as loss_module
        from torchtitanturbo.models.glm5 import patch as glm5_patch

        apply_patch()
        model_config = glm5.glm5_configs["debugmodel"]()

        dense_layer = model_config.layers[0]
        moe_layer = model_config.layers[1]
        self.assertIsNone(dense_layer.moe)
        self.assertIsInstance(
            moe_layer.moe.router,
            NpuGlm5TokenChoiceTopKRouter.Config,
        )
        linear_initializer = dense_layer.attention.wq_a.param_init["weight"]
        self.assertIsInstance(linear_initializer, partial)
        self.assertIs(linear_initializer.func, _npu_safe_trunc_normal_)

        expert_initializer = (
            moe_layer.moe.routed_experts.inner_experts.param_init["w1_EFD"]
        )
        self.assertIsInstance(expert_initializer, partial)
        self.assertIs(expert_initializer.func, _npu_safe_trunc_normal_)
        self.assertIs(
            loss_module._LossParallelCrossEntropy.forward,
            glm5_patch._npu_safe_loss_parallel_forward,
        )
        self.assertIs(
            loss_module._LossParallelCrossEntropy.backward,
            glm5_patch._npu_safe_loss_parallel_backward,
        )


if __name__ == "__main__":
    unittest.main()
