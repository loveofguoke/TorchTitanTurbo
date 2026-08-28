# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from functools import partial

import torch

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitanturbo.models.glm5.patch import (
    _convert_router_config,
    _gather_router_scores,
    _local_vocab_labels,
    _npu_safe_trunc_normal_,
    apply_patch,
    NpuGlm5TokenChoiceTopKRouter,
)


class TestGlm5RouterPatch(unittest.TestCase):
    def test_npu_router_runs_bfloat16_gate_once_in_float32(self):
        router = _convert_router_config(
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=Linear.Config(in_features=3, out_features=4, bias=True),
                top_k=2,
                score_func="sigmoid",
            )
        ).build()
        router.bfloat16()
        hidden_states = torch.randn(1, 2, 3, dtype=torch.bfloat16)
        calls = 0

        def count_gate_calls(*_args):
            nonlocal calls
            calls += 1

        hook = router.gate.register_forward_hook(count_gate_calls)
        try:
            _, _, scores = router(hidden_states)
        finally:
            hook.remove()

        expected_scores = torch.sigmoid(
            torch.nn.functional.linear(
                hidden_states.float(),
                router.gate.weight.float(),
                router.gate.bias.float(),
            )
        )
        self.assertEqual(calls, 1)
        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)

    def test_npu_router_preserves_torchtitan_router_contract(self):
        torch.manual_seed(19)
        inputs_TD = torch.randn(11, 6)
        expert_bias_E = torch.linspace(-0.05, 0.05, 8)
        gate = Linear.Config(in_features=6, out_features=8, bias=False)
        cases = (
            (
                TokenChoiceTopKRouter.Config(
                    num_experts=8,
                    gate=gate,
                    num_expert_groups=4,
                    num_limited_groups=2,
                    top_k=2,
                    score_func="sigmoid",
                    route_norm=True,
                    route_scale=2.5,
                ),
                expert_bias_E,
            ),
            (
                TokenChoiceTopKRouter.Config(
                    num_experts=8,
                    gate=gate,
                    top_k=2,
                    score_func="softmax",
                ),
                None,
            ),
            (
                TokenChoiceTopKRouter.Config(
                    num_experts=8,
                    gate=gate,
                    top_k=2,
                    score_func="sigmoid",
                    route_norm=True,
                    route_scale=1.25,
                    _debug_force_load_balance=True,
                ),
                expert_bias_E,
            ),
        )

        for reference_config, bias in cases:
            with self.subTest(config=reference_config):
                reference = reference_config.build()
                candidate = _convert_router_config(reference_config).build()
                candidate.load_state_dict(reference.state_dict(), strict=True)

                expected = reference(inputs_TD, bias)
                actual = candidate(inputs_TD, bias)

                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(actual_tensor, expected_tensor)

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
