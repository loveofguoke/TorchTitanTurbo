# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest

import torch

from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glm5 import build_model_config
from torchtitan.models.glm5.dsa import Glm5AllGatherKVFlexAttention
from torchtitanturbo.models.glm5.ops.triton_dsa import (
    TritonDsaAttention,
    TritonDsaCapabilityError,
    _TritonDsaFunction,
)
from torchtitanturbo.optimizations.registry import OptimizationProvidersConfig


_OVERRIDE = "torchtitanturbo.models.glm5.ops.triton_dsa.triton_dsa_training"


class TestGlm5TritonDsaRegistration(unittest.TestCase):
    def test_provider_registry_exposes_explicit_override(self):
        selected = OptimizationProvidersConfig(
            sparse_attention_backend="triton_dsa_training"
        ).build_selected()["sparse_attention"]
        self.assertTrue(selected.training_forward)
        self.assertTrue(selected.training_backward)
        self.assertFalse(selected.runtime_accepted)
        self.assertEqual(selected.override_import, _OVERRIDE)

    def test_override_is_explicit_and_replaces_non_cp_attention(self):
        config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        self.assertNotIsInstance(
            config.layers[0].attention.inner_attention,
            TritonDsaAttention.Config,
        )
        replacements = apply_overrides(
            OverrideConfig(imports=[_OVERRIDE]), config
        )
        self.assertGreater(len(replacements), 0)
        self.assertTrue(
            all(
                isinstance(
                    layer.attention.inner_attention, TritonDsaAttention.Config
                )
                for layer in config.layers
            )
        )

    def test_cp_attention_is_not_silently_replaced(self):
        config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(context_parallel_degree=2),
            seq_len=128,
        )
        apply_overrides(OverrideConfig(imports=[_OVERRIDE]), config)
        self.assertTrue(
            all(
                isinstance(
                    layer.attention.inner_attention,
                    Glm5AllGatherKVFlexAttention.Config,
                )
                for layer in config.layers
            )
        )

    def test_cpu_execution_is_rejected_without_fallback(self):
        q = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        k = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        v = torch.randn(4, 2, 6, dtype=torch.bfloat16)
        mask = torch.zeros(1, 4, 4)
        indices = torch.arange(4, dtype=torch.int32).unsqueeze(1)
        with self.assertRaisesRegex(TritonDsaCapabilityError, "CUDA or NPU"):
            _TritonDsaFunction.apply(q, k, v, mask, indices, 8**-0.5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Triton")
    def test_cuda_forward_backward_matches_selected_attention(self):
        torch.manual_seed(61)
        Q, K, N, H, V, S = 5, 7, 2, 16, 12, 3
        q = torch.randn(
            Q, N, H, device="cuda", dtype=torch.float16, requires_grad=True
        )
        k = torch.randn(
            K, N, H, device="cuda", dtype=torch.float16, requires_grad=True
        )
        v = torch.randn(
            K, N, V, device="cuda", dtype=torch.float16, requires_grad=True
        )
        indices = torch.tensor(
            [[0, 1, 2], [1, 0, 3], [2, 1, 0], [3, 2, 1], [4, 3, 2]],
            device="cuda",
            dtype=torch.int32,
        )
        mask = torch.full((1, Q, K), float("-inf"), device="cuda")
        mask[0] = torch.triu(mask[0], diagonal=1)
        scale = H**-0.5
        actual = _TritonDsaFunction.apply(q, k, v, mask, indices, scale)

        selected_k = k[indices.long()]
        selected_v = v[indices.long()]
        scores = torch.einsum("qnh,qsnh->qns", q, selected_k) * scale
        probabilities = scores.float().softmax(-1).to(q.dtype)
        expected = torch.einsum("qns,qsnv->qnv", probabilities, selected_v)
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual, (q, k, v), gradient, retain_graph=True
        )
        expected_grads = torch.autograd.grad(expected, (q, k, v), gradient)

        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(
                actual_grad, expected_grad, atol=4e-3, rtol=4e-3
            )


if __name__ == "__main__":
    unittest.main()
