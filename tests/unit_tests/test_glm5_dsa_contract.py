# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest

import torch

from torchtitan.models.glm5 import glm5_configs
from torchtitan.models.glm5.dsa import Glm5FlexAttention
from torchtitanturbo.models.glm5.patch import apply_patch


class TestGlm5DsaContract(unittest.TestCase):
    def test_patch_preserves_sparse_config_and_index_sources(self):
        before = glm5_configs["shared_dsa_debugmodel"]()
        apply_patch()
        apply_patch()
        after = glm5_configs["shared_dsa_debugmodel"]()
        self.assertEqual(after.index_sources, before.index_sources)
        for original, patched in zip(before.layers, after.layers, strict=True):
            self.assertIsInstance(patched.attention.inner_attention, Glm5FlexAttention.Config)
            self.assertEqual(patched.attention.inner_attention, original.attention.inner_attention)
            self.assertEqual(patched.attention.indexer.topk, original.attention.indexer.topk)

    @unittest.skipUnless(
        hasattr(torch, "npu") and torch.npu.is_available(),
        "requires an NPU and its FlexAttention compiler backend",
    )
    def test_npu_sparse_forward_backward_matches_dense_reference(self):
        torch.manual_seed(61)
        # Different Q/K lengths exercise the local-query/global-key CP boundary.
        Q, K, N, H = 128, 256, 2, 64
        q = torch.randn(Q, N, H, device="npu", requires_grad=True)
        k = torch.randn(K, N, H, device="npu", requires_grad=True)
        v = torch.randn(K, N, H, device="npu", requires_grad=True)
        rows = torch.arange(Q, device="npu")
        indices = torch.stack((rows, rows + Q), dim=-1)
        mask = torch.zeros(1, Q, K, device="npu")
        attention = Glm5FlexAttention.Config().build()
        actual = attention(q, k, v, mask, indices, scale=H**-0.5)
        allowed = torch.zeros(Q, K, device="npu", dtype=torch.bool)
        allowed.scatter_(1, indices, True)
        scores = torch.einsum("qnh,knh->nqk", q, k) * H**-0.5
        probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        expected = torch.einsum("nqk,knh->qnh", probabilities, v)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (q, k, v), gradient)
        expected_grads = torch.autograd.grad(expected, (q, k, v), gradient)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
