# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from dataclasses import replace

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms
from torchtitan.models.glm5 import build_model_config
from torchtitan.models.glm5.dsa import DsaMetadata, Glm5SparseAttention
from torchtitan_recipes.tests.models.glm5 import glm5_debugmodel
from torchtitanturbo.models.glm5.attention import (
    build_dsa_block_mask,
    NpuGlm5FlexAttention,
    NpuGlm5FlexTransform,
)
from torchtitanturbo.models.glm5.patch import apply_patch


class TestGlm5DsaContract(unittest.TestCase):
    def test_patch_preserves_sparse_config_and_index_sources(self):
        before = build_model_config(
            "shared_dsa_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        apply_patch()
        apply_patch()
        after = build_model_config(
            "shared_dsa_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        self.assertEqual(after.index_sources, before.index_sources)
        for original, patched in zip(before.layers, after.layers, strict=True):
            self.assertIsInstance(
                patched.attention.inner_attention.inner_attention,
                Glm5SparseAttention.Config,
            )
            self.assertEqual(
                patched.attention.inner_attention.inner_attention,
                original.attention.inner_attention.inner_attention,
            )
            if patched.attention.indexer is not None:
                self.assertEqual(
                    patched.attention.indexer.topk, original.attention.indexer.topk
                )

    def test_explicit_transform_preserves_original_config(self):
        original = glm5_debugmodel(seq_len=128, flavor="glm5_3_debugmodel_mtp")
        transformed = apply_transforms(original, [NpuGlm5FlexTransform()])
        before = list(original.model.traverse(Glm5SparseAttention.Config))
        after = list(transformed.model.traverse(Glm5SparseAttention.Config))
        self.assertEqual(len(before), len(after))
        self.assertGreater(len(after), 1)
        self.assertTrue(
            all(
                type(config) is Glm5SparseAttention.Config for _, config, _, _ in before
            )
        )
        self.assertTrue(
            all(
                type(config) is NpuGlm5FlexAttention.Config for _, config, _, _ in after
            )
        )

    def test_mask_membership_is_pointwise_and_deduplicated(self):
        metadata = DsaMetadata.from_positions(torch.arange(4))
        indices = torch.tensor([[0, 0, -1], [1, 0, 99], [2, 0, 0], [3, 1, -2]])
        mask = build_dsa_block_mask(indices, metadata, block_size=2)
        q, k = torch.arange(4)[:, None], torch.arange(4)[None, :]
        actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), q, k)
        expected = (indices[:, :, None] == k).any(dim=1) & (q >= k)
        torch.testing.assert_close(actual, expected)
        from torch.fx.experimental.proxy_tensor import make_fx

        graph = make_fx(mask.mask_mod)(torch.tensor(0), torch.tensor(0), q, k)
        self.assertFalse(
            any(node.target == torch.ops.aten.any.dim for node in graph.graph.nodes)
        )

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
        metadata = DsaMetadata.from_positions(torch.arange(K, device="npu"))
        metadata = replace(metadata, query_indices_Q=rows + Q)
        attention = NpuGlm5FlexAttention.Config().build()
        actual = attention(
            q, k, v, attention_metadata=metadata, topk_indices_QS=indices, scale=H**-0.5
        )
        allowed = torch.zeros(Q, K, device="npu", dtype=torch.bool)
        allowed.scatter_(1, indices, True)
        scores = torch.einsum("qnh,knh->nqk", q, k) * H**-0.5
        probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        expected = torch.einsum("nqk,knh->qnh", probabilities, v)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (q, k, v), gradient)
        expected_grads = torch.autograd.grad(expected, (q, k, v), gradient)
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
