# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import sys
import types
import unittest
from unittest import mock

import torch

from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.models.glm5 import glm5_configs
from torchtitanturbo.models.glm5.ops.sparse_mla import (
    NpuSparseMLA,
    _npu_sparse_mla_forward,
)


_OVERRIDE = "torchtitanturbo.models.glm5.ops.sparse_mla.npu_sparse_mla"


class TestGlm5FullDsaOverride(unittest.TestCase):
    def test_override_is_explicit_and_does_not_change_default_patch(self):
        default_config = glm5_configs["debugmodel"]()
        self.assertNotIsInstance(
            default_config.layers[0].attention.inner_attention,
            NpuSparseMLA.Config,
        )

        replacements = apply_overrides(
            OverrideConfig(imports=[_OVERRIDE]), default_config
        )

        self.assertEqual(len(replacements), 8)
        self.assertTrue(
            all(
                isinstance(layer.attention.inner_attention, NpuSparseMLA.Config)
                for layer in default_config.layers
            )
        )

    def test_forward_adapts_token_first_inputs_to_npu_operator(self):
        q_QNH = torch.randn(3, 2, 6, dtype=torch.bfloat16)
        kv_K1H = torch.randn(4, 1, 6, dtype=torch.bfloat16)
        mask_1QK = torch.zeros(1, 3, 4, dtype=torch.bfloat16)
        mask_1QK[0, 0, 1:] = float("-inf")
        topk_QS = torch.tensor(
            [[0, 1], [1, 0], [2, 1]], dtype=torch.int32
        )
        expected_QNC = torch.randn(3, 2, 4, dtype=torch.bfloat16)
        fake_torch_npu = types.SimpleNamespace(
            npu_sparse_flash_attention=mock.Mock(
                return_value=(expected_QNC.unsqueeze(0), None, None)
            )
        )

        with mock.patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            actual_QNC = _npu_sparse_mla_forward(
                q_QNH,
                kv_K1H,
                mask_1QK,
                topk_QS,
                scale=0.5,
                latent_dim=4,
                training=True,
            )

        torch.testing.assert_close(actual_QNC, expected_QNC)
        call = fake_torch_npu.npu_sparse_flash_attention.call_args
        self.assertEqual(call.args[0].shape, (1, 3, 2, 4))
        self.assertEqual(call.args[1].shape, (1, 4, 1, 4))
        self.assertEqual(call.kwargs["sparse_indices"].shape, (1, 3, 1, 2))
        self.assertEqual(
            call.kwargs["sparse_indices"][0, 0, 0].tolist(), [0, -1]
        )
        self.assertEqual(call.kwargs["sparse_mode"], 0)


if __name__ == "__main__":
    unittest.main()
