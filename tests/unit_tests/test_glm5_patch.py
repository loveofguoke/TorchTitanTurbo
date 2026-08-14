# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest

import torch

from torchtitanturbo.models.glm5.patch import (
    _gather_router_scores,
    apply_patch,
    NpuGlm5TokenChoiceTopKRouter,
)


class TestGlm5RouterPatch(unittest.TestCase):
    def test_plain_tensor_gather_matches_torch_forward_and_backward(self):
        index_BLK = torch.tensor([[[0, 2], [1, 3], [2, 0]]])
        actual_BLE = torch.randn(1, 3, 4, requires_grad=True)
        expected_BLE = actual_BLE.detach().clone().requires_grad_(True)

        actual_BLK = _gather_router_scores(actual_BLE, index_BLK)
        expected_BLK = torch.gather(expected_BLE, dim=-1, index=index_BLK)
        torch.testing.assert_close(actual_BLK, expected_BLK)

        grad_BLK = torch.randn_like(actual_BLK)
        actual_BLK.backward(grad_BLK)
        expected_BLK.backward(grad_BLK)
        torch.testing.assert_close(actual_BLE.grad, expected_BLE.grad)

    def test_patch_changes_only_glm5_router_configs(self):
        import torchtitan.models.glm5 as glm5

        apply_patch()
        model_config = glm5.glm5_configs["debugmodel"]()

        dense_layer = model_config.layers[0]
        moe_layer = model_config.layers[1]
        self.assertIsNone(dense_layer.moe)
        self.assertIsInstance(
            moe_layer.moe.router,
            NpuGlm5TokenChoiceTopKRouter.Config,
        )


if __name__ == "__main__":
    unittest.main()
