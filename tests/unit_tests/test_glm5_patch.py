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
