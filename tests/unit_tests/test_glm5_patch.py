# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from functools import partial
from unittest.mock import patch

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glm5 import build_model_config
from torchtitanturbo.models.glm5.patch import (
    _local_vocab_labels,
    _npu_safe_trunc_normal_,
    apply_patch,
)


class TestGlm5Patch(unittest.TestCase):
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
        expected = torch.where(expected_out_of_range, 0, safe_labels - 4)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_out_of_range, expected_out_of_range)

    def test_patch_updates_glm_flavor_initializers_and_is_idempotent(self):
        from torchtitan.components import loss as loss_module
        from torchtitanturbo.models.glm5 import patch as glm5_patch

        apply_patch()
        apply_patch()
        model_config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        linear_initializer = model_config.layers[0].attention.wq_a.param_init["weight"]
        self.assertIsInstance(linear_initializer, partial)
        self.assertIs(linear_initializer.func, _npu_safe_trunc_normal_)
        expert_initializer = model_config.layers[
            1
        ].feed_forward.routed_experts.w13.param_init["weight"]
        self.assertIsNotNone(expert_initializer)
        self.assertIs(
            loss_module._LossParallelCrossEntropy.forward,
            glm5_patch._npu_safe_loss_parallel_forward,
        )
        self.assertIs(
            loss_module._LossParallelCrossEntropy.backward,
            glm5_patch._npu_safe_loss_parallel_backward,
        )

    def test_init_and_loss_can_be_selected_independently(self):
        from torchtitanturbo.models.glm5 import patch as glm5_patch

        with (
            patch.object(glm5_patch, "_patch_vocab_parallel_loss") as loss,
            patch.object(glm5_patch, "_patch_glm5_param_initializers") as init,
        ):
            glm5_patch.apply_patch(initialization=False, loss=True)
            loss.assert_called_once()
            init.assert_not_called()
            loss.reset_mock()
            glm5_patch.apply_patch(initialization=False, loss=False)
            loss.assert_not_called()
            init.assert_not_called()

    def test_bootstrap_selector_rejects_unknown_patch_names(self):
        from torchtitanturbo.patch import _select_config_first_patches

        self.assertEqual(_select_config_first_patches("none"), ())
        self.assertEqual(
            _select_config_first_patches("glm_loss,rope"), ("rope", "glm_loss")
        )
        self.assertEqual(
            _select_config_first_patches(None), _select_config_first_patches("all")
        )
        with self.assertRaises(ValueError):
            _select_config_first_patches("roep")


if __name__ == "__main__":
    unittest.main()
