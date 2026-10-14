# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from functools import partial

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
        linear_initializer = model_config.layers[0].attention.wq_a.param_init[
            "weight"
        ]
        self.assertIsInstance(linear_initializer, partial)
        self.assertIs(linear_initializer.func, _npu_safe_trunc_normal_)
        expert_initializer = model_config.layers[1].moe.routed_experts.w13.param_init[
            "weight"
        ]
        self.assertIsNotNone(expert_initializer)
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
