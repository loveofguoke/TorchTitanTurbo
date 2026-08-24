# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from unittest.mock import patch

import torch

from torchtitanturbo.models.common.npu_rope import (
    _validate_torchtitan_rope_api,
    npu_apply_complex_rope,
    npu_reshape_for_broadcast,
)


class TestNpuRoPE(unittest.TestCase):
    def test_complex_cache_uses_token_first_positions(self):
        real = torch.arange(32, dtype=torch.float32).view(8, 4)
        cache = torch.complex(real, real + 100)
        positions_T = torch.tensor([0, 1, 2, 0, 1])

        actual_T1H = npu_reshape_for_broadcast(
            cache,
            (5, 3, 8),
            positions_T,
        )
        expected_T1H = cache[positions_T].view(5, 1, 4)

        self.assertEqual(actual_T1H.shape, (5, 1, 4))
        torch.testing.assert_close(actual_T1H, expected_T1H)

    def test_cache_without_positions_uses_token_dimension(self):
        cache = torch.arange(32, dtype=torch.float32).view(8, 4)

        actual_T1H = npu_reshape_for_broadcast(cache, (5, 3, 8))

        self.assertEqual(actual_T1H.shape, (5, 1, 4))
        torch.testing.assert_close(actual_T1H, cache[:5].view(5, 1, 4))

    @patch("torch_npu.npu_rotary_mul")
    def test_fused_complex_rope_matches_torchtitan_math(self, rotary_mul):
        def fake_rotary_mul(x, cos, sin, *, rotary_mode):
            self.assertEqual(rotary_mode, "interleave")
            x_pairs = x.reshape(*x.shape[:-1], -1, 2)
            rotated_pairs = torch.stack(
                (-x_pairs[..., 1], x_pairs[..., 0]), dim=-1
            )
            rotated = rotated_pairs.flatten(-2)
            return x * cos + rotated * sin

        rotary_mul.side_effect = fake_rotary_mul
        query_TNH = torch.randn(5, 3, 8, dtype=torch.bfloat16)
        key_TMH = torch.randn(5, 1, 8, dtype=torch.bfloat16)
        real = torch.randn(5, 1, 4)
        imag = torch.randn(5, 1, 4)
        norm = torch.sqrt(real.square() + imag.square())
        cache_T1C = torch.complex(real / norm, imag / norm)

        actual_query_TNH, actual_key_TMH = npu_apply_complex_rope(
            query_TNH,
            key_TMH,
            cache_T1C,
        )
        expected_query_TNH = torch.view_as_real(
            torch.view_as_complex(query_TNH.float().reshape(5, 3, 4, 2))
            * cache_T1C
        ).flatten(-2).to(query_TNH.dtype)
        expected_key_TMH = torch.view_as_real(
            torch.view_as_complex(key_TMH.float().reshape(5, 1, 4, 2))
            * cache_T1C
        ).flatten(-2).to(key_TMH.dtype)

        torch.testing.assert_close(actual_query_TNH, expected_query_TNH)
        torch.testing.assert_close(actual_key_TMH, expected_key_TMH)

    def test_adapter_matches_current_torchtitan_api(self):
        _validate_torchtitan_rope_api()


if __name__ == "__main__":
    unittest.main()
