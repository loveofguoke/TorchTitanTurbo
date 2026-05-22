# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from unittest.mock import MagicMock, patch

import torch

from torchtitanturbo.models.common.npu_attention import (
    NpuScaledDotProductAttention,
    NpuSDPAConverter,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model import ModelConfigConverter


class TestNpuScaledDotProductAttention(unittest.TestCase):
    """Tests for NpuScaledDotProductAttention module."""

    def test_config_build(self):
        """NpuScaledDotProductAttention.Config.build() creates module."""
        config = NpuScaledDotProductAttention.Config()
        attention = config.build()
        self.assertIsInstance(attention, NpuScaledDotProductAttention)

    @patch("torch_npu.npu_fusion_attention")
    def test_forward_calls_npu_operator(self, mock_npu_fusion_attention):
        """Forward calls torch_npu.npu_fusion_attention with correct format."""
        bsz, n_heads, seq_len, head_dim = 2, 4, 16, 64
        mock_output = torch.randn(bsz, n_heads, seq_len, head_dim)
        mock_npu_fusion_attention.return_value = (mock_output,)

        config = NpuScaledDotProductAttention.Config()
        attention = config.build()

        q = torch.randn(bsz, n_heads, seq_len, head_dim)
        k = torch.randn(bsz, n_heads, seq_len, head_dim)
        v = torch.randn(bsz, n_heads, seq_len, head_dim)

        out = attention(q, k, v)

        mock_npu_fusion_attention.assert_called_once()
        call_args = mock_npu_fusion_attention.call_args

        self.assertEqual(call_args[0][0], q)
        self.assertEqual(call_args[0][1], k)
        self.assertEqual(call_args[0][2], v)
        self.assertEqual(call_args[0][3], n_heads)
        self.assertEqual(call_args[1]["input_layout"], "BNSD")

        self.assertEqual(out.shape, torch.Size([bsz, n_heads, seq_len, head_dim]))

    @patch("torch_npu.npu_fusion_attention")
    def test_forward_with_custom_scale(self, mock_npu_fusion_attention):
        """Forward respects custom scale parameter."""
        mock_output = torch.randn(2, 4, 16, 64)
        mock_npu_fusion_attention.return_value = (mock_output,)

        config = NpuScaledDotProductAttention.Config()
        attention = config.build()

        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)
        v = torch.randn(2, 4, 16, 64)

        custom_scale = 0.5
        attention(q, k, v, scale=custom_scale)

        call_args = mock_npu_fusion_attention.call_args
        self.assertEqual(call_args[1]["scale"], custom_scale)


class TestNpuSDPAConverter(unittest.TestCase):
    """Tests for NpuSDPAConverter."""

    def test_converter_config(self):
        """Converter has valid Config."""
        config = NpuSDPAConverter.Config()
        self.assertIsInstance(config, ModelConfigConverter.Config)

    def test_converter_initialization(self):
        """Converter initializes correctly."""
        config = NpuSDPAConverter.Config()
        converter = NpuSDPAConverter(config)
        self.assertIsInstance(converter, NpuSDPAConverter)

    def test_convert_replaces_sdpa_configs(self):
        """Convert replaces ScaledDotProductAttention.Config."""
        from torchtitan.protocols.model import ModelConfig

        model_config = ModelConfig()
        model_config.attention = ScaledDotProductAttention.Config()

        converter = NpuSDPAConverter(NpuSDPAConverter.Config())
        converter.convert(model_config)

        self.assertIsInstance(
            model_config.attention, NpuScaledDotProductAttention.Config
        )


if __name__ == "__main__":
    unittest.main()
