# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import torch

from torchtitan.config import Configurable
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model import ModelConfigConverter
from torchtitanturbo.models.common.npu_attention import (
    NpuScaledDotProductAttention,
    NpuSDPAConverter,
)


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

        q = torch.randn(bsz, seq_len, n_heads, head_dim)
        k = torch.randn(bsz, seq_len, n_heads, head_dim)
        v = torch.randn(bsz, seq_len, n_heads, head_dim)

        out = attention(q, k, v)

        mock_npu_fusion_attention.assert_called_once()
        call_args = mock_npu_fusion_attention.call_args

        torch.testing.assert_close(call_args[0][0], q.transpose(1, 2))
        torch.testing.assert_close(call_args[0][1], k.transpose(1, 2))
        torch.testing.assert_close(call_args[0][2], v.transpose(1, 2))
        self.assertEqual(call_args[0][3], n_heads)
        self.assertEqual(call_args[0][4], "BNSD")

        self.assertEqual(out.shape, torch.Size([bsz, seq_len, n_heads, head_dim]))

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
        @dataclass(kw_only=True, slots=True)
        class ModelConfig(Configurable.Config):
            attention: Configurable.Config = field(
                default_factory=ScaledDotProductAttention.Config
            )

        model_config = ModelConfig()

        converter = NpuSDPAConverter(NpuSDPAConverter.Config())
        converter.convert(model_config)

        self.assertIsInstance(
            model_config.attention, NpuScaledDotProductAttention.Config
        )


if __name__ == "__main__":
    unittest.main()
