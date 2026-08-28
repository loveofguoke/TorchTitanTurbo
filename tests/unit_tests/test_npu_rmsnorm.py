# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import torch

from torchtitan.config import Configurable
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.model import ModelConfigConverter
from torchtitanturbo.models.common.npu_rmsnorm import (
    NpuRMSNorm,
    NpuRMSNormConverter,
)


class TestNpuRMSNorm(unittest.TestCase):
    """Tests for NpuRMSNorm module."""

    def test_config_build(self):
        """NpuRMSNorm.Config.build() creates module."""
        config = NpuRMSNorm.Config(normalized_shape=32)
        norm = config.build()
        self.assertIsInstance(norm, NpuRMSNorm)
        self.assertEqual(norm.normalized_shape, 32)

    def test_config_custom_eps(self):
        """NpuRMSNorm respects custom eps."""
        config = NpuRMSNorm.Config(normalized_shape=32, eps=1e-6)
        norm = config.build()
        self.assertEqual(norm.eps, 1e-6)

    def test_weight_initialization(self):
        """Weight is initialized to ones."""
        config = NpuRMSNorm.Config(normalized_shape=16)
        norm = config.build()
        self.assertTrue(torch.all(norm.weight == 1))

    @patch("torch_npu.npu_rms_norm")
    def test_forward_calls_npu_operator(self, mock_npu_rms_norm):
        """Forward calls torch_npu.npu_rms_norm."""
        mock_npu_rms_norm.return_value = (torch.randn(2, 10, 16),)

        config = NpuRMSNorm.Config(normalized_shape=16)
        norm = config.build()
        x = torch.randn(2, 10, 16)
        out = norm(x)

        mock_npu_rms_norm.assert_called_once()
        self.assertEqual(out.shape, torch.Size([2, 10, 16]))


class TestNpuRMSNormConverter(unittest.TestCase):
    """Tests for NpuRMSNormConverter."""

    def test_converter_config(self):
        """Converter has valid Config."""
        config = NpuRMSNormConverter.Config()
        self.assertIsInstance(config, ModelConfigConverter.Config)

    def test_converter_initialization(self):
        """Converter initializes correctly."""
        config = NpuRMSNormConverter.Config()
        converter = NpuRMSNormConverter(config)
        self.assertIsInstance(converter, NpuRMSNormConverter)

    def test_convert_replaces_rmsnorm_configs(self):
        """Convert replaces RMSNorm.Config with NpuRMSNorm.Config."""
        @dataclass(kw_only=True, slots=True)
        class ModelConfig(Configurable.Config):
            layers: list[Configurable.Config] = field(default_factory=list)

        model_config = ModelConfig(
            layers=[
                RMSNorm.Config(normalized_shape=32, eps=1e-5),
                RMSNorm.Config(normalized_shape=64, eps=1e-6),
            ]
        )

        converter = NpuRMSNormConverter(NpuRMSNormConverter.Config())
        converter.convert(model_config)

        self.assertIsInstance(model_config.layers[0], NpuRMSNorm.Config)
        self.assertIsInstance(model_config.layers[1], NpuRMSNorm.Config)
        self.assertEqual(model_config.layers[0].normalized_shape, 32)
        self.assertEqual(model_config.layers[1].normalized_shape, 64)


if __name__ == "__main__":
    unittest.main()
