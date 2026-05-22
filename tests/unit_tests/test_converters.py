# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest

from torchtitanturbo.models.common import (
    NpuGroupedExpertsConverter,
    NpuTokenDispatcherConverter,
    NpuRMSNormConverter,
    NpuSDPAConverter,
)
from torchtitan.protocols.model import ModelConfigConverter


class TestConverterInterface(unittest.TestCase):
    """Tests to ensure all converters have consistent interface."""

    def test_all_converters_have_config(self):
        """All converters have nested Config class."""
        converters = [
            NpuGroupedExpertsConverter,
            NpuTokenDispatcherConverter,
            NpuRMSNormConverter,
            NpuSDPAConverter,
        ]

        for converter_cls in converters:
            self.assertTrue(
                hasattr(converter_cls, "Config"),
                f"{converter_cls.__name__} missing Config class",
            )
            self.assertTrue(
                hasattr(converter_cls.Config, "__dataclass_fields__"),
                f"{converter_cls.__name__}.Config is not a dataclass",
            )

    def test_all_converter_configs_inherit_from_base(self):
        """All converter configs inherit from ModelConfigConverter.Config."""
        converters = [
            NpuGroupedExpertsConverter,
            NpuTokenDispatcherConverter,
            NpuRMSNormConverter,
            NpuSDPAConverter,
        ]

        for converter_cls in converters:
            config = converter_cls.Config()
            self.assertIsInstance(
                config,
                ModelConfigConverter.Config,
                f"{converter_cls.__name__}.Config must inherit from ModelConfigConverter.Config",
            )

    def test_all_converters_can_be_initialized(self):
        """All converters can be initialized with empty config."""
        converters = [
            NpuGroupedExpertsConverter,
            NpuTokenDispatcherConverter,
            NpuRMSNormConverter,
            NpuSDPAConverter,
        ]

        for converter_cls in converters:
            config = converter_cls.Config()
            converter = converter_cls(config)
            self.assertIsInstance(
                converter,
                converter_cls,
                f"{converter_cls.__name__} initialization failed",
            )

    def test_all_converters_have_convert_method(self):
        """All converters have convert() method."""
        converters = [
            NpuGroupedExpertsConverter,
            NpuTokenDispatcherConverter,
            NpuRMSNormConverter,
            NpuSDPAConverter,
        ]

        for converter_cls in converters:
            config = converter_cls.Config()
            converter = converter_cls(config)
            self.assertTrue(
                hasattr(converter, "convert"),
                f"{converter_cls.__name__} missing convert() method",
            )
            self.assertTrue(
                callable(converter.convert),
                f"{converter_cls.__name__}.convert is not callable",
            )


class TestConverterNaming(unittest.TestCase):
    """Tests to ensure converter naming follows convention."""

    def test_converter_names_end_with_converter(self):
        """All converter class names end with 'Converter'."""
        converters = [
            NpuGroupedExpertsConverter,
            NpuTokenDispatcherConverter,
            NpuRMSNormConverter,
            NpuSDPAConverter,
        ]

        for converter_cls in converters:
            self.assertTrue(
                converter_cls.__name__.endswith("Converter"),
                f"{converter_cls.__name__} should end with 'Converter'",
            )

    def test_converter_names_match_module_names(self):
        """Converter names follow Npu{Module}Converter pattern."""
        expected_patterns = {
            "NpuGroupedExpertsConverter": "NpuGroupedExperts",
            "NpuTokenDispatcherConverter": "NpuTokenDispatcher",
            "NpuRMSNormConverter": "NpuRMSNorm",
            "NpuSDPAConverter": "NpuScaledDotProductAttention",
        }

        for converter_name, expected_module in expected_patterns.items():
            converter_cls = getattr(
                __import__("torchtitanturbo.models.common", fromlist=[converter_name]),
                converter_name,
            )
            self.assertEqual(converter_cls.__name__, converter_name)


if __name__ == "__main__":
    unittest.main()
