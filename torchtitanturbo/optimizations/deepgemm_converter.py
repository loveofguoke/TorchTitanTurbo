"""Explicit model-config conversion for DeepGEMM-Ascend grouped experts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

from torchtitan.models.common.linear import GroupedLinear
from torchtitan.config.transform import ModelConfigConverter

logger = logging.getLogger(__name__)

from .deepgemm import DeepGemmGroupedBf16Provider


_deepgemm_grouped_linear_cache: dict[type, type] = {}


def _get_deepgemm_grouped_linear_cls(parent_cls: type) -> type:
    """Create a provider-backed grouped-linear subclass."""
    if parent_cls in _deepgemm_grouped_linear_cache:
        return _deepgemm_grouped_linear_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    class DeepGemmGroupedLinear(parent_cls):
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):
            pass

        def __init__(self, config: Config):
            super().__init__(config)
            self._deepgemm_provider = DeepGemmGroupedBf16Provider()

        def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
            return self._deepgemm_provider.forward(
                A_MK=input_RI,
                B_t_EKN=weight_EOI.transpose(-2, -1),
                offsets_E=offsets_E,
            )

    DeepGemmGroupedLinear.__name__ = f"DeepGemm{parent_cls.__name__}"
    DeepGemmGroupedLinear.__qualname__ = f"DeepGemm{parent_cls.__name__}"
    _deepgemm_grouped_linear_cache[parent_cls] = DeepGemmGroupedLinear
    return DeepGemmGroupedLinear


class DeepGemmGroupedExpertsConverter(ModelConfigConverter):
    """Replace only the grouped-MM seam with the opt-in DeepGEMM provider."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config):
        count = 0
        for _fqn, config, parent, attr in model_config.traverse(
            GroupedLinear.Config
        ):
            base_module_cls = type(config)._owner
            deepgemm_cls = _get_deepgemm_grouped_linear_cls(base_module_cls)
            config_cls = deepgemm_cls.Config
            new_config = config_cls(
                **{
                    field.name: getattr(config, field.name)
                    for field in fields(config)
                    if field.init
                }
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
            count += 1

        logger.info(
            "Converted %d GroupedLinear configs to DeepGEMM-Ascend", count
        )
        return model_config
