# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass

import torch
import torch_npu
from torch import nn

from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


class NpuRMSNorm(Module):
    """NPU-optimized RMSNorm using torch_npu.npu_rms_norm."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-5
        elementwise_affine: bool = True
        param_init: dict | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.normalized_shape = config.normalized_shape
        self.weight = nn.Parameter(torch.ones(config.normalized_shape))
        if config.param_init and "weight" in config.param_init:
            config.param_init["weight"](self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class NpuRMSNormConverter(ModelConfigConverter):
    """Replace RMSNorm with NPU-optimized NpuRMSNorm."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        self.config = config
        logger.info("NpuRMSNormConverter initialized")

    def convert(self, model_config) -> None:
        count = 0
        for fqn, cfg, parent, attr in model_config.traverse(RMSNorm.Config):
            new_config = NpuRMSNorm.Config(
                normalized_shape=cfg.normalized_shape,
                eps=cfg.eps,
                elementwise_affine=cfg.elementwise_affine,
            )
            if hasattr(cfg, "param_init"):
                new_config.param_init = cfg.param_init

            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
            count += 1

        if count > 0:
            logger.info(f"Converted {count} RMSNorm to NpuRMSNorm")
