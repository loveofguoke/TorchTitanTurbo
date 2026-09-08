# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass

import torch
import torch_npu
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.nn_modules import RMSNorm
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
        weight = self.weight
        # Pipeline stages exchange local tensors, while a stage's degree-one
        # FSDP mesh can still expose its final norm parameter as a DTensor.
        # torch-npu requires every argument to an op to use the same tensor
        # representation, so materialize only this mixed local/DTensor boundary.
        # ``full_tensor`` both gathers a sharded FSDP weight and remains
        # connected to the DTensor parameter for backward.
        if isinstance(weight, DTensor) and not isinstance(x, DTensor):
            weight = weight.full_tensor()
        elif (
            isinstance(weight, DTensor)
            and isinstance(x, DTensor)
            and weight.device_mesh != x.device_mesh
        ):
            # PP+TP can leave a degree-one FSDP axis on the parameter mesh,
            # while pipeline activations use the TP submesh. Older DTensor
            # dispatch cannot mix those meshes. Re-express the complete norm
            # weight as replicated on the activation mesh; both conversions
            # are autograd-aware, so gradients still reach the parameter.
            weight = DTensor.from_local(
                weight.full_tensor(),
                device_mesh=x.device_mesh,
                placements=tuple(Replicate() for _ in x.placements),
                run_check=False,
            )

        # FSDP2 may release the all-gather storage backing ``self.weight``
        # after forward. torch-npu's RMSNorm backward retains the exact tensor
        # passed here, so keep an autograd-connected allocation alive until
        # backward instead of retaining the FSDP-managed view.
        weight = weight.clone()
        return torch_npu.npu_rms_norm(x, weight, epsilon=self.eps)[0]

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
