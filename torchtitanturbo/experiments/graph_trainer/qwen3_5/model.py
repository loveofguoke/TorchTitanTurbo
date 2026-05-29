# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass

import torch

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    disable_active_parametrization,
)
from torchtitanturbo.models.qwen3_5 import Qwen35Model


class GraphTrainerQwen35Model(Qwen35Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        pass

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)
