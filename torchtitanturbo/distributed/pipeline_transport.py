# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Select PyTorch's ordinary PP communicator when NPU cannot split groups."""

import logging
from functools import wraps

import torch.distributed as dist
import torch.distributed.config as dist_config

from torchtitan.distributed import utils

logger = logging.getLogger(__name__)


def _configure_transport(device, backend):
    if device.type != "npu" or backend.supports_splitting:
        return False
    dist_config.pipeline_per_edge_p2p = False
    logger.warning(
        "NPU backend lacks split_group; using the ordinary shared PP communicator"
    )
    return True


def apply_patch():
    """Keep original distributed init, then select the supported transport.

    TODO: Remove when HCCL supports the per-edge split protocol. This is for
    ordinary Trainer eager/local compile, not captured GraphTrainer PP replay.
    https://github.com/pytorch/pytorch/issues/186548
    """
    original = utils.init_distributed
    if getattr(original, "_turbo_pipeline_transport", False):
        return True

    @wraps(original)
    def initialize(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("pipeline_parallel_degree", 1) > 1 and dist.is_initialized():
            device = utils.get_local_device()
            if device.type == "npu":
                backend = dist.group.WORLD._get_backend(device)
                _configure_transport(device, backend)
        return result

    initialize._turbo_pipeline_transport = True
    utils.init_distributed = initialize
    return True
