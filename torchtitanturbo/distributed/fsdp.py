# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from typing import Optional, Union


def _get_gradient_divide_factors(
    reduce_scatter_group: dist.ProcessGroup,
    all_reduce_group: Optional[dist.ProcessGroup],
    reduce_dtype: torch.dtype,
    device_type: str = "",
    factor: Optional[float] = None,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    Optional[float],
    Optional[float],
    Union[dist.ReduceOp, dist.ReduceOp.RedOpType],
    Union[dist.ReduceOp, dist.ReduceOp.RedOpType],
]:
    """NPU-patched gradient divide factors."""
    if device_type == "mtia" or device_type == "npu":
        force_sum_reduction_for_comms = True

    overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)

    data_parallel_size = reduce_scatter_group.size()
    if all_reduce_group is not None:
        data_parallel_size *= all_reduce_group.size()

    if factor is None:
        factor = float(data_parallel_size)

    if not overflow_risk and not force_sum_reduction_for_comms:
        if factor == data_parallel_size:
            if data_parallel_size == 1:
                return None, None, ReduceOp.SUM, ReduceOp.SUM
            return None, None, ReduceOp.AVG, ReduceOp.AVG
        else:
            return factor, None, ReduceOp.SUM, ReduceOp.SUM

    pre_factor: Optional[float]
    if overflow_risk:
        pre_factor = 1
        while factor % pre_factor == 0 and factor / pre_factor > pre_factor:
            pre_factor *= 2
        post_factor = factor / pre_factor
    else:
        pre_factor, post_factor = None, factor

    return pre_factor, post_factor, ReduceOp.SUM, ReduceOp.SUM


def apply_patch():
    import torch.distributed.fsdp._fully_shard._fsdp_collectives

    torch.distributed.fsdp._fully_shard._fsdp_collectives._get_gradient_divide_factors = _get_gradient_divide_factors
    from torchtitan.tools.logging import logger

    logger.info("Patched _get_gradient_divide_factors for NPU")
