# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.tools.logging import logger


def _qwen3_apply_non_moe_tp(
    model,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    """NPU-compatible tensor parallelism for Qwen3."""
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    attention_kernel_plan = prepare_module_input(
        input_layouts=(Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=True,
    )

    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), Replicate(), None),
                desired_input_layouts=(Replicate(), Replicate(), None),
            ),
            "attention.wq": colwise_parallel(use_local_output=False),
            "attention.wk": colwise_parallel(use_local_output=False),
            "attention.wv": colwise_parallel(use_local_output=False),
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            "attention.inner_attention": attention_kernel_plan,
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        f"{'Async ' if enable_async_tp else ''}Tensor Parallelism to the model"
    )


def apply_patch():
    try:
        from torchtitan.models.qwen3.infra import parallelize

        parallelize.apply_non_moe_tp = _qwen3_apply_non_moe_tp
        logger.info("Patched torchtitan.models.qwen3.infra.parallelize for NPU")
    except ImportError:
        logger.warning("Qwen3 model not available, skipping patch")
