# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile

from .model import GraphTrainerQwen35Model


def annotate_qwen3_5(model: GraphTrainerQwen35Model) -> None:
    """Attach module FQN annotations consumed by graph_trainer passes."""
    annotate_module_fqns(model)


def parallelize_qwen3_5(
    model: GraphTrainerQwen35Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply graph_trainer SimpleFSDP and graph compilation to Qwen3.5."""
    if parallel_dims.tp_enabled:
        raise NotImplementedError("Qwen3.5 graph_trainer TP is not enabled yet.")
    if parallel_dims.cp_enabled:
        raise NotImplementedError("Qwen3.5 graph_trainer CP is not enabled yet.")
    if parallel_dims.ep_enabled:
        raise NotImplementedError("Qwen3.5 graph_trainer EP is not enabled yet.")

    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}),
        i.e. {parallel_dims.seq_len_divisor}.
        """

    annotate_qwen3_5(model)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    # graph_trainer uses SimpleFSDP even when fsdp degree is 1 so parameter
    # casting follows the same path as existing llama3/qwen3 graph_trainer code.
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model
