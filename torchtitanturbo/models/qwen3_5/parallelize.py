# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.tensor_parallel import NoParallel
from .model import Qwen35Model
from torchtitan.tools.logging import logger


class Qwen35ExpertParallel(ParallelStyle):
    """Shard Qwen3.5 fused experts over the EP mesh.

    Qwen3.5 stores routed experts as fused ``gate_up_proj`` and ``down_proj``
    tensors, unlike the common TorchTitan MoE ``w1/w2/w3`` layout.  This style
    shards the expert dimension and records the local global-expert range so
    ``Qwen35MoeExperts.forward`` can compute only local expert contributions
    and sum them across the EP group.
    """

    @staticmethod
    def _partition_fn(name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        if module.num_experts % device_mesh.size() != 0:
            raise ValueError(
                "Qwen3.5 expert_parallel_degree must divide num_experts "
                f"({module.num_experts}), got {device_mesh.size()}."
            )
        local_experts = module.num_experts // device_mesh.size()
        module._ep_mesh = device_mesh
        module._local_expert_start = device_mesh.get_local_rank() * local_experts
        module._local_expert_count = local_experts
        module.register_parameter(
            "gate_up_proj",
            nn.Parameter(
                distribute_tensor(module.gate_up_proj, device_mesh, [Shard(0)])
            ),
        )
        module.register_parameter(
            "down_proj",
            nn.Parameter(distribute_tensor(module.down_proj, device_mesh, [Shard(0)])),
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, self._partition_fn)


def _has_moe(model: Qwen35Model) -> bool:
    return any(
        getattr(transformer_block, "moe_enabled", False)
        for transformer_block in model.layers.values()
    )


def _apply_qwen35_tp(
    model: Qwen35Model,
    tp_mesh: DeviceMesh,
    *,
    enable_loss_parallel: bool,
    enable_async_tp: bool,
    enable_cp: bool,
) -> None:
    """Apply TP to Qwen3.5 while keeping Gated DeltaNet replicated.

    The hybrid linear-attention layers carry recurrent state within the
    sequence and are not safe to shard head-wise yet.  Full-attention and dense
    FFN linears follow the Qwen3/Qwen3-VL style sharding plan.
    """

    top_level_plan = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "norm": NoParallel(local_output_grad_placements=(Replicate(),)),
        "lm_head": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
            use_local_output=not enable_loss_parallel,
        ),
    }
    parallelize_module(model, tp_mesh, top_level_plan)

    positions_sharding = Replicate() if enable_cp else None
    norm_plan = NoParallel(local_output_grad_placements=(Replicate(),))

    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": norm_plan,
            "ffn_norm": norm_plan,
        }
        if transformer_block.layer_type == "full_attention":
            layer_plan.update(
                {
                    "attention": PrepareModuleInput(
                        input_layouts=(
                            Replicate(),
                            Replicate(),
                            None,
                            positions_sharding,
                        ),
                        desired_input_layouts=(
                            Replicate(),
                            Replicate(),
                            None,
                            positions_sharding,
                        ),
                    ),
                    "attention.q_proj": ColwiseParallel(use_local_output=False),
                    "attention.k_proj": ColwiseParallel(use_local_output=False),
                    "attention.v_proj": ColwiseParallel(use_local_output=False),
                    "attention.q_norm": SequenceParallel(sequence_dim=2),
                    "attention.k_norm": SequenceParallel(sequence_dim=2),
                    "attention.o_proj": RowwiseParallel(output_layouts=Replicate()),
                }
            )
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward.w1": ColwiseParallel(),
                    "feed_forward.w2": RowwiseParallel(output_layouts=Replicate()),
                    "feed_forward.w3": ColwiseParallel(),
                }
            )
        else:
            layer_plan.update(
                {
                    "moe.shared_expert.w1": ColwiseParallel(),
                    "moe.shared_expert.w2": RowwiseParallel(
                        output_layouts=Replicate()
                    ),
                    "moe.shared_expert.w3": ColwiseParallel(),
                }
            )

        parallelize_module(transformer_block, tp_mesh, layer_plan)

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the Qwen3.5 model"
    )


def _apply_qwen35_ep(model: Qwen35Model, ep_mesh: DeviceMesh) -> None:
    for transformer_block in model.layers.values():
        if not getattr(transformer_block, "moe_enabled", False):
            continue
        parallelize_module(
            transformer_block.moe.experts,
            ep_mesh,
            Qwen35ExpertParallel(),
        )
    logger.info("Applied Expert Parallelism to Qwen3.5 fused MoE experts")


def _apply_qwen35_fsdp(
    model: Qwen35Model,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool,
    reshard_after_forward_policy: str,
    edp_mesh: DeviceMesh | None,
) -> None:
    """Apply FSDP for Qwen3.5.

    The top-level Qwen3.5 RMSNorm and lm_head can produce different unsharded
    gradient dtypes under FSDP mixed precision, so keep them in separate FSDP
    parameter groups instead of reusing the Llama helper's fused group.
    """

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.norm is not None:
        fully_shard(
            model.norm,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    if model.lm_head is not None:
        fully_shard(
            model.lm_head,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    expert_fsdp_config = fsdp_config
    if edp_mesh is not None:
        expert_fsdp_config = {
            **fsdp_config,
            "mesh": edp_mesh,
        }

    for transformer_block in model.layers.values():
        if transformer_block.moe_enabled:
            fully_shard(
                transformer_block.moe.experts,
                **expert_fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
    logger.info("Applied Qwen3.5 FSDP to the model")


def parallelize_qwen3_5(
    model: Qwen35Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply TorchTitan parallelisms to Qwen3.5."""
    if parallel_dims.cp_enabled:
        assert training.max_context_length % (parallel_dims.cp * 2) == 0, (
            f"Sequence length {training.max_context_length} must be divisible by "
            f"2 * context_parallel_degree ({parallel_dims.cp})."
        )

    if parallelism.full_dtensor:
        raise NotImplementedError("full_dtensor is not supported for Qwen3.5 yet.")

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")
        _apply_qwen35_tp(
            model,
            parallel_dims.get_mesh("tp"),
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
            enable_cp=parallel_dims.cp_enabled,
        )

    if parallel_dims.ep_enabled:
        if not _has_moe(model):
            raise ValueError("Qwen3.5 Expert Parallelism requires a MoE flavor.")
        _apply_qwen35_ep(model, parallel_dims.get_mesh("ep"))

    if parallel_dims.cp_enabled:
        cp_mesh = parallel_dims.get_mesh("cp")
        cp_modules = []
        for block in model.layers.values():
            layer_type = getattr(block, "layer_type", None)
            if layer_type == "full_attention":
                block.attention._cp_mesh = cp_mesh
                cp_modules.append(block.attention)
            elif layer_type == "linear_attention":
                block.linear_attn._cp_mesh = cp_mesh
                cp_modules.append(block.linear_attn)
        if cp_modules:
            logger.info("Applied Qwen3.5 NPU Context Parallel attention/recurrent path")

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    _apply_qwen35_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        edp_mesh=edp_mesh,
    )
    logger.info("Applied fully_shard to the Qwen3.5 model")
    return model
