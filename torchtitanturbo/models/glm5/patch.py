# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU compatibility patches for GLM-5."""

from dataclasses import dataclass, fields
from functools import partial
import math
import warnings

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.tools.logging import logger


_ORIGINAL_TRUNC_NORMAL = torch.nn.init.trunc_normal_


def _npu_safe_trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Initialize an NPU DTensor without a distributed scalar reduction.

    PyTorch 2.14 uses rejection sampling in ``trunc_normal_`` and evaluates
    ``mask.any()`` on every iteration. For a sharded DTensor that scalar
    result requires redistribution. On an Ascend multi-axis device mesh, the
    resulting lazy HCCL subgroup initialization can leave ranks entering
    different communicators. The inverse-CDF construction below samples the
    same truncated-normal distribution using only elementwise DTensor ops.
    """
    if not isinstance(tensor, DTensor):
        return _ORIGINAL_TRUNC_NORMAL(
            tensor,
            mean=mean,
            std=std,
            a=a,
            b=b,
            generator=generator,
        )
    if tensor.is_meta:
        return tensor
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    def norm_cdf(value: float) -> float:
        return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0

    lower = norm_cdf((a - mean) / std)
    upper = norm_cdf((b - mean) / std)
    with torch.no_grad():
        tensor.uniform_(2 * lower - 1, 2 * upper - 1, generator=generator)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def _replace_trunc_normal_initializers(param_init: dict) -> dict:
    converted = {}
    for name, initializer in param_init.items():
        if (
            isinstance(initializer, partial)
            and initializer.func is _ORIGINAL_TRUNC_NORMAL
        ):
            converted[name] = partial(
                _npu_safe_trunc_normal_,
                *initializer.args,
                **(initializer.keywords or {}),
            )
        else:
            converted[name] = initializer
    return converted


def _patch_glm5_param_initializers(glm5_module) -> None:
    glm5_module._LINEAR_INIT = _replace_trunc_normal_initializers(
        glm5_module._LINEAR_INIT
    )
    for name in ("_output_linear_init", "_depth_init", "_depth_experts_init"):
        current_factory = getattr(glm5_module, name)
        if getattr(current_factory, "_torchtitanturbo_npu_init_patched", False):
            continue

        def factory(*args, _factory=current_factory, **kwargs):
            return _replace_trunc_normal_initializers(
                _factory(*args, **kwargs)
            )

        factory._torchtitanturbo_npu_init_patched = True
        setattr(glm5_module, name, factory)


def _local_gather(scores: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(scores, dim=-1, index=index)


def _gather_router_scores(
    scores_TE: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
) -> torch.Tensor:
    """Run gather on rank-local shards while preserving DTensor metadata."""
    if not isinstance(scores_TE, DTensor):
        return _local_gather(scores_TE, topk_expert_ids_TK)

    assert isinstance(topk_expert_ids_TK, DTensor)
    gather_local = local_map(
        _local_gather,
        out_placements=(scores_TE.placements,),
        in_placements=(scores_TE.placements, topk_expert_ids_TK.placements),
        # local_map requires placement metadata for every DTensor input,
        # including integer routing indices that do not receive gradients.
        in_grad_placements=(
            scores_TE.placements,
            topk_expert_ids_TK.placements,
        ),
        device_mesh=scores_TE.device_mesh,
    )
    return gather_local(scores_TE, topk_expert_ids_TK)


class NpuFp32RouterLinear(Linear):
    """Linear gate with an explicit FP32 compute path for Ascend."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(
        self,
        input: torch.Tensor,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if compute_dtype is None:
            return super().forward(input)
        bias = None if self.bias is None else self.bias.to(compute_dtype)
        return F.linear(
            input.to(compute_dtype),
            self.weight.to(compute_dtype),
            bias,
        )


class NpuGlm5TokenChoiceTopKRouter(TokenChoiceTopKRouter):
    """GLM-5 router with a local DTensor gather boundary for NPU."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        pass

    def _debug_force_load_balance_routing(
        self, scores_TE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = scores_TE.shape[0]
        topk_expert_ids_TK = (
            torch.arange(
                num_tokens * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            ).reshape(num_tokens, self.top_k)
            % self.num_experts
        )
        topk_scores_TK = _gather_router_scores(
            scores_TE, topk_expert_ids_TK
        )
        return topk_expert_ids_TK, topk_scores_TK

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores_TE = self.gate(x_TD, compute_dtype=torch.float32)

        if self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_TE = self._get_node_limited_routing_scores(
                scores_for_choice_TE
            )
        _, topk_expert_ids_TK = torch.topk(
            scores_for_choice_TE,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        topk_scores_TK = _gather_router_scores(
            scores_TE, topk_expert_ids_TK
        )

        if self._debug_force_load_balance:
            topk_expert_ids_TK, topk_scores_TK = (
                self._debug_force_load_balance_routing(scores_TE)
            )

        if self.route_norm:
            denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_TK = topk_scores_TK / denominator
        topk_scores_TK = topk_scores_TK * self.route_scale

        return topk_scores_TK, topk_expert_ids_TK, scores_TE


def _convert_router_config(
    config: TokenChoiceTopKRouter.Config,
) -> NpuGlm5TokenChoiceTopKRouter.Config:
    config_kwargs = {
        config_field.name: getattr(config, config_field.name)
        for config_field in fields(config)
        if config_field.init
    }
    gate = config.gate
    gate_kwargs = {
        config_field.name: getattr(gate, config_field.name)
        for config_field in fields(gate)
        if config_field.init
    }
    config_kwargs["gate"] = NpuFp32RouterLinear.Config(**gate_kwargs)
    return NpuGlm5TokenChoiceTopKRouter.Config(**config_kwargs)


def apply_patch() -> None:
    """Apply NPU-safe initialization and router config replacements."""
    import torchtitan.models.glm5 as glm5_module

    _patch_glm5_param_initializers(glm5_module)
    current_factory = glm5_module.make_router_config
    if getattr(current_factory, "_torchtitanturbo_glm5_npu_patched", False):
        return

    def make_router_config(*args, **kwargs):
        return _convert_router_config(current_factory(*args, **kwargs))

    make_router_config._torchtitanturbo_glm5_npu_patched = True
    glm5_module.make_router_config = make_router_config
    logger.info("Patched GLM-5 router gather for NPU")
