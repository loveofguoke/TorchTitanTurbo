# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Keep GLM-5's final norm and LM head in independent FSDP units."""

from functools import wraps
from typing import Any

from torchtitan.tools.logging import logger


def apply_patch() -> None:
    """Split the untied final norm and LM head FSDP unit on NPU.

    Pipeline stages execute the final norm inside the stage and the LM head in
    the loss wrapper. Sharing one FSDP unit across those two execution
    boundaries can leave the NPU FSDP forward state associated with a previous
    microbatch. Keep each module in the FSDP unit that matches its actual
    forward boundary instead.
    """
    import torchtitan.distributed.fsdp as fsdp_module
    import torchtitan.models.glm5.parallelize as glm5_parallelize

    current = glm5_parallelize.apply_fsdp_to_decoder
    if getattr(current, "_torchtitanturbo_glm5_final_units_patched", False):
        return

    @wraps(current)
    def apply_fsdp_with_independent_final_units(
        model: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_fully_shard = fsdp_module.fully_shard

        @wraps(original_fully_shard)
        def fully_shard_with_independent_final_units(
            module_or_modules: Any,
            *shard_args: Any,
            **shard_kwargs: Any,
        ) -> Any:
            is_grouped_final_unit = (
                not model.enable_weight_tying
                and model.norm is not None
                and model.lm_head is not None
                and isinstance(module_or_modules, (list, tuple))
                and len(module_or_modules) == 2
                and module_or_modules[0] is model.norm
                and module_or_modules[1] is model.lm_head
            )
            if not is_grouped_final_unit:
                return original_fully_shard(
                    module_or_modules,
                    *shard_args,
                    **shard_kwargs,
                )

            original_fully_shard(model.norm, *shard_args, **shard_kwargs)
            return original_fully_shard(
                model.lm_head,
                *shard_args,
                **shard_kwargs,
            )

        fsdp_module.fully_shard = fully_shard_with_independent_final_units
        try:
            return current(model, *args, **kwargs)
        finally:
            fsdp_module.fully_shard = original_fully_shard

    setattr(
        apply_fsdp_with_independent_final_units,
        "_torchtitanturbo_glm5_final_units_patched",
        True,
    )
    glm5_parallelize.apply_fsdp_to_decoder = (
        apply_fsdp_with_independent_final_units
    )
    logger.info("Patched GLM-5 final norm and LM head into independent FSDP units")


__all__ = ["apply_patch"]
