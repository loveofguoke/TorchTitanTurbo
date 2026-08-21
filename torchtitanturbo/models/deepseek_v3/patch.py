# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from torchtitan.tools.logging import logger


def _deepseek_update_from_config(self, job_config, **kwargs):
    """NPU-patched DeepSeek config update."""
    from torchtitan.config import JobConfig

    seq_len = job_config.training.max_context_length
    if seq_len > self.max_seq_len:
        logger.warning(
            f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
        )
    self.max_seq_len = seq_len

    if (
        job_config.parallelism.pipeline_parallel_degree > 1
        and job_config.parallelism.pipeline_parallel_schedule == "ZBVZeroBubble"
    ):
        self.moe_args.use_grouped_mm = False

    if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
        raise NotImplementedError("CP support for FlexAttention is still in progress.")

    self.moe_args._debug_force_load_balance = (
        job_config.training.debug_moe_force_load_balance
    )


def apply_patch():
    try:
        from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
        from torchtitan.models.deepseek_v3 import deepseekv3_args

        DeepSeekV3ModelArgs.update_from_config = _deepseek_update_from_config

        for config in deepseekv3_args.values():
            config.use_flex_attn = False

        logger.info("Patched torchtitan.models.deepseek_v3 for NPU")
    except ImportError:
        logger.warning("DeepSeek V3 model not available, skipping patch")
