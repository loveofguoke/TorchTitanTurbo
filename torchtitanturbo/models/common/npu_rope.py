# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""NPU-optimized RoPE implementations using torch_npu.npu_rotary_mul."""

import torch
import torch_npu
from torchtitan.tools.logging import logger


def _complex_to_interleaved_cos_sin(
    freqs_cis: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cis.real.repeat_interleave(2, dim=-1)
    sin = freqs_cis.imag.repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2).to(dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(dtype)
    return cos, sin


def _wrap_dtensor_like(
    out_local: torch.Tensor, original_tensor: torch.Tensor, is_dt: bool
) -> torch.Tensor:
    if is_dt:
        from torch.distributed.tensor import DTensor

        return DTensor.from_local(
            out_local,
            device_mesh=original_tensor.device_mesh,
            placements=original_tensor.placements,
            run_check=False,
        )
    return out_local


def npu_apply_rotary_emb_cos_sin(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NPU-optimized cos/sin-format RoPE (Qwen3/GPT-OSS style).

    Uses torch_npu.npu_rotary_mul for fused rotary embedding.
    """
    from torch.distributed.tensor import DTensor
    from torchtitan.models.common.rope import (
        _maybe_wrap_positions,
        _reshape_for_broadcast_cos_sin,
    )

    xq_is_dt = isinstance(xq, DTensor)
    xk_is_dt = isinstance(xk, DTensor)
    xq_local = xq.to_local() if xq_is_dt else xq
    xk_local = xk.to_local() if xk_is_dt else xk
    rope_cache_local = (
        rope_cache.to_local() if isinstance(rope_cache, DTensor) else rope_cache
    )

    positions = _maybe_wrap_positions(positions, xq)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    head_dim = xq_local.shape[-1]
    rope_cache = _reshape_for_broadcast_cos_sin(rope_cache_local, xq_local, positions)
    cos = rope_cache[..., :head_dim].to(device=xq_local.device)
    sin = rope_cache[..., head_dim:].to(device=xq_local.device)

    xq_f = xq_local.float()
    xk_f = xk_local.float()
    xq_out = torch_npu.npu_rotary_mul(xq_f, cos, sin)
    xk_out = torch_npu.npu_rotary_mul(xk_f, cos, sin)

    xq_out = xq_out.type_as(xq_local)
    xk_out = xk_out.type_as(xk_local)

    xq_out = _wrap_dtensor_like(xq_out, xq, xq_is_dt)
    xk_out = _wrap_dtensor_like(xk_out, xk, xk_is_dt)

    return xq_out, xk_out


def npu_apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NPU-optimized complex-format RoPE (Llama3/4 style).

    Uses torch_npu.npu_rotary_mul with interleave mode.
    """
    from torch.distributed.tensor import DTensor
    from torchtitan.models.common.rope import _maybe_wrap_positions

    xq_is_dt = isinstance(xq, DTensor)
    xk_is_dt = isinstance(xk, DTensor)
    xq_local = xq.to_local() if xq_is_dt else xq
    xk_local = xk.to_local() if xk_is_dt else xk
    freqs_cis_local = (
        freqs_cis.to_local() if isinstance(freqs_cis, DTensor) else freqs_cis
    )

    positions = _maybe_wrap_positions(positions, xq)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    seqlen = xq_local.shape[1]
    if positions is None:
        freqs_cis = freqs_cis_local[0:seqlen]
    elif positions.size(0) == 1:
        freqs_cis = freqs_cis_local[positions.squeeze(0)]
    else:
        freqs_cis_expanded = freqs_cis_local[None, :, None, :].expand(
            xq_local.shape[0], -1, -1, -1
        )
        freqs_cis = torch.gather(
            freqs_cis_expanded,
            dim=1,
            index=positions.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, freqs_cis.shape[-1]),
        ).squeeze(2)

    xq_f = xq_local.float()
    xk_f = xk_local.float()

    cos, sin = _complex_to_interleaved_cos_sin(freqs_cis, xq_f.dtype)
    xq_out = torch_npu.npu_rotary_mul(xq_f, cos, sin, rotary_mode="interleave").type_as(
        xq_local
    )
    xk_out = torch_npu.npu_rotary_mul(
        xk_f, cos.to(xk_f.dtype), sin.to(xk_f.dtype), rotary_mode="interleave"
    ).type_as(xk_local)

    xq_out = _wrap_dtensor_like(xq_out, xq, xq_is_dt)
    xk_out = _wrap_dtensor_like(xk_out, xk, xk_is_dt)

    return xq_out, xk_out


def npu_apply_rotary_emb_single_complex(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """NPU-optimized single tensor RoPE (DeepSeek V3 MLA style).

    Uses torch_npu.npu_rotary_mul with interleave mode.
    """
    from torch.distributed.tensor import DTensor
    from torchtitan.models.common.rope import _maybe_wrap_positions

    is_dtensor = isinstance(x, DTensor)
    x_local = x.to_local() if is_dtensor else x
    freqs_cis_local = (
        freqs_cis.to_local() if isinstance(freqs_cis, DTensor) else freqs_cis
    )

    positions = _maybe_wrap_positions(positions, x)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    seqlen = x_local.shape[1]
    if positions is None:
        freqs_cis = freqs_cis_local[0:seqlen]
    elif positions.size(0) == 1:
        freqs_cis = freqs_cis_local[positions.squeeze(0)]
    else:
        freqs_cis_expanded = freqs_cis_local[None, :, None, :].expand(
            x_local.shape[0], -1, -1, -1
        )
        freqs_cis = torch.gather(
            freqs_cis_expanded,
            dim=1,
            index=positions.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, freqs_cis.shape[-1]),
        ).squeeze(2)

    x_f = x_local.float()

    cos, sin = _complex_to_interleaved_cos_sin(freqs_cis, x_f.dtype)
    y = torch_npu.npu_rotary_mul(x_f, cos, sin, rotary_mode="interleave")
    y = y.to(x_local.dtype)

    if is_dtensor:
        from torch.distributed.tensor import DTensor as _DTensor

        y = _DTensor.from_local(
            y, device_mesh=x.device_mesh, placements=x.placements, run_check=False
        )

    return y


def apply_patch():
    """Patch RoPE functions globally for NPU optimization.

    Uses unified replace_functions mechanism to handle both:
    - Module attributes: torchtitan.models.common.rope.xxx
    - Local references: from torchtitan.models.common.rope import xxx

    This ensures all code paths use NPU-optimized implementations.
    """
    from torchtitanturbo.tools.patch_utils import replace_functions
    from torchtitan.tools.logging import logger

    # Replace all three RoPE functions
    total_count = 0

    count = replace_functions(
        "apply_rotary_emb_complex",
        npu_apply_rotary_emb_complex,
        package="torchtitan",
    )
    total_count += count

    count = replace_functions(
        "apply_rotary_emb_cos_sin",
        npu_apply_rotary_emb_cos_sin,
        package="torchtitan",
    )
    total_count += count

    count = replace_functions(
        "apply_rotary_emb_single_complex",
        npu_apply_rotary_emb_single_complex,
        package="torchtitan",
    )
    total_count += count

    logger.info(f"Patched RoPE functions in {total_count} module references")
