# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Token-first TorchTitan ComplexRoPE support for Ascend NPU.

TorchTitan stores RoPE as a complex cache because complex multiplication is a
compact reference formula. The affected Ascend gather cannot index complex64,
so this adapter gathers real/imaginary FP32 components and reconstructs the
same angles. Rotation then uses ``npu_rotary_mul`` on real token-first tensors.
DTensor wrappers are restored after the local fused kernel.
"""

import inspect

import spmd_types as spmd
import torch
import torch_npu

import logging

logger = logging.getLogger(__name__)


@spmd.local_map(
    out_types=(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
        spmd.PartitionSpec(("dp", "cp"), None, None),
    )
)
def npu_reshape_for_broadcast(
    rope_cache: torch.Tensor,
    query_shape: torch.Size | tuple[int, ...],
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select a token-first RoPE cache without complex NPU gather.

    This matches ``torchtitan.models.common.rope._reshape_for_broadcast``:
    ``query_shape`` is ``[T, N, H]``, positions is ``[T]``, and the result is
    ``[T, 1, cache_width]``. Ascend gather does not accept a complex64 input,
    so complex caches gather their real and imaginary FP32 components before
    reconstruction.
    """
    cache_width = rope_cache.shape[-1]
    num_tokens = query_shape[0]
    if positions is None:
        selected = rope_cache[:num_tokens]
    elif torch.is_complex(rope_cache):
        selected = torch.complex(
            rope_cache.real[positions],
            rope_cache.imag[positions],
        )
    else:
        selected = rope_cache[positions]
    return selected.view(num_tokens, 1, cache_width)


def _wrap_dtensor_like(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    if not isinstance(reference, DTensor):
        return output
    return DTensor.from_local(
        output,
        device_mesh=reference.device_mesh,
        placements=reference.placements,
        run_check=False,
    )


def npu_apply_complex_rope(
    query: torch.Tensor,
    key: torch.Tensor | None,
    rope_cache: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply token-first complex RoPE with the fused Ascend kernel."""
    from torch.distributed.tensor import DTensor

    query_local = query.to_local() if isinstance(query, DTensor) else query
    key_local = key.to_local() if isinstance(key, DTensor) else key
    cache_local = (
        rope_cache.to_local() if isinstance(rope_cache, DTensor) else rope_cache
    )

    # Complex cache width is H/2. Interleaved real rotation needs each cosine
    # and sine duplicated for the adjacent even/odd feature pair.
    cos_1T1H = cache_local.real.repeat_interleave(2, dim=-1).unsqueeze(0)
    sin_1T1H = cache_local.imag.repeat_interleave(2, dim=-1).unsqueeze(0)
    if inverse:
        sin_1T1H = -sin_1T1H
    query_1TNH = query_local.float().unsqueeze(0)
    query_out_TNH = torch_npu.npu_rotary_mul(
        query_1TNH,
        cos_1T1H,
        sin_1T1H,
        rotary_mode="interleave",
    ).squeeze(0)
    query_out_TNH = query_out_TNH.type_as(query_local)
    query_result = _wrap_dtensor_like(query_out_TNH, query)
    if key_local is None:
        return query_result
    key_1TMH = key_local.float().unsqueeze(0)
    key_out_TMH = torch_npu.npu_rotary_mul(
        key_1TMH,
        cos_1T1H,
        sin_1T1H,
        rotary_mode="interleave",
    ).squeeze(0)
    key_out_TMH = key_out_TMH.type_as(key_local)
    return (
        query_result,
        _wrap_dtensor_like(key_out_TMH, key),
    )


def npu_reshape_complex_cache(self, query, positions=None):
    """Gather a complex cache through supported real FP32 NPU tensors."""
    cache = self.cache
    num_tokens = query.shape[0]
    if positions is None:
        selected = cache[:num_tokens]
    else:
        selected = torch.complex(cache.real[positions], cache.imag[positions])
    return selected.view(num_tokens, 1, cache.shape[-1])


def _validate_torchtitan_rope_api() -> None:
    """Fail clearly when TorchTitan changes the RoPE patch contract."""
    from torchtitan.models.common import rope as rope_module

    expected_reshape = ("self", "query", "positions")
    reshape_parameters = tuple(inspect.signature(rope_module.ComplexRoPE._reshape_cache).parameters)
    if reshape_parameters != expected_reshape:
        raise RuntimeError(
            "TorchTitan ComplexRoPE._reshape_cache API changed: expected "
            f"{expected_reshape}, got {reshape_parameters}. Update the NPU RoPE "
            "adapter before running training."
        )

    expected_apply = ("query", "key", "rope_cache", "inverse")
    apply_parameters = tuple(
        inspect.signature(rope_module.ComplexRoPE.apply_rotary_emb).parameters
    )
    if apply_parameters != expected_apply:
        raise RuntimeError(
            "TorchTitan ComplexRoPE.apply_rotary_emb API changed: expected "
            f"{expected_apply}, got {apply_parameters}. Update the NPU RoPE "
            "adapter before running training."
        )


def apply_patch() -> None:
    """Patch TorchTitan's token-first RoPE cache selection for NPU."""
    from torchtitan.models.common.rope import ComplexRoPE

    _validate_torchtitan_rope_api()
    ComplexRoPE._reshape_cache = npu_reshape_complex_cache
    ComplexRoPE.apply_rotary_emb = staticmethod(npu_apply_complex_rope)
    logger.info("Patched ComplexRoPE cache selection and rotation for NPU")
