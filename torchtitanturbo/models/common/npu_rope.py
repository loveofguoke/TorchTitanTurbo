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

from torchtitan.tools.logging import logger


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
    key: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    query_1TNH = query_local.float().unsqueeze(0)
    key_1TMH = key_local.float().unsqueeze(0)
    query_out_TNH = torch_npu.npu_rotary_mul(
        query_1TNH,
        cos_1T1H,
        sin_1T1H,
        rotary_mode="interleave",
    ).squeeze(0)
    key_out_TMH = torch_npu.npu_rotary_mul(
        key_1TMH,
        cos_1T1H,
        sin_1T1H,
        rotary_mode="interleave",
    ).squeeze(0)
    query_out_TNH = query_out_TNH.type_as(query_local)
    key_out_TMH = key_out_TMH.type_as(key_local)
    return (
        _wrap_dtensor_like(query_out_TNH, query),
        _wrap_dtensor_like(key_out_TMH, key),
    )


def _validate_torchtitan_rope_api() -> None:
    """Fail clearly when TorchTitan changes the RoPE patch contract."""
    from torchtitan.models.common import rope as rope_module

    expected_reshape = ("rope_cache", "query_shape", "positions")
    reshape_parameters = tuple(
        inspect.signature(rope_module._reshape_for_broadcast).parameters
    )
    if reshape_parameters != expected_reshape:
        raise RuntimeError(
            "TorchTitan _reshape_for_broadcast API changed: expected "
            f"{expected_reshape}, got {reshape_parameters}. Update the NPU RoPE "
            "adapter before running training."
        )

    expected_apply = ("query", "key", "rope_cache")
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

    from torchtitanturbo.tools.patch_utils import replace_functions

    _validate_torchtitan_rope_api()
    count = replace_functions(
        "_reshape_for_broadcast",
        npu_reshape_for_broadcast,
        package="torchtitan",
    )
    if count == 0:
        raise RuntimeError(
            "TorchTitan _reshape_for_broadcast was not found; the NPU RoPE "
            "adapter cannot be applied."
        )
    ComplexRoPE.apply_rotary_emb = staticmethod(npu_apply_complex_rope)
    logger.info(
        "Patched token-first RoPE cache selection in "
        f"{count} module references and ComplexRoPE rotation"
    )
