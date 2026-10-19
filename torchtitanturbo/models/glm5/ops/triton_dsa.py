"""Opt-in Triton sparse-attention training kernels for current GLM-5 DSA.

The public contract is the expanded training layout used by TorchTitan:
q[Q,N,H], k[K,N,H], v[K,N,V], and selected indices[Q,S].  The implementation
provides an explicit autograd dual and is never imported by Turbo's default
patch bootstrap.  CUDA and Triton-Ascend use the same mathematical kernels;
hardware acceptance remains an experiment obligation.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.glm5.dsa import Glm5FlexAttention


class TritonDsaCapabilityError(ValueError):
    """The requested tensor contract is unsupported by this kernel suite."""


@triton.jit
def _selected_scores_kernel(
    q,
    k,
    indices,
    allowed,
    scores,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    head = query_head % NUM_HEADS
    slot = (tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)).to(tl.int64)
    channel = tl.arange(0, BLOCK_H).to(tl.int64)
    slot_mask = slot < TOP_K
    index = tl.load(indices + query * TOP_K + slot, mask=slot_mask, other=-1)
    valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
    safe_index = tl.maximum(index, 0).to(tl.int64)
    valid &= tl.load(allowed + query * TOP_K + slot, mask=slot_mask, other=0) != 0
    q_value = tl.load(
        q + query_head * HEAD_DIM + channel,
        mask=channel < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    k_offset = (
        (safe_index[:, None] * NUM_HEADS + head) * HEAD_DIM
        + channel[None, :]
    )
    k_value = tl.load(
        k + k_offset,
        mask=valid[:, None] & (channel[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    score = tl.sum(k_value * q_value[None, :], axis=1) * SCALE
    tl.store(
        scores + query_head * TOP_K + slot,
        tl.where(valid, score, float("-inf")),
        mask=slot_mask,
    )


@triton.jit
def _selected_output_kernel(
    probabilities,
    v,
    indices,
    output,
    NUM_HEADS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    head = query_head % NUM_HEADS
    channel = (tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    accumulator = tl.zeros((BLOCK_V,), dtype=tl.float32)
    # TOP_K is 2048 for the released GLM model. Keep this as a loop in the
    # generated program instead of unrolling hundreds of copies into the IR.
    for slot_start in tl.range(0, TOP_K, BLOCK_S):
        slot = (slot_start + tl.arange(0, BLOCK_S)).to(tl.int64)
        slot_mask = slot < TOP_K
        index = tl.load(indices + query * TOP_K + slot, mask=slot_mask, other=-1)
        valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
        safe_index = tl.maximum(index, 0).to(tl.int64)
        probability = tl.load(
            probabilities + query_head * TOP_K + slot,
            mask=slot_mask,
            other=0.0,
        ).to(tl.float32)
        v_offset = (
            (safe_index[:, None] * NUM_HEADS + head) * VALUE_DIM
            + channel[None, :]
        )
        value = tl.load(
            v + v_offset,
            mask=valid[:, None] & (channel[None, :] < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(probability[:, None] * value, axis=0)
    tl.store(
        output + query_head * VALUE_DIM + channel,
        accumulator,
        mask=channel < VALUE_DIM,
    )


@triton.jit
def _probability_grad_kernel(
    v,
    indices,
    grad_output,
    grad_probability,
    NUM_HEADS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    head = query_head % NUM_HEADS
    slot = (tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)).to(tl.int64)
    slot_mask = slot < TOP_K
    index = tl.load(indices + query * TOP_K + slot, mask=slot_mask, other=-1)
    valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
    safe_index = tl.maximum(index, 0).to(tl.int64)
    accumulator = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for channel_start in tl.static_range(0, VALUE_DIM, BLOCK_V):
        channel = (channel_start + tl.arange(0, BLOCK_V)).to(tl.int64)
        channel_mask = channel < VALUE_DIM
        grad_value = tl.load(
            grad_output + query_head * VALUE_DIM + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        v_offset = (
            (safe_index[:, None] * NUM_HEADS + head) * VALUE_DIM
            + channel[None, :]
        )
        value = tl.load(
            v + v_offset,
            mask=valid[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(value * grad_value[None, :], axis=1)
    tl.store(
        grad_probability + query_head * TOP_K + slot,
        accumulator,
        mask=slot_mask,
    )


@triton.jit
def _softmax_grad_kernel(
    probabilities,
    grad_probability,
    grad_score,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    query_head = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, BLOCK_S).to(tl.int64)
    mask = slot < TOP_K
    probability = tl.load(
        probabilities + query_head * TOP_K + slot, mask=mask, other=0.0
    ).to(tl.float32)
    grad = tl.load(
        grad_probability + query_head * TOP_K + slot, mask=mask, other=0.0
    ).to(tl.float32)
    delta = tl.sum(probability * grad, axis=0)
    tl.store(
        grad_score + query_head * TOP_K + slot,
        probability * (grad - delta) * SCALE,
        mask=mask,
    )


@triton.jit
def _query_grad_kernel(
    grad_score,
    k,
    indices,
    grad_q,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    head = query_head % NUM_HEADS
    channel = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for slot_start in tl.range(0, TOP_K, BLOCK_S):
        slot = (slot_start + tl.arange(0, BLOCK_S)).to(tl.int64)
        slot_mask = slot < TOP_K
        index = tl.load(indices + query * TOP_K + slot, mask=slot_mask, other=-1)
        valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
        safe_index = tl.maximum(index, 0).to(tl.int64)
        ds = tl.load(
            grad_score + query_head * TOP_K + slot,
            mask=slot_mask,
            other=0.0,
        ).to(tl.float32)
        k_offset = (
            (safe_index[:, None] * NUM_HEADS + head) * HEAD_DIM
            + channel[None, :]
        )
        key = tl.load(
            k + k_offset,
            mask=valid[:, None] & (channel[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(ds[:, None] * key, axis=0)
    tl.store(
        grad_q + query_head * HEAD_DIM + channel,
        accumulator,
        mask=channel < HEAD_DIM,
    )


@triton.jit
def _key_grad_kernel(
    grad_score,
    q,
    indices,
    grad_k,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    head = query_head % NUM_HEADS
    slot = (tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)).to(tl.int64)
    channel = (tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    slot_mask = slot < TOP_K
    channel_mask = channel < HEAD_DIM
    index = tl.load(indices + query * TOP_K + slot, mask=slot_mask, other=-1)
    valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
    safe_index = tl.maximum(index, 0).to(tl.int64)
    ds = tl.load(
        grad_score + query_head * TOP_K + slot, mask=slot_mask, other=0.0
    ).to(tl.float32)
    query_value = tl.load(
        q + query_head * HEAD_DIM + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    offset = (
        (safe_index[:, None] * NUM_HEADS + head) * HEAD_DIM
        + channel[None, :]
    )
    tl.atomic_add(
        grad_k + offset,
        ds[:, None] * query_value[None, :],
        mask=valid[:, None] & channel_mask[None, :],
    )


@triton.jit
def _value_grad_kernel(
    probabilities,
    grad_output,
    indices,
    grad_v,
    NUM_HEADS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    query_head = tl.program_id(0).to(tl.int64)
    query = query_head // NUM_HEADS
    head = query_head % NUM_HEADS
    slot = (tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)).to(tl.int64)
    channel = (tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    slot_mask = slot < TOP_K
    channel_mask = channel < VALUE_DIM
    index = tl.load(indices + query * TOP_K + slot, mask=slot_mask, other=-1)
    valid = slot_mask & (index >= 0) & (index < NUM_KEYS)
    safe_index = tl.maximum(index, 0).to(tl.int64)
    probability = tl.load(
        probabilities + query_head * TOP_K + slot, mask=slot_mask, other=0.0
    ).to(tl.float32)
    grad_value = tl.load(
        grad_output + query_head * VALUE_DIM + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    offset = (
        (safe_index[:, None] * NUM_HEADS + head) * VALUE_DIM
        + channel[None, :]
    )
    tl.atomic_add(
        grad_v + offset,
        probability[:, None] * grad_value[None, :],
        mask=valid[:, None] & channel_mask[None, :],
    )


def _validate_inputs(q, k, v, mask, indices):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise TritonDsaCapabilityError("q, k, and v must have rank 3.")
    Q, N, H = q.shape
    K = k.shape[0]
    if Q <= 0 or K <= 0 or N <= 0 or H <= 0:
        raise TritonDsaCapabilityError("DSA tensor dimensions must be positive.")
    if k.shape != (K, N, H) or v.shape[:2] != (K, N):
        raise TritonDsaCapabilityError("Expected q[Q,N,H], k[K,N,H], v[K,N,V].")
    if mask.shape != (1, Q, K) or indices.ndim != 2 or indices.shape[0] != Q:
        raise TritonDsaCapabilityError("Expected mask[1,Q,K] and indices[Q,S].")
    if indices.shape[1] <= 0:
        raise TritonDsaCapabilityError("The selected-key dimension must be positive.")
    if q.device.type not in {"cuda", "npu"}:
        raise TritonDsaCapabilityError("Triton DSA training requires CUDA or NPU.")
    if any(t.device != q.device for t in (k, v, mask, indices)):
        raise TritonDsaCapabilityError("All DSA tensors must share one device.")
    if (
        q.dtype not in {torch.float16, torch.bfloat16}
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TritonDsaCapabilityError("q, k, and v must share FP16 or BF16 dtype.")
    if indices.dtype not in {torch.int32, torch.int64}:
        raise TritonDsaCapabilityError("Selected indices must be int32 or int64.")
    if H > 512 or v.shape[-1] > 512:
        raise TritonDsaCapabilityError("Current kernels require head/value dim <= 512.")


def _forward(q, k, v, mask, indices, scale):
    _validate_inputs(q, k, v, mask, indices)
    q, k, v, mask, indices = (t.contiguous() for t in (q, k, v, mask, indices))
    Q, N, H = q.shape
    K, _, V = v.shape
    S = indices.shape[1]
    safe = indices.clamp(0, K - 1).long()
    allowed = (
        (indices >= 0)
        & (indices < K)
        & (mask[0].gather(-1, safe) == 0)
    ).contiguous()
    scores = torch.empty((Q, N, S), device=q.device, dtype=torch.float32)
    block_s = 16
    _selected_scores_kernel[(Q * N, triton.cdiv(S, block_s))](
        q, k, indices, allowed, scores,
        NUM_HEADS=N, HEAD_DIM=H, NUM_KEYS=K, TOP_K=S, SCALE=scale,
        BLOCK_S=block_s, BLOCK_H=triton.next_power_of_2(H), num_warps=4,
    )
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    probabilities = probabilities.masked_fill(~allowed.unsqueeze(1), 0.0).contiguous()
    output = torch.empty((Q, N, V), device=q.device, dtype=q.dtype)
    block_v = min(64, triton.next_power_of_2(V))
    _selected_output_kernel[(Q * N, triton.cdiv(V, block_v))](
        probabilities, v, indices, output,
        NUM_HEADS=N, VALUE_DIM=V, NUM_KEYS=K, TOP_K=S,
        BLOCK_S=block_s, BLOCK_V=block_v, num_warps=4,
    )
    return output, probabilities


def _backward(q, k, v, indices, probabilities, grad_output, scale):
    q, k, v, indices, probabilities, grad_output = (
        t.contiguous() for t in (q, k, v, indices, probabilities, grad_output)
    )
    Q, N, H = q.shape
    K, _, V = v.shape
    S = indices.shape[1]
    block_s = 8
    block_v = min(64, triton.next_power_of_2(V))
    grad_probability = torch.empty_like(probabilities)
    _probability_grad_kernel[(Q * N, triton.cdiv(S, block_s))](
        v, indices, grad_output, grad_probability,
        NUM_HEADS=N, VALUE_DIM=V, NUM_KEYS=K, TOP_K=S,
        BLOCK_S=block_s, BLOCK_V=block_v, num_warps=4,
    )
    grad_score = torch.empty_like(probabilities)
    _softmax_grad_kernel[(Q * N,)](
        probabilities, grad_probability, grad_score,
        TOP_K=S, SCALE=scale, BLOCK_S=triton.next_power_of_2(S), num_warps=8,
    )
    block_h = min(64, triton.next_power_of_2(H))
    grad_q = torch.empty_like(q)
    _query_grad_kernel[(Q * N, triton.cdiv(H, block_h))](
        grad_score, k, indices, grad_q,
        NUM_HEADS=N, HEAD_DIM=H, NUM_KEYS=K, TOP_K=S,
        BLOCK_S=block_s, BLOCK_H=block_h, num_warps=4,
    )
    grad_k_fp32 = torch.zeros_like(k, dtype=torch.float32)
    _key_grad_kernel[(Q * N, triton.cdiv(S, block_s), triton.cdiv(H, block_h))](
        grad_score, q, indices, grad_k_fp32,
        NUM_HEADS=N, HEAD_DIM=H, NUM_KEYS=K, TOP_K=S,
        BLOCK_S=block_s, BLOCK_H=block_h, num_warps=4,
    )
    grad_v_fp32 = torch.zeros_like(v, dtype=torch.float32)
    _value_grad_kernel[(Q * N, triton.cdiv(S, block_s), triton.cdiv(V, block_v))](
        probabilities, grad_output, indices, grad_v_fp32,
        NUM_HEADS=N, VALUE_DIM=V, NUM_KEYS=K, TOP_K=S,
        BLOCK_S=block_s, BLOCK_V=block_v, num_warps=4,
    )
    return grad_q, grad_k_fp32.to(k.dtype), grad_v_fp32.to(v.dtype)


class _TritonDsaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, indices, scale):
        output, probabilities = _forward(q, k, v, mask, indices, scale)
        ctx.scale = scale
        ctx.save_for_backward(q, k, v, indices, probabilities)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, indices, probabilities = ctx.saved_tensors
        grad_q, grad_k, grad_v = _backward(
            q, k, v, indices, probabilities, grad_output, ctx.scale
        )
        return grad_q, grad_k, grad_v, None, None, None


# The function is local with respect to the outer DTensor placements. This is
# the same registration boundary used by TorchTitan's KDA/GDN kernel wrappers.
spmd.register_local_autograd_function(_TritonDsaFunction)


class TritonDsaAttention(Glm5FlexAttention):
    """Current GLM-5 selected attention with paired Triton forward/backward."""

    @dataclass(kw_only=True, slots=True)
    class Config(Glm5FlexAttention.Config):
        pass

    def forward(
        self,
        q_QNH,
        k_KNH,
        v_KNV,
        attention_metadata,
        topk_indices_QS,
        *,
        scale=None,
    ):
        if scale is None:
            scale = q_QNH.shape[-1] ** -0.5
        return _TritonDsaFunction.apply(
            q_QNH,
            k_KNH,
            v_KNV,
            attention_metadata,
            topk_indices_QS,
            scale,
        )


@override(
    target=Glm5FlexAttention.Config,
    description="Use paired Triton forward/backward kernels for GLM-5 DSA.",
)
def triton_dsa_training(cfg: Glm5FlexAttention.Config) -> TritonDsaAttention.Config:
    """Replace only non-CP GLM DSA attention selected by this override."""
    return derive(cfg, TritonDsaAttention.Config)


__all__ = [
    "TritonDsaAttention",
    "TritonDsaCapabilityError",
    "triton_dsa_training",
]
