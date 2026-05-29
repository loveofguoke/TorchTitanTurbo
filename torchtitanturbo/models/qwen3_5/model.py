# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Qwen3.5 text components are adapted from the official Hugging Face
# Transformers Qwen3.5 implementation:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py

import dataclasses
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce
from torch.distributed.tensor import DTensor

from torchtitan.models.common.attention import (
    AttentionMasksType,
    ScaledDotProductAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import _rotate_half
from torchtitan.models.utils import (
    get_dense_model_nparams_and_flops,
    get_moe_model_nparams_and_flops,
)
from torchtitan.protocols.module import Module


class _ContextParallelAllGatherSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group, world_size: int, rank: int):
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        local_tensor = tensor.contiguous()
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor, group=group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.chunk(ctx.world_size, dim=1)[ctx.rank].contiguous()
        dist.all_reduce(grad_input, group=ctx.group)
        return grad_input, None, None, None


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Align with the official FLA l2norm fallback used by Qwen3.5."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Torch fallback for Qwen3.5 Gated DeltaNet.

    This mirrors the official Transformers fallback and intentionally avoids
    optional CUDA-only FLA kernels so the debug model can run on Ascend NPU.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (query.shape[-1] ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def apply_partial_rotary_emb_cos_sin(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3.5 partial RoPE to only the leading rotary dimensions."""
    bsz, seqlen, _, _ = xq.shape
    rotary_dim = rope_cache.shape[-1] // 2
    if positions is None:
        rope_cache = rope_cache[:seqlen].view(1, seqlen, 1, rotary_dim * 2)
    elif positions.size(0) == 1:
        rope_cache = rope_cache[positions.squeeze(0)].view(
            1, seqlen, 1, rotary_dim * 2
        )
    else:
        rope_cache = rope_cache[positions].view(bsz, seqlen, 1, rotary_dim * 2)

    cos = rope_cache[..., :rotary_dim].to(device=xq.device, dtype=xq.dtype)
    sin = rope_cache[..., rotary_dim:].to(device=xq.device, dtype=xq.dtype)
    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]
    xq_out = (xq_rot * cos) + (_rotate_half(xq_rot) * sin)
    xk_out = (xk_rot * cos) + (_rotate_half(xk_rot) * sin)
    return (
        torch.cat([xq_out.type_as(xq), xq_pass], dim=-1),
        torch.cat([xk_out.type_as(xk), xk_pass], dim=-1),
    )


class Qwen35RMSNorm(RMSNorm):
    """Qwen3.5 RMSNorm uses a zero offset and multiplies by ``1 + weight``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        output = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        weight = (1.0 + self.weight).to(dtype=input_dtype)
        return output.to(dtype=input_dtype) * weight


class Qwen35RMSNormGated(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_float = hidden_states.to(torch.float32)
        variance = hidden_states_float.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states_float * torch.rsqrt(
            variance + self.variance_epsilon
        )
        gate = F.silu(gate.to(torch.float32)).to(input_dtype)
        return self.weight.to(input_dtype) * hidden_states.to(input_dtype) * gate


class Qwen35GatedDeltaNet(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_k_heads: int
        num_v_heads: int
        head_k_dim: int
        head_v_dim: int
        conv_kernel_size: int = 4
        rms_norm_eps: float = 1e-6
        in_proj_qkv: Linear.Config
        in_proj_z: Linear.Config
        in_proj_b: Linear.Config
        in_proj_a: Linear.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.conv_dim = self.key_dim * 2 + self.value_dim

        conv1d_cls = Module.from_nn_module(nn.Conv1d)
        self.conv1d = conv1d_cls(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen35RMSNormGated.Config(
            hidden_size=self.head_v_dim,
            eps=config.rms_norm_eps,
            param_init={"weight": nn.init.ones_},
        ).build()
        self.out_proj = config.out_proj.build()
        self.in_proj_qkv = config.in_proj_qkv.build()
        self.in_proj_z = config.in_proj_z.build()
        self.in_proj_b = config.in_proj_b.build()
        self.in_proj_a = config.in_proj_a.build()
        self._cp_mesh = None

    def _cp_gather_seq(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._cp_mesh is None:
            return tensor
        return _ContextParallelAllGatherSeq.apply(
            tensor,
            self._cp_mesh.get_group(),
            self._cp_mesh.size(),
            self._cp_mesh.get_local_rank(),
        )

    def _init_self_parameters(self) -> None:
        with torch.no_grad():
            self.dt_bias.fill_(1.0)
            self.A_log.uniform_(0, 16).log_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_seq_len = hidden_states.shape[1]
        local_positions = positions
        global_positions = None
        if self._cp_mesh is not None:
            hidden_states = self._cp_gather_seq(hidden_states)
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = self._cp_gather_seq(attention_mask)
            if positions is not None:
                global_positions = self._cp_gather_seq(positions)
                sort_idx = torch.argsort(global_positions, dim=1)
                hidden_states = torch.gather(
                    hidden_states,
                    1,
                    sort_idx.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
                )
                global_positions = torch.gather(global_positions, 1, sort_idx)
                if attention_mask is not None and attention_mask.ndim == 2:
                    attention_mask = torch.gather(attention_mask, 1, sort_idx)

        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(
                dtype=hidden_states.dtype
            )

        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )

        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.to(dtype=hidden_states.dtype).exp() * F.softplus(
            self.in_proj_a(hidden_states) + self.dt_bias.to(dtype=hidden_states.dtype)
        )
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        core_attn_out, _ = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)

        if self._cp_mesh is not None:
            if local_positions is not None and global_positions is not None:
                gather_idx = (
                    global_positions[:, None, :] == local_positions[:, :, None]
                ).to(torch.int64).argmax(dim=-1)
                output = torch.gather(
                    output,
                    1,
                    gather_idx.unsqueeze(-1).expand(-1, -1, output.shape[-1]),
                ).contiguous()
            else:
                rank = self._cp_mesh.get_local_rank()
                start = rank * local_seq_len
                output = output[:, start : start + local_seq_len].contiguous()

        return output


class Qwen35Attention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_heads: int
        n_kv_heads: int
        head_dim: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        o_proj: Linear.Config
        q_norm: Qwen35RMSNorm.Config
        k_norm: Qwen35RMSNorm.Config
        mask_type: str = "causal"
        inner_attention: ScaledDotProductAttention.Config = field(
            default_factory=ScaledDotProductAttention.Config
        )

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.o_proj = config.o_proj.build()
        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()
        self.inner_attention = config.inner_attention.build()
        self._cp_mesh = None

    def _cp_gather_seq(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._cp_mesh is None:
            return tensor
        return _ContextParallelAllGatherSeq.apply(
            tensor,
            self._cp_mesh.get_group(),
            self._cp_mesh.size(),
            self._cp_mesh.get_local_rank(),
        )

    def _cp_gather_positions(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._cp_mesh is None:
            return positions
        local_positions = positions.contiguous()
        gathered = [
            torch.empty_like(local_positions) for _ in range(self._cp_mesh.size())
        ]
        dist.all_gather(gathered, local_positions, group=self._cp_mesh.get_group())
        return torch.cat(gathered, dim=1)

    def _context_parallel_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run local queries against all-gathered KV for qwen3.5 CP on NPU."""
        tp_mesh = None
        tp_placements = None
        if isinstance(query_states, DTensor):
            assert isinstance(key_states, DTensor) and isinstance(value_states, DTensor)
            tp_mesh = query_states.device_mesh
            tp_placements = query_states.placements
            query_states = query_states.to_local()
            key_states = key_states.to_local()
            value_states = value_states.to_local()
        if isinstance(positions, DTensor):
            positions = positions.to_local()

        batch_size, local_seq_len, _, _ = query_states.shape
        key_states = self._cp_gather_seq(key_states)
        value_states = self._cp_gather_seq(value_states)

        if positions is None:
            rank = self._cp_mesh.get_local_rank() if self._cp_mesh is not None else 0
            positions = torch.arange(
                rank * local_seq_len,
                (rank + 1) * local_seq_len,
                dtype=torch.int32,
                device=query_states.device,
            ).expand(batch_size, -1)
        all_positions = self._cp_gather_positions(positions)
        sort_idx = torch.argsort(all_positions, dim=1)
        key_states = torch.gather(
            key_states,
            1,
            sort_idx[:, :, None, None].expand(
                -1, -1, key_states.shape[2], key_states.shape[3]
            ),
        )
        value_states = torch.gather(
            value_states,
            1,
            sort_idx[:, :, None, None].expand(
                -1, -1, value_states.shape[2], value_states.shape[3]
            ),
        )
        all_positions = torch.gather(all_positions, 1, sort_idx)

        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2)

        if self.num_key_value_groups > 1:
            local_n_heads = query_states.shape[1]
            local_n_kv_heads = key_states.shape[1]
            _, global_seq_len, _, _ = value_states.transpose(1, 2).shape
            key_states = key_states[:, :, None, :, :].expand(
                batch_size,
                local_n_kv_heads,
                self.num_key_value_groups,
                global_seq_len,
                self.head_dim,
            )
            value_states = value_states[:, :, None, :, :].expand(
                batch_size,
                local_n_kv_heads,
                self.num_key_value_groups,
                global_seq_len,
                self.head_dim,
            )
            key_states = key_states.reshape(
                batch_size, local_n_heads, global_seq_len, self.head_dim
            )
            value_states = value_states.reshape(
                batch_size, local_n_heads, global_seq_len, self.head_dim
            )

        causal_mask = all_positions[:, None, None, :] <= positions[:, None, :, None]
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            scale=self.scaling,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        if tp_mesh is not None:
            attn_output = DTensor.from_local(
                attn_output,
                device_mesh=tp_mesh,
                placements=tp_placements,
                run_check=False,
            )
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, -1, self.head_dim
        )

        query_states, key_states = apply_partial_rotary_emb_cos_sin(
            query_states,
            key_states,
            freqs_cis,
            positions,
        )

        if self._cp_mesh is not None:
            attn_output = self._context_parallel_attention(
                query_states,
                key_states,
                value_states,
                positions,
            )
        else:
            attn_output = self.inner_attention(
                query_states,
                key_states,
                value_states,
                attention_masks=attention_masks,
                scale=self.scaling,
                enable_gqa=self.num_key_value_groups > 1,
            ).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen35MoeExperts(Module):
    """Qwen3.5 MoE experts with official fused gate/up projection layout."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int
        num_experts: int

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self._ep_mesh = None
        self._local_expert_start = 0
        self._local_expert_count = config.num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * config.intermediate_size,
                config.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                config.intermediate_size,
            )
        )

    def _local_expert_bounds(self) -> tuple[int, int]:
        if isinstance(self.gate_up_proj, DTensor):
            return self._local_expert_start, self._local_expert_count
        return 0, self.num_experts

    def _maybe_reduce_ep_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._ep_mesh is None:
            return hidden_states
        return differentiable_all_reduce(
            hidden_states,
            group=self._ep_mesh.get_group(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        gate_up_proj = (
            self.gate_up_proj.to_local()
            if isinstance(self.gate_up_proj, DTensor)
            else self.gate_up_proj
        )
        down_proj = (
            self.down_proj.to_local()
            if isinstance(self.down_proj, DTensor)
            else self.down_proj
        )
        expert_start, local_expert_count = self._local_expert_bounds()

        if torch.compiler.is_compiling():
            if (
                hidden_states.device.type == "npu"
                or torch.are_deterministic_algorithms_enabled()
            ):
                final_hidden_states = torch.zeros_like(hidden_states)
                for local_expert_idx in range(local_expert_count):
                    expert_idx = expert_start + local_expert_idx
                    expert_weights = (
                        (top_k_index == expert_idx).to(top_k_weights.dtype)
                        * top_k_weights
                    ).sum(dim=1)
                    gate, up = F.linear(
                        hidden_states,
                        gate_up_proj[local_expert_idx],
                    ).chunk(2, dim=-1)
                    current_hidden_states = F.silu(gate) * up
                    current_hidden_states = F.linear(
                        current_hidden_states,
                        down_proj[local_expert_idx],
                    )
                    final_hidden_states = final_hidden_states + (
                        current_hidden_states * expert_weights[:, None]
                    )
                return self._maybe_reduce_ep_output(final_hidden_states)

            gate_up = torch.einsum(
                "th,eih->tei",
                hidden_states,
                gate_up_proj,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expert_hidden = F.silu(gate) * up
            expert_output = torch.einsum(
                "tei,ehi->teh",
                expert_hidden,
                down_proj,
            )
            local_top_k_index = top_k_index - expert_start
            local_expert_mask = (
                (local_top_k_index >= 0) & (local_top_k_index < local_expert_count)
            )
            expert_weights = F.one_hot(
                local_top_k_index.clamp(min=0, max=local_expert_count - 1),
                num_classes=local_expert_count,
            ).to(dtype=top_k_weights.dtype)
            expert_weights = (
                expert_weights
                * local_expert_mask.unsqueeze(-1).to(top_k_weights.dtype)
                * top_k_weights.unsqueeze(-1)
            ).sum(dim=1)
            final_hidden_states = (expert_output * expert_weights.unsqueeze(-1)).sum(
                dim=1
            )
            return self._maybe_reduce_ep_output(final_hidden_states)

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            expert_hit = expert_hit[
                (expert_hit[:, 0] >= expert_start)
                & (expert_hit[:, 0] < expert_start + local_expert_count)
            ]

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            local_expert_idx = expert_idx - expert_start
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(
                current_state, gate_up_proj[local_expert_idx]
            ).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states,
                down_proj[local_expert_idx],
            )
            current_hidden_states = (
                current_hidden_states
                * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )
        return self._maybe_reduce_ep_output(final_hidden_states)


class Qwen35MoeTopKRouter(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_experts: int
        top_k: int

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        router_top_value, router_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        router_top_value = router_top_value / router_top_value.sum(
            dim=-1, keepdim=True
        )
        router_scores = router_top_value.to(router_logits.dtype)
        return router_logits, router_scores, router_indices


class Qwen35SparseMoeBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        experts: Qwen35MoeExperts.Config
        gate: Qwen35MoeTopKRouter.Config
        shared_expert: FeedForward.Config
        shared_expert_gate: Linear.Config

        @property
        def num_experts(self) -> int:
            return self.experts.num_experts

        @property
        def router(self) -> Qwen35MoeTopKRouter.Config:
            return self.gate

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = config.gate.build()
        self.experts = config.experts.build()
        self.shared_expert = config.shared_expert.build()
        self.shared_expert_gate = config.shared_expert_gate.build()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(
            hidden_states_reshaped,
            selected_experts,
            routing_weights,
        )
        shared_expert_output = (
            torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )
        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)


class Qwen35TransformerBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_type: str
        linear_attn: Qwen35GatedDeltaNet.Config | None
        attention: Qwen35Attention.Config | None
        attention_norm: Qwen35RMSNorm.Config
        ffn_norm: Qwen35RMSNorm.Config
        feed_forward: FeedForward.Config | None = None
        moe: Qwen35SparseMoeBlock.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.layer_type = config.layer_type
        self.moe_enabled = config.moe is not None
        if self.layer_type == "linear_attention":
            assert config.linear_attn is not None
            self.linear_attn = config.linear_attn.build()
        elif self.layer_type == "full_attention":
            assert config.attention is not None
            self.attention = config.attention.build()
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        h = self.attention_norm(x)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(h, positions=positions)
        else:
            h = self.attention(h, freqs_cis, attention_masks, positions)
        x = residual + h
        h = self.ffn_norm(x)
        h = self.moe(h) if self.moe_enabled else self.feed_forward(h)
        return x + h


class Qwen35Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        vocab_size: int = 248320
        enable_weight_tying: bool = False

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            seq_len = trainer_config.training.seq_len
            parallelism = trainer_config.parallelism
            if seq_len > self.rope.max_seq_len:
                self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)
            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                full_attention = next(
                    (
                        layer.attention
                        for layer in self.layers
                        if getattr(layer, "attention", None) is not None
                    ),
                    None,
                )
                if full_attention is not None:
                    if full_attention.n_heads % tp != 0:
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide "
                            f"n_heads ({full_attention.n_heads})."
                        )
                    if full_attention.n_kv_heads % tp != 0:
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide "
                            f"n_kv_heads ({full_attention.n_kv_heads})."
                        )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            full_attn = next(
                (
                    layer.attention
                    for layer in self.layers
                    if getattr(layer, "attention", None) is not None
                ),
                None,
            )
            n_heads = full_attn.n_heads if full_attn is not None else 1
            head_dim = full_attn.head_dim if full_attn is not None else self.dim
            moe_config = next(
                (
                    layer.moe
                    for layer in self.layers
                    if getattr(layer, "moe", None) is not None
                ),
                None,
            )
            if moe_config is not None:
                return get_moe_model_nparams_and_flops(
                    self,
                    model,
                    n_heads,
                    2 * head_dim,
                    seq_len,
                )
            return get_dense_model_nparams_and_flops(
                model,
                len(self.layers),
                n_heads,
                2 * head_dim,
                seq_len,
                self.enable_weight_tying,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def _build_rope_cache(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        # Decoder/RoPE already owns a persistent cos/sin cache. Reusing it keeps
        # arange/outer/cos/sin out of torch.compile graphs, which is important
        # for torch_npu Inductor/DVM fallback memory behavior.
        rope_cache = self.freqs_cis
        if rope_cache.device != hidden_states.device:
            rope_cache = rope_cache.to(device=hidden_states.device)
        return rope_cache

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        freqs_cis = self._build_rope_cache(h, h.shape[1], positions)

        for layer in self.layers.values():
            h = layer(h, freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.enable_weight_tying:
            assert self.tok_embeddings is not None and self.lm_head is not None
            self.tok_embeddings.weight = self.lm_head.weight
        super().init_states(buffer_device=buffer_device)
