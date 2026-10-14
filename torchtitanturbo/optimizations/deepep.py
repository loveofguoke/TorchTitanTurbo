"""Lifecycle-safe factory for the DeepEP-Ascend EPBuffer."""

from __future__ import annotations

from dataclasses import dataclass, fields
from importlib import import_module
from types import ModuleType
from typing import Any

import torch

from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    BaseEPTokenDispatcher,
    DeepEPTokenDispatcher,
    EPDispatchMetadata,
)
from torchtitan.config.transform import ModelConfigConverter
import logging

logger = logging.getLogger(__name__)


class DeepEpCapabilityError(ValueError):
    """The requested DeepEP-Ascend buffer contract is unsupported."""


@dataclass(frozen=True, slots=True)
class DeepEpBufferRequest:
    platform: str
    num_ranks: int
    num_max_tokens_per_rank: int
    hidden_dim: int
    num_topk: int
    use_fp8_dispatch: bool = False
    allow_multiple_reduction: bool = True
    do_handle_copy: bool = False


def validate_deepep_buffer_request(request: DeepEpBufferRequest) -> None:
    if request.platform != "npu":
        raise DeepEpCapabilityError("DeepEP-Ascend requires an NPU runtime.")
    if request.num_ranks <= 0 or (
        request.num_ranks != 1 and request.num_ranks % 2 != 0
    ):
        raise DeepEpCapabilityError(
            "DeepEP-Ascend requires one rank for local validation or a positive "
            "even number of ranks for multi-rank communication."
        )
    if min(
        request.num_max_tokens_per_rank,
        request.hidden_dim,
        request.num_topk,
    ) <= 0:
        raise DeepEpCapabilityError(
            "token capacity, hidden_dim, and num_topk must be positive."
        )
    if not request.allow_multiple_reduction:
        raise DeepEpCapabilityError(
            "DeepEP-Ascend requires allow_multiple_reduction=True."
        )
    if request.do_handle_copy:
        raise DeepEpCapabilityError(
            "DeepEP-Ascend requires do_handle_copy=False."
        )


class DeepEpAscendBufferProvider:
    """Build an explicit, reusable DeepEP-Ascend buffer for one EP group."""

    name = "deepep_ascend_epbuffer"

    def __init__(self, library: ModuleType | None = None) -> None:
        self._library = library

    def _load_library(self) -> ModuleType:
        if self._library is None:
            try:
                self._library = import_module("deep_ep")
            except ImportError as error:
                raise RuntimeError(
                    "DeepEP-Ascend was selected but deep_ep is not installed. "
                    "Install the pinned source revision with its DeepJIT submodule."
                ) from error
        return self._library

    def build(
        self,
        *,
        group: Any,
        num_max_tokens_per_rank: int,
        hidden_dim: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
    ) -> Any:
        request = DeepEpBufferRequest(
            platform="npu",
            num_ranks=group.size(),
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden_dim=hidden_dim,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
        )
        validate_deepep_buffer_request(request)
        return self._load_library().EPBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden_dim,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            deterministic=deterministic,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
            explicitly_destroy=True,
        )

    def dispatch_training(
        self,
        *,
        buffer: Any,
        x_TH: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        topk_weights_TK: torch.Tensor,
        num_experts: int,
        expert_alignment: int = 1,
    ) -> "DeepEpDispatchResult":
        """Dispatch with an autograd edge whose backward performs combine."""
        handle_box = _DeepEpHandleBox()
        recv_x_RH, recv_weights_RK = _DeepEpDispatch.apply(
            buffer,
            handle_box,
            x_TH,
            topk_indices_TK,
            topk_weights_TK,
            num_experts,
            expert_alignment,
        )
        if handle_box.handle is None:
            raise RuntimeError("DeepEP dispatch did not return a communication handle.")
        return DeepEpDispatchResult(
            recv_x_RH=recv_x_RH,
            recv_weights_RK=recv_weights_RK,
            num_tokens_per_local_expert_e=_tokens_per_local_expert(
                handle_box.handle
            ),
            handle=handle_box.handle,
        )

    def combine_training(
        self,
        *,
        buffer: Any,
        expert_output_RH: torch.Tensor,
        handle: Any,
    ) -> torch.Tensor:
        """Combine with an autograd edge whose backward performs dispatch."""
        return _DeepEpCombine.apply(buffer, handle, expert_output_RH)

    def provenance(self) -> dict[str, object]:
        library = self._load_library()
        return {
            "provider": self.name,
            "version": getattr(library, "__version__", "unknown"),
            "compile_capability": "unsupported",
            "graph_capture": False,
            "training_backward": True,
            "allow_hybrid_mode": False,
            "allow_multiple_reduction": True,
            "do_handle_copy": False,
            "explicit_destroy": True,
        }


@dataclass(slots=True)
class _DeepEpHandleBox:
    handle: Any | None = None


@dataclass(frozen=True, slots=True)
class DeepEpDispatchResult:
    recv_x_RH: torch.Tensor
    recv_weights_RK: torch.Tensor
    num_tokens_per_local_expert_e: torch.Tensor
    handle: Any


@dataclass(slots=True)
class _DeepEpBufferPoolEntry:
    group: Any
    buffer: Any
    ref_count: int


_buffer_pool: dict[tuple[int, int, int, int, bool, bool], _DeepEpBufferPoolEntry] = {}


def _acquire_buffer(
    provider: DeepEpAscendBufferProvider,
    *,
    group: Any,
    num_max_tokens_per_rank: int,
    hidden_dim: int,
    num_topk: int,
    use_fp8_dispatch: bool,
    deterministic: bool,
) -> tuple[Any, tuple[int, int, int, int, bool, bool]]:
    key = (
        id(group),
        num_max_tokens_per_rank,
        hidden_dim,
        num_topk,
        use_fp8_dispatch,
        deterministic,
    )
    entry = _buffer_pool.get(key)
    if entry is None:
        buffer = provider.build(
            group=group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden_dim=hidden_dim,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            deterministic=deterministic,
        )
        entry = _DeepEpBufferPoolEntry(group=group, buffer=buffer, ref_count=0)
        _buffer_pool[key] = entry
    entry.ref_count += 1
    return entry.buffer, key


def _release_buffer(key: tuple[int, int, int, int, bool, bool]) -> None:
    entry = _buffer_pool[key]
    entry.ref_count -= 1
    if entry.ref_count == 0:
        entry.buffer.destroy()
        del _buffer_pool[key]


def _wait_deferred(result: Any) -> Any:
    """Resolve the official deferred EP result on the current compute stream."""
    return result.wait() if hasattr(result, "wait") else result


def _tokens_per_local_expert(handle: Any) -> torch.Tensor:
    """Convert DeepEP's expert prefix sums to grouped-MM segment lengths."""
    prefix_sum_e = handle.psum_num_recv_tokens_per_expert
    return torch.diff(prefix_sum_e, prepend=prefix_sum_e.new_zeros(1)).to(
        torch.int32
    )


class _DeepEpDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        handle_box,
        x_TH,
        topk_indices_TK,
        topk_weights_TK,
        num_experts,
        expert_alignment,
    ):
        pending = buffer.dispatch(
            x_TH,
            topk_idx=topk_indices_TK,
            topk_weights=topk_weights_TK,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            do_expand=True,
            do_zero_padding=True,
            async_with_compute_stream=True,
            defer_epilogue=True,
        )
        recv_x_RH, _recv_indices, recv_weights_RK, handle = _wait_deferred(
            pending
        )
        if recv_weights_RK is None:
            raise RuntimeError(
                "DeepEP training dispatch must return received routing weights."
            )
        handle_box.handle = handle
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.input_dtype = x_TH.dtype
        ctx.weight_dtype = topk_weights_TK.dtype
        return recv_x_RH, recv_weights_RK

    @staticmethod
    def backward(ctx, grad_recv_x_RH, grad_recv_weights_RK):
        pending = ctx.buffer.combine(
            grad_recv_x_RH,
            handle=ctx.handle,
            topk_weights=grad_recv_weights_RK,
            async_with_compute_stream=True,
            defer_epilogue=True,
        )
        grad_x_TH, grad_weights_TK = _wait_deferred(pending)
        grad_x_TH = grad_x_TH.to(ctx.input_dtype)
        if grad_weights_TK is not None:
            grad_weights_TK = grad_weights_TK.to(ctx.weight_dtype)
        return None, None, grad_x_TH, None, grad_weights_TK, None, None


class _DeepEpCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, handle, expert_output_RH):
        pending = buffer.combine(
            expert_output_RH,
            handle=handle,
            async_with_compute_stream=True,
            defer_epilogue=True,
        )
        combined_TH, _combined_weights = _wait_deferred(pending)
        ctx.buffer = buffer
        ctx.handle = handle
        return combined_TH

    @staticmethod
    def backward(ctx, grad_output_TH):
        pending = ctx.buffer.dispatch(
            grad_output_TH,
            handle=ctx.handle,
            do_expand=True,
            do_zero_padding=True,
            async_with_compute_stream=True,
            defer_epilogue=True,
        )
        grad_expert_RH, _indices, _weights, _handle = _wait_deferred(pending)
        return None, None, grad_expert_RH


class AscendDeepEPTokenDispatcher(BaseEPTokenDispatcher):
    """TorchTitan EP dispatcher backed by the DeepEP-Ascend expanded layout."""

    provider_cls = DeepEpAscendBufferProvider

    @dataclass(kw_only=True, slots=True)
    class Config(DeepEPTokenDispatcher.Config):
        use_fp8_dispatch: bool = False
        deterministic: bool = False
        expert_alignment: int = 1

    def __init__(self, config: Config):
        super().__init__(config)
        if config.num_max_tokens_per_rank is None:
            raise ValueError(
                "DeepEP-Ascend requires num_max_tokens_per_rank for buffer "
                "initialization."
            )
        if config.hidden_dim is None:
            raise ValueError(
                "DeepEP-Ascend requires hidden_dim for buffer initialization."
            )
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.hidden_dim = config.hidden_dim
        self.use_fp8_dispatch = config.use_fp8_dispatch
        self.deterministic = config.deterministic
        self.expert_alignment = config.expert_alignment
        self.provider = self.provider_cls()
        self.buffer = None
        self._buffer_pool_key = None

    def init_buffer(self) -> None:
        assert self.ep_mesh is not None
        if self._buffer_pool_key is not None:
            self.close()
        group = self.ep_mesh.get_group()
        self.buffer, self._buffer_pool_key = _acquire_buffer(
            self.provider,
            group=group,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            hidden_dim=self.hidden_dim,
            num_topk=self.top_k,
            use_fp8_dispatch=self.use_fp8_dispatch,
            deterministic=self.deterministic,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, EPDispatchMetadata]:
        del num_local_tokens_per_expert_E
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "DeepEP-Ascend does not currently support torch.compile."
            )
        if self.buffer is None:
            raise RuntimeError(
                "DeepEP-Ascend buffer is not initialized; wire the EP mesh first."
            )
        result = self.provider.dispatch_training(
            buffer=self.buffer,
            x_TH=x_TD,
            topk_indices_TK=topk_expert_ids_TK,
            topk_weights_TK=topk_scores_TK,
            num_experts=self.num_experts,
            expert_alignment=self.expert_alignment,
        )
        metadata = EPDispatchMetadata(state=result)
        return (
            result.recv_x_RH,
            result.num_tokens_per_local_expert_e,
            metadata,
        )

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: EPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        del x_TD
        if self.buffer is None:
            raise RuntimeError(
                "DeepEP-Ascend buffer is not initialized; wire the EP mesh first."
            )
        state = metadata.state
        assert isinstance(state, DeepEpDispatchResult)
        per_row_weight_R1 = state.recv_weights_RK.reshape(
            routed_output_RD.shape[0], -1
        ).sum(dim=-1, keepdim=True)
        weighted_output_RD = routed_output_RD * per_row_weight_R1.to(
            routed_output_RD.dtype
        )
        return self.provider.combine_training(
            buffer=self.buffer,
            expert_output_RH=weighted_output_RD,
            handle=state.handle,
        )

    def close(self) -> None:
        """Destroy the communication buffer before its process group."""
        if self.buffer is not None:
            assert self._buffer_pool_key is not None
            _release_buffer(self._buffer_pool_key)
            self.buffer = None
            self._buffer_pool_key = None


class AscendDeepEPTokenDispatcherConverter(ModelConfigConverter):
    """Replace an explicitly selected GPU DeepEP dispatcher with Ascend DeepEP."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        use_fp8_dispatch: bool = False
        deterministic: bool = False
        expert_alignment: int = 1

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config):
        count = 0
        for _fqn, routed_config, _parent, _attr in model_config.traverse(
            RoutedExperts.Config
        ):
            dispatcher = routed_config.token_dispatcher
            if not isinstance(dispatcher, DeepEPTokenDispatcher.Config):
                continue
            inherited = {
                field.name: getattr(dispatcher, field.name)
                for field in fields(dispatcher)
                if field.init
                and field.name
                in {
                    "num_experts",
                    "top_k",
                    "num_max_tokens_per_rank",
                    "hidden_dim",
                }
            }
            routed_config.token_dispatcher = AscendDeepEPTokenDispatcher.Config(
                **inherited,
                use_fp8_dispatch=self.config.use_fp8_dispatch,
                deterministic=self.config.deterministic,
                expert_alignment=self.config.expert_alignment,
            )
            count += 1

        logger.info(
            "Converted %d DeepEP dispatcher configs to DeepEP-Ascend", count
        )
        return model_config
