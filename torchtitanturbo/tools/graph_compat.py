# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Opt-in Ascend graph-mode compatibility patches.

Every patch is gated by a ``TORCHTITAN_*`` environment variable. Importing
TorchTitanTurbo for ordinary eager training therefore preserves its existing
behavior.

The patches address different compiler stages and must not be conflated:

* autotuner guards change how generated Triton candidates are benchmarked;
* safe grouped MM gives Dynamo/AOT a traceable custom-op schema, fake kernel,
  and backward for empty-expert cases;
* DTensor strategy registration teaches distributed propagation about complex
  pointwise values;
* PP metadata P2P replaces object communication with tensor communication;
* NPUGraph policies decide which captured FX graphs are eligible for replay.

Each feature is default-off because a graph workaround can affect compilation
coverage or launch behavior even when its eager mathematics is equivalent.
"""

from __future__ import annotations

import inspect
import operator
import os
from typing import Callable

import torch

from torchtitan.tools.logging import logger


_COMPLEX_STRATEGY_REGISTERED = False


def _enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value not in (None, "0", "1"):
        raise ValueError(f"{name} must be 0 or 1; got {value!r}")
    return value == "1"


def _grouped_mm_weight_grad(
    B_t: torch.Tensor,
    expert_ids: torch.Tensor,
    row_grad_B: torch.Tensor,
) -> torch.Tensor:
    grad_B = torch.zeros_like(B_t)
    grad_B.index_add_(0, expert_ids, row_grad_B)
    return grad_B


def _install_vetted_pointwise_autotune() -> None:
    import torch_npu._inductor.runtime.triton_heuristics as runtime

    cls = runtime.NPUCachingAutotuner
    if getattr(
        cls._bench_with_launch_args,
        "_torchtitanturbo_vetted_pointwise_autotune",
        False,
    ):
        return

    signature = inspect.signature(cls._bench_with_launch_args)
    expected_parameters = ("self", "launcher", "launch_args", "reset_args", "kwargs")
    if tuple(signature.parameters) != expected_parameters:
        raise RuntimeError(
            "Unsupported torch_npu NPUCachingAutotuner._bench_with_launch_args "
            f"signature: {signature}"
        )

    def bench_with_launch_args(self, launcher, launch_args, reset_args, **kwargs):
        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        def kernel_call():
            cloned_args, cloned_kwargs = self.clone_args(*launch_args, **kwargs)
            self.reset_to_zero_args(*reset_args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                stream=stream,
            )

        if self.inductor_meta.get(
            "profile_bandwidth_with_do_bench_using_profiling", False
        ):
            return runtime.do_bench_using_profiling_npu(kernel_call, rep=1)

        is_vetted_pointwise = (
            self.heuristic_type == runtime.HeuristicType.POINTWISE
        )
        return runtime.benchmarker.benchmark_gpu(
            kernel_call,
            rep=1,
            device_type="npu",
            is_vetted_benchmarking=is_vetted_pointwise,
        )

    bench_with_launch_args._torchtitanturbo_vetted_pointwise_autotune = True
    cls._bench_with_launch_args = bench_with_launch_args
    logger.info("Enabled vetted NPU pointwise autotuning")


def _install_zero_numel_triton_guard() -> None:
    import torch_npu._inductor.runtime.triton_heuristics as runtime

    def has_zero_numel(self, args) -> bool:
        for name, value in zip(self.fn.arg_names, args):
            if not name.endswith("numel") or name.startswith("r"):
                continue
            try:
                if operator.index(value) == 0:
                    logger.info(
                        "Skipping zero-numel Triton kernel %s (%s=0)",
                        self.get_fn_name(),
                        name,
                    )
                    return True
            except TypeError:
                continue
        return False

    def patch(cls) -> None:
        if getattr(cls.run, "_torchtitanturbo_zero_numel_guard", False):
            return
        original_run = cls.run

        def run(self, *args, stream, benchmark_run=False, **kwargs):
            if has_zero_numel(self, args):
                return None
            return original_run(
                self,
                *args,
                stream=stream,
                benchmark_run=benchmark_run,
                **kwargs,
            )

        run._torchtitanturbo_zero_numel_guard = True
        cls.run = run

    patch(runtime.NPUCachingAutotuner)
    patch(runtime.NPUSymbolicGroupedAutotuner)
    logger.info("Enabled zero-numel Triton launch guard")


def _install_safe_empty_grouped_mm() -> None:
    """Register a compile-visible grouped MM that tolerates empty experts.

    MoE routing is data dependent: a legal step may send zero tokens to one or
    more experts. Padding each empty segment to one row gives the backend valid
    non-empty offsets; positions for real rows are remembered so padding is
    removed after compute. Fake and autograd registrations let Dynamo/AOT reason
    about the op without executing device code during tracing.
    """
    from torchtitan.models.common.moe import GroupedExperts

    namespace = "torchtitanturbo_graph"
    try:
        safe_grouped_mm = torch.ops.torchtitanturbo_graph.safe_grouped_mm.default
    except (AttributeError, RuntimeError):

        @torch.library.custom_op(
            f"{namespace}::safe_grouped_mm", mutates_args=()
        )
        def safe_grouped_mm(
            A: torch.Tensor,
            B_t: torch.Tensor,
            offs: torch.Tensor,
        ) -> torch.Tensor:
            group_sizes = torch.diff(torch.cat((offs.new_zeros((1,)), offs)))
            if bool(torch.all(group_sizes > 0).item()):
                return torch._grouped_mm(A, B_t, offs=offs)
            if A.shape[0] == 0:
                return A.new_empty((0, B_t.shape[-1]))
            padded_sizes = torch.clamp_min(group_sizes, 1)
            padded_offs = torch.cumsum(padded_sizes, 0, dtype=torch.int32)
            expert_ids = torch.repeat_interleave(
                torch.arange(B_t.shape[0], device=A.device), group_sizes
            )
            shifts = torch.cumsum((group_sizes == 0).to(torch.int64), 0)
            shifts = shifts - (group_sizes == 0).to(torch.int64)
            positions = torch.arange(A.shape[0], device=A.device)
            positions = positions + shifts.index_select(0, expert_ids)
            padded_A = A.new_zeros((padded_offs[-1].item(), A.shape[-1]))
            padded_A.index_copy_(0, positions, A)
            output = torch._grouped_mm(padded_A, B_t, offs=padded_offs)
            return output.index_select(0, positions)

        @safe_grouped_mm.register_fake
        def safe_grouped_mm_fake(
            A: torch.Tensor,
            B_t: torch.Tensor,
            offs: torch.Tensor,
        ) -> torch.Tensor:
            del offs
            return A.new_empty((A.shape[0], B_t.shape[-1]))

        @torch.library.custom_op(
            f"{namespace}::safe_grouped_mm_backward", mutates_args=()
        )
        def backward_op(
            grad_output: torch.Tensor,
            A: torch.Tensor,
            B_t: torch.Tensor,
            offs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if A.shape[0] == 0:
                return torch.zeros_like(A), torch.zeros_like(B_t)
            sizes = torch.diff(torch.cat((offs.new_zeros((1,)), offs)))
            expert_ids = torch.repeat_interleave(
                torch.arange(B_t.shape[0], device=A.device), sizes
            )
            row_weights = B_t.index_select(0, expert_ids)
            grad_A = torch.bmm(
                grad_output.unsqueeze(1), row_weights.transpose(-2, -1)
            ).squeeze(1)
            row_grad_B = A.unsqueeze(-1) * grad_output.unsqueeze(-2)
            grad_B = _grouped_mm_weight_grad(B_t, expert_ids, row_grad_B)
            return grad_A, grad_B

        @backward_op.register_fake
        def backward_op_fake(
            grad_output: torch.Tensor,
            A: torch.Tensor,
            B_t: torch.Tensor,
            offs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del grad_output, offs
            return torch.empty_like(A), torch.empty_like(B_t)

        def setup_context(ctx, inputs, output):
            del output
            ctx.save_for_backward(*inputs)

        def backward(ctx, grad_output):
            A, B_t, offs = ctx.saved_tensors
            grad_A, grad_B = backward_op(grad_output, A, B_t, offs)
            return grad_A, grad_B, None

        safe_grouped_mm.register_autograd(backward, setup_context=setup_context)

    if not getattr(GroupedExperts, "_torchtitanturbo_empty_gmm", False):

        def grouped_mm(self, *, A, B_t, offs):
            del self
            return safe_grouped_mm(A, B_t, offs)

        GroupedExperts._grouped_mm = grouped_mm
        GroupedExperts._torchtitanturbo_empty_gmm = True
    logger.info("Enabled empty-expert grouped-mm compatibility")


def _register_complex_dtensor_strategy() -> None:
    global _COMPLEX_STRATEGY_REGISTERED
    if _COMPLEX_STRATEGY_REGISTERED:
        return
    from torch.distributed.tensor._ops._pointwise_ops import (
        _register_single_dim_pointwise,
    )

    try:
        _register_single_dim_pointwise(torch.ops.aten.complex.default)
    except (AssertionError, RuntimeError) as error:
        if "already" not in str(error).lower():
            raise
    _COMPLEX_STRATEGY_REGISTERED = True
    logger.info("Registered aten.complex DTensor pointwise strategy")


def _install_pipeline_metadata_p2p() -> None:
    import torch.distributed as dist
    from torch.distributed.pipelining.stage import PipelineStage

    if getattr(PipelineStage, "_torchtitanturbo_meta_p2p", False):
        return

    def recv_meta(self, src_stage):
        objects = [None]
        dist.recv_object_list(
            objects,
            src=self._resolve_peer_global_rank(src_stage),
            group=self.group,
            device=self.device,
            use_batch=False,
        )
        return objects[0]

    def send_meta(self, meta, dst_stage):
        dist.send_object_list(
            [meta],
            dst=self._resolve_peer_global_rank(dst_stage),
            group=self.group,
            device=self.device,
            use_batch=False,
        )

    PipelineStage._recv_meta = recv_meta
    PipelineStage._send_meta = send_meta
    PipelineStage._torchtitanturbo_meta_p2p = True
    logger.info("Enabled non-batched PipelineStage metadata P2P")


def _install_npugraph_skip_policy() -> None:
    skip_all = _enabled("TORCHTITAN_NPUGRAPH_SKIP_ALL")
    unsafe_ops = {
        item.strip()
        for item in os.environ.get("TORCHTITAN_NPUGRAPH_UNSAFE_OPS", "").split(",")
        if item.strip()
    }
    if not skip_all and not unsafe_ops:
        return
    import torch_npu.utils._graph_tree as backend
    from torch._inductor.cudagraph_utils import format_default_skip_message

    if getattr(backend.check_for_skip, "_torchtitanturbo_skip_policy", False):
        return
    original = backend.check_for_skip

    def check_for_skip(graph_module, num_fixed):
        if skip_all:
            logger.info("NPUGraph replay disabled by compatibility profile")
            return format_default_skip_message(
                "NPUGraph replay disabled by compatibility profile"
            )
        targets = {
            str(node.target)
            for module in graph_module.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
            if node.op == "call_function"
        }
        matched = sorted(targets & unsafe_ops)
        if matched:
            logger.info(
                "NPUGraph replay skipped for incompatible ops: %s",
                ",".join(matched),
            )
            return format_default_skip_message(
                f"configured incompatible op ({matched[0]})"
            )
        return original(graph_module, num_fixed)

    check_for_skip._torchtitanturbo_skip_policy = True
    backend.check_for_skip = check_for_skip
    logger.info("Enabled NPUGraph compatibility skip policy")


def _install_npugraph_capture_mode() -> None:
    mode = os.environ.get("TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE")
    if mode in (None, ""):
        return
    if mode not in ("global", "thread_local", "relaxed"):
        raise ValueError(
            "TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE must be global, "
            f"thread_local, or relaxed; got {mode!r}"
        )
    import torch_npu

    if getattr(torch.npu.graph, "_torchtitanturbo_capture_mode", False):
        return
    original = torch.npu.graph

    def graph(*args, **kwargs):
        kwargs["capture_error_mode"] = mode
        return original(*args, **kwargs)

    graph._torchtitanturbo_capture_mode = True
    torch.npu.graph = graph
    torch_npu.npu.graph = graph
    logger.info("Enabled NPUGraph capture error mode %s", mode)


def apply_graph_compat_patches() -> None:
    """Apply only graph compatibility features explicitly requested by env."""

    patches: tuple[tuple[str, Callable[[], None]], ...] = (
        (
            "TORCHTITAN_VETTED_POINTWISE_AUTOTUNE",
            _install_vetted_pointwise_autotune,
        ),
        ("TORCHTITAN_SAFE_ZERO_NUMEL_TRITON", _install_zero_numel_triton_guard),
        ("TORCHTITAN_SAFE_EMPTY_GROUPED_MM", _install_safe_empty_grouped_mm),
        (
            "TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY",
            _register_complex_dtensor_strategy,
        ),
    )
    for variable, patch in patches:
        if _enabled(variable):
            patch()
    pipeline_batch = os.environ.get("TORCHTITAN_PIPELINE_META_USE_BATCH")
    if pipeline_batch not in (None, "0", "1"):
        raise ValueError(
            "TORCHTITAN_PIPELINE_META_USE_BATCH must be 0 or 1; "
            f"got {pipeline_batch!r}"
        )
    if pipeline_batch == "0":
        _install_pipeline_metadata_p2p()
    _install_npugraph_skip_policy()
    _install_npugraph_capture_mode()
