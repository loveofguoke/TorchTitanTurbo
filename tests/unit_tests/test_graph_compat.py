# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import os

import pytest
import torch

from torchtitanturbo.patch import set_environ_variable
from torchtitanturbo.tools.graph_compat import (
    _enabled,
    _grouped_mm_weight_grad,
    _install_npugraph_capture_mode,
    _install_vetted_pointwise_autotune,
)


def test_task_queue_profile_is_not_overwritten(monkeypatch) -> None:
    monkeypatch.setenv("TORCHTITAN_TASK_QUEUE_ENABLE", "0")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2")

    set_environ_variable()

    assert os.environ["TASK_QUEUE_ENABLE"] == "0"


def test_invalid_boolean_graph_option_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TORCHTITAN_SAFE_EMPTY_GROUPED_MM", "yes")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        _enabled("TORCHTITAN_SAFE_EMPTY_GROUPED_MM")


def test_empty_npugraph_capture_mode_is_unset(monkeypatch) -> None:
    monkeypatch.setenv("TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE", "")

    _install_npugraph_capture_mode()


def test_grouped_mm_weight_grad_preserves_transposed_layout() -> None:
    B_t = torch.randn(3, 4, 5).transpose(-2, -1)
    expert_ids = torch.tensor([0, 1, 1, 2])
    row_grad_B = torch.randn(4, 5, 4)

    grad_B = _grouped_mm_weight_grad(B_t, expert_ids, row_grad_B)

    assert grad_B.stride() == B_t.stride()


@pytest.mark.parametrize(
    ("heuristic_type", "is_vetted"),
    [("POINTWISE", True), ("REDUCTION", False)],
)
def test_only_pointwise_autotune_is_marked_vetted(
    monkeypatch, heuristic_type: str, is_vetted: bool
) -> None:
    import torch_npu._inductor.runtime.triton_heuristics as runtime

    cls = runtime.NPUCachingAutotuner
    original = cls._bench_with_launch_args
    recorded = {}

    class DeviceInterface:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_raw_stream(device):
            assert device == 0
            return "stream"

    class Autotuner:
        inductor_meta = {}

        @staticmethod
        def get_device_interface():
            return DeviceInterface

        @staticmethod
        def clone_args(*args, **kwargs):
            return args, kwargs

        @staticmethod
        def reset_to_zero_args(*args, **kwargs):
            return None

    Autotuner.heuristic_type = getattr(runtime.HeuristicType, heuristic_type)

    def benchmark_gpu(kernel_call, **kwargs):
        recorded.update(kwargs)
        kernel_call()
        return 1.0

    def launcher(*args, **kwargs):
        recorded["stream"] = kwargs["stream"]

    monkeypatch.setattr(runtime.benchmarker, "benchmark_gpu", benchmark_gpu)
    try:
        _install_vetted_pointwise_autotune()
        result = cls._bench_with_launch_args(
            Autotuner(), launcher, (torch.ones(1),), (torch.ones(1),)
        )
    finally:
        cls._bench_with_launch_args = original

    assert result == 1.0
    assert recorded["is_vetted_benchmarking"] is is_vetted
    assert recorded["stream"] == "stream"
