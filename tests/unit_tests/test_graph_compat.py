# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import os
from contextlib import contextmanager

import pytest
import torch

from torchtitanturbo.patch import set_environ_variable
from torchtitanturbo.tools.graph_compat import (
    _enabled,
    _grouped_mm_weight_grad,
    _install_first_all_reduce_host_barrier,
    _install_npugraph_capture_mode,
    _install_pipeline_metadata_force_eager,
    _install_pipeline_metadata_store_transport,
    _install_vetted_pointwise_autotune,
    _pipeline_stage_precompile_shapes,
    _pipeline_stage_precompile_dtype,
    _precompile_pipeline_stage,
    _wait_for_pipeline_stage_precompile,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ()),
        ("", ()),
        ("128,256", ((128, 256),)),
        ("128,256;1024,256", ((128, 256), (1024, 256))),
    ],
)
def test_pipeline_stage_precompile_shapes(monkeypatch, value, expected) -> None:
    if value is None:
        monkeypatch.delenv(
            "TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE", raising=False
        )
    else:
        monkeypatch.setenv("TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE", value)

    assert _pipeline_stage_precompile_shapes() == expected


@pytest.mark.parametrize("value", ["128", "128,0", "x,256", "1,2,3"])
def test_invalid_pipeline_stage_precompile_shape_is_rejected(
    monkeypatch, value
) -> None:
    monkeypatch.setenv("TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_SHAPE", value)

    with pytest.raises(ValueError, match="PIPELINE_STAGE_PRECOMPILE_SHAPE"):
        _pipeline_stage_precompile_shapes()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("float32", torch.float32), ("bfloat16", torch.bfloat16)],
)
def test_pipeline_stage_precompile_dtype(monkeypatch, value, expected) -> None:
    monkeypatch.setenv("TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_DTYPE", value)

    assert _pipeline_stage_precompile_dtype() is expected


@pytest.mark.parametrize("value", [None, "", "int64", "not_a_dtype"])
def test_invalid_pipeline_stage_precompile_dtype_is_rejected(
    monkeypatch, value
) -> None:
    if value is None:
        monkeypatch.delenv(
            "TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_DTYPE", raising=False
        )
    else:
        monkeypatch.setenv("TORCHTITAN_PIPELINE_STAGE_PRECOMPILE_DTYPE", value)

    with pytest.raises(ValueError, match="PIPELINE_STAGE_PRECOMPILE_DTYPE"):
        _pipeline_stage_precompile_dtype()


def test_pipeline_stage_precompile_runs_forward_backward_once() -> None:
    module = torch.nn.Linear(4, 4)

    class Stage:
        is_first = False
        submod = module
        device = torch.device("cpu")
        stage_index = 3

    stage = Stage()
    original_weight = module.weight.detach().clone()
    original_grad = torch.full_like(module.weight, 3.0)
    module.weight.grad = original_grad.clone()

    _precompile_pipeline_stage(stage, None, None, ((2, 4),), torch.float32)
    _precompile_pipeline_stage(stage, None, None, ((2, 4),), torch.float32)

    assert stage._torchtitanturbo_stage_precompiled is True
    torch.testing.assert_close(module.weight.grad, original_grad)
    assert module.bias.grad is None
    torch.testing.assert_close(module.weight, original_weight)


def test_pipeline_stage_precompile_uses_host_store_barrier(monkeypatch) -> None:
    recorded = {}

    class Store:
        def set(self, key, value):
            recorded["set"] = (key, value)

        def wait(self, keys, timeout):
            recorded["wait"] = (keys, timeout)

    class Stage:
        stage_index = 2
        num_stages = 4
        group_size = 4

    monkeypatch.setattr(
        "torch.distributed.distributed_c10d._get_default_store", lambda: Store()
    )

    _wait_for_pipeline_stage_precompile(Stage())

    assert recorded["set"] == (
        "torchtitanturbo/pipeline_stage_precompile/2",
        b"ready",
    )
    assert recorded["wait"][0] == [
        f"torchtitanturbo/pipeline_stage_precompile/{index}" for index in range(4)
    ]


def test_pipeline_metadata_store_transport_round_trip(monkeypatch) -> None:
    import torch.distributed.pipelining.stage as stage_module

    storage = {}

    class Store:
        def set(self, key, value):
            storage[key] = value

        def get(self, key):
            return storage[key]

    class FakePipelineStage:
        stage_index = 2

        @staticmethod
        def _resolve_peer_global_rank(stage_index):
            return stage_index + 10

    monkeypatch.setattr(stage_module, "PipelineStage", FakePipelineStage)
    monkeypatch.setattr(
        "torch.distributed.distributed_c10d._get_default_store", lambda: Store()
    )

    _install_pipeline_metadata_store_transport()
    stage = FakePipelineStage()
    payload = {"shape": (128, 256), "dtype": "float32"}
    stage._send_meta(payload, 3)

    stage.stage_index = 3
    assert stage._recv_meta(2) == payload


def test_pipeline_metadata_force_eager_preserves_original_protocol(
    monkeypatch,
) -> None:
    import torch.distributed.pipelining.stage as stage_module

    state = {"active": False, "calls": []}

    @contextmanager
    def set_stance(value):
        assert value == "force_eager"
        state["active"] = True
        try:
            yield
        finally:
            state["active"] = False

    class FakePipelineStage:
        def _forward_metadata_inference(self, args, kwargs=None, has_backward=False):
            state["calls"].append(("forward", state["active"]))
            return args, kwargs, has_backward

        def _backward_metadata_inference(self, **kwargs):
            state["calls"].append(("backward", state["active"]))
            return kwargs

    monkeypatch.setattr(stage_module, "PipelineStage", FakePipelineStage)
    monkeypatch.setattr(torch.compiler, "set_stance", set_stance)

    _install_pipeline_metadata_force_eager()
    stage = FakePipelineStage()
    assert stage._forward_metadata_inference((torch.ones(1),), has_backward=True)[2]
    stage._backward_metadata_inference(target=torch.ones(1))

    assert state["calls"] == [("forward", True), ("backward", True)]


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
def test_all_reduce_waits_on_host_store_for_each_sequence(monkeypatch) -> None:
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    recorded = {"calls": 0, "set": [], "wait": []}

    class Store:
        def set(self, key, value):
            recorded["set"].append((key, value))

        def wait(self, keys, timeout):
            recorded["wait"].append(keys)

    def original(*args, **kwargs):
        recorded["calls"] += 1
        return "work"

    monkeypatch.setattr(dist, "all_reduce", original)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(c10d, "_get_default_store", lambda: Store())

    _install_first_all_reduce_host_barrier()

    assert dist.all_reduce(torch.ones(1)) == "work"
    assert dist.all_reduce(torch.ones(1)) == "work"
    assert recorded["set"] == [
        ("torchtitanturbo/all_reduce_ready/0-1/0/1", b"ready"),
        ("torchtitanturbo/all_reduce_ready/0-1/1/1", b"ready"),
    ]
    assert recorded["wait"] == [
        [
            "torchtitanturbo/all_reduce_ready/0-1/0/0",
            "torchtitanturbo/all_reduce_ready/0-1/0/1",
        ],
        [
            "torchtitanturbo/all_reduce_ready/0-1/1/0",
            "torchtitanturbo/all_reduce_ready/0-1/1/1",
        ],
    ]
    assert recorded["calls"] == 2
