# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torchtitanturbo.distributed.pipeline_transport as transport

from torchtitanturbo.distributed.pipeline_transport import _configure_transport


@pytest.mark.parametrize(
    "device,splitting,changed",
    [("npu", False, True), ("npu", True, False), ("cuda", False, False)],
)
def test_only_unsupported_npu_split_uses_shared_communicator(
    device, splitting, changed
):
    config = SimpleNamespace(pipeline_per_edge_p2p=True)
    with patch("torchtitanturbo.distributed.pipeline_transport.dist_config", config):
        actual = _configure_transport(
            SimpleNamespace(type=device),
            SimpleNamespace(supports_splitting=splitting),
            use_torchcomms=False,
        )
    assert actual is changed
    assert config.pipeline_per_edge_p2p is (not changed)


def test_torchcomms_cannot_be_silently_forced_to_ordinary_transport():
    with pytest.raises(ValueError, match="TorchComms"):
        _configure_transport(
            SimpleNamespace(type="npu"),
            SimpleNamespace(supports_splitting=False),
            use_torchcomms=True,
        )


@pytest.mark.parametrize("degree", [1, 2])
def test_wrapper_queries_backend_only_after_original_initialization(
    monkeypatch, degree
):
    events = []
    result = object()
    config = SimpleNamespace(pipeline_per_edge_p2p=True)

    def initialize(**kwargs):
        events.append("init")
        assert kwargs["pipeline_parallel_degree"] == degree
        return result

    def device():
        assert events == ["init"]
        events.append("device")
        return SimpleNamespace(type="npu")

    def backend(device):
        assert events == ["init", "device"]
        assert device.type == "npu"
        events.append("backend")
        return SimpleNamespace(supports_splitting=False)

    monkeypatch.setattr(transport.utils, "init_distributed", initialize)
    monkeypatch.setattr(transport.utils, "get_local_device", device)
    monkeypatch.setattr(transport.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        transport.dist,
        "group",
        SimpleNamespace(WORLD=SimpleNamespace(_get_backend=backend)),
    )
    monkeypatch.setattr(
        transport.dist.distributed_c10d, "_use_torchcomms_enabled", lambda: False
    )
    monkeypatch.setattr(transport, "dist_config", config)
    transport.apply_patch()
    wrapped = transport.utils.init_distributed
    transport.apply_patch()
    assert transport.utils.init_distributed is wrapped
    assert wrapped(pipeline_parallel_degree=degree) is result
    assert events == (["init"] if degree == 1 else ["init", "device", "backend"])
    assert config.pipeline_per_edge_p2p is (degree == 1)
