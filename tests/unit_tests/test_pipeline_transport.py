# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
