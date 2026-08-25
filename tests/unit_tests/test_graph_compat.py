# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import os

import pytest

from torchtitanturbo.patch import set_environ_variable
from torchtitanturbo.tools.graph_compat import _enabled


def test_task_queue_profile_is_not_overwritten(monkeypatch) -> None:
    monkeypatch.setenv("TORCHTITAN_TASK_QUEUE_ENABLE", "0")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2")

    set_environ_variable()

    assert os.environ["TASK_QUEUE_ENABLE"] == "0"


def test_invalid_boolean_graph_option_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TORCHTITAN_SAFE_EMPTY_GROUPED_MM", "yes")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        _enabled("TORCHTITAN_SAFE_EMPTY_GROUPED_MM")
