# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import os

import pytest

from torchtitanturbo.patch import set_environ_variable


@pytest.mark.parametrize("requested", ["0", "1", "2"])
def test_task_queue_profile_is_respected(monkeypatch, requested: str) -> None:
    monkeypatch.setenv("TORCHTITAN_TASK_QUEUE_ENABLE", requested)
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2" if requested != "2" else "0")

    set_environ_variable()

    assert os.environ["TASK_QUEUE_ENABLE"] == requested


def test_task_queue_defaults_to_two(monkeypatch) -> None:
    monkeypatch.delenv("TORCHTITAN_TASK_QUEUE_ENABLE", raising=False)
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")

    set_environ_variable()

    assert os.environ["TASK_QUEUE_ENABLE"] == "2"


def test_invalid_task_queue_profile_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TORCHTITAN_TASK_QUEUE_ENABLE", "yes")

    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        set_environ_variable()


def test_flex_attention_dkdv_tasklist_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv(
        "TORCHTITAN_FLEX_ATTENTION_BWD_DKDV_TASKLIST", raising=False
    )
    monkeypatch.delenv(
        "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST",
        raising=False,
    )

    set_environ_variable()

    assert (
        os.environ[
            "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST"
        ]
        == "0"
    )


def test_flex_attention_dkdv_tasklist_vendor_override_is_respected(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "TORCHTITAN_FLEX_ATTENTION_BWD_DKDV_TASKLIST", raising=False
    )
    monkeypatch.setenv(
        "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST", "1"
    )

    set_environ_variable()

    assert (
        os.environ[
            "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST"
        ]
        == "1"
    )


@pytest.mark.parametrize("requested", ["0", "1"])
def test_flex_attention_dkdv_tasklist_profile_is_respected(
    monkeypatch, requested: str
) -> None:
    monkeypatch.setenv(
        "TORCHTITAN_FLEX_ATTENTION_BWD_DKDV_TASKLIST", requested
    )
    monkeypatch.setenv(
        "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST",
        "1" if requested == "0" else "0",
    )

    set_environ_variable()

    assert (
        os.environ[
            "TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST"
        ]
        == requested
    )


def test_invalid_flex_attention_dkdv_tasklist_profile_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "TORCHTITAN_FLEX_ATTENTION_BWD_DKDV_TASKLIST", "yes"
    )

    with pytest.raises(ValueError, match="must be 0 or 1"):
        set_environ_variable()
