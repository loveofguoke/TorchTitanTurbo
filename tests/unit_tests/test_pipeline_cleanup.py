# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from torchtitanturbo.distributed.pipeline_cleanup import (
    _after_stage_cleanup,
    apply_patch,
)


@pytest.mark.parametrize("failure", [None, "before", "during"])
def test_callback_runs_only_after_original_cleanup_and_restores_stage(failure):
    events = []
    stage = SimpleNamespace(
        _post_metadata_inference_cleanup=lambda: events.append("stage")
    )
    original = stage._post_metadata_inference_cleanup

    def run():
        events.append("run")
        if failure == "before":
            raise RuntimeError("P2P failed before metadata backup")
        try:
            if failure == "during":
                raise RuntimeError("metadata failed")
            return 7
        finally:
            stage._post_metadata_inference_cleanup()

    callbacks = OrderedDict([(0, lambda: events.append("callback"))])
    if failure:
        with pytest.raises(RuntimeError):
            _after_stage_cleanup(stage, callbacks, run)
    else:
        assert _after_stage_cleanup(stage, callbacks, run) == 7
    assert stage._post_metadata_inference_cleanup is original
    assert events == (["run"] if failure == "before" else ["run", "stage", "callback"])


def test_existing_public_api_is_untouched():
    target = SimpleNamespace(register_post_metadata_inference_cleanup=object())
    with patch(
        "torchtitanturbo.distributed.pipeline_cleanup.schedules._PipelineSchedule",
        target,
    ):
        assert apply_patch() is False


def test_registration_returns_removable_handle_and_preserves_order():
    class Schedule:
        def _initialize_pp_stages(
            self,
            stages,
            args,
            kwargs,
            target,
            fwd_initialized,
            bwd_initialized,
            loss_kwargs=None,
        ):
            return fwd_initialized, bwd_initialized

    with patch(
        "torchtitanturbo.distributed.pipeline_cleanup.schedules._PipelineSchedule",
        Schedule,
    ):
        assert apply_patch() is True
        schedule = Schedule()
        first = lambda: None
        second = lambda: None
        handle = schedule.register_post_metadata_inference_cleanup(first)
        schedule.register_post_metadata_inference_cleanup(second)
        assert list(schedule._post_metadata_inference_cleanups.values()) == [
            first,
            second,
        ]
        handle.remove()
        assert list(schedule._post_metadata_inference_cleanups.values()) == [second]
