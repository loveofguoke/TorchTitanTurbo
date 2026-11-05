# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from torchtitanturbo.distributed import pipeline_cleanup

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


@pytest.mark.parametrize(
    "mode",
    ["dynamic", "static", "initialized", "p2p-error", "metadata-error", "removed"],
)
def test_initialization_wrapper_preserves_cleanup_boundary(mode):
    events = []

    class Stage:
        def __init__(self, name, dynamic):
            self.name = name
            self.device = SimpleNamespace(type="npu")
            self._inference_mode = (
                pipeline_cleanup.InferenceMode.DYNAMIC if dynamic else object()
            )

        def _post_metadata_inference_cleanup(self):
            events.append(self.name)

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
            if fwd_initialized and bwd_initialized:
                return True, True
            if mode == "p2p-error":
                raise RuntimeError("P2P error")
            try:
                if mode == "metadata-error":
                    raise RuntimeError("metadata error")
            finally:
                for stage in stages:
                    stage._post_metadata_inference_cleanup()
            return True, True

    stages = [Stage("first", mode != "static"), Stage("last", False)]
    with patch.object(pipeline_cleanup, "PipelineStage", Stage), patch.object(
        pipeline_cleanup.schedules, "_PipelineSchedule", Schedule
    ):
        assert apply_patch()
        schedule = Schedule()
        handle = schedule.register_post_metadata_inference_cleanup(
            lambda: events.append("callback")
        )
        if mode == "removed":
            handle.remove()
        initialized = mode == "initialized"
        if mode.endswith("error"):
            with pytest.raises(RuntimeError):
                schedule._initialize_pp_stages(
                    stages, (), {}, None, initialized, initialized
                )
        else:
            assert schedule._initialize_pp_stages(
                stages, (), {}, None, initialized, initialized
            ) == (True, True)
    if mode in ("initialized", "p2p-error"):
        assert events == []
    else:
        assert events == ["first", "last"] + (
            [] if mode in ("static", "removed") else ["callback"]
        )
    assert all(
        "_post_metadata_inference_cleanup" not in vars(stage) for stage in stages
    )
