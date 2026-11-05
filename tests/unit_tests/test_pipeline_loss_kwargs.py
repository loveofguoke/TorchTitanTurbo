# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import pytest
import torch

from torchtitanturbo.distributed.pipeline_loss_kwargs import _wrap_loss, _wrap_step


class Schedule:
    _n_microbatches = 2

    def _maybe_compute_loss(
        self, stage, output, target_mbs, mb_index, loss_kwargs=None
    ):
        value = output.square() / loss_kwargs["denominator"]
        loss_kwargs["loss_metrics"]["value"] = value.detach()
        return value

    def step(self, values, *, loss_kwargs=None, fail=False):
        # The old API passes one dict to every loss. Execution order need not
        # match microbatch order in interleaved schedules.
        outputs = [
            self._maybe_compute_loss(None, values[i], None, i, loss_kwargs)
            for i in (1, 0)
        ]
        if fail:
            raise RuntimeError("stage failed")
        return sum(outputs)


Schedule.step = _wrap_step(Schedule.step)
Schedule._maybe_compute_loss = _wrap_loss(Schedule._maybe_compute_loss)


def test_distinct_microbatch_normalizers_metrics_and_gradients():
    values = torch.tensor([2.0, 3.0], requires_grad=True)
    kwargs = [
        {"denominator": 2.0, "loss_metrics": {}},
        {"denominator": 9.0, "loss_metrics": {}},
    ]
    schedule = Schedule()
    result = schedule.step(values, loss_kwarg_mbs=kwargs)
    result.backward()
    torch.testing.assert_close(result, torch.tensor(3.0), rtol=0, atol=0)
    torch.testing.assert_close(values.grad, torch.tensor([2.0, 2 / 3]))
    assert kwargs[0]["loss_metrics"]["value"] == 2
    assert kwargs[1]["loss_metrics"]["value"] == 1
    assert not hasattr(schedule, "_turbo_loss_kwarg_mbs")


def test_failure_restores_schedule_and_rejects_incomplete_kwargs():
    schedule = Schedule()
    with pytest.raises(RuntimeError, match="stage failed"):
        schedule.step(
            torch.ones(2),
            fail=True,
            loss_kwarg_mbs=[{"denominator": 1, "loss_metrics": {}} for _ in range(2)],
        )
    assert not hasattr(schedule, "_turbo_loss_kwarg_mbs")
    with pytest.raises(ValueError, match="one dict"):
        schedule.step(torch.ones(2), loss_kwarg_mbs=[{}])
