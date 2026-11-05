# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Bridge per-microbatch loss kwargs to older pipeline schedule wheels."""

import inspect
from functools import wraps

from torch.distributed.pipelining import schedules


def _wrap_step(original):
    @wraps(original)
    def step(self, *args, loss_kwarg_mbs=None, **kwargs):
        if loss_kwarg_mbs is None:
            return original(self, *args, **kwargs)
        if len(loss_kwarg_mbs) != self._n_microbatches or not all(
            isinstance(value, dict) for value in loss_kwarg_mbs
        ):
            raise ValueError("loss_kwarg_mbs requires one dict per microbatch")
        common = kwargs.pop("loss_kwargs", None)
        if common is not None:
            if not isinstance(common, dict):
                raise ValueError("loss_kwargs must be a dict")
            if any(common.keys() & value.keys() for value in loss_kwarg_mbs):
                raise ValueError("Common and microbatch loss kwargs must not overlap")
            loss_kwarg_mbs = [{**common, **value} for value in loss_kwarg_mbs]
        previous = getattr(self, "_turbo_loss_kwarg_mbs", None)
        self._turbo_loss_kwarg_mbs = loss_kwarg_mbs
        try:
            # Dynamic metadata inference uses the first representative microbatch.
            return original(self, *args, loss_kwargs=loss_kwarg_mbs[0], **kwargs)
        finally:
            if previous is None:
                del self._turbo_loss_kwarg_mbs
            else:
                self._turbo_loss_kwarg_mbs = previous

    return step


def _wrap_loss(original):
    @wraps(original)
    def loss(self, stage, output, target_mbs, mb_index, loss_kwargs=None):
        per_microbatch = getattr(self, "_turbo_loss_kwarg_mbs", None)
        if per_microbatch is not None:
            loss_kwargs = per_microbatch[mb_index]
        return original(self, stage, output, target_mbs, mb_index, loss_kwargs)

    return loss


def apply_patch():
    """Backport loss dispatch, retaining distinct metrics/normalizers for every MB.

    TODO: Remove with the NPU wheel's public loss_kwarg_mbs API.
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/schedules.py
    """
    targets = (schedules.PipelineScheduleSingle, schedules.PipelineScheduleMulti)
    missing = [
        target
        for target in targets
        if "loss_kwarg_mbs" not in inspect.signature(target.step).parameters
    ]
    if not missing:
        return False
    base = schedules._PipelineSchedule
    if getattr(base, "_turbo_loss_kwargs_installed", False):
        return True
    expected = ("self", "stage", "output", "target_mbs", "mb_index", "loss_kwargs")
    if tuple(inspect.signature(base._maybe_compute_loss).parameters) != expected:
        raise RuntimeError("Unsupported pipeline loss dispatch signature")
    for target in missing:
        if "loss_kwargs" not in inspect.signature(target.step).parameters:
            raise RuntimeError("Unsupported pipeline step loss signature")
    base._maybe_compute_loss = _wrap_loss(base._maybe_compute_loss)
    for target in missing:
        target.step = _wrap_step(target.step)
    base._turbo_loss_kwargs_installed = True
    return True
