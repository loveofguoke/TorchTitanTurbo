# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Backport the pipeline metadata-cleanup callback API on older NPU wheels."""

import inspect
from collections import OrderedDict
from functools import wraps

from torch.distributed.pipelining import schedules
from torch.distributed.pipelining.stage import InferenceMode, PipelineStage
from torch.utils.hooks import RemovableHandle


def _after_stage_cleanup(stage, callbacks, run):
    """Run callbacks at the existing last-stage cleanup boundary, not on P2P errors."""
    name = "_post_metadata_inference_cleanup"
    previous_override = vars(stage).get(name)
    had_override = name in vars(stage)
    original = getattr(stage, name)

    def cleanup():
        original()
        for callback in tuple(callbacks.values()):
            callback()

    setattr(stage, name, cleanup)
    try:
        return run()
    finally:
        if had_override:
            setattr(stage, name, previous_override)
        else:
            delattr(stage, name)


def apply_patch():
    """Install only when the public callback API is absent; newer wheels are untouched.

    TODO: Remove when the NPU wheel contains the upstream callback API used by
    TorchTitan's FSDP reset. https://github.com/pytorch/pytorch/issues/203178
    """
    target = schedules._PipelineSchedule
    if hasattr(target, "register_post_metadata_inference_cleanup"):
        return False
    original = target._initialize_pp_stages
    signature = inspect.signature(original)
    expected = (
        "self",
        "stages",
        "args",
        "kwargs",
        "target",
        "fwd_initialized",
        "bwd_initialized",
        "loss_kwargs",
    )
    if tuple(signature.parameters) != expected:
        raise RuntimeError(f"Unsupported pipeline metadata signature: {signature}")

    def register(self, callback):
        if not hasattr(self, "_post_metadata_inference_cleanups"):
            self._post_metadata_inference_cleanups = OrderedDict()
        callbacks = self._post_metadata_inference_cleanups
        handle = RemovableHandle(callbacks)
        callbacks[handle.id] = callback
        return handle

    @wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        stages = bound.arguments["stages"]
        dynamic = [stage for stage in stages if isinstance(stage, PipelineStage)]
        callbacks = getattr(self, "_post_metadata_inference_cleanups", {})
        if not callbacks or not any(
            stage._inference_mode == InferenceMode.DYNAMIC for stage in dynamic
        ):
            return original(self, *args, **kwargs)
        if any(stage.device.type != "npu" for stage in dynamic):
            raise ValueError("Pipeline cleanup backport supports NPU stages only")
        return _after_stage_cleanup(
            dynamic[-1], callbacks, lambda: original(self, *args, **kwargs)
        )

    target.register_post_metadata_inference_cleanup = register
    target._initialize_pp_stages = initialize
    return True
