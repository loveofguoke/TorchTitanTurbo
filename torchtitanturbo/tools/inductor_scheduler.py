"""Opt-in adapter for TorchNPU inheriting NVIDIA-only Inductor scheduler hooks.

Remove this bridge when TorchNPU supplies these hooks itself. It does not
construct a CUDA scheduler or change NPU kernels, tensor math or remat policy.
"""

import inspect


def _fusion_pair_priority(self, node1, node2):
    from torch_npu._inductor import config

    if config.enable_welford:
        from torch_npu._inductor.choices import (
            is_same_welford_output_group,
            is_welford_epilogue_fusion,
        )

        if is_same_welford_output_group(node1, node2):
            return 0
        if is_welford_epilogue_fusion(node1, node2):
            return 1
    # Match CUDACombinedScheduling's non-NVIDIA-template priority branch.
    return self._triton_scheduling.get_fusion_pair_priority(node1, node2) + 2


def apply_patch() -> bool:
    """Bridge only the observed inherited-hook gap; leave fixed versions alone."""
    from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
    from torch._inductor.scheduler import BaseScheduling
    from torch_npu._inductor.codegen.npu_combined_scheduling import (
        NPUCombinedScheduling,
    )

    cls = NPUCombinedScheduling
    marker = "_torchtitanturbo_npu_scheduler_hooks"
    if getattr(cls, marker, False):
        return True
    method = cls.can_fuse_reduction_pair
    if "_nv_universal_gemm_scheduling" not in method.__code__.co_names:
        return False
    if not issubclass(cls, CUDACombinedScheduling):
        raise RuntimeError("Unexpected TorchNPU scheduler inheritance")
    for name in (
        "can_fuse_reduction_pair",
        "can_fuse_reduction_epilogue",
        "get_fusion_pair_priority",
    ):
        if tuple(inspect.signature(getattr(cls, name)).parameters) != (
            "self",
            "node1",
            "node2",
        ):
            raise RuntimeError(f"Unsupported TorchNPU scheduler hook: {name}")
    cls.can_fuse_reduction_pair = BaseScheduling.can_fuse_reduction_pair
    if (
        "_nv_universal_gemm_scheduling"
        in cls.can_fuse_reduction_epilogue.__code__.co_names
    ):
        cls.can_fuse_reduction_epilogue = BaseScheduling.can_fuse_reduction_epilogue
    cls.get_fusion_pair_priority = _fusion_pair_priority
    setattr(cls, marker, True)
    return True
