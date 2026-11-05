"""Default-off bridge for the observed TorchNPU reduction-factory signature gap."""

import inspect


def apply_patch() -> bool:
    """Preserve the NPU factory; reject rather than discard strict semantics."""
    from torch._inductor import lowering
    from torch_npu._inductor import lowering as npu_lowering

    original = lowering.make_reduction
    if getattr(original, "_torchtitanturbo_reduction_bridge", False):
        return True
    if "strict_reduction" in inspect.signature(original).parameters:
        return False
    if original is not npu_lowering.make_reduction:
        raise RuntimeError("Unexpected reduction factory owner")

    def make_reduction(
        reduction_type, override_return_dtype=None, *, strict_reduction=False
    ):
        if strict_reduction:
            raise NotImplementedError(
                "This TorchNPU build does not support strict reduction ordering"
            )
        return original(reduction_type, override_return_dtype=override_return_dtype)

    make_reduction._torchtitanturbo_reduction_bridge = True
    lowering.make_reduction = make_reduction
    return True
