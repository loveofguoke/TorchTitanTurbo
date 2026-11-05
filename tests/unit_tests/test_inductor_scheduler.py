from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from torchtitanturbo.tools.inductor_scheduler import _fusion_pair_priority, apply_patch


def test_priority_preserves_welford_and_non_template_order():
    from torch_npu._inductor import config

    scheduling = SimpleNamespace(
        _triton_scheduling=SimpleNamespace(
            get_fusion_pair_priority=Mock(return_value=4)
        )
    )
    a, b = object(), object()
    with patch.object(config, "enable_welford", False):
        assert _fusion_pair_priority(scheduling, a, b) == 6
    with (
        patch.object(config, "enable_welford", True),
        patch(
            "torch_npu._inductor.choices.is_same_welford_output_group",
            return_value=True,
        ),
    ):
        assert _fusion_pair_priority(scheduling, a, b) == 0
    with (
        patch.object(config, "enable_welford", True),
        patch(
            "torch_npu._inductor.choices.is_same_welford_output_group",
            return_value=False,
        ),
        patch(
            "torch_npu._inductor.choices.is_welford_epilogue_fusion", return_value=True
        ),
    ):
        assert _fusion_pair_priority(scheduling, a, b) == 1


def test_adapter_is_idempotent_and_does_not_change_cuda_class():
    from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
    from torch_npu._inductor.codegen.npu_combined_scheduling import (
        NPUCombinedScheduling,
    )

    original = CUDACombinedScheduling.can_fuse_reduction_pair
    assert apply_patch()
    assert apply_patch()
    scheduler = NPUCombinedScheduling.__new__(NPUCombinedScheduling)
    assert scheduler.can_fuse_reduction_pair(object(), object())
    assert not scheduler.can_fuse_reduction_epilogue(object(), object())
    assert CUDACombinedScheduling.can_fuse_reduction_pair is original


def test_reduction_bridge_preserves_native_factory_and_rejects_strict():
    from torch._inductor import lowering
    from torch_npu._inductor import lowering as npu_lowering
    from torchtitanturbo.tools.inductor_reduction import apply_patch as apply_reduction

    native = Mock(return_value=object())

    def factory(reduction_type, override_return_dtype=None):
        return native(reduction_type, override_return_dtype=override_return_dtype)

    with patch.object(lowering, "make_reduction", factory), patch.object(
        npu_lowering, "make_reduction", factory
    ):
        assert apply_reduction()
        assert apply_reduction()
        result = lowering.make_reduction(
            "sum", override_return_dtype="dtype", strict_reduction=False
        )
        assert result is native.return_value
        native.assert_called_once_with("sum", override_return_dtype="dtype")
        with pytest.raises(NotImplementedError, match="strict reduction"):
            lowering.make_reduction("sum", strict_reduction=True)
        assert native.call_count == 1
