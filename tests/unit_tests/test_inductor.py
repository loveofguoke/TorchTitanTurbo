# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchtitanturbo.compiler import inductor


def test_complex_dtensor_strategy_registration_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(
        inductor,
        "_COMPLEX_DTENSOR_STRATEGY_REGISTERED",
        False,
    )
    with patch(
        "torch.distributed.tensor._ops._pointwise_ops."
        "_register_single_dim_pointwise"
    ) as register:
        assert inductor._register_complex_dtensor_strategy()
        assert not inductor._register_complex_dtensor_strategy()

    register.assert_called_once_with(torch.ops.aten.complex.default)


def test_invalid_complex_dtensor_strategy_flag_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(
        "TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY",
        "yes",
    )
    with pytest.raises(ValueError, match="must be 0 or 1"):
        inductor._env_flag_enabled(
            "TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY"
        )


@pytest.mark.parametrize(
    ("requested", "initial"),
    [("0", True), ("1", False)],
)
def test_flex_attention_dkdv_tasklist_updates_preloaded_config(
    monkeypatch, requested: str, initial: bool
) -> None:
    import torch_npu._inductor.config as npu_config

    flex_attention_config = SimpleNamespace(bwd_dkdv_tasklist=initial)
    monkeypatch.setattr(
        npu_config, "flex_attention", flex_attention_config, raising=False
    )
    monkeypatch.setenv(
        inductor._FLEX_ATTENTION_BWD_DKDV_TASKLIST_ENV, requested
    )

    assert inductor._configure_flex_attention_dkdv_tasklist()
    assert flex_attention_config.bwd_dkdv_tasklist is (requested == "1")
    assert not inductor._configure_flex_attention_dkdv_tasklist()
