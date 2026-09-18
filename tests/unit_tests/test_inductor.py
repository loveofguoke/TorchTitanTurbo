# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

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
