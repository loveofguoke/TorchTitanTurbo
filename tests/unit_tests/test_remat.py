# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from unittest.mock import patch

import torch

from torchtitanturbo.distributed.remat import apply_patch


def test_remat_patch_preserves_non_npu_storage_path():
    from torch_remat import _placeholder

    original = _placeholder._null_storage
    apply_patch()
    patched = _placeholder._null_storage

    if original is patched:
        # The package import bootstrap may already have installed the patch.
        assert getattr(patched, "_torchtitanturbo_npu_remat_storage", False)

    storage = patched(16, torch.device("cpu"))
    assert storage.nbytes() == 16


def test_remat_patch_allocates_owning_storage_for_npu():
    from torch_remat import _placeholder

    apply_patch()
    fake_storage = object()
    fake_tensor = type(
        "FakeTensor", (), {"untyped_storage": lambda self: fake_storage}
    )()

    with patch("torch.empty", return_value=fake_tensor) as empty:
        storage = _placeholder._null_storage(37, torch.device("npu:0"))

    assert storage is fake_storage
    empty.assert_called_once_with(
        37, dtype=torch.uint8, device=torch.device("npu:0")
    )
