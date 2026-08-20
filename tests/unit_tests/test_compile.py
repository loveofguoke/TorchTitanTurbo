# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from types import SimpleNamespace

from torchtitanturbo.tools.compile import validate_npu_compile_config


def _config(**overrides):
    values = {
        "enable": True,
        "components": ["model"],
        "backend": "inductor",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestNpuCompileConfig(unittest.TestCase):
    def test_inductor_accepts_default_shape_policy(self):
        validate_npu_compile_config(_config())

    def test_npugraphs_accepts_model_compilation(self):
        validate_npu_compile_config(_config(backend="npugraphs"))

    def test_npugraphs_rejects_loss_compilation(self):
        with self.assertRaisesRegex(ValueError, "model compilation only"):
            validate_npu_compile_config(
                _config(
                    backend="npugraphs",
                    components=["model", "loss"],
                )
            )


if __name__ == "__main__":
    unittest.main()
