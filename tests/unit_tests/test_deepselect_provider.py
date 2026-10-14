# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest

import torch

from torchtitanturbo.optimizations.deepselect import (
    DeepSelectCapabilityError,
    DeepSelectRequest,
    validate_deepselect_request,
)


class TestDeepSelectCapability(unittest.TestCase):
    def test_npu_accepts_aligned_bfloat16_dsa_request(self):
        validate_deepselect_request(
            DeepSelectRequest(
                platform="npu",
                dtype=torch.bfloat16,
                num_rows=128,
                row_width=4096,
                topk=512,
                row_stride_bytes=8192,
                input_stride_requirement=256,
            )
        )

    def test_npu_rejects_glm_fp32_index_scores(self):
        with self.assertRaisesRegex(
            DeepSelectCapabilityError,
            "does not support torch.float32",
        ):
            validate_deepselect_request(
                DeepSelectRequest(
                    platform="npu",
                    dtype=torch.float32,
                    num_rows=128,
                    row_width=4096,
                    topk=512,
                    row_stride_bytes=16384,
                )
            )

    def test_rejects_topk_above_public_limit(self):
        with self.assertRaisesRegex(DeepSelectCapabilityError, "topk <="):
            validate_deepselect_request(
                DeepSelectRequest(
                    platform="cuda",
                    dtype=torch.bfloat16,
                    num_rows=8,
                    row_width=8192,
                    topk=4097,
                    row_stride_bytes=16384,
                )
            )

    def test_rejects_unaligned_row_stride(self):
        with self.assertRaisesRegex(DeepSelectCapabilityError, "aligned row stride"):
            validate_deepselect_request(
                DeepSelectRequest(
                    platform="cuda",
                    dtype=torch.bfloat16,
                    num_rows=8,
                    row_width=4097,
                    topk=512,
                    row_stride_bytes=8194,
                    input_stride_requirement=256,
                )
            )


if __name__ == "__main__":
    unittest.main()
