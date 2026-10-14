# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest

import torch

from torchtitanturbo.optimizations.deepgemm import (
    _DeepGemmGroupedBf16,
    DeepGemmCapabilityError,
    DeepGemmGroupedRequest,
    validate_deepgemm_grouped_request,
)


def valid_request(**overrides) -> DeepGemmGroupedRequest:
    values = {
        "platform": "npu",
        "input_dtype": torch.bfloat16,
        "weight_dtype": torch.bfloat16,
        "offsets_dtype": torch.int32,
        "num_tokens": 2048,
        "input_dim": 7168,
        "output_dim": 2048,
        "num_experts": 32,
        "requires_grad": False,
    }
    values.update(overrides)
    return DeepGemmGroupedRequest(**values)


class TestDeepGemmCapability(unittest.TestCase):
    def test_accepts_forward_only_bf16_npu_request(self):
        validate_deepgemm_grouped_request(valid_request())

    def test_accepts_training_request(self):
        validate_deepgemm_grouped_request(valid_request(requires_grad=True))

    def test_rejects_non_bf16_weight(self):
        with self.assertRaisesRegex(DeepGemmCapabilityError, "BF16 weights"):
            validate_deepgemm_grouped_request(
                valid_request(weight_dtype=torch.float32)
            )

    def test_rejects_non_npu_platform(self):
        with self.assertRaisesRegex(DeepGemmCapabilityError, "requires NPU"):
            validate_deepgemm_grouped_request(valid_request(platform="cuda"))

    def test_rejects_all_empty_local_batch_until_verified(self):
        with self.assertRaisesRegex(DeepGemmCapabilityError, "routed token"):
            validate_deepgemm_grouped_request(valid_request(num_tokens=0))


class _ReferenceDeepGemm:
    @staticmethod
    def _ranges(offsets):
        start = 0
        for offset in offsets.tolist():
            end = int(offset)
            yield start, end
            start = end

    def m_grouped_bf16_gemm_nt_contiguous(self, a, b, out, counts):
        for expert, (start, end) in enumerate(self._ranges(counts)):
            out[start:end].copy_(a[start:end] @ b[expert].T)

    def m_grouped_bf16_gemm_nn_contiguous(self, a, b, out, counts):
        for expert, (start, end) in enumerate(self._ranges(counts)):
            out[start:end].copy_(a[start:end] @ b[expert])

    def k_grouped_bf16_gemm_tn_contiguous(
        self, a_t, b, out, counts_cpu, counts, *args
    ):
        for expert, (start, end) in enumerate(self._ranges(counts)):
            out[expert].copy_(a_t[start:end].T @ b[start:end])


class TestDeepGemmAutograd(unittest.TestCase):
    def test_backward_computes_input_and_weight_gradients(self):
        torch.manual_seed(0)
        a = torch.randn(5, 3, dtype=torch.bfloat16, requires_grad=True)
        b_t = torch.randn(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
        offsets = torch.tensor([2, 5], dtype=torch.int32)

        out = _DeepGemmGroupedBf16.apply(_ReferenceDeepGemm(), a, b_t, offsets)
        out.float().sum().backward()

        a_ref = a.detach().clone().requires_grad_()
        b_ref = b_t.detach().clone().requires_grad_()
        expected = torch.cat(
            [a_ref[:2] @ b_ref[0], a_ref[2:] @ b_ref[1]], dim=0
        )
        expected.float().sum().backward()
        torch.testing.assert_close(a.grad, a_ref.grad)
        torch.testing.assert_close(b_t.grad, b_ref.grad)


if __name__ == "__main__":
    unittest.main()
