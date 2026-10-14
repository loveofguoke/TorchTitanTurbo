# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import unittest
from unittest import mock

import torch

from torchtitan.models.common.moe import RoutedExperts

from torchtitanturbo.optimizations.deepep import (
    AscendDeepEPTokenDispatcher,
    AscendDeepEPTokenDispatcherConverter,
    DeepEpAscendBufferProvider,
    DeepEpDispatchResult,
    DeepEpBufferRequest,
    DeepEpCapabilityError,
    validate_deepep_buffer_request,
)
from torchtitanturbo.optimizations.flashmla import (
    FlashMlaCapabilityError,
    FlashMlaSparseRequest,
    validate_flashmla_sparse_request,
)
from torchtitanturbo.optimizations.tilekernels import (
    TileKernelsCapabilityError,
    TileKernelsTopKRequest,
    validate_tilekernels_topk_request,
)
from torchtitanturbo.optimizations.registry import OptimizationProvidersConfig


class TestTileKernelsCapability(unittest.TestCase):
    def test_accepts_glm_style_fp32_router_scores(self):
        validate_tilekernels_topk_request(
            TileKernelsTopKRequest(
                platform="npu",
                dtype=torch.float32,
                num_tokens=128,
                num_experts=256,
                topk=8,
            )
        )

    def test_rejects_topk_above_kernel_limit(self):
        with self.assertRaisesRegex(TileKernelsCapabilityError, r"min\(32"):
            validate_tilekernels_topk_request(
                TileKernelsTopKRequest(
                    platform="npu",
                    dtype=torch.float32,
                    num_tokens=128,
                    num_experts=256,
                    topk=33,
                )
            )


class TestFlashMlaCapability(unittest.TestCase):
    def test_accepts_official_ascend_sparse_forward_shape(self):
        validate_flashmla_sparse_request(
            FlashMlaSparseRequest(
                platform="npu",
                query_dtype=torch.bfloat16,
                kv_dtype=torch.bfloat16,
                indices_dtype=torch.int32,
                num_query_heads=64,
                query_key_dim=512,
                value_dim=512,
                requires_grad=False,
            )
        )

    def test_rejects_training_without_backward(self):
        with self.assertRaisesRegex(FlashMlaCapabilityError, "forward-only"):
            validate_flashmla_sparse_request(
                FlashMlaSparseRequest(
                    platform="npu",
                    query_dtype=torch.bfloat16,
                    kv_dtype=torch.bfloat16,
                    indices_dtype=torch.int32,
                    num_query_heads=64,
                    query_key_dim=512,
                    value_dim=512,
                    requires_grad=True,
                )
            )


class TestDeepEpCapability(unittest.TestCase):
    def test_accepts_single_rank_for_local_validation(self):
        validate_deepep_buffer_request(
            DeepEpBufferRequest(
                platform="npu",
                num_ranks=1,
                num_max_tokens_per_rank=256,
                hidden_dim=7168,
                num_topk=8,
            )
        )

    def test_dispatch_and_combine_have_autograd_duals(self):
        class Pending:
            def __init__(self, value):
                self.value = value

            def wait(self):
                return self.value

        class Buffer:
            def dispatch(self, x, **kwargs):
                if kwargs.get("handle") is not None:
                    return Pending((x, None, None, kwargs["handle"]))
                weights = kwargs["topk_weights"]
                handle = type(
                    "Handle",
                    (),
                    {
                        "psum_num_recv_tokens_per_expert": torch.tensor(
                            [2, 4], dtype=torch.int32
                        )
                    },
                )()
                return Pending((x, None, weights, handle))

            def combine(self, x, **kwargs):
                weights = kwargs.get("topk_weights")
                return Pending((x, weights))

        provider = DeepEpAscendBufferProvider()
        buffer = Buffer()
        x = torch.randn(4, 3, requires_grad=True)
        weights = torch.randn(4, 2, requires_grad=True)
        indices = torch.zeros(4, 2, dtype=torch.int64)
        dispatched = provider.dispatch_training(
            buffer=buffer,
            x_TH=x,
            topk_indices_TK=indices,
            topk_weights_TK=weights,
            num_experts=2,
        )
        combined = provider.combine_training(
            buffer=buffer,
            expert_output_RH=dispatched.recv_x_RH,
            handle=dispatched.handle,
        )
        (combined.sum() + dispatched.recv_weights_RK.sum()).backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        torch.testing.assert_close(weights.grad, torch.ones_like(weights))

    def test_dispatcher_matches_torchtitan_dispatch_combine_contract(self):
        class Group:
            def size(self):
                return 2

        group = Group()

        class Mesh:
            def get_group(self):
                return group

        class Buffer:
            destroyed = False

            def destroy(self):
                self.destroyed = True

        class Provider:
            def build(self, **kwargs):
                self.build_kwargs = kwargs
                return Buffer()

            def dispatch_training(self, **kwargs):
                return DeepEpDispatchResult(
                    recv_x_RH=kwargs["x_TH"].repeat_interleave(2, dim=0),
                    recv_weights_RK=torch.tensor(
                        [[0.25], [0.75], [0.5], [0.5]], dtype=torch.float32
                    ),
                    num_tokens_per_local_expert_e=torch.tensor(
                        [2, 2], dtype=torch.int32
                    ),
                    handle=object(),
                )

            def combine_training(self, **kwargs):
                return kwargs["expert_output_RH"]

        config = AscendDeepEPTokenDispatcher.Config(
            num_experts=4,
            top_k=2,
            num_max_tokens_per_rank=8,
            hidden_dim=3,
        )
        dispatcher = config.build()
        self.assertIsInstance(dispatcher, AscendDeepEPTokenDispatcher)
        dispatcher.provider = Provider()
        with mock.patch.object(
            AscendDeepEPTokenDispatcher,
            "ep_mesh",
            new_callable=mock.PropertyMock,
            return_value=Mesh(),
        ):
            dispatcher.init_buffer()

        x_TD = torch.ones(2, 3)
        routed_RD, counts_e, metadata = dispatcher.dispatch(
            x_TD,
            torch.full((2, 2), 0.5),
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
            torch.ones(4, dtype=torch.int32),
        )
        combined_RD = dispatcher.combine(routed_RD, metadata, x_TD)

        torch.testing.assert_close(counts_e, torch.tensor([2, 2], dtype=torch.int32))
        torch.testing.assert_close(
            combined_RD,
            torch.tensor(
                [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75],
                 [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
            ),
        )
        buffer = dispatcher.buffer
        dispatcher.close()
        self.assertTrue(buffer.destroyed)
        self.assertIsNone(dispatcher.buffer)

    def test_dispatchers_share_one_buffer_per_ep_group(self):
        class Group:
            def size(self):
                return 2

        class Mesh:
            group = Group()

            def get_group(self):
                return self.group

        class Buffer:
            destroyed = False

            def destroy(self):
                self.destroyed = True

        class Provider:
            num_builds = 0

            def build(self, **kwargs):
                type(self).num_builds += 1
                return Buffer()

        config = AscendDeepEPTokenDispatcher.Config(
            num_experts=4,
            top_k=2,
            num_max_tokens_per_rank=8,
            hidden_dim=3,
        )
        first = config.build()
        second = config.build()
        first.provider = Provider()
        second.provider = Provider()
        mesh = Mesh()
        with mock.patch.object(
            AscendDeepEPTokenDispatcher,
            "ep_mesh",
            new_callable=mock.PropertyMock,
            return_value=mesh,
        ):
            first.init_buffer()
            second.init_buffer()

        self.assertIs(first.buffer, second.buffer)
        self.assertEqual(Provider.num_builds, 1)
        buffer = first.buffer
        first.close()
        self.assertFalse(buffer.destroyed)
        second.close()
        self.assertTrue(buffer.destroyed)

    def test_converter_replaces_only_explicit_deepep_dispatchers(self):
        from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
        from torchtitan.config.parallelism import ParallelismConfig
        from torchtitan.models.glm5 import build_model_config

        model_config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        for _fqn, routed, _parent, _attr in model_config.traverse(
            RoutedExperts.Config
        ):
            routed.token_dispatcher = DeepEPTokenDispatcher.Config(
                num_experts=routed.token_dispatcher.num_experts,
                top_k=routed.token_dispatcher.top_k,
                num_max_tokens_per_rank=256,
                hidden_dim=model_config.dim,
            )
        converter = AscendDeepEPTokenDispatcherConverter(
            AscendDeepEPTokenDispatcherConverter.Config()
        )
        converted = converter.convert(model_config)

        dispatchers = [
            config.token_dispatcher
            for _fqn, config, _parent, _attr in converted.traverse(
                RoutedExperts.Config
            )
        ]
        self.assertTrue(dispatchers)
        self.assertTrue(
            all(
                isinstance(config, AscendDeepEPTokenDispatcher.Config)
                for config in dispatchers
            )
        )
        self.assertTrue(
            all(
                isinstance(config, DeepEPTokenDispatcher.Config)
                for config in dispatchers
            )
        )

    def test_accepts_official_ascend_ep_constraints(self):
        validate_deepep_buffer_request(
            DeepEpBufferRequest(
                platform="npu",
                num_ranks=8,
                num_max_tokens_per_rank=4096,
                hidden_dim=7168,
                num_topk=8,
            )
        )

    def test_rejects_odd_rank_count(self):
        with self.assertRaisesRegex(DeepEpCapabilityError, "even"):
            validate_deepep_buffer_request(
                DeepEpBufferRequest(
                    platform="npu",
                    num_ranks=3,
                    num_max_tokens_per_rank=4096,
                    hidden_dim=7168,
                    num_topk=8,
                )
            )


class TestOptimizationProvidersConfig(unittest.TestCase):
    def test_defaults_select_no_external_provider(self):
        config = OptimizationProvidersConfig()
        self.assertEqual(config.build_selected(), {})
        self.assertEqual(config.selected_names()["moe_gemm"], "torch")

    def test_explicit_variants_build_without_importing_external_libraries(self):
        config = OptimizationProvidersConfig(
            dsa_topk_backend="deepselect",
            moe_topk_backend="tilekernels",
            moe_gemm_backend="deepgemm_ascend",
            ep_dispatch_backend="deepep_ascend",
            sparse_attention_backend="flashmla_ascend",
        )
        self.assertEqual(
            set(config.build_selected()),
            {
                "dsa_topk",
                "moe_topk",
                "moe_gemm",
                "ep_dispatch",
                "sparse_attention",
            },
        )

    def test_invalid_backend_fails_instead_of_silently_using_reference(self):
        with self.assertRaisesRegex(ValueError, "Unsupported moe_gemm_backend"):
            OptimizationProvidersConfig(moe_gemm_backend="unknown")


if __name__ == "__main__":
    unittest.main()
