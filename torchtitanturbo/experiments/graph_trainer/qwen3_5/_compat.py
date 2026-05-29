# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Qwen3.5 graph_trainer compatibility shims.

These shims are intentionally scoped to the Qwen3.5 graph_trainer path. They
bridge small API gaps between TorchTitan main and the installed PyTorch build,
and each patch is feature-detected so newer torch/torch_npu stacks remain
untouched.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import sys
import types

import torch


_AO_LIBRARY = None


def _patch_fsdp_data_parallel_mesh_dims() -> None:
    try:
        import torch.distributed.fsdp as fsdp
    except Exception:
        return

    if hasattr(fsdp, "DataParallelMeshDims"):
        return

    @dataclass(frozen=True)
    class DataParallelMeshDims:
        shard: str | tuple[str, ...] | None = None
        replicate: str | None = None

    fsdp.DataParallelMeshDims = DataParallelMeshDims


def _patch_opaque_object_api() -> None:
    try:
        import torch._library.opaque_object as opaque_object
    except Exception:
        return

    if hasattr(opaque_object, "is_opaque_value"):
        return

    def is_opaque_value(value) -> bool:
        return opaque_object.is_opaque_value_type(type(value))

    opaque_object.is_opaque_value = is_opaque_value


def _patch_activation_offload_import() -> None:
    global _AO_LIBRARY
    module_name = "torch._functorch._activation_offloading"
    ops_name = f"{module_name}.offload_ops"
    if ops_name in sys.modules:
        return

    package = types.ModuleType(module_name)
    package.__path__ = []
    ops = types.ModuleType(ops_name)
    sys.modules.setdefault(module_name, package)
    sys.modules.setdefault(ops_name, ops)

    if hasattr(torch.ops, "ao") and hasattr(torch.ops.ao, "offload"):
        return

    try:
        _AO_LIBRARY = torch.library.Library("ao", "DEF")
        _AO_LIBRARY.define("offload(Tensor input) -> Tensor")
        _AO_LIBRARY.define("reload(Tensor input, Device device) -> Tensor")
        _AO_LIBRARY.define(
            "wait_tensor(Tensor input, Tensor? keepalive=None, "
            "Tensor? dependency=None) -> Tensor"
        )
    except Exception:
        pass


def _patch_checkpoint_policy_api() -> None:
    try:
        from torch.utils.checkpoint import CheckpointPolicy
    except Exception:
        return

    if not hasattr(CheckpointPolicy, "MUST_CPU_OFFLOAD"):
        CheckpointPolicy.MUST_CPU_OFFLOAD = CheckpointPolicy.MUST_SAVE


def _patch_flex_attention_api() -> None:
    try:
        import torch.nn.attention.flex_attention as flex_attention
    except Exception:
        return

    if hasattr(flex_attention, "_MaskModWrapper"):
        return

    class _MaskModWrapper:
        def __eq__(self, other):
            return type(self) is type(other)

        def __hash__(self):
            return hash(type(self))

    flex_attention._MaskModWrapper = _MaskModWrapper


def _patch_torch_compiler_api() -> None:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return

    @contextmanager
    def _compiler_noop_context():
        yield

    if not hasattr(compiler, "_non_strict_tracing_context"):
        compiler._non_strict_tracing_context = _compiler_noop_context
    if not hasattr(compiler, "_patch_engine_backward"):
        compiler._patch_engine_backward = _compiler_noop_context


def _patch_fx_print_readable_api() -> None:
    try:
        from torch.fx.graph_module import GraphModule
    except Exception:
        return

    if getattr(GraphModule.print_readable, "_torchtitanturbo_patched", False):
        return

    orig_print_readable = GraphModule.print_readable

    def print_readable(self, *args, **kwargs):
        kwargs.pop("additional_meta", None)
        return orig_print_readable(self, *args, **kwargs)

    print_readable._torchtitanturbo_patched = True
    GraphModule.print_readable = print_readable


def apply_qwen35_graph_trainer_compat() -> None:
    _patch_fsdp_data_parallel_mesh_dims()
    _patch_opaque_object_api()
    _patch_activation_offload_import()
    _patch_checkpoint_policy_api()
    _patch_flex_attention_api()
    _patch_torch_compiler_api()
    _patch_fx_print_readable_api()
