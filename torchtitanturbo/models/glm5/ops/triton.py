# Copyright (c) 2025 Huawei Technologies Co., Ltd.

"""Register GLM-5 Triton kernels for the Triton-Ascend backend."""

from dataclasses import dataclass

from torchtitan.config import derive, override
from torchtitan.models.glm5.model import DSAIndexerTopK, SparseMLA
from torchtitan.models.glm5.ops.triton import (
    TritonDSAIndexerTopK,
    TritonSparseMLA,
)


class AscendTritonDSAIndexerTopK(TritonDSAIndexerTopK):
    """Run the shared DSA index-score kernel through Triton-Ascend."""

    required_device_type = "npu"

    @dataclass(kw_only=True, slots=True)
    class Config(TritonDSAIndexerTopK.Config):
        pass


class AscendTritonSparseMLA(TritonSparseMLA):
    """Run the shared SparseMLA kernels through Triton-Ascend."""

    required_device_type = "npu"

    @dataclass(kw_only=True, slots=True)
    class Config(TritonSparseMLA.Config):
        pass


@override(
    target=DSAIndexerTopK.Config,
    description="Use the Triton-Ascend GLM-5 DSA indexer.",
)
def npu_triton_dsa_indexer(
    cfg: DSAIndexerTopK.Config,
) -> AscendTritonDSAIndexerTopK.Config:
    return derive(cfg, AscendTritonDSAIndexerTopK.Config)


@override(
    target=SparseMLA.Config,
    description="Use the Triton-Ascend GLM-5 SparseMLA forward and backward.",
)
def npu_triton_sparse_mla(
    cfg: SparseMLA.Config,
) -> AscendTritonSparseMLA.Config:
    return derive(cfg, AscendTritonSparseMLA.Config)


__all__ = [
    "AscendTritonDSAIndexerTopK",
    "AscendTritonSparseMLA",
    "npu_triton_dsa_indexer",
    "npu_triton_sparse_mla",
]
