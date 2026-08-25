# Copyright (c) 2025 Huawei Technologies Co., Ltd.

"""Graph-stable static workspace layout prototype for Ascend kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkspaceRegion:
    name: str
    offset_bytes: int
    size_bytes: int


def plan_graph_workspace(
    regions: tuple[tuple[str, int], ...],
    *,
    alignment_bytes: int = 512,
) -> tuple[WorkspaceRegion, ...]:
    """Assign stable aligned offsets without allocating an NPU tensor."""

    if alignment_bytes <= 0:
        raise ValueError("alignment_bytes must be positive")
    if len({name for name, _ in regions}) != len(regions):
        raise ValueError("workspace region names must be unique")
    if any(size_bytes < 0 for _, size_bytes in regions):
        raise ValueError("workspace region sizes must be non-negative")

    result = []
    offset = 0
    for name, size_bytes in regions:
        offset = (
            (offset + alignment_bytes - 1) // alignment_bytes
        ) * alignment_bytes
        result.append(WorkspaceRegion(name, offset, size_bytes))
        offset += size_bytes
    return tuple(result)


__all__ = ["WorkspaceRegion", "plan_graph_workspace"]
