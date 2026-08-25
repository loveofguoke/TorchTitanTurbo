# Copyright (c) 2025 Huawei Technologies Co., Ltd.

"""Static double-buffer schedule for an eventual HCCL overlap integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HcclOverlapStage:
    clock: int
    chunk_id: int
    phase: str
    buffer_bank: str
    buffer_slot: int


def plan_hccl_double_buffer(num_chunks: int) -> tuple[HcclOverlapStage, ...]:
    """Describe dispatch/compute/combine ownership without launching HCCL."""

    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    stages = []
    for chunk_id in range(num_chunks):
        slot = chunk_id % 2
        stages.append(
            HcclOverlapStage(chunk_id, chunk_id, "dispatch", "input", slot)
        )
        stages.append(
            HcclOverlapStage(chunk_id + 1, chunk_id, "compute", "input", slot)
        )
        stages.append(
            HcclOverlapStage(chunk_id + 2, chunk_id, "combine", "output", slot)
        )
    phase_order = {"dispatch": 0, "compute": 1, "combine": 2}
    return tuple(
        sorted(
            stages,
            key=lambda stage: (stage.clock, phase_order[stage.phase]),
        )
    )


__all__ = ["HcclOverlapStage", "plan_hccl_double_buffer"]
