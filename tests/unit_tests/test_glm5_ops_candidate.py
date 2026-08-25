# Copyright (c) 2025 Huawei Technologies Co., Ltd.

import unittest

from torchtitanturbo.models.glm5.ops_candidate.communication.hccl_overlap_plan import (
    plan_hccl_double_buffer,
)
from torchtitanturbo.models.glm5.ops_candidate.memory.graph_workspace_plan import (
    plan_graph_workspace,
)


class Glm5NpuCandidatePlanTest(unittest.TestCase):
    def test_hccl_plan_alternates_buffers(self) -> None:
        stages = plan_hccl_double_buffer(3)
        dispatch = [stage for stage in stages if stage.phase == "dispatch"]

        self.assertEqual([stage.buffer_slot for stage in dispatch], [0, 1, 0])
        self.assertEqual(len(stages), 9)

    def test_graph_workspace_uses_stable_aligned_offsets(self) -> None:
        regions = plan_graph_workspace(
            (("score", 600), ("probability", 100)),
            alignment_bytes=512,
        )

        self.assertEqual(regions[0].offset_bytes, 0)
        self.assertEqual(regions[1].offset_bytes, 1024)


if __name__ == "__main__":
    unittest.main()
