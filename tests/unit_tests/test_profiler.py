# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import os
import unittest
from unittest.mock import patch

from torchtitanturbo.tools.profiler import NpuProfilerOptions


class TestNpuProfilerOptions(unittest.TestCase):
    def test_default_options_use_synchronous_parsing(self):
        with patch.dict(os.environ, {}, clear=True):
            options = NpuProfilerOptions.from_environment()

        self.assertEqual(options.level, "level1")
        self.assertIsNone(options.ranks)
        self.assertTrue(options.record_shapes)
        self.assertTrue(options.profile_memory)
        self.assertTrue(options.online_parse)
        self.assertEqual(options.parse_mode, "sync")
        self.assertEqual(options.host_system, ())

    def test_probe_options_are_loaded_from_environment(self):
        environment = {
            "TORCHTITAN_NPU_PROFILER_LEVEL": "level0",
            "TORCHTITAN_NPU_PROFILER_RANKS": "2,0,2",
            "TORCHTITAN_NPU_PROFILER_RECORD_SHAPES": "false",
            "TORCHTITAN_NPU_PROFILER_PROFILE_MEMORY": "0",
            "TORCHTITAN_NPU_PROFILER_EXPORT_TYPES": "text",
            "TORCHTITAN_NPU_PROFILER_HOST_SYSTEM": "cpu,mem",
            "TORCHTITAN_NPU_PROFILER_WITH_MODULES": "true",
            "TORCHTITAN_NPU_PROFILER_WITH_STACK": "true",
            "TORCHTITAN_NPU_PROFILER_WITH_FLOPS": "true",
            "TORCHTITAN_NPU_PROFILER_MSPROF_TX": "true",
            "TORCHTITAN_NPU_PROFILER_EXPORT_STACKS": "true",
            "TORCHTITAN_NPU_PROFILER_PARSE_MODE": "sync",
            "TORCHTITAN_NPU_PROFILER_GC_DETECT_THRESHOLD": "1.5",
        }
        with patch.dict(os.environ, environment, clear=True):
            options = NpuProfilerOptions.from_environment()

        self.assertEqual(options.level, "level0")
        self.assertEqual(options.ranks, (0, 2))
        self.assertFalse(options.record_shapes)
        self.assertFalse(options.profile_memory)
        self.assertEqual(options.export_types, ("text",))
        self.assertEqual(options.parse_mode, "sync")
        self.assertEqual(options.host_system, ("cpu", "mem"))
        self.assertTrue(options.with_modules)
        self.assertTrue(options.with_flops)
        self.assertTrue(options.msprof_tx)
        self.assertTrue(options.export_stacks)
        self.assertFalse(options.export_memory_timeline)
        self.assertEqual(options.gc_detect_threshold, 1.5)
        self.assertTrue(options.profiles_rank(2))
        self.assertFalse(options.profiles_rank(1))

    def test_level0_rejects_aic_metrics(self):
        environment = {
            "TORCHTITAN_NPU_PROFILER_LEVEL": "level0",
            "TORCHTITAN_NPU_PROFILER_AIC_METRICS": "pipe_utilization",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "Level0"):
                NpuProfilerOptions.from_environment()

    def test_legacy_online_parse_switch_maps_to_offline(self):
        environment = {"TORCHTITAN_NPU_PROFILER_ONLINE_PARSE": "false"}
        with patch.dict(os.environ, environment, clear=True):
            options = NpuProfilerOptions.from_environment()

        self.assertEqual(options.parse_mode, "offline")
        self.assertFalse(options.online_parse)

    def test_stack_export_requires_stack_capture_and_sync_parse(self):
        with patch.dict(
            os.environ,
            {"TORCHTITAN_NPU_PROFILER_EXPORT_STACKS": "true"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "WITH_STACK"):
                NpuProfilerOptions.from_environment()
        with patch.dict(
            os.environ,
            {
                "TORCHTITAN_NPU_PROFILER_EXPORT_STACKS": "true",
                "TORCHTITAN_NPU_PROFILER_WITH_STACK": "true",
                "TORCHTITAN_NPU_PROFILER_PARSE_MODE": "offline",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "synchronous"):
                NpuProfilerOptions.from_environment()

    def test_memory_timeline_requires_memory_shape_and_callsite_data(self):
        base = {
            "TORCHTITAN_NPU_PROFILER_EXPORT_MEMORY_TIMELINE": "true",
            "TORCHTITAN_NPU_PROFILER_RECORD_SHAPES": "true",
            "TORCHTITAN_NPU_PROFILER_PROFILE_MEMORY": "true",
            "TORCHTITAN_NPU_PROFILER_WITH_MODULES": "true",
        }
        with patch.dict(os.environ, base, clear=True):
            self.assertTrue(
                NpuProfilerOptions.from_environment().export_memory_timeline
            )
        requirements = (
            ("RECORD_SHAPES", "false", "RECORD_SHAPES"),
            ("PROFILE_MEMORY", "false", "PROFILE_MEMORY"),
            ("WITH_MODULES", "false", "WITH_STACK"),
        )
        for name, value, message in requirements:
            environment = dict(base)
            environment["TORCHTITAN_NPU_PROFILER_" + name] = value
            with self.subTest(name=name), patch.dict(
                os.environ, environment, clear=True
            ):
                with self.assertRaisesRegex(ValueError, message):
                    NpuProfilerOptions.from_environment()


if __name__ == "__main__":
    unittest.main()
