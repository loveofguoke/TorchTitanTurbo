# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import os
import pickle
import time

import torch
import torch_npu

from torchtitan.tools.logging import logger


MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


class NpuMemoryProfiler:
    """NPU-optimized memory profiler using torch_npu.npu.memory."""

    def __init__(self, step_num, freq, snapshot_dir, leaf_folder, rank):
        torch_npu.npu.memory._record_memory_history(stacks="all")
        self.step_num = step_num
        self.freq = freq
        self._snapshot_dir = snapshot_dir
        self._leaf_folder = leaf_folder
        self._rank = rank

    def step(self, exit_ctx=False):
        self.step_num += 1
        if not exit_ctx and self.step_num % self.freq != 0:
            return
        if not exit_ctx:
            curr_step = self.step_num
            dir_name = f"step_{curr_step:012d}"
        else:
            curr_step = self.step_num - 1
            dir_name = f"step_{curr_step:012d}_exit"
        curr_snapshot_dir = os.path.join(
            self._snapshot_dir, dir_name, self._leaf_folder
        )
        if not os.path.exists(curr_snapshot_dir):
            os.makedirs(curr_snapshot_dir, exist_ok=True)
        logger.info(f"Dumping memory snapshot at step {curr_step}")
        begin = time.monotonic()
        output_file = os.path.join(
            curr_snapshot_dir, f"{self._rank:06d}_step_{curr_step}.pickle"
        )
        with open(output_file, "wb") as output:
            pickle.dump(torch.npu.memory._snapshot(), output, protocol=4)
        logger.info(
            f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
        )


def build_torch_profiler_npu(self, *, global_step, base_folder, leaf_folder):
    """NPU-optimized torch profiler build."""
    cfg = self._config
    if not cfg.enable_profiling:
        return None

    trace_dir = os.path.join(base_folder, cfg.save_traces_folder)
    profile_freq, warmup, active = (
        cfg.profile_freq,
        cfg.profiler_warmup,
        cfg.profiler_active,
    )
    rank = torch.distributed.get_rank()

    def trace_handler(prof):
        curr_trace_dir_name = f"iteration_{prof.step_num}"
        curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name, leaf_folder)
        if not os.path.exists(curr_trace_dir):
            os.makedirs(curr_trace_dir, exist_ok=True)
        logger.info(f"Dumping profiler traces at step {prof.step_num}")
        begin = time.monotonic()
        output_file = os.path.join(curr_trace_dir, f"rank{rank}_trace.json")
        prof.export_chrome_trace(output_file)
        logger.info(
            f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
        )

    logger.info(f"Profiling active. Traces will be saved at {trace_dir}")
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir, exist_ok=True)

    wait = profile_freq - (active + warmup)
    assert wait >= 0, "profile_freq must be >= warmup + active"

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[
            torch_npu.profiler.ExportType.Text,
            torch_npu.profiler.ExportType.Db,
        ],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        mstx_domain_include=[],
        mstx_domain_exclude=[],
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[
            torch_npu.profiler.HostSystem.CPU,
            torch_npu.profiler.HostSystem.MEM,
        ],
        sys_io=False,
        sys_interconnection=False,
    )

    gpu_device_profiled = None
    if torch.npu.is_available():
        gpu_device_profiled = torch_npu.profiler.ProfilerActivity.NPU
    elif torch.cuda.is_available():
        gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
    elif torch.xpu.is_available():
        gpu_device_profiled = torch.profiler.ProfilerActivity.XPU

    torch_profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            gpu_device_profiled,
        ],
        schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        experimental_config=experimental_config,
    )
    torch_profiler.__enter__()
    torch_profiler.step_num = global_step
    return torch_profiler


def build_memory_profiler_npu(self, *, global_step, base_folder, leaf_folder):
    """NPU-optimized memory profiler build."""
    cfg = self._config
    if not cfg.enable_memory_snapshot:
        return None

    snapshot_dir = os.path.join(base_folder, cfg.save_memory_snapshot_folder)
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir, exist_ok=True)
    rank = torch.distributed.get_rank()

    logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
    return NpuMemoryProfiler(
        global_step, cfg.profile_freq, snapshot_dir, leaf_folder, rank
    )


def apply_patch():
    import torchtitan.tools.profiler

    torchtitan.tools.profiler.Profiler.build_torch_profiler = build_torch_profiler_npu
    torchtitan.tools.profiler.Profiler.build_memory_profiler = build_memory_profiler_npu
    logger.info("Patched torchtitan.tools.profiler.Profiler for NPU")
