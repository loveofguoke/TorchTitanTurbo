# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Ascend NPU implementation of TorchTitan's profiler lifecycle.

TorchTitan owns the device-neutral profiler schedule and lifecycle. This
module only translates that contract to ``torch_npu.profiler``. NPU-only
controls use environment variables so TorchTitan's public config remains
device independent and the patch can follow multiple TorchTitan revisions.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.tools.logging import logger


ENV_PREFIX = "TORCHTITAN_NPU_PROFILER_"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{ENV_PREFIX + name} must be a boolean, got {value!r}"
    )


def _env_choices(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None:
        return default
    result = tuple(
        part.strip().lower() for part in value.split(",") if part.strip()
    )
    if not result:
        raise ValueError(f"{ENV_PREFIX + name} must not be empty")
    return result


def _env_optional_choices(
    name: str, default: tuple[str, ...] = ()
) -> tuple[str, ...]:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none", "off"}:
        return ()
    return tuple(
        part.strip().lower() for part in value.split(",") if part.strip()
    )


def _env_parse_mode() -> str:
    value = os.environ.get(ENV_PREFIX + "PARSE_MODE")
    if value is not None:
        return value.strip().lower()
    # Preserve the first implementation's public environment contract.
    return "sync" if _env_bool("ONLINE_PARSE", True) else "offline"


def _env_optional_float(name: str) -> float | None:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None or value.strip().lower() in {"", "none", "off"}:
        return None
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{ENV_PREFIX + name} must be a number") from error


def _env_ranks() -> tuple[int, ...] | None:
    value = os.environ.get(ENV_PREFIX + "RANKS", "all").strip().lower()
    if value in {"all", "*", "-1"}:
        return None
    try:
        ranks = tuple(sorted({int(part.strip()) for part in value.split(",")}))
    except ValueError as error:
        raise ValueError(
            f"{ENV_PREFIX}RANKS must be 'all' or comma-separated integers"
        ) from error
    if not ranks or any(rank < 0 for rank in ranks):
        raise ValueError(f"{ENV_PREFIX}RANKS contains an invalid rank: {value!r}")
    return ranks


@dataclass(frozen=True)
class NpuProfilerOptions:
    """NPU-only profiler controls loaded from environment variables."""

    level: str = "level1"
    ranks: tuple[int, ...] | None = None
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_modules: bool = False
    with_flops: bool = False
    export_stacks: bool = False
    export_memory_timeline: bool = False
    parse_mode: str = "sync"
    export_types: tuple[str, ...] = ("text", "db")
    aic_metrics: str = "none"
    l2_cache: bool = False
    op_attr: bool = False
    data_simplification: bool = True
    record_op_args: bool = False
    gc_detect_threshold: float | None = None
    msprof_tx: bool = False
    mstx: bool = False
    mstx_domain_include: tuple[str, ...] = ()
    mstx_domain_exclude: tuple[str, ...] = ()
    host_system: tuple[str, ...] = ()
    system_io: bool = False
    system_interconnection: bool = False

    @classmethod
    def from_environment(cls) -> "NpuProfilerOptions":
        level = os.environ.get(ENV_PREFIX + "LEVEL", "level1").strip().lower()
        aic_metrics = os.environ.get(
            ENV_PREFIX + "AIC_METRICS", "none"
        ).strip().lower()
        options = cls(
            level=level,
            ranks=_env_ranks(),
            record_shapes=_env_bool("RECORD_SHAPES", True),
            profile_memory=_env_bool("PROFILE_MEMORY", True),
            with_stack=_env_bool("WITH_STACK", False),
            with_modules=_env_bool("WITH_MODULES", False),
            with_flops=_env_bool("WITH_FLOPS", False),
            export_stacks=_env_bool("EXPORT_STACKS", False),
            export_memory_timeline=_env_bool(
                "EXPORT_MEMORY_TIMELINE", False
            ),
            parse_mode=_env_parse_mode(),
            export_types=_env_choices("EXPORT_TYPES", ("text", "db")),
            aic_metrics=aic_metrics,
            l2_cache=_env_bool("L2_CACHE", False),
            op_attr=_env_bool("OP_ATTR", False),
            data_simplification=_env_bool("DATA_SIMPLIFICATION", True),
            record_op_args=_env_bool("RECORD_OP_ARGS", False),
            gc_detect_threshold=_env_optional_float("GC_DETECT_THRESHOLD"),
            msprof_tx=_env_bool("MSPROF_TX", False),
            mstx=_env_bool("MSTX", False),
            mstx_domain_include=_env_optional_choices("MSTX_DOMAIN_INCLUDE"),
            mstx_domain_exclude=_env_optional_choices("MSTX_DOMAIN_EXCLUDE"),
            host_system=_env_optional_choices("HOST_SYSTEM"),
            system_io=_env_bool("SYSTEM_IO", False),
            system_interconnection=_env_bool("SYSTEM_INTERCONNECTION", False),
        )
        options.validate()
        return options

    def validate(self) -> None:
        levels = {"level_none", "level0", "level1", "level2"}
        metrics = {
            "none",
            "pipe_utilization",
            "arithmetic_utilization",
            "memory",
            "memory_l0",
            "memory_ub",
            "resource_conflict_ratio",
            "l2_cache",
            "memory_access",
        }
        if self.level not in levels:
            raise ValueError(f"unsupported NPU profiler level: {self.level!r}")
        if self.parse_mode not in {"sync", "async", "offline"}:
            raise ValueError(
                f"unsupported NPU profiler parse mode: {self.parse_mode!r}"
            )
        if self.aic_metrics not in metrics:
            raise ValueError(f"unsupported AIC metric: {self.aic_metrics!r}")
        if (
            self.level in {"level_none", "level0"}
            and self.aic_metrics != "none"
        ):
            raise ValueError(f"{self.level} does not collect AIC metrics")
        unsupported_exports = set(self.export_types) - {"text", "db"}
        if unsupported_exports:
            raise ValueError(
                f"unsupported NPU profiler export types: {sorted(unsupported_exports)}"
            )
        unsupported_host_system = set(self.host_system) - {
            "cpu",
            "mem",
            "disk",
            "network",
            "osrt",
            "numa",
        }
        if unsupported_host_system:
            raise ValueError(
                "unsupported NPU host system collectors: "
                f"{sorted(unsupported_host_system)}"
            )
        if self.gc_detect_threshold is not None and self.gc_detect_threshold < 0:
            raise ValueError("NPU profiler GC threshold must be non-negative")
        if self.mstx_domain_include and self.mstx_domain_exclude:
            raise ValueError(
                "MSTX include and exclude domain filters are mutually exclusive"
            )
        if (
            self.mstx_domain_include or self.mstx_domain_exclude
        ) and not self.mstx:
            raise ValueError("MSTX domain filters require MSTX collection")
        if self.export_stacks and not self.with_stack:
            raise ValueError("NPU stack export requires WITH_STACK=true")
        if self.export_stacks and self.parse_mode != "sync":
            raise ValueError("NPU stack export requires synchronous parsing")
        if self.export_memory_timeline and not self.record_shapes:
            raise ValueError(
                "NPU memory timeline export requires RECORD_SHAPES=true"
            )
        if self.export_memory_timeline and not self.profile_memory:
            raise ValueError(
                "NPU memory timeline export requires PROFILE_MEMORY=true"
            )
        if self.export_memory_timeline and not (
            self.with_stack or self.with_modules
        ):
            raise ValueError(
                "NPU memory timeline export requires WITH_STACK=true or "
                "WITH_MODULES=true"
            )

    @property
    def online_parse(self) -> bool:
        """Return whether parsing is enabled during the training process."""

        return self.parse_mode != "offline"

    def profiles_rank(self, rank: int) -> bool:
        return self.ranks is None or rank in self.ranks


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _torch_npu() -> Any:
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("torch_npu is required for NPU profiling") from error
    return torch_npu


def _resolve_member(namespace: Any, name: str) -> Any:
    """Resolve a snake-case option to a torch_npu enum member."""

    aliases = {
        "level0": "Level0",
        "level1": "Level1",
        "level2": "Level2",
        "level_none": "Level_none",
        "none": "AiCoreNone",
        "pipe_utilization": "PipeUtilization",
        "arithmetic_utilization": "ArithmeticUtilization",
        "memory": "Memory",
        "memory_l0": "MemoryL0",
        "memory_ub": "MemoryUB",
        "resource_conflict_ratio": "ResourceConflictRatio",
        "l2_cache": "L2Cache",
        "memory_access": "MemoryAccess",
        "text": "Text",
        "db": "Db",
        "cpu": "CPU",
        "mem": "MEM",
        "disk": "DISK",
        "network": "NETWORK",
        "osrt": "OSRT",
        "numa": "NUMA",
    }
    return getattr(namespace, aliases[name])


def _experimental_config(torch_npu: Any, options: NpuProfilerOptions) -> Any:
    export_types = [
        _resolve_member(torch_npu.profiler.ExportType, value)
        for value in options.export_types
    ]
    host_system = [
        _resolve_member(torch_npu.profiler.HostSystem, value)
        for value in options.host_system
    ]
    return torch_npu.profiler._ExperimentalConfig(
        export_type=export_types,
        profiler_level=_resolve_member(
            torch_npu.profiler.ProfilerLevel, options.level
        ),
        aic_metrics=_resolve_member(
            torch_npu.profiler.AiCMetrics, options.aic_metrics
        ),
        l2_cache=options.l2_cache,
        msprof_tx=options.msprof_tx,
        op_attr=options.op_attr,
        data_simplification=options.data_simplification,
        record_op_args=options.record_op_args,
        gc_detect_threshold=options.gc_detect_threshold,
        mstx=options.mstx,
        mstx_domain_include=list(options.mstx_domain_include),
        mstx_domain_exclude=list(options.mstx_domain_exclude),
        host_sys=host_system,
        sys_io=options.system_io,
        sys_interconnection=options.system_interconnection,
    )


class NpuMemoryProfiler:
    """Record periodic NPU allocator snapshots using TorchTitan's contract."""

    def __init__(
        self,
        step_num: int,
        freq: int,
        snapshot_dir: str,
        leaf_folder: str,
        rank: int,
        max_entries: int,
    ) -> None:
        torch_npu = _torch_npu()
        torch_npu.npu.memory._record_memory_history(
            stacks="python", max_entries=max_entries
        )
        self.step_num = step_num
        self.freq = freq
        self._snapshot_dir = snapshot_dir
        self._leaf_folder = leaf_folder
        self._rank = rank

    def step(self, exit_ctx: bool = False) -> None:
        self.step_num += 1
        if not exit_ctx and self.step_num % self.freq != 0:
            return
        current_step = self.step_num if not exit_ctx else self.step_num - 1
        directory_name = (
            f"step_{current_step:012d}"
            if not exit_ctx
            else f"step_{current_step:012d}_exit"
        )
        output_directory = os.path.join(
            self._snapshot_dir, directory_name, self._leaf_folder
        )
        os.makedirs(output_directory, exist_ok=True)
        logger.info(f"Dumping NPU memory snapshot at step {current_step}")
        begin = time.monotonic()
        output_file = os.path.join(
            output_directory,
            f"{self._rank:06d}_step_{current_step}.pickle",
        )
        with open(output_file, "wb") as output:
            pickle.dump(torch.npu.memory._snapshot(), output, protocol=4)
        logger.info(
            "Finished dumping NPU memory snapshot in "
            f"{time.monotonic() - begin:.2f} seconds"
        )


def build_torch_profiler_npu(self, *, global_step, base_folder, leaf_folder):
    """Build an Ascend profiler from TorchTitan's profiler configuration."""

    config = self._config
    if not config.enable_profiling:
        return None

    options = NpuProfilerOptions.from_environment()
    rank = _distributed_rank()
    if not options.profiles_rank(rank):
        logger.info(f"NPU profiling disabled for unselected rank {rank}")
        return None

    torch_npu = _torch_npu()
    trace_dir = os.path.join(base_folder, config.save_traces_folder)
    os.makedirs(trace_dir, exist_ok=True)

    warmup = config.profiler_warmup
    active = config.profiler_active
    wait = config.profile_freq - warmup - active
    if wait < 0:
        raise ValueError("profile_freq must be >= profiler_warmup + profiler_active")
    # Preserve torch.profiler.schedule semantics: None means its default of
    # zero (repeat until the profiler context exits). Experiment launchers can
    # request one bounded capture explicitly with profiler_repeat=1.
    repeat = config.profiler_repeat if config.profiler_repeat is not None else 0
    skip_first = (
        config.profiler_skip_first
        if config.profiler_skip_first is not None
        else 0
    )
    skip_first_wait = (
        config.profiler_skip_first_wait
        if config.profiler_skip_first_wait is not None
        else 0
    )

    official_trace_handler = torch_npu.profiler.tensorboard_trace_handler(
        trace_dir,
        worker_name=f"rank_{rank}",
        analyse_flag=options.online_parse,
        async_mode=options.parse_mode == "async",
    )

    def on_trace_ready(profiler: Any) -> None:
        official_trace_handler(profiler)
        step = int(getattr(profiler, "step_num", 0))
        prefix = f"rank_{rank}_step_{step}"
        if options.export_stacks:
            stack_directory = os.path.join(trace_dir, "stacks")
            os.makedirs(stack_directory, exist_ok=True)
            profiler.export_stacks(
                os.path.join(stack_directory, prefix + "_npu_stacks.log"),
                metric="self_npu_time_total",
            )
            profiler.export_stacks(
                os.path.join(stack_directory, prefix + "_cpu_stacks.log"),
                metric="self_cpu_time_total",
            )
        if options.export_memory_timeline:
            memory_directory = os.path.join(trace_dir, "memory_timeline")
            os.makedirs(memory_directory, exist_ok=True)
            device = f"npu:{torch.npu.current_device()}"
            for suffix in (".html", ".json.gz", "_raw.json.gz"):
                profiler.export_memory_timeline(
                    output_path=os.path.join(
                        memory_directory,
                        prefix + "_memory_timeline" + suffix,
                    ),
                    device=device,
                )

    logger.info(
        "NPU profiling active: "
        f"rank={rank}, level={options.level}, aic_metrics={options.aic_metrics}, "
        f"wait={wait}, warmup={warmup}, active={active}, repeat={repeat}, "
        f"skip_first={skip_first}, record_shapes={options.record_shapes}, "
        f"profile_memory={options.profile_memory}, with_stack={options.with_stack}, "
        f"with_modules={options.with_modules}, with_flops={options.with_flops}, "
        f"parse_mode={options.parse_mode}, "
        f"msprof_tx={options.msprof_tx}, mstx={options.mstx}, "
        f"export_stacks={options.export_stacks}, "
        f"export_memory_timeline={options.export_memory_timeline}, "
        f"output={trace_dir}"
    )

    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
            skip_first_wait=skip_first_wait,
        ),
        on_trace_ready=on_trace_ready,
        record_shapes=options.record_shapes,
        profile_memory=options.profile_memory,
        with_stack=options.with_stack,
        with_flops=options.with_flops,
        with_modules=options.with_modules,
        experimental_config=_experimental_config(torch_npu, options),
    )
    profiler.add_metadata("torchtitan_rank", str(rank))
    profiler.add_metadata("torchtitan_profiler_level", options.level)
    profiler.add_metadata("torchtitan_parse_mode", options.parse_mode)
    profiler.__enter__()
    profiler.step_num = global_step
    return profiler


def build_memory_profiler_npu(self, *, global_step, base_folder, leaf_folder):
    """Build the NPU allocator snapshot profiler."""

    config = self._config
    if not config.enable_memory_snapshot:
        return None
    snapshot_dir = os.path.join(base_folder, config.save_memory_snapshot_folder)
    os.makedirs(snapshot_dir, exist_ok=True)
    rank = _distributed_rank()
    snapshot_freq = (
        config.profile_freq
        if config.memory_snapshot_freq is None
        else config.memory_snapshot_freq
    )
    logger.info(f"NPU memory snapshots will be saved at {snapshot_dir}")
    return NpuMemoryProfiler(
        global_step,
        snapshot_freq,
        snapshot_dir,
        leaf_folder,
        rank,
        config.memory_snapshot_max_entries,
    )


def apply_patch() -> None:
    """Install the NPU implementations on TorchTitan's Profiler class."""

    import torchtitan.tools.profiler

    torchtitan.tools.profiler.Profiler.build_torch_profiler = build_torch_profiler_npu
    torchtitan.tools.profiler.Profiler.build_memory_profiler = (
        build_memory_profiler_npu
    )
    logger.info("Patched torchtitan.tools.profiler.Profiler for Ascend NPU")
