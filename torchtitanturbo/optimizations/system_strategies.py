"""Typed prototypes for training-system optimization strategies.

The strategies in this module do not reimplement TorchTitan algorithms. They
validate an experiment request and translate it to TorchTitan's existing typed
configuration surface. Unsupported research directions fail explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ExecutionMode = Literal["eager", "graph_trainer"]
ParallelPlan = Literal["manual", "autoparallel"]
TpSchedule = Literal["sync", "async"]
FsdpOverlap = Literal["default", "ag_rs", "dense_region", "ag_rs_and_dense"]
EpOverlap = Literal["off", "enabled"]
ActivationMemory = Literal[
    "unchanged",
    "default_sac",
    "full_recompute",
    "selective_recompute",
    "cpu_offload",
    "selective_recompute_and_cpu_offload",
]
StateOffload = Literal["none", "cpu"]
CheckpointIo = Literal["unchanged", "sync", "async", "async_pinned"]
AdvancedMemory = Literal["none", "gtp_rematerialization", "nvme"]
InductorCompilation = Literal["unchanged", "regional", "full"]


class SystemStrategyError(ValueError):
    """Raised when a strategy cannot preserve the requested training contract."""


@dataclass(frozen=True, slots=True)
class SystemOptimizationConfig:
    """Select independent placement, scheduling, memory, and I/O strategies."""

    execution_mode: ExecutionMode = "eager"
    parallel_plan: ParallelPlan = "manual"
    tp_schedule: TpSchedule = "sync"
    fsdp_overlap: FsdpOverlap = "default"
    ep_overlap: EpOverlap = "off"
    ep_overlap_chunk_dim: Literal["batch", "seq"] = "batch"
    ep_overlap_module_fqn: Literal["layers.*", "layers.*.moe"] = "layers.*"
    activation_memory: ActivationMemory = "unchanged"
    activation_cpu_budget_gb: float = 100.0
    activation_cpu_prefetch_layers: int = 1
    activation_cpu_defer_layers: int = 1
    state_offload: StateOffload = "none"
    checkpoint_io: CheckpointIo = "unchanged"
    advanced_memory: AdvancedMemory = "none"
    inductor_compilation: InductorCompilation = "unchanged"
    numerics_changing_optim: bool = False
    precompile_artifact_dir: str = ""

    def validate(
        self,
        *,
        tensor_parallel_degree: int = 1,
        data_parallel_shard_degree: int = 1,
        expert_parallel_degree: int = 1,
        pipeline_parallel_degree: int = 1,
    ) -> None:
        """Validate strategy composition before constructing a training command."""
        degrees = {
            "tensor_parallel_degree": tensor_parallel_degree,
            "data_parallel_shard_degree": data_parallel_shard_degree,
            "expert_parallel_degree": expert_parallel_degree,
            "pipeline_parallel_degree": pipeline_parallel_degree,
        }
        for name, degree in degrees.items():
            if degree < 1:
                raise SystemStrategyError(f"{name} must be at least 1, got {degree}.")

        graph_only = {
            "autoparallel": self.parallel_plan == "autoparallel",
            "async TP": self.tp_schedule == "async",
            "FSDP overlap": self.fsdp_overlap != "default",
            "EP overlap": self.ep_overlap != "off",
            "activation memory policy": self.activation_memory != "unchanged",
            "Inductor compilation": self.inductor_compilation != "unchanged",
            "numerics-changing optimization": self.numerics_changing_optim,
            "precompile": bool(self.precompile_artifact_dir),
        }
        requested_graph_features = [name for name, enabled in graph_only.items() if enabled]
        if self.execution_mode != "graph_trainer" and requested_graph_features:
            names = ", ".join(requested_graph_features)
            raise SystemStrategyError(
                f"The current prototypes require graph_trainer for: {names}."
            )

        if self.tp_schedule == "async" and tensor_parallel_degree <= 1:
            raise SystemStrategyError("Async TP requires tensor_parallel_degree > 1.")
        if self.fsdp_overlap != "default" and data_parallel_shard_degree <= 1:
            raise SystemStrategyError("FSDP overlap requires data_parallel_shard_degree > 1.")
        if self.ep_overlap != "off" and expert_parallel_degree <= 1:
            raise SystemStrategyError("EP overlap requires expert_parallel_degree > 1.")
        if self.ep_overlap != "off" and self.fsdp_overlap in {
            "dense_region",
            "ag_rs_and_dense",
        }:
            raise SystemStrategyError(
                "FSDP dense-region overlap does not compose with EP overlap; "
                "TorchTitan would otherwise ignore the dense-region request."
            )
        if (
            self.ep_overlap_chunk_dim == "seq"
            and self.ep_overlap_module_fqn != "layers.*.moe"
        ):
            raise SystemStrategyError(
                "Sequence EP overlap is only valid for layers.*.moe."
            )
        if self.activation_cpu_budget_gb <= 0:
            raise SystemStrategyError("activation_cpu_budget_gb must be positive.")
        if self.activation_cpu_prefetch_layers < 0:
            raise SystemStrategyError(
                "activation_cpu_prefetch_layers must be non-negative."
            )
        if self.activation_cpu_defer_layers < 0:
            raise SystemStrategyError(
                "activation_cpu_defer_layers must be non-negative."
            )
        if self.advanced_memory == "gtp_rematerialization":
            raise SystemStrategyError(
                "GTP rematerialization has no TorchTitan executor yet; use this "
                "value only as a tracked backlog identity."
            )
        if self.advanced_memory == "nvme":
            raise SystemStrategyError(
                "NVMe offload has no PyTorch-native TorchTitan executor yet; it "
                "must not silently import DeepSpeed or fall back to CPU offload."
            )
        if self.precompile_artifact_dir and pipeline_parallel_degree > 1:
            raise SystemStrategyError(
                "GraphPP does not support compile.precompile_artifact_dir in "
                "the current TorchTitan version."
            )

    def torchtitan_overrides(
        self,
        *,
        tensor_parallel_degree: int = 1,
        data_parallel_shard_degree: int = 1,
        expert_parallel_degree: int = 1,
        pipeline_parallel_degree: int = 1,
    ) -> list[str]:
        """Return stable CLI overrides for the supported TorchTitan prototypes."""
        self.validate(
            tensor_parallel_degree=tensor_parallel_degree,
            data_parallel_shard_degree=data_parallel_shard_degree,
            expert_parallel_degree=expert_parallel_degree,
            pipeline_parallel_degree=pipeline_parallel_degree,
        )
        args: list[str] = []
        # GraphTrainer is selected by the recipe's Trainer.Config type. It is
        # not a mode of the regular Trainer and has no --compile.enable or
        # --compile.mode switch. The overrides below are valid only after the
        # caller selects a GraphTrainer recipe.
        if self.parallel_plan == "autoparallel":
            args.append("--compile.enable_autoparallel")
        if self.tp_schedule == "async":
            args.append("--compile.enable_async_tensor_parallel")
        if self.fsdp_overlap in {"ag_rs", "ag_rs_and_dense"}:
            args.append("--compile.enable_fsdp_ag_rs_overlap")
        if self.fsdp_overlap in {"dense_region", "ag_rs_and_dense"}:
            args.append("--compile.enable_fsdp_dense_region_overlap")
        if self.ep_overlap != "off":
            args.extend(
                [
                    "--compile.ep_overlap.enabled",
                    "--compile.ep_overlap.chunk_dim",
                    self.ep_overlap_chunk_dim,
                    "--compile.ep_overlap.module_fqn",
                    self.ep_overlap_module_fqn,
                ]
            )
        if self.inductor_compilation != "unchanged":
            args.extend(
                ["--compile.inductor_compilation", self.inductor_compilation]
            )
        if self.numerics_changing_optim:
            args.append("--compile.numerics_changing_optim")
        if self.precompile_artifact_dir:
            args.extend(
                ["--compile.precompile_artifact_dir", self.precompile_artifact_dir]
            )

        memory_policy = {
            "unchanged": "default",
            "default_sac": "default",
            "full_recompute": "full",
            "selective_recompute": "eager",
            "cpu_offload": "sac_and_offload",
            "selective_recompute_and_cpu_offload": "sac_and_offload",
        }[self.activation_memory]
        if self.execution_mode == "graph_trainer" and self.activation_memory != "unchanged":
            args.extend(["--compile.memory_policy", memory_policy])

        if self.activation_memory in {
            "cpu_offload",
            "selective_recompute_and_cpu_offload",
        }:
            args.extend(
                [
                    "--compile.cpu_offload_budget_gb",
                    str(self.activation_cpu_budget_gb),
                    "--compile.cpu_offload_prefetch_n_layers",
                    str(self.activation_cpu_prefetch_layers),
                    "--compile.cpu_offload_defer_n_layers",
                    str(self.activation_cpu_defer_layers),
                ]
            )
        if self.state_offload == "cpu":
            args.append("--training.enable_cpu_offload")

        checkpoint_mode = {
            "unchanged": "disabled",
            "sync": "disabled",
            "async": "async",
            "async_pinned": "async_with_pinned_mem",
        }[self.checkpoint_io]
        if self.checkpoint_io != "unchanged":
            args.extend(["--checkpointer.async_mode", checkpoint_mode])
        return args

    def recipe_family(self) -> ExecutionMode:
        """Return the Trainer family the experiment must select explicitly."""
        return self.execution_mode

    def provenance(self) -> dict[str, object]:
        """Return the requested strategy identity for manifests and reports."""
        return {
            "schema_version": 1,
            "owner": "torchtitan_existing_implementations",
            "recipe_family": self.recipe_family(),
            "strategies": asdict(self),
            "experimental": True,
        }
