"""Render validated TorchTitan overrides for system optimization experiments."""

from __future__ import annotations

import argparse
import json
import shlex

from .system_strategies import SystemOptimizationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-mode", choices=("eager", "graph_trainer"), default="eager"
    )
    parser.add_argument(
        "--parallel-plan", choices=("manual", "autoparallel"), default="manual"
    )
    parser.add_argument("--tp-schedule", choices=("sync", "async"), default="sync")
    parser.add_argument(
        "--fsdp-overlap",
        choices=("default", "ag_rs", "dense_region", "ag_rs_and_dense"),
        default="default",
    )
    parser.add_argument("--ep-overlap", choices=("off", "enabled"), default="off")
    parser.add_argument(
        "--ep-overlap-chunk-dim", choices=("batch", "seq"), default="batch"
    )
    parser.add_argument(
        "--ep-overlap-module-fqn",
        choices=("layers.*", "layers.*.moe"),
        default="layers.*",
    )
    parser.add_argument(
        "--activation-memory",
        choices=(
            "unchanged",
            "default_sac",
            "full_recompute",
            "selective_recompute",
            "cpu_offload",
            "selective_recompute_and_cpu_offload",
        ),
        default="unchanged",
    )
    parser.add_argument("--activation-cpu-budget-gb", type=float, default=100.0)
    parser.add_argument("--activation-cpu-prefetch-layers", type=int, default=1)
    parser.add_argument("--activation-cpu-defer-layers", type=int, default=1)
    parser.add_argument("--state-offload", choices=("none", "cpu"), default="none")
    parser.add_argument(
        "--checkpoint-io",
        choices=("unchanged", "sync", "async", "async_pinned"),
        default="unchanged",
    )
    parser.add_argument(
        "--inductor-compilation",
        choices=("unchanged", "regional", "full"),
        default="unchanged",
    )
    parser.add_argument("--numerics-changing-optim", action="store_true")
    parser.add_argument("--precompile-artifact-dir", default="")
    parser.add_argument("--tensor-parallel-degree", type=int, default=1)
    parser.add_argument("--data-parallel-shard-degree", type=int, default=1)
    parser.add_argument("--expert-parallel-degree", type=int, default=1)
    parser.add_argument("--pipeline-parallel-degree", type=int, default=1)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    return parser


def render(args: argparse.Namespace) -> str:
    config = SystemOptimizationConfig(
        execution_mode=args.execution_mode,
        parallel_plan=args.parallel_plan,
        tp_schedule=args.tp_schedule,
        fsdp_overlap=args.fsdp_overlap,
        ep_overlap=args.ep_overlap,
        ep_overlap_chunk_dim=args.ep_overlap_chunk_dim,
        ep_overlap_module_fqn=args.ep_overlap_module_fqn,
        activation_memory=args.activation_memory,
        activation_cpu_budget_gb=args.activation_cpu_budget_gb,
        activation_cpu_prefetch_layers=args.activation_cpu_prefetch_layers,
        activation_cpu_defer_layers=args.activation_cpu_defer_layers,
        state_offload=args.state_offload,
        checkpoint_io=args.checkpoint_io,
        inductor_compilation=args.inductor_compilation,
        numerics_changing_optim=args.numerics_changing_optim,
        precompile_artifact_dir=args.precompile_artifact_dir,
    )
    overrides = config.torchtitan_overrides(
        tensor_parallel_degree=args.tensor_parallel_degree,
        data_parallel_shard_degree=args.data_parallel_shard_degree,
        expert_parallel_degree=args.expert_parallel_degree,
        pipeline_parallel_degree=args.pipeline_parallel_degree,
    )
    if args.format == "shell":
        return shlex.join(overrides)
    return json.dumps(
        {
            "recipe_family": config.recipe_family(),
            "overrides": overrides,
            "provenance": config.provenance(),
        },
        indent=2,
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(render(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
