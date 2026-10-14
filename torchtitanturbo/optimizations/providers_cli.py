"""Inspect default-off optimization provider selections before an experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .registry import OptimizationProvidersConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsa-topk-backend", choices=("torch", "deepselect"), default="torch"
    )
    parser.add_argument(
        "--moe-topk-backend", choices=("torch", "tilekernels"), default="torch"
    )
    parser.add_argument(
        "--moe-gemm-backend",
        choices=("torch", "deepgemm_ascend"),
        default="torch",
    )
    parser.add_argument(
        "--ep-dispatch-backend",
        choices=("torchao", "deepep_ascend"),
        default="torchao",
    )
    parser.add_argument(
        "--sparse-attention-backend",
        choices=("flex_attention", "flashmla_ascend"),
        default="flex_attention",
    )
    return parser


def render(args: argparse.Namespace) -> str:
    config = OptimizationProvidersConfig(
        dsa_topk_backend=args.dsa_topk_backend,
        moe_topk_backend=args.moe_topk_backend,
        moe_gemm_backend=args.moe_gemm_backend,
        ep_dispatch_backend=args.ep_dispatch_backend,
        sparse_attention_backend=args.sparse_attention_backend,
    )
    return json.dumps(
        {
            "schema_version": 1,
            "selected": config.selected_names(),
            "package_capabilities": [
                asdict(item) for item in config.package_capabilities()
            ],
            "runtime_accepted": False,
            "next_gate": "provider-specific runtime validation",
        },
        indent=2,
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    print(render(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
