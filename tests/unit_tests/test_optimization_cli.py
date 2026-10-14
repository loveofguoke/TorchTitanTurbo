from __future__ import annotations

import json

import pytest

from torchtitanturbo.optimizations.cli import build_parser, render
from torchtitanturbo.optimizations.system_strategies import SystemStrategyError


def test_default_cli_is_a_true_noop():
    payload = json.loads(render(build_parser().parse_args([])))

    assert payload["recipe_family"] == "eager"
    assert payload["overrides"] == []


def test_graph_cli_emits_current_torchtitan_fields():
    args = build_parser().parse_args(
        [
            "--execution-mode", "graph_trainer",
            "--tensor-parallel-degree", "2",
            "--tp-schedule", "async",
            "--inductor-compilation", "full",
            "--checkpoint-io", "async",
            "--format", "shell",
        ]
    )

    command = render(args)

    assert "--compile.enable_async_tensor_parallel" in command
    assert "--compile.inductor_compilation full" in command
    assert "--checkpointer.async_mode async" in command


def test_cli_rejects_unsupported_graphpp_precompile_combination():
    args = build_parser().parse_args(
        [
            "--execution-mode", "graph_trainer",
            "--pipeline-parallel-degree", "2",
            "--precompile-artifact-dir", "artifacts/graph-trainer",
        ]
    )

    with pytest.raises(SystemStrategyError, match="GraphPP does not support"):
        render(args)
