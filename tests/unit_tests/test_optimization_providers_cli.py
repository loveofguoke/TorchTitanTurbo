from __future__ import annotations

import json
from unittest.mock import patch

from torchtitanturbo.optimizations.providers_cli import build_parser, render


def test_default_provider_cli_keeps_reference_backends():
    payload = json.loads(render(build_parser().parse_args([])))

    assert payload["selected"]["dsa_topk"] == "torch"
    assert payload["selected"]["ep_dispatch"] == "torchao"
    assert payload["runtime_accepted"] is False
    assert all(
        not item["selected"] for item in payload["package_capabilities"]
    )


def test_selected_provider_cli_preserves_missing_dependency_evidence():
    args = build_parser().parse_args(
        ["--ep-dispatch-backend", "deepep_ascend"]
    )
    with patch(
        "torchtitanturbo.optimizations.registry.find_spec", return_value=None
    ):
        payload = json.loads(render(args))

    capability = next(
        item
        for item in payload["package_capabilities"]
        if item["seam"] == "ep_dispatch"
    )
    assert capability["package"] == "deep_ep"
    assert capability["installed"] is False
    assert payload["runtime_accepted"] is False
