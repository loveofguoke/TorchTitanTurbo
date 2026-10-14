from __future__ import annotations

from unittest.mock import patch

from torchtitanturbo.optimizations.registry import OptimizationProvidersConfig


def test_reference_backends_do_not_require_optional_packages():
    capabilities = OptimizationProvidersConfig().package_capabilities()

    assert len(capabilities) == 5
    assert all(not item.selected for item in capabilities)
    assert all(item.installed is None for item in capabilities)


def test_selected_provider_reports_missing_package_without_importing_it():
    config = OptimizationProvidersConfig(dsa_topk_backend="deepselect")

    with patch(
        "torchtitanturbo.optimizations.registry.find_spec", return_value=None
    ) as mocked_find_spec:
        capabilities = {item.seam: item for item in config.package_capabilities()}

    mocked_find_spec.assert_called_once_with("deep_select")
    assert capabilities["dsa_topk"].selected
    assert capabilities["dsa_topk"].installed is False
    assert "not discoverable" in capabilities["dsa_topk"].reason


def test_package_presence_is_not_reported_as_runtime_acceptance():
    config = OptimizationProvidersConfig(ep_dispatch_backend="deepep_ascend")

    with patch(
        "torchtitanturbo.optimizations.registry.find_spec", return_value=object()
    ):
        capability = {
            item.seam: item for item in config.package_capabilities()
        }["ep_dispatch"]

    assert capability.installed is True
    assert "runtime validation" in capability.reason
