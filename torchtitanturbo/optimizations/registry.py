"""Typed selection for default-off optimization providers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

from .deepgemm import DeepGemmGroupedBf16Provider
from .deepep import DeepEpAscendBufferProvider
from .deepselect import DeepSelectTopKProvider
from .flashmla import FlashMlaAscendSparseProvider
from .tilekernels import TileKernelsMoeTopKProvider
from .system_strategies import SystemOptimizationConfig


DsaTopKBackend = Literal["torch", "deepselect"]
MoeTopKBackend = Literal["torch", "tilekernels"]
MoeGemmBackend = Literal["torch", "deepgemm_ascend"]
EpDispatchBackend = Literal["torchao", "deepep_ascend"]
SparseAttentionBackend = Literal[
    "flex_attention",
    "flashmla_ascend",
    "triton_dsa_training",
]


@dataclass(frozen=True, slots=True)
class OverrideProviderSelection:
    """A provider selected through TorchTitan's explicit override mechanism."""

    name: str
    override_import: str
    training_forward: bool
    training_backward: bool
    runtime_accepted: bool = False


@dataclass(frozen=True, slots=True)
class ProviderPackageCapability:
    """Static package-presence evidence; never a runtime acceptance claim."""

    seam: str
    backend: str
    package: str | None
    selected: bool
    installed: bool | None
    reason: str


@dataclass(frozen=True, slots=True)
class OptimizationProvidersConfig:
    """Select one explicit implementation at each replaceable model seam."""

    dsa_topk_backend: DsaTopKBackend = "torch"
    moe_topk_backend: MoeTopKBackend = "torch"
    moe_gemm_backend: MoeGemmBackend = "torch"
    ep_dispatch_backend: EpDispatchBackend = "torchao"
    sparse_attention_backend: SparseAttentionBackend = "flex_attention"

    def __post_init__(self) -> None:
        allowed = {
            "dsa_topk_backend": {"torch", "deepselect"},
            "moe_topk_backend": {"torch", "tilekernels"},
            "moe_gemm_backend": {"torch", "deepgemm_ascend"},
            "ep_dispatch_backend": {"torchao", "deepep_ascend"},
            "sparse_attention_backend": {
                "flex_attention",
                "flashmla_ascend",
                "triton_dsa_training",
            },
        }
        for field_name, valid_values in allowed.items():
            value = getattr(self, field_name)
            if value not in valid_values:
                choices = ", ".join(sorted(valid_values))
                raise ValueError(
                    f"Unsupported {field_name}={value!r}; choose one of: {choices}."
                )

    def build_selected(self) -> dict[str, object]:
        """Construct selected providers without importing their libraries."""
        providers: dict[str, object] = {}
        if self.dsa_topk_backend == "deepselect":
            providers["dsa_topk"] = DeepSelectTopKProvider()
        if self.moe_topk_backend == "tilekernels":
            providers["moe_topk"] = TileKernelsMoeTopKProvider()
        if self.moe_gemm_backend == "deepgemm_ascend":
            providers["moe_gemm"] = DeepGemmGroupedBf16Provider()
        if self.ep_dispatch_backend == "deepep_ascend":
            providers["ep_dispatch"] = DeepEpAscendBufferProvider()
        if self.sparse_attention_backend == "flashmla_ascend":
            providers["sparse_attention"] = FlashMlaAscendSparseProvider()
        elif self.sparse_attention_backend == "triton_dsa_training":
            providers["sparse_attention"] = OverrideProviderSelection(
                name="triton_dsa_training",
                override_import=(
                    "torchtitanturbo.models.glm5.ops.triton_dsa."
                    "triton_dsa_training"
                ),
                training_forward=True,
                training_backward=True,
            )
        return providers

    def selected_names(self) -> dict[str, str]:
        """Return stable experiment identity without loading providers."""
        return {
            "dsa_topk": self.dsa_topk_backend,
            "moe_topk": self.moe_topk_backend,
            "moe_gemm": self.moe_gemm_backend,
            "ep_dispatch": self.ep_dispatch_backend,
            "sparse_attention": self.sparse_attention_backend,
        }

    def package_capabilities(self) -> tuple[ProviderPackageCapability, ...]:
        """Inspect optional package presence without importing device runtimes."""
        defaults = {
            "dsa_topk": "torch",
            "moe_topk": "torch",
            "moe_gemm": "torch",
            "ep_dispatch": "torchao",
            "sparse_attention": "flex_attention",
        }
        packages = {
            ("dsa_topk", "deepselect"): "deep_select",
            ("moe_topk", "tilekernels"): "tile_kernels",
            ("moe_gemm", "deepgemm_ascend"): "deep_gemm",
            ("ep_dispatch", "deepep_ascend"): "deep_ep",
            ("sparse_attention", "flashmla_ascend"): "flash_mla",
            ("sparse_attention", "triton_dsa_training"): "triton",
        }
        capabilities: list[ProviderPackageCapability] = []
        for seam, backend in self.selected_names().items():
            selected = backend != defaults[seam]
            package = packages.get((seam, backend))
            if not selected:
                capabilities.append(
                    ProviderPackageCapability(
                        seam,
                        backend,
                        None,
                        False,
                        None,
                        "Reference backend selected; no optional package required.",
                    )
                )
                continue
            installed = find_spec(package) is not None
            capabilities.append(
                ProviderPackageCapability(
                    seam,
                    backend,
                    package,
                    True,
                    installed,
                    (
                        "Package is discoverable; dtype, device, shape, backward, "
                        "and graph contracts still require runtime validation."
                        if installed
                        else f"Required optional package {package!r} is not discoverable."
                    ),
                )
            )
        return tuple(capabilities)


@dataclass(frozen=True, slots=True)
class OptimizationExperimentConfig:
    """Compose operator providers with whole-training system strategies."""

    providers: OptimizationProvidersConfig = OptimizationProvidersConfig()
    system: SystemOptimizationConfig = SystemOptimizationConfig()

    def provenance(self) -> dict[str, object]:
        """Return one manifest payload without importing optional libraries."""
        return {
            "schema_version": 1,
            "providers": self.providers.selected_names(),
            "system": self.system.provenance(),
        }
