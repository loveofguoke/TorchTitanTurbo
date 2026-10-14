import pytest

from torchtitanturbo.optimizations.system_strategies import (
    SystemOptimizationConfig,
    SystemStrategyError,
)


def test_combined_experiment_config_keeps_reference_defaults():
    from torchtitanturbo.optimizations.registry import OptimizationExperimentConfig

    provenance = OptimizationExperimentConfig().provenance()

    assert provenance["providers"]["moe_gemm"] == "torch"
    assert provenance["system"]["strategies"]["parallel_plan"] == "manual"


def test_reference_strategy_is_default_off():
    config = SystemOptimizationConfig()

    assert config.torchtitan_overrides() == []
    assert config.recipe_family() == "eager"
    assert config.provenance()["strategies"]["parallel_plan"] == "manual"


def test_graph_strategy_translates_to_existing_torchtitan_flags():
    config = SystemOptimizationConfig(
        execution_mode="graph_trainer",
        parallel_plan="autoparallel",
        tp_schedule="async",
        fsdp_overlap="ag_rs",
        ep_overlap="enabled",
        ep_overlap_module_fqn="layers.*.moe",
        activation_memory="selective_recompute_and_cpu_offload",
        activation_cpu_budget_gb=24.0,
        state_offload="cpu",
        checkpoint_io="async_pinned",
    )

    args = config.torchtitan_overrides(
        tensor_parallel_degree=2,
        data_parallel_shard_degree=2,
        expert_parallel_degree=2,
    )

    expected_flags = {
        "--compile.enable_autoparallel",
        "--compile.enable_async_tensor_parallel",
        "--compile.enable_fsdp_ag_rs_overlap",
        "--compile.ep_overlap.enabled",
        "--training.enable_cpu_offload",
    }
    assert expected_flags.issubset(args)
    assert "--compile.enable" not in args
    assert "--compile.mode" not in args
    assert "--compile.ep_overlap.strategy" not in args
    assert config.recipe_family() == "graph_trainer"
    assert args[args.index("--compile.memory_policy") + 1] == "sac_and_offload"
    assert args[args.index("--checkpointer.async_mode") + 1] == (
        "async_with_pinned_mem"
    )


@pytest.mark.parametrize(
    ("config", "degrees", "message"),
    [
        (
            SystemOptimizationConfig(parallel_plan="autoparallel"),
            {},
            "require graph_trainer",
        ),
        (
            SystemOptimizationConfig(activation_memory="full_recompute"),
            {},
            "require graph_trainer",
        ),
        (
            SystemOptimizationConfig(
                execution_mode="graph_trainer", tp_schedule="async"
            ),
            {},
            "tensor_parallel_degree",
        ),
        (
            SystemOptimizationConfig(
                execution_mode="graph_trainer", fsdp_overlap="ag_rs"
            ),
            {},
            "data_parallel_shard_degree",
        ),
        (
            SystemOptimizationConfig(
                execution_mode="graph_trainer", ep_overlap="enabled"
            ),
            {},
            "expert_parallel_degree",
        ),
        (
            SystemOptimizationConfig(advanced_memory="gtp_rematerialization"),
            {},
            "no TorchTitan executor",
        ),
        (
            SystemOptimizationConfig(advanced_memory="nvme"),
            {},
            "no PyTorch-native TorchTitan executor",
        ),
    ],
)
def test_invalid_or_unimplemented_strategy_fails_explicitly(
    config, degrees, message
):
    with pytest.raises(SystemStrategyError, match=message):
        config.validate(**degrees)


def test_sequence_ep_overlap_requires_moe_scope():
    config = SystemOptimizationConfig(
        execution_mode="graph_trainer",
        ep_overlap="enabled",
        ep_overlap_chunk_dim="seq",
        ep_overlap_module_fqn="layers.*",
    )

    with pytest.raises(SystemStrategyError, match=r"layers\.\*\.moe"):
        config.validate(expert_parallel_degree=2)


def test_dense_fsdp_overlap_rejects_ep_overlap_instead_of_silent_skip():
    config = SystemOptimizationConfig(
        execution_mode="graph_trainer",
        fsdp_overlap="dense_region",
        ep_overlap="enabled",
    )

    with pytest.raises(SystemStrategyError, match="does not compose"):
        config.validate(data_parallel_shard_degree=2, expert_parallel_degree=2)


def test_graph_compilation_and_precompile_use_current_official_fields():
    config = SystemOptimizationConfig(
        execution_mode="graph_trainer",
        inductor_compilation="full",
        numerics_changing_optim=True,
        precompile_artifact_dir="artifacts/graph-trainer",
    )

    args = config.torchtitan_overrides()

    assert args[args.index("--compile.inductor_compilation") + 1] == "full"
    assert "--compile.numerics_changing_optim" in args
    assert args[args.index("--compile.precompile_artifact_dir") + 1] == (
        "artifacts/graph-trainer"
    )


def test_graphpp_rejects_unsupported_precompile_artifact():
    config = SystemOptimizationConfig(
        execution_mode="graph_trainer",
        precompile_artifact_dir="artifacts/graph-trainer",
    )

    with pytest.raises(SystemStrategyError, match="GraphPP does not support"):
        config.torchtitan_overrides(pipeline_parallel_degree=2)
