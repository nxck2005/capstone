"""The CPU lane's exclusions are an audited, deliberately closed set."""

from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPECTED = {
    "primary_runtime": {
        ("tests/test_env.py", "test_cuda_build"),
        ("tests/test_env.py", "test_assert_cuda_helper_agrees"),
        ("tests/test_env.py", "test_torch_version_matches_params"),
        ("tests/test_env.py", "test_torchvision_version_matches_params"),
        ("tests/test_env.py", "test_cuda_is_available"),
        ("tests/test_env.py", "test_environment_record_is_fully_populated"),
        ("tests/test_ldpc.py", "test_adapter_clean_high_snr_all_modulations"),
        ("tests/test_g8_bler_runner.py", "test_context_authenticates_registered_candidate_once"),
        ("tests/test_g8_bler_runner.py", "test_full_strength_is_rejected_before_root_or_adapter"),
        ("tests/test_g8_bler_runner.py", "test_bounded_authorization_requires_explicit_fresh_nonproduction_root"),
        ("tests/test_g8_bler_runner.py", "test_shard_bounds_and_unknown_execution_class_are_closed"),
        ("tests/test_g8_bler_runner.py", "test_measurement_is_batch_partition_invariant"),
        ("tests/test_g8_bler_runner.py", "test_measurement_rejects_bad_batch_size"),
        ("tests/test_g8_bler_runner.py", "test_one_bounded_unit_uses_claim_request_result_link_transaction"),
        ("tests/test_g8_bler_runner.py", "test_smoke_record_builder_is_path_and_time_free"),
        ("tests/test_g8_bler_runner.py", "test_candidate_runner_contract_registers_against_isolated_campaign_state"),
        ("tests/test_g8_bler_runner.py", "test_runner_contract_migration_recovers_the_complete_v2_v3_matrix"),
        ("tests/test_g8_bler_runner.py", "test_cached_request_and_result_validation_does_not_reauthenticate_large_artifacts"),
    },
    "external_ldpc_fixture": {
        ("tests/test_ldpc.py", "test_srsran_encoder_and_rate_matched_fixture_exact"),
    },
    "external_dataset": {
        ("tests/test_w4_smoke_runner.py", "test_worklist_is_deterministic_and_free_of_duplicate_identities"),
        ("tests/test_w4_smoke_runner.py", "test_worklist_is_ordered_by_stable_sample_id_not_loader_order"),
        ("tests/test_w4_smoke_runner.py", "test_a_cifar_worklist_row_is_never_task_scored"),
        ("tests/test_w4_smoke_runner.py", "test_every_distinct_cell_has_its_own_config_hash"),
        ("tests/test_w4_smoke_runner.py", "test_the_two_snr_points_do_not_share_a_config_hash"),
        ("tests/test_w4_smoke_runner.py", "test_groups_differing_in_dataset_ratio_modulation_or_rate_differ"),
        ("tests/test_w4_smoke_runner.py", "test_every_row_of_one_cell_shares_that_cell_s_hash"),
        ("tests/test_w4_smoke_runner.py", "test_resolved_selections_must_agree_with_the_plan_row"),
        ("tests/test_w4_smoke_runner.py", "test_archived_run_configs_round_trip_and_reproduce_their_own_hash"),
        ("tests/test_w4_smoke_runner.py", "test_the_index_covers_every_cell_and_names_no_duplicate_hash"),
        ("tests/test_w4_smoke_runner.py", "test_the_root_digest_is_not_usable_as_a_cell_config_hash"),
        ("tests/test_w4_smoke_runner.py", "test_the_root_digest_changes_when_any_cell_changes"),
        *{
            ("tests/test_transparency_bitrate_probe.py", name)
            for name in (
                "test_committed_probe_evidence_verifies",
                "test_missing_or_unexpected_summary_field_fails",
                "test_summary_identity_and_scope_mutations_fail",
                "test_missing_validation_cell_fails",
                "test_duplicate_budget_axis_sample_cell_fails",
                "test_stable_id_outside_validation_manifest_fails",
                "test_per_image_numeric_and_status_mutations_fail",
                "test_accuracy_and_metric_aggregate_mutations_fail",
                "test_best_axis_mutation_fails",
                "test_file_hash_disagreement_fails",
                "test_bootstrap_result_mutation_fails",
                "test_threshold_forecast_mutation_fails",
                "test_cache_manifest_missing_entry_fails",
                "test_cache_manifest_root_mutation_fails",
                "test_unreachable_implementation_commit_fails",
                "test_unreachable_measurement_commit_fails",
                "test_wrong_commit_ancestry_fails",
                "test_critical_source_changed_between_a_and_b_fails",
                "test_execution_source_identity_mutations_fail",
                "test_missing_or_unexpected_execution_source_fails",
                "test_shards_from_differing_commits_fail",
                "test_tracked_cache_or_codestream_is_rejected",
                "test_codec_binding_accepts_the_declared_am80_drift",
                "test_codec_binding_accepts_no_drift_without_a_record",
                "test_codec_binding_rejects_a_stale_record_when_nothing_drifted",
                "test_codec_binding_rejects_undeclared_drift_without_a_record",
                "test_codec_binding_rejects_an_archive_that_fails_its_own_hash",
                "test_codec_binding_rejects_any_additional_codec_drift",
                "test_codec_binding_rejects_a_change_to_the_probes_own_axis_ladder",
                "test_codec_binding_rejects_a_defective_record",
            )
        },
    },
    "frozen_checkpoint": {
        ("tests/test_frozen_reference_classifier.py", "test_exact_checkpoint_is_accepted_and_frozen"),
        ("tests/test_frozen_reference_classifier.py", "test_wrong_checkpoint_hash_is_rejected"),
        ("tests/test_frozen_reference_classifier.py", "test_wrong_config_or_manifest_identity_is_rejected"),
        ("tests/test_frozen_reference_classifier.py", "test_missing_or_extra_state_dict_key_is_rejected"),
        ("tests/test_frozen_reference_classifier.py", "test_smoke_or_non_g1_checkpoint_is_rejected"),
        ("tests/test_frozen_reference_classifier.py", "test_incomplete_full_run_lineage_is_rejected"),
    },
}


def _marker_names(decorator: ast.expr) -> set[str]:
    if not isinstance(decorator, ast.Attribute) or decorator.attr not in EXPECTED:
        return set()
    value = decorator.value
    if not isinstance(value, ast.Attribute) or value.attr != "mark":
        return set()
    if not isinstance(value.value, ast.Name) or value.value.id != "pytest":
        return set()
    return {decorator.attr}


def _marked_functions() -> dict[str, set[tuple[str, str]]]:
    found = {name: set() for name in EXPECTED}
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = str(path.relative_to(REPO))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                for marker in _marker_names(decorator):
                    found[marker].add((relative, node.name))
    return found


def test_cpu_exclusion_allowlist_is_exact():
    assert _marked_functions() == EXPECTED
