"""AM-87 G8_F corpus-plan derivation and fail-closed mutation coverage."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from baseline import g8_f_corpus_plan as plan
from baseline import g8_pascal_production as production
from baseline import g8_e_corrected_v3s as v3s


@pytest.fixture(scope="module")
def committed_plan() -> dict:
    # AM-87 is immutable historical support evidence after AM-88 superseded only
    # its Cartesian execution multiplicity.  Structural verification remains
    # exact; current-parameter rebuilding belongs to the additive AM-88 plan.
    value = json.loads(plan.PLAN_PATH.read_bytes())
    return plan.verify_plan_value(value, expected=value)


def _reid(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("plan_id", None)
    value["plan_id"] = plan.PLAN_PREFIX + plan.sha256_bytes(plan.canonical_json(value))
    return value


def _reject(value: dict, expected: dict) -> None:
    with pytest.raises(plan.G8FCorpusPlanError):
        plan.verify_plan_value(_reid(value), expected=expected)


def test_committed_plan_is_plan_only_and_has_exact_breadth(committed_plan: dict) -> None:
    assert committed_plan["artifact_quality_projection"]["quality_count"] == 120  # literal-ok: frozen AM-87 quality count
    assert committed_plan["training_membership"]["stable_id_count"] == 8469  # literal-ok: frozen train-manifest count
    assert committed_plan["multiplicity_and_feasibility"]["exact_attempt_count"] == 1_016_280  # literal-ok: exact 120 x 8469 plan
    boundary = committed_plan["protected_boundary"]
    assert boundary["plan_only"] is True
    assert boundary["f0_execution_authorized"] is False
    assert boundary["corpus_materialized"] is False
    assert boundary["materialized_object_count"] == 0
    assert boundary["optimizer_steps"] == boundary["pass_two"] == boundary["test_access"] == 0
    assert boundary["confessor_started"] is False


def test_projection_deduplicates_real_phy_aliases(committed_plan: dict) -> None:
    aliases = [
        row for row in committed_plan["artifact_quality_projection"]["qualities"]
        if row["source_structural_identity_count"] > 1
    ]
    assert aliases
    assert any(
        len({(item["ratio"], item["modulation"], item["ldpc_rate"]) for item in row["source_structural_identities"]}) > 1
        for row in aliases
    )


def _candidate_and_structural() -> tuple[dict, dict]:
    candidate = {
        "candidate_id": "candidate-a",
        "composition_candidate_identity": "composition-a",
        "dataset": "imagenette160",
        "source_codec": "jpeg2000",
        "ratio": "r_1_6",
        "encode_axis_px": 96,  # literal-ok: configured Imagenette axis fixture
        "modulation": "qpsk",
        "ldpc_rate": "1/2",
        "snr_db": 7,  # literal-ok: configured training-SNR fixture
        "packet_config_id": "packet-a",
    }
    structural = {
        "dataset": candidate["dataset"],
        "source_codec": candidate["source_codec"],
        "ratio": candidate["ratio"],
        "encode_axis_px": candidate["encode_axis_px"],
        "modulation": candidate["modulation"],
        "ldpc_rate": candidate["ldpc_rate"],
        "packet_config_id": candidate["packet_config_id"],
        "payload_budget_bytes": 1063,  # literal-ok: frozen authority budget fixture
        "packet_accounting": {
            "payload_bytes": 1063,  # literal-ok: same frozen authority budget fixture
            "reconciles": True,
            "channel_reconciles": True,
            "channel_uses_exact": True,
        },
    }
    return candidate, structural


def test_phy_only_fields_cannot_create_artificial_quality_multiplicity(committed_plan: dict) -> None:
    candidate, structural = _candidate_and_structural()
    codec_id = committed_plan["frozen_bindings"]["g8d_codec_configuration"]["codec_configuration_id"]
    original = plan.project_artifact_quality(candidate, structural, codec_configuration_id=codec_id)
    changed_candidate = dict(candidate)
    changed_structural = dict(structural)
    changed_candidate.update(
        candidate_id="candidate-b",
        composition_candidate_identity="composition-b",
        ratio="r_1_3",
        snr_db=-8,  # literal-ok: configured grid endpoint fixture
        modulation="qam16",
        ldpc_rate="5/6",
        packet_config_id="packet-b",
    )
    changed_structural.update(
        ratio="r_1_3", modulation="qam16", ldpc_rate="5/6", packet_config_id="packet-b"
    )
    changed = plan.project_artifact_quality(changed_candidate, changed_structural, codec_configuration_id=codec_id)
    assert changed == original
    assert plan.quality_id(changed) == plan.quality_id(original)


@pytest.mark.parametrize("field,value", [("payload_budget_bytes", 1064), ("encode_axis_px", 128)])  # literal-ok: identity-mutation fixtures
def test_every_direct_artifact_field_changes_quality_identity(committed_plan: dict, field: str, value: int) -> None:
    candidate, structural = _candidate_and_structural()
    codec_id = committed_plan["frozen_bindings"]["g8d_codec_configuration"]["codec_configuration_id"]
    first = plan.project_artifact_quality(candidate, structural, codec_configuration_id=codec_id)
    if field == "payload_budget_bytes":
        structural[field] = value
        structural["packet_accounting"] = dict(structural["packet_accounting"], payload_bytes=value)
    else:
        candidate[field] = value
        structural[field] = value
    second = plan.project_artifact_quality(candidate, structural, codec_configuration_id=codec_id)
    assert plan.quality_id(first) != plan.quality_id(second)


def test_adding_foreign_quality_is_rejected(committed_plan: dict) -> None:
    mutated = copy.deepcopy(committed_plan)
    row = copy.deepcopy(mutated["artifact_quality_projection"]["qualities"][0])
    row["identity"]["payload_budget_bytes"] += 1
    row["quality_id"] = plan.quality_id(row["identity"])
    mutated["artifact_quality_projection"]["qualities"].append(row)
    mutated["artifact_quality_projection"]["qualities"].sort(key=lambda item: item["quality_id"])
    _reject(mutated, committed_plan)


def test_removing_expected_quality_is_rejected(committed_plan: dict) -> None:
    mutated = copy.deepcopy(committed_plan)
    mutated["artifact_quality_projection"]["qualities"].pop()
    _reject(mutated, committed_plan)


def test_reordering_qualities_is_rejected(committed_plan: dict) -> None:
    mutated = copy.deepcopy(committed_plan)
    mutated["artifact_quality_projection"]["qualities"][0:2] = reversed(
        mutated["artifact_quality_projection"]["qualities"][0:2]
    )
    _reject(mutated, committed_plan)


def test_duplicate_artifact_identity_is_rejected(committed_plan: dict) -> None:
    mutated = copy.deepcopy(committed_plan)
    mutated["artifact_quality_projection"]["qualities"][1] = copy.deepcopy(
        mutated["artifact_quality_projection"]["qualities"][0]
    )
    _reject(mutated, committed_plan)


def test_changing_projected_identity_is_rejected(committed_plan: dict) -> None:
    mutated = copy.deepcopy(committed_plan)
    mutated["artifact_quality_projection"]["qualities"][0]["identity"]["encode_axis_px"] += 1
    mutated["artifact_quality_projection"]["qualities"][0]["quality_id"] = plan.quality_id(
        mutated["artifact_quality_projection"]["qualities"][0]["identity"]
    )
    _reject(mutated, committed_plan)


def _stable_id_for_split(split: str) -> str:
    with (plan.REPO_ROOT / "data/manifests/imagenette160.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == split:
                return row["stable_sample_id"]
    raise AssertionError(f"manifest has no {split} row")


@pytest.mark.parametrize("split", ["val", "test"])
def test_validation_or_test_stable_id_is_rejected(committed_plan: dict, split: str) -> None:
    mutated = copy.deepcopy(committed_plan)
    mutated["training_membership"]["stable_ids"][0] = _stable_id_for_split(split)
    mutated["training_membership"]["stable_ids"].sort()
    _reject(mutated, committed_plan)


def test_changed_feasibility_classification_is_rejected(committed_plan: dict) -> None:
    mutated = copy.deepcopy(committed_plan)
    audit = mutated["artifact_quality_projection"]["qualities"][0]["validation_feasibility_audit"]
    audit["classification"] = "foreign_feasibility_class"
    _reject(mutated, committed_plan)


def test_changed_corpus_multiplicity_is_rejected(committed_plan: dict) -> None:
    mutated = copy.deepcopy(committed_plan)
    mutated["multiplicity_and_feasibility"]["exact_attempt_count"] += 1
    _reject(mutated, committed_plan)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("e7_handoff", "handoff_id", "g8ee7handoff-" + "0" * 64),  # literal-ok: digest mutation
        ("pass_one", "state_id", "g8epassone-" + "0" * 64),  # literal-ok: digest mutation
        ("specification", "sha256", "0" * 64),  # literal-ok: digest mutation
    ],
)
def test_changed_e7_pass_one_or_spec_binding_is_rejected(
    committed_plan: dict, section: str, field: str, value: str
) -> None:
    mutated = copy.deepcopy(committed_plan)
    if section == "specification":
        mutated["amendment"][section][field] = value
    else:
        mutated["frozen_bindings"][section][field] = value
    _reject(mutated, committed_plan)


def test_am87_post_campaign_compatibility_preserves_frozen_g8c_verification() -> None:
    verified = production.validate_production_contracts()
    assert verified["campaign_id"].startswith("g8p-")


@pytest.mark.parametrize("mutation", ["foreign_parameter_path", "current_sha", "protected_counter"])
def test_am87_post_campaign_compatibility_mutations_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    value = json.loads(production.POST_CAMPAIGN_SOURCE_COMPATIBILITY.read_bytes())
    if mutation == "foreign_parameter_path":
        value["allowed_parameter_paths"].append("baseline.ldpc_max_iters")
        value["allowed_parameter_paths"].sort()
    elif mutation == "current_sha":
        value["entries"][0]["current_sha256"] = "0" * 64
    else:
        value["protected_boundary"]["training"] = 1
    body = {key: child for key, child in value.items() if key != "compatibility_id"}
    value["compatibility_id"] = "g8postsource-" + production.sha256_bytes(
        production.canonical_json(body)
    )
    path = production.POST_CAMPAIGN_SOURCE_COMPATIBILITY.parent / (
        f".test-{tmp_path.name}-{mutation}-compatibility.json"
    )
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    monkeypatch.setattr(production, "POST_CAMPAIGN_SOURCE_COMPATIBILITY", path)
    try:
        with pytest.raises(production.ProductionContractError):
            production.validate_production_contracts()
    finally:
        path.unlink(missing_ok=True)


def test_am87_g8e_source_compatibility_mutations_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = json.loads(v3s.V3S_SOURCE_MANIFEST_PATH.read_bytes())
    v3s.validate_source_manifest(source)
    compatibility = json.loads(v3s.AM87_SOURCE_COMPATIBILITY_PATH.read_bytes())
    compatibility["entries"][0]["current_sha256"] = "0" * 64
    body = {key: child for key, child in compatibility.items() if key != "compatibility_id"}
    compatibility["compatibility_id"] = "g8esourcecompat-" + v3s.sha256_bytes(
        v3s.canonical_json(body)
    )
    path = v3s.AM87_SOURCE_COMPATIBILITY_PATH.parent / (
        f".test-{tmp_path.name}-g8e-source-compatibility.json"
    )
    path.write_text(json.dumps(compatibility, indent=2, sort_keys=True) + "\n", encoding="ascii")
    monkeypatch.setattr(v3s, "AM87_SOURCE_COMPATIBILITY_PATH", path)
    try:
        with pytest.raises(v3s.G8EV3SError):
            v3s.validate_source_manifest(source)
    finally:
        path.unlink(missing_ok=True)


def test_plan_never_imports_or_calls_test_access_boundary() -> None:
    source = Path(plan.__file__).read_text(encoding="utf-8")
    assert "data.test_access" not in source
    assert "canonicalize_source" not in source
    assert "encode_to_budget(" not in source
