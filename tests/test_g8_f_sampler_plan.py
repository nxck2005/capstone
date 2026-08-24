"""Focused AM-88 balanced-sampler invariants and fail-closed mutations."""

from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from baseline import g8_f_sampler_plan as sampler
from baseline import g8_pascal_production as production
from baseline import g8_e_corrected_v3s as v3s
import verify_g8_f_sampler_plan as independent


@pytest.fixture(scope="module")
def committed() -> dict:
    return independent.verify_sampler_plan()


def _reid(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("plan_id", None)
    value["plan_id"] = sampler.PLAN_PREFIX + sampler.sha256_bytes(sampler.canonical_json(value))
    return value


def _reject(tmp_path: Path, value: dict) -> None:
    path = tmp_path / "sampler.json"
    path.write_bytes(sampler.rendered_json(_reid(value)))
    with pytest.raises(independent.G8FSamplerVerificationError):
        independent.verify_sampler_plan(path)


def _pairs() -> tuple[dict, list[tuple[str, str]], dict[str, int]]:
    support, quality_ids, _rows = sampler._load_am87_support()  # noqa: SLF001
    ids, labels, ids_by_class, _splits = sampler._training_membership()  # noqa: SLF001
    assert ids == support["training_membership"]["stable_ids"]
    pairs, _cycle = sampler.derive_assignments(
        quality_ids, ids_by_class, seed=sampler.EXPECTED_SEED, variants_per_image=sampler.EXPECTED_VARIANTS
    )
    return support, pairs, labels


def test_exact_nominal_attempt_count_and_all_training_ids(committed: dict) -> None:
    evidence = committed["assignment_evidence"]
    assert evidence["nominal_attempt_count"] == 8_469 * 6 == 50_814  # literal-ok: frozen AM-88 arithmetic
    assert evidence["participating_training_id_count"] == 8_469  # literal-ok: frozen training count


def test_every_training_id_has_six_distinct_supported_qualities() -> None:
    support, pairs, _labels = _pairs()
    support_ids = {row["quality_id"] for row in support["artifact_quality_projection"]["qualities"]}
    by_image: dict[str, list[str]] = {}
    for stable_id, quality_id in pairs:
        by_image.setdefault(stable_id, []).append(quality_id)
    assert len(by_image) == 8_469  # literal-ok: frozen training count
    assert all(len(values) == len(set(values)) == 6 for values in by_image.values())  # literal-ok: AM-88 multiplicity
    assert {quality_id for _stable_id, quality_id in pairs} == support_ids


def test_no_validation_or_test_id_participates() -> None:
    _support, pairs, _labels = _pairs()
    _ids, _labels2, _by_class, split_by_id = sampler._training_membership()  # noqa: SLF001
    assert {split_by_id[stable_id] for stable_id, _quality_id in pairs} == {"train"}


def test_all_120_qualities_receive_globally_balanced_attempts(committed: dict) -> None:
    counts = committed["assignment_evidence"]["quality_attempt_counts"]
    assert len(counts) == 120  # literal-ok: immutable AM-87 support count
    assert set(counts.values()) == {423, 424}  # literal-ok: floor/ceil of 50814/120
    assert max(counts.values()) - min(counts.values()) == 1


def test_every_class_is_arithmetically_balanced(committed: dict) -> None:
    balance = committed["assignment_evidence"]["balance"]["per_class"]
    assert set(balance) == {str(label) for label in range(10)}  # literal-ok: Imagenette classes
    assert all(stats["range"] <= 1 for stats in balance.values())
    assert all(stats["attempts"] > 0 for stats in balance.values())


def test_no_duplicate_pair() -> None:
    _support, pairs, _labels = _pairs()
    assert len(pairs) == len(set(pairs)) == 50_814  # literal-ok: exact AM-88 pair count


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("sampler", "seed", "foreign-seed"),
        ("sampler", "variants_per_training_image", 5),  # literal-ok: mutation fixture
        ("sampler", "version", "foreign-version"),
    ],
)
def test_seed_variants_and_version_cannot_be_silently_changed(
    tmp_path: Path, committed: dict, section: str, field: str, value: object
) -> None:
    mutated = copy.deepcopy(committed)
    mutated[section][field] = value
    _reject(tmp_path, mutated)


def test_changing_seed_changes_pair_and_plan_identity() -> None:
    _ids, _labels, by_class, _splits = sampler._training_membership()  # noqa: SLF001
    _support, quality_ids, _rows = sampler._load_am87_support()  # noqa: SLF001
    first, _ = sampler.derive_assignments(quality_ids, by_class, seed=sampler.EXPECTED_SEED, variants_per_image=6)
    second, _ = sampler.derive_assignments(quality_ids, by_class, seed="foreign-seed", variants_per_image=6)
    assert sampler._pair_digest(first) != sampler._pair_digest(second)  # noqa: SLF001


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "foreign"])
def test_quality_support_mutations_fail_closed(tmp_path: Path, committed: dict, mutation: str) -> None:
    mutated = copy.deepcopy(committed)
    values = mutated["support"]["quality_ids_in_am87_order"]
    if mutation == "missing":
        values.pop()
    elif mutation == "extra":
        values.append("g8fquality-" + "0" * 64)
    elif mutation == "reordered":
        values[0:2] = reversed(values[0:2])
    else:
        values[0] = "g8fquality-" + "f" * 64
    _reject(tmp_path, mutated)


@pytest.mark.parametrize("field", ["stable_id_count", "stable_id_set_sha256", "stable_id_label_mapping_sha256"])
def test_training_membership_or_class_label_mutation_fails_closed(
    tmp_path: Path, committed: dict, field: str
) -> None:
    mutated = copy.deepcopy(committed)
    if field == "stable_id_count":
        mutated["training_membership"][field] -= 1
    else:
        mutated["training_membership"][field] = "0" * 64
    _reject(tmp_path, mutated)


def test_assignment_api_cannot_accept_pass_one_or_validation_performance() -> None:
    _support, quality_ids, _rows = sampler._load_am87_support()  # noqa: SLF001
    _ids, _labels, by_class, _splits = sampler._training_membership()  # noqa: SLF001
    with pytest.raises(TypeError):
        sampler.derive_assignments(
            quality_ids,
            by_class,
            seed=sampler.EXPECTED_SEED,
            variants_per_image=6,
            pass_one_score=0.99,  # type: ignore[call-arg]  # literal-ok: forbidden-input fixture
        )
    with pytest.raises(TypeError):
        sampler.derive_assignments(
            quality_ids,
            by_class,
            seed=sampler.EXPECTED_SEED,
            variants_per_image=6,
            validation_e4_feasibility=True,  # type: ignore[call-arg]
        )


def test_typed_codec_infeasibility_never_resamples(committed: dict) -> None:
    semantics = committed["f1_outcome_semantics"]
    assert semantics["typed_image_codec_infeasibility"] == "record_omitted_assigned_pair_no_replacement_no_resampling"
    assert semantics["neighbour_quality_substitution"] is False
    assert semantics["outage_image_substitution"] is False
    assert all(
        semantics[field] == "HOLD"
        for field in ("unexpected_codec_or_decoder_failure", "runtime_exception", "foreign_or_corrupt_identity", "unverified_artifact")
    )


def test_protected_pre_f0_counters_are_zero(committed: dict) -> None:
    boundary = committed["protected_boundary"]
    assert boundary["f0_execution_authorized"] is False
    assert boundary["corpus_materialized"] is False
    assert boundary["materialized_object_count"] == 0
    assert all(
        boundary[field] == 0
        for field in ("image_payloads_decoded", "jpeg2000_invocations", "classifier_inference", "optimizer_steps", "pass_two", "test_access")
    )
    assert boundary["confessor_started"] is False


def test_metadata_planner_has_no_payload_codec_classifier_optimizer_or_test_calls() -> None:
    source = Path(sampler.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "data.test_access" not in imports
    forbidden_calls = {
        "canonicalize_source", "encode_to_budget", "decode", "predict", "forward",
        "backward", "step", "load_test_dataset",
    }
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert not (called & forbidden_calls)


def test_am87_support_projection_still_reproduces_exactly() -> None:
    _support, quality_ids, _rows = sampler._load_am87_support()  # noqa: SLF001
    assert sampler._derive_am87_support_ids() == quality_ids  # noqa: SLF001


def test_independent_verifier_reproduces_committed_digests(committed: dict) -> None:
    verified = independent.verify_sampler_plan()
    assert verified["plan_id"] == committed["plan_id"]
    assert verified["assignment_evidence"]["ordered_pair_sha256"] == "c7c29729c2e0a94646c9e1ef16f45fe240badfb4a1baf82f27883681bac65229"
    assert verified["assignment_evidence"]["pair_set_sha256"] == "255eab85aca45e4ca910f040b0752ea773630c642c8697579df056104dcb594e"


def test_am88_post_campaign_compatibility_mutation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = json.loads(production.AM88_POST_CAMPAIGN_SOURCE_COMPATIBILITY.read_bytes())
    value["protected_boundary"]["training"] = 1
    body = {key: child for key, child in value.items() if key != "compatibility_id"}
    value["compatibility_id"] = "g8postsource-" + production.sha256_bytes(production.canonical_json(body))
    path = tmp_path / "am88-post.json"
    path.write_bytes(sampler.rendered_json(value))
    monkeypatch.setattr(production, "AM88_POST_CAMPAIGN_SOURCE_COMPATIBILITY", path)
    with pytest.raises(production.ProductionContractError):
        production.validate_production_contracts()


def test_am88_g8e_compatibility_mutation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = json.loads(v3s.AM88_SOURCE_COMPATIBILITY_PATH.read_bytes())
    value["entries"][0]["current_sha256"] = "0" * 64
    body = {key: child for key, child in value.items() if key != "compatibility_id"}
    value["compatibility_id"] = "g8esourcecompat-" + v3s.sha256_bytes(v3s.canonical_json(body))
    path = tmp_path / "am88-g8e.json"
    path.write_bytes(sampler.rendered_json(value))
    monkeypatch.setattr(v3s, "AM88_SOURCE_COMPATIBILITY_PATH", path)
    source = json.loads(v3s.V3S_SOURCE_MANIFEST_PATH.read_bytes())
    with pytest.raises(v3s.G8EV3SError):
        v3s.validate_source_manifest(source)
