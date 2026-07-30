"""Frozen constant-class outage policy: selection, freezing and keyed sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import baseline.classical.outage as outage
from baseline.classical.outage import (
    EVIDENCE_LABELS,
    OUTAGE_POLICY_SCHEMA_VERSION,
    SELECTION_SPLIT,
    OutagePolicy,
    OutagePolicyError,
    ValidationLabelCounts,
    build_outage_policy_record,
    class_count,
    configured_rng_identity_fields,
    count_validation_labels,
    keyed_uniform_random_label,
    load_outage_policy,
    policy_from_record,
    select_outage_policy,
    sensitivity_labels,
)
from config.params import REPO_ROOT, get
from data.manifests import ManifestRow

OUTAGE_DATASET = "imagenette160"
ARTIFACT_PATH = REPO_ROOT / "results/baseline/w4/outage_policy.json"


@pytest.fixture
def committed_record() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _counts(class_counts: tuple[int, ...]) -> ValidationLabelCounts:
    """A label-count fixture, so tie-break shapes the manifest cannot hold can be tested."""

    return ValidationLabelCounts(
        dataset="fixture",
        split=SELECTION_SPLIT,
        manifest_path="fixture.csv",
        manifest_sha256="0" * len("0" * 64),
        class_count=len(class_counts),
        class_counts=class_counts,
        validation_count=sum(class_counts),
    )


# --------------------------------------------------------------------------
# Selection over the committed manifest
# --------------------------------------------------------------------------


def test_full_validation_counts_reproduce_the_frozen_artifact(committed_record: dict) -> None:
    counts = count_validation_labels(OUTAGE_DATASET, REPO_ROOT)
    assert list(counts.class_counts) == committed_record["class_counts"]
    assert counts.validation_count == committed_record["validation_count"]
    assert counts.manifest_sha256 == committed_record["manifest_sha256"]
    assert counts.class_count == committed_record["class_count"]
    assert list(counts.tied_maximum_classes()) == committed_record["tied_maximum_classes"]
    assert counts.selected_class() == committed_record["selected_class"]


def test_selection_counts_the_whole_validation_split_not_a_subset() -> None:
    counts = count_validation_labels(OUTAGE_DATASET, REPO_ROOT)
    assert counts.validation_count == int(
        get(f"datasets.{OUTAGE_DATASET}.{SELECTION_SPLIT}_images")
    )
    assert sum(counts.class_counts) == counts.validation_count


def test_selected_class_is_the_true_maximum() -> None:
    policy, counts = select_outage_policy(OUTAGE_DATASET, REPO_ROOT)
    assert policy.selected_count == max(counts.class_counts)
    assert counts.class_counts[policy.selected_class] == policy.selected_count


def test_selection_is_reproducible() -> None:
    first, _ = select_outage_policy(OUTAGE_DATASET, REPO_ROOT)
    second, _ = select_outage_policy(OUTAGE_DATASET, REPO_ROOT)
    assert first == second


def test_selection_reads_validation_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Train and test rows are discarded before any counting happens.

    The crafted manifest gives class 9 an overwhelming majority in ``train`` and
    ``test`` and gives class 3 the validation maximum.  A selection that leaked
    either split would return 9.
    """

    rows = tuple(
        [ManifestRow(f"{index:016x}", 9, "train") for index in range(50)]
        + [ManifestRow(f"{index + 100:016x}", 9, "test") for index in range(50)]
        + [ManifestRow(f"{index + 200:016x}", 3, SELECTION_SPLIT) for index in range(5)]
        + [ManifestRow(f"{index + 300:016x}", 1, SELECTION_SPLIT) for index in range(2)]
    )
    monkeypatch.setattr(outage, "validate_manifest_bytes", lambda dataset, payload: rows)
    monkeypatch.setattr(
        outage,
        "get",
        lambda path: 7 if path.endswith(f".{SELECTION_SPLIT}_images") else get(path),
    )
    counts = count_validation_labels(OUTAGE_DATASET, REPO_ROOT)
    assert counts.validation_count == 7
    assert counts.class_counts[9] == 0
    assert counts.selected_class() == 3


def test_selection_rejects_a_manifest_with_no_validation_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = (ManifestRow("0" * 16, 0, "train"), ManifestRow("1" * 16, 1, "test"))
    monkeypatch.setattr(outage, "validate_manifest_bytes", lambda dataset, payload: rows)
    with pytest.raises(OutagePolicyError, match="no val rows"):
        count_validation_labels(OUTAGE_DATASET, REPO_ROOT)


# --------------------------------------------------------------------------
# Tie-break and the measured-versus-assumed accuracy distinction
# --------------------------------------------------------------------------


def test_tie_break_selects_the_lowest_class_index() -> None:
    counts = _counts((4, 9, 9, 9, 1))
    assert counts.tied_maximum_classes() == (1, 2, 3)
    assert counts.selected_class() == 1


def test_a_unique_maximum_needs_no_tie_break() -> None:
    counts = _counts((4, 9, 11, 9, 1))
    assert counts.tied_maximum_classes() == (2,)
    assert counts.selected_class() == 2


def test_measured_accuracy_is_count_derived_and_not_one_over_n() -> None:
    """An imbalanced validation split must not report ``1 / n_classes``."""

    counts = _counts((10, 30, 10, 10, 40))
    selected = counts.selected_class()
    policy = OutagePolicy(
        dataset="fixture",
        split=SELECTION_SPLIT,
        manifest_sha256="0" * 64,
        class_count=counts.class_count,
        selected_class=selected,
        selected_count=counts.class_counts[selected],
        validation_count=counts.validation_count,
        tied_maximum_classes=counts.tied_maximum_classes(),
        class_counts=counts.class_counts,
    )
    assert selected == 4
    assert policy.measured_validation_accuracy == 40 / 100
    assert policy.measured_validation_accuracy != 1 / counts.class_count


def test_committed_accuracy_equals_one_over_n_only_because_the_split_is_stratified(
    committed_record: dict,
) -> None:
    """Document *why* the real number coincides with the theoretical one.

    ``data.manifests._validate_counts_and_stratification`` enforces an exactly
    stratified validation split, so every class ties at ``val_images /
    classes``.  The recorded accuracy is still a measured ratio: it carries an
    explicit numerator and denominator, and the tie-break really fires across
    all ten classes.  A hardcoded ``1/n`` would be indistinguishable here and
    would silently survive a manifest whose stratification changed, which is why
    the artifact and the verifier check counts rather than the float.
    """

    counts = count_validation_labels(OUTAGE_DATASET, REPO_ROOT)
    n_classes = class_count(OUTAGE_DATASET)
    assert len(counts.tied_maximum_classes()) == n_classes
    assert committed_record["numerator"] == counts.class_counts[0]
    assert committed_record["denominator"] == counts.validation_count
    assert (
        committed_record["measured_validation_accuracy"]
        == committed_record["numerator"] / committed_record["denominator"]
    )
    assert committed_record["tie_break_applied"] is True
    assert committed_record["accuracy_derivation"] == "selected_count / validation_count"


# --------------------------------------------------------------------------
# The frozen artifact
# --------------------------------------------------------------------------


def test_committed_artifact_loads_and_carries_the_required_labels(
    committed_record: dict,
) -> None:
    policy = load_outage_policy(ARTIFACT_PATH, expected_dataset=OUTAGE_DATASET)
    assert policy.dataset == OUTAGE_DATASET
    assert policy.split == SELECTION_SPLIT
    assert tuple(committed_record["evidence_labels"]) == EVIDENCE_LABELS
    assert committed_record["schema_version"] == OUTAGE_POLICY_SCHEMA_VERSION
    assert committed_record["frozen_before"] == get("baseline.outage_class_frozen_before")
    assert committed_record["test_split_access"]["test_split_sealed"] is True
    assert committed_record["status"] == "frozen"


def test_artifact_record_is_deterministic_apart_from_its_provenance_fields() -> None:
    first = build_outage_policy_record(
        OUTAGE_DATASET,
        selection_source_commit="a" * 40,
        generated_at="2026-01-01T00:00:00+00:00",
        repo_root=REPO_ROOT,
    )
    second = build_outage_policy_record(
        OUTAGE_DATASET,
        selection_source_commit="b" * 40,
        generated_at="2026-02-02T00:00:00+00:00",
        repo_root=REPO_ROOT,
    )
    volatile = {"selection_source_commit", "generated_at"}
    assert {k: v for k, v in first.items() if k not in volatile} == {
        k: v for k, v in second.items() if k not in volatile
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"selected_class": 1}, "lowest-index tie-break"),
        ({"selection_policy": "uniform_random"}, "different outage policy"),
        ({"selection_rule": "most_frequent"}, "different selection rule"),
        ({"tie_break": "highest_class_index"}, "different tie-break"),
        ({"frozen_before": "never"}, "different freeze point"),
        ({"split": "test"}, "not 'val'"),
        ({"schema_version": OUTAGE_POLICY_SCHEMA_VERSION + 1}, "schema_version"),
        ({"evidence_labels": ["bounded"]}, "wrong evidence labels"),
        ({"numerator": 99}, "numerator/denominator"),
        ({"denominator": 999}, "numerator/denominator"),
        ({"selected_count": 99}, "selected count disagrees"),
        ({"measured_validation_accuracy": 0.5}, "not selected_count / validation_count"),
        ({"class_count": 9}, "class count disagrees"),
        ({"tied_maximum_classes": [0]}, "tied-maximum classes are wrong"),
    ],
)
def test_artifact_mutations_fail_closed(
    committed_record: dict, mutation: dict, message: str
) -> None:
    record = dict(committed_record) | mutation
    with pytest.raises(OutagePolicyError, match=message):
        policy_from_record(record)


def test_artifact_missing_a_required_field_fails(committed_record: dict) -> None:
    record = {k: v for k, v in committed_record.items() if k != "selected_class"}
    with pytest.raises(OutagePolicyError, match="incomplete"):
        policy_from_record(record)


def test_artifact_that_hardcodes_one_over_n_without_matching_counts_fails(
    committed_record: dict,
) -> None:
    """The mutation the coincidence makes dangerous.

    Halving one class count leaves the *theoretical* ``1/n`` unchanged, so an
    artifact that reported ``0.1`` regardless would still pass a float check.
    Recomputing from counts catches it.
    """

    counts = list(committed_record["class_counts"])
    counts[0] = counts[0] // 2
    record = dict(committed_record) | {
        "class_counts": counts,
        "validation_count": sum(counts),
        "denominator": sum(counts),
        "measured_validation_accuracy": 1 / committed_record["class_count"],
    }
    with pytest.raises(OutagePolicyError):
        policy_from_record(record)


def test_artifact_dataset_mismatch_fails_closed() -> None:
    with pytest.raises(OutagePolicyError, match="not 'cifar10'"):
        load_outage_policy(ARTIFACT_PATH, expected_dataset="cifar10")


def test_artifact_manifest_mismatch_fails_closed() -> None:
    with pytest.raises(OutagePolicyError, match="different split manifest"):
        load_outage_policy(
            ARTIFACT_PATH,
            expected_dataset=OUTAGE_DATASET,
            expected_manifest_sha256="0" * 64,
        )


def test_missing_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(OutagePolicyError, match="cannot read"):
        load_outage_policy(tmp_path / "absent.json")


# --------------------------------------------------------------------------
# Runtime prediction
# --------------------------------------------------------------------------


def test_runtime_prediction_does_not_reload_or_reselect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction must touch no manifest and no label, ever."""

    policy = load_outage_policy(ARTIFACT_PATH, expected_dataset=OUTAGE_DATASET)

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("prediction reselected the outage class")

    monkeypatch.setattr(outage, "count_validation_labels", _forbidden)
    monkeypatch.setattr(outage, "validate_manifest_bytes", _forbidden)
    assert [policy.predict() for _ in range(5)] == [policy.selected_class] * 5


def test_constant_prediction_is_invariant_across_rows_systems_and_order() -> None:
    policy = load_outage_policy(ARTIFACT_PATH, expected_dataset=OUTAGE_DATASET)
    predictions = [policy.predict() for _ in range(100)]
    assert len(set(predictions)) == 1
    # The prediction takes no argument at all, so no system, sample, batch or
    # ordering can be threaded into it even by accident.
    assert policy.predict.__code__.co_argcount == 1


def test_outage_correctness_is_binary() -> None:
    policy = load_outage_policy(ARTIFACT_PATH, expected_dataset=OUTAGE_DATASET)
    outcomes = {policy.is_correct(label) for label in range(policy.class_count)}
    assert outcomes == {True, False}
    for label in range(policy.class_count):
        assert isinstance(policy.is_correct(label), bool)
    assert policy.is_correct(policy.selected_class) is True


# --------------------------------------------------------------------------
# Keyed sensitivity variant
# --------------------------------------------------------------------------


def test_sensitivity_rng_identity_matches_the_central_contract() -> None:
    fields = configured_rng_identity_fields()
    assert set(fields) == set(get("artifacts.rng_identity_fields.outage_label"))
    assert "purpose=outage_label" in get("baseline.outage_rng_key")
    assert not any(field.startswith("purpose") for field in fields)


def test_sensitivity_label_is_in_range() -> None:
    n_classes = class_count(OUTAGE_DATASET)
    labels = [
        keyed_uniform_random_label(
            split_manifest_hash="m",
            stable_sample_id=f"{index:016x}",
            channel_seed=0,
            n_classes=n_classes,
        )
        for index in range(200)
    ]
    assert set(labels) <= set(range(n_classes))
    assert len(set(labels)) > 1


def test_sensitivity_is_invariant_to_row_order_batching_and_other_rows() -> None:
    n_classes = class_count(OUTAGE_DATASET)
    rows = [{"stable_sample_id": f"{index:016x}"} for index in range(24)]
    kwargs = {
        "split_manifest_hash": "manifest",
        "channel_seed": 7,
        "n_classes": n_classes,
    }
    straight = sensitivity_labels(rows, **kwargs)
    reversed_rows = sensitivity_labels(list(reversed(rows)), **kwargs)
    assert tuple(reversed(reversed_rows)) == straight

    batched: list[int] = []
    for size in (1, 5, 7, 11):
        batched = []
        for start in range(0, len(rows), size):
            batched.extend(sensitivity_labels(rows[start : start + size], **kwargs))
        assert tuple(batched) == straight

    subset = sensitivity_labels(rows[3:5], **kwargs)
    assert subset == straight[3:5]


def test_sensitivity_is_unmoved_by_intervening_draws() -> None:
    n_classes = class_count(OUTAGE_DATASET)
    kwargs = {
        "split_manifest_hash": "manifest",
        "stable_sample_id": "0" * 16,
        "channel_seed": 3,
        "n_classes": n_classes,
    }
    first = keyed_uniform_random_label(**kwargs)
    np.random.default_rng(1234).standard_normal(1000)
    for index in range(50):
        keyed_uniform_random_label(
            split_manifest_hash="other",
            stable_sample_id=f"{index:016x}",
            channel_seed=99,
            n_classes=n_classes,
        )
    assert keyed_uniform_random_label(**kwargs) == first


@pytest.mark.parametrize(
    "override",
    [
        {"split_manifest_hash": "different"},
        {"stable_sample_id": "f" * 16},
        {"channel_seed": 8},
    ],
)
def test_sensitivity_varies_with_every_declared_identity_component(
    override: dict,
) -> None:
    n_classes = class_count(OUTAGE_DATASET)
    base = {
        "split_manifest_hash": "manifest",
        "stable_sample_id": "0" * 16,
        "channel_seed": 7,
        "n_classes": n_classes,
    }
    mutated = base | override

    def _identity(values: dict) -> dict:
        return {key: value for key, value in values.items() if key != "n_classes"}

    # One drawn label can collide by chance over ten classes, so compare the
    # keyed *stream*: that is what must move when any declared component moves.
    long_base = outage.keyed_generator("outage_label", _identity(base)).integers(
        0, n_classes, size=64
    )
    long_mutated = outage.keyed_generator("outage_label", _identity(mutated)).integers(
        0, n_classes, size=64
    )
    assert not np.array_equal(long_base, long_mutated)


def test_sensitivity_rejects_an_incomplete_or_extended_identity() -> None:
    with pytest.raises(ValueError, match="missing"):
        outage.keyed_generator("outage_label", {"split_manifest_hash": "m"})
    with pytest.raises(ValueError, match="extra"):
        outage.keyed_generator(
            "outage_label",
            {
                "split_manifest_hash": "m",
                "stable_sample_id": "0" * 16,
                "channel_seed": 1,
                "purpose": "outage_label",
            },
        )


def test_sensitivity_rejects_a_non_positive_class_count() -> None:
    with pytest.raises(OutagePolicyError, match="positive integer"):
        keyed_uniform_random_label(
            split_manifest_hash="m",
            stable_sample_id="0" * 16,
            channel_seed=1,
            n_classes=0,
        )


def test_sensitivity_is_never_the_primary_outcome(committed_record: dict) -> None:
    """The frozen constant remains the policy; the keyed draw is secondary."""

    assert committed_record["selection_policy"] == get("baseline.outage_policy")
    assert committed_record["sensitivity_policy"] == get(
        "baseline.outage_policy_sensitivity"
    )
    assert committed_record["selection_policy"] != committed_record["sensitivity_policy"]
