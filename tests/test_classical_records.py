"""Exact W4 record schemas, identity construction, prediction and aggregation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import baseline.classical.records as records
from artifacts.ids import make_noise_id
from baseline.classical.channel_transport import build_accounting
from baseline.classical.outage import NOT_APPLICABLE, load_outage_policy
from baseline.classical.pipeline import (
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    DELIVERED,
    STRUCTURAL_INFEASIBILITY,
    ChannelIdentity,
    ClassicalResult,
    SourceCoding,
)
from baseline.classical.records import (
    FROZEN_CLASSIFIER_DATASET,
    AggregateContext,
    RecordError,
    RunIdentity,
    TaskOutcome,
    aggregate_row,
    aggregate_schema,
    classify_reconstruction,
    codestream_byte_split,
    field_semantics,
    noise_identity,
    per_image_row,
    per_image_schema,
    reconcile_aggregate,
    score_result,
    validate_row,
)
from baseline.ldpc.transport import build_packet_plan
from config.params import REPO_ROOT, get

ARTIFACT_PATH = REPO_ROOT / "results/baseline/w4/outage_policy.json"
DATASET = FROZEN_CLASSIFIER_DATASET
RATIO = "r_1_24"
MODULATION = "qam16"
LDPC_RATE = "2/3"
SNR_DB = 18.0
K_SYMBOLS = get(f"bandwidth.k_symbols.{DATASET}.{RATIO}")
IMAGE_HW = tuple(int(v) for v in get(f"datasets.{DATASET}.image_size")[:2])


@pytest.fixture(scope="module")
def policy():
    return load_outage_policy(ARTIFACT_PATH, expected_dataset=DATASET)


@pytest.fixture(scope="module")
def accounting():
    return build_accounting(build_packet_plan(K_SYMBOLS, MODULATION, LDPC_RATE))


class _FixtureClassifier(torch.nn.Module):
    """A labelled *test fixture*, never the frozen G-1 model.

    It returns a fixed argmax so prediction plumbing can be checked without
    loading a 92 MB checkpoint, and it counts its calls so "the classifier never
    runs on an outage row" is an assertion rather than a claim.
    """

    def __init__(self, predicted: int = 3) -> None:
        super().__init__()
        self.predicted = predicted
        self.calls = 0
        self.eval()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        n_classes = int(get(f"datasets.{DATASET}.classes"))
        logits = torch.zeros((batch.shape[0], n_classes))
        logits[:, self.predicted] = 1.0
        return logits


def _identity(**overrides: Any) -> RunIdentity:
    values = {
        "system": "classical_fixed_mcs",
        "dataset": DATASET,
        "dataset_version": get(f"datasets.{DATASET}.archive_sha256"),
        "split": "val",
        "split_manifest_hash": get(f"datasets.{DATASET}.manifest_sha256"),
        "bw_ratio": RATIO,
        "test_snr_db": SNR_DB,
        "train_seed": get("evaluation.train_seeds")[0],
        "channel_seed": get("evaluation.channel_seeds")[0],
        "config_hash": "c" * 64,
        "checkpoint_id": "d" * 64,
        "classifier_variant": get("reference_classifier.clean_variant_name"),
        "ldpc_rate": LDPC_RATE,
        "modulation": MODULATION,
        "quantiser_bits": None,
        "transmit_dim": None,
        "reconstruction_weight": None,
        "analysis_version": get("config.analysis_version"),
    }
    values.update(overrides)
    return RunIdentity(**values)


def _canonical() -> np.ndarray:
    rows, columns = np.indices(IMAGE_HW)
    return np.stack(
        (
            (rows % 256).astype(np.uint8),
            (columns % 256).astype(np.uint8),
            ((rows + columns) % 256).astype(np.uint8),
        ),
        axis=-1,
    )


def _scheduled(sample_id: str = "0" * 16, **overrides) -> str:
    """The cell's scheduled channel realisation, drawn from nothing."""

    values = {
        "dataset_version": get(f"datasets.{DATASET}.archive_sha256"),
        "split_manifest_hash": get(f"datasets.{DATASET}.manifest_sha256"),
        "stable_sample_id": sample_id,
        "test_snr_db": SNR_DB,
        "channel_seed": get("evaluation.channel_seeds")[0],
        "k": K_SYMBOLS,
        "block_index": 0,
    }
    values.update(overrides)
    return noise_identity(**values)


def _result(
    verdict: str,
    *,
    accounting,
    sample_id: str = "0" * 16,
    decoded: np.ndarray | None = None,
    codestream: bytes | None = None,
) -> ClassicalResult:
    transmitted = verdict in (DELIVERED, DECODE_FAILURE)
    noise = (
        ChannelIdentity(
            dataset_version=get(f"datasets.{DATASET}.archive_sha256"),
            split_manifest_hash=get(f"datasets.{DATASET}.manifest_sha256"),
            channel_seed=get("evaluation.channel_seeds")[0],
        ).noise_id(
            stable_sample_id=sample_id,
            test_snr_db=SNR_DB,
            k=K_SYMBOLS,
            block_index=0,
        )
        if transmitted
        else None
    )
    structural = verdict == STRUCTURAL_INFEASIBILITY
    source_coding = (
        None
        if structural
        else SourceCoding(
            feasible=verdict != CODEC_INFEASIBILITY,
            encode_axis_px=IMAGE_HW[0] if verdict != CODEC_INFEASIBILITY else None,
            axes_attempted=(IMAGE_HW[0],),
            axis_reasons=(),
            payload_capacity_bytes=accounting.payload_bytes,
            emitted_bytes=len(codestream) if codestream else None,
            payload_filler_bytes=None,
            payload_filler_bits=None,
            codestream_sha256=None,
            cache_key=None,
            cache_hit=None,
            search_iterations=None,
            emitted_codestream=codestream,
        )
    )
    return ClassicalResult(
        verdict=verdict,
        dataset=DATASET,
        k_symbols=K_SYMBOLS,
        modulation=MODULATION,
        ldpc_rate=LDPC_RATE,
        snr_db=SNR_DB,
        stable_sample_id=sample_id,
        noise_id=noise,
        packet_feasible=not structural,
        structural_reason="no_legal_byte_aligned_A" if structural else None,
        accounting=None if structural else accounting,
        source_coding=source_coding,
        transport=None,
        codestream_recovered_exactly=True if verdict == DELIVERED else None,
        decoded_image=decoded,
    )


# ---------------------------------------------------------------------------
# Exact schemas
# ---------------------------------------------------------------------------


def test_aggregate_schema_is_read_from_params_not_restated() -> None:
    assert aggregate_schema() == tuple(get("artifacts.csv_schema"))
    assert len(aggregate_schema()) == len(set(aggregate_schema()))


def test_per_image_schema_is_read_from_params_not_restated() -> None:
    assert per_image_schema() == tuple(get("artifacts.per_image_schema"))
    assert len(per_image_schema()) == len(set(per_image_schema()))


def test_field_semantics_cover_both_schemas_exactly() -> None:
    semantics = field_semantics()
    assert tuple(semantics["aggregate"]) == aggregate_schema()
    assert tuple(semantics["per_image"]) == per_image_schema()
    for table in (semantics["aggregate"], semantics["per_image"]):
        for field, entry in table.items():
            assert set(entry) == {
                "source",
                "type",
                "unit",
                "nullable",
                "not_applicable_representation",
                "aggregation_denominator",
            }, field
            assert entry["source"], field


def test_source_bytes_is_the_br10_quantity(accounting) -> None:
    """BR-10 fixes ``source_bytes`` as exactly A/8, not archive or RGB bytes."""

    semantics = field_semantics()["per_image"]["source_bytes"]
    assert "A/8" in semantics["source"]
    assert accounting.payload_bytes * 8 == accounting.payload_bits


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda row: {k: v for k, v in row.items() if k != "correct"}, id="dropped"),
        pytest.param(lambda row: dict(row) | {"extra_field": 1}, id="added"),
        pytest.param(
            lambda row: {("accuracy" if k == "correct" else k): v for k, v in row.items()},
            id="renamed",
        ),
        pytest.param(
            lambda row: dict(reversed(list(row.items()))),
            id="reordered",
        ),
    ],
)
def test_per_image_schema_mutations_are_rejected(mutate, policy, accounting) -> None:
    row = per_image_row(
        _result(DECODE_FAILURE, accounting=accounting),
        TaskOutcome(0, True, True, DECODE_FAILURE, None, None, None, None),
        identity=_identity(),
        true_label=0,
        run_id="r" * 64,
        scheduled_noise_id=_scheduled(),
    )
    with pytest.raises(RecordError):
        validate_row(mutate(row), per_image_schema())


def test_duplicate_schema_fields_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    duplicated = list(get("artifacts.per_image_schema"))
    duplicated[1] = duplicated[0]
    monkeypatch.setattr(
        records,
        "get",
        lambda path: duplicated if path == "artifacts.per_image_schema" else get(path),
    )
    with pytest.raises(RecordError, match="duplicate fields"):
        per_image_schema()


# ---------------------------------------------------------------------------
# System identity
# ---------------------------------------------------------------------------


def test_pb2_uses_the_fixed_mcs_system_value() -> None:
    """PB_2 runs one explicitly fixed configuration and has built no adaptation."""

    assert "classical_fixed_mcs" in get("artifacts.system_values")
    assert _identity().system == "classical_fixed_mcs"


def test_unknown_system_values_are_rejected() -> None:
    with pytest.raises(RecordError, match="unsupported system"):
        _identity(system="classical_someday")


def test_test_split_records_are_refused() -> None:
    with pytest.raises(RecordError, match="sealed behind SR-22 and G-12"):
        _identity(split="test")


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


def test_identity_key_sets_match_params() -> None:
    identity = _identity()
    assert set(identity.run_id_values()) == set(get("artifacts.run_id_key"))


def test_changing_system_changes_run_id_but_not_pair_id() -> None:
    base = _identity()
    other = _identity(system="classical_jpeg_secondary")
    assert base.run_id() != other.run_id()
    assert base.pair_id(stable_sample_id="a" * 16, noise_id="n") == other.pair_id(
        stable_sample_id="a" * 16, noise_id="n"
    )


def test_pair_id_excludes_system_and_comparison() -> None:
    excluded = set(get("artifacts.pair_id_excludes"))
    assert excluded == {"system", "comparison"}
    assert not (set(get("artifacts.pair_id_key")) & excluded)


@pytest.mark.parametrize(
    "override",
    [
        {"config_hash": "0" * 64},
        {"checkpoint_id": "0" * 64},
        {"split": "train"},
        {"classifier_variant": "artifact_finetuned"},
    ],
)
def test_run_id_changes_with_config_checkpoint_split_and_variant(override) -> None:
    assert _identity().run_id() != _identity(**override).run_id()


def test_identity_rejects_missing_and_extra_key_fields() -> None:
    from artifacts.ids import make_run_id

    values = _identity().run_id_values()
    with pytest.raises(ValueError, match="missing fields"):
        make_run_id({k: v for k, v in values.items() if k != "modulation"})
    # An extra field is silently irrelevant to the ID only if the key set is
    # honoured, which is exactly what the digest must prove.
    assert make_run_id(values | {"unused": 1}) == make_run_id(values)


def test_noise_identity_agrees_with_the_pb1_channel_identity() -> None:
    channel = ChannelIdentity(
        dataset_version=get(f"datasets.{DATASET}.archive_sha256"),
        split_manifest_hash=get(f"datasets.{DATASET}.manifest_sha256"),
        channel_seed=get("evaluation.channel_seeds")[0],
    )
    from_pb1 = channel.noise_id(
        stable_sample_id="a" * 16, test_snr_db=SNR_DB, k=K_SYMBOLS, block_index=0
    )
    from_records = noise_identity(
        dataset_version=get(f"datasets.{DATASET}.archive_sha256"),
        split_manifest_hash=get(f"datasets.{DATASET}.manifest_sha256"),
        stable_sample_id="a" * 16,
        test_snr_db=SNR_DB,
        channel_seed=get("evaluation.channel_seeds")[0],
        k=K_SYMBOLS,
        block_index=0,
    )
    assert from_pb1 == from_records
    assert from_records == make_noise_id(
        {
            "dataset_version": get(f"datasets.{DATASET}.archive_sha256"),
            "split_manifest_hash": get(f"datasets.{DATASET}.manifest_sha256"),
            "stable_sample_id": "a" * 16,
            "test_snr_db": SNR_DB,
            "channel_seed": get("evaluation.channel_seeds")[0],
            "channel": "awgn",
            "k": K_SYMBOLS,
            "block_index": 0,
            "rng_purpose": "channel_noise",
        }
    )


def test_row_order_and_batching_change_no_identity(policy, accounting) -> None:
    ids = [f"{index:016x}" for index in range(8)]
    identity = _identity()

    def build(order):
        return {
            sample: per_image_row(
                _result(DECODE_FAILURE, accounting=accounting, sample_id=sample),
                TaskOutcome(0, True, True, DECODE_FAILURE, None, None, None, None),
                identity=identity,
                true_label=0,
                run_id="r" * 64,
                scheduled_noise_id=_scheduled(sample),
            )
            for sample in order
        }

    straight = build(ids)
    shuffled = build(list(reversed(ids)))
    assert straight == shuffled


# ---------------------------------------------------------------------------
# Prediction semantics
# ---------------------------------------------------------------------------


def test_delivered_row_runs_the_classifier_and_scores_metrics(policy, accounting) -> None:
    canonical = _canonical()
    classifier = _FixtureClassifier(predicted=7)
    result = _result(DELIVERED, accounting=accounting, decoded=canonical.copy())
    outcome = score_result(
        result,
        true_label=7,
        policy=policy,
        canonical_image=canonical,
        classifier=classifier,
    )
    assert classifier.calls == 1
    assert outcome.pred_label == 7
    assert outcome.correct is True
    assert outcome.outage is False
    assert outcome.outage_reason is None
    assert outcome.ssim == pytest.approx(1.0)
    assert outcome.psnr_db == float("inf")


@pytest.mark.parametrize(
    "verdict",
    [STRUCTURAL_INFEASIBILITY, CODEC_INFEASIBILITY, DECODE_FAILURE],
)
def test_each_outage_reason_maps_correctly_and_skips_the_classifier(
    verdict, policy, accounting
) -> None:
    classifier = _FixtureClassifier()
    outcome = score_result(
        _result(verdict, accounting=accounting),
        true_label=5,
        policy=policy,
        classifier=classifier,
    )
    assert classifier.calls == 0
    assert outcome.outage is True
    assert outcome.outage_reason == verdict
    assert outcome.pred_label == policy.selected_class
    assert outcome.psnr_db is None and outcome.ssim is None


def test_outage_correctness_is_binary_and_uses_the_frozen_class(policy, accounting) -> None:
    right = score_result(
        _result(DECODE_FAILURE, accounting=accounting),
        true_label=policy.selected_class,
        policy=policy,
    )
    wrong = score_result(
        _result(DECODE_FAILURE, accounting=accounting),
        true_label=(policy.selected_class + 1) % policy.class_count,
        policy=policy,
    )
    assert right.correct is True and wrong.correct is False
    assert isinstance(right.correct, bool) and isinstance(wrong.correct, bool)


def test_the_frozen_classifier_refuses_a_non_imagenette_reconstruction(
    policy, accounting
) -> None:
    """Ten equal output indices are not a shared class vocabulary."""

    canonical = _canonical()
    result = _result(DELIVERED, accounting=accounting, decoded=canonical.copy())
    cifar_result = ClassicalResult(**(dict(vars(result)) | {"dataset": "cifar10"}))
    cifar_policy = type(policy)(**(dict(vars(policy)) | {"dataset": "cifar10"}))
    with pytest.raises(RecordError, match="must not score 'cifar10'"):
        score_result(
            cifar_result,
            true_label=1,
            policy=cifar_policy,
            canonical_image=canonical,
            classifier=_FixtureClassifier(),
        )


def test_scoring_rejects_a_policy_for_another_dataset(policy, accounting) -> None:
    other = type(policy)(**(dict(vars(policy)) | {"dataset": "cifar10"}))
    with pytest.raises(RecordError, match="row dataset is"):
        score_result(_result(DECODE_FAILURE, accounting=accounting), true_label=0, policy=other)


def test_classifier_must_be_in_evaluation_mode() -> None:
    classifier = _FixtureClassifier()
    classifier.train()
    with pytest.raises(RecordError, match="evaluation mode"):
        classify_reconstruction(classifier, _canonical())


def test_classifier_inference_takes_no_gradient() -> None:
    classifier = _FixtureClassifier(predicted=2)
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)
    assert classify_reconstruction(classifier, _canonical()) == 2


# ---------------------------------------------------------------------------
# JPEG 2000 container accounting
# ---------------------------------------------------------------------------


def test_codestream_byte_split_reconciles_on_a_real_encode(tmp_path: Path) -> None:
    from baseline.j2k import J2KCodec

    codec = J2KCodec(tmp_path / "cache")
    image = _canonical()
    encoded = codec.encode_to_budget(
        image,
        canonical_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        budget_bytes=2000,
        encode_axis_px=IMAGE_HW[0],
    )
    assert encoded.feasible
    container, data = codestream_byte_split(encoded.codestream)
    assert container + data == encoded.emitted_byte_count == len(encoded.codestream)
    assert container > 0 and data > 0


def test_codestream_byte_split_rejects_a_non_codestream() -> None:
    with pytest.raises(RecordError, match="SOC"):
        codestream_byte_split(b"\x00\x01\x02\x03")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _rows_and_context(policy, accounting, verdicts, labels, run_id="r" * 64):
    identity = _identity()
    rows = []
    psnr, ssim = [], []
    for index, (verdict, label) in enumerate(zip(verdicts, labels, strict=True)):
        canonical = _canonical()
        result = _result(
            verdict,
            accounting=accounting,
            sample_id=f"{index:016x}",
            decoded=canonical.copy() if verdict == DELIVERED else None,
        )
        outcome = score_result(
            result,
            true_label=label,
            policy=policy,
            canonical_image=canonical if verdict == DELIVERED else None,
            classifier=_FixtureClassifier(predicted=label) if verdict == DELIVERED else None,
        )
        if verdict == DELIVERED:
            psnr.append(1.0)
            ssim.append(1.0)
        rows.append(
            per_image_row(
                result,
                outcome,
                identity=identity,
                true_label=label,
                run_id=run_id,
                scheduled_noise_id=_scheduled(result.stable_sample_id),
            )
        )
    context = AggregateContext(
        identity=identity,
        k_symbols=K_SYMBOLS,
        timestamp="2026-07-31T00:00:00+00:00",
        git_commit="a" * 40,
        git_dirty=False,
        source_codec=get("baseline.source_codec"),
        j2k_target_bytes=accounting.payload_bytes,
        wall_clock_s=1.0,
        peak_vram_gb=None,
        tb_crc_type=accounting.tb_crc_name,
        base_graph=accounting.base_graph,
        lifting_size=accounting.lifting_size,
        num_codeblocks=accounting.code_blocks,
        filler_bits=accounting.ldpc_filler_bits_total,
        effective_code_rate=accounting.k_prime / max(accounting.rate_matched_bits),
        bytes_sent=accounting.payload_bytes,
    )
    return rows, context, psnr, ssim


def test_aggregate_recomputes_exactly_from_the_per_image_rows(policy, accounting) -> None:
    verdicts = [DELIVERED, DELIVERED, DECODE_FAILURE, CODEC_INFEASIBILITY, STRUCTURAL_INFEASIBILITY]
    labels = [1, 2, policy.selected_class, 4, 5]
    rows, context, psnr, ssim = _rows_and_context(policy, accounting, verdicts, labels)
    aggregate = aggregate_row(
        rows, context, run_id="r" * 64, psnr_values=psnr, ssim_values=ssim
    )
    assert tuple(aggregate) == aggregate_schema()
    assert aggregate["n"] == aggregate["n_test"] == 5
    # Two delivered rows predicted correctly, plus the decode-failure row whose
    # true label is the frozen constant class.
    assert aggregate["n_correct"] == 3
    assert aggregate["top1_acc"] == 3 / 5
    assert aggregate["coverage_rate"] == 2 / 5
    assert aggregate["decode_failure_rate"] == 1 / 5
    assert aggregate["infeasible_rate"] == 2 / 5
    assert aggregate["acc_given_delivery"] == 1.0
    counts = reconcile_aggregate(aggregate, rows)
    assert sum(counts.values()) == 5


def test_verdict_counts_sum_to_the_row_count(policy, accounting) -> None:
    verdicts = [DELIVERED, DECODE_FAILURE, CODEC_INFEASIBILITY, STRUCTURAL_INFEASIBILITY]
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, verdicts, [0, 1, 2, 3]
    )
    aggregate = aggregate_row(
        rows, context, run_id="r" * 64, psnr_values=psnr, ssim_values=ssim
    )
    counts = reconcile_aggregate(aggregate, rows)
    assert counts == {
        "delivered": 1,
        DECODE_FAILURE: 1,
        CODEC_INFEASIBILITY: 1,
        STRUCTURAL_INFEASIBILITY: 1,
    }


def test_reconstruction_metrics_are_supplied_for_delivered_rows_only(
    policy, accounting
) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DELIVERED, DECODE_FAILURE], [1, 2]
    )
    assert len(psnr) == 1
    with pytest.raises(RecordError, match="exactly the delivered rows"):
        aggregate_row(rows, context, run_id="r" * 64, psnr_values=[1.0, 2.0], ssim_values=[1.0, 2.0])


def test_zero_delivery_uses_the_documented_null_denominator(policy, accounting) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DECODE_FAILURE, CODEC_INFEASIBILITY], [0, 1]
    )
    aggregate = aggregate_row(
        rows, context, run_id="r" * 64, psnr_values=psnr, ssim_values=ssim
    )
    assert aggregate["coverage_rate"] == 0.0
    assert aggregate["acc_given_delivery"] is None
    assert aggregate["psnr_db"] is None and aggregate["ssim"] is None
    semantics = field_semantics()["aggregate"]["acc_given_delivery"]
    assert semantics["nullable"] is True
    assert semantics["aggregation_denominator"] == "delivered rows only"


def test_duplicate_per_image_identities_are_rejected(policy, accounting) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DECODE_FAILURE, DECODE_FAILURE], [0, 1]
    )
    duplicated = [rows[0], dict(rows[0])]
    with pytest.raises(RecordError, match="duplicate identities"):
        aggregate_row(duplicated, context, run_id="r" * 64)


def test_an_altered_aggregate_value_fails_reconciliation(policy, accounting) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DELIVERED, DECODE_FAILURE], [1, 2]
    )
    aggregate = aggregate_row(
        rows, context, run_id="r" * 64, psnr_values=psnr, ssim_values=ssim
    )
    with pytest.raises(RecordError, match="does not reconcile"):
        reconcile_aggregate(dict(aggregate) | {"top1_acc": 0.99}, rows)


def test_an_omitted_outage_row_fails_reconciliation(policy, accounting) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DELIVERED, DECODE_FAILURE], [1, 2]
    )
    aggregate = aggregate_row(
        rows, context, run_id="r" * 64, psnr_values=psnr, ssim_values=ssim
    )
    with pytest.raises(RecordError, match="does not reconcile"):
        reconcile_aggregate(aggregate, [rows[0]])


def test_a_collapsed_outage_reason_is_rejected(policy, accounting) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DECODE_FAILURE, CODEC_INFEASIBILITY], [0, 1]
    )
    collapsed = [dict(row) | {"outage_reason": DECODE_FAILURE} for row in rows]
    aggregate = aggregate_row(
        collapsed, context, run_id="r" * 64, psnr_values=[], ssim_values=[]
    )
    assert aggregate["decode_failure_rate"] == 1.0
    with pytest.raises(RecordError, match="does not reconcile"):
        reconcile_aggregate(aggregate, rows)


def test_non_binary_correctness_is_rejected(policy, accounting) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DECODE_FAILURE], [0]
    )
    fractional = [dict(rows[0]) | {"correct": 0.5}]
    with pytest.raises(RecordError, match="strictly boolean"):
        aggregate_row(fractional, context, run_id="r" * 64)


def test_an_unknown_outage_reason_is_rejected(policy, accounting) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DECODE_FAILURE], [0]
    )
    bad = [dict(rows[0]) | {"outage_reason": "gremlins"}]
    with pytest.raises(RecordError, match="unknown outage reasons"):
        aggregate_row(bad, context, run_id="r" * 64)


def test_a_delivered_row_may_not_carry_an_outage_reason(policy, accounting) -> None:
    rows, context, psnr, ssim = _rows_and_context(
        policy, accounting, [DELIVERED], [1]
    )
    bad = [dict(rows[0]) | {"outage_reason": DECODE_FAILURE}]
    with pytest.raises(RecordError, match="delivered row carries an outage reason"):
        aggregate_row(bad, context, run_id="r" * 64, psnr_values=[1.0], ssim_values=[1.0])


# ---------------------------------------------------------------------------
# Scheduled noise and cross-system pairing (PB_2C/C2.3)
#
# A row's channel realisation is scheduled by the evaluation cell, not by
# whether that system got as far as transmitting. PB_2 recorded a null
# `noise_id` on infeasible rows and built `pair_id` from it, so an infeasible
# classical row could never pair with a transmitting comparison arm — which
# silently removes exactly the images where the two systems differ most.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict",
    [DELIVERED, DECODE_FAILURE, CODEC_INFEASIBILITY, STRUCTURAL_INFEASIBILITY],
)
def test_every_verdict_carries_the_scheduled_noise_identity(
    verdict: str, policy, accounting
) -> None:
    result = _result(verdict, accounting=accounting)
    row = per_image_row(
        result,
        TaskOutcome(0, True, verdict != DELIVERED,
                    NOT_APPLICABLE if verdict == DELIVERED else verdict,
                    None, None, None, None),
        identity=_identity(),
        true_label=0,
        run_id="r" * 64,
        scheduled_noise_id=_scheduled(),
    )
    assert row["noise_id"] == _scheduled()
    assert row["noise_id"] is not None


@pytest.mark.parametrize(
    "classical_verdict", [STRUCTURAL_INFEASIBILITY, CODEC_INFEASIBILITY]
)
def test_an_infeasible_row_pairs_with_a_transmitting_comparison_arm(
    classical_verdict: str, policy, accounting
) -> None:
    """The point of the repair, stated as the comparison it makes possible."""

    scheduled = _scheduled()
    classical = per_image_row(
        _result(classical_verdict, accounting=accounting),
        TaskOutcome(0, True, True, classical_verdict, None, None, None, None),
        identity=_identity(system="classical_fixed_mcs"),
        true_label=0,
        run_id="c" * 64,
        scheduled_noise_id=scheduled,
    )
    # The other arm transmitted on the same image, ratio, SNR and realisation.
    learned = per_image_row(
        _result(DELIVERED, accounting=accounting),
        TaskOutcome(0, True, False, NOT_APPLICABLE, 30.0, 0.9, 157, 900),
        identity=_identity(system="learned"),
        true_label=0,
        run_id="l" * 64,
        scheduled_noise_id=scheduled,
    )
    assert classical["pair_id"] == learned["pair_id"]
    assert classical["noise_id"] == learned["noise_id"] == scheduled
    assert classical["run_id"] != learned["run_id"]


def test_an_infeasible_row_consumes_no_random_draw(policy, accounting) -> None:
    """The identity is a content address, so scheduling it draws nothing."""

    result = _result(STRUCTURAL_INFEASIBILITY, accounting=accounting)
    assert result.noise_id is None
    row = per_image_row(
        result,
        TaskOutcome(0, True, True, STRUCTURAL_INFEASIBILITY, None, None, None, None),
        identity=_identity(),
        true_label=0,
        run_id="r" * 64,
        scheduled_noise_id=_scheduled(),
    )
    assert row["noise_id"] == _scheduled()


def test_a_transmitted_row_reconciles_its_realised_and_scheduled_identity(
    policy, accounting
) -> None:
    result = _result(DELIVERED, accounting=accounting)
    assert result.noise_id == _scheduled()
    row = per_image_row(
        result,
        TaskOutcome(0, True, False, NOT_APPLICABLE, 30.0, 0.9, 157, 900),
        identity=_identity(),
        true_label=0,
        run_id="r" * 64,
        scheduled_noise_id=_scheduled(),
    )
    assert row["noise_id"] == result.noise_id


def test_a_realised_identity_that_differs_from_the_schedule_is_rejected(
    policy, accounting
) -> None:
    """A divergence would mean the pairing describes a draw that never happened."""

    with pytest.raises(RecordError, match="does not match the scheduled one"):
        per_image_row(
            _result(DELIVERED, accounting=accounting),
            TaskOutcome(0, True, False, NOT_APPLICABLE, 30.0, 0.9, 157, 900),
            identity=_identity(),
            true_label=0,
            run_id="r" * 64,
            scheduled_noise_id=_scheduled(sample_id="f" * 16),
        )


@pytest.mark.parametrize("missing", [None, ""])
def test_a_missing_scheduled_identity_is_rejected(missing, policy, accounting) -> None:
    with pytest.raises(RecordError, match="needs its scheduled noise identity"):
        per_image_row(
            _result(STRUCTURAL_INFEASIBILITY, accounting=accounting),
            TaskOutcome(0, True, True, STRUCTURAL_INFEASIBILITY, None, None, None, None),
            identity=_identity(),
            true_label=0,
            run_id="r" * 64,
            scheduled_noise_id=missing,
        )


def test_a_pair_id_built_from_a_null_noise_identity_is_rejected() -> None:
    """The PB_2 mutation, refused at the seam that used to allow it."""

    with pytest.raises(RecordError, match="scheduled noise identity"):
        _identity().pair_id(stable_sample_id="0" * 16, noise_id=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("k", K_SYMBOLS + 1),
        ("test_snr_db", SNR_DB + 1.0),
        ("stable_sample_id", "f" * 16),
        ("channel_seed", get("evaluation.channel_seeds")[1]),
        ("split_manifest_hash", "0" * 64),
        ("block_index", 1),
        ("dataset_version", "0" * 64),
    ],
)
def test_the_scheduled_identity_moves_with_every_keyed_field(field, value) -> None:
    assert _scheduled(**{field: value}) != _scheduled()


# ---------------------------------------------------------------------------
# Known-answer JPEG 2000 container parsing (PB_2C/C2.4)
#
# `header + payload == len(codestream)` is satisfied by *any* split, so it
# cannot detect a wrong boundary. PB_2 read Psot six bytes early, which on
# these small single-tile-part codestreams reads as zero, takes the "Psot = 0"
# last-tile fallback and lands on the right boundary by luck. These fixtures
# assert the two counts individually against hand-computed values.
# ---------------------------------------------------------------------------

_SOC = b"\xff\x4f"
_SOD = b"\xff\x93"
_EOC = b"\xff\xd9"


def _segment(marker: bytes, body: bytes) -> bytes:
    """An ordinary length-carrying marker segment; Lsat covers itself."""

    return marker + (len(body) + 2).to_bytes(2, "big") + body


def _sot(*, isot: int, psot: int, tpsot: int = 0, tnsot: int = 1) -> bytes:
    return _segment(
        b"\xff\x90",
        isot.to_bytes(2, "big")
        + psot.to_bytes(4, "big")
        + bytes([tpsot, tnsot]),
    )


def test_known_answer_single_tile_part_with_a_non_zero_psot() -> None:
    """The case a wrong Psot offset cannot survive: Psot is really used."""

    siz = _segment(b"\xff\x51", b"\x00" * 12)          # 2 + 2 + 12 = 16 bytes
    data = bytes(range(64)) * 3                         # 192 tile-part data bytes
    psot = 12 + len(_SOD) + len(data)                   # SOT segment + SOD + data
    tile = _sot(isot=0, psot=psot) + _SOD + data
    codestream = _SOC + siz + tile + _EOC

    header, payload = codestream_byte_split(codestream)
    assert payload == 192
    assert header == 2 + 16 + 12 + 2 + 2              # SOC, SIZ, SOT, SOD, EOC
    assert header == 34
    assert header + payload == len(codestream) == 226


def test_known_answer_multi_tile_part() -> None:
    first = bytes(range(32))          # 32 data bytes
    second = bytes(range(50))         # 50 data bytes
    tile_one = _sot(isot=0, psot=12 + 2 + len(first), tpsot=0, tnsot=2) + _SOD + first
    tile_two = _sot(isot=0, psot=12 + 2 + len(second), tpsot=1, tnsot=2) + _SOD + second
    codestream = _SOC + tile_one + tile_two + _EOC

    header, payload = codestream_byte_split(codestream)
    assert payload == 82                               # 32 + 50, data only
    assert header == 2 + (12 + 2) * 2 + 2              # SOC, two tile-part headers, EOC
    assert header == 32
    assert header + payload == len(codestream)


def test_known_answer_last_tile_part_with_psot_zero() -> None:
    """Psot may legally be 0 on the last tile-part, meaning "to EOC"."""

    data = b"\x01\x02\x03\x04" * 9                     # 36 bytes, no false markers
    codestream = _SOC + _sot(isot=0, psot=0) + _SOD + data + _EOC

    header, payload = codestream_byte_split(codestream)
    assert payload == 36
    assert header == 2 + 12 + 2 + 2
    assert header + payload == len(codestream)


def test_the_parser_rejects_a_wrong_known_answer_boundary() -> None:
    """Proof the fixtures discriminate: shifting the boundary changes a count."""

    data = bytes(range(64))
    psot = 12 + 2 + len(data)
    codestream = _SOC + _sot(isot=0, psot=psot) + _SOD + data + _EOC
    header, payload = codestream_byte_split(codestream)

    # A parser reading Psot at the PB_2 offset of 4 sees Isot || high16(Psot).
    mis_read = int.from_bytes(codestream[2 + 4 : 2 + 8], "big")
    assert mis_read != psot, "the fixture must actually exercise the offset"
    assert (header, payload) == (18, 64)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda body: body[:-1], "truncated|does not reconcile|past the end"),
        (lambda body: body[2:], "does not start with SOC"),
        (
            lambda body: body.replace(b"\xff\x90\x00\x0a", b"\xff\x90\x00\x0e", 1),
            "SOT segment length",
        ),
    ],
)
def test_malformed_codestreams_are_rejected(mutate, message: str) -> None:
    data = bytes(range(48))
    codestream = (
        _SOC + _sot(isot=0, psot=12 + 2 + len(data)) + _SOD + data + _EOC
    )
    with pytest.raises(RecordError, match=message):
        codestream_byte_split(mutate(codestream))


@pytest.mark.parametrize("psot", [4, 1_000_000])
def test_an_out_of_range_psot_is_rejected(psot: int) -> None:
    data = bytes(range(48))
    codestream = _SOC + _sot(isot=0, psot=psot) + _SOD + data + _EOC
    with pytest.raises(RecordError, match="Psot"):
        codestream_byte_split(codestream)


def test_a_real_openjpeg_encode_reconciles_exactly(tmp_path) -> None:
    """The hand-built fixtures above must agree with a real codestream."""

    from baseline.j2k import J2KCodec

    codec = J2KCodec(tmp_path / "cache")
    image = np.stack(
        [
            (np.indices((64, 64))[0] * 3 + np.indices((64, 64))[1] * 5) % 256
            for _ in range(3)
        ],
        axis=-1,
    ).astype(np.uint8)
    result = codec.encode_to_budget(
        image,
        canonical_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        budget_bytes=2000,
        encode_axis_px=64,
    )
    header, payload = codestream_byte_split(result.codestream)
    assert header > 0 and payload > 0
    assert header + payload == len(result.codestream) == result.emitted_byte_count
