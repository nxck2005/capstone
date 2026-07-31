#!/usr/bin/env python
"""Crash-resumable bounded W4 classical-baseline integration smoke.

Bounded validation/plumbing integration. NOT the BR-4 full validation sweep,
NOT a G-8 operating-point selection, and NOT test evidence.

The run is a deterministic worklist, and every completed row is appended to a
JSONL partial file and fsynced before the next row starts, so a session killed
at 80% leaves 80% of its rows on disk. Resuming re-validates the source commit,
config hash, checkpoint hash, manifest hash and worklist hash before reusing a
single row: a partial file from another commit or configuration is rejected
rather than silently mixed with new rows.

Two datasets are kept strictly apart. Imagenette-160 rows are task-scored with
the frozen G-1 classifier and the frozen outage policy. CIFAR-10 rows are a
transport, verdict, accounting and cache plumbing smoke with **no** classifier
inference and no task accuracy at all — the frozen checkpoint is an
Imagenette-160 model, and ten equal output indices are not a shared class
vocabulary.

Usage:
    .venv/bin/python tools/run_classical_baseline_w4_smoke.py
    .venv/bin/python tools/run_classical_baseline_w4_smoke.py --max-rows 6
    .venv/bin/python tools/run_classical_baseline_w4_smoke.py --restart
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.classical.outage import (  # noqa: E402
    EVIDENCE_LABELS,
    keyed_uniform_random_label,
    load_outage_policy,
    write_json_atomically,
)
from baseline.classical.pipeline import (  # noqa: E402
    DELIVERED,
    ChannelIdentity,
    run_classical_pipeline,
)
from baseline.classical.records import (  # noqa: E402
    FROZEN_CLASSIFIER_DATASET,
    AggregateContext,
    codestream_byte_split,
    RunIdentity,
    noise_identity,
    aggregate_row,
    aggregate_schema,
    field_semantics,
    per_image_row,
    per_image_schema,
    reconcile_aggregate,
    score_result,
)
from baseline.j2k import J2KCodec  # noqa: E402
from env import assert_j2k_runtime, loaded_openjpeg_version  # noqa: E402
from config.params import get  # noqa: E402
from config.run_config import canonical_sha256, config_hash, load_experiment  # noqa: E402
from data.preprocessing import codec_input  # noqa: E402
from data.registry import load_dataset, manifest_sha256  # noqa: E402
from models.frozen_reference_classifier import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_HASH,
    load_frozen_reference_classifier,
)

PLAN_PATH = Path("configs/classical-baseline-w4-smoke-plan.yaml")

PROMINENT_DECLARATION = (
    "Bounded validation/plumbing integration for the W4 classical baseline. "
    "This is not the BR-4 full validation sweep, not a G-8 operating-point "
    "selection, and not test evidence. No operating ratio was selected, no "
    "candidate configurations were compared, no model was trained or "
    "fine-tuned, and the test split was never opened. Accuracy figures here are "
    "plumbing observations at the stated sample size, not experimental results "
    "and not estimates of test performance."
)

CIFAR_DECLARATION = (
    "transport-only plumbing smoke\n"
    "no task accuracy\n"
    "no frozen-classifier inference\n"
    "not comparable to Imagenette task labels"
)


class SmokeError(RuntimeError):
    """A bounded-run contract violation."""


# ---------------------------------------------------------------------------
# Small durable-IO helpers
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise SmokeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one row and fsync it, so an abrupt exit cannot lose it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv_atomically(path: Path, schema: tuple[str, ...], rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(schema), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row[field]) for field in schema})
    body = buffer.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(body).hexdigest()


def _write_bytes_atomically(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_value(value: Any) -> Any:
    """The documented not-applicable representation is an empty CSV cell."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


# ---------------------------------------------------------------------------
# Plan and worklist
# ---------------------------------------------------------------------------


def load_plan(repo_root: Path = REPO) -> dict[str, Any]:
    body = yaml.safe_load((repo_root / PLAN_PATH).read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise SmokeError("smoke plan must be a mapping")
    if tuple(body.get("evidence_labels", ())) != EVIDENCE_LABELS:
        raise SmokeError("smoke plan carries the wrong evidence labels")
    return body


#: Loading a dataset re-verifies the extraction and re-parses the manifest, so
#: it is cached per (dataset, split) rather than repeated once per row.
_DATASETS: dict[tuple[str, str], Any] = {}
_SAMPLE_INDEX: dict[tuple[str, str], dict[str, int]] = {}


def _dataset(dataset: str, split: str):
    key = (dataset, split)
    if key not in _DATASETS:
        data = load_dataset(dataset, split)
        _DATASETS[key] = data
        _SAMPLE_INDEX[key] = {
            data.source_sample(index).stable_sample_id: index
            for index in range(len(data))
        }
    return _DATASETS[key]


def _ordered_samples(dataset: str, split: str, count: int) -> list[tuple[str, int]]:
    """Deterministic stable-ID order, never loader order."""

    data = _dataset(dataset, split)
    pairs = []
    for index in range(len(data)):
        sample = data.source_sample(index)
        pairs.append((sample.stable_sample_id, sample.label))
    pairs.sort()
    if count > len(pairs):
        raise SmokeError(f"{dataset}/{split}: asked for {count} of {len(pairs)} samples")
    return pairs[:count]


def build_worklist(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """One deterministic, fully-keyed unit of work per row."""

    work: list[dict[str, Any]] = []

    def extend(group: str, spec: dict[str, Any], snrs: list[float], task_scored: bool) -> None:
        samples = _ordered_samples(spec["dataset"], spec["split"], int(spec["sample_count"]))
        for snr in snrs:
            for stable_sample_id, label in samples:
                work.append(
                    {
                        "group": group,
                        "dataset": spec["dataset"],
                        "split": spec["split"],
                        "stable_sample_id": stable_sample_id,
                        "true_label": int(label),
                        "bw_ratio": spec["bw_ratio"],
                        "modulation": spec["modulation"],
                        "ldpc_rate": spec["ldpc_rate"],
                        "encode_axis_px": spec.get("encode_axis_px"),
                        "test_snr_db": float(snr),
                        "train_seed": int(spec["train_seed"]),
                        "channel_seed": int(spec["channel_seed"]),
                        "block_index": int(spec.get("block_index", 0)),
                        "task_scored": task_scored,
                        "experiment_config": spec["experiment_config"],
                    }
                )

    cifar = plan["cifar10_transport_only"]
    extend("cifar10_transport_only", cifar, list(cifar["snr_db"]), False)
    imagenette = plan["imagenette160_task_scored"]
    extend("imagenette160_task_scored", imagenette, list(imagenette["snr_db"]), True)
    for fixture in plan["fixtures"]:
        extend(
            f"fixture:{fixture['name']}",
            fixture,
            [float(fixture["snr_db"])],
            bool(fixture.get("task_scored", False)),
        )
    for item in work:
        item["work_id"] = canonical_sha256(
            {key: value for key, value in item.items() if key != "true_label"}
        )
    if len({item["work_id"] for item in work}) != len(work):
        raise SmokeError("worklist contains duplicate work identities")
    return work


def worklist_hash(work: list[dict[str, Any]]) -> str:
    return canonical_sha256([item["work_id"] for item in work])


# ---------------------------------------------------------------------------
# Per-cell configuration
# ---------------------------------------------------------------------------


def cell_key(item: dict[str, Any]) -> tuple[str, int, int, float]:
    """The identity of the configuration a row runs under.

    A cell is one experiment file plus one point on each declared sweep axis.
    Everything else that changes the row — dataset, ratio, modulation, LDPC
    rate, encode axis — is fixed *inside* the experiment file, so two rows
    sharing this key genuinely share a resolved configuration.
    """

    return (
        item["experiment_config"],
        int(item["train_seed"]),
        int(item["channel_seed"]),
        float(item["test_snr_db"]),
    )


def resolve_cell_configs(work: list[dict[str, Any]]) -> dict[tuple, tuple[Any, str]]:
    """One concrete ``RunConfig`` per distinct cell, hashed centrally.

    PB_2 resolved a single configuration at 18 dB and reused its hash for the
    −8 dB cell, both fixtures and the CIFAR-10 rows, so `config_hash` named a
    cell most rows were not in.  Resolving per cell is the repair, and it goes
    through the ordinary `load_experiment`/`config_hash` pair rather than any
    private formula.
    """

    configs: dict[tuple, tuple[Any, str]] = {}
    for item in work:
        key = cell_key(item)
        if key in configs:
            continue
        source, train_seed, channel_seed, test_snr_db = key
        run_config = load_experiment(
            source,
            train_seed=train_seed,
            channel_seed=channel_seed,
            test_snr_db=int(test_snr_db) if test_snr_db.is_integer() else test_snr_db,
        )
        resolved = run_config.resolved
        # The fingerprint must agree with the work item it will be attached to,
        # or the archive would describe a different cell than the one that ran.
        for field, expected in (
            ("dataset", item["dataset"]),
            ("split", item["split"]),
            ("bw_ratio", item["bw_ratio"]),
            ("modulation", item["modulation"]),
            ("ldpc_rate", item["ldpc_rate"]),
            ("encode_axis_px", item["encode_axis_px"]),
        ):
            if resolved[field] != expected:
                raise SmokeError(
                    f"{source}: resolved {field}={resolved[field]!r} but the plan "
                    f"row for {item['group']} says {expected!r}"
                )
        if float(resolved["test_snr_db"]) != float(item["test_snr_db"]):
            raise SmokeError(f"{source}: resolved test_snr_db disagrees with the plan")
        configs[key] = (run_config, config_hash(run_config))
    return configs


def config_hash_root(configs: dict[tuple, tuple[Any, str]]) -> str:
    """One digest over every cell hash, for the resume binding.

    Deliberately not a substitute for the per-cell hashes: it exists so an
    interrupted run cannot resume against a differently-configured worklist.
    """

    return canonical_sha256(
        sorted(digest for _config, digest in configs.values())
    )


def write_run_config_artifacts(
    configs: dict[tuple, tuple[Any, str]],
    directory: Path,
    *,
    relative_to: Path | None = None,
) -> list[dict[str, Any]]:
    """Archive each concrete configuration under its own hash, and index them.

    Index paths are relative to the evidence directory, so the bundle describes
    itself and can be verified wherever it is unpacked.
    """

    directory.mkdir(parents=True, exist_ok=True)
    base = relative_to or directory.parent
    index: list[dict[str, Any]] = []
    for key in sorted(configs, key=lambda item: configs[item][1]):
        run_config, digest = configs[key]
        path = directory / f"{digest}.json"
        write_json_atomically(path, run_config.to_dict())
        resolved = run_config.resolved
        index.append(
            {
                "config_hash": digest,
                "relative_path": str(
                    path.relative_to(base) if path.is_relative_to(base) else path.name
                ),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "run_config": run_config.experiment,
                "dataset": resolved["dataset"],
                "bw_ratio": resolved["bw_ratio"],
                "test_snr_db": resolved["test_snr_db"],
                "train_seed": resolved["train_seed"],
                "channel_seed": resolved["channel_seed"],
                "modulation": resolved["modulation"],
                "ldpc_rate": resolved["ldpc_rate"],
                "encode_axis_px": resolved["encode_axis_px"],
            }
        )
    return index


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _identity_for(item: dict[str, Any], *, plan: dict[str, Any], run_config_hash: str,
                  checkpoint_id: str) -> RunIdentity:
    dataset = item["dataset"]
    return RunIdentity(
        system="classical_fixed_mcs",
        dataset=dataset,
        dataset_version=get(f"datasets.{dataset}.{get('config.dataset_version_rule')}"),
        split=item["split"],
        split_manifest_hash=manifest_sha256(dataset),
        bw_ratio=item["bw_ratio"],
        test_snr_db=item["test_snr_db"],
        # Per row, not per run: PB_2 read one group's train seed for every row.
        train_seed=int(item["train_seed"]),
        channel_seed=item["channel_seed"],
        config_hash=run_config_hash,
        checkpoint_id=checkpoint_id,
        classifier_variant=get("reference_classifier.clean_variant_name"),
        ldpc_rate=item["ldpc_rate"],
        modulation=item["modulation"],
        quantiser_bits=None,
        transmit_dim=None,
        reconstruction_weight=None,
        analysis_version=get("config.analysis_version"),
    )


def run_row(
    item: dict[str, Any],
    *,
    plan: dict[str, Any],
    codec: J2KCodec,
    policy,
    classifier,
    run_config_hash: str,
    checkpoint_id: str,
    device: str,
) -> dict[str, Any]:
    """Execute one worklist row and return its durable record."""

    started = time.perf_counter()
    dataset = item["dataset"]
    data = _dataset(dataset, item["split"])
    index = _SAMPLE_INDEX[(dataset, item["split"])].get(item["stable_sample_id"])
    if index is None:
        raise SmokeError(f"{dataset}: sample {item['stable_sample_id']} is absent")
    product, _label = data[index]

    identity = _identity_for(
        item, plan=plan, run_config_hash=run_config_hash, checkpoint_id=checkpoint_id
    )
    channel_identity = ChannelIdentity(
        dataset_version=identity.dataset_version,
        split_manifest_hash=identity.split_manifest_hash,
        channel_seed=item["channel_seed"],
    )
    k_symbols = get(f"bandwidth.k_symbols.{dataset}.{item['bw_ratio']}")
    # Scheduled by the evaluation cell, not by whether the row got to transmit.
    # Computing it draws nothing: it is a content address over the cell, so an
    # infeasible row and a transmitting comparison arm agree on it.
    scheduled_noise_id = noise_identity(
        dataset_version=identity.dataset_version,
        split_manifest_hash=identity.split_manifest_hash,
        stable_sample_id=item["stable_sample_id"],
        test_snr_db=item["test_snr_db"],
        channel_seed=item["channel_seed"],
        k=k_symbols,
        block_index=item["block_index"],
    )
    result = run_classical_pipeline(
        product,
        dataset=dataset,
        k_symbols=k_symbols,
        modulation=item["modulation"],
        ldpc_rate=item["ldpc_rate"],
        snr_db=item["test_snr_db"],
        codec=codec,
        channel_identity=channel_identity,
        encode_axis_px=item["encode_axis_px"],
        block_index=item["block_index"],
        device=device,
    )

    record: dict[str, Any] = {
        "work_id": item["work_id"],
        "group": item["group"],
        "task_scored": item["task_scored"],
        "verdict": result.verdict,
        "wall_clock_s": None,  # set once the whole row path has been walked
        "summary": result.summary(),
        "k_symbols": k_symbols,
        "config_hash": run_config_hash,
        # Three separate facts, deliberately not collapsed into one field: what
        # the cell scheduled, what the channel actually drew, and whether a draw
        # happened at all. An infeasible row keeps the first and honestly
        # reports the other two as absent.
        "scheduled_noise_id": scheduled_noise_id,
        "actual_noise_id": result.noise_id,
        "noise_consumed": result.noise_id is not None,
    }
    if result.source_coding is not None:
        record["source_coding"] = {
            "encode_axis_px": result.source_coding.encode_axis_px,
            "payload_capacity_bytes": result.source_coding.payload_capacity_bytes,
            "emitted_bytes": result.source_coding.emitted_bytes,
            "payload_filler_bytes": result.source_coding.payload_filler_bytes,
            "codestream_sha256": result.source_coding.codestream_sha256,
            "cache_key": result.source_coding.cache_key,
            "cache_hit": result.source_coding.cache_hit,
            "axis_reasons": [list(pair) for pair in result.source_coding.axis_reasons],
        }
        # The container/data split is a property of the emitted codestream, not
        # of the task, so it is recorded for CIFAR-10 transport-only rows too:
        # BR-11's overhead fraction is exactly what that smoke exists to expose.
        codestream = result.source_coding.emitted_codestream
        if isinstance(codestream, bytes):
            container, data = codestream_byte_split(codestream)
            record["source_coding"]["header_bytes"] = container
            record["source_coding"]["payload_bytes"] = data
    if result.transport is not None:
        record["transport"] = {
            "realised_symbol_energy": result.transport.realised_symbol_energy,
            "papr_db": result.transport.papr_db,
            "crc_ok": result.transport.crc_ok,
            "unit_noise_sha256": result.transport.unit_noise_sha256,
            "per_packet_power_rescaling_applied": (
                result.transport.per_packet_power_rescaling_applied
            ),
            "interleaver": result.transport.interleaver,
        }
    record["codestream_recovered_exactly"] = result.codestream_recovered_exactly

    if not item["task_scored"]:
        # CIFAR-10 and the transport-only fixture: no classifier, no task score.
        record["task"] = None
        record["wall_clock_s"] = time.perf_counter() - started
        return record

    if dataset != FROZEN_CLASSIFIER_DATASET:
        raise SmokeError(
            f"refusing to task-score {dataset!r} with the "
            f"{FROZEN_CLASSIFIER_DATASET!r} frozen classifier"
        )
    canonical = codec_input(product)
    outcome = score_result(
        result,
        true_label=item["true_label"],
        policy=policy,
        canonical_image=canonical if result.verdict == DELIVERED else None,
        classifier=classifier if result.verdict == DELIVERED else None,
        device=device,
    )
    row = per_image_row(
        result,
        outcome,
        identity=identity,
        true_label=item["true_label"],
        run_id=identity.run_id(),
        scheduled_noise_id=scheduled_noise_id,
    )
    record["per_image"] = row
    record["task"] = {
        "psnr_db": outcome.psnr_db,
        "ssim": outcome.ssim,
        "header_bytes": outcome.header_bytes,
        "payload_bytes": outcome.payload_bytes,
        # Secondary comparison only; never the primary outcome.
        "sensitivity_label": keyed_uniform_random_label(
            split_manifest_hash=identity.split_manifest_hash,
            stable_sample_id=item["stable_sample_id"],
            channel_seed=item["channel_seed"],
            n_classes=policy.class_count,
        ),
    }
    # Stamped last, deliberately: the row's cost includes source coding,
    # transport, classifier inference, reconstruction metrics, outage scoring
    # and record construction. PB_2 stopped the clock at the end of
    # `run_classical_pipeline`, so the classifier -- the most expensive part of
    # a delivered row -- was never counted.
    record["wall_clock_s"] = time.perf_counter() - started
    return record


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def _progress_identity(
    *,
    source_commit: str,
    config_hash_root: str,
    checkpoint_id: str,
    manifest_hashes: dict[str, str],
    work_hash: str,
    plan_sha256: str,
) -> dict[str, Any]:
    return {
        "source_commit": source_commit,
        "config_hash_root": config_hash_root,
        "checkpoint_id": checkpoint_id,
        "manifest_sha256": manifest_hashes,
        "worklist_sha256": work_hash,
        "plan_sha256": plan_sha256,
    }


def _load_partial(
    partial_path: Path, progress_path: Path, identity: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Reuse durable rows only when every binding still validates."""

    if not partial_path.is_file() or not progress_path.is_file():
        return {}
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    recorded = progress.get("identity")
    if recorded != identity:
        differing = sorted(
            key
            for key in set(recorded or {}) | set(identity)
            if (recorded or {}).get(key) != identity.get(key)
        )
        raise SmokeError(
            "refusing to resume: the partial run was produced under a different "
            f"binding ({differing}). Delete the partial files or use --restart."
        )
    rows: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(
        partial_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        work_id = row.get("work_id")
        if not isinstance(work_id, str) or not work_id:
            raise SmokeError(f"partial row {number} has no work_id")
        if work_id in rows:
            raise SmokeError(f"partial file duplicates work_id {work_id}")
        rows[work_id] = _restore_row(row)
    return rows


def _restore_row(row: dict[str, Any]) -> dict[str, Any]:
    """Undo the JSONL key sort for the one field whose order is normative.

    Partial rows are written with ``sort_keys=True`` so the file's bytes are
    deterministic, which alphabetises the per-image record. The schema's field
    *order* is part of the contract, so it is restored here rather than
    weakening ``validate_row`` to ignore order.
    """

    record = row.get("per_image")
    if isinstance(record, dict):
        schema = per_image_schema()
        missing = sorted(set(schema) - set(record))
        unexpected = sorted(set(record) - set(schema))
        if missing or unexpected:
            raise SmokeError(
                f"partial per-image row fields differ: missing={missing}, "
                f"unexpected={unexpected}"
            )
        row["per_image"] = {field: record[field] for field in schema}
    return row


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------


def finalise(
    *,
    plan: dict[str, Any],
    work: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    identity_binding: dict[str, Any],
    timestamp: str,
    cell_configs: dict[tuple, tuple[Any, str]],
    openjpeg_version: str,
    wall_clock_s: float,
    evidence_dir: Path,
    git_dirty: bool,
) -> dict[str, Any]:
    """Deterministically build final evidence from the validated partial rows."""

    missing = [item["work_id"] for item in work if item["work_id"] not in rows]
    if missing:
        raise SmokeError(f"{len(missing)} worklist rows are still missing")

    ordered = [rows[item["work_id"]] for item in work]
    task_rows = [row for row in ordered if row.get("per_image") is not None]
    cifar_rows = [row for row in ordered if row["group"] == "cifar10_transport_only"]

    per_image = [row["per_image"] for row in task_rows]
    per_image_hash = _write_csv_atomically(
        evidence_dir / "per_image.csv", per_image_schema(), per_image
    )

    # One aggregate per (group, SNR) cell; the two SNR points are reported
    # separately and never pooled, because pooling them would invent a
    # comparison this bounded run does not make.
    aggregates: list[dict[str, Any]] = []
    cells: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for item, row in zip(work, ordered, strict=True):
        if row.get("per_image") is None:
            continue
        cells.setdefault((item["group"], item["test_snr_db"]), []).append(row)

    for (group, snr), group_rows in sorted(cells.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        first_item = next(
            item
            for item in work
            if item["group"] == group and item["test_snr_db"] == snr
        )
        dataset = first_item["dataset"]
        identity = _identity_for(
            first_item,
            plan=plan,
            run_config_hash=cell_configs[cell_key(first_item)][1],
            checkpoint_id=identity_binding["checkpoint_id"],
        )
        delivered = [row for row in group_rows if row["verdict"] == DELIVERED]
        accounting = next(
            (row["summary"]["accounting"] for row in group_rows if row["summary"]["accounting"]),
            None,
        )
        context = AggregateContext(
            identity=identity,
            k_symbols=group_rows[0]["k_symbols"],
            timestamp=timestamp,
            git_commit=identity_binding["source_commit"],
            git_dirty=git_dirty,
            source_codec=get("baseline.source_codec"),
            j2k_target_bytes=(accounting or {}).get("payload_bytes"),
            wall_clock_s=sum(row["wall_clock_s"] for row in group_rows),
            peak_vram_gb=None,
            tb_crc_type=(accounting or {}).get("tb_crc_name"),
            base_graph=(accounting or {}).get("base_graph"),
            lifting_size=(accounting or {}).get("lifting_size"),
            num_codeblocks=(accounting or {}).get("code_blocks"),
            filler_bits=(accounting or {}).get("ldpc_filler_bits_total"),
            effective_code_rate=(
                (accounting or {}).get("k_prime") / max((accounting or {}).get("rate_matched_bits"))
                if accounting
                else None
            ),
            bytes_sent=(accounting or {}).get("payload_bytes"),
            papr_db_values=tuple(
                row["transport"]["papr_db"] for row in group_rows if row.get("transport")
            ),
            # BR-11/AM-81: every row that emitted a codestream, which includes
            # decode failures. Restricting these to delivered rows made an
            # all-decode-failure cell report no overhead at all.
            header_bytes_values=tuple(
                row["source_coding"]["header_bytes"]
                for row in group_rows
                if (row.get("source_coding") or {}).get("header_bytes") is not None
            ),
            payload_bytes_values=tuple(
                row["source_coding"]["payload_bytes"]
                for row in group_rows
                if (row.get("source_coding") or {}).get("payload_bytes") is not None
            ),
        )
        aggregate = aggregate_row(
            [row["per_image"] for row in group_rows],
            context,
            run_id=identity.run_id(),
            psnr_values=[row["task"]["psnr_db"] for row in delivered],
            ssim_values=[row["task"]["ssim"] for row in delivered],
        )
        reconcile_aggregate(aggregate, [row["per_image"] for row in group_rows])
        aggregate["_cell"] = {"group": group, "dataset": dataset, "test_snr_db": snr}
        aggregates.append(aggregate)

    # The final, immutable raw-row artifact. The partial file is execution
    # state; verification must read a complete file in deterministic worklist
    # order, so a truncated run can never be mistaken for a finished one.
    final_rows_path = REPO / plan["outputs"]["final_rows"]
    raw_body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in ordered
    ).encode("utf-8")
    _write_bytes_atomically(final_rows_path, raw_body)
    raw_rows_hash = hashlib.sha256(raw_body).hexdigest()

    aggregate_hash = _write_csv_atomically(
        evidence_dir / "aggregate.csv",
        aggregate_schema(),
        [{k: v for k, v in row.items() if k != "_cell"} for row in aggregates],
    )

    summary = {
        "schema_version": 1,
        "complete": True,
        "status": "COMPLETE",
        "evidence_labels": list(EVIDENCE_LABELS),
        "prominent_declaration": PROMINENT_DECLARATION,
        "timestamp": timestamp,
        "wall_clock_s": wall_clock_s,
        "openjpeg_version": openjpeg_version,
        "openjpeg_preflight_preceded_artifacts": True,
        "execution_source_commit": identity_binding["source_commit"],
        "git_dirty": git_dirty,
        "config_hash_root": identity_binding["config_hash_root"],
        "config_hashes": {
            "/".join(str(part) for part in key): digest
            for key, (_config, digest) in sorted(
                cell_configs.items(), key=lambda kv: kv[1][1]
            )
        },
        "checkpoint_id": identity_binding["checkpoint_id"],
        "classifier_config_hash": EXPECTED_CONFIG_HASH,
        "manifest_sha256": identity_binding["manifest_sha256"],
        "worklist_sha256": identity_binding["worklist_sha256"],
        "plan_sha256": identity_binding["plan_sha256"],
        "per_image_csv_sha256": per_image_hash,
        "aggregate_csv_sha256": aggregate_hash,
        "raw_rows_sha256": raw_rows_hash,
        "raw_rows_count": len(ordered),
        "br4_sweep_completed": False,
        "g8_status": "unresolved",
        "j2k_resolutions_issue_status": "resolved_by_am80",
        "operating_point_selected": False,
        "training_performed": False,
        "test_split_access": {
            "test_accessed": False,
            "test_inference": False,
            "test_accuracy_computed": False,
            "test_split_sealed": True,
        },
        "cifar10_transport_only": _cifar_summary(cifar_rows, plan),
        "imagenette160_task_scored": _task_summary(aggregates, task_rows, plan),
        "fixtures": _fixture_summary(work, rows, plan),
    }
    return summary


def _cifar_summary(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    spec = plan["cifar10_transport_only"]
    verdicts: dict[str, int] = {}
    for row in rows:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    if any(row.get("per_image") is not None or row.get("task") for row in rows):
        raise SmokeError("a CIFAR-10 row carries a task score")
    return {
        "declaration": CIFAR_DECLARATION,
        "dataset": spec["dataset"],
        "split": spec["split"],
        "sample_count": int(spec["sample_count"]),
        "encode_axis_px": spec["encode_axis_px"],
        "bw_ratio": spec["bw_ratio"],
        "modulation": spec["modulation"],
        "ldpc_rate": spec["ldpc_rate"],
        "snr_db": list(spec["snr_db"]),
        "verdict_counts": verdicts,
        "task_accuracy": None,
        "classifier_inference_performed": False,
        "top1_acc": None,
        "n_correct": None,
        "codestream_exact": [row["codestream_recovered_exactly"] for row in rows],
        "cache_hits": [
            (row.get("source_coding") or {}).get("cache_hit") for row in rows
        ],
        "header_bytes": [
            (row.get("source_coding") or {}).get("header_bytes") for row in rows
        ],
        "payload_bytes": [
            (row.get("source_coding") or {}).get("payload_bytes") for row in rows
        ],
        "accounting": next(
            (row["summary"]["accounting"] for row in rows if row["summary"]["accounting"]),
            None,
        ),
    }


def _task_summary(
    aggregates: list[dict[str, Any]], rows: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    spec = plan["imagenette160_task_scored"]
    cells = []
    for aggregate in aggregates:
        cell = dict(aggregate["_cell"])
        if cell["group"] != "imagenette160_task_scored":
            # A targeted fixture is reported under `fixtures`, never pooled into
            # the bounded subset's accuracy.
            continue
        delivered = aggregate["coverage_rate"] * aggregate["n"]
        cell.update(
            {
                "n": aggregate["n"],
                "sample_count": aggregate["n"],
                "top1_acc": aggregate["top1_acc"],
                "n_correct": aggregate["n_correct"],
                "coverage_rate": aggregate["coverage_rate"],
                "acc_given_delivery": aggregate["acc_given_delivery"],
                "decode_failure_rate": aggregate["decode_failure_rate"],
                "infeasible_rate": aggregate["infeasible_rate"],
                "psnr_db_mean": aggregate["psnr_db"],
                "ssim_mean": aggregate["ssim"],
                "psnr_ssim_denominator": int(round(delivered)),
                "papr_db_mean": aggregate["papr_db"],
                "bytes_sent": aggregate["bytes_sent"],
                "header_bytes_mean": aggregate["header_bytes"],
                "payload_bytes_mean": aggregate["payload_bytes"],
                "run_id": aggregate["run_id"],
            }
        )
        cells.append(cell)
    return {
        "declaration": (
            "Task-scored with the frozen G-1 Imagenette-160 classifier and the "
            "frozen constant-class outage policy. Sample sizes are stated with "
            "every number; these are plumbing observations, not experimental "
            "results."
        ),
        "dataset": spec["dataset"],
        "split": spec["split"],
        "sample_count": int(spec["sample_count"]),
        "sample_selection_rule": plan["sample_selection"]["rule"],
        "bw_ratio": spec["bw_ratio"],
        "modulation": spec["modulation"],
        "ldpc_rate": spec["ldpc_rate"],
        "snr_db": list(spec["snr_db"]),
        "cells": cells,
        "sensitivity_labels_are_secondary": True,
        "cached_j2k_repeat": _cache_summary(rows),
    }


def _cache_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evidence that the second SNR point re-used the first point's codestreams.

    The same image at the same axis and the same payload budget is the same
    encode, so the low-SNR pass must hit the cache and reproduce a byte-identical
    codestream rather than re-encoding. A miss here would mean the cache key is
    keyed on something it should not be.
    """

    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_coding = row.get("source_coding") or {}
        key = source_coding.get("cache_key")
        if key:
            by_key.setdefault(key, []).append(source_coding)
    repeats = {
        key: entries for key, entries in by_key.items() if len(entries) > 1
    }
    return {
        "distinct_cache_keys": len(by_key),
        "repeated_cache_keys": len(repeats),
        "repeats_were_cache_hits": all(
            entry["cache_hit"] for entries in repeats.values() for entry in entries[1:]
        ),
        "repeats_reproduced_identical_codestreams": all(
            len({entry["codestream_sha256"] for entry in entries}) == 1
            for entries in repeats.values()
        ),
    }


def _fixture_summary(
    work: list[dict[str, Any]], rows: dict[str, dict[str, Any]], plan: dict[str, Any]
) -> list[dict[str, Any]]:
    summaries = []
    for fixture in plan["fixtures"]:
        name = f"fixture:{fixture['name']}"
        matching = [rows[item["work_id"]] for item in work if item["group"] == name]
        observed = sorted({row["verdict"] for row in matching})
        if observed != [fixture["expected_verdict"]]:
            raise SmokeError(
                f"{name}: expected {fixture['expected_verdict']}, observed {observed}"
            )
        summaries.append(
            {
                "name": fixture["name"],
                "labelled": "targeted infeasibility fixture, not an ordinary smoke row",
                "dataset": fixture["dataset"],
                "bw_ratio": fixture["bw_ratio"],
                "modulation": fixture["modulation"],
                "ldpc_rate": fixture["ldpc_rate"],
                "expected_verdict": fixture["expected_verdict"],
                "observed_verdict": observed[0],
                "task_scored": bool(fixture.get("task_scored", False)),
                "axis_reasons": [
                    (row.get("source_coding") or {}).get("axis_reasons") for row in matching
                ],
            }
        )
    return summaries


def build_accounting_examples(
    work: list[dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Concrete reconciliations, including one through classifier and outage."""

    examples = []
    seen: set[str] = set()
    for item in work:
        row = rows[item["work_id"]]
        key = f"{item['group']}|{row['verdict']}"
        if key in seen:
            continue
        seen.add(key)
        accounting = row["summary"]["accounting"]
        source_coding = row.get("source_coding") or {}
        example = {
            "group": item["group"],
            "dataset": item["dataset"],
            "stable_sample_id": item["stable_sample_id"],
            "bw_ratio": item["bw_ratio"],
            "modulation": item["modulation"],
            "ldpc_rate": item["ldpc_rate"],
            "test_snr_db": item["test_snr_db"],
            "verdict": row["verdict"],
            "accounting": accounting,
            "source_coding": source_coding or None,
            "transport": row.get("transport"),
            "codestream_recovered_exactly": row["codestream_recovered_exactly"],
        }
        if accounting and source_coding.get("emitted_bytes") is not None:
            header = source_coding.get("header_bytes")
            payload = source_coding.get("payload_bytes")
            example["byte_reconciliation"] = {
                "bytes_sent_A_over_8": accounting["payload_bytes"],
                "emitted_codestream_bytes": source_coding["emitted_bytes"],
                "header_bytes_container": header,
                "payload_bytes_entropy_data": payload,
                "payload_filler_bytes": source_coding["payload_filler_bytes"],
                "reconciles": (
                    header + payload + source_coding["payload_filler_bytes"]
                    == accounting["payload_bytes"]
                    if header is not None and payload is not None
                    else None
                ),
                "channel_bits_G": accounting["channel_bits"],
                "channel_uses_k_symbols": accounting["k_symbols"],
            }
        if row.get("per_image"):
            example["task_path"] = {
                "true_label": row["per_image"]["true_label"],
                "pred_label": row["per_image"]["pred_label"],
                "correct": row["per_image"]["correct"],
                "outage": row["per_image"]["outage"],
                "outage_reason": row["per_image"]["outage_reason"],
                "source_bytes": row["per_image"]["source_bytes"],
                "psnr_db": (row.get("task") or {}).get("psnr_db"),
                "ssim": (row.get("task") or {}).get("ssim"),
                "classifier_ran": not row["per_image"]["outage"],
            }
        examples.append(example)
    return {
        "schema_version": 1,
        "evidence_labels": list(EVIDENCE_LABELS),
        "prominent_declaration": PROMINENT_DECLARATION,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="stop after this many newly executed rows (interruption drill)")
    parser.add_argument("--restart", action="store_true",
                        help="discard any partial run and start over")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    plan = load_plan()

    # Before *anything* is created. `assert_j2k_runtime` used to fire on the
    # first encode, which is after the results directory, the outage policy and
    # the frozen classifier had already been touched -- so a wrong OpenJPEG
    # left a half-built evidence directory behind. SR-21/AM-75 require the
    # check to precede artifact creation; this is the one implementation, not a
    # second version parser.
    assert_j2k_runtime()
    openjpeg_version = loaded_openjpeg_version(required=True)

    evidence_dir = REPO / plan["outputs"]["evidence_dir"]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    partial_path = REPO / plan["outputs"]["partial_rows"]
    progress_path = REPO / plan["outputs"]["progress"]

    source_commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=normal")
    plan_sha256 = hashlib.sha256((REPO / PLAN_PATH).read_bytes()).hexdigest()

    policy = load_outage_policy(
        REPO / plan["outputs"]["outage_policy"],
        expected_dataset=FROZEN_CLASSIFIER_DATASET,
        expected_manifest_sha256=manifest_sha256(FROZEN_CLASSIFIER_DATASET),
    )
    work = build_worklist(plan)
    cell_configs = resolve_cell_configs(work)
    run_config_index = write_run_config_artifacts(
        cell_configs,
        REPO / plan["outputs"]["run_configs_dir"],
        relative_to=evidence_dir,
    )
    manifest_hashes = {
        dataset: manifest_sha256(dataset)
        for dataset in sorted({item["dataset"] for item in work})
    }
    identity_binding = _progress_identity(
        source_commit=source_commit,
        config_hash_root=config_hash_root(cell_configs),
        checkpoint_id=EXPECTED_CHECKPOINT_SHA256,
        manifest_hashes=manifest_hashes,
        work_hash=worklist_hash(work),
        plan_sha256=plan_sha256,
    )

    if arguments.restart:
        partial_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    rows = _load_partial(partial_path, progress_path, identity_binding)
    if rows:
        timestamp = json.loads(progress_path.read_text(encoding="utf-8"))["timestamp"]
        print(f"resuming with {len(rows)} durable rows of {len(work)}")
    else:
        timestamp = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")

    pending = [item for item in work if item["work_id"] not in rows]
    classifier = None
    if pending and any(item["task_scored"] for item in pending):
        classifier = load_frozen_reference_classifier(arguments.device, repo_root=REPO)
    codec = J2KCodec(REPO / plan["cache_dir"])

    started = time.perf_counter()
    executed = 0
    for item in pending:
        if arguments.max_rows is not None and executed >= arguments.max_rows:
            break
        record = run_row(
            item,
            plan=plan,
            codec=codec,
            policy=policy,
            classifier=classifier,
            run_config_hash=cell_configs[cell_key(item)][1],
            checkpoint_id=EXPECTED_CHECKPOINT_SHA256,
            device=arguments.device,
        )
        _append_jsonl(partial_path, record)
        rows[item["work_id"]] = record
        executed += 1
        write_json_atomically(
            progress_path,
            {
                "complete": False,
                "schema_version": 1,
                "evidence_labels": list(EVIDENCE_LABELS),
                "identity": identity_binding,
                "timestamp": timestamp,
                "completed_rows": len(rows),
                "total_rows": len(work),
                "git_dirty": bool(dirty),
            },
        )
        print(
            f"[{len(rows)}/{len(work)}] {item['group']} "
            f"{item['stable_sample_id']} snr={item['test_snr_db']} -> {record['verdict']}"
        )

    if len(rows) < len(work):
        write_json_atomically(
            progress_path,
            {
                "complete": False,
                "schema_version": 1,
                "evidence_labels": list(EVIDENCE_LABELS),
                "identity": identity_binding,
                "timestamp": timestamp,
                "completed_rows": len(rows),
                "total_rows": len(work),
                "git_dirty": bool(dirty),
            },
        )
        print(f"incomplete: {len(rows)}/{len(work)} rows durable, complete=false")
        return 0

    # Derived from every durable row, not from this session's elapsed time.
    # PB_2 measured `perf_counter()` from *after* the resume load, so a run
    # resumed once reported a total smaller than the sum of its own aggregate
    # rows.
    wall_clock = sum(row["wall_clock_s"] for row in rows.values())
    summary = finalise(
        plan=plan,
        work=work,
        rows=rows,
        identity_binding=identity_binding,
        timestamp=timestamp,
        cell_configs=cell_configs,
        openjpeg_version=openjpeg_version,
        wall_clock_s=wall_clock,
        evidence_dir=evidence_dir,
        git_dirty=bool(dirty),
    )
    # The partial file is execution state and has now been superseded by the
    # immutable final one; leaving it behind invites it being read as evidence.
    partial_path.unlink(missing_ok=True)
    write_json_atomically(REPO / plan["outputs"]["summary"], summary)
    write_json_atomically(
        REPO / plan["outputs"]["accounting_examples"],
        build_accounting_examples(work, rows),
    )
    write_json_atomically(
        REPO / plan["outputs"]["resolved_config"],
        {
            "schema_version": 2,
            "complete": True,
            "evidence_labels": list(EVIDENCE_LABELS),
            "prominent_declaration": PROMINENT_DECLARATION,
            "plan_config": str(PLAN_PATH),
            "plan_sha256": plan_sha256,
            "config_hash_root": identity_binding["config_hash_root"],
            # An execution-level *index*, not one configuration pretending to
            # describe every cell. Each entry points at the archived concrete
            # RunConfig that produced its rows.
            "run_configs": run_config_index,
            "field_semantics": field_semantics(),
            "execution_source_commit": source_commit,
            "git_dirty": bool(dirty),
        },
    )
    write_json_atomically(
        progress_path,
        {
            "complete": True,
            "schema_version": 1,
            "evidence_labels": list(EVIDENCE_LABELS),
            "identity": identity_binding,
            "timestamp": timestamp,
            "completed_rows": len(rows),
            "total_rows": len(work),
            "git_dirty": bool(dirty),
        },
    )
    print(f"complete: {len(rows)}/{len(work)} rows; evidence in {plan['outputs']['evidence_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
