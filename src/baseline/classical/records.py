"""Schema-exact W4 per-image and aggregate records, identities and aggregation.

This is the layer PB_1 deliberately stopped short of.  It takes a
``ClassicalResult`` — one image's verdict, accounting, measurements and, when the
link held, its decoded pixels — and turns it into rows conforming *exactly* to
``params.artifacts.per_image_schema`` and ``params.artifacts.csv_schema``.

Three things are worth stating because getting them wrong would be invisible in
the numbers:

**The frozen classifier is dataset-bound.**  The adjudicated G-1 checkpoint is an
Imagenette-160 model.  CIFAR-10 also has ten class indices, but they are a
different vocabulary, so scoring a CIFAR reconstruction with it would produce a
number that looks like an accuracy and means nothing.  ``score_result`` refuses
it.  CIFAR remains a transport/accounting/cache plumbing smoke with no task
score at all.

**Outage rows are scored, not dropped.**  A row the link could not deliver still
gets a prediction — the frozen constant class from ``outage.py`` — because
silently discarding it would inflate the baseline's accuracy by exactly the
fraction of the grid it fails on, which is the fraction that matters.  The
classifier never runs on such a row and no substitute pixels are invented.

**Schemas are read, never restated.**  Field names and order come from
``params.artifacts.*`` at runtime.  A second hand-written list in production code
is precisely the thing that lets a schema change pass unnoticed, so the only
hand-written table here is ``FIELD_SEMANTICS``, which *annotates* the configured
fields and is asserted to cover them exactly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from artifacts.ids import (
    make_analysis_cell_id,
    make_noise_id,
    make_pair_id,
    make_run_id,
)
from baseline.classical.outage import NOT_APPLICABLE, OutagePolicy
from baseline.classical.pipeline import (
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    DELIVERED,
    STRUCTURAL_INFEASIBILITY,
    VERDICTS,
    ClassicalResult,
)
from config.params import get
from data.preprocessing import reconstruction_input, reconstruction_metrics

#: The dataset whose label vocabulary the frozen G-1 checkpoint was trained on.
#: Nothing else may be scored with it.
FROZEN_CLASSIFIER_DATASET = "imagenette160"

#: The four verdicts that are outages, mapped to the ``outage_reason`` value
#: recorded for them.  ``DELIVERED`` is deliberately absent.
OUTAGE_REASONS = {
    STRUCTURAL_INFEASIBILITY: STRUCTURAL_INFEASIBILITY,
    CODEC_INFEASIBILITY: CODEC_INFEASIBILITY,
    DECODE_FAILURE: DECODE_FAILURE,
}

_CHANNEL_MODEL = "awgn"
_J2K_SOC = 0xFF4F
_J2K_SOT = 0xFF90
_J2K_SOD = 0xFF93
_J2K_EOC = 0xFFD9
#: Markers that stand alone: they carry no two-byte length segment.
_J2K_DELIMITERS = frozenset({_J2K_SOC, _J2K_SOD, _J2K_EOC})
_MARKER_BYTES = 2
_LENGTH_BYTES = 2
#: SOT segment layout (ISO/IEC 15444-1 A.4.2):
#:     SOT 2 | Lsot 2 | Isot 2 | Psot 4 | TPsot 1 | TNsot 1
#: so Psot begins **six** bytes into the segment, not four. PB_2 read at 4,
#: which is `Isot || high16(Psot)`; for these small single-tile-part
#: codestreams that reads as zero, silently taking the `Psot = 0` last-tile
#: fallback and landing on the right boundary by luck. A multi-tile-part or
#: large codestream would have been mis-split with no error (AM-81).
_PSOT_OFFSET = 6  # literal-ok: SOT(2) + Lsot(2) + Isot(2) precede Psot
_PSOT_BYTES = 4  # literal-ok: Psot is a 32-bit tile-part length
_SOT_SEGMENT_LENGTH = 10  # literal-ok: Lsot(2)+Isot(2)+Psot(4)+TPsot(1)+TNsot(1)

#: The only splits a record may describe.  Deliberately a whitelist: the sealed
#: split is refused by *not appearing here*, so this module contains no bare
#: reference to it at all and `tests/test_classical_pipeline.py`'s standing
#: invariant over `src/baseline/classical/` keeps its teeth.
_PERMITTED_SPLITS = ("train", "val")


class RecordError(RuntimeError):
    """A record, schema or identity contract violation."""


# ---------------------------------------------------------------------------
# Configured schemas
# ---------------------------------------------------------------------------


def aggregate_schema() -> tuple[str, ...]:
    return _schema("artifacts.csv_schema")


def per_image_schema() -> tuple[str, ...]:
    return _schema("artifacts.per_image_schema")


def _schema(parameter_path: str) -> tuple[str, ...]:
    fields = get(parameter_path)
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
        raise RecordError(f"params.{parameter_path} must be a list of field names")
    if len(set(fields)) != len(fields):
        duplicates = sorted({f for f in fields if fields.count(f) > 1})
        raise RecordError(f"params.{parameter_path} has duplicate fields: {duplicates}")
    return tuple(fields)


def validate_row(row: Mapping[str, Any], schema: Sequence[str]) -> dict[str, Any]:
    """Reject any row that is not exactly the configured fields, in order."""

    if not isinstance(row, Mapping):
        raise RecordError("record must be a mapping")
    actual = tuple(row)
    expected = tuple(schema)
    if actual == expected:
        return dict(row)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RecordError(
            f"record fields differ: missing={missing}, unexpected={unexpected}"
        )
    raise RecordError(
        f"record field order differs from the configured schema: "
        f"{[f for a, f in zip(actual, expected, strict=True) if a != f][:3]}"
    )


def require_system(system: str) -> str:
    values = get("artifacts.system_values")
    if system not in values:
        raise RecordError(f"unsupported system {system!r}; params allows {values}")
    return system


# ---------------------------------------------------------------------------
# Field semantics (PB_2D §6.5)
# ---------------------------------------------------------------------------


def _semantics(
    source: str,
    kind: str,
    unit: str | None,
    *,
    nullable: bool = False,
    na: str = "n/a",
    denominator: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "type": kind,
        "unit": unit,
        "nullable": nullable,
        "not_applicable_representation": na,
        "aggregation_denominator": denominator,
    }


_NULL = "JSON null / empty CSV cell"

#: How the three byte columns are resolved.  ``params.baseline.container_policy``
#: puts every emitted container byte *inside* the payload budget, and BR-10 fixes
#: the per-image ``source_bytes`` as exactly ``A/8``.  So the transport block is
#: the total, and ``header_bytes``/``payload_bytes`` split the emitted codestream
#: inside it; the remainder is payload filler, which has no schema column and is
#: therefore recorded in the evidence rather than folded into either column.
BYTE_ACCOUNTING_NOTE = (
    "BR-11 as amended by AM-81, which defines these columns arithmetically "
    "rather than leaving them to interpretation. "
    "bytes_sent = source_bytes = A/8, the complete byte-aligned transport "
    "payload capacity placed on the channel (identical to the per-image "
    "source_bytes fixed by BR-10). "
    "header_bytes = all structural codestream bytes: SOC, every main-header "
    "marker segment, every SOT marker segment, every tile-part header through "
    "and including SOD, EOC, and the equivalent structural bytes of every "
    "tile-part, counted inside bytes_sent per params.baseline.container_policy. "
    "payload_bytes = all tile-part data bytes after SOD and before the next "
    "tile-part boundary. This is deliberately NOT described as pure "
    "entropy-coded sample data: JPEG 2000 tile-part data may also carry "
    "packet-header information, so the narrower wording would be false. "
    "emitted_codestream_bytes = header_bytes + payload_bytes exactly. "
    "payload_filler_bytes = bytes_sent - emitted_codestream_bytes is the zero "
    "filler padding the codestream out to A/8; it has no schema column and is "
    "reported in accounting_examples.json instead of being folded into either "
    "column, so no denominator is silently mixed. "
    "Aggregate denominator: both columns are means over every row that emitted "
    "a codestream, which includes delivered AND decode_failure rows and "
    "excludes structural_infeasibility and codec_infeasibility, where no "
    "codestream exists; they are null only when the aggregate contains no "
    "emitted codestream at all. "
    "LDPC/channel bits (G = k x Qm) and the channel-use count (k symbols) are "
    "separate quantities and are never reported as bytes here."
)

AGGREGATE_FIELD_SEMANTICS: dict[str, dict[str, Any]] = {
    "run_id": _semantics("make_run_id over params.artifacts.run_id_key", "sha256 hex", None),
    "timestamp": _semantics("run initialisation, captured once and reused on resume", "ISO 8601 UTC", None),
    "git_commit": _semantics("clean runner-ready execution source commit", "sha1 hex", None),
    "git_dirty": _semantics("worktree state at execution; must be false for evidence", "bool", None),
    "config_hash": _semantics("config.run_config.config_hash of the resolved RunConfig", "sha256 hex", None),
    "checkpoint_id": _semantics("sha256 of the exact frozen G-1 checkpoint file bytes", "sha256 hex", None),
    "system": _semantics("one of params.artifacts.system_values", "enum", None),
    "dataset": _semantics("configured dataset name", "enum", None),
    "split": _semantics("manifest split; always val in PB_2", "enum", None),
    "n": _semantics("number of per-image rows in this aggregate", "count", "images"),
    "k": _semantics("params.bandwidth.k_symbols[dataset][bw_ratio]", "count", "channel symbols"),
    "bw_ratio": _semantics("configured bandwidth ratio name", "enum", None),
    "channel": _semantics("params.channel.models_supported member", "enum", None),
    "train_snr_db": _semantics("not applicable: the classical arm is not trained", "float", "dB", nullable=True, na=_NULL),
    "test_snr_db": _semantics("evaluated SNR point from params.channel.test_snr_grid_db", "float", "dB"),
    "train_seed": _semantics("params.evaluation.train_seeds member; the paired analysis cell", "int", None),
    "channel_seed": _semantics("params.evaluation.channel_seeds member", "int", None),
    "lambda": _semantics("not applicable: no learned rate-distortion trade-off", "float", None, nullable=True, na=_NULL),
    "source_codec": _semantics("params.baseline.source_codec", "enum", None),
    "jpeg_quality": _semantics("not applicable: the codec of record is JPEG 2000, not JPEG", "int", None, nullable=True, na=_NULL),
    "j2k_target_bytes": _semantics("payload capacity A/8 handed to encode_to_budget", "int", "bytes"),
    "ldpc_rate": _semantics("configured nominal LDPC rate", "enum", None),
    "modulation": _semantics("configured modulation", "enum", None),
    "top1_acc": _semantics("n_correct / n, counting outage rows", "float", None, denominator="n (all rows, outages included)"),
    "n_correct": _semantics("sum of per-image correct, outage rows included", "count", "images"),
    "n_test": _semantics("number of evaluated rows despite the legacy field name; equals n, and is NOT a test-split count", "count", "images"),
    "psnr_db": _semantics("mean reconstruction PSNR", "float", "dB", nullable=True, na=_NULL, denominator="delivered rows only"),
    "ssim": _semantics("mean reconstruction SSIM", "float", None, nullable=True, na=_NULL, denominator="delivered rows only"),
    "bytes_sent": _semantics("A/8, the complete transport-block payload; see BYTE_ACCOUNTING_NOTE", "int", "bytes", denominator="fixed per configuration"),
    "header_bytes": _semantics("mean structural JPEG 2000 codestream bytes per BR-11/AM-81; see BYTE_ACCOUNTING_NOTE", "float", "bytes", nullable=True, na=_NULL, denominator="rows that emitted a codestream (delivered and decode_failure); null only when none did"),
    "payload_bytes": _semantics("mean tile-part data bytes per BR-11/AM-81; see BYTE_ACCOUNTING_NOTE", "float", "bytes", nullable=True, na=_NULL, denominator="rows that emitted a codestream (delivered and decode_failure); null only when none did"),
    "papr_db": _semantics("mean realised peak-to-average power ratio", "float", "dB", nullable=True, na=_NULL, denominator="transmitted rows only (delivered + decode_failure)"),
    "decode_failure_rate": _semantics("decode_failure count / n", "float", None, denominator="n"),
    "infeasible_rate": _semantics("(structural + codec infeasible) / n", "float", None, denominator="n"),
    "coverage_rate": _semantics("delivered count / n", "float", None, denominator="n"),
    "acc_given_delivery": _semantics("delivered_correct / delivered_count", "float", None, nullable=True, na=_NULL, denominator="delivered rows only"),
    "test_subset": _semantics("not applicable: no test split was opened", "int", "images", nullable=True, na=_NULL),
    "wall_clock_s": _semantics("measured runner wall time for this aggregate", "float", "seconds"),
    "peak_vram_gb": _semantics("torch peak reserved VRAM, or null on CPU-only inference", "float", "GB", nullable=True, na=_NULL),
    "classifier_variant": _semantics("params.reference_classifier.clean_variant_name", "enum", None),
    "quantiser_bits": _semantics("not applicable: no learned quantiser", "int", "bits", nullable=True, na=_NULL),
    "transmit_dim": _semantics("not applicable: no learned transmit dimension", "int", None, nullable=True, na=_NULL),
    "entropy_stream_bytes": _semantics("not applicable: no learned entropy coder", "int", "bytes", nullable=True, na=_NULL),
    "entropy_table_bytes": _semantics("not applicable: no learned entropy tables", "int", "bytes", nullable=True, na=_NULL),
    "side_information_bytes": _semantics("not applicable: control plane is out of band for all systems", "int", "bytes", nullable=True, na=_NULL),
    "tb_crc_type": _semantics("TransportAccounting.tb_crc_name", "enum", None),
    "base_graph": _semantics("TransportAccounting.base_graph", "int", None),
    "lifting_size": _semantics("TransportAccounting.lifting_size", "int", None),
    "num_codeblocks": _semantics("TransportAccounting.code_blocks", "count", None),
    "filler_bits": _semantics("TransportAccounting.ldpc_filler_bits_total (LDPC filler, not payload filler)", "int", "bits"),
    "effective_code_rate": _semantics("K' / max(E_r), the worst-block realised rate", "float", None),
    "model_param_count": _semantics("not applicable: no transmitted learned model", "int", None, nullable=True, na=_NULL),
}

PER_IMAGE_FIELD_SEMANTICS: dict[str, dict[str, Any]] = {
    "run_id": _semantics("identical to the aggregate run_id", "sha256 hex", None),
    "pair_id": _semantics("make_pair_id; excludes system and comparison", "sha256 hex", None),
    "noise_id": _semantics("the scheduled channel realisation for this evaluation cell, from make_noise_id; present on every row including outages, and equal to PB_1's realised ChannelIdentity.noise_id whenever the row transmitted", "sha256 hex", None),
    "analysis_cell_id": _semantics("make_analysis_cell_id over [train_seed, channel_seed]", "sha256 hex", None),
    "dataset": _semantics("configured dataset name", "enum", None),
    "dataset_version": _semantics("params.datasets[dataset].archive_sha256", "sha256 hex", None),
    "split": _semantics("manifest split; always val in PB_2", "enum", None),
    "stable_sample_id": _semantics("sha256 of original source bytes, truncated", "hex", None),
    "bw_ratio": _semantics("configured bandwidth ratio name", "enum", None),
    "test_snr_db": _semantics("evaluated SNR point", "float", "dB"),
    "true_label": _semantics("authoritative manifest label", "int", None),
    "pred_label": _semantics("classifier argmax when delivered, else the frozen constant outage class", "int", None),
    "correct": _semantics("pred_label == true_label; strictly binary on every row", "bool", None),
    "outage": _semantics("true when the row was not delivered", "bool", None),
    "outage_reason": _semantics("structural_infeasibility / codec_infeasibility / decode_failure; null when delivered", "enum", None, nullable=True, na=_NULL),
    "source_bytes": _semantics("A/8 exactly, per BR-10", "int", "bytes"),
}


def field_semantics() -> dict[str, Any]:
    """The machine-readable field-semantics table, checked against the schemas."""

    for schema, table, name in (
        (aggregate_schema(), AGGREGATE_FIELD_SEMANTICS, "csv_schema"),
        (per_image_schema(), PER_IMAGE_FIELD_SEMANTICS, "per_image_schema"),
    ):
        missing = sorted(set(schema) - set(table))
        unexpected = sorted(set(table) - set(schema))
        if missing or unexpected:
            raise RecordError(
                f"field semantics for {name} differ from params: "
                f"missing={missing}, unexpected={unexpected}"
            )
    return {
        "byte_accounting_note": BYTE_ACCOUNTING_NOTE,
        "aggregate": {
            field: AGGREGATE_FIELD_SEMANTICS[field] for field in aggregate_schema()
        },
        "per_image": {
            field: PER_IMAGE_FIELD_SEMANTICS[field] for field in per_image_schema()
        },
    }


# ---------------------------------------------------------------------------
# JPEG 2000 container accounting
# ---------------------------------------------------------------------------


def codestream_byte_split(codestream: bytes) -> tuple[int, int]:
    """Split a raw JPEG 2000 codestream into (container bytes, data bytes).

    Container bytes are everything that is not entropy-coded packet data: SOC,
    every main-header marker segment, each tile-part header up to and including
    its SOD marker, and the trailing EOC.  The two returned counts always sum to
    ``len(codestream)`` — that identity is the point, since an approximate
    header figure would make BR-11's overhead fraction unfalsifiable.
    """

    if not isinstance(codestream, bytes):
        raise RecordError("codestream must be bytes")
    total = len(codestream)
    position = 0
    container = 0
    data = 0

    def marker_at(offset: int) -> int:
        if offset + _MARKER_BYTES > total:
            raise RecordError("truncated JPEG 2000 codestream")
        return int.from_bytes(codestream[offset : offset + _MARKER_BYTES], "big")

    if marker_at(position) != _J2K_SOC:
        raise RecordError("codestream does not start with SOC")
    container += _MARKER_BYTES
    position += _MARKER_BYTES

    tile_part_start: int | None = None
    tile_part_length = 0
    while position < total:
        marker = marker_at(position)
        if marker == _J2K_EOC:
            container += _MARKER_BYTES
            position += _MARKER_BYTES
            continue
        if marker == _J2K_SOD:
            container += _MARKER_BYTES
            position += _MARKER_BYTES
            if tile_part_start is None:
                raise RecordError("SOD outside a tile-part")
            end = (
                tile_part_start + tile_part_length
                if tile_part_length
                else _next_tile_boundary(codestream, position)
            )
            if not position <= end <= total:
                raise RecordError("tile-part length runs outside the codestream")
            data += end - position
            position = end
            tile_part_start, tile_part_length = None, 0
            continue
        if marker in _J2K_DELIMITERS:
            raise RecordError(f"unexpected delimiter marker {marker:#06x}")
        segment_start = position
        position += _MARKER_BYTES
        if position + _LENGTH_BYTES > total:
            raise RecordError("truncated marker segment length")
        length = int.from_bytes(codestream[position : position + _LENGTH_BYTES], "big")
        if length < _LENGTH_BYTES:
            raise RecordError("invalid marker segment length")
        if marker == _J2K_SOT:
            if length != _SOT_SEGMENT_LENGTH:
                raise RecordError(
                    f"SOT segment length is {length}, expected {_SOT_SEGMENT_LENGTH}"
                )
            psot_start = segment_start + _PSOT_OFFSET
            if psot_start + _PSOT_BYTES > total:
                raise RecordError("truncated SOT segment: Psot runs past the end")
            tile_part_start = segment_start
            tile_part_length = int.from_bytes(
                codestream[psot_start : psot_start + _PSOT_BYTES], "big"
            )
            if tile_part_length:
                # Psot counts from the first byte of SOT to the end of the
                # tile-part, so it must at least cover the segment it sits in.
                if tile_part_length < _MARKER_BYTES + length:
                    raise RecordError(
                        f"Psot {tile_part_length} is smaller than its own SOT segment"
                    )
                if tile_part_start + tile_part_length > total:
                    raise RecordError("Psot runs past the end of the codestream")
        container += _MARKER_BYTES + length
        position = position + length
        if position > total:
            raise RecordError("marker segment runs past the end of the codestream")

    if container + data != total:
        raise RecordError("codestream byte split does not reconcile")
    return container, data


def _next_tile_boundary(codestream: bytes, position: int) -> int:
    """Locate the end of a tile-part whose Psot was left at zero.

    Psot may legally be 0 for the last tile-part, meaning "to EOC".  Scanning
    for the next SOT or the EOC is the standard's own fallback.
    """

    total = len(codestream)
    offset = position
    while offset + _MARKER_BYTES <= total:
        marker = int.from_bytes(codestream[offset : offset + _MARKER_BYTES], "big")
        if marker in (_J2K_SOT, _J2K_EOC):
            return offset
        offset += 1
    return total


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunIdentity:
    """Every run-level value the configured identity key sets consume.

    Held as one object so a caller cannot build ``run_id`` from one set of
    values and ``pair_id`` from another.
    """

    system: str
    dataset: str
    dataset_version: str
    split: str
    split_manifest_hash: str
    bw_ratio: str
    test_snr_db: float
    train_seed: int
    channel_seed: int
    config_hash: str
    checkpoint_id: str
    classifier_variant: str
    ldpc_rate: str
    modulation: str
    quantiser_bits: int | None
    transmit_dim: int | None
    reconstruction_weight: float | None
    analysis_version: int

    def __post_init__(self) -> None:
        require_system(self.system)
        if self.split not in _PERMITTED_SPLITS:
            raise RecordError(
                f"records may only describe {list(_PERMITTED_SPLITS)}; "
                f"{self.split!r} is refused, and the evaluation split stays "
                "sealed behind SR-22 and G-12"
            )

    def run_id_values(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "split_manifest_hash": self.split_manifest_hash,
            "bw_ratio": self.bw_ratio,
            "test_snr_db": self.test_snr_db,
            "train_seed": self.train_seed,
            "channel_seed": self.channel_seed,
            "config_hash": self.config_hash,
            "checkpoint_id": self.checkpoint_id,
            "classifier_variant": self.classifier_variant,
            "ldpc_rate": self.ldpc_rate,
            "modulation": self.modulation,
            "quantiser_bits": self.quantiser_bits,
            "transmit_dim": self.transmit_dim,
            "lambda": self.reconstruction_weight,
            "analysis_version": self.analysis_version,
        }

    def run_id(self) -> str:
        return make_run_id(self.run_id_values())

    def analysis_cell_id(self) -> str:
        return make_analysis_cell_id(
            {"train_seed": self.train_seed, "channel_seed": self.channel_seed}
        )

    def pair_id(self, *, stable_sample_id: str, noise_id: str) -> str:
        """System-independent pairing identity.

        ``noise_id`` here is the **scheduled** channel realisation for the cell,
        never the optional transport result.  A row that never transmitted still
        has a scheduled realisation, and if it paired on ``None`` it could not
        share a ``pair_id`` with a comparison arm that did transmit — which is
        exactly the paired comparison ER-10 exists to make.

        ``params.artifacts.pair_id_excludes`` names ``system`` and
        ``comparison``, and ``make_pair_id`` asserts neither appears in the key,
        so the same image at the same cell, ratio, SNR and noise realisation
        joins across arms.
        """

        if not isinstance(noise_id, str) or not noise_id:
            raise RecordError(
                "pair_id needs the scheduled noise identity; a row that did not "
                "transmit still has one"
            )
        return make_pair_id(
            {
                "analysis_cell_id": self.analysis_cell_id(),
                "stable_sample_id": stable_sample_id,
                "bw_ratio": self.bw_ratio,
                "test_snr_db": self.test_snr_db,
                "noise_id": noise_id,
            }
        )


def noise_identity(
    *,
    dataset_version: str,
    split_manifest_hash: str,
    stable_sample_id: str,
    test_snr_db: float,
    channel_seed: int,
    k: int,
    block_index: int,
) -> str:
    """Rebuild PB_1's ``noise_id`` through the one shared implementation."""

    return make_noise_id(
        {
            "dataset_version": dataset_version,
            "split_manifest_hash": split_manifest_hash,
            "stable_sample_id": stable_sample_id,
            "test_snr_db": test_snr_db,
            "channel_seed": channel_seed,
            "channel": _CHANNEL_MODEL,
            "k": k,
            "block_index": block_index,
            "rng_purpose": "channel_noise",
        }
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskOutcome:
    """One image's scored result: prediction, correctness and metrics."""

    pred_label: int
    correct: bool
    outage: bool
    outage_reason: str | None
    psnr_db: float | None
    ssim: float | None
    header_bytes: int | None
    payload_bytes: int | None


def classify_reconstruction(
    classifier: torch.nn.Module,
    decoded_image: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> int:
    """Run the frozen classifier on one reconstruction, inference-only."""

    if classifier.training:
        raise RecordError("the frozen classifier must be in evaluation mode")
    tensor = reconstruction_input(decoded_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(tensor)
    if logits.requires_grad:
        raise RecordError("classifier inference produced a gradient-tracking output")
    return int(torch.argmax(logits, dim=-1).item())


def score_result(
    result: ClassicalResult,
    *,
    true_label: int,
    policy: OutagePolicy,
    canonical_image: np.ndarray | None = None,
    classifier: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
    classifier_dataset: str = FROZEN_CLASSIFIER_DATASET,
) -> TaskOutcome:
    """Score one ``ClassicalResult``, running the classifier only if delivered."""

    if result.verdict not in VERDICTS:
        raise RecordError(f"unknown verdict {result.verdict!r}")
    if policy.dataset != result.dataset:
        raise RecordError(
            f"frozen outage policy is for {policy.dataset!r}, "
            f"row dataset is {result.dataset!r}"
        )
    if result.verdict != DELIVERED:
        # No reconstruction exists.  The frozen constant prediction is applied
        # and the classifier is never consulted; no substitute pixels are made.
        #
        # The *byte* columns are a different matter (AM-81).  A decode failure
        # still put a real codestream on the channel, so its container overhead
        # is measurable and belongs in the aggregate; only the two
        # infeasibility verdicts, where no codestream exists, are null here.
        # PB_2 blanked all three, so an all-decode-failure cell reported no
        # overhead at all — the regime where overhead dominates the budget.
        header, payload = _emitted_byte_split(result)
        pred_label = policy.predict()
        return TaskOutcome(
            pred_label=pred_label,
            correct=policy.is_correct(true_label),
            outage=True,
            outage_reason=OUTAGE_REASONS[result.verdict],
            psnr_db=None,
            ssim=None,
            header_bytes=header,
            payload_bytes=payload,
        )

    if result.dataset != classifier_dataset:
        raise RecordError(
            f"the frozen reference classifier is a {classifier_dataset!r} model and "
            f"must not score {result.dataset!r} reconstructions: ten equal output "
            "indices are not a shared class vocabulary"
        )
    if classifier is None:
        raise RecordError("a delivered row needs the frozen classifier")
    if canonical_image is None:
        raise RecordError("a delivered row needs its canonical source pixels")
    if result.decoded_image is None:
        raise RecordError("a delivered row carries no decoded image")

    pred_label = classify_reconstruction(classifier, result.decoded_image, device=device)
    # Both sides go through the one canonical uint8-to-unit-interval conversion:
    # ``params.preprocessing.psnr_data_range`` is 1.0, so handing raw 0..255
    # pixels to the metrics would be rejected, and converting only one side
    # would silently compare different scales.
    metrics = reconstruction_metrics(
        reconstruction_input(canonical_image),
        reconstruction_input(result.decoded_image),
    )
    header, payload = _emitted_byte_split(result)
    return TaskOutcome(
        pred_label=pred_label,
        correct=pred_label == int(true_label),
        outage=False,
        outage_reason=NOT_APPLICABLE,
        psnr_db=metrics.psnr_db,
        ssim=metrics.ssim,
        header_bytes=header,
        payload_bytes=payload,
    )


def _emitted_byte_split(result: ClassicalResult) -> tuple[int | None, int | None]:
    """The BR-11 split for any row that emitted a codestream, delivered or not."""

    source_coding = result.source_coding
    if source_coding is None or source_coding.emitted_bytes is None:
        return None, None
    codestream = source_coding.emitted_codestream
    if not isinstance(codestream, bytes):
        return None, None
    container, data = codestream_byte_split(codestream)
    if container + data != source_coding.emitted_bytes:
        raise RecordError("codestream byte split disagrees with the emitted byte count")
    return container, data


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def per_image_row(
    result: ClassicalResult,
    outcome: TaskOutcome,
    *,
    identity: RunIdentity,
    true_label: int,
    run_id: str,
    scheduled_noise_id: str,
) -> dict[str, Any]:
    """Build one schema-exact per-image row.

    ``scheduled_noise_id`` is the channel realisation the *evaluation cell*
    schedules for this row, computed without drawing anything.  It is required,
    and it is what both ``noise_id`` and ``pair_id`` carry, for all four
    verdicts.  When the row really did transmit, PB_1's realised identity must
    equal it — asserted here rather than assumed, because a silent divergence
    would mean the recorded pairing describes a different draw than the one the
    channel actually made.
    """

    if not isinstance(scheduled_noise_id, str) or not scheduled_noise_id:
        raise RecordError("every row needs its scheduled noise identity")
    if result.noise_id is not None and result.noise_id != scheduled_noise_id:
        raise RecordError(
            "the realised noise identity does not match the scheduled one: "
            f"{result.noise_id} != {scheduled_noise_id}"
        )
    if result.accounting is None:
        source_bytes = None
    else:
        source_bytes = result.accounting.payload_bytes
    row = {
        "run_id": run_id,
        "pair_id": identity.pair_id(
            stable_sample_id=result.stable_sample_id,
            noise_id=scheduled_noise_id,
        ),
        "noise_id": scheduled_noise_id,
        "analysis_cell_id": identity.analysis_cell_id(),
        "dataset": identity.dataset,
        "dataset_version": identity.dataset_version,
        "split": identity.split,
        "stable_sample_id": result.stable_sample_id,
        "bw_ratio": identity.bw_ratio,
        "test_snr_db": result.snr_db,
        "true_label": int(true_label),
        "pred_label": int(outcome.pred_label),
        "correct": bool(outcome.correct),
        "outage": bool(outcome.outage),
        "outage_reason": outcome.outage_reason,
        "source_bytes": source_bytes,
    }
    return validate_row(row, per_image_schema())


@dataclass(frozen=True)
class AggregateContext:
    """The run-level facts an aggregate row needs beyond its per-image rows."""

    identity: RunIdentity
    k_symbols: int
    timestamp: str
    git_commit: str
    git_dirty: bool
    source_codec: str
    j2k_target_bytes: int
    wall_clock_s: float
    peak_vram_gb: float | None
    tb_crc_type: str | None
    base_graph: int | None
    lifting_size: int | None
    num_codeblocks: int | None
    filler_bits: int | None
    effective_code_rate: float | None
    bytes_sent: int | None
    papr_db_values: tuple[float, ...] = ()
    header_bytes_values: tuple[int, ...] = ()
    payload_bytes_values: tuple[int, ...] = ()


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def aggregate_row(
    rows: Iterable[Mapping[str, Any]],
    context: AggregateContext,
    *,
    run_id: str,
    psnr_values: Sequence[float] = (),
    ssim_values: Sequence[float] = (),
) -> dict[str, Any]:
    """Recompute every aggregate quantity from the per-image rows themselves."""

    per_image = [dict(row) for row in rows]
    for row in per_image:
        validate_row(row, per_image_schema())
    n = len(per_image)
    if n == 0:
        raise RecordError("an aggregate needs at least one per-image row")

    identifiers = [(row["run_id"], row["pair_id"]) for row in per_image]
    if len(set(identifiers)) != n:
        raise RecordError("per-image rows carry duplicate identities")

    for row in per_image:
        if not isinstance(row["correct"], bool) or not isinstance(row["outage"], bool):
            raise RecordError("per-image correct and outage must be strictly boolean")

    n_correct = sum(1 for row in per_image if row["correct"])
    delivered = [row for row in per_image if not row["outage"]]
    reasons = [row["outage_reason"] for row in per_image if row["outage"]]
    decode_failures = reasons.count(DECODE_FAILURE)
    structural = reasons.count(STRUCTURAL_INFEASIBILITY)
    codec = reasons.count(CODEC_INFEASIBILITY)
    unknown = sorted(set(reasons) - set(OUTAGE_REASONS))
    if unknown:
        raise RecordError(f"unknown outage reasons: {unknown}")
    if any(row["outage_reason"] is not NOT_APPLICABLE for row in delivered):
        raise RecordError("a delivered row carries an outage reason")
    if len(delivered) + decode_failures + structural + codec != n:
        raise RecordError("verdict counts do not reconcile with the row count")

    delivered_count = len(delivered)
    delivered_correct = sum(1 for row in delivered if row["correct"])
    if len(psnr_values) != delivered_count or len(ssim_values) != delivered_count:
        raise RecordError(
            "reconstruction metrics must be supplied for exactly the delivered rows"
        )

    row = {
        "run_id": run_id,
        "timestamp": context.timestamp,
        "git_commit": context.git_commit,
        "git_dirty": context.git_dirty,
        "config_hash": context.identity.config_hash,
        "checkpoint_id": context.identity.checkpoint_id,
        "system": context.identity.system,
        "dataset": context.identity.dataset,
        "split": context.identity.split,
        "n": n,
        "k": context.k_symbols,
        "bw_ratio": context.identity.bw_ratio,
        "channel": _CHANNEL_MODEL,
        "train_snr_db": None,
        "test_snr_db": context.identity.test_snr_db,
        "train_seed": context.identity.train_seed,
        "channel_seed": context.identity.channel_seed,
        "lambda": context.identity.reconstruction_weight,
        "source_codec": context.source_codec,
        "jpeg_quality": None,
        "j2k_target_bytes": context.j2k_target_bytes,
        "ldpc_rate": context.identity.ldpc_rate,
        "modulation": context.identity.modulation,
        "top1_acc": n_correct / n,
        "n_correct": n_correct,
        "n_test": n,
        "psnr_db": _mean(psnr_values),
        "ssim": _mean(ssim_values),
        "bytes_sent": context.bytes_sent,
        "header_bytes": _mean(context.header_bytes_values),
        "payload_bytes": _mean(context.payload_bytes_values),
        "papr_db": _mean(context.papr_db_values),
        "decode_failure_rate": decode_failures / n,
        "infeasible_rate": (structural + codec) / n,
        "coverage_rate": delivered_count / n,
        "acc_given_delivery": (
            delivered_correct / delivered_count if delivered_count else None
        ),
        "test_subset": None,
        "wall_clock_s": context.wall_clock_s,
        "peak_vram_gb": context.peak_vram_gb,
        "classifier_variant": context.identity.classifier_variant,
        "quantiser_bits": context.identity.quantiser_bits,
        "transmit_dim": context.identity.transmit_dim,
        "entropy_stream_bytes": None,
        "entropy_table_bytes": None,
        "side_information_bytes": None,
        "tb_crc_type": context.tb_crc_type,
        "base_graph": context.base_graph,
        "lifting_size": context.lifting_size,
        "num_codeblocks": context.num_codeblocks,
        "filler_bits": context.filler_bits,
        "effective_code_rate": context.effective_code_rate,
        "model_param_count": None,
    }
    return validate_row(row, aggregate_schema())


def reconcile_aggregate(
    aggregate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently recompute the identities an aggregate row asserts.

    Used by the verifier, and by tests, so the checks do not live only inside
    the builder that would have to be wrong for them to matter.
    """

    n = len(rows)
    if n == 0:
        raise RecordError("cannot reconcile an empty aggregate")
    n_correct = sum(1 for row in rows if row["correct"])
    delivered = [row for row in rows if not row["outage"]]
    reasons = [row["outage_reason"] for row in rows if row["outage"]]
    counts = {
        "delivered": len(delivered),
        DECODE_FAILURE: reasons.count(DECODE_FAILURE),
        STRUCTURAL_INFEASIBILITY: reasons.count(STRUCTURAL_INFEASIBILITY),
        CODEC_INFEASIBILITY: reasons.count(CODEC_INFEASIBILITY),
    }
    expected = {
        "n": n,
        "n_test": n,
        "n_correct": n_correct,
        "top1_acc": n_correct / n,
        "coverage_rate": counts["delivered"] / n,
        "decode_failure_rate": counts[DECODE_FAILURE] / n,
        "infeasible_rate": (
            counts[STRUCTURAL_INFEASIBILITY] + counts[CODEC_INFEASIBILITY]
        )
        / n,
        "acc_given_delivery": (
            sum(1 for row in delivered if row["correct"]) / counts["delivered"]
            if counts["delivered"]
            else None
        ),
    }
    if sum(counts.values()) != n:
        raise RecordError("verdict counts do not sum to the row count")
    differing = sorted(
        field for field, value in expected.items() if aggregate.get(field) != value
    )
    if differing:
        raise RecordError(f"aggregate does not reconcile with its rows: {differing}")
    return counts
