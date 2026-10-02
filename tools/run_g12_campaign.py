#!/usr/bin/env python3
"""The single G-12 test campaign (SR-22, ER-1, ER-6, ``test_campaign_is_single``).

    python tools/run_g12_campaign.py run --worker 0 --workers 4 --device cuda:0
    python tools/run_g12_campaign.py status
    python tools/run_g12_campaign.py closeout      # after every unit exists

``run`` refuses unless the committed freeze manifest reproduces from this tree
and nothing protected changed after its code commit.  Units are partitioned by
ordinal across workers; each finished unit is written atomically, so a crashed
worker resumes the *same* campaign (same freeze) without re-running finished
units.  Every opening of the test split is appended to the runtime access log.
``closeout`` writes the per-image release files, ``results/per_image_manifest.csv``
and the aggregate CSV, then runs the frozen ER-10 analysis into
``results/inference_summary.csv``.

``--rehearsal`` runs the identical code on the validation split before the
freeze: the view, split label, output directories and the freeze source (built
in memory, never written) are the only differences, and nothing is written
under ``results/``.  It never reaches a test loader.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402


@dataclass(frozen=True)
class Mode:
    rehearsal: bool
    split: str
    runtime: Path
    results_dir: Path
    per_image_dir: Path
    per_image_manifest: Path
    inference_summary: Path
    j2k_cache_dir: str

    @property
    def units_dir(self) -> Path:
        return self.runtime / "units"

    @property
    def access_log(self) -> Path:
        return self.runtime / "test_access_log.jsonl"

    @property
    def marker(self) -> Path:
        return self.runtime / "campaign.json"


CAMPAIGN = Mode(
    rehearsal=False,
    split="test",
    runtime=REPO / "checkpoints/g12_campaign",
    results_dir=REPO / "results/g12",
    per_image_dir=REPO / str(get("artifacts.per_image_dir")) / "g12",
    per_image_manifest=REPO / str(get("artifacts.per_image_manifest")),
    inference_summary=REPO / str(get("artifacts.inference_summary_file")),
    j2k_cache_dir="checkpoints/g12_campaign/j2k_cache",
)
_REHEARSAL_ROOT = REPO / "checkpoints/g12_rehearsal_full"
REHEARSAL = Mode(
    rehearsal=True,
    split="val",
    runtime=_REHEARSAL_ROOT,
    results_dir=_REHEARSAL_ROOT / "results",
    per_image_dir=_REHEARSAL_ROOT / "per_image",
    per_image_manifest=_REHEARSAL_ROOT / "results/per_image_manifest.csv",
    inference_summary=_REHEARSAL_ROOT / "results/inference_summary.csv",
    j2k_cache_dir="checkpoints/g12_rehearsal_full/j2k_cache",
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"G-12 campaign HOLD: {message}")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _freeze(mode: Mode) -> dict[str, Any]:
    """The committed, reproducing freeze manifest (or an in-memory one to rehearse)."""

    import g12_freeze  # noqa: PLC0415

    if mode.rehearsal:
        return g12_freeze.build(code_commit=_git("rev-parse", "HEAD"))
    path = REPO / str(get("artifacts.freeze_manifest_file"))
    _require(path.is_file(), "no freeze manifest; G-12 has not been frozen")
    value = json.loads(path.read_bytes())
    rebuilt = g12_freeze.build(code_commit=value["code_commit"])
    _require(g12_freeze._encode(rebuilt) == path.read_bytes(), "freeze manifest does not reproduce from this tree")
    changed = _git("diff", "--name-only", value["code_commit"], "HEAD", "--", *g12_freeze.PROTECTED)
    _require(not changed, f"protected source changed after the freeze: {changed}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", *g12_freeze.PROTECTED, str(path.relative_to(REPO)))
    _require(not dirty, f"working tree differs from the committed freeze:\n{dirty}")
    return value


class TestView:
    """The guarded test split in stable-ID order, with the ValidationView interface."""

    def __init__(self) -> None:
        from data.test_access import load_test_dataset  # noqa: PLC0415  (the sole SR-22 boundary)

        self._source = load_test_dataset("imagenette160")
        pairs = sorted((str(self._source.source_sample(index).stable_sample_id), index) for index in range(len(self._source)))
        self.stable_ids = [stable_id for stable_id, _ in pairs]
        self._index = {stable_id: index for stable_id, index in pairs}
        self.labels = {stable_id: int(self._source.source_sample(index).label) for stable_id, index in pairs}
        self._products: dict[str, Any] = {}

    def product(self, stable_id: str) -> Any:
        if stable_id not in self._products:
            self._products[stable_id] = self._source[self._index[stable_id]][0]
        return self._products[stable_id]

    def label(self, stable_id: str) -> int:
        return self.labels[stable_id]

    def index_of(self, stable_id: str) -> int:
        return self.stable_ids.index(stable_id)

    def canonical_tensor(self, stable_id: str) -> Any:
        from data.preprocessing import evaluation_input  # noqa: PLC0415

        return evaluation_input(self.product(stable_id))


class CachedValidationView:
    """The rehearsal's validation view, caching canonical products like ``TestView``."""

    def __init__(self) -> None:
        from evaluation.w10_backends import ValidationView  # noqa: PLC0415

        self._view = ValidationView()
        self.stable_ids = list(self._view.stable_ids)
        self.labels = dict(self._view.labels)
        self._products: dict[str, Any] = {}

    def product(self, stable_id: str) -> Any:
        if stable_id not in self._products:
            self._products[stable_id] = self._view.product(stable_id)
        return self._products[stable_id]

    def label(self, stable_id: str) -> int:
        return self.labels[stable_id]

    def index_of(self, stable_id: str) -> int:
        return self.stable_ids.index(stable_id)

    def canonical_tensor(self, stable_id: str) -> Any:
        from data.preprocessing import evaluation_input  # noqa: PLC0415

        return evaluation_input(self.product(stable_id))


def _denominator(mode: Mode) -> int:
    return int(get("datasets.imagenette160.val_images" if mode.rehearsal else "datasets.imagenette160.test_images"))


def _unit_path(mode: Mode, unit: dict[str, Any]) -> Path:
    from evaluation.g12_scope import unit_stem  # noqa: PLC0415

    return mode.units_dir / f"{unit_stem(unit)}.json.gz"


def _read_unit(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read())


def _streams(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    streams = [{"classifier_variant": aggregate["primary_classifier_variant"], "rows": aggregate["per_image"]}]
    for extra in aggregate.get("secondary_streams", ()):
        streams.append({"classifier_variant": extra["classifier_variant"], "rows": extra["per_image"]})
    return streams


def _unit_for_mode(mode: Mode, unit: dict[str, Any]) -> dict[str, Any]:
    return {**unit, "split": mode.split}


def _verify_unit(mode: Mode, value: dict[str, Any], unit: dict[str, Any], stable_ids: list[str] | None, freeze_id: str) -> None:
    from evaluation.g12_scope import arm_for  # noqa: PLC0415

    _require(value["unit"] == unit and value["freeze_id"] == freeze_id, f"unit {unit['ordinal']} belongs to another campaign")
    arm = arm_for(unit["role"])
    _require([stream["classifier_variant"] for stream in value["streams"]] == list(arm.classifier_variants), f"unit {unit['ordinal']} scorer streams differ")
    for stream in value["streams"]:
        rows = stream["rows"]
        ids = [row["stable_sample_id"] for row in rows]
        _require(len(rows) == _denominator(mode) and len(set(ids)) == len(ids), f"unit {unit['ordinal']} does not cover the split exactly once")
        if stable_ids is not None:
            _require(ids == stable_ids, f"unit {unit['ordinal']} rows are not the complete ordered split")
        _require(all(row["split"] == mode.split and float(row["test_snr_db"]) == float(unit["snr_db"]) for row in rows), f"unit {unit['ordinal']} rows differ from the unit")
        _require(hashlib.sha256(b"".join(_canonical(row) for row in rows)).hexdigest() == stream["rows_sha256"], f"unit {unit['ordinal']} row digest differs")
        _require(sum(1 for row in rows if row["correct"]) == stream["n_correct"], f"unit {unit['ordinal']} n_correct does not recompute")


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.GzipFile(temporary, "wb", mtime=0) as handle:
        handle.write(_canonical(value))
    os.replace(temporary, path)


def run(mode: Mode, worker: int, workers: int, device: str) -> int:
    import torch

    from evaluation.g12_bindings import resolve
    from evaluation.g12_dispatch import dispatch
    from evaluation.g12_scope import work_units

    freeze = _freeze(mode)
    mode.runtime.mkdir(parents=True, exist_ok=True)
    if mode.marker.exists():
        marker = json.loads(mode.marker.read_bytes())
        _require(marker["freeze_id"] == freeze["freeze_id"], "a different G-12 campaign already opened this runtime; it must be closed or invalidated, never mixed")
    else:
        mode.marker.write_bytes(_canonical({"freeze_id": freeze["freeze_id"], "code_commit": freeze["code_commit"], "split": mode.split, "opened_at": dt.datetime.now(dt.timezone.utc).isoformat()}))
    bindings = resolve(REPO)
    _require([item["binding_id"] for item in bindings] == [item["binding_id"] for item in freeze["checkpoints"]], "live bindings differ from the frozen checkpoints")
    torch.set_num_threads(max(1, (os.cpu_count() or workers) // workers))
    units = [_unit_for_mode(mode, unit) for unit in work_units() if int(unit["ordinal"]) % workers == worker]
    pending = []
    for unit in units:
        path = _unit_path(mode, unit)
        if path.exists():
            _verify_unit(mode, _read_unit(path), unit, None, freeze["freeze_id"])
        else:
            pending.append(unit)
    print(f"worker {worker}/{workers} on {device}: {len(units) - len(pending)} done, {len(pending)} pending", flush=True)
    if not pending:
        return 0
    with mode.access_log.open("a") as log:
        log.write(json.dumps({"event": f"{mode.split}_split_opened", "worker": worker, "device": device, "pid": os.getpid(), "freeze_id": freeze["freeze_id"], "at": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
    view = CachedValidationView() if mode.rehearsal else TestView()
    _require(len(view.stable_ids) == _denominator(mode), "split denominator differs")
    backend = dispatch(REPO, device=device, bindings=bindings, view=view, split=mode.split, j2k_cache_dir=mode.j2k_cache_dir)
    for unit in pending:
        started = time.monotonic()
        with torch.no_grad():
            aggregate = dict(backend(unit))
        streams = _streams(aggregate)
        for stream in streams:
            stream["rows_sha256"] = hashlib.sha256(b"".join(_canonical(row) for row in stream["rows"])).hexdigest()
            stream["n_correct"] = sum(1 for row in stream["rows"] if row["correct"])
        summary = {key: value for key, value in aggregate.items() if key not in ("per_image", "secondary_streams")}
        value = {
            "unit": unit,
            "freeze_id": freeze["freeze_id"],
            "code_commit": freeze["code_commit"],
            "device": device,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "wall_clock_s": round(time.monotonic() - started, 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None,  # literal-ok: bytes per GiB
            "summary": summary,
            "streams": streams,
        }
        _verify_unit(mode, value, unit, view.stable_ids, freeze["freeze_id"])
        _write_atomic(_unit_path(mode, unit), value)
        print(f"{unit['ordinal']:3d} {unit['role']:28s} c{unit['train_seed']} {unit['snr_db']:>4} dB  {streams[0]['n_correct']}/{len(streams[0]['rows'])}  {value['wall_clock_s']:.0f}s", flush=True)
    return 0


def status(mode: Mode) -> int:
    from evaluation.g12_scope import work_units

    units = [_unit_for_mode(mode, unit) for unit in work_units()]
    done = [unit for unit in units if _unit_path(mode, unit).exists()]
    print(f"{len(done)}/{len(units)} G-12 units complete ({mode.split})")
    return 0


def _csv_row(mode: Mode, value: dict[str, Any], stream: dict[str, Any]) -> dict[str, Any]:
    unit, summary, rows = value["unit"], value["summary"], stream["rows"]
    binding = summary.get("binding", {}) or {}
    measured = summary.get("measurements", {}) or {}
    n = len(rows)
    delivered = [row for row in rows if not row["outage"]]
    return {
        "run_id": rows[0]["run_id"] if rows else None,
        "timestamp": value.get("finished_utc"),
        "git_commit": value["code_commit"],
        "git_dirty": False,
        "config_hash": binding.get("config_hash"),
        "checkpoint_id": binding.get("checkpoint_id"),
        "system": unit["system"],
        "dataset": "imagenette160",
        "split": mode.split,
        "n": n,
        "k": int(get(f"bandwidth.k_symbols.imagenette160.{unit['bw_ratio']}")),
        "bw_ratio": unit["bw_ratio"],
        "channel": "awgn",
        "train_snr_db": get("channel.train_snr_db_fixed"),
        "test_snr_db": unit["snr_db"],
        "train_seed": unit["train_seed"],
        "channel_seed": unit["channel_seed"],
        "lambda": None,
        "source_codec": binding.get("codec"),
        "jpeg_quality": binding.get("quality") if binding.get("codec") == "jpeg" else None,
        "j2k_target_bytes": None,
        "ldpc_rate": binding.get("ldpc_rate"),
        "modulation": binding.get("modulation"),
        "top1_acc": stream["n_correct"] / n if n else None,
        "n_correct": stream["n_correct"],
        "n_test": n,
        "psnr_db": measured.get("psnr_db"),
        "ssim": measured.get("ssim"),
        "bytes_sent": measured.get("bytes_sent", rows[0]["source_bytes"] if rows else None),
        "header_bytes": measured.get("header_bytes"),
        "payload_bytes": measured.get("payload_bytes"),
        "papr_db": summary["mean_papr_db"],
        "decode_failure_rate": sum(1 for row in rows if row["outage_reason"] == "decode_failure") / n if n else None,
        "infeasible_rate": sum(1 for row in rows if row["outage_reason"] in ("structural_infeasibility", "codec_infeasibility")) / n if n else None,
        "coverage_rate": len(delivered) / n if n else None,
        "acc_given_delivery": (sum(1 for row in delivered if row["correct"]) / len(delivered)) if delivered else None,
        "test_subset": False,
        "wall_clock_s": value["wall_clock_s"],
        "peak_vram_gb": value["peak_vram_gb"],
        "classifier_variant": stream["classifier_variant"],
        "quantiser_bits": binding.get("quantiser_bits"),
        "transmit_dim": binding.get("transmit_dim"),
        "entropy_stream_bytes": None,
        "entropy_table_bytes": None,
        "side_information_bytes": None,
        "tb_crc_type": measured.get("tb_crc_type"),
        "base_graph": measured.get("base_graph"),
        "lifting_size": measured.get("lifting_size"),
        "num_codeblocks": measured.get("num_codeblocks"),
        "filler_bits": measured.get("filler_bits"),
        "effective_code_rate": measured.get("effective_code_rate"),
        "model_param_count": None,
    }


def _descriptive(outcomes: Any) -> dict[str, Any]:
    """ER-10's per-cell curves, between-cell range and per-cell McNemar checks."""

    from evaluation.g12_analysis import mcnemar_exact  # noqa: PLC0415

    curves = {}
    for system, values in outcomes.correct.items():
        per_cell = values.mean(axis=0)  # cells × SNRs
        curves[system] = {
            "per_cell_accuracy": {str(cell): [float(v) for v in per_cell[j]] for j, cell in enumerate(outcomes.cells)},
            "cell_mean_accuracy": [float(v) for v in per_cell.mean(axis=0)],
            "between_cell_range": [float(v) for v in per_cell.max(axis=0) - per_cell.min(axis=0)],
        }
    mcnemar = []
    learned = outcomes.correct["learned"]
    for comparator in ("classical_adaptive", "er9_digital"):
        other = outcomes.correct[comparator]
        for j, cell in enumerate(outcomes.cells):
            for s, snr in enumerate(outcomes.snrs):
                a_only = int(((learned[:, j, s] == 1) & (other[:, j, s] == 0)).sum())
                b_only = int(((learned[:, j, s] == 0) & (other[:, j, s] == 1)).sum())
                mcnemar.append({"comparator": comparator, "cell": cell, "snr_db": snr, "learned_only": a_only, "comparator_only": b_only, "p_value": mcnemar_exact(a_only, b_only)})
    return {"role": get("evaluation.paired_test_role"), "curves": curves, "mcnemar": mcnemar}


def closeout(mode: Mode) -> int:
    from evaluation import g12_analysis
    from evaluation.g12_scope import arm_for, work_units

    freeze = _freeze(mode)
    units = [_unit_for_mode(mode, unit) for unit in work_units()]
    missing = [unit["ordinal"] for unit in units if not _unit_path(mode, unit).exists()]
    _require(not missing, f"{len(missing)} units are missing: {missing[:10]}")
    mode.per_image_dir.mkdir(parents=True, exist_ok=True)
    mode.results_dir.mkdir(parents=True, exist_ok=True)
    schema = list(get("artifacts.csv_schema"))
    manifest_rows, csv_rows, analysis_rows = [], [], []
    reference_ids = None
    for unit in units:
        value = _read_unit(_unit_path(mode, unit))
        _verify_unit(mode, value, unit, reference_ids, freeze["freeze_id"])
        reference_ids = reference_ids or [row["stable_sample_id"] for row in value["streams"][0]["rows"]]
        arm = arm_for(unit["role"])
        for stream in value["streams"]:
            raw = b"".join(_canonical(row) for row in stream["rows"])
            digest = hashlib.sha256(raw).hexdigest()
            name = f"{_unit_path(mode, unit).name[:-len('.json.gz')]}.{stream['classifier_variant']}.jsonl.gz"
            with gzip.GzipFile(mode.per_image_dir / name, "wb", mtime=0) as handle:
                handle.write(raw)
            manifest_rows.append({"file": f"g12/{name}", "rows_sha256": digest, "rows": len(stream["rows"]), "run_id": stream["rows"][0]["run_id"], "system": unit["system"], "bw_ratio": unit["bw_ratio"], "train_seed": unit["train_seed"], "test_snr_db": unit["snr_db"], "classifier_variant": stream["classifier_variant"]})
            csv_rows.append(_csv_row(mode, value, stream))
            if stream["classifier_variant"] == arm.classifier_variants[0] and unit["bw_ratio"] == "r_1_6":
                for row in stream["rows"]:
                    analysis_rows.append({"system": unit["system"], "bw_ratio": unit["bw_ratio"], "stable_sample_id": row["stable_sample_id"], "train_seed": unit["train_seed"], "test_snr_db": row["test_snr_db"], "correct": row["correct"]})
    mode.per_image_manifest.parent.mkdir(parents=True, exist_ok=True)
    with mode.per_image_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (mode.results_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema)
        writer.writeheader()
        writer.writerows(csv_rows)

    systems = {name: (name, "r_1_6") for name in ("learned", "classical_adaptive", "er9_digital", "classical_fixed_mcs")}
    outcomes = g12_analysis.outcomes_from_rows(analysis_rows, systems=systems, cells=(0, 1, 2))
    window = (freeze["h2_window"]["low_snr_db"], freeze["h2_window"]["high_snr_db"])
    result = g12_analysis.analyse(outcomes, h2_window=window)
    per_image_sha = hashlib.sha256(mode.per_image_manifest.read_bytes()).hexdigest()
    result["split"] = mode.split
    result["freeze_id"] = freeze["freeze_id"]
    result["code_commit"] = freeze["code_commit"]
    result["per_image_manifest_sha256"] = per_image_sha
    result["descriptive"] = _descriptive(outcomes)
    (mode.results_dir / "analysis.json").write_bytes(json.dumps(result, indent=1, sort_keys=True, allow_nan=False).encode() + b"\n")
    _write_inference_summary(mode, result, freeze, per_image_sha)
    print(f"G-12 closeout ({mode.split}): {len(csv_rows)} result rows, {len(manifest_rows)} per-image streams")
    for key in ("H1", "H2", "H3", "H4"):
        if key in result:
            print(f"{key}: supported={result[key]['supported']}")
    return 0


def _write_inference_summary(mode: Mode, result: dict[str, Any], freeze: dict[str, Any], per_image_sha: str) -> None:
    from training.deterministic_core import canonical_sha256  # noqa: PLC0415

    schema = list(get("artifacts.inference_summary_schema"))
    common = {
        "estimand": get("evaluation.estimand"),
        "bw_ratio": "r_1_6",
        "seed_aggregation": get("evaluation.seed_aggregation"),
        "source_run_ids": None,
        "per_image_artifact_sha256": per_image_sha,
        "analysis_commit": freeze["code_commit"],
        "analysis_config_hash": canonical_sha256(freeze["analysis_version"]),
    }
    rows = []
    for key, comparator in (("H1", "classical_adaptive"), ("H4", "er9_digital")):
        value = result[key]
        rows.append({**common, "hypothesis": key, "systems": f"learned-{comparator}", "snr_region": f"<= {get('channel.train_snr_db_fixed')} dB", "window": None,
                     "estimate": value["effect_size"], "ci_low": value["effect_ci"][0], "ci_high": value["effect_ci"][1], "p_value": value["calibrated_p"],
                     "calibration_statistic": f"R_obs={value['r_obs']}", "support_decision": value["supported"]})
        for point in value["per_snr"]:
            rows.append({**common, "hypothesis": f"{key}-point", "systems": f"learned-{comparator}", "snr_region": f"{point['snr_db']} dB", "window": None,
                         "estimate": point["delta"], "ci_low": point["ci_low"], "ci_high": point["ci_high"], "p_value": None, "calibration_statistic": None, "support_decision": None})
    if "H2" in result:
        value = result["H2"]
        rows.append({**common, "hypothesis": "H2", "systems": "learned-classical_fixed_mcs", "snr_region": None, "window": f"{value['window_snr_db'][0]}->{value['window_snr_db'][1]} dB",
                     "estimate": value["difference_in_differences"], "ci_low": value["difference_in_differences_ci"][0], "ci_high": value["difference_in_differences_ci"][1], "p_value": None,
                     "calibration_statistic": f"D_C_lower={value['classical_drop_one_sided_lower']:.4f};D_L_upper={value['learned_drop_one_sided_upper']:.4f}", "support_decision": value["supported"]})
    value = result["H3"]
    rows.append({**common, "hypothesis": "H3", "systems": "learned-classical_adaptive", "snr_region": "full grid", "window": None,
                 "estimate": value["slope_per_db"], "ci_low": value["slope_ci"][0], "ci_high": value["slope_ci"][1], "p_value": None,
                 "calibration_statistic": json.dumps(value["clauses"], sort_keys=True), "support_decision": value["supported"]})
    mode.inference_summary.parent.mkdir(parents=True, exist_ok=True)
    with mode.inference_summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rehearsal", action="store_true", help="run the identical code on the validation split, before the freeze")
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--worker", type=int, required=True)
    runner.add_argument("--workers", type=int, required=True)
    runner.add_argument("--device", default="cuda:0")
    sub.add_parser("status")
    sub.add_parser("closeout")
    args = parser.parse_args()
    mode = REHEARSAL if args.rehearsal else CAMPAIGN
    if args.command == "run":
        return run(mode, args.worker, args.workers, args.device)
    return status(mode) if args.command == "status" else closeout(mode)


if __name__ == "__main__":
    raise SystemExit(main())
