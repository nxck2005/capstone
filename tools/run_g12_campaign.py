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
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402

RUNTIME = REPO / "checkpoints/g12_campaign"
UNITS_DIR = RUNTIME / "units"
ACCESS_LOG = RUNTIME / "test_access_log.jsonl"
CAMPAIGN_MARKER = RUNTIME / "campaign.json"
RESULTS_DIR = REPO / "results/g12"
PER_IMAGE_DIR = REPO / str(get("artifacts.per_image_dir")) / "g12"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"G-12 campaign HOLD: {message}")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _freeze() -> dict[str, Any]:
    """The committed, reproducing freeze manifest and the commit it binds."""

    import g12_freeze  # noqa: PLC0415

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


def _unit_path(unit: dict[str, Any]) -> Path:
    from evaluation.g12_scope import unit_stem  # noqa: PLC0415

    return UNITS_DIR / f"{unit_stem(unit)}.json.gz"


def _read_unit(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read())


def _streams(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    streams = [{"classifier_variant": aggregate["primary_classifier_variant"], "rows": aggregate["per_image"]}]
    for extra in aggregate.get("secondary_streams", ()):
        streams.append({"classifier_variant": extra["classifier_variant"], "rows": extra["per_image"]})
    return streams


def _verify_unit(value: dict[str, Any], unit: dict[str, Any], stable_ids: list[str] | None, freeze_id: str) -> None:
    from evaluation.g12_scope import arm_for  # noqa: PLC0415

    _require(value["unit"] == unit and value["freeze_id"] == freeze_id, f"unit {unit['ordinal']} belongs to another campaign")
    arm = arm_for(unit["role"])
    _require([stream["classifier_variant"] for stream in value["streams"]] == list(arm.classifier_variants), f"unit {unit['ordinal']} scorer streams differ")
    for stream in value["streams"]:
        rows = stream["rows"]
        ids = [row["stable_sample_id"] for row in rows]
        if stable_ids is not None:
            _require(ids == stable_ids, f"unit {unit['ordinal']} rows are not the complete ordered test split")
        _require(all(row["split"] == "test" and float(row["test_snr_db"]) == float(unit["snr_db"]) for row in rows), f"unit {unit['ordinal']} rows differ from the unit")
        _require(hashlib.sha256(b"".join(_canonical(row) for row in rows)).hexdigest() == stream["rows_sha256"], f"unit {unit['ordinal']} row digest differs")


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.GzipFile(temporary, "wb", mtime=0) as handle:
        handle.write(_canonical(value))
    os.replace(temporary, path)


def run(worker: int, workers: int, device: str) -> int:
    import torch

    from evaluation.g12_bindings import resolve
    from evaluation.g12_dispatch import dispatch
    from evaluation.g12_scope import work_units

    freeze = _freeze()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if CAMPAIGN_MARKER.exists():
        marker = json.loads(CAMPAIGN_MARKER.read_bytes())
        _require(marker["freeze_id"] == freeze["freeze_id"], "a different G-12 campaign already opened test; it must be closed or invalidated, never mixed")
    else:
        CAMPAIGN_MARKER.write_bytes(_canonical({"freeze_id": freeze["freeze_id"], "code_commit": freeze["code_commit"], "opened_at": dt.datetime.now(dt.timezone.utc).isoformat()}))
    bindings = resolve(REPO)
    _require([item["binding_id"] for item in bindings] == [item["binding_id"] for item in freeze["checkpoints"]], "live bindings differ from the frozen checkpoints")
    torch.set_num_threads(max(1, (os.cpu_count() or workers) // workers))
    units = [unit for unit in work_units() if int(unit["ordinal"]) % workers == worker]
    pending = []
    for unit in units:
        path = _unit_path(unit)
        if path.exists():
            _verify_unit(_read_unit(path), unit, None, freeze["freeze_id"])
        else:
            pending.append(unit)
    print(f"worker {worker}/{workers} on {device}: {len(units) - len(pending)} done, {len(pending)} pending", flush=True)
    if not pending:
        return 0
    with ACCESS_LOG.open("a") as log:
        log.write(json.dumps({"event": "test_split_opened", "worker": worker, "device": device, "pid": os.getpid(), "freeze_id": freeze["freeze_id"], "at": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
    view = TestView()
    _require(len(view.stable_ids) == int(get("datasets.imagenette160.test_images")), "test denominator differs")
    backend = dispatch(REPO, device=device, bindings=bindings, view=view)
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
            "wall_clock_s": round(time.monotonic() - started, 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None,  # literal-ok: bytes per GiB
            "summary": summary,
            "streams": streams,
        }
        _verify_unit(value, unit, view.stable_ids, freeze["freeze_id"])
        _write_atomic(_unit_path(unit), value)
        print(f"{unit['ordinal']:3d} {unit['role']:28s} c{unit['train_seed']} {unit['snr_db']:>4} dB  {streams[0]['n_correct']}/{len(streams[0]['rows'])}  {value['wall_clock_s']:.0f}s", flush=True)
    return 0


def status() -> int:
    from evaluation.g12_scope import work_units

    units = work_units()
    done = [unit for unit in units if _unit_path(unit).exists()]
    print(f"{len(done)}/{len(units)} G-12 units complete")
    return 0


def _csv_row(value: dict[str, Any], stream: dict[str, Any]) -> dict[str, Any]:
    unit, summary, rows = value["unit"], value["summary"], stream["rows"]
    binding = summary.get("binding", {}) or {}
    n = len(rows)
    delivered = [row for row in rows if not row["outage"]]
    return {
        "run_id": rows[0]["run_id"] if rows else None,
        "timestamp": None,
        "git_commit": value["code_commit"],
        "git_dirty": False,
        "config_hash": None,
        "checkpoint_id": binding.get("checkpoint_id"),
        "system": unit["system"],
        "dataset": "imagenette160",
        "split": "test",
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
        "jpeg_quality": None,
        "j2k_target_bytes": None,
        "ldpc_rate": binding.get("ldpc_rate"),
        "modulation": binding.get("modulation"),
        "top1_acc": stream["n_correct"] / n if n else None,
        "n_correct": stream["n_correct"],
        "n_test": n,
        "psnr_db": None,
        "ssim": None,
        "bytes_sent": None,
        "header_bytes": None,
        "payload_bytes": None,
        "papr_db": summary.get("papr_db_mean", summary.get("papr_db")),
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
        "tb_crc_type": None,
        "base_graph": None,
        "lifting_size": None,
        "num_codeblocks": None,
        "filler_bits": None,
        "effective_code_rate": None,
        "model_param_count": None,
    }


def closeout() -> int:
    from evaluation import g12_analysis
    from evaluation.g12_scope import arm_for, work_units

    freeze = _freeze()
    units = work_units()
    missing = [unit["ordinal"] for unit in units if not _unit_path(unit).exists()]
    _require(not missing, f"{len(missing)} units are missing: {missing[:10]}")
    PER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schema = list(get("artifacts.csv_schema"))
    manifest_rows, csv_rows, analysis_rows = [], [], []
    reference_ids = None
    for unit in units:
        value = _read_unit(_unit_path(unit))
        _verify_unit(value, unit, reference_ids, freeze["freeze_id"])
        reference_ids = reference_ids or [row["stable_sample_id"] for row in value["streams"][0]["rows"]]
        arm = arm_for(unit["role"])
        for stream in value["streams"]:
            raw = b"".join(_canonical(row) for row in stream["rows"])
            digest = hashlib.sha256(raw).hexdigest()
            name = f"{_unit_path(unit).name[:-len('.json.gz')]}.{stream['classifier_variant']}.jsonl.gz"
            with gzip.GzipFile(PER_IMAGE_DIR / name, "wb", mtime=0) as handle:
                handle.write(raw)
            manifest_rows.append({"file": f"g12/{name}", "rows_sha256": digest, "rows": len(stream["rows"]), "run_id": stream["rows"][0]["run_id"], "system": unit["system"], "bw_ratio": unit["bw_ratio"], "train_seed": unit["train_seed"], "test_snr_db": unit["snr_db"], "classifier_variant": stream["classifier_variant"]})
            csv_rows.append(_csv_row(value, stream))
            if stream["classifier_variant"] == arm.classifier_variants[0] and unit["bw_ratio"] == "r_1_6":
                for row in stream["rows"]:
                    analysis_rows.append({"system": unit["system"], "bw_ratio": unit["bw_ratio"], "stable_sample_id": row["stable_sample_id"], "train_seed": unit["train_seed"], "test_snr_db": row["test_snr_db"], "correct": row["correct"]})
    with (REPO / str(get("artifacts.per_image_manifest"))).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (RESULTS_DIR / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema)
        writer.writeheader()
        writer.writerows(csv_rows)

    systems = {name: (name, "r_1_6") for name in ("learned", "classical_adaptive", "er9_digital", "classical_fixed_mcs")}
    outcomes = g12_analysis.outcomes_from_rows(analysis_rows, systems=systems, cells=(0, 1, 2))
    window = (freeze["h2_window"]["low_snr_db"], freeze["h2_window"]["high_snr_db"])
    result = g12_analysis.analyse(outcomes, h2_window=window)
    per_image_sha = hashlib.sha256((REPO / str(get("artifacts.per_image_manifest"))).read_bytes()).hexdigest()
    result["freeze_id"] = freeze["freeze_id"]
    result["code_commit"] = freeze["code_commit"]
    result["per_image_manifest_sha256"] = per_image_sha
    (RESULTS_DIR / "analysis.json").write_bytes(json.dumps(result, indent=1, sort_keys=True, allow_nan=False).encode() + b"\n")
    _write_inference_summary(result, freeze, per_image_sha)
    print(f"G-12 closeout: {len(csv_rows)} result rows, {len(manifest_rows)} per-image streams")
    for key in ("H1", "H2", "H3", "H4"):
        if key in result:
            print(f"{key}: supported={result[key]['supported']}")
    return 0


def _write_inference_summary(result: dict[str, Any], freeze: dict[str, Any], per_image_sha: str) -> None:
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
    with (REPO / str(get("artifacts.inference_summary_file"))).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--worker", type=int, required=True)
    runner.add_argument("--workers", type=int, required=True)
    runner.add_argument("--device", default="cuda:0")
    sub.add_parser("status")
    sub.add_parser("closeout")
    args = parser.parse_args()
    if args.command == "run":
        return run(args.worker, args.workers, args.device)
    return status() if args.command == "status" else closeout()


if __name__ == "__main__":
    raise SystemExit(main())
