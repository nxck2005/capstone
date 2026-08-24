"""Compact, fail-closed G8_F/F1 corpus closeout and offline verification.

The live path authenticates every frozen request, result, and referenced object
through :mod:`baseline.g8_f_materializer` before deriving compact evidence.  It
never decodes an image, invokes a codec, imports a classifier, trains, selects,
or accesses validation/test data.
"""

from __future__ import annotations

import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from baseline.g8_f_f0 import (
    AUTHORIZATION_PATH,
    PROFILE_ID,
    ZERO_COUNTERS,
    rendered_json,
    verify_f0_authorization,
)
from baseline.g8_f_materializer import (
    CODEC_CONFIGURATION_HASH,
    CODEC_CONFIGURATION_ID,
    F1Assignment,
    G8FMaterializationHold,
    MANIFEST_SHA256,
    ORDERED_PAIR_SHA256,
    PAIR_SET_SHA256,
    SAMPLER_PLAN_ID,
    SAMPLER_PLAN_SHA256,
    canonical_json,
    load_frozen_assignments,
    validate_exact_result_prefix,
)
from baseline.g8_f_sampler_plan import (
    AM87_PLAN_FILE_SHA256,
    AM87_PLAN_ID,
    EXPECTED_ATTEMPTS,
    EXPECTED_QUALITY_COUNT,
    EXPECTED_TRAINING_COUNT,
    EXPECTED_VARIANTS,
)
from config.params import REPO_ROOT

SCHEMA_VERSION = 1
COMPLETION_ROLE = "g8_f_f1_completion"
COMPLETION_PREFIX = "g8ff1completion-"
CORPUS_PREFIX = "g8fcorpus-"
STATUS = "F1_GREEN_EXACT_50814_ASSIGNMENT_ARTIFACT_CORPUS_AUTHENTICATED_AND_FROZEN"
LAUNCH_PATH = REPO_ROOT / "results/baseline/g8_f/f1_launch_authorization.json"
MANIFEST_PATH = REPO_ROOT / "results/baseline/g8_f/f1_corpus_manifest.csv"
COMPLETION_PATH = REPO_ROOT / "results/baseline/g8_f/f1_completion.json"
MONITOR_CLOSEOUT_PATH = REPO_ROOT / "results/baseline/g8_f/f1_monitor_closeout.json"
RUNTIME_PATH = REPO_ROOT / "results/baseline/g8_f/runtime"

MATERIALIZED = "materialized_verified_artifact"
INFEASIBLE = "typed_image_codec_infeasibility"
OMISSION = "record_omitted_assigned_pair_no_replacement_no_resampling"
EXPECTED_F0_SHA256 = "391cd81553ed2de869ddf3ad1f0a401523781289342eefaddc7ad27cb005517e"
EXPECTED_LAUNCH_SHA256 = "4265b69660d40fe72cc7957a1709e33c46384fc82ea59e12e075d2363f36a111"
EXPECTED_LAUNCH_ID = "g8ff1launch-a88fc23774b38763858e2fec717bf27f0f79893bb6b768708c0f3d38a570ee74"
EXPECTED_SOURCE_COMMIT = "6f06aa81ae2d624bae0d406904982f3a61278d93"
EXPECTED_LOCK_SHA256 = "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82"
EXPECTED_GPU_UUID = "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a"
EXPECTED_GPU_NAME = "NVIDIA TITAN Xp"
EXPECTED_DEVICE = "cuda:0"
EXPECTED_HOST = "confessor"
EXPECTED_MONITOR_MISMATCHES = ((37030, 37028), (42819, 42817), (46021, 46019))

MANIFEST_FIELDS = (
    "ordinal",
    "assignment_id",
    "stable_sample_id",
    "class_label",
    "quality_id",
    "payload_budget_bytes",
    "encode_axis_px",
    "outcome",
    "request_id",
    "result_id",
    "request_record_sha256",
    "result_record_sha256",
    "codestream_sha256",
    "codestream_bytes",
    "codestream_path",
    "reconstruction_sha256",
    "reconstruction_bytes",
    "reconstruction_path",
    "omission_state",
)

PROTECTED_COUNTERS = {
    "artifact_classifier_inference": 0,
    "artifact_classifier_optimizer_steps": 0,
    "artifact_fine_tuning": 0,
    "pass_two": 0,
    "fallback": 0,
    "ratio_adjudication": 0,
    "learned_system_training": 0,
    "test_access": 0,
}


class G8FF1CloseoutHold(RuntimeError):
    """A condition that prevents F1 closeout."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8FF1CloseoutHold(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _file_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    name = str(path if relative_to is None else path.relative_to(relative_to))
    return {"path": name, "bytes": len(raw), "sha256": _sha(raw)}


def _canonical_csv(rows: Sequence[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("ascii")


def _read_manifest(raw: bytes) -> list[dict[str, str]]:
    _require(raw.decode("ascii").encode("ascii") == raw, "F1 corpus manifest is not ASCII")
    reader = csv.DictReader(io.StringIO(raw.decode("ascii"), newline=""))
    _require(tuple(reader.fieldnames or ()) == MANIFEST_FIELDS, "F1 corpus manifest header differs")
    rows = list(reader)
    _require(_canonical_csv(rows) == raw, "F1 corpus manifest is not canonical CSV")
    return rows


def _exact_namespace(root: Path, name: str, total: int) -> list[Path]:
    _require(root.is_dir() and not root.is_symlink(), f"F1 {name} namespace is missing, symlinked, or not a directory")
    paths = sorted(root.iterdir(), key=lambda path: path.name)
    expected = [f"{ordinal:05d}.json" for ordinal in range(total)]
    _require([path.name for path in paths] == expected, f"F1 {name} namespace is not exact complete prefix {total}/{total}")
    _require(all(path.is_file() and not path.is_symlink() for path in paths), f"F1 {name} namespace contains a non-regular path")
    return paths


def _deduplicate_object(target: dict[str, dict[str, Any]], value: dict[str, Any], kind: str) -> None:
    entry = {"sha256": value["sha256"], "bytes": value["bytes"], "path": value["path"]}
    previous = target.setdefault(entry["sha256"], entry)
    _require(previous == entry, f"{kind} content identity has conflicting metadata")


def collect_authenticated_runtime(
    runtime_root: Path,
    assignments: Sequence[F1Assignment],
    *,
    expected_total: int | None = None,
) -> dict[str, Any]:
    """Authenticate a stopped complete runtime and derive deterministic records.

    ``expected_total`` exists only so synthetic mutation tests can exercise this
    exact closeout path without constructing the production 50,814-row corpus.
    Production callers always pass the frozen assignment count.
    """

    runtime_root = Path(runtime_root)
    total = len(assignments) if expected_total is None else expected_total
    _require(len(assignments) == total, "supplied F1 assignment authority is incomplete")
    completed = validate_exact_result_prefix(runtime_root, assignments, expected_scientific=True)
    _require(completed == total, f"authenticated F1 prefix is incomplete: {completed}/{total}")
    requests = _exact_namespace(runtime_root / "requests", "request", total)
    results = _exact_namespace(runtime_root / "results", "result", total)

    rows: list[dict[str, Any]] = []
    per_quality: dict[str, Counter[str]] = defaultdict(Counter)
    per_class: dict[int, Counter[str]] = defaultdict(Counter)
    per_image: dict[str, Counter[str]] = defaultdict(Counter)
    class_quality: dict[int, Counter[str]] = defaultdict(Counter)
    outcomes: Counter[str] = Counter()
    codestreams: dict[str, dict[str, Any]] = {}
    reconstructions: dict[str, dict[str, Any]] = {}
    ordered_requests: list[list[Any]] = []
    ordered_results: list[list[Any]] = []
    referenced_codestream_bytes = 0
    referenced_reconstruction_bytes = 0

    for ordinal, (assignment, request_path, result_path) in enumerate(zip(assignments, requests, results, strict=True)):
        request_raw = request_path.read_bytes()
        result_raw = result_path.read_bytes()
        request = json.loads(request_raw)
        result = json.loads(result_raw)
        outcome = result["outcome"]
        _require(outcome in {MATERIALIZED, INFEASIBLE}, f"unexpected F1 outcome at ordinal {ordinal}: {outcome!r}")
        outcomes[outcome] += 1
        bucket = "materialized" if outcome == MATERIALIZED else "infeasible"
        for summary in (per_quality[assignment.quality_id], per_class[assignment.label], per_image[assignment.stable_sample_id]):
            summary["assigned"] += 1
            summary[bucket] += 1
        class_quality[assignment.label][assignment.quality_id] += 1

        codestream = result["codestream"]
        reconstruction = result["reconstruction"]
        if outcome == MATERIALIZED:
            _deduplicate_object(codestreams, codestream, "codestream")
            _deduplicate_object(reconstructions, reconstruction, "reconstruction")
            referenced_codestream_bytes += codestream["bytes"]
            referenced_reconstruction_bytes += reconstruction["bytes"]
            omission_state = ""
        else:
            _require(codestream is None and reconstruction is None, "typed F1 omission references an object")
            omission_state = OMISSION

        request_sha = _sha(request_raw)
        result_sha = _sha(result_raw)
        ordered_requests.append([ordinal, request["request_id"], request_sha])
        ordered_results.append([ordinal, result["result_id"], result_sha])
        rows.append(
            {
                "ordinal": ordinal,
                "assignment_id": assignment.assignment_id,
                "stable_sample_id": assignment.stable_sample_id,
                "class_label": assignment.label,
                "quality_id": assignment.quality_id,
                "payload_budget_bytes": assignment.payload_budget_bytes,
                "encode_axis_px": assignment.encode_axis_px,
                "outcome": outcome,
                "request_id": request["request_id"],
                "result_id": result["result_id"],
                "request_record_sha256": request_sha,
                "result_record_sha256": result_sha,
                "codestream_sha256": "" if codestream is None else codestream["sha256"],
                "codestream_bytes": "" if codestream is None else codestream["bytes"],
                "codestream_path": "" if codestream is None else codestream["path"],
                "reconstruction_sha256": "" if reconstruction is None else reconstruction["sha256"],
                "reconstruction_bytes": "" if reconstruction is None else reconstruction["bytes"],
                "reconstruction_path": "" if reconstruction is None else reconstruction["path"],
                "omission_state": omission_state,
            }
        )

    _require(sum(outcomes.values()) == total, "F1 outcome total differs from assignments")
    _require(set(per_image) == {assignment.stable_sample_id for assignment in assignments}, "F1 observed image membership differs")
    _require(all(value["assigned"] == EXPECTED_VARIANTS for value in per_image.values()) if total == EXPECTED_ATTEMPTS else True, "AM-88 six-assignments-per-image balance differs")
    if total == EXPECTED_ATTEMPTS:
        _require(len(per_quality) == EXPECTED_QUALITY_COUNT, "not every AM-87 quality appears in observed assignments")
        _require({value["assigned"] for value in per_quality.values()} <= {423, 424}, "AM-88 global quality balance differs")
        _require(all(max(counts.values()) - min(counts.values()) <= 1 for counts in class_quality.values()), "AM-88 within-class quality balance differs")

    codestream_entries = sorted(codestreams.values(), key=lambda entry: entry["sha256"])
    reconstruction_entries = sorted(reconstructions.values(), key=lambda entry: entry["sha256"])
    return {
        "rows": rows,
        "manifest_bytes": _canonical_csv(rows),
        "authenticated_prefix": completed,
        "outcomes": {
            "total_assignments": total,
            MATERIALIZED: outcomes[MATERIALIZED],
            INFEASIBLE: outcomes[INFEASIBLE],
            "unexpected_or_other": total - outcomes[MATERIALIZED] - outcomes[INFEASIBLE],
        },
        "digests": {
            "ordered_request_record_sha256": _sha(canonical_json(ordered_requests)),
            "request_record_set_sha256": _sha(canonical_json(sorted([entry[1:] for entry in ordered_requests]))),
            "ordered_result_record_sha256": _sha(canonical_json(ordered_results)),
            "result_record_set_sha256": _sha(canonical_json(sorted([entry[1:] for entry in ordered_results]))),
            "codestream_object_set_sha256": _sha(canonical_json(codestream_entries)),
            "reconstruction_object_set_sha256": _sha(canonical_json(reconstruction_entries)),
        },
        "summaries": {
            "per_quality": [
                {"quality_id": key, **{name: value.get(name, 0) for name in ("assigned", "materialized", "infeasible")}}
                for key, value in sorted(per_quality.items())
            ],
            "per_class": [
                {"class_label": key, **{name: value.get(name, 0) for name in ("assigned", "materialized", "infeasible")}}
                for key, value in sorted(per_class.items())
            ],
            "per_image": [
                {"stable_sample_id": key, **{name: value.get(name, 0) for name in ("assigned", "materialized", "infeasible")}}
                for key, value in sorted(per_image.items())
            ],
        },
        "objects": {
            "unique_codestream_objects": len(codestream_entries),
            "unique_reconstruction_objects": len(reconstruction_entries),
            "referenced_codestream_bytes_all_rows": referenced_codestream_bytes,
            "referenced_reconstruction_bytes_all_rows": referenced_reconstruction_bytes,
            "unique_codestream_bytes": sum(entry["bytes"] for entry in codestream_entries),
            "unique_reconstruction_bytes": sum(entry["bytes"] for entry in reconstruction_entries),
        },
    }


def _tree_usage(root: Path) -> dict[str, int]:
    apparent = allocated = files = 0
    for directory, directories, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            _require(not (base / name).is_symlink(), f"worker corpus tree contains symlinked directory: {base / name}")
        for name in names:
            path = base / name
            _require(path.is_file() and not path.is_symlink(), f"worker corpus tree contains non-regular file: {path}")
            stat = path.stat()
            files += 1
            apparent += stat.st_size
            allocated += stat.st_blocks * 512  # literal-ok: POSIX st_blocks units are 512 bytes
    return {"regular_files": files, "apparent_bytes": apparent, "allocated_bytes": allocated}


def _lock_is_free(path: Path) -> bool:
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        finally:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    return True


def _writer_pids(runtime_root: Path) -> list[int]:
    found: list[int] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = [part.decode("utf-8", "replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
            if any(Path(arg).name == "run_g8_f_f1.py" for arg in argv) and "--start" in argv and str(runtime_root) in argv:
                found.append(int(proc.name))
        except (OSError, ValueError):
            continue
    return sorted(found)


def _tmux_active(session: str) -> bool:
    try:
        return subprocess.run(["tmux", "has-session", "-t", session], check=False, capture_output=True, timeout=5).returncode == 0  # literal-ok: bounded local process query
    except (OSError, subprocess.SubprocessError):
        return False


def _ops_evidence(ops_root: Path) -> dict[str, Any]:
    required = ("launch.sh", "launch-command.txt", "started-at.txt", "exit-status.txt", "f1.stdout.log", "f1.stderr.log")
    _require(all((ops_root / name).is_file() and not (ops_root / name).is_symlink() for name in required), "detached F1 launcher records are incomplete")
    bindings = {name: _file_binding(ops_root / name) for name in required}
    exit_status = int((ops_root / "exit-status.txt").read_text(encoding="ascii").strip())
    _require(exit_status == 0, f"detached F1 launcher exited nonzero: {exit_status}")
    started_at = (ops_root / "started-at.txt").read_text(encoding="ascii").strip()
    _require(started_at.endswith("Z"), "detached F1 start timestamp is invalid")
    final_progress: dict[str, Any] | None = None
    for line in reversed((ops_root / "f1.stdout.log").read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "completed" in value:
            final_progress = value
            break
    _require(final_progress is not None and final_progress.get("completed") == EXPECTED_ATTEMPTS and final_progress.get("total") == EXPECTED_ATTEMPTS, "detached F1 stdout lacks exact terminal progress")
    _require((ops_root / "f1.stderr.log").stat().st_size == 0, "detached F1 stderr is nonempty")
    return {
        "mechanism": "tmux_session_g8f-f1_running_launch.sh_with_atomic_exit_status_record",
        "tmux_session": "g8f-f1",
        "started_at": started_at,
        "exit_status": exit_status,
        "final_progress": final_progress,
        "records": bindings,
    }


def _monitor_evidence(source_path: Path, log_path: Path) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    log = log_path.read_text(encoding="utf-8")
    result_sample = "completed, latest, result_prefix = ordinal_namespace(RESULTS)"
    request_sample = "requests, _latest_request, request_prefix = ordinal_namespace(REQUESTS)"
    _require(result_sample in source and request_sample in source and source.index(result_sample) < source.index(request_sample), "monitor sampling order cannot explain transient snapshots")
    observed = tuple((int(a), int(b)) for a, b in re.findall(r"request/result mismatch (\d+)/(\d+)", log))
    _require(observed == EXPECTED_MONITOR_MISMATCHES, f"monitor mismatch history differs: {observed!r}")
    _require("status='F1 COMPLETED' completed=50814/50814" in log, "monitor did not observe terminal file counts")
    return {
        "classification": "OPERATIONAL_MONITOR_FALSE_POSITIVES_SAMPLING_RACE",
        "scientific_hold": False,
        "observed_request_result_snapshots": [{"requests": request, "results": result} for request, result in observed],
        "cause": "read_only_monitor_enumerated_the_growing_results_namespace_before_the_growing_requests_namespace;the_two_counts_were_not_one_atomic_snapshot",
        "final_runtime_resolution": "exact_live_authentication_requires_50814_requests_50814_results_no_orphan_no_hole_no_foreign_record_and_all_referenced_objects_valid",
        "source": _file_binding(source_path),
        "log": _file_binding(log_path),
        "sampling_order": ["results_namespace", "requests_namespace"],
        "terminal_count_observed": True,
    }


def build_closeout(
    runtime_root: Path,
    ops_root: Path,
    monitor_source: Path,
    monitor_log: Path,
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[bytes, dict[str, Any]]:
    """Build immutable evidence only after full live authentication succeeds."""

    from run_g8_f_f1 import verify_separate_f1_launch

    runtime_root = Path(runtime_root).resolve()
    _require(runtime_root.name == "runtime", "unexpected F1 runtime location")
    _require(not runtime_root.is_symlink(), "F1 runtime root is symlinked")
    f0 = verify_f0_authorization(AUTHORIZATION_PATH, live_runtime=True, require_zero_prefix=False)
    launch = verify_separate_f1_launch(LAUNCH_PATH, AUTHORIZATION_PATH, f0)
    _require(_sha(AUTHORIZATION_PATH.read_bytes()) == EXPECTED_F0_SHA256, "active F0 file SHA differs")
    _require(_sha(LAUNCH_PATH.read_bytes()) == EXPECTED_LAUNCH_SHA256 and launch["launch_id"] == EXPECTED_LAUNCH_ID, "F1 launch lineage differs")
    _require(f0["source"]["intended_f1_source_commit"] == EXPECTED_SOURCE_COMMIT, "F1 source commit differs")
    _require(f0["execution"]["execution_profile_id"] == PROFILE_ID and f0["execution"]["sole_writer_host"] == EXPECTED_HOST, "F1 profile/host differs")
    _require(f0["execution"]["device"] == EXPECTED_DEVICE and f0["execution"]["selected_gpu_uuid"] == EXPECTED_GPU_UUID, "F1 device/GPU differs")
    _require(f0["execution"]["lock_file_sha256"] == EXPECTED_LOCK_SHA256, "Pascal lock differs")

    writer_pids = _writer_pids(runtime_root)
    _require(not writer_pids, f"F1 scientific writer is still running: {writer_pids}")
    _require(_lock_is_free(runtime_root / "f1.lock"), "F1 scientific writer lock is still owned")
    _require(not _tmux_active("g8f-f1"), "detached F1 tmux session is still active")
    ops = _ops_evidence(Path(ops_root))
    assignments = load_frozen_assignments()
    _require(len(assignments) == EXPECTED_ATTEMPTS, "frozen F1 assignment count differs")
    collected = collect_authenticated_runtime(runtime_root, assignments, expected_total=EXPECTED_ATTEMPTS)
    manifest_raw = collected.pop("manifest_bytes")
    rows = collected.pop("rows")
    del rows
    manifest_sha = _sha(manifest_raw)

    result_last = runtime_root / "results" / f"{EXPECTED_ATTEMPTS - 1:05d}.json"
    completed_at = _utc(result_last.stat().st_mtime)
    started_at = ops["started_at"]
    elapsed = dt.datetime.fromisoformat(completed_at.replace("Z", "+00:00")).timestamp() - dt.datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
    _require(elapsed > 0, "F1 elapsed time is invalid")
    runtime_usage = _tree_usage(runtime_root)
    cache_usage = _tree_usage(runtime_root / "backend_j2k_cache")
    disk = shutil.disk_usage(runtime_root)
    monitor = _monitor_evidence(Path(monitor_source), Path(monitor_log))

    corpus_identity = {
        "manifest_sha256": manifest_sha,
        "ordered_result_record_sha256": collected["digests"]["ordered_result_record_sha256"],
        "result_record_set_sha256": collected["digests"]["result_record_set_sha256"],
        "codestream_object_set_sha256": collected["digests"]["codestream_object_set_sha256"],
        "reconstruction_object_set_sha256": collected["digests"]["reconstruction_object_set_sha256"],
        "f0_authorization_id": f0["authorization_id"],
        "f1_launch_id": launch["launch_id"],
        "ordered_pair_sha256": ORDERED_PAIR_SHA256,
        "pair_set_sha256": PAIR_SET_SHA256,
    }
    corpus_id = CORPUS_PREFIX + _sha(canonical_json(corpus_identity))
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": COMPLETION_ROLE,
        "phase": "G8_F",
        "checkpoint": "F1",
        "status": STATUS,
        "corpus_id": corpus_id,
        "lineage": {
            "f0": {"path": str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)), "authorization_id": f0["authorization_id"], "file_sha256": EXPECTED_F0_SHA256},
            "f1_launch": {"path": str(LAUNCH_PATH.relative_to(REPO_ROOT)), "launch_id": launch["launch_id"], "file_sha256": EXPECTED_LAUNCH_SHA256},
            "f1_source_commit": EXPECTED_SOURCE_COMMIT,
            "am87": {"path": "results/baseline/g8_f/corpus_plan.json", "plan_id": AM87_PLAN_ID, "file_sha256": AM87_PLAN_FILE_SHA256},
            "am88": {"path": "results/baseline/g8_f/am88_sampler_plan.json", "plan_id": SAMPLER_PLAN_ID, "file_sha256": SAMPLER_PLAN_SHA256},
            "ordered_pair_sha256": ORDERED_PAIR_SHA256,
            "pair_set_sha256": PAIR_SET_SHA256,
            "training_manifest_sha256": MANIFEST_SHA256,
            "codec_configuration_id": CODEC_CONFIGURATION_ID,
            "codec_configuration_hash": CODEC_CONFIGURATION_HASH,
        },
        "execution": {
            "worker_hostname": EXPECTED_HOST,
            "runtime_root": str(runtime_root),
            "execution_profile_id": PROFILE_ID,
            "device": EXPECTED_DEVICE,
            "gpu_name": EXPECTED_GPU_NAME,
            "gpu_uuid": EXPECTED_GPU_UUID,
            "lock_file": f0["execution"]["lock_file"],
            "lock_file_sha256": EXPECTED_LOCK_SHA256,
            "openjpeg_version": f0["codec"]["openjpeg_version"],
            "glymur_version": f0["codec"]["glymur_version"],
            "started_at": started_at,
            "completed_at": completed_at,
            "completed_at_basis": "mtime_of_authenticated_terminal_result_50813",
            "elapsed_seconds": elapsed,
            "average_assignments_per_hour": EXPECTED_ATTEMPTS / elapsed * 3600,
        },
        "worker_termination": {
            "scientific_writer_running": False,
            "scientific_writer_pids": writer_pids,
            "second_writer_detected": False,
            "writer_lock_free": True,
            "tmux_session_active": False,
            "detached_launcher": ops,
        },
        "coverage": {
            "assignments": EXPECTED_ATTEMPTS,
            "requests": EXPECTED_ATTEMPTS,
            "results": EXPECTED_ATTEMPTS,
            "authenticated_prefix": collected["authenticated_prefix"],
            "orphan_requests": 0,
            "result_holes": 0,
            "foreign_ordinals": 0,
            "foreign_assignments": 0,
            "duplicate_assignments": 0,
            "unassigned_results": 0,
        },
        "outcomes": collected["outcomes"],
        "digests": collected["digests"],
        "corpus_manifest": {
            "path": str(Path(manifest_path).relative_to(REPO_ROOT)),
            "rows": EXPECTED_ATTEMPTS,
            "bytes": len(manifest_raw),
            "sha256": manifest_sha,
            "format": "canonical_ascii_csv_one_row_per_frozen_assignment_in_ordinal_order",
        },
        "assignment_balance": {
            "training_images": EXPECTED_TRAINING_COUNT,
            "qualities": EXPECTED_QUALITY_COUNT,
            "variants_per_image": EXPECTED_VARIANTS,
            "duplicate_pairs": 0,
            "global_quality_assignment_count_min": min(row["assigned"] for row in collected["summaries"]["per_quality"]),
            "global_quality_assignment_count_max": max(row["assigned"] for row in collected["summaries"]["per_quality"]),
            "within_class_quality_assignment_range_max": 1,
            "membership_changed_after_outcomes": False,
            "successful_subset_resampled_or_rebalanced": False,
        },
        "descriptive_counts": collected["summaries"],
        "objects": collected["objects"],
        "data_membership": {"split": "train", "training_stable_ids": EXPECTED_TRAINING_COUNT, "validation_ids": 0, "test_ids": 0, "test_access": 0},
        "storage_custody": {
            "worker_hostname": EXPECTED_HOST,
            "runtime_root": str(runtime_root),
            "runtime": runtime_usage,
            "backend_cache": cache_usage,
            "filesystem_total_bytes": disk.total,
            "filesystem_used_bytes": disk.used,
            "filesystem_free_bytes": disk.free,
            "owner": runtime_root.stat().st_uid,
            "group": runtime_root.stat().st_gid,
            "mode_octal": oct(runtime_root.stat().st_mode & 0o777),
            "another_durable_copy": False,
            "git_bulk_objects_committed": False,
            "deletion_authorized": False,
        },
        "monitor_incident": monitor,
        "f2_readiness": {
            "deterministic_mapping": "AM88_assignment_to_authenticated_F1_result_to_materialized_reconstruction_or_typed_omission",
            "materialized_training_policy": "consume_only_the_exact_materialized_reconstruction_referenced_by_the_assignment_row",
            "typed_omission_policy": OMISSION,
            "replacement_artifacts": 0,
            "ready_for_separately_authorized_loader_and_training": True,
            "f2_authorized": False,
            "f2_launched": False,
        },
        "protected_counters": PROTECTED_COUNTERS,
        "protected_counter_basis": {
            "f0_starting_zero_counters": {key: ZERO_COUNTERS[key] for key in ("artifact_classifier_inference", "artifact_classifier_optimizer_steps", "pass_two", "fallback_invoked", "ratio_adjudicated", "learned_system_training", "test_access")},
            "f1_source_closure_contains_no_classifier_optimizer_selection_or_test_path": True,
            "closeout_performed_record_and_object_authentication_only": True,
        },
        "prior_science_recomputation": {
            "g8_c": False,
            "g8_d": False,
            "g8_e": False,
            "pass_one": False,
            "g1": False,
            "am87": False,
            "am88": False,
        },
        "terminal_statement": "F1 GREEN - EXACT 50,814-ASSIGNMENT ARTIFACT CORPUS AUTHENTICATED AND FROZEN; WORKER CLOSED; F2/BR-12 TRAINING REQUIRES SEPARATE OWNER AUTHORIZATION.",
    }
    body["completion_id"] = COMPLETION_PREFIX + _sha(canonical_json(body))
    return manifest_raw, body


def _int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except ValueError:
        raise G8FF1CloseoutHold(f"F1 manifest {field} is not an integer") from None


def verify_closeout(
    completion_path: Path = COMPLETION_PATH,
    manifest_path: Path = MANIFEST_PATH,
    *,
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Verify tracked closeout offline; optionally re-authenticate all live bytes."""

    from run_g8_f_f1 import verify_separate_f1_launch

    completion_raw = Path(completion_path).read_bytes()
    completion = json.loads(completion_raw)
    _require(completion_raw == rendered_json(completion), "F1 completion artifact is not canonical rendered JSON")
    body = dict(completion)
    completion_id = body.pop("completion_id", None)
    _require(completion_id == COMPLETION_PREFIX + _sha(canonical_json(body)), "F1 completion content identity differs")
    _require(completion.get("schema_version") == SCHEMA_VERSION and completion.get("artifact_role") == COMPLETION_ROLE, "F1 completion header differs")
    _require(completion.get("status") == STATUS, "F1 completion is not green/frozen")
    _require(completion.get("protected_counters") == PROTECTED_COUNTERS, "F1 closeout later-stage counter is nonzero or missing")
    _require(completion["data_membership"] == {"split": "train", "training_stable_ids": EXPECTED_TRAINING_COUNT, "validation_ids": 0, "test_ids": 0, "test_access": 0}, "F1 closeout validation/test membership differs")
    _require(completion["f2_readiness"]["f2_authorized"] is False and completion["f2_readiness"]["f2_launched"] is False, "F2 was opened")

    f0 = verify_f0_authorization(AUTHORIZATION_PATH, require_zero_prefix=False)
    launch = verify_separate_f1_launch(LAUNCH_PATH, AUTHORIZATION_PATH, f0)
    _require(completion["lineage"]["f0"] == {"path": str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)), "authorization_id": f0["authorization_id"], "file_sha256": EXPECTED_F0_SHA256}, "F1 completion F0 lineage differs")
    _require(completion["lineage"]["f1_launch"]["launch_id"] == launch["launch_id"] and completion["lineage"]["f1_launch"]["file_sha256"] == EXPECTED_LAUNCH_SHA256, "F1 completion launch lineage differs")
    _require(completion["lineage"]["ordered_pair_sha256"] == ORDERED_PAIR_SHA256 and completion["lineage"]["pair_set_sha256"] == PAIR_SET_SHA256, "F1 completion AM-88 digest differs")

    manifest_raw = Path(manifest_path).read_bytes()
    manifest_binding = completion["corpus_manifest"]
    _require(len(manifest_raw) == manifest_binding["bytes"] and _sha(manifest_raw) == manifest_binding["sha256"], "F1 corpus manifest bytes differ")
    rows = _read_manifest(manifest_raw)
    assignments = load_frozen_assignments()
    _require(len(rows) == len(assignments) == EXPECTED_ATTEMPTS, "F1 corpus manifest is not exact 50,814 rows")
    ordered_results: list[list[Any]] = []
    result_set: list[list[str]] = []
    codestreams: dict[str, dict[str, Any]] = {}
    reconstructions: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    seen_results: set[str] = set()
    for ordinal, (row, assignment) in enumerate(zip(rows, assignments, strict=True)):
        _require(_int(row, "ordinal") == ordinal, "F1 corpus manifest has duplicate, missing, or reordered ordinal")
        _require(row["assignment_id"] == assignment.assignment_id and row["stable_sample_id"] == assignment.stable_sample_id and row["quality_id"] == assignment.quality_id, "F1 corpus manifest contains a foreign assignment")
        _require(_int(row, "class_label") == assignment.label and _int(row, "payload_budget_bytes") == assignment.payload_budget_bytes and _int(row, "encode_axis_px") == assignment.encode_axis_px, "F1 corpus manifest assignment body differs")
        pair = (row["stable_sample_id"], row["quality_id"])
        _require(pair not in seen_pairs, "F1 corpus manifest contains duplicate assignment pair")
        seen_pairs.add(pair)
        _require(row["result_id"] not in seen_results, "F1 corpus manifest contains duplicate result ID")
        seen_results.add(row["result_id"])
        outcome = row["outcome"]
        _require(outcome in {MATERIALIZED, INFEASIBLE}, "F1 corpus manifest contains unexpected outcome")
        counts[outcome] += 1
        if outcome == MATERIALIZED:
            _require(not row["omission_state"], "materialized F1 row carries omission state")
            _require(row["codestream_path"] == f"objects/codestream/{row['codestream_sha256']}.j2k", "F1 manifest codestream path differs")
            _require(row["reconstruction_path"] == f"objects/reconstruction/{row['reconstruction_sha256']}.rgb", "F1 manifest reconstruction path differs")
            _deduplicate_object(codestreams, {"sha256": row["codestream_sha256"], "bytes": _int(row, "codestream_bytes"), "path": row["codestream_path"]}, "codestream")
            _deduplicate_object(reconstructions, {"sha256": row["reconstruction_sha256"], "bytes": _int(row, "reconstruction_bytes"), "path": row["reconstruction_path"]}, "reconstruction")
        else:
            _require(row["omission_state"] == OMISSION, "typed F1 omission semantics differ")
            _require(all(not row[field] for field in ("codestream_sha256", "codestream_bytes", "codestream_path", "reconstruction_sha256", "reconstruction_bytes", "reconstruction_path")), "typed F1 omission carries object metadata")
        ordered_results.append([ordinal, row["result_id"], row["result_record_sha256"]])
        result_set.append([row["result_id"], row["result_record_sha256"]])

    _require(counts[MATERIALIZED] == completion["outcomes"][MATERIALIZED] and counts[INFEASIBLE] == completion["outcomes"][INFEASIBLE], "F1 manifest outcome counts differ from completion")
    _require(completion["outcomes"]["unexpected_or_other"] == 0, "F1 completion records unexpected outcomes")
    _require(_sha(canonical_json(ordered_results)) == completion["digests"]["ordered_result_record_sha256"], "ordered F1 result-record digest differs")
    _require(_sha(canonical_json(sorted(result_set))) == completion["digests"]["result_record_set_sha256"], "F1 result-record set digest differs")
    _require(_sha(canonical_json(sorted(codestreams.values(), key=lambda entry: entry["sha256"]))) == completion["digests"]["codestream_object_set_sha256"], "F1 codestream-object set digest differs")
    _require(_sha(canonical_json(sorted(reconstructions.values(), key=lambda entry: entry["sha256"]))) == completion["digests"]["reconstruction_object_set_sha256"], "F1 reconstruction-object set digest differs")
    corpus_identity = {
        "manifest_sha256": manifest_binding["sha256"],
        "ordered_result_record_sha256": completion["digests"]["ordered_result_record_sha256"],
        "result_record_set_sha256": completion["digests"]["result_record_set_sha256"],
        "codestream_object_set_sha256": completion["digests"]["codestream_object_set_sha256"],
        "reconstruction_object_set_sha256": completion["digests"]["reconstruction_object_set_sha256"],
        "f0_authorization_id": f0["authorization_id"],
        "f1_launch_id": launch["launch_id"],
        "ordered_pair_sha256": ORDERED_PAIR_SHA256,
        "pair_set_sha256": PAIR_SET_SHA256,
    }
    _require(completion["corpus_id"] == CORPUS_PREFIX + _sha(canonical_json(corpus_identity)), "F1 corpus content identity differs")

    if runtime_root is not None:
        live = collect_authenticated_runtime(Path(runtime_root), assignments, expected_total=EXPECTED_ATTEMPTS)
        _require(live["manifest_bytes"] == manifest_raw, "live F1 runtime does not reproduce frozen corpus manifest")
        _require(live["digests"] == completion["digests"], "live F1 runtime digests differ from completion")
        _require(live["outcomes"] == completion["outcomes"], "live F1 runtime outcomes differ from completion")
        _require(live["objects"] == completion["objects"], "live F1 runtime object accounting differs from completion")
    return completion


def verify_monitor_closeout(
    path: Path = MONITOR_CLOSEOUT_PATH,
    *,
    completion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the post-evidence Discord delivery and polling transition."""

    raw = Path(path).read_bytes()
    value = json.loads(raw)
    _require(raw == rendered_json(value), "F1 monitor closeout is not canonical rendered JSON")
    body = dict(value)
    monitor_id = body.pop("monitor_closeout_id", None)
    _require(monitor_id == "g8fmonitorcloseout-" + _sha(canonical_json(body)), "F1 monitor closeout identity differs")
    _require(value.get("artifact_role") == "g8_f_f1_discord_monitor_closeout" and value.get("status") == "COMPLETE", "F1 monitor closeout header differs")
    completion = verify_closeout() if completion is None else completion
    _require(value["completion_id"] == completion["completion_id"] and value["corpus_id"] == completion["corpus_id"], "F1 monitor closeout references other evidence")
    _require(value["delivery"] == {
        "assignments": "50814/50814",
        "delivered": True,
        "f2_closed": True,
        "http_status": 204,
        "materialized": 44039,
        "message_status": "F1 COMPLETE AUTHENTICATED",
        "timestamp": "2026-08-24T18:36:16Z",
        "typed_infeasible": 6775,
    }, "F1 authenticated Discord completion delivery differs")
    _require(value["transition"] == {
        "active_f1_polling": False,
        "service_active": "inactive",
        "timer_active": "inactive",
        "timer_enabled": "disabled",
        "webhook_configuration_deleted": False,
    }, "F1 Discord monitor polling transition differs")
    _require(value["secret_disclosure"] is False and value["f2_launched"] is False, "F1 monitor closeout crossed a protected boundary")
    return value
