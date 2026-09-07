#!/usr/bin/env python3
"""Verify ER-9 source/authentication artifacts and terminal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash, load_experiment  # noqa: E402
from evaluation.er9_protocol import factorisation_for_dimension  # noqa: E402
from evaluation.er9_search import all_configured_pairs, feasible_pairs, packetisation_floor, stage1_candidates, select_stage1, select_stage2, stage2_candidates  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from gen_er9_source_manifest import CRITICAL_SOURCES  # noqa: E402


SOURCE_DEFAULT = REPO / "results/learned/er9/er_execution_source_manifest.json"
STAGE1_DEFAULT = REPO / "results/learned/er9/er9_stage1_execution_authorization.json"
STAGE1_SELECTION_DEFAULT = REPO / "results/learned/er9/er9_stage1_selection.json"
STAGE2_DEFAULT = REPO / "results/learned/er9/er9_stage2_execution_authorization.json"
STAGE2_SELECTION_DEFAULT = REPO / "results/learned/er9/er9_stage2_selection.json"
PRODUCTION_DEFAULT = REPO / "results/learned/er9/er9_final_production_manifest.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _source_bytes(commit: str, path: str) -> tuple[bytes, str]:
    raw = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True, check=True).stdout
    blob = _git("rev-parse", f"{commit}:{path}")
    return raw, blob


def _repo_file(relative: str) -> Path:
    path = (REPO / relative).resolve()
    if REPO.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError(f"ER-9 artifact is missing or unsafe: {relative}")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    required = {
        "schema_version", "artifact_role", "status", "source_commit",
        "source_commit_comparison", "entries", "entry_count",
        "scientific_results_included", "er9_training", "randomized_er2_training",
        "g11", "test_access", "test", "manifest_id",
    }
    if set(value) != required:
        raise ValueError("ER source manifest schema differs")
    body = dict(value)
    identifier = body.pop("manifest_id")
    if identifier != "ersource-" + canonical_sha256(body):
        raise ValueError("ER source manifest ID differs")
    commit = str(value["source_commit"])
    if len(commit) != 40 or _git("rev-parse", "--verify", f"{commit}^{{commit}}") != commit:  # literal-ok: Git SHA-1 width
        raise ValueError("ER source manifest commit is invalid")
    entries = value["entries"]
    expected = []
    for path_name, role in CRITICAL_SOURCES:
        raw, blob = _source_bytes(commit, path_name)
        expected.append({
            "path": path_name,
            "role": role,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": blob,
        })
    if entries != expected or value["entry_count"] != len(expected):
        raise ValueError("ER source manifest source bytes differ")
    listing = _git("ls-tree", "-r", "--name-only", commit, "--", "results/learned/er9")
    if any(line == "results/learned/er9" or line.startswith("results/learned/er9/") for line in listing.splitlines()):
        raise ValueError("ER source epoch contains scientific results")
    if value["scientific_results_included"] is not False or value["er9_training"] != 0 or value["randomized_er2_training"] != 0 or value["g11"] != 0 or value["test_access"] != 0 or value["test"] != "SEALED":
        raise ValueError("ER source manifest pre-science counters differ")
    return value


def verify_stage1_authorization(path: Path, manifest_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    required = {
        "schema_version", "artifact_role", "status", "authorization_scope", "source_commit",
        "source_manifest_id", "source_manifest_sha256", "config_path", "config_hash",
        "execution_profile_selection", "dataset", "split", "ratio", "k_symbols",
        "evaluation_snr_db", "search_seed_cell", "packet_floor", "metadata_bits",
        "configured_pair_count", "admissible_pair_count", "admissible_pairs", "rejected_pairs",
        "stage1_rule", "stage1_candidates", "stage1_training_count", "stage2_authorization",
        "final_seed_scope", "pre_execution_counters", "test", "authorization_id",
    }
    if set(value) != required:
        raise ValueError("ER-9 Stage-1 authorization schema differs")
    body = dict(value)
    identifier = body.pop("authorization_id")
    if identifier != "er9stage1auth-" + canonical_sha256(body):
        raise ValueError("ER-9 Stage-1 authorization ID differs")
    manifest = verify_source_manifest(manifest_path)
    if value["status"] != "FROZEN_BEFORE_FIRST_ER9_OPTIMIZER_STEP" or value["authorization_scope"] != "ER9_STAGE1_ONLY":
        raise ValueError("Stage-1 authorization status/scope differs")
    if value["source_commit"] != manifest["source_commit"] or value["source_manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Stage-1 authorization source binding differs")
    if value["source_manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Stage-1 authorization manifest hash differs")
    config = load_experiment("configs/er9-digital.yaml", train_seed=0, channel_seed=0)  # literal-ok: first zipped seed cell
    if value["config_hash"] != config_hash(config) or value["config_path"] != "configs/er9-digital.yaml":
        raise ValueError("Stage-1 authorization config binding differs")
    floor = packetisation_floor(int(config.resolved["k"]))
    if value["packet_floor"] != floor.as_dict() or value["k_symbols"] != int(config.resolved["k"]):
        raise ValueError("Stage-1 exact packet floor differs")
    if value["dataset"] != "imagenette160" or value["split"] != "train" or value["ratio"] != "r_1_6" or value["evaluation_snr_db"] != 7:
        raise ValueError("Stage-1 dataset/ratio/SNR binding differs")
    if value["metadata_bits"] != 1:  # literal-ok: AM-96 selector-only metadata
        raise ValueError("Stage-1 metadata accounting differs")
    all_pairs = all_configured_pairs()
    admissible = feasible_pairs(floor.payload_bits)
    expected_stage1 = stage1_candidates(floor.payload_bits)
    if value["configured_pair_count"] != len(all_pairs) or value["admissible_pair_count"] != len(admissible):
        raise ValueError("Stage-1 pair counts differ")
    if value["admissible_pairs"] != [candidate.as_dict() for candidate in admissible]:
        raise ValueError("Stage-1 admissible pair list differs")
    expected_rejected = [
        {**candidate.as_dict(), "reason": "raw_bound_exceeds_A_floor"}
        for candidate in all_pairs if candidate not in admissible
    ]
    if value["rejected_pairs"] != expected_rejected:
        raise ValueError("Stage-1 rejected pair list differs")
    if value["stage1_candidates"] != [candidate.as_dict() for candidate in expected_stage1] or value["stage1_training_count"] != len(expected_stage1):
        raise ValueError("Stage-1 candidate list/count differs")
    if value["stage1_rule"] != {
        "quantiser_bits": 2,  # literal-ok: AM-96 stage-1 minimum width
        "candidate_order": "ascending_numeric_transmit_dim",
        "selection_metric": "exact_validation_n_correct_at_7db_real_digital_chain",
        "tie_break": "smallest_transmit_dim",
        "cross_product": False,
    }:
        raise ValueError("Stage-1 search rule differs")
    if value["execution_profile_selection"] != {
        "execution_profile_id": "local_4060_cu130",
        "device": "cuda:0",
        "status": "selected_before_first_scientific_measurement",
        "sole_writer": True,
    }:
        raise ValueError("Stage-1 execution-profile selection differs")
    if value["search_seed_cell"] != {"train_seed": 0, "channel_seed": 0}:  # literal-ok: first zipped seed cell
        raise ValueError("Stage-1 seed cell differs")
    if value["pre_execution_counters"] != {
        "er9_training": 0,
        "randomized_er2_training": 0,
        "g11": 0,
        "w10": 0,
        "learned_test_inference": 0,
        "model_facing_test_access": 0,
    }:
        raise ValueError("Stage-1 pre-execution counters differ")
    if value["test"] != "SEALED":
        raise ValueError("Stage-1 test boundary differs")
    return value


def _verify_evaluation_body(path: Path, expected_prefix: str, candidate: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"ER-9 evaluation is missing or unsafe: {path}")
    body = json.loads(path.read_bytes())
    if not isinstance(body, dict):
        raise ValueError(f"ER-9 evaluation is not an object: {path}")
    evaluation_id = body.get("evaluation_id")
    without_id = dict(body)
    without_id.pop("evaluation_id", None)
    if evaluation_id != expected_prefix + canonical_sha256(without_id):
        raise ValueError(f"ER-9 evaluation ID differs: {path}")
    if body.get("artifact_role") != "ER9_STAGE_SEARCH_CANDIDATE_EVALUATION" or body.get("candidate") != candidate or body.get("test_access") != 0:
        raise ValueError(f"ER-9 evaluation candidate/scope differs: {path}")
    real = body.get("real_chain_validation", {})
    if real.get("validation_total") != int(get("datasets.imagenette160.val_images")):
        raise ValueError(f"ER-9 evaluation denominator differs: {path}")
    if not isinstance(real.get("validation_n_correct"), int) or isinstance(real.get("validation_n_correct"), bool):
        raise ValueError(f"ER-9 evaluation count is not exact: {path}")
    if real.get("snr_db") != 7 or real.get("transmit_dim") != candidate["transmit_dim"] or real.get("quantiser_bits") != candidate["quantiser_bits"]:
        raise ValueError(f"ER-9 evaluation protocol binding differs: {path}")
    if real.get("selected") not in real.get("candidates", []):
        raise ValueError(f"ER-9 selected PHY is not one of the evaluated candidates: {path}")
    return body


def verify_stage1_selection(path: Path, auth_path: Path, manifest_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("ER-9 Stage-1 selection is not an object")
    body = dict(value)
    identifier = body.pop("selection_id", None)
    if identifier != "er9stage1selection-" + canonical_sha256(body):
        raise ValueError("ER-9 Stage-1 selection ID differs")
    auth = verify_stage1_authorization(auth_path, manifest_path)
    if value.get("authorization_id") != auth["authorization_id"]:
        raise ValueError("ER-9 Stage-1 selection authorization differs")
    if value.get("candidate_order") != auth["stage1_candidates"]:
        raise ValueError("ER-9 Stage-1 candidate order differs")
    expected_files = [
        f"results/learned/er9/stage1_evaluations/D{candidate['transmit_dim']}_b{candidate['quantiser_bits']}.json"
        for candidate in auth["stage1_candidates"]
    ]
    if value.get("candidate_evaluation_files") != expected_files:
        raise ValueError("ER-9 Stage-1 evaluation file order differs")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != auth["stage1_training_count"]:
        raise ValueError("ER-9 Stage-1 rows/count differs")
    expected_rows: list[dict[str, Any]] = []
    for candidate, row, relative in zip(auth["stage1_candidates"], rows, expected_files, strict=True):
        if row.get("transmit_dim") != candidate["transmit_dim"] or row.get("quantiser_bits") != candidate["quantiser_bits"]:
            raise ValueError("ER-9 Stage-1 row order differs")
        evaluation_path = REPO / str(relative)
        evaluation = _verify_evaluation_body(evaluation_path, "er9stage1eval-", candidate)
        if evaluation["evaluation_id"] != row["evaluation_id"]:
            raise ValueError("ER-9 Stage-1 evaluation ID binding differs")
        if row["n_correct"] != evaluation["real_chain_validation"]["validation_n_correct"] or row["n_total"] != evaluation["real_chain_validation"]["validation_total"]:
            raise ValueError("ER-9 Stage-1 count binding differs")
        if row["selected_phy"] != evaluation["real_chain_validation"]["selected"]:
            raise ValueError("ER-9 Stage-1 PHY binding differs")
        if row.get("runtime_root") != f"checkpoints/er9/stage1/D{candidate['transmit_dim']}_b{candidate['quantiser_bits']}":
            raise ValueError("ER-9 Stage-1 runtime binding differs")
        expected_rows.append(row)
    if value.get("test_access") != 0 or value.get("status") != "STAGE1_COMPLETE_STAGE2_NOT_AUTHORIZED_IN_THIS_ARTIFACT":
        raise ValueError("ER-9 Stage-1 selection scope/status differs")
    if value.get("selected") != dict(select_stage1(expected_rows)):
        raise ValueError("ER-9 Stage-1 selected row differs")
    return value


def verify_stage2_authorization(path: Path, stage1_path: Path, stage1_auth_path: Path, manifest_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("ER-9 Stage-2 authorization is not an object")
    body = dict(value)
    identifier = body.pop("authorization_id", None)
    if identifier != "er9stage2auth-" + canonical_sha256(body):
        raise ValueError("ER-9 Stage-2 authorization ID differs")
    stage1 = verify_stage1_selection(stage1_path, stage1_auth_path, manifest_path)
    stage1_auth = verify_stage1_authorization(stage1_auth_path, manifest_path)
    if value.get("artifact_role") != "ER9_STAGE2_EXECUTION_AUTHORIZATION" or value.get("authorization_scope") != "ER9_STAGE2_ONLY":
        raise ValueError("ER-9 Stage-2 authorization role/scope differs")
    source_commit = str(value.get("source_commit"))
    if len(source_commit) != 40 or _git("rev-parse", "--verify", f"{source_commit}^{{commit}}") != source_commit:  # literal-ok: Git SHA-1 width
        raise ValueError("ER-9 Stage-2 authorization source commit is invalid")
    if value.get("implementation_commit") != stage1_auth["source_commit"] or value.get("config_path") != "configs/er9-digital.yaml" or value.get("config_hash") != stage1_auth["config_hash"]:
        raise ValueError("ER-9 Stage-2 source/config binding differs")
    if value.get("stage1_authorization_id") != stage1_auth["authorization_id"] or value.get("source_manifest_sha256") != stage1_auth["source_manifest_sha256"]:
        raise ValueError("ER-9 Stage-2 Stage-1 authorization binding differs")
    selected_dimension = int(stage1["selected"]["transmit_dim"])
    floor = packetisation_floor(int(stage1_auth["k_symbols"]))
    expected = [candidate.as_dict() for candidate in stage2_candidates(floor.payload_bits, selected_dimension)]
    if value.get("status") != "FROZEN_BEFORE_STAGE2_OPTIMIZER_STEP" or value.get("cross_product") is not False:
        raise ValueError("ER-9 Stage-2 status/search rule differs")
    if value.get("source_manifest_id") != stage1_auth["source_manifest_id"] or value.get("stage1_selection_id") != stage1["selection_id"]:
        raise ValueError("ER-9 Stage-2 lineage differs")
    if value.get("stage1_selection_sha256") != hashlib.sha256(stage1_path.read_bytes()).hexdigest():
        raise ValueError("ER-9 Stage-2 selection hash differs")
    if value.get("selected_transmit_dim") != selected_dimension or value.get("stage1_reused_candidate") != {"transmit_dim": selected_dimension, "quantiser_bits": 2}:  # literal-ok: AM-96 Stage-2 reuse
        raise ValueError("ER-9 Stage-2 reused candidate differs")
    if value.get("stage2_candidate_order") != expected or value.get("stage2_training_count") != max(len(expected) - 1, 0):
        raise ValueError("ER-9 Stage-2 candidate/count differs")
    if value.get("stage2_bits_order") != "ascending_numeric" or value.get("stage2_selection_metric") != "exact_validation_n_correct_at_7db_real_digital_chain" or value.get("stage2_tie_break") != "smallest_quantiser_bits":
        raise ValueError("ER-9 Stage-2 policy differs")
    if value.get("test") != "SEALED":
        raise ValueError("ER-9 Stage-2 test boundary differs")
    return value


def verify_stage2_selection(path: Path, stage2_auth_path: Path, stage1_path: Path, stage1_auth_path: Path, manifest_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("ER-9 Stage-2 selection is not an object")
    body = dict(value)
    identifier = body.pop("selection_id", None)
    if identifier != "er9stage2selection-" + canonical_sha256(body):
        raise ValueError("ER-9 Stage-2 selection ID differs")
    auth = verify_stage2_authorization(stage2_auth_path, stage1_path, stage1_auth_path, manifest_path)
    if value.get("authorization_id") != auth["authorization_id"] or value.get("candidate_order") != auth["stage2_candidate_order"]:
        raise ValueError("ER-9 Stage-2 selection lineage/order differs")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(auth["stage2_candidate_order"]):
        raise ValueError("ER-9 Stage-2 row count differs")
    stage1 = json.loads(stage1_path.read_bytes())
    for candidate, row in zip(auth["stage2_candidate_order"], rows, strict=True):
        if row.get("transmit_dim") != candidate["transmit_dim"] or row.get("quantiser_bits") != candidate["quantiser_bits"]:
            raise ValueError("ER-9 Stage-2 row order differs")
        if bool(row.get("reused_stage1")):
            if candidate != auth["stage1_reused_candidate"]:
                raise ValueError("ER-9 Stage-2 reused row is not b=2")
            expected = next(item for item in stage1["rows"] if item["transmit_dim"] == candidate["transmit_dim"] and item["quantiser_bits"] == candidate["quantiser_bits"])
            if row.get("evaluation_id") != expected["evaluation_id"] or row.get("n_correct") != expected["n_correct"]:
                raise ValueError("ER-9 Stage-2 reused row differs")
        else:
            evaluation_path = REPO / "results/learned/er9/stage2_evaluations" / f"D{candidate['transmit_dim']}_b{candidate['quantiser_bits']}.json"
            evaluation = _verify_evaluation_body(evaluation_path, "er9stage2eval-", candidate)
            if row.get("evaluation_id") != evaluation["evaluation_id"]:
                raise ValueError("ER-9 Stage-2 evaluation binding differs")
            if row.get("n_correct") != evaluation["real_chain_validation"]["validation_n_correct"]:
                raise ValueError("ER-9 Stage-2 count binding differs")
    if value.get("selected") != dict(select_stage2(rows)) or value.get("test_access") != 0 or value.get("status") != "SELECTED_PAIR_FROZEN":
        raise ValueError("ER-9 Stage-2 selected row/scope differs")
    for candidate, row in zip(auth["stage2_candidate_order"], rows, strict=True):
        if row.get("runtime_root") != (
            f"checkpoints/er9/stage1/D{candidate['transmit_dim']}_b{candidate['quantiser_bits']}"
            if row.get("reused_stage1")
            else f"checkpoints/er9/stage2/D{candidate['transmit_dim']}_b{candidate['quantiser_bits']}"
        ):
            raise ValueError("ER-9 Stage-2 runtime binding differs")
    if value.get("selected_factorisation") != factorisation_for_dimension(int(value["selected"]["transmit_dim"])).as_dict():
        raise ValueError("ER-9 Stage-2 factorisation differs")
    return value


def verify_final_production(path: Path, stage1_path: Path, stage1_auth_path: Path, stage2_path: Path, stage2_auth_path: Path, manifest_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("ER-9 production manifest is not an object")
    stage1_auth = verify_stage1_authorization(stage1_auth_path, manifest_path)
    stage1 = verify_stage1_selection(stage1_path, stage1_auth_path, manifest_path)
    stage2_auth = verify_stage2_authorization(stage2_auth_path, stage1_path, stage1_auth_path, manifest_path)
    stage2 = verify_stage2_selection(stage2_path, stage2_auth_path, stage1_path, stage1_auth_path, manifest_path)
    pair = value.get("selected_pair")
    if pair != {"transmit_dim": stage2["selected"]["transmit_dim"], "quantiser_bits": stage2["selected"]["quantiser_bits"]}:
        raise ValueError("ER-9 production selected pair differs")
    if value.get("selected_factorisation") != factorisation_for_dimension(int(pair["transmit_dim"])).as_dict():
        raise ValueError("ER-9 production factorisation differs")
    architecture_path = _repo_file(str(value["architecture_difference"]))
    if _sha(architecture_path) != value.get("architecture_difference_sha256"):
        raise ValueError("ER-9 architecture-difference hash differs")
    architecture = json.loads(architecture_path.read_bytes())
    if (
        architecture.get("artifact_role") != "ER9_ARCHITECTURE_DIFFERENCE_AUDIT"
        or architecture.get("declared_difference") != "channel_interface"
        or architecture.get("architecture_difference_paths") != ["channel_interface"]
        or architecture.get("only_declared_difference") is not True
        or architecture.get("test_access") != 0
    ):
        raise ValueError("ER-9 architecture-difference closure differs")
    expected_cells = [(0, 0), (1, 1), (2, 2)]  # literal-ok: three preregistered zipped cells
    cells = value.get("seed_cells", [])
    if [(int(cell["train_seed"]), int(cell["channel_seed"])) for cell in cells] != expected_cells:
        raise ValueError("ER-9 production seed cells differ")
    for index, cell in enumerate(cells):
        if cell.get("candidate") != pair or cell.get("test_access") != 0:
            raise ValueError("ER-9 production cell candidate/scope differs")
        if bool(cell.get("promoted_search_run")) != (index == 0):
            raise ValueError("ER-9 production promotion rule differs")
        cell_cfg = load_experiment(
            "configs/er9-digital.yaml",
            train_seed=int(cell["train_seed"]),
            channel_seed=int(cell["channel_seed"]),
        )
        if cell.get("config_hash") != config_hash(cell_cfg):
            raise ValueError("ER-9 production cell config hash differs")
        if cell.get("completion_sha256") != _sha(_repo_file(str(cell["runtime_root"]) + "/run_completion.json")):
            raise ValueError("ER-9 production completion hash differs")
    expected_search_count = int(stage1_auth["stage1_training_count"] + stage2_auth["stage2_training_count"])
    if value.get("search_training_count") != expected_search_count:
        raise ValueError("ER-9 production search count differs")
    if value.get("full_configured_pair_count") != stage1_auth["configured_pair_count"] or value.get("admissible_pair_count") != stage1_auth["admissible_pair_count"]:
        raise ValueError("ER-9 production pair counts differ")
    floor = stage1_auth["packet_floor"]
    budget = value.get("budget_proof", {})
    if budget.get("A_floor_bits") != floor["payload_bits"] or budget.get("selected_pair_feasible") is not True or budget.get("selected_raw_bound_bits") != int(pair["transmit_dim"]) * int(pair["quantiser_bits"]) + int(stage1_auth["metadata_bits"]):
        raise ValueError("ER-9 production budget proof differs")
    if value.get("packet_floor") != floor or value.get("test_access") != 0 or value.get("test") != "SEALED":
        raise ValueError("ER-9 production packet/scope differs")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--stage1", type=Path, default=STAGE1_DEFAULT)
    parser.add_argument("--stage1-selection", type=Path, default=STAGE1_SELECTION_DEFAULT)
    parser.add_argument("--stage2", type=Path, default=STAGE2_DEFAULT)
    parser.add_argument("--stage2-selection", type=Path, default=STAGE2_SELECTION_DEFAULT)
    parser.add_argument("--production", type=Path, default=PRODUCTION_DEFAULT)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args(argv)
    source = args.source if args.source.is_absolute() else REPO / args.source
    stage1 = args.stage1 if args.stage1.is_absolute() else REPO / args.stage1
    stage1_selection = args.stage1_selection if args.stage1_selection.is_absolute() else REPO / args.stage1_selection
    stage2 = args.stage2 if args.stage2.is_absolute() else REPO / args.stage2
    stage2_selection = args.stage2_selection if args.stage2_selection.is_absolute() else REPO / args.stage2_selection
    production = args.production if args.production.is_absolute() else REPO / args.production
    manifest = verify_source_manifest(source)
    print(f"ER source manifest PASS: {manifest['manifest_id']}")
    if not args.source_only:
        authorization = verify_stage1_authorization(stage1, source)
        print(f"ER-9 Stage-1 authorization PASS: {authorization['authorization_id']}")
    if args.terminal:
        selection1 = verify_stage1_selection(stage1_selection, stage1, source)
        print(f"ER-9 Stage-1 selection PASS: {selection1['selection_id']}")
        authorization2 = verify_stage2_authorization(stage2, stage1_selection, stage1, source)
        print(f"ER-9 Stage-2 authorization PASS: {authorization2['authorization_id']}")
        selection2 = verify_stage2_selection(stage2_selection, stage2, stage1_selection, stage1, source)
        print(f"ER-9 Stage-2 selection PASS: {selection2['selection_id']}")
        final = verify_final_production(production, stage1_selection, stage1, stage2_selection, stage2, source)
        print(f"ER-9 terminal production PASS: selected ({final['selected_pair']['transmit_dim']},{final['selected_pair']['quantiser_bits']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
