#!/usr/bin/env python3
"""Run exactly one PAPR-constrained training lifecycle (AM-98). Owner-invoked only.

This wrapper reuses the frozen W8 trainer and validation/selection machinery with
the PAPR-constrained model factory.  It is not invoked by CI and must not run
until the owner separately freezes the PAPR training authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.downstream_v4 import immutable_write, read_json  # noqa: E402
from evaluation.w8_validation import evaluate_validation, select_checkpoint_epoch  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.papr_constrained import (  # noqa: E402
    PAPR_AUTHORITY_PATH,
    PAPR_CAMPAIGN_ID,
    PAPR_GPU_UUID,
    PAPR_RUN_ID,
    PAPR_SELECTED_CHECKPOINT_PATH,
    PaprConstrainedTrainer,
    load_papr_config,
    papr_cap_db,
)
from training.w8_final import W8SourceLineage  # noqa: E402
from config.w8_execution import authenticate_w8_gpu  # noqa: E402

SUMMARY_FILE = "validation_summaries.jsonl"


def _summaries_path(runtime_root: Path) -> Path:
    return runtime_root / SUMMARY_FILE


def _load_summaries(runtime_root: Path) -> list[dict]:
    path = _summaries_path(runtime_root)
    if not path.is_file():
        return []
    values = []
    for line in path.read_text().splitlines():
        if line.strip():
            values.append(json.loads(line))
    return values


def _append_summary(runtime_root: Path, summary: dict, sidecar: dict) -> None:
    path = _summaries_path(runtime_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"summary": summary, "checkpoint_path": sidecar["checkpoint_path"]}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def _trainer(authority: dict):
    from config.run_config import config_hash as run_config_hash

    config = load_papr_config()
    if run_config_hash(config) != authority["config_hash"]:
        raise SystemExit("PAPR config hash differs from the frozen authority")
    binding = authenticate_w8_gpu(config_hash=run_config_hash(config), expected_gpu_uuid=PAPR_GPU_UUID)
    lineage = W8SourceLineage(
        authority["source_commit"],
        authority["source_manifest"]["manifest_id"],
        authority["source_manifest"]["sha256"],
    )
    runtime_root = REPO / str(authority["runtime_root"])
    trainer = PaprConstrainedTrainer(
        config,
        device="cuda:0",
        runtime_root=runtime_root,
        source_lineage=lineage,
        profile_binding=binding,
        campaign_id=PAPR_CAMPAIGN_ID,
        run_id=PAPR_RUN_ID,
        num_workers=int(config.parameters["learned_system"]["dataloader_workers"]),
    )
    return trainer, config


def action_status(authority_path: Path) -> int:
    from verify_papr_training_authorization import verify_authority

    authority = verify_authority(authority_path)
    runtime_root = REPO / str(authority["runtime_root"])
    print(f"PAPR authority {authority['authority_id']}; runtime {runtime_root}; summaries {len(_load_summaries(runtime_root))}")
    return 0


def action_train(authority_path: Path) -> int:
    from verify_papr_training_authorization import verify_authority

    authority = verify_authority(authority_path)
    if (REPO / PAPR_SELECTED_CHECKPOINT_PATH).exists():
        raise SystemExit("a PAPR selected checkpoint already exists; the one run is immutable")
    trainer, _config = _trainer(authority)
    runtime_root = trainer.runtime_root
    if (runtime_root / "latest.json").is_file():
        trainer.resume()
    total = int(get("learned_system.epochs.imagenette160"))
    summaries = _load_summaries(runtime_root)
    if len(summaries) != trainer.completed_epoch + 1:
        raise SystemExit("PAPR validation history does not match the authenticated prefix")
    for epoch in range(trainer.completed_epoch + 1, total):
        from data.djscc_training import TrainingDJSCCDataset

        dataset = TrainingDJSCCDataset(str(trainer.config.resolved["dataset"]), 0, epoch, repo_root=REPO)
        record = trainer.train_epoch(epoch, dataset)
        sidecar = trainer.save_checkpoint(record)
        evaluation = evaluate_validation(trainer, checkpoint_id=sidecar["checkpoint_id"], repo_root=REPO)
        _append_summary(runtime_root, evaluation.summary, sidecar)
        summaries = _load_summaries(runtime_root)
        print(f"PAPR epoch {epoch + 1}/{total} n_correct={evaluation.summary['n_correct']}")
    _select(runtime_root, authority, summaries)
    return 0


def action_select(authority_path: Path) -> int:
    from verify_papr_training_authorization import verify_authority

    authority = verify_authority(authority_path)
    runtime_root = REPO / str(authority["runtime_root"])
    summaries = _load_summaries(runtime_root)
    _select(runtime_root, authority, summaries)
    return 0


def _select(runtime_root: Path, authority: dict, summaries: list[dict]) -> None:
    total = int(get("learned_system.epochs.imagenette160"))
    if len(summaries) != total:
        raise SystemExit(f"PAPR selection requires {total} complete validation summaries")
    selection = select_checkpoint_epoch([item["summary"] for item in summaries], expected_epochs=total)
    record = summaries[int(selection["selected_epoch"])]
    checkpoint_path = runtime_root / str(record["checkpoint_path"])
    from evaluation.downstream_v4 import sha256_file

    body = {
        "schema_version": 1,
        "artifact_role": "PAPR_CONSTRAINED_SELECTED_CHECKPOINT",
        "authority_id": authority["authority_id"],
        "source_manifest": authority["source_manifest"],
        "config_hash": authority["config_hash"],
        "papr_cap_db": papr_cap_db(),
        "train_seed": 0,
        "channel_seed": 0,
        "selected_epoch": int(selection["selected_epoch"]),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO)),
        "validation_n_correct": int(selection["n_correct"]),
        "validation_n_total": int(selection["n_total"]),
        "w8_selection": selection,
        "training_runs": 1,
        "test": "SEALED",
        "test_access": 0,
    }
    body["selection_id"] = "paprselected-" + canonical_sha256(body)
    immutable_write(REPO / PAPR_SELECTED_CHECKPOINT_PATH, body)
    immutable_write(runtime_root / "run_completion.json", body)
    print(f"PAPR selected checkpoint: {body['selection_id']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "train", "select"))
    parser.add_argument("--authority", type=Path, default=REPO / PAPR_AUTHORITY_PATH)
    args = parser.parse_args(argv)
    if args.action == "status":
        return action_status(args.authority)
    if args.action == "train":
        return action_train(args.authority)
    return action_select(args.authority)


if __name__ == "__main__":
    raise SystemExit(main())
