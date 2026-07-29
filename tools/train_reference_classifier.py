#!/usr/bin/env python3
"""Run the reference classifier only in explicit full or bounded smoke mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash, load_reference_classifier_config  # noqa: E402
from env import set_deterministic_backend  # noqa: E402
from training.reference_classifier import ReferenceClassifierTrainer  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--smoke-val-batches", type=int, default=2)
    parser.add_argument("--full-run", action="store_true")
    return parser


def _smoke_root(value: Path | None) -> Path:
    root = (value or REPO / "checkpoints" / "smoke" / "reference-classifier").resolve()
    allowed = (REPO / "checkpoints" / "smoke").resolve()
    if not root.is_relative_to(allowed):
        raise ValueError(f"smoke output root must remain under {allowed}: {root}")
    return root


def _production_paths() -> dict[str, Path]:
    """Return the fixed AM-78 artifact locations for a real full run."""

    return {
        "artifact_dir": REPO / get("artifacts.classifier_artifact_dir"),
        "resolved_config": REPO / get("artifacts.classifier_resolved_config_file"),
        "epochs": REPO / get("artifacts.classifier_epoch_log_file"),
        "validation_summary": REPO / get("artifacts.classifier_validation_summary_file"),
        "best_checkpoint": REPO / get("artifacts.classifier_best_checkpoint_metadata_file"),
        "checkpoint_dir": REPO / get("artifacts.classifier_checkpoint_dir"),
    }


def _paths_for_run(*, smoke: bool, output_root: Path | None) -> dict[str, Path]:
    if not smoke:
        if output_root is not None:
            raise ValueError("--output-root is smoke-only; full runs use configured artifact paths")
        return _production_paths()
    root = _smoke_root(output_root)
    return {
        "artifact_dir": root,
        "resolved_config": root / "resolved_config.json",
        "epochs": root / "epochs.jsonl",
        "validation_summary": root / "validation_summary.json",
        "best_checkpoint": root / "best_checkpoint.json",
        "checkpoint_dir": root / "checkpoints",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _repository_relative_posix(path: Path, *, repository: Path = REPO) -> str:
    """Return a portable repository-relative path, rejecting external targets."""

    root = repository.resolve()
    candidate = path.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"checkpoint path must remain under repository {root}: {candidate}") from None
    return relative.as_posix()


def _write_epoch_log(path: Path, training_history: list[dict[str, Any]], validation_history: list[dict[str, Any]]) -> None:
    validation_by_epoch = {int(record["epoch"]): record for record in validation_history}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in training_history:
            training_summary = {
                key: value for key, value in record.items() if key != "sample_order"
            }
            line = {
                "training": training_summary,
                "validation": validation_by_epoch.get(int(record["epoch"])),
            }
            stream.write(json.dumps(line, sort_keys=True) + "\n")


def main() -> int:
    args = _parser().parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.full_run and (args.smoke_steps != 3 or args.smoke_val_batches != 2):
        raise ValueError("--smoke-* controls are incompatible with --full-run")
    # This must precede construction, data views and every artifact directory.
    # The helper applies and read-back verifies the configured SR-12 mappings;
    # this contract has no separate deterministic-algorithms policy to apply.
    set_deterministic_backend()
    config = load_reference_classifier_config(args.config, dataset=args.dataset)
    smoke = not args.full_run
    paths = _paths_for_run(smoke=smoke, output_root=args.output_root)
    # Full lineage is established by trainer.run_epochs() before it may create
    # any production artifact. Smoke output has no production lineage.
    if smoke:
        paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    trainer = ReferenceClassifierTrainer(config, device=args.device)
    if args.resume:
        trainer.resume(args.resume, execution_mode="full" if args.full_run else "smoke")
    if args.full_run:
        final_epoch = int(get("reference_classifier.epochs")) - 1
    else:
        final_epoch = trainer.state.completed_epoch + 1
    records = trainer.run_epochs(
        final_epoch=final_epoch,
        checkpoint_dir=paths["checkpoint_dir"],
        num_workers=args.num_workers,
        execution_mode="full" if args.full_run else "smoke",
        full_run_requested=args.full_run,
        smoke_steps=None if args.full_run else args.smoke_steps,
        smoke_val_batches=None if args.full_run else args.smoke_val_batches,
        run_complete=args.full_run,
        g1_eligible=args.full_run,
    )
    final = records[-1]
    checkpoint_ids = {
        int(record["epoch"]): str(record["checkpoint_id"])
        for record in trainer.state.checkpoint_history
    }
    checkpoint_ids.update({record.epoch: record.checkpoint_id for record in records})
    best_epoch = trainer.state.best_epoch
    best_checkpoint_id = checkpoint_ids.get(best_epoch) if best_epoch is not None else None
    best_checkpoint = (
        paths["checkpoint_dir"] / f"epoch-{best_epoch}.pt" if best_checkpoint_id is not None else None
    )
    best_checkpoint_path = (
        _repository_relative_posix(best_checkpoint) if best_checkpoint is not None else None
    )
    final_checkpoint_path = _repository_relative_posix(final.path)
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    _write_json(paths["resolved_config"], config.to_dict())
    _write_epoch_log(paths["epochs"], trainer.state.training_history, trainer.state.validation_history)
    _write_json(
        paths["validation_summary"],
        {
            "administrative": {"device": args.device, "num_workers": args.num_workers},
            "config_hash": config_hash(config),
            "run_complete": args.full_run,
            "g1_eligible": args.full_run,
            "best_epoch": best_epoch,
            "best_validation_top1": trainer.state.best_validation_top1,
            "final_epoch": trainer.state.completed_epoch,
            "final_checkpoint_id": final.checkpoint_id,
        },
    )
    _write_json(
        paths["best_checkpoint"],
        {
            "best_epoch": best_epoch,
            "best_validation_top1": trainer.state.best_validation_top1,
            "best_checkpoint": best_checkpoint_path,
            "best_checkpoint_id": best_checkpoint_id,
            "final_epoch": trainer.state.completed_epoch,
            "final_checkpoint": final_checkpoint_path,
            "final_checkpoint_id": final.checkpoint_id,
        },
    )
    print(
        f"checkpoint={final.path} checkpoint_id={final.checkpoint_id} "
        f"run_complete={args.full_run} g1_eligible={args.full_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
