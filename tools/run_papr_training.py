#!/usr/bin/env python3
"""Run exactly one PAPR-constrained training lifecycle (AM-98). Owner-invoked only.

This wrapper reuses the frozen W8 trainer mechanics with the PAPR-constrained
model factory and the explicit PAPR artifact namespace.  It is not invoked by CI
and must not run until the owner separately freezes the PAPR training authority.

Custody: the worker must be checked out at the exact commit that carries the
frozen authority bytes; that commit's protected source must equal the frozen
downstream source epoch.  The authority/evidence publication commits may
advance HEAD only before launch.  The sole writer is this process; epochs are
committed by the authenticated checkpoint publication, and per-epoch validation
records are re-derivable evidence under ``validation/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.w8_execution import authenticate_w8_gpu  # noqa: E402
from evaluation.w8_validation import evaluate_validation, select_checkpoint_epoch, _validate_summary  # noqa: E402
from runtime.source_epochs import load_w10_manifest  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.papr_constrained import (  # noqa: E402
    PAPR_AUTHORITY_PATH,
    PAPR_CAMPAIGN_ID,
    PAPR_CHANNEL_SEED,
    PAPR_EPOCHS,
    PAPR_GPU_UUID,
    PAPR_RUN_ID,
    PAPR_SELECTED_CHECKPOINT_PATH,
    PAPR_TRAIN_SEED,
    PAPR_TRAINING_POLICY,
    PaprConstrainedTrainer,
    load_papr_config,
)
from training.papr_lifecycle import (  # noqa: E402
    PAPR_VALIDATION_NAMESPACE,
    assert_authority_source_closure,
    execution_commit_for,
    epoch_chain_digest,
    load_validation_summaries,
    publish_selected_and_completion,
    validate_summary_chain,
    verify_papr_authority,
)
from training.w8_final import W8SourceLineage, publish_immutable_json  # noqa: E402
from training.w8_protocol import W8_VALIDATION_BATCH_SIZE  # noqa: E402


def _trainer(authority: dict, *, root: Path):
    config = load_papr_config()
    if run_config_hash(config) != authority["config_hash"]:
        raise SystemExit("PAPR config hash differs from the frozen authority")
    if authority.get("protocol_config_hash") is None:
        raise SystemExit("PAPR authority carries no protocol config hash")
    manifest = load_w10_manifest(root, live=True)
    execution_commit = execution_commit_for(root, root / PAPR_AUTHORITY_PATH)
    assert_authority_source_closure(root, authority, execution_commit)
    if execution_commit != _git_head(root):
        raise SystemExit(
            "PAPR worker must run at the exact commit that carries the frozen authority"
        )
    binding = authenticate_w8_gpu(config_hash=run_config_hash(config), expected_gpu_uuid=PAPR_GPU_UUID)
    lineage = W8SourceLineage(
        execution_commit,
        manifest["manifest_id"],
        authority["source_manifest"]["sha256"],
    )
    runtime_root = root / str(authority["runtime_root"])
    trainer = PaprConstrainedTrainer(
        config,
        device="cuda:0",
        runtime_root=runtime_root,
        source_lineage=lineage,
        profile_binding=binding,
        campaign_id=PAPR_CAMPAIGN_ID,
        run_id=PAPR_RUN_ID,
        policy=PAPR_TRAINING_POLICY,
        num_workers=int(config.parameters["learned_system"]["dataloader_workers"]),
    )
    return trainer, config, execution_commit


def _git_head(root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit("cannot resolve the PAPR worker HEAD")
    return result.stdout.strip()


def _authenticate_summary_prefix(trainer, summaries: list[dict]) -> None:
    for epoch, summary in enumerate(summaries):
        _validate_summary(
            summary,
            expected_epoch=epoch,
            namespace=PAPR_VALIDATION_NAMESPACE,
        )
        if summary["campaign_id"] != trainer.campaign_id or summary["run_id"] != trainer.run_id:
            raise SystemExit("PAPR validation summary run differs")
        sidecar_path = trainer.runtime_root / f"checkpoints/epoch-{epoch:04d}.sidecar.json"
        sidecar = json.loads(sidecar_path.read_bytes())
        trainer._validate_sidecar(sidecar)
        if sidecar["checkpoint_id"] != summary["checkpoint_id"]:
            raise SystemExit("PAPR validation summary checkpoint binding differs")


def _publish_summary(runtime_root: Path, summary: dict) -> None:
    from evaluation.w8_validation import _validate_summary

    _validate_summary(
        summary,
        expected_epoch=int(summary["epoch"]),
        namespace=PAPR_VALIDATION_NAMESPACE,
    )
    publish_immutable_json(runtime_root / f"validation/epoch-{int(summary['epoch']):04d}.json", summary)


def _catch_up_validation(trainer, *, repo: Path) -> list[dict]:
    """Recompute any missing trailing validation records from authenticated checkpoints."""

    summaries = load_validation_summaries(trainer.runtime_root)
    if len(summaries) > trainer.completed_epoch + 1:
        raise SystemExit("PAPR validation history runs past the authenticated checkpoint prefix")
    _authenticate_summary_prefix(trainer, summaries)
    for epoch in range(len(summaries), trainer.completed_epoch + 1):
        sidecar = trainer.load_checkpoint_epoch(epoch)
        evaluation = evaluate_validation(
            trainer,
            checkpoint_id=sidecar["checkpoint_id"],
            repo_root=repo,
            namespace=PAPR_VALIDATION_NAMESPACE,
        )
        _publish_summary(trainer.runtime_root, evaluation.summary)
        summaries.append(evaluation.summary)
    if trainer.completed_epoch >= 0:
        trainer.resume()
    return summaries


def _run_chain(trainer, *, repo: Path) -> list[dict]:
    if (trainer.runtime_root / "latest.json").is_file():
        trainer.resume()
    elif trainer.runtime_root.exists() and any(trainer.runtime_root.iterdir()):
        trainer.discard_unpublished_genesis_suffix()
    summaries = _catch_up_validation(trainer, repo=repo)
    while trainer.completed_epoch < PAPR_EPOCHS - 1:
        from data.djscc_training import TrainingDJSCCDataset

        next_epoch = trainer.completed_epoch + 1
        dataset = TrainingDJSCCDataset(str(trainer.config.resolved["dataset"]), PAPR_TRAIN_SEED, next_epoch, repo_root=repo)
        record = trainer.train_epoch(next_epoch, dataset)
        sidecar = trainer.save_checkpoint(record)
        evaluation = evaluate_validation(
            trainer,
            checkpoint_id=sidecar["checkpoint_id"],
            repo_root=repo,
            namespace=PAPR_VALIDATION_NAMESPACE,
        )
        _publish_summary(trainer.runtime_root, evaluation.summary)
        summaries = load_validation_summaries(trainer.runtime_root)
        print(f"PAPR epoch {next_epoch + 1}/{PAPR_EPOCHS} n_correct={evaluation.summary['n_correct']}")
    if len(summaries) != PAPR_EPOCHS:
        raise SystemExit("PAPR training did not publish the complete validation trajectory")
    return summaries


def _select(root: Path, authority: dict, *, authority_path: Path, execution_commit: str) -> dict:
    trainer, _config, _commit = _trainer(authority, root=root)
    if (trainer.runtime_root / "latest.json").is_file():
        trainer.resume()
    if trainer.completed_epoch != PAPR_EPOCHS - 1:
        raise SystemExit("PAPR selection requires the complete 100-epoch chain")
    if (root / PAPR_SELECTED_CHECKPOINT_PATH).exists():
        raise SystemExit("a PAPR selected checkpoint already exists; the one run is immutable")
    chain = _chain_summary(root, authority, execution_commit)
    summaries = validate_summary_chain(root, authority, chain)
    selection = select_checkpoint_epoch(
        summaries, expected_epochs=PAPR_EPOCHS, namespace=PAPR_VALIDATION_NAMESPACE
    )
    selected_epoch = int(selection["selected_epoch"])
    sidecar = json.loads(
        (trainer.runtime_root / f"checkpoints/epoch-{selected_epoch:04d}.sidecar.json").read_bytes()
    )
    checkpoint_relative = f"{authority['runtime_root']}/checkpoints/epoch-{selected_epoch:04d}.pt"
    checkpoint = root / checkpoint_relative
    selected, completion = publish_selected_and_completion(
        root,
        authority=authority,
        authority_path=authority_path,
        execution_commit=execution_commit,
        chain=chain,
        summaries=summaries,
        selection=selection,
        selected_sidecar=sidecar,
        checkpoint_relative=checkpoint_relative,
        checkpoint_bytes=checkpoint.stat().st_size,
    )
    print(f"PAPR selected checkpoint: {selected['selection_id']}")
    print(f"PAPR completion: {completion['completion_id']}")
    return selected


def _chain_summary(root: Path, authority: dict, execution_commit: str) -> dict:
    """Rebuild the chain aggregate from the runtime without model inference."""

    entries = []
    runtime = root / str(authority["runtime_root"])
    total_steps = 0
    total_skips = 0
    total_opportunities = 0
    max_papr = None
    initialization = None
    for epoch in range(PAPR_EPOCHS):
        record = json.loads((runtime / f"epochs/epoch-{epoch:04d}.json").read_bytes())
        sidecar = json.loads((runtime / f"checkpoints/epoch-{epoch:04d}.sidecar.json").read_bytes())
        total_steps += int(record["optimizer_steps"])
        total_skips += int(record["grad_scaler_skips"])
        total_opportunities += int(record["optimizer_step_opportunities"])
        if max_papr is None:
            max_papr = float(record["papr_checks"]["max_observed_db"])
        else:
            max_papr = max(max_papr, float(record["papr_checks"]["max_observed_db"]))
        if initialization is None:
            initialization = dict(sidecar["initialization"])
        entries.append(
            {
                "epoch": epoch,
                "epoch_record_sha256": sidecar["epoch_record_sha256"],
                "checkpoint_id": sidecar["checkpoint_id"],
                "global_optimizer_step": sidecar["global_optimizer_step"],
            }
        )
    if initialization is None or "initial_model_state_sha256" not in initialization:
        raise SystemExit("PAPR chain initialization is incomplete")
    return {
        "entries": entries,
        "epoch_chain_digest": epoch_chain_digest(entries),
        "optimizer_steps": total_steps,
        "grad_scaler_skips": total_skips,
        "optimizer_step_opportunities": total_opportunities,
        "global_optimizer_step": total_steps,
        "max_observed_papr_db": float(max_papr),
        "initialization": initialization,
    }


def action_status(authority_path: Path) -> int:
    authority = verify_papr_authority(REPO, authority_path)
    runtime_root = REPO / str(authority["runtime_root"])
    summaries = load_validation_summaries(REPO / str(authority["runtime_root"]))
    print(
        f"PAPR authority {authority['authority_id']}; runtime {runtime_root}; "
        f"validation records {len(summaries)}"
    )
    return 0


def action_train(authority_path: Path) -> int:
    authority = verify_papr_authority(REPO, authority_path)
    if (REPO / PAPR_SELECTED_CHECKPOINT_PATH).exists():
        raise SystemExit("a PAPR selected checkpoint already exists; the one run is immutable")
    trainer, _config, execution_commit = _trainer(authority, root=REPO)
    _run_chain(trainer, repo=REPO)
    _select(REPO, authority, authority_path=authority_path, execution_commit=execution_commit)
    return 0


def action_select(authority_path: Path) -> int:
    authority = verify_papr_authority(REPO, authority_path)
    execution_commit = execution_commit_for(REPO, authority_path)
    _select(REPO, authority, authority_path=authority_path, execution_commit=execution_commit)
    return 0


def action_verify(authority_path: Path) -> int:
    from training.papr_lifecycle import verify_papr_terminal

    value = verify_papr_terminal(REPO, authority_path=authority_path)
    print(f"PAPR terminal PASS: {value['completion_id']} cap_max={value['papr_max_observed_db']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "train", "select", "verify"))
    parser.add_argument("--authority", type=Path, default=REPO / PAPR_AUTHORITY_PATH)
    args = parser.parse_args(argv)
    if args.action == "status":
        return action_status(args.authority)
    if args.action == "train":
        return action_train(args.authority)
    if args.action == "select":
        return action_select(args.authority)
    return action_verify(args.authority)


if __name__ == "__main__":
    raise SystemExit(main())
