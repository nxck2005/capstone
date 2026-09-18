#!/usr/bin/env python3
"""Verify the frozen one-run PAPR-constrained training authority (AM-98)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from evaluation.downstream_v4 import read_json, require  # noqa: E402
from runtime.source_epochs import load_w10_manifest, source_record  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.papr_constrained import (  # noqa: E402
    PAPR_AUTHORITY_PATH,
    PAPR_SELECTED_CHECKPOINT_PATH,
    load_papr_config,
)

AUTHORITY = REPO / PAPR_AUTHORITY_PATH
SELECTED = REPO / PAPR_SELECTED_CHECKPOINT_PATH


def verify_authority(path: Path = AUTHORITY) -> dict:
    value = read_json(path, "PAPR training authority")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    require(identifier == "paprtrainingauth-" + canonical_sha256(body), "PAPR authority ID differs")
    require(value.get("schema_version") == 1 and value.get("authority_kind") == "W10_PAPR_CONSTRAINED_TRAINING_AUTHORITY", "PAPR authority role differs")
    require(value.get("status") == "FROZEN_PRE_EXECUTION" and value.get("authorization_scope") == "PAPR_CONSTRAINED_TRAINING_ONLY", "PAPR authority scope/status differs")
    source = load_w10_manifest(REPO, live=True)
    require(value.get("source_manifest") == source_record(REPO, source) and value.get("source_binding") == source and value.get("source_commit") == source["source_commit"], "PAPR source binding differs")
    config = load_papr_config()
    require(value.get("config_path") == "configs/learned-papr-constrained-r1-6.yaml" and value.get("config_hash") == run_config_hash(config), "PAPR config binding differs")
    protocol = value.get("protocol")
    require(isinstance(protocol, dict), "PAPR protocol is missing")
    require(protocol.get("papr_cap_db") == float(get("evaluation.w10_papr_cap_db")) == 3.0, "PAPR cap differs")  # literal-ok: AM-98 frozen cap
    require(protocol.get("training_runs") == int(get("evaluation.w10_papr_training_runs")) == 1, "PAPR run count differs")  # literal-ok: AM-98 one run
    require(protocol.get("system") == "learned_papr_constrained" and protocol.get("bw_ratio") == "r_1_6", "PAPR system/ratio differs")
    require(protocol.get("fresh_initialization") is True and protocol.get("transfer_initialization_permitted") is False, "PAPR fresh-initialization boundary differs")
    require(protocol.get("additional_seeds_permitted") is False and protocol.get("outcome_conditioned_tuning_permitted") is False, "PAPR seed/tuning boundary differs")
    require(protocol.get("papr_domain") == "symbol_domain_not_oversampled_waveform", "PAPR domain differs")
    require(protocol.get("protected_counters") == {
        "papr_constrained_training": 0,
        "production_training": 0,
        "randomized_er2_training": 0,
        "er9_training": 0,
        "g10_adjudications": 0,
        "g11": 0,
        "w10": 0,
        "learned_test_inference": 0,
        "model_facing_test_access": 0,
    }, "PAPR protected counters differ")
    require(value.get("training_count") == 1 and value.get("w10_authorized") is False, "PAPR authorization boundary differs")
    require(value.get("test_authorized") is False and value.get("test") == "SEALED" and value.get("test_access") == 0, "PAPR test boundary differs")
    require(value.get("host") == "confessor" and value.get("device") == "cuda:0", "PAPR host/device differs")
    require(value.get("gpu_uuid") == "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b" and value.get("gpu_name") == "NVIDIA GeForce GTX 1080 Ti", "PAPR worker is not the frozen W8 GTX")  # literal-ok: frozen profile identity
    require(value.get("cuda_visible_devices") == value.get("gpu_uuid"), "PAPR CUDA binding differs")
    return value


def verify_selected(path: Path = SELECTED) -> dict:
    value = read_json(path, "PAPR selected checkpoint")
    body = dict(value)
    identifier = body.pop("selection_id", None)
    require(identifier == "paprselected-" + canonical_sha256(body), "PAPR selection ID differs")
    require(value.get("papr_cap_db") == 3.0 and value.get("train_seed") == 0 and value.get("channel_seed") == 0, "PAPR selected identity differs")  # literal-ok: AM-98 frozen cap/cell
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "PAPR selected checkpoint crossed test boundary")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args(argv)
    verify_authority()
    if args.terminal:
        verify_selected()
    print("PAPR constrained training verifier PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
