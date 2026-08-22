"""G8_E E7 closeout verification: live gate outcome and semantic mutations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_pass_one as pass_one

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.external_dataset
def test_live_g8_e_complete_verifier_returns_green():
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools/verify_g8_e_complete.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["status"] == "PASS"
    assert report["counters"]["pass_one_executed_count"] == 1
    assert all(
        report["counters"][name] == 0
        for name in (
            "training", "pass_two", "pass_three", "fallback_invoked",
            "ratio_adjudicated", "test_access", "learned_system_training",
            "g8_f_execution",
        )
    )
    assert report["verdict"].startswith("G8_E GREEN")
    assert report["pass_one_cells_without_selection"] >= 0


def test_e4_semantics_mutations_are_refused():
    e4 = json.loads(pass_one.v3s.V3S_E4_PATH.read_bytes())
    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location("verify_g8_e_complete", REPO / "tools/verify_g8_e_complete.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    module._verify_e4_semantics(e4)

    def mutated(change) -> dict:
        value = json.loads(json.dumps(e4))
        change(value)
        return value

    with pytest.raises(SystemExit):
        module._verify_e4_semantics(mutated(lambda v: v["objects"][0].__setitem__("total_count", 999)))
    with pytest.raises(SystemExit):
        module._verify_e4_semantics(mutated(lambda v: v["objects"][0].__setitem__("status", "foreign")))
    with pytest.raises(SystemExit):
        module._verify_e4_semantics(
            mutated(lambda v: v["objects"][0].__setitem__(
                "delivered_count", v["objects"][0]["delivered_count"] - 1))
        )
    with pytest.raises(SystemExit):
        module._verify_e4_semantics(mutated(lambda v: v.__setitem__("object_count", 287)))
    with pytest.raises(SystemExit):
        module._verify_e4_semantics(
            mutated(lambda v: v.__setitem__("outage_accuracy", dict(v["outage_accuracy"], numerator=99)))
        )


def test_corpus_training_only_validation_refuses_mutations(tmp_path, monkeypatch):
    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location("verify_g8_e_complete", REPO / "tools/verify_g8_e_complete.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus_raw = module.CORPUS_SPEC_PATH.read_bytes()
    binding = {
        "sha256": v3.sha256_bytes(corpus_raw),
        "training_only": True,
        "materialized": False,
        "resolved_lineage": {"state_sha256": "0" * 64},
    }
    context = {"contract": json.loads(pass_one.v3s.V3S_CONTRACT_PATH.read_bytes())}
    # The real frozen corpus specification passes its own training-only audit.
    module.validate_corpus_training_only(context, binding)

    def with_tmp_corpus(mutated: dict):
        path = tmp_path / "corpus_spec.json"
        path.write_bytes(v3.rendered_json(mutated))
        monkeypatch.setattr(module, "CORPUS_SPEC_PATH", path)

    good = json.loads(corpus_raw)
    with pytest.raises(SystemExit):
        with_tmp_corpus(dict(good, materialized_object_count=5))
        module.validate_corpus_training_only(context, binding)
    with pytest.raises(SystemExit):
        with_tmp_corpus(dict(good, generation_rules=dict(good["generation_rules"], output_split="val")))
        module.validate_corpus_training_only(context, binding)
