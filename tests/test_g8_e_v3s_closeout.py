"""Corrected worker-successor closeout layer: identity loading and equivalence."""

from __future__ import annotations

from pathlib import Path

import pytest

from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_corrected_v3s as v3s
from baseline import g8_e_v3s_closeout as closeout

import aggregate_g8_e_corrected_v3s as aggregate_cli  # noqa: E402
import merge_g8_e_corrected_v3s as merge_cli  # noqa: E402
import run_g8_e_corrected_v3s as runner_cli  # noqa: E402
from prove_g8_e_v3s_lifecycle import (  # noqa: E402
    _authorization,
    _br11_stub,
    fixture_context,
)


def _completed_fixture_runtime(tmp_path: Path):
    fixture = fixture_context()
    authorization = tmp_path / "synthetic-owner-authorization.json"
    _authorization(authorization, fixture)
    runtime = tmp_path / "runtime"
    common = [
        "--campaign-id", fixture["contract"]["campaign_id"],
        "--runtime-root", str(runtime),
        "--authorization", str(authorization),
    ]
    assert runner_cli.main(["--start", *common], fixture=dict(fixture)) == 0
    v3.publish_e2_completion(
        runtime_root=runtime,
        contract=fixture["contract"],
        authority=fixture["authority"],
        production=False,
    )
    return fixture, authorization, runtime


def _context_kwargs(fixture, authorization: Path) -> dict:
    return {
        "contract": fixture["contract"],
        "authority": fixture["authority"],
        "data_identity": fixture["data_identity"],
        "sample_ids": fixture["sample_ids"],
        "sample_labels": fixture["sample_labels"],
        "authorization_path": authorization,
    }


def test_load_bound_data_identity_authenticates_the_production_contract():
    """The exact production load now succeeds from tracked bytes alone."""

    if not v3s.V3S_CONTRACT_PATH.is_file():
        pytest.skip("worker-successor contract is frozen on the worker host")
    contract, _ = v3._rendered_object(v3s.V3S_CONTRACT_PATH, "v3s measurement contract")
    identity = closeout.load_bound_data_identity(contract)
    assert identity["data_identity_id"] == contract["scientific_data_identity"]["id"]
    sample_ids, labels = v3.frozen_validation_metadata(identity)
    assert len(sample_ids) == contract["scientific_data_identity"]["validation_count"]
    assert set(labels) == set(sample_ids)


@pytest.mark.external_dataset
def test_summary_block_is_rejected_against_the_live_rebuild_before_payload():
    """The exact operation that failed in production must still refuse.

    The live rebuild authenticates archive bytes before any comparison, so
    this regression needs the authenticated Imagenette archive and runs on
    hosts that hold it.
    """

    if not v3s.V3S_CONTRACT_PATH.is_file():
        pytest.skip("worker-successor contract is frozen on the worker host")
    contract, _ = v3._rendered_object(v3s.V3S_CONTRACT_PATH, "v3s measurement contract")
    with pytest.raises(ValueError):
        v3.verify_live_validation_identity(contract["scientific_data_identity"])


def test_load_bound_data_identity_rejects_a_substituted_contract_block():
    if not v3s.V3S_CONTRACT_PATH.is_file():
        pytest.skip("worker-successor contract is frozen on the worker host")
    contract, _ = v3._rendered_object(v3s.V3S_CONTRACT_PATH, "v3s measurement contract")
    tampered = {key: child for key, child in contract.items()}
    tampered["scientific_data_identity"] = dict(contract["scientific_data_identity"], id="f" * 64)
    with pytest.raises(ValueError):
        closeout.load_bound_data_identity(tampered)


def test_active_context_refuses_sample_injection_without_a_contract():
    with pytest.raises(ValueError):
        closeout.active_context(
            runtime_root=v3s.V3S_RUNTIME_ROOT,
            authorization_path=v3s.V3S_AUTHORIZATION_PATH,
            sample_ids=("x",),
            sample_labels={"x": 0},
        )


def test_synthetic_closeout_reproduces_bound_cli_outputs_byte_for_byte(tmp_path, monkeypatch):
    import baseline.g8_d as g8d

    monkeypatch.setattr(g8d, "account_br11", _br11_stub)
    fixture, authorization, runtime = _completed_fixture_runtime(tmp_path)
    kwargs = _context_kwargs(fixture, authorization)

    assert closeout.verify_e2_complete(**kwargs, runtime_root=runtime)["phase"] == "E2_COMPLETE"

    _, _, corrected_e3_sha = closeout.publish_e3(runtime_root=runtime, **kwargs)
    e3 = runtime / "e3_exact_set_closure.json"
    e3.unlink()
    merge_fixture = {
        "contract": fixture["contract"],
        "authority": fixture["authority"],
        "sample_ids": fixture["sample_ids"],
        "sample_labels": fixture["sample_labels"],
    }
    assert merge_cli.main(["--execute", "--runtime-root", str(runtime)], fixture=merge_fixture) == 0
    assert v3.sha256_file(e3) == corrected_e3_sha

    _, _, corrected_e4_sha = closeout.publish_e4(
        runtime_root=runtime,
        e3_path=e3,
        e3_sha256=corrected_e3_sha,
        **kwargs,
    )
    e4 = runtime / "e4_count_derived.json"
    e4.unlink()
    assert aggregate_cli.main(
        ["--execute", "--runtime-root", str(runtime), "--e3", str(e3), "--e3-sha256", corrected_e3_sha],
        fixture=merge_fixture,
    ) == 0
    assert v3.sha256_file(e4) == corrected_e4_sha

    result = closeout.verify_e4_complete(
        e4_path=e4,
        e3_path=e3,
        e3_sha256=corrected_e3_sha,
        **kwargs,
    )
    assert result["phase"] == "E4_COMPLETE"
    assert result["completion"]["status"] == "E2_COMPLETE"
    assert result["e3"]["observed_work_unit_count"] == len(fixture["sample_ids"]) * 3
    eligible = [obj for obj in result["e4"]["objects"] if obj["status"] == "eligible"]
    assert eligible
    for obj in eligible:
        assert obj["total_count"] == len(fixture["sample_ids"])
        assert obj["codec_infeasibility_count"] >= 0


def test_verify_e2_complete_detects_record_tampering(tmp_path, monkeypatch):
    import baseline.g8_d as g8d

    monkeypatch.setattr(g8d, "account_br11", _br11_stub)
    fixture, authorization, runtime = _completed_fixture_runtime(tmp_path)
    record = sorted((runtime / "records").iterdir())[0]
    record.write_bytes(record.read_bytes() + b" ")
    with pytest.raises(ValueError):
        closeout.verify_e2_complete(**_context_kwargs(fixture, authorization), runtime_root=runtime)


def test_verify_e3_complete_enforces_the_exact_supplied_sha(tmp_path, monkeypatch):
    import baseline.g8_d as g8d

    monkeypatch.setattr(g8d, "account_br11", _br11_stub)
    fixture, authorization, runtime = _completed_fixture_runtime(tmp_path)
    _, _, e3_sha = closeout.publish_e3(runtime_root=runtime, **_context_kwargs(fixture, authorization))
    kwargs = _context_kwargs(fixture, authorization)
    fixture_e3 = runtime / "e3_exact_set_closure.json"
    with pytest.raises(ValueError):
        closeout.verify_e3_complete(e3_path=fixture_e3, e3_sha256="0" * 64, **kwargs)
    result = closeout.verify_e3_complete(e3_path=fixture_e3, e3_sha256=e3_sha, **kwargs)
    assert result["e3"]["missing_count"] == 0
    assert result["e3"]["required_work_unit_count"] == result["e3"]["observed_work_unit_count"]
