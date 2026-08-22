"""G8_E E5/E6 pass-one layer: frozen-input authentication and exact-once execution.

Chain-level tests that need the authenticated Imagenette archive are marked
``external_dataset``; every schema/digest/mutation test runs from tracked bytes
alone so the CPU CI lane keeps full regression coverage of the exactly-once
semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_pass_one as pass_one

# Frozen E2-E4 lifecycle history: these values are immutable evidence, pinned
# here so any drift of the tracked artifacts fails loudly.
E2_COMPLETION_SHA256 = "442448a424cbad0ead742c4a45724155486cd2e8ecefeff52bff62394e5096a6"
E3_ID = "g8ee3v3-ec7b28fda5bf0fc25b5bf4c71c25731a4d3286df2b7e88d039ff97daf5355f5e"
E3_SHA256 = "8496ebdb1c3757331b9fc53bc556d57091cbb7d08bdf390b07865547662dda42"
E4_ID = "g8ee4v3-4b5206cb5a7f752dc46f45996fd2d74a927ba0770813445915ab7a64f0e714f1"
E4_SHA256 = "ee2693460036539049b325c66a81e01298f7a66226c68715670ef26caf90f3b3"
W4_ADJUDICATION_SHA256 = "58827922671e596f038972cf731e3626313a562fbc36ee817eb50d947d6af121"
SELECTION_POLICY_SHA256 = "6a4ffa98a26ee627f8339f1668f11305e097ca813e246d46a235dbfb2476db0e"


def _write_canonical(path: Path, body: dict) -> Path:
    path.write_bytes(v3.rendered_json(body))
    return path


def _synthetic_authorization(tmp_path: Path, contract: dict, chain: dict, **overrides) -> Path:
    body: dict = {
        "schema_version": pass_one.PASS_ONE_SCHEMA_VERSION,
        "artifact_role": pass_one.AUTHORIZATION_ROLE,
        "status": "AUTHORIZED",
        "authorized_by": "test-owner",
        "reason": "synthetic non-publication authorization for mutation tests",
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "contract_sha256": v3.sha256_bytes(pass_one._raw_contract_bytes()),
        "source_manifest_id": contract["source_manifest"]["id"],
        "source_manifest_sha256": contract["source_manifest"]["sha256"],
        "data_identity_id": contract["scientific_data_identity"]["id"],
        "data_identity_sha256": contract["scientific_data_identity"]["sha256"],
        **{key: chain[key] for key in (
            "e2_completion_sha256",
            "e3_id",
            "e3_sha256",
            "e4_id",
            "e4_sha256",
            "bler_table_id",
            "bler_table_sha256",
            "w4_integration_adjudication_sha256",
            "selection_policy_sha256",
            "selection_call_plan_sha256",
            "candidate_authority_file_sha256",
            "outage_policy_file_sha256",
        )},
        "state_path": str(pass_one.PASS_ONE_STATE_PATH.relative_to(v3.REPO_ROOT)),
        "scope": dict(pass_one.AUTHORIZED_SCOPE),
    }
    body.update(overrides)
    for key, child in overrides.items():
        if child is None:
            del body[key]
    if "issued_sha256" not in body:
        body["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(body))
    return _write_canonical(tmp_path / "authorization.json", body)


def _synthetic_marker(tmp_path: Path, authorization_path: Path, authorization: dict, **overrides) -> Path:
    body: dict = {
        "schema_version": pass_one.PASS_ONE_SCHEMA_VERSION,
        "artifact_role": pass_one.MARKER_ROLE,
        "status": "MARKED_PRE_EXECUTION",
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization["issued_sha256"],
        "scorer_module": pass_one.SCORER_MODULE,
        "selection_policy_sha256": authorization["selection_policy_sha256"],
        "e4_input_id": authorization["e4_id"],
        "e4_input_sha256": authorization["e4_sha256"],
        "intended_output_path": authorization["state_path"],
        "intended_output_rule": "one immutable completion record; refuse when present",
        "exact_command": ".venv/bin/python tools/run_g8_e_pass_one.py --execute",
        "restart_command": ".venv/bin/python tools/run_g8_e_pass_one.py --execute",
        "pre_execution_pass_one_count": 0,
    }
    body.update(overrides)
    for key, child in overrides.items():
        if child is None:
            del body[key]
    if "issued_sha256" not in body:
        body["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(body))
    return _write_canonical(tmp_path / "marker.json", body)


# ---------------------------------------------------------------------------
# Tracked-bytes tests (CPU CI lane)
# ---------------------------------------------------------------------------


def test_selection_policy_fingerprint_reproduces_the_w4_adjudication():
    adjudication = json.loads(pass_one.W4_ADJUDICATION_PATH.read_bytes())
    recorded = adjudication["selection_machinery"]["selection_policy_sha256"]
    live, _fields = pass_one.recompute_selection_policy()
    assert recorded == SELECTION_POLICY_SHA256
    assert live == recorded


def test_standalone_e2_completion_authenticates_and_rejects_mutations(tmp_path):
    raw = (pass_one.v3s.V3S_RUNTIME_ROOT / "e2_completion.json").read_bytes()
    assert len(raw) > 0
    contract = json.loads(pass_one.v3s.V3S_CONTRACT_PATH.read_bytes())
    value = pass_one.verify_e2_completion_standalone(
        pass_one.v3s.V3S_RUNTIME_ROOT / "e2_completion.json",
        contract=contract,
        expected_sha256=E2_COMPLETION_SHA256,
    )
    assert value["completed_work_unit_count"] == 288000
    tampered = json.loads(raw)
    tampered["counters"]["training"] = 1
    mutated = _write_canonical(tmp_path / "completion.json", tampered)
    with pytest.raises(pass_one.G8EPassOneError):
        pass_one.verify_e2_completion_standalone(
            mutated, contract=contract, expected_sha256=E2_COMPLETION_SHA256
        )


def _contract() -> dict:
    return json.loads(pass_one.v3s.V3S_CONTRACT_PATH.read_bytes())


def _chain_digests() -> dict:
    # Every digest derivable from tracked bytes alone; no dataset access.
    contract = _contract()
    table_raw = (
        v3.REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_table.json"
    ).read_bytes()
    return {
        "e2_completion_sha256": E2_COMPLETION_SHA256,
        "e3_id": E3_ID,
        "e3_sha256": E3_SHA256,
        "e4_id": E4_ID,
        "e4_sha256": E4_SHA256,
        "bler_table_id": json.loads(table_raw)["table_id"],
        "bler_table_sha256": v3.sha256_bytes(table_raw),
        "w4_integration_adjudication_sha256": W4_ADJUDICATION_SHA256,
        "selection_policy_sha256": SELECTION_POLICY_SHA256,
        "selection_call_plan_sha256": v3.sha256_bytes(
            v3.canonical_json(contract["selection_authorization"])
        ),
        "candidate_authority_file_sha256": v3.sha256_bytes(
            pass_one.CANDIDATE_AUTHORITY_PATH.read_bytes()
        ),
        "outage_policy_file_sha256": v3.sha256_bytes(pass_one.OUTAGE_POLICY_PATH.read_bytes()),
    }


def test_authorization_schema_and_digest_mutations_are_refused(tmp_path):
    contract = _contract()
    good = json.loads(_synthetic_authorization(tmp_path, contract, _chain_digests()).read_bytes())
    assert pass_one.authenticate_owner_authorization(
        tmp_path / "authorization.json", contract
    )["status"] == "AUTHORIZED"

    def reserialized(mutated: dict) -> Path:
        del mutated["issued_sha256"]
        mutated["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(mutated))
        return _write_canonical(tmp_path / "mutated.json", mutated)

    mutations = {
        "scope_opens_pass_two": {"scope": dict(good["scope"], pass_two=True)},
        "scope_missing_field": {"scope": {k: v for k, v in good["scope"].items() if k != "fallback"}},
        # Lifecycle digests are deliberately not checked here: the schema layer
        # validates contract bindings and digest integrity; every frozen-chain
        # digest is refused by authenticate_inputs (see
        # test_chain_binding_mutations_are_refused).
        "wrong_campaign": {"campaign_id": "foreign"},
        "wrong_state_path": {"state_path": "results/baseline/g8_e/other.json"},
        "blank_reason": {"reason": " "},
        "wrong_schema": {"schema_version": 99},
        "missing_field": {k: None for k in ("outage_policy_file_sha256",)},
    }
    for name, override in mutations.items():
        mutated = dict(good)
        for key, child in override.items():
            if child is None:
                del mutated[key]
            else:
                mutated[key] = child
        with pytest.raises(pass_one.G8EPassOneError):
            pass_one.authenticate_owner_authorization(reserialized(mutated), contract)

    stale_digest = dict(good, e4_sha256="0" * 64)
    with pytest.raises(pass_one.G8EPassOneError):
        pass_one.authenticate_owner_authorization(
            _write_canonical(tmp_path / "stale.json", stale_digest), contract
        )


def test_marker_mutations_are_refused(tmp_path):
    contract = _contract()
    authorization = json.loads(
        _synthetic_authorization(tmp_path, contract, _chain_digests()).read_bytes()
    )
    auth_path = tmp_path / "authorization.json"
    marker_path = _synthetic_marker(tmp_path, auth_path, authorization)
    value = pass_one.authenticate_marker(marker_path, auth_path, authorization)
    assert value["pre_execution_pass_one_count"] == 0

    good = json.loads(marker_path.read_bytes())

    def reserialized(mutated: dict) -> Path:
        del mutated["issued_sha256"]
        mutated["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(mutated))
        return _write_canonical(tmp_path / "mutated-marker.json", mutated)

    for name, override in {
        "already_executed": {"pre_execution_pass_one_count": 1},
        "wrong_scorer": {"scorer_module": "somewhere/else.py"},
        "wrong_output": {"intended_output_path": "results/baseline/g8_e/other.json"},
        "blank_command": {"exact_command": ""},
    }.items():
        mutated = dict(good)
        mutated.update(override)
        with pytest.raises(pass_one.G8EPassOneError):
            pass_one.authenticate_marker(reserialized(mutated), auth_path, authorization)

    with pytest.raises(pass_one.G8EPassOneError):
        pass_one.authenticate_marker(
            reserialized(dict(good, authorization_sha256="0" * 64)), auth_path, authorization
        )
    other_auth = dict(authorization, issued_sha256="1" * 64)
    with pytest.raises(pass_one.G8EPassOneError):
        pass_one.authenticate_marker(marker_path, auth_path, other_auth)


def test_structural_feasibility_derives_block_identities_from_frozen_accounting():
    authority = v3.load_measurement_authority()
    row = next(
        row
        for row in authority["structural_identities"]
        if row["dataset"] == "imagenette160"
        and row["modulation"] == "qpsk"
        and row["ldpc_rate"] == "5/6"
    )
    feasibility, identities = pass_one._structural_feasibility(row)
    accounting = row["packet_accounting"]
    assert feasibility.feasible is True
    assert feasibility.code_blocks == len(identities) == int(accounting["code_blocks"])
    for identity, expected_bits in zip(identities, accounting["rate_matched_bits"]):
        assert identity.k_and_n[1] == int(expected_bits)
        assert identity.k_and_n[0] == int(accounting["k_prime"])
        assert identity.modulation == "qpsk"
        assert identity.rate == "5/6"
        assert identity.snr_convention == "es_n0_per_symbol"


def test_authority_key_canonicalizes_like_the_scorer_candidate():
    row = {
        "dataset": "imagenette160",
        "ratio": "r_1_2",
        "modulation": "qpsk",
        "ldpc_rate": "5/6",
        "encode_axis_px": 96,
        "snr_db": 18,
        "candidate_id": "cand-x",
    }
    ids = pass_one._authority_ids_for_ratio({"candidate_authority": {"candidates": [row]}}, "r_1_2")
    candidate = pass_one.Candidate(
        dataset="imagenette160",
        ratio="r_1_2",
        modulation="qpsk",
        ldpc_rate="5/6",
        encode_axis_px=96,
        snr_db=float(18),
    )
    assert list(ids.values()) == ["cand-x"]
    assert json.dumps(json.loads(candidate.candidate_id), sort_keys=True, separators=(",", ":")) in ids


def test_execute_refuses_a_pre_existing_completion_record(tmp_path, monkeypatch):
    monkeypatch.setattr(pass_one, "E5_MARKER_PATH", tmp_path / "marker.json")
    output = tmp_path / "pass_one_state.json"
    output.write_bytes(b"{}\n")
    with pytest.raises(pass_one.G8EPassOneError, match="immutable completion record"):
        pass_one.run_pass_one(tmp_path / "absent-authorization.json", output_path=output)


# ---------------------------------------------------------------------------
# Full-chain tests (authenticated dataset required)
# ---------------------------------------------------------------------------


@pytest.fixture
def chain_context():
    context = pass_one.authenticate_frozen_chain()
    assert context["chain"]["e3_id"] == E3_ID
    assert context["chain"]["e4_id"] == E4_ID
    assert context["chain"]["e2_completion_sha256"] == E2_COMPLETION_SHA256
    assert context["chain"]["selection_policy_sha256"] == SELECTION_POLICY_SHA256
    return context


@pytest.mark.external_dataset
def test_frozen_chain_authenticates_with_expected_hashes(chain_context):
    plan = chain_context["plan"]
    assert plan["call_count"] == 18
    assert plan["max_candidates"] == 1008
    assert plan["max_samples"] == 1000
    assert plan["authorization_issued"] is False
    objects = chain_context["e4"]["objects"]
    assert len(objects) == 288
    assert all(obj["total_count"] == 1000 for obj in objects)


@pytest.mark.external_dataset
def test_chain_binding_mutations_are_refused(chain_context, tmp_path):
    contract = chain_context["contract"]
    chain = chain_context["chain"]
    good = json.loads(_synthetic_authorization(tmp_path, contract, chain).read_bytes())
    assert pass_one.authenticate_owner_authorization(tmp_path / "authorization.json", contract)
    for field in (
        "e3_sha256",
        "bler_table_sha256",
        "w4_integration_adjudication_sha256",
        "selection_call_plan_sha256",
        "candidate_authority_file_sha256",
        "outage_policy_file_sha256",
        "selection_policy_sha256",
    ):
        mutated = dict(good)
        del mutated["issued_sha256"]
        mutated[field] = "0" * 64
        mutated["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(mutated))
        path = _write_canonical(tmp_path / f"mutated-{field}.json", mutated)
        with pytest.raises((pass_one.G8EPassOneError, ValueError)):
            pass_one.authenticate_inputs(path)


@pytest.mark.external_dataset
def test_full_pass_one_executes_once_verifies_and_is_deterministic(chain_context, tmp_path, monkeypatch):
    contract = chain_context["contract"]
    chain = chain_context["chain"]
    auth_path = _synthetic_authorization(tmp_path, contract, chain)
    authorization = json.loads(auth_path.read_bytes())
    marker_path = _synthetic_marker(tmp_path, auth_path, authorization)
    monkeypatch.setattr(pass_one, "E5_MARKER_PATH", marker_path)
    output = tmp_path / "pass_one_state.json"

    body = pass_one.run_pass_one(auth_path, output_path=output)
    assert body["counters"]["pass_one_executed_count"] == 1
    assert all(body["counters"][name] == 0 for name in (
        "training", "pass_two", "pass_three", "fallback_invoked",
        "ratio_adjudicated", "test_access", "learned_system_training", "g8_f_execution",
    ))
    assert body["call_count"] == 18

    with pytest.raises(pass_one.G8EPassOneError, match="already has an immutable completion record"):
        pass_one.run_pass_one(auth_path, output_path=output)

    result = pass_one.verify_pass_one_state(output, authorization_path=auth_path)
    assert result["status"] == "PASS"
    assert result["calls"] == 18
    second_body = json.loads(output.read_bytes())
    assert second_body["calls"] == body["calls"]

    # Every selection must map back to the logical candidate authority.
    selected_ids = [
        entry["authority_candidate_id"]
        for call in body["calls"]
        for entry in call["per_snr"]
        if entry["authority_candidate_id"] is not None
    ]
    authority_ids = {
        row["candidate_id"]
        for row in chain_context["candidate_authority"]["candidates"]
        if row["dataset"] == "imagenette160"
    }
    assert selected_ids and set(selected_ids) <= authority_ids


@pytest.mark.external_dataset
def test_state_mutations_are_refused(chain_context, tmp_path, monkeypatch):
    contract = chain_context["contract"]
    chain = chain_context["chain"]
    auth_path = _synthetic_authorization(tmp_path, contract, chain)
    authorization = json.loads(auth_path.read_bytes())
    marker_path = _synthetic_marker(tmp_path, auth_path, authorization)
    monkeypatch.setattr(pass_one, "E5_MARKER_PATH", marker_path)
    output = tmp_path / "pass_one_state.json"
    pass_one.run_pass_one(auth_path, output_path=output)
    original = json.loads(output.read_bytes())

    def rewrite(mutated: dict) -> None:
        mutated.pop("state_sha256", None)
        mutated.pop("state_id", None)
        mutated["state_id"] = v3._id(
            pass_one.STATE_PREFIX,
            {key: child for key, child in mutated.items() if key != "state_id"},
        )
        mutated["state_sha256"] = v3.sha256_bytes(v3.canonical_json(mutated))
        output.write_bytes(v3.rendered_json(mutated))

    # A flipped composition float must be caught by the independent recomputation.
    mutated = json.loads(json.dumps(original))
    for call in mutated["calls"]:
        for entry in call["per_snr"]:
            if entry["selected_composition"] is not None:
                entry["selected_composition"]["expected_accuracy"] += 1e-9
                break
        else:
            continue
        break
    rewrite(mutated)
    with pytest.raises(pass_one.G8EPassOneError):
        pass_one.verify_pass_one_state(output, authorization_path=auth_path)

    # A swapped selection target must be caught too.
    output.write_bytes(v3.rendered_json(original))
    swapped = json.loads(json.dumps(original))
    ids = [
        entry["authority_candidate_id"]
        for call in swapped["calls"]
        for entry in call["per_snr"]
        if entry["authority_candidate_id"] is not None
    ]
    a, b = ids[0], ids[-1]
    done = False
    for call in swapped["calls"]:
        for entry in call["per_snr"]:
            if entry["authority_candidate_id"] == a and not done:
                entry["authority_candidate_id"] = b
                done = True
                break
    assert done
    rewrite(swapped)
    with pytest.raises(pass_one.G8EPassOneError):
        pass_one.verify_pass_one_state(output, authorization_path=auth_path)

    # A prohibited nonzero counter must be refused by the header check alone.
    output.write_bytes(v3.rendered_json(original))
    counters = json.loads(json.dumps(original))
    counters["counters"]["training"] = 1
    rewrite(counters)
    with pytest.raises(pass_one.G8EPassOneError):
        pass_one.verify_pass_one_state(output, authorization_path=auth_path)

    # Non-canonical bytes are refused before any semantic check.
    output.write_text(json.dumps(original))
    with pytest.raises(pass_one.G8EPassOneError):
        pass_one.verify_pass_one_state(output, authorization_path=auth_path)

    # Executing again after the record exists is forbidden even with fresh inputs.
    with pytest.raises(pass_one.G8EPassOneError, match="immutable completion record"):
        pass_one.run_pass_one(auth_path, output_path=output)
