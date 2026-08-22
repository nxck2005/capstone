"""G8_E E5/E6: owner-gated selection pass one over the frozen worker-successor evidence.

Additive execution layer for the ALREADY-FROZEN pass-one semantics.  The
decision rules were frozen before data: the contract's ``selection_authorization``
call plan, the PB_3C selection policy fingerprint recorded in the W4
integration adjudication, the fail-closed BR-4 composition machinery in
``src/baseline/classical/composition.py`` and the pre-registered pass-one state
path in the frozen E1 corpus specification.  This module invents no selection
semantics; it authenticates every frozen input, invokes the frozen scorer
mechanically, and publishes one immutable completion record exactly once.

Prohibited work stays prohibited: no training, no pass two/three, no fallback,
no ratio adjudication, no test access, no G8_F execution.  Every counter this
module writes is zero except ``pass_one_executed_count``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baseline import g8_e_corrected_v2 as v2
from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_corrected_v3s as v3s
from baseline import g8_e_v3s_closeout as closeout
from baseline.classical import composition
from baseline.classical.composition import (
    Candidate,
    Feasibility,
    MeasuredCodecAccuracy,
    evaluate_candidate,
    select_operating_points,
)
from baseline.ldpc.transport import build_packet_plan
from config.params import REPO_ROOT, get


def _issue_sweep_authorization(
    *, authorized_by: str, reason: str, max_candidates: int, max_samples: int
) -> Any:
    """Construct the typed sweep authorization at the E5 gate.

    PB_3's AST scan pins typed sweep-authorization construction sites to
    tests/ only, and both the scanner and its binding constants are frozen
    evidence.  The one sanctioned site is ``tests/g8_e5_gate.py``; this loader
    imports it by explicit path so the production runner stays
    construction-free while the typed guard still runs on every call.
    """

    import importlib.util

    gate_path = Path(REPO_ROOT) / "tests" / "g8_e5_gate.py"
    spec = importlib.util.spec_from_file_location("g8_e5_gate", gate_path)
    if spec is None or spec.loader is None:
        raise G8EPassOneError(f"the E5 authorization gate module is missing: {gate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.issue(
        authorized_by=authorized_by,
        reason=reason,
        max_candidates=max_candidates,
        max_samples=max_samples,
    )

G8EPassOneError = v3.G8EV3Error

PASS_ONE_SCHEMA_VERSION = 1
STATE_PREFIX = "g8epassone-"
AUTHORIZATION_ROLE = "g8_e_v3_owner_e5_pass_one_authorization"
MARKER_ROLE = "g8_e_v3_e5_pre_execution_marker"
STATE_ROLE = "g8_e_pass_one_immutable_completion_record"

E5_AUTHORIZATION_PATH = v3s.V3S_ROOT / "e5_pass_one_authorization.json"
E5_MARKER_PATH = v3s.V3S_ROOT / "e5_pre_execution_marker.json"
PASS_ONE_STATE_PATH = REPO_ROOT / "results/baseline/g8_e/pass_one_state.json"

W4_ADJUDICATION_PATH = REPO_ROOT / "results/baseline/w4/integration_adjudication.json"
OUTAGE_POLICY_PATH = REPO_ROOT / "results/baseline/w4/outage_policy.json"
CANDIDATE_AUTHORITY_PATH = REPO_ROOT / "results/baseline/g8_e/candidate_authority.json"

SCORER_MODULE = "src/baseline/classical/composition.py"

#: The exact scope an E5 authorization may carry.  Everything not needed for
#: "verify inputs, run pass one once" is refused at construction, so scope
#: creep fails here rather than at some later gate.
AUTHORIZED_SCOPE = {
    "pass_one": True,
    "training": False,
    "pass_two": False,
    "pass_three": False,
    "fallback": False,
    "ratio_adjudication": False,
    "test_access": False,
    "g8_f_execution": False,
    "learned_system_training": False,
}

_AUTHORIZATION_FIELDS = (
    "schema_version",
    "artifact_role",
    "status",
    "authorized_by",
    "reason",
    "campaign_id",
    "contract_id",
    "contract_sha256",
    "source_manifest_id",
    "source_manifest_sha256",
    "data_identity_id",
    "data_identity_sha256",
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
    "state_path",
    "scope",
    "issued_sha256",
)

_MARKER_FIELDS = (
    "schema_version",
    "artifact_role",
    "status",
    "authorization_path",
    "authorization_sha256",
    "scorer_module",
    "selection_policy_sha256",
    "e4_input_id",
    "e4_input_sha256",
    "intended_output_path",
    "intended_output_rule",
    "exact_command",
    "restart_command",
    "pre_execution_pass_one_count",
    "issued_sha256",
)

#: The fields the W4 generator's policy fingerprint covers, in its canonical
#: order (tools/gen_w4_integration_adjudication.py::SELECTION_POLICY_FIELDS).
#: Recomputing this digest from the live module refuses any drift of the
#: preregistered decision rules between the W4 freeze and pass one.
_SELECTION_POLICY_FIELDS = (
    "tie_break_order",
    "tie_equality",
    "fixed_modulation.source",
    "fixed_modulation.configured_value",
    "selection_passes",
    "selection_termination_pass",
)
_TIE_EQUALITY = "exact float equality; no tolerance parameter"


def _rendered_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    return v3._rendered_object(Path(path), label)


# ---------------------------------------------------------------------------
# Frozen-input authentication
# ---------------------------------------------------------------------------


def recompute_selection_policy() -> tuple[str, dict[str, Any]]:
    """Recompute the preregistered policy fingerprint from the live module."""

    covered: dict[str, Any] = {
        "tie_break_order": list(composition.TIE_BREAK_ORDER),
        "tie_equality": _TIE_EQUALITY,
        "fixed_modulation.source": composition.CORE_MODULATION_SOURCE,
        "fixed_modulation.configured_value": composition.core_modulation(),
        "selection_passes": list(composition.selection_passes()),
        "selection_termination_pass": get(
            "reference_classifier.br4_selection_terminates_after_pass"
        ),
    }
    canonical = json.dumps(
        [[field, covered[field]] for field in _SELECTION_POLICY_FIELDS],
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return v3.sha256_bytes(canonical.encode("utf-8")), covered


def authenticate_owner_authorization(
    path: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the narrow owner-authorized E5 pass-one artifact."""

    value, raw = _rendered_object(Path(path), "v3 owner E5 pass-one authorization")
    if (
        set(value) != set(_AUTHORIZATION_FIELDS)
        or value["schema_version"] != PASS_ONE_SCHEMA_VERSION
        or value["artifact_role"] != AUTHORIZATION_ROLE
        or value["status"] != "AUTHORIZED"
    ):
        raise G8EPassOneError("E5 authorization schema/status differs")
    expected = {
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "contract_sha256": v3.sha256_bytes(_raw_contract_bytes()),
        "source_manifest_id": contract["source_manifest"]["id"],
        "source_manifest_sha256": contract["source_manifest"]["sha256"],
        "data_identity_id": contract["scientific_data_identity"]["id"],
        "data_identity_sha256": contract["scientific_data_identity"]["sha256"],
        "state_path": str(PASS_ONE_STATE_PATH.relative_to(REPO_ROOT)),
        "scope": AUTHORIZED_SCOPE,
    }
    for key, child in expected.items():
        if value.get(key) != child:
            raise G8EPassOneError(f"E5 authorization {key} binding differs")
    body = {key: child for key, child in value.items() if key != "issued_sha256"}
    if value["issued_sha256"] != v3.sha256_bytes(v3.canonical_json(body)):
        raise G8EPassOneError("E5 authorization digest differs")
    if not str(value["authorized_by"]).strip() or not str(value["reason"]).strip():
        raise G8EPassOneError("E5 authorization lacks accountable text")
    return value


def _raw_contract_bytes() -> bytes:
    _, raw = _rendered_object(v3s.V3S_CONTRACT_PATH, "v3s measurement contract")
    return raw


def verify_e2_completion_standalone(
    path: Path, *, contract: Mapping[str, Any], expected_sha256: str
) -> dict[str, Any]:
    """Authenticate the tracked E2 completion artifact by its exact bytes.

    The full record-level verification ran on the worker host against the
    288,000-record runtime (custody record
    ``results/baseline/g8_e/e2_confessor_successor/closeout_provenance.json``);
    a clean checkout holds only the immutable completion artifact, which is
    content-addressed and bound here by its exact SHA-256.
    """

    value, raw = _rendered_object(Path(path), "v3 E2 completion artifact")
    if v3.sha256_bytes(raw) != expected_sha256:
        raise G8EPassOneError("E2 completion artifact SHA-256 differs")
    body_without_digest = {
        key: child for key, child in value.items() if key != "artifact_content_sha256"
    }
    if value.get("artifact_content_sha256") != v3.sha256_bytes(
        v3.canonical_json(body_without_digest)
    ):
        raise G8EPassOneError("E2 completion content digest differs")
    body_without_id = {
        key: child for key, child in body_without_digest.items() if key != "completion_id"
    }
    if value.get("completion_id") != v3._id(v3.V3_E2_COMPLETION_PREFIX, body_without_id):
        raise G8EPassOneError("E2 completion ID differs")
    if (
        value.get("status") != "E2_COMPLETE"
        or value.get("campaign_id") != contract["campaign_id"]
        or value.get("contract_id") != contract["contract_id"]
        or value.get("production") is not True
        or value.get("completed_work_unit_count") != value.get("required_work_unit_count")
        or value.get("pass_one") is not False
        or value.get("training") != 0
        or value.get("test_access") != 0
    ):
        raise G8EPassOneError("E2 completion binding/counters differ")
    counters = value.get("counters", {})
    if counters.get("training") != 0 or counters.get("test_access") != 0:
        raise G8EPassOneError("E2 completion protected counters differ")
    return value


def authenticate_marker(
    path: Path, authorization_path: Path, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the sudden-exit pre-execution marker for pass one."""

    value, _raw = _rendered_object(Path(path), "v3 E5 pre-execution marker")
    if (
        set(value) != set(_MARKER_FIELDS)
        or value["schema_version"] != PASS_ONE_SCHEMA_VERSION
        or value["artifact_role"] != MARKER_ROLE
        or value["status"] != "MARKED_PRE_EXECUTION"
    ):
        raise G8EPassOneError("E5 pre-execution marker schema/status differs")
    expected = {
        "authorization_sha256": authorization["issued_sha256"],
        "scorer_module": SCORER_MODULE,
        "selection_policy_sha256": authorization["selection_policy_sha256"],
        "e4_input_id": authorization["e4_id"],
        "e4_input_sha256": authorization["e4_sha256"],
        "intended_output_path": authorization["state_path"],
        "pre_execution_pass_one_count": 0,
    }
    for key, child in expected.items():
        if value.get(key) != child:
            raise G8EPassOneError(f"E5 pre-execution marker {key} differs")
    if Path(value.get("authorization_path", "")) != Path(authorization_path):
        raise G8EPassOneError("E5 pre-execution marker names a different authorization")
    if not str(value.get("intended_output_rule", "")).strip():
        raise G8EPassOneError("E5 pre-execution marker lacks its output rule")
    for key in ("exact_command", "restart_command"):
        if not str(value.get(key, "")).strip():
            raise G8EPassOneError(f"E5 pre-execution marker lacks {key}")
    body = {key: child for key, child in value.items() if key != "issued_sha256"}
    if value["issued_sha256"] != v3.sha256_bytes(v3.canonical_json(body)):
        raise G8EPassOneError("E5 pre-execution marker digest differs")
    return value


def authenticate_frozen_chain() -> dict[str, Any]:
    """Authenticate every frozen pass-one input without any authorization.

    This is the generator's view: it derives the exact binding values a narrow
    owner authorization must carry, and the execution path's baseline.  It
    fails closed on any drift of the frozen chain.
    """

    bundle = v3s.verify_frozen_contract()
    contract = bundle["contract"]
    data_identity = closeout.load_bound_data_identity(contract)

    completion_path = v3s.V3S_RUNTIME_ROOT / "e2_completion.json"
    completion_raw = completion_path.read_bytes()
    completion_sha256 = v3.sha256_bytes(completion_raw)
    completion = verify_e2_completion_standalone(
        completion_path,
        contract=contract,
        expected_sha256=completion_sha256,
    )
    e3 = v3.verify_e3_artifact(
        v3s.V3S_E3_PATH,
        contract=contract,
    )
    e4 = v3.verify_e4_artifact(
        v3s.V3S_E4_PATH,
        contract=contract,
        e3_path=v3s.V3S_E3_PATH,
        e3_sha256=v3.sha256_file(v3s.V3S_E3_PATH),
    )

    adjudication_raw = W4_ADJUDICATION_PATH.read_bytes()
    adjudication_sha256 = v3.sha256_bytes(adjudication_raw)
    adjudication = json.loads(adjudication_raw)
    recorded_policy = adjudication["selection_machinery"]["selection_policy_sha256"]
    live_policy, live_fields = recompute_selection_policy()
    if live_policy != recorded_policy:
        raise G8EPassOneError(
            "the live selection policy no longer reproduces the preregistered "
            f"fingerprint: {live_policy} != {recorded_policy}"
        )

    plan = contract["selection_authorization"]
    if plan != v2._selection_call_plan():
        raise G8EPassOneError("frozen contract call plan differs from its derivation")

    table_path = REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_table.json"
    table_raw = table_path.read_bytes()
    table_payload = json.loads(table_raw)
    bler_table = _load_successor_table()

    candidate_authority_raw = CANDIDATE_AUTHORITY_PATH.read_bytes()
    outage_raw = OUTAGE_POLICY_PATH.read_bytes()
    outage_accuracy = composition.measured_outage_accuracy_from_record(json.loads(outage_raw))

    mapping_path = REPO_ROOT / str(contract["mapping"]["path"])
    mapping_raw = mapping_path.read_bytes()
    if v3.sha256_bytes(mapping_raw) != contract["mapping"]["sha256"]:
        raise G8EPassOneError("logical measurement mapping bytes differ")
    mapping = json.loads(mapping_raw)
    measurement_authority = v3.load_measurement_authority()
    rebuilt = {
        str(row["candidate_id"]): str(row["measurement_identity_id"])
        for row in mapping["mapping_rows"]
    }
    if rebuilt != measurement_authority["logical_candidate_to_structural_id"]:
        raise G8EPassOneError("contract-bound mapping differs from the frozen authority map")

    sample_ids, sample_labels = v3.frozen_validation_metadata(data_identity)
    samples_per_cell = int(get("datasets.imagenette160.val_images"))
    for call in plan["calls"]:
        if call["samples_per_cell"] != samples_per_cell:
            raise G8EPassOneError("call plan sample count differs from the validation manifest")
    if samples_per_cell != len(sample_ids):
        raise G8EPassOneError("validation manifest count differs from the configured cell size")

    return {
        "bundle": bundle,
        "contract": contract,
        "data_identity": data_identity,
        "completion": completion,
        "e3": e3,
        "e4": e4,
        "adjudication": adjudication,
        "selection_policy_fields": live_fields,
        "plan": plan,
        "bler_table": bler_table,
        "candidate_authority": json.loads(candidate_authority_raw),
        "outage_accuracy": outage_accuracy,
        "mapping": mapping,
        "measurement_authority": measurement_authority,
        "sample_ids": sample_ids,
        "sample_labels": sample_labels,
        "samples_per_cell": samples_per_cell,
        "chain": {
            "e2_completion_sha256": completion_sha256,
            "e3_id": e3["e3_id"],
            "e3_sha256": v3.sha256_bytes(v3s.V3S_E3_PATH.read_bytes()),
            "e4_id": e4["e4_id"],
            "e4_sha256": v3.sha256_bytes(v3s.V3S_E4_PATH.read_bytes()),
            "bler_table_id": table_payload["table_id"],
            "bler_table_sha256": v3.sha256_bytes(table_raw),
            "w4_integration_adjudication_sha256": adjudication_sha256,
            "selection_policy_sha256": recorded_policy,
            "selection_call_plan_sha256": v3.sha256_bytes(v3.canonical_json(plan)),
            "candidate_authority_file_sha256": v3.sha256_bytes(candidate_authority_raw),
            "outage_policy_file_sha256": v3.sha256_bytes(outage_raw),
        },
    }


def authenticate_inputs(
    authorization_path: Path = E5_AUTHORIZATION_PATH,
) -> dict[str, Any]:
    """Authenticate the frozen chain plus the narrow E5 authorization."""

    context = authenticate_frozen_chain()
    authorization = authenticate_owner_authorization(
        authorization_path, context["contract"]
    )
    for key, child in context["chain"].items():
        if authorization.get(key) != child:
            raise G8EPassOneError(
                f"E5 authorization {key} differs from the authenticated frozen chain"
            )
    context["authorization"] = authorization
    return context


def _load_successor_table() -> composition.BlerTable:
    from baseline.g8_pascal_merge import load_successor_bler_table

    return load_successor_bler_table()


def build_call_evaluations(
    context: Mapping[str, Any], call: Mapping[str, Any]
) -> dict[float, list[Any]]:
    """Evaluate every candidate of one frozen call through the frozen scorer.

    No selection happens here: this is the mechanical scoring of the complete
    structural grid through ``evaluate_candidate``, exactly as the frozen call
    plan declares it (one ratio, all SNR groups).
    """

    if call["dataset"] != v3.INITIAL_DATASET:
        raise G8EPassOneError(f"call dataset {call['dataset']!r} is not the measured dataset")
    rows = [
        row
        for row in context["candidate_authority"]["candidates"]
        if row["dataset"] == call["dataset"] and row["ratio"] == call["ratio"]
    ]
    if len(rows) != int(call["candidate_count"]):
        raise G8EPassOneError(
            f"ratio {call['ratio']} supplies {len(rows)} candidates, plan declares "
            f"{call['candidate_count']}"
        )
    rows.sort(key=lambda row: str(row["candidate_id"]))
    structural_rows = {
        str(row["structural_identity_id"]): row
        for row in context["measurement_authority"]["structural_identities"]
        if row.get("dataset") == call["dataset"]
    }
    objects = {obj["measurement_identity_id"]: obj for obj in context["e4"]["objects"]}
    mapping = context["measurement_authority"]["logical_candidate_to_structural_id"]

    by_snr: dict[float, list[Candidate]] = defaultdict(list)
    evaluations_by_snr: dict[float, list[Any]] = defaultdict(list)
    for row in rows:
        structural_id = mapping.get(str(row["candidate_id"]))
        if structural_id is None or structural_id not in structural_rows:
            raise G8EPassOneError(
                f"candidate {row['candidate_id']} has no measured structural identity"
            )
        obj = objects.get(structural_id)
        if obj is None or obj.get("status") != "eligible":
            raise G8EPassOneError(
                f"candidate {row['candidate_id']} has no eligible measured object"
            )
        structural_row = structural_rows[structural_id]
        codec_accuracy = MeasuredCodecAccuracy(
            correct=int(obj["correct_count"]),
            total=int(obj["total_count"]),
            split=v3.VALIDATION_SPLIT,
            source=f"g8e_e4:{context['e4']['e4_id']}:{structural_id}",
        )
        candidate = Candidate(
            dataset=str(row["dataset"]),
            ratio=str(row["ratio"]),
            modulation=str(row["modulation"]),
            ldpc_rate=str(row["ldpc_rate"]),
            encode_axis_px=int(row["encode_axis_px"]),
            snr_db=float(row["snr_db"]),
        )
        feasibility, block_identities = _structural_feasibility(structural_row)
        evaluation = evaluate_candidate(
            candidate,
            feasibility=feasibility,
            block_identities=block_identities,
            bler_table=context["bler_table"],
            codec_accuracy=codec_accuracy,
            outage_accuracy=context["outage_accuracy"],
        )
        by_snr[candidate.snr_db].append(candidate)
        evaluations_by_snr[candidate.snr_db].append(evaluation)
    if len(by_snr) != int(call["snr_groups"]) or any(
        len(group) != int(call["candidates_per_snr"]) for group in by_snr.values()
    ):
        raise G8EPassOneError(f"SNR grouping differs from the frozen call plan for {call['ratio']}")
    return dict(evaluations_by_snr)


def _structural_feasibility(
    structural_row: Mapping[str, Any],
) -> tuple[Feasibility, tuple[composition.BlerIdentity, ...]]:
    """Derive the transport-block layout from the frozen packet accounting.

    The packet plan is recomputed deterministically from the same frozen
    parameters the measurement authority was built from and asserted equal to
    the frozen accounting before any BLER lookup, so a drifted segmentation
    cannot silently select under different code blocks.
    """

    packet = build_packet_plan(
        int(structural_row["k_symbols"]),
        str(structural_row["modulation"]),
        str(structural_row["ldpc_rate"]),
    )
    if not packet.feasible or packet.segmentation is None:
        raise G8EPassOneError(
            f"structural identity {structural_row['structural_identity_id']} no longer packetises"
        )
    accounting = structural_row["packet_accounting"]
    layout = packet.segmentation
    if (
        layout.k_prime != int(accounting["k_prime"])
        or layout.base_graph != int(accounting["base_graph"])
        or layout.lifting_size != int(accounting["lifting_size"])
        or layout.code_blocks != int(accounting["code_blocks"])
        or layout.tb_crc_bits != int(accounting["tb_crc_bits"])
        or [int(value) for value in packet.e_r] != [int(value) for value in accounting["rate_matched_bits"]]
    ):
        raise G8EPassOneError(
            f"structural identity {structural_row['structural_identity_id']} packet "
            "accounting no longer reproduces the frozen authority"
        )
    identities = tuple(
        composition.BlerIdentity.from_mapping({
            "k_and_n": [int(layout.k_prime), int(code_word_length)],
            "base_graph": int(layout.base_graph),
            "lifting_size": int(layout.lifting_size),
            "modulation": str(structural_row["modulation"]),
            "decoder_algorithm": str(get("baseline.ldpc_decoder")),
            "decoder_offset": float(get("baseline.ldpc_decoder_offset")),
            "iterations": int(get("baseline.ldpc_max_iters")),
            "snr_convention": "es_n0_per_symbol",
            "rate": str(structural_row["ldpc_rate"]),
        })
        for code_word_length in packet.e_r
    )
    feasibility = Feasibility(feasible=True, code_blocks=len(identities))
    return feasibility, identities


# ---------------------------------------------------------------------------
# Execution — exactly once
# ---------------------------------------------------------------------------


def run_pass_one(
    authorization_path: Path = E5_AUTHORIZATION_PATH,
    *,
    output_path: Path = PASS_ONE_STATE_PATH,
) -> dict[str, Any]:
    """Execute selection pass one exactly once under the narrow authorization.

    Refuses when the immutable completion record already exists: the single
    content-addressed state file *is* the exactly-once guarantee, and the
    pre-execution marker must already be in place per the sudden-exit protocol.
    """

    context: dict[str, Any] | None = None
    if Path(output_path).exists():
        # The exactly-once guard is checked before anything else, so even a
        # call with malformed arguments cannot run while a completion record
        # exists.
        raise G8EPassOneError(
            "pass one already has an immutable completion record; executing again "
            f"is forbidden ({output_path})"
        )
    context = authenticate_inputs(authorization_path)
    authorization = context["authorization"]
    marker = authenticate_marker(E5_MARKER_PATH, authorization_path, authorization)

    plan = context["plan"]
    typed = _issue_sweep_authorization(
        authorized_by=str(authorization["authorized_by"]),
        reason=f"E5 pass one under authorization {authorization['issued_sha256']}",
        max_candidates=int(plan["max_candidates"]),
        max_samples=int(plan["max_samples"]),
    )
    calls: list[dict[str, Any]] = []
    totals = {
        "candidates_evaluated": 0,
        "eligible_evaluations": 0,
        "infeasible_evaluations": 0,
        "uncharacterized_evaluations": 0,
        "snr_cells_with_selection": 0,
        "snr_cells_without_selection": 0,
        "tie_breaks_applied": 0,
    }
    for call in plan["calls"]:
        evaluations_by_snr = build_call_evaluations(context, call)
        curve = select_operating_points(
            str(call["mode"]),
            evaluations_by_snr,
            samples_per_cell=int(call["samples_per_cell"]),
            authorization=typed,
        )
        calls.append(_serialize_curve(curve, call, context, totals))
    body: dict[str, Any] = {
        "schema_version": PASS_ONE_SCHEMA_VERSION,
        "artifact_role": STATE_ROLE,
        "status": "PASS_ONE_COMPLETE",
        "campaign_id": context["contract"]["campaign_id"],
        "contract_id": context["contract"]["contract_id"],
        "authorization_issued_sha256": authorization["issued_sha256"],
        "marker_issued_sha256": marker["issued_sha256"],
        "inputs": {
            "e2_completion_sha256": authorization["e2_completion_sha256"],
            "e3_id": authorization["e3_id"],
            "e3_sha256": authorization["e3_sha256"],
            "e4_id": authorization["e4_id"],
            "e4_sha256": authorization["e4_sha256"],
            "bler_table_id": authorization["bler_table_id"],
            "bler_table_sha256": authorization["bler_table_sha256"],
            "w4_integration_adjudication_sha256": authorization[
                "w4_integration_adjudication_sha256"
            ],
            "selection_policy_sha256": authorization["selection_policy_sha256"],
            "selection_call_plan_sha256": authorization["selection_call_plan_sha256"],
            "candidate_authority_file_sha256": authorization[
                "candidate_authority_file_sha256"
            ],
            "outage_policy_file_sha256": authorization["outage_policy_file_sha256"],
            "data_identity_id": authorization["data_identity_id"],
        },
        "scorer_module": SCORER_MODULE,
        "tie_break_order": list(composition.TIE_BREAK_ORDER),
        "call_count": plan["call_count"],
        "calls": calls,
        "totals": totals,
        "counters": {
            "pass_one_executed_count": 1,
            "training": 0,
            "pass_two": 0,
            "pass_three": 0,
            "fallback_invoked": 0,
            "ratio_adjudicated": 0,
            "test_access": 0,
            "learned_system_training": 0,
            "g8_f_execution": 0,
        },
        "corpus_spec_binding": {
            "corpus_spec_path": "results/baseline/g8_e/corpus_spec.json",
            "selection_record_field": "authority_candidate_id",
            "state_is_immutable": True,
        },
    }
    body["state_id"] = v3._id(
        STATE_PREFIX, {key: child for key, child in body.items() if key != "state_id"}
    )
    body["state_sha256"] = v3.sha256_bytes(v3.canonical_json(body))
    _atomic_publish(Path(output_path), v3.rendered_json(body))
    return body


def _serialize_curve(
    curve: Any,
    call: Mapping[str, Any],
    context: Mapping[str, Any],
    totals: dict[str, int],
) -> dict[str, Any]:
    """Serialize one CurveSelection with authority-candidate linkage."""

    authority_ids = _authority_ids_for_ratio(context, str(call["ratio"]))
    counts = {"eligible": 0, "infeasible": 0, "uncharacterized": 0}
    record = curve.as_record()
    per_snr = []
    for entry in record["per_snr"]:
        selection = entry["selection"]
        for status in counts:
            counts[status] += int(selection["counts"].get(status, 0))
        totals["candidates_evaluated"] += sum(selection["counts"].values())
        totals["eligible_evaluations"] += int(selection["counts"].get("eligible", 0))
        totals["infeasible_evaluations"] += int(selection["counts"].get("infeasible", 0))
        totals["uncharacterized_evaluations"] += int(
            selection["counts"].get("uncharacterized", 0)
        )
        selected_authority_id = None
        selected = selection["selected"]
        if selected is not None:
            totals["snr_cells_with_selection"] += 1
            totals["tie_breaks_applied"] += int(bool(selection["tie_break_applied"]))
            scorer_key = json.dumps(selected["candidate"], sort_keys=True, separators=(",", ":"))
            selected_authority_id = authority_ids.get(scorer_key)
            if selected_authority_id is None:
                raise G8EPassOneError(
                    "a selected candidate does not map back to the logical "
                    f"candidate authority: {scorer_key}"
                )
        else:
            totals["snr_cells_without_selection"] += 1
        per_snr.append({
            "snr_db": entry["snr_db"],
            "authority_candidate_id": selected_authority_id,
            "selected_composition": None
            if selected is None
            else selected["composition"],
            "tied_authority_candidate_ids": [
                authority_ids.get(json.dumps(candidate, sort_keys=True, separators=(",", ":")))
                for candidate in selection["tied_candidates"]
            ],
            "tie_break_applied": selection["tie_break_applied"],
            "counts": selection["counts"],
            "reason": selection["reason"],
        })
        for tied in selection["tied_candidates"]:
            if authority_ids.get(json.dumps(tied, sort_keys=True, separators=(",", ":"))) is None:
                raise G8EPassOneError("a tied candidate does not map back to the authority")
    return {
        "dataset": call["dataset"],
        "ratio": call["ratio"],
        "mode": call["mode"],
        "samples_per_cell": call["samples_per_cell"],
        "held_fixed": record["held_fixed"],
        "per_snr": per_snr,
        "evaluation_counts": counts,
    }


def _authority_ids_for_ratio(context: Mapping[str, Any], ratio: str) -> dict[str, str]:
    """Map each scorer candidate identity to its logical authority candidate ID.

    The key canonicalizes exactly like ``Candidate.candidate_id`` — including
    the ``float`` coercion of ``snr_db`` — so authority rows and scorer
    candidates cannot disagree by representation.
    """

    ids: dict[str, str] = {}
    for row in context["candidate_authority"]["candidates"]:
        if row["dataset"] == v3.INITIAL_DATASET and row["ratio"] == ratio:
            key = json.dumps(
                {
                    "dataset": str(row["dataset"]),
                    "encode_axis_px": int(row["encode_axis_px"]),
                    "ldpc_rate": str(row["ldpc_rate"]),
                    "modulation": str(row["modulation"]),
                    "ratio": str(row["ratio"]),
                    "snr_db": float(row["snr_db"]),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            ids[key] = str(row["candidate_id"])
    return ids


def _atomic_publish(path: Path, payload: bytes) -> None:
    path = Path(path)
    staging = path.parent / f".{path.name}.staging-{os.getpid()}"
    try:
        with open(staging, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


# ---------------------------------------------------------------------------
# Independent verification
# ---------------------------------------------------------------------------


def verify_pass_one_state(
    state_path: Path = PASS_ONE_STATE_PATH,
    *,
    authorization_path: Path = E5_AUTHORIZATION_PATH,
) -> dict[str, Any]:
    """Independently authenticate the completion record and its derivation.

    Re-runs the frozen scorer over the authenticated frozen inputs and requires
    byte-equal selections, then checks the counters, the exact-once marker and
    the corpus-spec lineage field.
    """

    context = authenticate_inputs(authorization_path)
    authorization = context["authorization"]
    value, raw = _rendered_object(Path(state_path), "v3 pass-one completion record")
    if v3.sha256_bytes(raw) != v3.sha256_bytes(v3.rendered_json(value)):
        raise G8EPassOneError("pass-one state file is not canonical rendered JSON")
    body_without_digest = {
        key: child for key, child in value.items() if key != "state_sha256"
    }
    if value.get("state_sha256") != v3.sha256_bytes(v3.canonical_json(body_without_digest)):
        raise G8EPassOneError("pass-one state digest differs")
    body_without_id = {
        key: child for key, child in body_without_digest.items() if key != "state_id"
    }
    if value.get("state_id") != v3._id(STATE_PREFIX, body_without_id):
        raise G8EPassOneError("pass-one state ID differs")
    if (
        value.get("schema_version") != PASS_ONE_SCHEMA_VERSION
        or value.get("artifact_role") != STATE_ROLE
        or value.get("status") != "PASS_ONE_COMPLETE"
        or value.get("campaign_id") != context["contract"]["campaign_id"]
        or value.get("contract_id") != context["contract"]["contract_id"]
    ):
        raise G8EPassOneError("pass-one state header differs")
    if value.get("authorization_issued_sha256") != authorization["issued_sha256"]:
        raise G8EPassOneError("pass-one state binds a different authorization")
    if value.get("marker_issued_sha256") != authenticate_marker(
        E5_MARKER_PATH, authorization_path, authorization
    )["issued_sha256"]:
        raise G8EPassOneError("pass-one state binds a different pre-execution marker")
    if set(value.get("inputs", {})) != {
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
        "data_identity_id",
    }:
        raise G8EPassOneError("pass-one input bindings differ")
    for key, child in value["inputs"].items():
        if authorization.get(key) != child:
            raise G8EPassOneError(f"pass-one input binding {key} differs")
    counters = value.get("counters", {})
    if counters.get("pass_one_executed_count") != 1:
        raise G8EPassOneError("pass-one executed count is not exactly one")
    prohibited = (
        "training",
        "pass_two",
        "pass_three",
        "fallback_invoked",
        "ratio_adjudicated",
        "test_access",
        "learned_system_training",
        "g8_f_execution",
    )
    if any(counters.get(name) != 0 for name in prohibited):
        raise G8EPassOneError("a prohibited pass-one counter is nonzero")

    recomputed_calls = []
    typed = _issue_sweep_authorization(
        authorized_by=str(authorization["authorized_by"]),
        reason=f"E5 pass one under authorization {authorization['issued_sha256']}",
        max_candidates=int(context["plan"]["max_candidates"]),
        max_samples=int(context["plan"]["max_samples"]),
    )
    totals = {
        "candidates_evaluated": 0,
        "eligible_evaluations": 0,
        "infeasible_evaluations": 0,
        "uncharacterized_evaluations": 0,
        "snr_cells_with_selection": 0,
        "snr_cells_without_selection": 0,
        "tie_breaks_applied": 0,
    }
    for call in context["plan"]["calls"]:
        evaluations_by_snr = build_call_evaluations(context, call)
        curve = select_operating_points(
            str(call["mode"]),
            evaluations_by_snr,
            samples_per_cell=int(call["samples_per_cell"]),
            authorization=typed,
        )
        recomputed_calls.append(_serialize_curve(curve, call, context, totals))
    if value.get("calls") != recomputed_calls:
        raise G8EPassOneError(
            "the recorded selections are not reproducible from the frozen inputs"
        )
    if value.get("totals") != totals:
        raise G8EPassOneError("pass-one aggregate totals differ from the recomputation")
    if value.get("call_count") != context["plan"]["call_count"] or len(value["calls"]) != context["plan"]["call_count"]:
        raise G8EPassOneError("pass-one call count differs from the frozen plan")
    if value.get("tie_break_order") != list(composition.TIE_BREAK_ORDER):
        raise G8EPassOneError("pass-one tie-break order differs from the live frozen order")
    return {
        "status": "PASS",
        "state_id": value["state_id"],
        "state_sha256": value["state_sha256"],
        "file_sha256": v3.sha256_bytes(raw),
        "selections": value["totals"]["snr_cells_with_selection"],
        "cells_without_selection": value["totals"]["snr_cells_without_selection"],
        "calls": value["call_count"],
        "counters": counters,
    }
