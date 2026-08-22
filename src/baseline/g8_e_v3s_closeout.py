"""Additive post-completion closeout layer for the worker-successor epoch.

The frozen v3s lifecycle wrappers pass the contract's seven-key
scientific-data-identity summary block to loaders that require the complete
data identity object: ``g8_e_corrected_v3s.verify_active_e2`` calls
``frozen_validation_metadata(contract["scientific_data_identity"])`` and the
production branches of ``tools/merge_g8_e_corrected_v3s.py`` and
``tools/aggregate_g8_e_corrected_v3s.py`` repeat the pattern.  The summary
block carries no ``manifest_bytes``, so every production E2-completion/E3/E4
invocation fails before any check runs.  The defect is closeout-layer only:
the production runner already authenticates the full identity FILE and its
regression test
(``tests/test_g8_e_e2_successor.py::test_live_identity_check_rejects_the_contract_summary_block``)
pins that rule, so no measurement semantics are in question.

This module restores exactly that pre-registered rule for E2-completion, E3
and E4 verification and publication.  It is additive: no byte-bound source of
the frozen epoch changes, ``verify_frozen_contract`` still authenticates the
complete frozen chain with live sources and live data, and every scientific
primitive (durable-prefix authentication, completion-artifact verification,
exact-set closure, cache authentication, count-derived aggregation) remains
the frozen v3 implementation invoked unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_corrected_v3s as v3s

G8EV3SError = v3s.G8EV3SError

sha256_file = v3.sha256_file


def load_bound_data_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Load and authenticate the full data identity FILE bound by the contract.

    The tracked repository path named by ``contract["scientific_data_identity"]``
    is the authenticator; its bytes must reproduce both the bound data identity
    ID and the bound file SHA-256.  This mirrors the production runner's gate.
    """

    block = contract["scientific_data_identity"]
    path = v3.REPO_ROOT / str(block["path"])
    value, raw = v3._rendered_object(path, "v3 reused scientific data identity")
    if value.get("data_identity_id") != block["id"] or v3.sha256_bytes(raw) != block["sha256"]:
        raise G8EV3SError("the contract-bound scientific data identity file differs from its contract block")
    return dict(value)


def active_context(
    *,
    runtime_root: Path,
    authorization_path: Path,
    contract: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
    data_identity: Mapping[str, Any] | None = None,
    sample_ids: tuple[str, ...] | None = None,
    sample_labels: Mapping[str, int] | None = None,
    verify_live_sources: bool = True,
    verify_live_data: bool = True,
) -> dict[str, Any]:
    """Authenticate the epoch exactly like ``v3s.verify_active_e2``, corrected.

    One code path serves production and synthetic fixtures: when ``contract``
    is None the frozen tracked epoch is authenticated (live sources and live
    data included) and the full data identity is loaded from its contract-bound
    path; fixtures inject every object explicitly and are marked non-production
    by their record labels downstream.  Production sample identities always
    derive from the FULL data identity object, never from a contract summary
    block.
    """

    bundle: dict[str, Any]
    production = contract is None
    if production and (sample_ids is not None or sample_labels is not None):
        raise G8EV3SError("production closeout derives sample identities from the bound data identity file")
    if production:
        bundle = dict(v3s.verify_frozen_contract(verify_live_sources=verify_live_sources, verify_live_data=verify_live_data))
        contract = bundle["contract"]
    else:
        bundle = {"contract": dict(contract)}
        contract = bundle["contract"]
    resolved_identity = load_bound_data_identity(contract) if data_identity is None else dict(data_identity)
    resolved_authority = v3s.load_measurement_authority() if authority is None else dict(authority)
    if production:
        derived_ids, derived_labels = v3.frozen_validation_metadata(resolved_identity)
    else:
        if sample_ids is None or sample_labels is None:
            raise G8EV3SError("synthetic closeout contexts must inject exact sample identities")
        derived_ids, derived_labels = tuple(sample_ids), dict(sample_labels)
    authorization = v3s.authenticate_owner_authorization(authorization_path, contract, resolved_identity)
    state = v3s.verify_runtime_prefix_readonly(
        runtime_root=Path(runtime_root),
        contract=contract,
        authority=resolved_authority,
        sample_ids=derived_ids,
    )
    return {
        **bundle,
        "contract": contract,
        "authority": resolved_authority,
        "data_identity": resolved_identity,
        "sample_ids": derived_ids,
        "sample_labels": derived_labels,
        "authorization": authorization,
        "state": state,
        "production": production,
        "phase": "ACTIVE_E2",
    }


def verify_e2_complete(
    *,
    runtime_root: Path = v3s.V3S_RUNTIME_ROOT,
    authorization_path: Path = v3s.V3S_AUTHORIZATION_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    """Corrected equivalent of ``v3s.verify_e2_complete``."""

    context = active_context(runtime_root=runtime_root, authorization_path=authorization_path, **kwargs)
    observed, digest = v3.verify_e2_completion_artifact(
        runtime_root=runtime_root,
        contract=context["contract"],
        authority=context["authority"],
        production=bool(context["production"]),
    )
    return {**context, "completion": observed, "completion_sha256": digest, "phase": "E2_COMPLETE"}


def verify_e3_complete(
    *,
    e3_path: Path = v3s.V3S_E3_PATH,
    e3_sha256: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Corrected equivalent of ``v3s.verify_e3_complete``."""

    injected_runtime_root = kwargs.pop("runtime_root", None)
    derived_runtime_root = Path(e3_path).parent
    if injected_runtime_root is not None and Path(injected_runtime_root) != derived_runtime_root:
        raise G8EV3SError("E3 path and runtime root disagree")
    complete = verify_e2_complete(runtime_root=derived_runtime_root, **kwargs)
    value = v3.verify_e3_artifact(e3_path, contract=complete["contract"], expected_sha256=e3_sha256)
    return {**complete, "e3": value, "e3_sha256": sha256_file(e3_path), "phase": "E3_COMPLETE"}


def verify_e4_complete(
    *,
    e4_path: Path = v3s.V3S_E4_PATH,
    e3_path: Path = v3s.V3S_E3_PATH,
    e3_sha256: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Corrected equivalent of ``v3s.verify_e4_complete``."""

    e3_complete = verify_e3_complete(e3_path=e3_path, e3_sha256=e3_sha256, **kwargs)
    bound_e3_sha = e3_sha256 or sha256_file(e3_path)
    value = v3.verify_e4_artifact(
        e4_path,
        contract=e3_complete["contract"],
        e3_path=e3_path,
        e3_sha256=bound_e3_sha,
    )
    return {**e3_complete, "e4": value, "e4_sha256": sha256_file(e4_path), "phase": "E4_COMPLETE"}


def publish_e3(
    *,
    runtime_root: Path = v3s.V3S_RUNTIME_ROOT,
    authorization_path: Path = v3s.V3S_AUTHORIZATION_PATH,
    authenticate_caches: bool = True,
    **kwargs: Any,
) -> tuple[dict[str, Any], Path, str]:
    """Publish the E3 exact-set closure through the corrected loader.

    Production invocations must use the frozen epoch runtime root; injected
    synthetic contexts may target an isolated temporary runtime only.
    """

    context = active_context(runtime_root=runtime_root, authorization_path=authorization_path, **kwargs)
    if context["production"]:
        if Path(runtime_root).resolve() != v3s.V3S_RUNTIME_ROOT.resolve():
            raise G8EV3SError("worker E3 must use the frozen v3s runtime root")
    v3.verify_e2_completion_artifact(
        runtime_root=runtime_root,
        contract=context["contract"],
        authority=context["authority"],
        production=bool(context["production"]),
    )
    return v3.publish_e3_artifact(
        authority=context["authority"],
        sample_ids=context["sample_ids"],
        sample_labels=context["sample_labels"],
        runtime_root=runtime_root,
        contract=context["contract"],
        production=bool(context["production"]),
        authenticate_caches=authenticate_caches,
    )


def publish_e4(
    *,
    runtime_root: Path = v3s.V3S_RUNTIME_ROOT,
    authorization_path: Path = v3s.V3S_AUTHORIZATION_PATH,
    e3_path: Path | None = None,
    e3_sha256: str,
    **kwargs: Any,
) -> tuple[dict[str, Any], Path, str]:
    """Publish E4 from one exact SHA-bound E3 closure through the corrected loader.

    Production invocations must use the frozen epoch runtime root and E3 path;
    injected synthetic contexts may target isolated temporary paths only.
    """

    resolved_e3_path = Path(e3_path) if e3_path is not None else v3s.V3S_E3_PATH
    context = active_context(runtime_root=runtime_root, authorization_path=authorization_path, **kwargs)
    if context["production"]:
        if (
            Path(runtime_root).resolve() != v3s.V3S_RUNTIME_ROOT.resolve()
            or resolved_e3_path.resolve() != v3s.V3S_E3_PATH.resolve()
        ):
            raise G8EV3SError("worker E4 must use the frozen v3s runtime/E3 paths")
    v3.verify_e2_completion_artifact(
        runtime_root=runtime_root,
        contract=context["contract"],
        authority=context["authority"],
        production=bool(context["production"]),
    )
    return v3.publish_e4_artifact(
        authority=context["authority"],
        sample_ids=context["sample_ids"],
        runtime_root=runtime_root,
        contract=context["contract"],
        e3_path=resolved_e3_path,
        e3_sha256=e3_sha256,
        production=bool(context["production"]),
    )
