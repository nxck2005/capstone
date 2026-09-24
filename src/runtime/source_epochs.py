"""The prospective W10/PAPR downstream source epochs (AM-98, AM-99).

The historical W9 v4 manifest (`results/learned/w9/downstream_source_manifest_v4.json`)
remains immutable.  Successor-v1 is the AM-98 freeze that produced zero PAPR and
zero W10 science; it is preserved as superseded-before-science evidence.
Successor-v2 (AM-99) remains immutable history.  Successor-v3 is the
superseded-before-science repaired epoch.  Successor-v4 is the pre-science epoch
over the narrow authority-generation contract repair; it produced the one closed
PAPR-constrained training lifecycle and therefore is *not* superseded-before-
science history.  The active-epoch closure check lets closed-authority verifiers
keep verifying against their own embedded bytes while the successor governs new
work.

Successor-v5 is the post-PAPR/pre-selection source repair: the frozen JPEG
selector was missing its ``canonical_sha256`` binding, so the first selection
launch failed before any selection artifact existed.  v5 binds the exact
immutable v4 bytes and the terminal PAPR completion as a dedicated
``transition_from_v4`` record, records the honest post-PAPR/pre-future-work
state (one closed lifecycle, zero selections, zero W10 work, test sealed), and
never claims that v4 was superseded before science.

Successor-v6 is the post-JPEG/pre-ER-12 custody and source repair.  It binds
the completed JPEG selection executed under historical v5, its lossless
publication carrier and descriptor, and the unchanged v5 bytes.  It records
one closed JPEG selection while keeping ER-12, W10 units, W10 authority, G-12
and test access unopened.

Successor-v7 is a post-JPEG/pre-ER-12 non-scientific static-policy repair.  It binds the exact v6 bytes, adds honest literal annotations to the JPEG carrier implementation without changing values, and re-authenticates the carrier while preserving the closed PAPR and JPEG history.

Successor-v8 truthfully records the failed v7 ER-12 startup before candidate selection. It binds the exact v7 bytes and repairs only ER-12's consumption of the frozen ER-9 checkpoint identity; the PAPR and JPEG lifecycles remain bound to their historical v4 and v5 sources.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runtime.source_guard import (
    PROTECTED_PREFIXES,
    assert_manifest_commit_bytes,
    build_manifest,
    committed_source_differences,
    git_tree_hashes,
    working_tree_source_differences,
)
from training.deterministic_core import canonical_sha256

W10_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest.json"
W10_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V1"
W10_MANIFEST_PREFIX = "w10downstreamsource-"
W10_V2_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v2.json"
W10_V2_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V2"
W10_V2_MANIFEST_PREFIX = "w10downstreamsourcev2-"
W10_V3_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v3.json"
W10_V3_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V3"
W10_V3_MANIFEST_PREFIX = "w10downstreamsourcev3-"
W10_V4_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v4.json"
W10_V4_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V4"
W10_V4_MANIFEST_PREFIX = "w10downstreamsourcev4-"
W10_V5_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v5.json"
W10_V5_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V5"
W10_V5_MANIFEST_PREFIX = "w10downstreamsourcev5-"
W10_V6_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v6.json"
W10_V6_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V6"
W10_V6_MANIFEST_PREFIX = "w10downstreamsourcev6-"
W10_V7_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v7.json"
W10_V7_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V7"
W10_V7_MANIFEST_PREFIX = "w10downstreamsourcev7-"
W10_V7_MANIFEST_ID = "w10downstreamsourcev7-e3121abde952dde9a6dbe04fd35b8e7160adbe9ce416c516c867beb0bf764f88"
W10_V7_MANIFEST_SHA256 = "4adb904f86c444624eec67b21dd33ebcf0ce211f402a9641b06661255b195473"
W10_V7_SOURCE_COMMIT = "cc705300dc0dd33dc23498ff127b56f1f92427ad"
W10_V8_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v8.json"
W10_V8_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V8"
W10_V8_MANIFEST_PREFIX = "w10downstreamsourcev8-"
W10_V9_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v9.json"
W10_V9_MANIFEST_KIND = "W10_VALIDATION_CONTINUATION_SOURCE_SUCCESSOR_V9"
W10_V9_MANIFEST_PREFIX = "w10downstreamsourcev9-"
W10_V8_MANIFEST_ID = "w10downstreamsourcev8-e13ba8905c1910645794ffdf5d7a06c55c55b8a01beec84a392a4a59f9bc48fc"
W10_V8_MANIFEST_SHA256 = "c37ce2c845acaa111b9ee0d49552867fc671e2ffe7e593589102c8d7e1a1f3c3"
W10_V8_AUTHORITY_ID = "w10rehearsalauth-18582571f86b71aadcd546808f1199410d1c55cd0237def20689d985aa21eb7d"
W10_V8_AUTHORITY_SHA256 = "0db7e8900c8cb4415e34e62167e8065dba5ecda8648f28d5e2885729c20abdb0"
W10_V8_CUSTODY_PATH = "results/learned/w10/w10_v8_failed_prefix_custody.json"
W10_V8_CUSTODY_ID = "w10v8prefix-a4cbafd6366eb105c24c3806c9182689c81ff4323aee4d45138d75ad4b80171f"
W10_V8_CUSTODY_SHA256 = "4b20751860633e815bb63236ad5796bfba0092496ec59456428d465d6f7b53a1"
W10_V8_PREFIX_DIGEST = "4f83fc7d5889ab1209f7e3c66a7d9e485e9c940509c981252c8c05d31f9769a5"
W10_V8_EXECUTION_COMMIT = "c906ab920cba652c8ea681b4cefda869d5e829b2"
W10_MANIFEST_KINDS = (
    W10_MANIFEST_KIND,
    W10_V2_MANIFEST_KIND,
    W10_V3_MANIFEST_KIND,
    W10_V4_MANIFEST_KIND,
    W10_V5_MANIFEST_KIND,
    W10_V6_MANIFEST_KIND,
    W10_V7_MANIFEST_KIND,
    W10_V8_MANIFEST_KIND,
    W10_V9_MANIFEST_KIND,
)
W10_EPOCHS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9")
_EPOCH_FOR_KIND = {
    W10_MANIFEST_KIND: "v1",
    W10_V2_MANIFEST_KIND: "v2",
    W10_V3_MANIFEST_KIND: "v3",
    W10_V4_MANIFEST_KIND: "v4",
    W10_V5_MANIFEST_KIND: "v5",
    W10_V6_MANIFEST_KIND: "v6",
    W10_V7_MANIFEST_KIND: "v7",
    W10_V8_MANIFEST_KIND: "v8",
    W10_V9_MANIFEST_KIND: "v9",
}
HISTORICAL_SOURCE_PATH = "results/learned/w9/downstream_source_manifest_v4.json"
HISTORICAL_SOURCE_COMMIT = "22fde3e0ba0c8ad7a92356587eb95780ded89ee6"

# The closed PAPR lifecycle executed under the immutable v4 epoch.  These are
# historical facts: a successor may govern new work but must bind exactly these
# identities, and the PAPR authority must keep verifying against v4 bytes.
W10_V4_MANIFEST_ID = "w10downstreamsourcev4-28af9f881b27af36f7cfbceb313e81fbf0081b0baab455475f9c709d65e53a5a"
W10_V4_MANIFEST_SHA256 = "54f2f736dd75cba612a3bdc803880c86379af1557c3cc09b0df8070744edfb9a"
W10_V4_SOURCE_COMMIT = "982688ab119959d3b861af302f896c8e952a4c60"
PAPR_COMPLETION_SOURCE_PATH = "results/learned/w10/papr_training_completion.json"
PAPR_COMPLETION_PREFIX = "paprcompletion-"
PAPR_COMPLETION_ID = "paprcompletion-2d23d342ad7fe2661c9a3926f5e025b3d7174c290f5e56983d05efbbc65d74ca"
PAPR_COMPLETION_SHA256 = "ced0e6a955071252863558dd9b1db41fc93d408435cdb2131d729e4b18afdfa4"
PAPR_EPOCH_COUNT = 100  # literal-ok: closed PAPR lifecycle epoch count
W10_V5_TRANSITION_KIND = "post_papr_pre_selection_source_repair"
W10_V5_REPAIR_REASON = "jpeg_selector_missing_canonical_sha256_import"
W10_V5_MANIFEST_ID = "w10downstreamsourcev5-00e19a1c1fbf0a6052347a58db57d7236e1efde9914a978e3b45408844339652"
W10_V5_MANIFEST_SHA256 = "4fe35441999bc5771fa6fea10692378eab76ce6e5ac626d68202d80802c81ae3"
W10_V5_SOURCE_COMMIT = "1e0d26e8f12b791aa4ba099dc3488111e1342880"
W10_V6_MANIFEST_ID = "w10downstreamsourcev6-4a6839536aa0b7cf8663ae35bf0906e5df6f52e4555e3abe460959ce615347d5"
W10_V6_MANIFEST_SHA256 = "01ea25273494241c42bf2b4dfdf3873d3196893068cb06ab35af29806936c53a"
W10_V6_SOURCE_COMMIT = "e626e13491c00fb245cf467d54cd53d743717711"
W10_V6_TRANSITION_KIND = "post_jpeg_pre_er12_source_repair"
W10_V6_REPAIR_REASONS = (
    "jpeg_analytic_packet_metadata_json_normalization",
    "jpeg_selection_lossless_publication_carrier",
    "post_jpeg_clean_checkout_for_er12",
)
W10_V7_TRANSITION_KIND = "post_jpeg_pre_er12_static_policy_repair"
W10_V7_REPAIR_REASON = "jpeg_carrier_literal_policy_annotations"
W10_V8_TRANSITION_KIND = "post_er12_startup_failure_pre_candidate_source_repair"
W10_V8_REPAIR_REASON = "er12_er9_checkpoint_binding_key_repair"
_W10_V7_JPEG_BINDING_FIELDS = (
    "jpeg_validation_selection_id",
    "jpeg_validation_selection_contract_sha256",
    "jpeg_validation_selection_raw_sha256",
    "jpeg_validation_selection_raw_bytes",
    "jpeg_validation_selection_source_epoch",
    "jpeg_validation_selection_carrier_path",
    "jpeg_validation_selection_carrier_sha256",
    "jpeg_validation_selection_carrier_bytes",
    "jpeg_validation_selection_carrier_descriptor_path",
    "jpeg_validation_selection_carrier_descriptor_id",
    "jpeg_validation_selection_carrier_descriptor_sha256",
)

# Successor-v5 must freeze before any of these post-PAPR artifacts can exist.
# They are freeze-time facts recorded in the transition; they are deliberately
# not re-asserted by live verification, because the selections and the W10
# authority are legitimate later work under this same epoch.
W10_JPEG_SELECTION_SOURCE_PATH = "results/learned/w10/jpeg_validation_selection.json"
W10_ER12_SELECTION_SOURCE_PATH = "results/learned/w10/er12_validation_selection.json"
W10_AUTHORITY_SOURCE_PATH = "results/learned/w10/w10_rehearsal_authorization.json"
W10_CLOSEOUT_SOURCE_PATH = "results/learned/w10/w10_rehearsal_closeout.json"
W10_RUNTIME_SOURCE_PATH = "checkpoints/w10_rehearsal"
G12_FREEZE_MANIFEST_SOURCE_PATH = "results/freeze_manifest.json"

W10_V5_PRE_FUTURE_WORK_STATE = {
    "papr_constrained_training_runs": 1,
    "papr_lifecycle_closed": True,
    "jpeg_validation_selection_count": 0,
    "er12_validation_selection_count": 0,
    "w10_authority_frozen": False,
    "w10_scientific_units": 0,
    "g12_freeze_manifest": False,
    "test": "SEALED",
    "test_access": 0,
}
W10_V6_PRE_FUTURE_WORK_STATE = {
    "papr_constrained_training_runs": 1,
    "papr_lifecycle_closed": True,
    "jpeg_validation_selection_count": 1,
    "jpeg_validation_selection_closed": True,
    "er12_validation_selection_count": 0,
    "w10_authority_frozen": False,
    "w10_scientific_units": 0,
    "g12_freeze_manifest": False,
    "test": "SEALED",
    "test_access": 0,
}
W10_V7_PRE_FUTURE_WORK_STATE = dict(W10_V6_PRE_FUTURE_WORK_STATE)
W10_V8_PRE_FUTURE_WORK_STATE = dict(W10_V7_PRE_FUTURE_WORK_STATE)
W10_V9_PRE_FUTURE_WORK_STATE = {
    "papr_constrained_training_runs": 1,
    "papr_lifecycle_closed": True,
    "jpeg_validation_selection_count": 1,
    "er12_validation_selection_count": 1,
    "v8_w10_authority_frozen": True,
    "v8_w10_scientific_units_complete": 126,
    "v8_w10_failed_ordinal": 126,
    "v9_continuation_authority_frozen": False,
    "v9_w10_scientific_units": 0,
    "g12_freeze_manifest": False,
    "test": "SEALED",
    "test_access": 0,
}

W10_RELEVANT_CONFIG_PATHS = (
    "configs/er9-digital-pascal-v4.yaml",
    "configs/learned-er2-randomized-pascal-v4.yaml",
    "configs/learned-papr-constrained-r1-6.yaml",
    "spec/params.generated.yaml",
)

# Historical successor-v1 declared this exact boundary and must keep verifying
# against its own bytes; successor-v2 adds the worker-local W10 rehearsal root.
W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES = (
    "results/learned/er9/",
    "results/learned/er2_randomized/",
    "results/learned/g11/",
    "results/learned/w9/",
    "results/learned/w10/",
    "checkpoints/er9_pascal_v4/",
    "checkpoints/er2_randomized_pascal_v4/",
    "checkpoints/papr_constrained_pascal_v4/",
    "checkpoints/smoke/",
    "checkpoints/er9_production_pascal_v4/",
)

W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES = (
    *W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES,
    "checkpoints/w10_rehearsal/",
)

W10_GOVERNS = (
    "papr_constrained_training",
    "w10_validation_rehearsal",
)

WORKING_TREE_GUARD = {
    "checks_committed_source_commit_to_head": True,
    "checks_unstaged_protected_source": True,
    "checks_staged_protected_source": True,
    "checks_untracked_protected_source": True,
    "runtime_and_evidence_only_after_freeze": True,
}


class SourceEpochHold(RuntimeError):
    """The active downstream source epoch is missing or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceEpochHold(message)


def build_w10_manifest(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the successor epoch at the exact clean HEAD it binds."""

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    base["allowed_evidence_runtime_prefixes"] = list(W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_science_state"] = {
        "papr_constrained_training_runs": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }
    base["manifest_id"] = W10_MANIFEST_PREFIX + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


def assert_w10_manifest_contract(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == 2, "W10 source manifest schema differs")
    _require(manifest.get("manifest_kind") in W10_MANIFEST_KINDS, "W10 source manifest kind differs")
    _require(manifest.get("source_commit_comparison") == "exact_clean_HEAD_at_freeze", "W10 source freeze rule differs")
    _require(manifest.get("protected_source_prefixes") == list(PROTECTED_PREFIXES), "W10 protected source boundary differs")
    if manifest.get("manifest_kind") in {
        W10_V2_MANIFEST_KIND,
        W10_V3_MANIFEST_KIND,
        W10_V4_MANIFEST_KIND,
        W10_V5_MANIFEST_KIND,
        W10_V6_MANIFEST_KIND,
        W10_V7_MANIFEST_KIND,
        W10_V8_MANIFEST_KIND,
        W10_V9_MANIFEST_KIND,
    }:
        _require(manifest.get("allowed_evidence_runtime_prefixes") == list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES), "W10 successor evidence boundary differs")
    else:
        _require(manifest.get("allowed_evidence_runtime_prefixes") == list(W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES), "W10 v1 evidence boundary differs")
    _require(manifest.get("working_tree_guard") == WORKING_TREE_GUARD, "W10 working-tree guard differs")
    _require(manifest.get("governs") == list(W10_GOVERNS), "W10 governed lifecycle differs")
    _require(manifest.get("historical_w9_downstream_manifest", {}).get("source_commit") == HISTORICAL_SOURCE_COMMIT, "W10 historical source binding differs")
    source_commit = manifest.get("source_commit")
    _require(isinstance(source_commit, str) and len(source_commit) == 40, "W10 source commit is malformed")  # literal-ok: Git SHA-1 width
    trees = manifest.get("tree_hashes")
    _require(
        isinstance(trees, Mapping)
        and set(trees) == {"repository", "src", "tools", "configs", "spec", "tests"}
        and all(isinstance(value, str) and len(value) == 40 for value in trees.values()),  # literal-ok: Git SHA-1 width
        "W10 tree closure differs",
    )
    relevant = manifest.get("relevant_config_sha256")
    _require(
        isinstance(relevant, Mapping)
        and tuple(sorted(relevant)) == tuple(sorted(W10_RELEVANT_CONFIG_PATHS)),
        "W10 relevant config closure differs",
    )
    lock = manifest.get("requirements_pascal_lock_sha256")
    _require(isinstance(lock, str) and len(lock) == 64, "W10 Pascal lock identity is malformed")  # literal-ok: SHA-256 width
    pre_science = manifest.get("pre_science_state")
    if manifest.get("manifest_kind") == W10_V5_MANIFEST_KIND:
        _require("pre_science_state" not in manifest, "W10 v5 must not reuse the zero-science pre-science state")
        transition = manifest.get("transition_from_v4")
        _require(isinstance(transition, Mapping), "W10 v5 transition record is missing")
        _require(transition.get("transition_kind") == W10_V5_TRANSITION_KIND, "W10 v5 transition kind differs")
        _require(transition.get("repair_reason") == W10_V5_REPAIR_REASON, "W10 v5 repair reason differs")
        _require(
            transition.get("predecessor_papr_constrained_training_runs") == 1,
            "W10 v5 must record exactly one closed PAPR training lifecycle",
        )
        _require(
            transition.get("jpeg_validation_selection_count") == 0
            and transition.get("er12_validation_selection_count") == 0,
            "W10 v5 must record a pre-selection freeze",
        )
        _require(
            transition.get("w10_authority_frozen") is False
            and transition.get("w10_scientific_units") == 0
            and transition.get("g12_freeze_manifest") is False
            and transition.get("test") == "SEALED"
            and transition.get("test_access") == 0,
            "W10 v5 transition is not pre-future-work",
        )
        _require(
            isinstance(transition.get("predecessor_papr_completion_id"), str)
            and str(transition["predecessor_papr_completion_id"]).startswith(PAPR_COMPLETION_PREFIX),
            "W10 v5 does not bind a PAPR completion",
        )
        _require("superseded_successor" not in manifest, "W10 v5 must not claim a zero-science supersession")
        _require(
            isinstance(manifest.get("pre_future_work_state"), Mapping)
            and dict(manifest["pre_future_work_state"]) == W10_V5_PRE_FUTURE_WORK_STATE,
            "W10 v5 pre-future-work state differs",
        )
    elif manifest.get("manifest_kind") == W10_V6_MANIFEST_KIND:
        _require("pre_science_state" not in manifest, "W10 v6 must not reuse the zero-science pre-science state")
        _require("superseded_successor" not in manifest, "W10 v6 must not claim a zero-science supersession")
        transition = manifest.get("transition_from_v5")
        _require(isinstance(transition, Mapping), "W10 v6 transition record is missing")
        _require(transition.get("transition_kind") == W10_V6_TRANSITION_KIND, "W10 v6 transition kind differs")
        _require(
            transition.get("repair_reasons") == list(W10_V6_REPAIR_REASONS),
            "W10 v6 repair reasons differ",
        )
        _require(
            transition.get("predecessor_papr_constrained_training_runs") == 1
            and transition.get("jpeg_validation_selection_count") == 1
            and transition.get("jpeg_validation_selection_closed") is True
            and transition.get("er12_validation_selection_count") == 0
            and transition.get("w10_authority_frozen") is False
            and transition.get("w10_scientific_units") == 0
            and transition.get("g12_freeze_manifest") is False
            and transition.get("test") == "SEALED"
            and transition.get("test_access") == 0,
            "W10 v6 JPEG selection count/transition is not post-JPEG/pre-ER-12",
        )
        _require(
            isinstance(manifest.get("pre_future_work_state"), Mapping)
            and dict(manifest["pre_future_work_state"]) == W10_V6_PRE_FUTURE_WORK_STATE,
            "W10 v6 pre-future-work state differs",
        )
    elif manifest.get("manifest_kind") == W10_V7_MANIFEST_KIND:
        _require("pre_science_state" not in manifest, "W10 v7 must not reuse the zero-science pre-science state")
        _require("superseded_successor" not in manifest, "W10 v7 must not claim a zero-science supersession")
        transition = manifest.get("transition_from_v6")
        _require(isinstance(transition, Mapping), "W10 v7 transition record is missing")
        _require(transition.get("transition_kind") == W10_V7_TRANSITION_KIND, "W10 v7 transition kind differs")
        _require(transition.get("repair_reason") == W10_V7_REPAIR_REASON, "W10 v7 repair reason differs")
        _require(
            transition.get("predecessor_papr_constrained_training_runs") == 1
            and transition.get("jpeg_validation_selection_count") == 1
            and transition.get("jpeg_validation_selection_closed") is True
            and transition.get("er12_validation_selection_count") == 0
            and transition.get("w10_authority_frozen") is False
            and transition.get("w10_scientific_units") == 0
            and transition.get("g12_freeze_manifest") is False
            and transition.get("test") == "SEALED"
            and transition.get("test_access") == 0,
            "W10 v7 transition is not post-JPEG/pre-ER-12",
        )
        _require(
            isinstance(manifest.get("pre_future_work_state"), Mapping)
            and dict(manifest["pre_future_work_state"]) == W10_V7_PRE_FUTURE_WORK_STATE,
            "W10 v7 pre-future-work state differs",
        )
    elif manifest.get("manifest_kind") == W10_V9_MANIFEST_KIND:
        _require("pre_science_state" not in manifest and "superseded_successor" not in manifest, "W10 v9 must record executed v8 science")
        transition = manifest.get("transition_from_v8")
        _require(isinstance(transition, Mapping), "W10 v9 transition record is missing")
        _require(transition.get("transition_kind") == "failed_partial_science_to_suffix_continuation", "W10 v9 transition kind differs")
        _require(transition.get("v8_scientific_work_executed") is True and transition.get("completed_ordinals") == [0, 125] and transition.get("failed_ordinal") == 126, "W10 v9 historical frontier differs")
        _require(dict(manifest.get("pre_future_work_state", {})) == W10_V9_PRE_FUTURE_WORK_STATE, "W10 v9 pre-future-work state differs")
    elif manifest.get("manifest_kind") == W10_V8_MANIFEST_KIND:
        _require("pre_science_state" not in manifest, "W10 v8 must not reuse the zero-science pre-science state")
        _require("superseded_successor" not in manifest, "W10 v8 must not claim a zero-science supersession")
        transition = manifest.get("transition_from_v7")
        _require(isinstance(transition, Mapping), "W10 v8 transition record is missing")
        expected_transition = {
            "path": W10_V7_SOURCE_PATH,
            "manifest_kind": W10_V7_MANIFEST_KIND,
            "manifest_id": W10_V7_MANIFEST_ID,
            "sha256": W10_V7_MANIFEST_SHA256,
            "source_commit": W10_V7_SOURCE_COMMIT,
            "transition_kind": W10_V8_TRANSITION_KIND,
            "repair_reason": W10_V8_REPAIR_REASON,
            "predecessor_papr_constrained_training_runs": 1,  # literal-ok: one closed historical PAPR lifecycle
            "jpeg_validation_selection_count": 1,  # literal-ok: one closed historical JPEG selection
            "jpeg_validation_selection_closed": True,
            "er12_validation_selection_count": 0,  # literal-ok: no completed ER-12 selection
            "w10_authority_frozen": False,
            "w10_scientific_units": 0,  # literal-ok: W10 rehearsal remains unopened
            "g12_freeze_manifest": False,
            "test": "SEALED",
            "test_access": 0,  # literal-ok: sealed test boundary
        }
        _require(dict(transition) == expected_transition, "W10 v8 transition is not post-failed-ER-12-startup/pre-candidate-selection")
        _require(
            isinstance(manifest.get("pre_future_work_state"), Mapping)
            and dict(manifest["pre_future_work_state"]) == W10_V8_PRE_FUTURE_WORK_STATE,
            "W10 v8 pre-future-work state differs",
        )
    else:
        _require(
            isinstance(pre_science, Mapping)
            and pre_science.get("papr_constrained_training_runs") == 0
            and pre_science.get("w10_authority_frozen") is False
            and pre_science.get("w10_scientific_units") == 0
            and pre_science.get("g12_freeze_manifest") is False
            and pre_science.get("test") == "SEALED"
            and pre_science.get("test_access") == 0,
            "W10 pre-science state differs",
        )


def successor_path(root: Path, *, epoch: str = "v2") -> Path:
    """Return one explicitly named successor path; no implicit fallback."""

    names = {
        "v1": W10_SOURCE_PATH,
        "v2": W10_V2_SOURCE_PATH,
        "v3": W10_V3_SOURCE_PATH,
        "v4": W10_V4_SOURCE_PATH,
        "v5": W10_V5_SOURCE_PATH,
        "v6": W10_V6_SOURCE_PATH,
        "v7": W10_V7_SOURCE_PATH,
        "v8": W10_V8_SOURCE_PATH,
        "v9": W10_V9_SOURCE_PATH,
    }
    _require(epoch in names, f"unknown W10 successor epoch: {epoch}")
    return Path(root) / names[epoch]


def manifest_prefix(kind: str) -> str:
    if kind == W10_V9_MANIFEST_KIND:
        return W10_V9_MANIFEST_PREFIX
    if kind == W10_V8_MANIFEST_KIND:
        return W10_V8_MANIFEST_PREFIX
    if kind == W10_V7_MANIFEST_KIND:
        return W10_V7_MANIFEST_PREFIX
    if kind == W10_V6_MANIFEST_KIND:
        return W10_V6_MANIFEST_PREFIX
    if kind == W10_V5_MANIFEST_KIND:
        return W10_V5_MANIFEST_PREFIX
    if kind == W10_V4_MANIFEST_KIND:
        return W10_V4_MANIFEST_PREFIX
    if kind == W10_V3_MANIFEST_KIND:
        return W10_V3_MANIFEST_PREFIX
    if kind == W10_V2_MANIFEST_KIND:
        return W10_V2_MANIFEST_PREFIX
    return W10_MANIFEST_PREFIX


def epoch_for_kind(kind: str) -> str:
    """The explicit successor epoch one manifest kind belongs to."""

    _require(kind in _EPOCH_FOR_KIND, f"unknown W10 successor manifest kind: {kind}")
    return _EPOCH_FOR_KIND[kind]


def predecessor_binding(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The exact predecessor record a successor epoch declares.

    Successors v2-v4 declare ``superseded_successor`` (zero-science).
    Successors v5-v8 declare post-science transitions from v4 through v7
    respectively, and must never be described as before-science supersessions.
    """

    kind = manifest.get("manifest_kind")
    if kind in {W10_V2_MANIFEST_KIND, W10_V3_MANIFEST_KIND, W10_V4_MANIFEST_KIND}:
        value = manifest.get("superseded_successor")
    elif kind == W10_V5_MANIFEST_KIND:
        value = manifest.get("transition_from_v4")
    elif kind == W10_V6_MANIFEST_KIND:
        value = manifest.get("transition_from_v5")
    elif kind == W10_V7_MANIFEST_KIND:
        value = manifest.get("transition_from_v6")
    elif kind == W10_V8_MANIFEST_KIND:
        value = manifest.get("transition_from_v7")
    elif kind == W10_V9_MANIFEST_KIND:
        value = manifest.get("transition_from_v8")
    else:
        value = None
    return value if isinstance(value, Mapping) else None


def active_manifest_path(root: Path) -> Path:
    for epoch in ("v9", "v8", "v7", "v6", "v5", "v4", "v3", "v2"):
        candidate = successor_path(root, epoch=epoch)
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return successor_path(root, epoch="v1")


def load_w10_manifest(root: Path, *, live: bool = True, epoch: str | None = None) -> dict[str, Any]:
    path = successor_path(root, epoch=epoch) if epoch is not None else active_manifest_path(root)
    _require(path.is_file() and not path.is_symlink(), "W10 successor source manifest is missing")
    value = json.loads(path.read_bytes())
    body = dict(value)
    identifier = body.pop("manifest_id", None)
    kind = value.get("manifest_kind")
    _require(kind in W10_MANIFEST_KINDS, "W10 successor manifest kind differs")
    _require(identifier == manifest_prefix(str(kind)) + canonical_sha256(body), "W10 source manifest ID differs")
    assert_w10_manifest_contract(value)
    if kind == W10_V2_MANIFEST_KIND:
        _require_w10_v2_supersession(root, value)
    if kind == W10_V3_MANIFEST_KIND:
        _require_w10_v3_supersession(root, value)
    if kind == W10_V4_MANIFEST_KIND:
        _require_w10_v4_supersession(root, value)
    if kind == W10_V5_MANIFEST_KIND:
        _require_w10_v5_transition(root, value)
    if kind == W10_V6_MANIFEST_KIND:
        _require_w10_v6_transition(root, value)
    if kind == W10_V7_MANIFEST_KIND:
        _require_w10_v7_transition(root, value)
    if kind == W10_V8_MANIFEST_KIND:
        _require_w10_v8_transition(root, value)
    if kind == W10_V9_MANIFEST_KIND:
        _require_w10_v9_transition(root, value)
    if live:
        assert_clean_active_source_closure(root, value)
    return value


def _require_w10_v2_supersession(root: Path, value: Mapping[str, Any]) -> None:
    """Successor-v2 names the exact superseded-before-science v1 bytes."""

    superseded = value.get("superseded_successor")
    _require(isinstance(superseded, Mapping), "W10 v2 supersession record is missing")
    v1_path = successor_path(root, epoch="v1")
    _require(v1_path.is_file() and not v1_path.is_symlink(), "superseded W10 v1 manifest is missing")
    raw = v1_path.read_bytes()
    _require(superseded.get("path") == W10_SOURCE_PATH, "W10 v2 superseded path differs")
    _require(superseded.get("manifest_id") == json.loads(raw).get("manifest_id"), "W10 v2 superseded ID differs")
    _require(superseded.get("sha256") == hashlib.sha256(raw).hexdigest(), "W10 v2 superseded bytes differ")
    _require(
        superseded.get("superseded_before_science") is True
        and superseded.get("papr_constrained_training_runs") == 0
        and superseded.get("w10_scientific_units") == 0
        and superseded.get("test_access") == 0,
        "W10 v2 supersession is not zero-science",
    )


def _require_w10_v3_supersession(root: Path, value: Mapping[str, Any]) -> None:
    """Successor-v3 names the exact superseded-before-science v2 bytes."""

    superseded = value.get("superseded_successor")
    _require(isinstance(superseded, Mapping), "W10 v3 supersession record is missing")
    v2_path = successor_path(root, epoch="v2")
    _require(v2_path.is_file() and not v2_path.is_symlink(), "superseded W10 v2 manifest is missing")
    raw = v2_path.read_bytes()
    predecessor = json.loads(raw)
    _require(superseded.get("path") == W10_V2_SOURCE_PATH, "W10 v3 superseded path differs")
    _require(superseded.get("manifest_kind") == W10_V2_MANIFEST_KIND, "W10 v3 superseded kind differs")
    _require(superseded.get("manifest_id") == predecessor.get("manifest_id"), "W10 v3 superseded ID differs")
    _require(superseded.get("sha256") == hashlib.sha256(raw).hexdigest(), "W10 v3 superseded bytes differ")
    _require(superseded.get("source_commit") == predecessor.get("source_commit"), "W10 v3 predecessor source commit differs")
    _require(
        superseded.get("superseded_before_science") is True
        and superseded.get("papr_constrained_training_runs") == 0
        and superseded.get("w10_scientific_units") == 0
        and superseded.get("test_access") == 0,
        "W10 v3 supersession is not zero-science",
    )


def _require_w10_v4_supersession(root: Path, value: Mapping[str, Any]) -> None:
    """Successor-v4 names the exact superseded-before-science v3 bytes."""

    superseded = value.get("superseded_successor")
    _require(isinstance(superseded, Mapping), "W10 v4 supersession record is missing")
    v3_path = successor_path(root, epoch="v3")
    _require(v3_path.is_file() and not v3_path.is_symlink(), "superseded W10 v3 manifest is missing")
    raw = v3_path.read_bytes()
    predecessor = json.loads(raw)
    _require(superseded.get("path") == W10_V3_SOURCE_PATH, "W10 v4 superseded path differs")
    _require(superseded.get("manifest_kind") == W10_V3_MANIFEST_KIND, "W10 v4 superseded kind differs")
    _require(superseded.get("manifest_id") == predecessor.get("manifest_id"), "W10 v4 superseded ID differs")
    _require(superseded.get("sha256") == hashlib.sha256(raw).hexdigest(), "W10 v4 superseded bytes differ")
    _require(superseded.get("source_commit") == predecessor.get("source_commit"), "W10 v4 predecessor source commit differs")
    _require(
        superseded.get("superseded_before_science") is True
        and superseded.get("papr_constrained_training_runs") == 0
        and superseded.get("w10_scientific_units") == 0
        and superseded.get("test_access") == 0,
        "W10 v4 supersession is not zero-science",
    )


def _require_w10_v5_transition(root: Path, value: Mapping[str, Any]) -> None:
    """Successor-v5 is a post-PAPR/pre-selection repair of the closed v4 epoch.

    It binds the exact immutable v4 bytes and authenticates the terminal PAPR
    completion that proves the predecessor produced exactly one closed
    validation-only lifecycle.  The declared zero selection/authority/unit/G-12
    counts are freeze-boundary facts; ``assert_v5_freeze_boundary`` enforces
    them when the manifest is built, and this verifier enforces that they were
    declared and that the bound predecessor bytes still authenticate.
    """

    transition = value.get("transition_from_v4")
    _require(isinstance(transition, Mapping), "W10 v5 transition record is missing")
    predecessor_path = successor_path(root, epoch="v4")
    _require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "W10 v5 predecessor manifest is missing")
    raw = predecessor_path.read_bytes()
    predecessor = json.loads(raw)
    _require(transition.get("path") == W10_V4_SOURCE_PATH, "W10 v5 predecessor path differs")
    _require(
        transition.get("manifest_kind") == W10_V4_MANIFEST_KIND
        and predecessor.get("manifest_kind") == W10_V4_MANIFEST_KIND,
        "W10 v5 predecessor kind differs",
    )
    _require(
        transition.get("manifest_id") == predecessor.get("manifest_id") == W10_V4_MANIFEST_ID,
        "W10 v5 predecessor ID differs",
    )
    _require(
        transition.get("sha256") == hashlib.sha256(raw).hexdigest() == W10_V4_MANIFEST_SHA256,
        "W10 v5 predecessor bytes differ",
    )
    _require(
        transition.get("source_commit") == predecessor.get("source_commit") == W10_V4_SOURCE_COMMIT,
        "W10 v5 predecessor source commit differs",
    )
    _require(transition.get("transition_kind") == W10_V5_TRANSITION_KIND, "W10 v5 transition kind differs")
    _require(transition.get("repair_reason") == W10_V5_REPAIR_REASON, "W10 v5 repair reason differs")
    _require(
        transition.get("predecessor_papr_constrained_training_runs") == 1,
        "W10 v5 must record exactly one closed PAPR training lifecycle",
    )
    _require(
        transition.get("jpeg_validation_selection_count") == 0
        and transition.get("er12_validation_selection_count") == 0,
        "W10 v5 must record a pre-selection freeze",
    )
    _require(
        transition.get("w10_authority_frozen") is False
        and transition.get("w10_scientific_units") == 0
        and transition.get("g12_freeze_manifest") is False
        and transition.get("test") == "SEALED"
        and transition.get("test_access") == 0,
        "W10 v5 transition is not pre-future-work",
    )
    completion_path = root / PAPR_COMPLETION_SOURCE_PATH
    _require(completion_path.is_file() and not completion_path.is_symlink(), "W10 v5 PAPR completion evidence is missing")
    completion_raw = completion_path.read_bytes()
    completion = json.loads(completion_raw)
    completion_body = dict(completion)
    completion_identifier = completion_body.pop("completion_id", None)
    _require(
        completion_identifier == PAPR_COMPLETION_PREFIX + canonical_sha256(completion_body),
        "W10 v5 PAPR completion ID differs",
    )
    _require(
        transition.get("predecessor_papr_completion_path") == PAPR_COMPLETION_SOURCE_PATH,
        "W10 v5 PAPR completion path differs",
    )
    _require(
        completion_identifier == transition.get("predecessor_papr_completion_id") == PAPR_COMPLETION_ID,
        "W10 v5 PAPR completion ID binding differs",
    )
    _require(
        transition.get("predecessor_papr_completion_sha256") == hashlib.sha256(completion_raw).hexdigest() == PAPR_COMPLETION_SHA256,
        "W10 v5 PAPR completion bytes differ",
    )
    _require(completion.get("training_runs") == 1, "W10 v5 PAPR completion run count differs")
    _require(
        completion.get("test") == "SEALED" and completion.get("test_access") == 0,
        "W10 v5 PAPR completion crossed the test boundary",
    )
    _require(completion.get("epochs") == PAPR_EPOCH_COUNT, "W10 v5 PAPR completion epoch count differs")
    _require(
        completion.get("source_manifest") == source_record(root, predecessor, path=predecessor_path)
        and completion.get("scientific_source_commit") == W10_V4_SOURCE_COMMIT,
        "W10 v5 PAPR completion source epoch differs",
    )


def _require_w10_v6_transition(root: Path, value: Mapping[str, Any]) -> None:
    """Authenticate v6's exact v5 predecessor and completed JPEG carrier."""

    from evaluation.w10_jpeg_carrier import (
        JPEG_CARRIER_DESCRIPTOR_PATH,
        JPEG_CARRIER_PATH,
        JPEG_SELECTION_CONTRACT_SHA256,
        JPEG_SELECTION_ID,
        JPEG_SELECTION_RAW_BYTES,
        JPEG_SELECTION_RAW_SHA256,
        JPEG_SELECTION_SOURCE_RECORD,
        check_jpeg_carrier,
    )

    transition = value.get("transition_from_v5")
    _require(isinstance(transition, Mapping), "W10 v6 transition record is missing")
    predecessor_path = successor_path(root, epoch="v5")
    _require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "W10 v6 predecessor manifest is missing")
    predecessor_raw = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_raw)
    _require(transition.get("path") == W10_V5_SOURCE_PATH, "W10 v6 predecessor path differs")
    _require(
        transition.get("manifest_kind") == W10_V5_MANIFEST_KIND
        and predecessor.get("manifest_kind") == W10_V5_MANIFEST_KIND,
        "W10 v6 predecessor kind differs",
    )
    _require(
        transition.get("manifest_id") == predecessor.get("manifest_id") == W10_V5_MANIFEST_ID,
        "W10 v6 predecessor ID differs",
    )
    _require(
        transition.get("sha256") == hashlib.sha256(predecessor_raw).hexdigest() == W10_V5_MANIFEST_SHA256,
        "W10 v6 predecessor bytes differ",
    )
    _require(
        transition.get("source_commit") == predecessor.get("source_commit") == W10_V5_SOURCE_COMMIT,
        "W10 v6 predecessor source commit differs",
    )
    _require(transition.get("transition_kind") == W10_V6_TRANSITION_KIND, "W10 v6 transition kind differs")
    _require(transition.get("repair_reasons") == list(W10_V6_REPAIR_REASONS), "W10 v6 repair reasons differ")
    _require(transition.get("jpeg_validation_selection_count") == 1, "W10 v6 JPEG selection count differs")
    _require(transition.get("jpeg_validation_selection_closed") is True, "W10 v6 JPEG selection is not closed")
    _require(transition.get("er12_validation_selection_count") == 0, "W10 v6 ER-12 selection count differs")
    _require(transition.get("w10_authority_frozen") is False, "W10 v6 authority state differs")
    _require(transition.get("w10_scientific_units") == 0, "W10 v6 scientific unit count differs")
    _require(transition.get("g12_freeze_manifest") is False, "W10 v6 G-12 state differs")
    _require(transition.get("test") == "SEALED" and transition.get("test_access") == 0, "W10 v6 transition crossed the test boundary")
    loaded = check_jpeg_carrier(root)
    provenance = loaded.provenance
    _require(loaded.value.get("selection_id") == JPEG_SELECTION_ID, "W10 v6 JPEG selection ID differs")
    _require(loaded.value.get("contract_sha256") == JPEG_SELECTION_CONTRACT_SHA256, "W10 v6 JPEG contract differs")
    _require(loaded.value.get("source_epoch") == JPEG_SELECTION_SOURCE_RECORD, "W10 v6 JPEG source epoch differs")
    _require(provenance.get("raw_sha256") == JPEG_SELECTION_RAW_SHA256, "W10 v6 JPEG raw SHA differs")
    _require(provenance.get("raw_bytes") == JPEG_SELECTION_RAW_BYTES, "W10 v6 JPEG raw byte count differs")
    _require(provenance.get("carrier_path") == JPEG_CARRIER_PATH, "W10 v6 JPEG carrier path differs")
    _require(provenance.get("carrier_descriptor_path") == JPEG_CARRIER_DESCRIPTOR_PATH, "W10 v6 JPEG descriptor path differs")
    for field, expected in (
        ("jpeg_validation_selection_id", JPEG_SELECTION_ID),
        ("jpeg_validation_selection_contract_sha256", JPEG_SELECTION_CONTRACT_SHA256),
        ("jpeg_validation_selection_raw_sha256", JPEG_SELECTION_RAW_SHA256),
        ("jpeg_validation_selection_raw_bytes", JPEG_SELECTION_RAW_BYTES),
        ("jpeg_validation_selection_source_epoch", JPEG_SELECTION_SOURCE_RECORD),
        ("jpeg_validation_selection_carrier_path", JPEG_CARRIER_PATH),
        ("jpeg_validation_selection_carrier_sha256", provenance.get("carrier_sha256")),
        ("jpeg_validation_selection_carrier_bytes", provenance.get("carrier_bytes")),
        ("jpeg_validation_selection_carrier_descriptor_path", JPEG_CARRIER_DESCRIPTOR_PATH),
        ("jpeg_validation_selection_carrier_descriptor_id", provenance.get("carrier_descriptor_id")),
        ("jpeg_validation_selection_carrier_descriptor_sha256", provenance.get("carrier_descriptor_sha256")),
    ):
        _require(transition.get(field) == expected, f"W10 v6 JPEG binding differs: {field}")


def _v7_jpeg_binding(loaded: Any) -> dict[str, Any]:
    provenance = loaded.provenance
    return {
        "jpeg_validation_selection_id": loaded.value.get("selection_id"),
        "jpeg_validation_selection_contract_sha256": loaded.value.get("contract_sha256"),
        "jpeg_validation_selection_raw_sha256": provenance.get("raw_sha256"),
        "jpeg_validation_selection_raw_bytes": provenance.get("raw_bytes"),
        "jpeg_validation_selection_source_epoch": loaded.value.get("source_epoch"),
        "jpeg_validation_selection_carrier_path": provenance.get("carrier_path"),
        "jpeg_validation_selection_carrier_sha256": provenance.get("carrier_sha256"),
        "jpeg_validation_selection_carrier_bytes": provenance.get("carrier_bytes"),
        "jpeg_validation_selection_carrier_descriptor_path": provenance.get("carrier_descriptor_path"),
        "jpeg_validation_selection_carrier_descriptor_id": provenance.get("carrier_descriptor_id"),
        "jpeg_validation_selection_carrier_descriptor_sha256": provenance.get("carrier_descriptor_sha256"),
    }


def _require_w10_v7_transition(root: Path, value: Mapping[str, Any]) -> None:
    """Authenticate v7 against exact v6 bytes and independently verify the JPEG carrier."""

    from evaluation.w10_jpeg_carrier import (
        JPEG_SELECTION_CONTRACT_SHA256,
        JPEG_SELECTION_ID,
        JPEG_SELECTION_RAW_BYTES,
        JPEG_SELECTION_RAW_SHA256,
        JPEG_SELECTION_SOURCE_RECORD,
        check_jpeg_carrier,
    )

    transition = value.get("transition_from_v6")
    _require(isinstance(transition, Mapping), "W10 v7 transition record is missing")
    predecessor_path = successor_path(root, epoch="v6")
    _require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "W10 v7 predecessor manifest is missing")
    predecessor_raw = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_raw)
    _require(transition.get("path") == W10_V6_SOURCE_PATH, "W10 v7 predecessor path differs")
    _require(
        transition.get("manifest_kind") == W10_V6_MANIFEST_KIND
        and predecessor.get("manifest_kind") == W10_V6_MANIFEST_KIND,
        "W10 v7 predecessor kind differs",
    )
    _require(
        transition.get("manifest_id") == predecessor.get("manifest_id") == W10_V6_MANIFEST_ID,
        "W10 v7 predecessor ID differs",
    )
    _require(
        transition.get("sha256") == hashlib.sha256(predecessor_raw).hexdigest() == W10_V6_MANIFEST_SHA256,
        "W10 v7 predecessor bytes differ",
    )
    _require(
        transition.get("source_commit") == predecessor.get("source_commit") == W10_V6_SOURCE_COMMIT,
        "W10 v7 predecessor source commit differs",
    )
    predecessor_transition = predecessor.get("transition_from_v5")
    _require(isinstance(predecessor_transition, Mapping), "W10 v7 frozen v6 JPEG transition is missing")
    loaded = check_jpeg_carrier(root)
    _require(loaded.value.get("selection_id") == JPEG_SELECTION_ID, "W10 v7 JPEG selection ID differs")
    _require(loaded.value.get("contract_sha256") == JPEG_SELECTION_CONTRACT_SHA256, "W10 v7 JPEG contract differs")
    _require(loaded.value.get("source_epoch") == JPEG_SELECTION_SOURCE_RECORD, "W10 v7 JPEG historical source differs")
    _require(loaded.provenance.get("raw_sha256") == JPEG_SELECTION_RAW_SHA256, "W10 v7 JPEG raw SHA differs")
    _require(loaded.provenance.get("raw_bytes") == JPEG_SELECTION_RAW_BYTES, "W10 v7 JPEG raw byte count differs")
    actual_binding = _v7_jpeg_binding(loaded)
    frozen_binding = {field: predecessor_transition.get(field) for field in _W10_V7_JPEG_BINDING_FIELDS}
    _require(actual_binding == frozen_binding, "W10 v7 JPEG carrier identity differs from frozen v6")
    expected_transition = {
        "path": W10_V6_SOURCE_PATH,
        "manifest_kind": W10_V6_MANIFEST_KIND,
        "manifest_id": W10_V6_MANIFEST_ID,
        "sha256": W10_V6_MANIFEST_SHA256,
        "source_commit": W10_V6_SOURCE_COMMIT,
        "transition_kind": W10_V7_TRANSITION_KIND,
        "repair_reason": W10_V7_REPAIR_REASON,
        "predecessor_papr_constrained_training_runs": 1,
        "jpeg_validation_selection_count": 1,
        "jpeg_validation_selection_closed": True,
        "er12_validation_selection_count": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
        **frozen_binding,
    }
    _require(dict(transition) == expected_transition, "W10 v7 transition fields differ")


def _require_w10_v8_transition(root: Path, value: Mapping[str, Any]) -> None:
    """Authenticate v8 against the exact failed-startup v7 predecessor."""

    transition = value.get("transition_from_v7")
    _require(isinstance(transition, Mapping), "W10 v8 transition record is missing")
    predecessor_path = successor_path(root, epoch="v7")
    _require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "W10 v8 predecessor manifest is missing")
    predecessor_raw = predecessor_path.read_bytes()
    predecessor = load_w10_manifest(root, live=False, epoch="v7")
    _require(transition.get("path") == W10_V7_SOURCE_PATH, "W10 v8 predecessor path differs")
    _require(transition.get("manifest_kind") == W10_V7_MANIFEST_KIND, "W10 v8 predecessor kind differs")
    _require(
        transition.get("manifest_id") == predecessor.get("manifest_id") == W10_V7_MANIFEST_ID,
        "W10 v8 predecessor ID differs",
    )
    _require(
        transition.get("sha256") == hashlib.sha256(predecessor_raw).hexdigest() == W10_V7_MANIFEST_SHA256,
        "W10 v8 predecessor bytes differ",
    )
    _require(
        transition.get("source_commit") == predecessor.get("source_commit") == W10_V7_SOURCE_COMMIT,
        "W10 v8 predecessor source commit differs",
    )
    expected_transition = {
        "path": W10_V7_SOURCE_PATH,
        "manifest_kind": W10_V7_MANIFEST_KIND,
        "manifest_id": W10_V7_MANIFEST_ID,
        "sha256": W10_V7_MANIFEST_SHA256,
        "source_commit": W10_V7_SOURCE_COMMIT,
        "transition_kind": W10_V8_TRANSITION_KIND,
        "repair_reason": W10_V8_REPAIR_REASON,
        "predecessor_papr_constrained_training_runs": 1,  # literal-ok: one closed historical PAPR lifecycle
        "jpeg_validation_selection_count": 1,  # literal-ok: one closed historical JPEG selection
        "jpeg_validation_selection_closed": True,
        "er12_validation_selection_count": 0,  # literal-ok: no completed ER-12 selection
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,  # literal-ok: W10 rehearsal remains unopened
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,  # literal-ok: sealed test boundary
    }
    _require(dict(transition) == expected_transition, "W10 v8 transition fields differ")


def v8_continuation_custody(root: Path) -> dict[str, Any]:
    """Authenticate immutable published facts before a v9 source freeze."""

    root = Path(root)
    source_path = root / W10_V8_SOURCE_PATH
    authority_path = root / W10_AUTHORITY_SOURCE_PATH
    custody_path = root / W10_V8_CUSTODY_PATH
    for path in (source_path, authority_path, custody_path):
        _require(path.is_file() and not path.is_symlink(), f"W10 v8 custody input is missing or unsafe: {path}")
    source = load_w10_manifest(root, live=False, epoch="v8")
    _require(source["manifest_id"] == W10_V8_MANIFEST_ID and hashlib.sha256(source_path.read_bytes()).hexdigest() == W10_V8_MANIFEST_SHA256, "W10 v8 source bytes differ")
    authority = json.loads(authority_path.read_bytes())
    authority_body = dict(authority)
    authority_id = authority_body.pop("authority_id", None)
    _require(authority_id == W10_V8_AUTHORITY_ID == "w10rehearsalauth-" + canonical_sha256(authority_body), "W10 v8 authority ID differs")
    _require(hashlib.sha256(authority_path.read_bytes()).hexdigest() == W10_V8_AUTHORITY_SHA256, "W10 v8 authority bytes differ")
    _require(authority.get("source_binding") == source and authority.get("source_commit") == source["source_commit"], "W10 v8 authority source differs")
    _require(authority.get("validation_only") is True and authority.get("test") == "SEALED" and authority.get("test_access") == 0, "W10 v8 authority test boundary differs")
    custody = json.loads(custody_path.read_bytes())
    custody_body = dict(custody)
    custody_id = custody_body.pop("custody_id", None)
    _require(custody_id == W10_V8_CUSTODY_ID == "w10v8prefix-" + canonical_sha256(custody_body), "W10 v8 custody ID differs")
    _require(hashlib.sha256(custody_path.read_bytes()).hexdigest() == W10_V8_CUSTODY_SHA256, "W10 v8 custody bytes differ")
    _require(custody.get("authority", {}).get("authority_id") == authority_id and custody.get("authority", {}).get("sha256") == W10_V8_AUTHORITY_SHA256, "W10 v8 custody authority differs")
    _require(custody.get("source", {}).get("manifest_id") == source["manifest_id"] and custody.get("source", {}).get("sha256") == W10_V8_MANIFEST_SHA256, "W10 v8 custody source differs")
    _require(custody.get("execution_commit") == W10_V8_EXECUTION_COMMIT and custody.get("status") == "FAILED_PARTIAL_NOT_CLOSEOUT", "W10 v8 failed execution differs")
    _require(custody.get("completed_ordinals") == [0, 125] and custody.get("failed_ordinal") == 126 and custody.get("unit_count") == 126 and custody.get("stream_count") == 168, "W10 v8 prefix frontier differs")
    units = custody.get("units")
    streams = custody.get("streams")
    _require(isinstance(units, list) and [item.get("ordinal") for item in units] == list(range(126)), "W10 v8 ordered unit custody differs")
    _require(isinstance(streams, list) and len(streams) == 168 and all(isinstance(item.get("ordinal"), int) and 0 <= item["ordinal"] < 126 for item in streams), "W10 v8 ordered scorer custody differs")
    _require(len({item.get("path") for item in units}) == 126 and len({item.get("path") for item in streams}) == 168, "W10 v8 custody paths are not unique")
    _require(custody.get("complete_prefix_digest") == canonical_sha256({"units": units, "streams": streams}), "W10 v8 custody prefix digest does not recompute")
    _require(custody.get("complete_prefix_digest") == W10_V8_PREFIX_DIGEST, "W10 v8 prefix digest differs")
    _require(custody.get("validation_only") is True and custody.get("test") == "SEALED" and custody.get("test_access") == 0 and custody.get("g12_unopened") is True, "W10 v8 custody test boundary differs")
    return custody


def _v9_transition(root: Path) -> dict[str, Any]:
    custody = v8_continuation_custody(root)
    papr_path = Path(root) / PAPR_COMPLETION_SOURCE_PATH
    _require(papr_path.is_file() and not papr_path.is_symlink(), "W10 v9 PAPR completion is missing or unsafe")
    _require(hashlib.sha256(papr_path.read_bytes()).hexdigest() == PAPR_COMPLETION_SHA256, "W10 v9 PAPR completion bytes differ")
    papr_completion = json.loads(papr_path.read_bytes())
    _require(papr_completion.get("completion_id") == PAPR_COMPLETION_ID, "W10 v9 PAPR completion ID differs")
    authority = json.loads((Path(root) / W10_AUTHORITY_SOURCE_PATH).read_bytes())
    selections = {str(item["scope"]["role"]): item["selection"] for item in authority["bindings"]}
    for role, epoch in (("dec9_jpeg_secondary", "v5"), ("er12_label_upper_bound", "v8")):
        selection = selections[role]
        historical = load_w10_manifest(root, live=False, epoch=epoch)
        _require(selection["source_epoch"] == source_record(root, historical), f"W10 v9 {role} historical source epoch differs")
        if selection["carrier_path"] is None:
            path = Path(root) / selection["artifact"]["path"]
            _require(path.is_file() and not path.is_symlink(), f"W10 v9 {role} selection is missing or unsafe")
            _require(hashlib.sha256(path.read_bytes()).hexdigest() == selection["raw_sha256"], f"W10 v9 {role} selection bytes differ")
            continue
        for path_field, sha_field in (("carrier_path", "carrier_sha256"), ("carrier_descriptor_path", "carrier_descriptor_sha256")):
            path = Path(root) / selection[path_field]
            _require(path.is_file() and not path.is_symlink(), f"W10 v9 {role} carrier is missing or unsafe")
            _require(hashlib.sha256(path.read_bytes()).hexdigest() == selection[sha_field], f"W10 v9 {role} carrier bytes differ")
    return {
        "path": W10_V8_SOURCE_PATH,
        "manifest_kind": W10_V8_MANIFEST_KIND,
        "manifest_id": W10_V8_MANIFEST_ID,
        "sha256": W10_V8_MANIFEST_SHA256,
        "source_commit": custody["source"]["source_commit"],
        "transition_kind": "failed_partial_science_to_suffix_continuation",
        "v8_scientific_work_executed": True,
        "v8_authority_id": W10_V8_AUTHORITY_ID,
        "v8_authority_sha256": W10_V8_AUTHORITY_SHA256,
        "v8_execution_commit": W10_V8_EXECUTION_COMMIT,
        "historical_custody_path": W10_V8_CUSTODY_PATH,
        "historical_custody_id": W10_V8_CUSTODY_ID,
        "historical_custody_sha256": W10_V8_CUSTODY_SHA256,
        "historical_prefix_digest": W10_V8_PREFIX_DIGEST,
        "completed_ordinals": [0, 125],
        "failed_ordinal": 126,
        "papr_completion_id": PAPR_COMPLETION_ID,
        "papr_completion_sha256": PAPR_COMPLETION_SHA256,
        "jpeg_selection_id": selections["dec9_jpeg_secondary"]["selection_id"],
        "jpeg_selection_source_epoch": selections["dec9_jpeg_secondary"]["source_epoch"],
        "jpeg_selection_raw_sha256": selections["dec9_jpeg_secondary"]["raw_sha256"],
        "er12_selection_id": selections["er12_label_upper_bound"]["selection_id"],
        "er12_selection_source_epoch": selections["er12_label_upper_bound"]["source_epoch"],
        "er12_selection_raw_sha256": selections["er12_label_upper_bound"]["raw_sha256"],
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }


def _require_w10_v9_transition(root: Path, value: Mapping[str, Any]) -> None:
    _require(dict(value.get("transition_from_v8", {})) == _v9_transition(root), "W10 v9 transition differs from immutable v8 custody")


def _v6_forbidden_paths(root: Path) -> tuple[tuple[str, str], ...]:
    return (
        (W10_ER12_SELECTION_SOURCE_PATH, "ER-12 validation selection"),
        (W10_AUTHORITY_SOURCE_PATH, "W10 rehearsal authority"),
        (W10_CLOSEOUT_SOURCE_PATH, "W10 rehearsal closeout"),
        (W10_RUNTIME_SOURCE_PATH, "W10 rehearsal runtime"),
        (G12_FREEZE_MANIFEST_SOURCE_PATH, "G-12 test freeze manifest"),
    )


def assert_v7_freeze_boundary(root: Path) -> Any:
    """Require the carrier-only, post-JPEG/pre-ER-12 frontier before v7."""

    from evaluation.w10_jpeg_carrier import JPEG_SELECTION_PATH, check_jpeg_carrier

    root = Path(root)
    for relative, label in (
        (W10_ER12_SELECTION_SOURCE_PATH, "ER-12 validation selection"),
        (W10_AUTHORITY_SOURCE_PATH, "W10 rehearsal authority"),
        (W10_CLOSEOUT_SOURCE_PATH, "W10 rehearsal closeout"),
        (W10_RUNTIME_SOURCE_PATH, "W10 rehearsal runtime"),
        (G12_FREEZE_MANIFEST_SOURCE_PATH, "G-12 test freeze manifest"),
    ):
        path = root / relative
        _require(not path.exists() and not path.is_symlink(), f"{label} already exists at the successor-v7 freeze boundary")
    raw_selection = root / JPEG_SELECTION_PATH
    _require(
        not raw_selection.exists() and not raw_selection.is_symlink(),
        "raw JPEG selection JSON must remain absent at the successor-v7 carrier-only boundary",
    )
    for relative in (
        PAPR_COMPLETION_SOURCE_PATH,
        "results/learned/w10/papr_selected_checkpoint.json",
        "results/learned/w10/papr_training_authorization.json",
    ):
        path = root / relative
        if path.exists() or path.is_symlink():
            _require(path.is_file() and not path.is_symlink(), f"v7 freeze boundary evidence is unsafe: {relative}")
            try:
                evidence = json.loads(path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SourceEpochHold(f"v7 freeze boundary evidence is corrupt: {relative}: {exc}") from None
            _require(evidence.get("test_access") == 0 and evidence.get("test") == "SEALED", f"v7 freeze boundary test access differs: {relative}")
    return check_jpeg_carrier(root)


def build_w10_manifest_v7(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the static-policy repair over exact v6 bytes after completed JPEG."""

    from evaluation.w10_jpeg_carrier import (
        JPEG_CARRIER_DESCRIPTOR_PATH,
        JPEG_CARRIER_PATH,
        JPEG_SELECTION_CONTRACT_SHA256,
        JPEG_SELECTION_ID,
        JPEG_SELECTION_RAW_BYTES,
        JPEG_SELECTION_RAW_SHA256,
        JPEG_SELECTION_SOURCE_RECORD,
    )

    root = Path(root)
    output_path = successor_path(root, epoch="v7")
    _require(not output_path.exists() and not output_path.is_symlink(), "W10 v7 manifest already exists at its freeze boundary")
    loaded = assert_v7_freeze_boundary(root)
    predecessor_path = successor_path(root, epoch="v6")
    _require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "W10 v7 requires the frozen v6 predecessor")
    predecessor_raw = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_raw)
    _require(predecessor.get("manifest_kind") == W10_V6_MANIFEST_KIND, "W10 v7 predecessor is not successor-v6")
    _require(predecessor.get("manifest_id") == W10_V6_MANIFEST_ID, "W10 v7 predecessor is not the exact v6 epoch")
    _require(hashlib.sha256(predecessor_raw).hexdigest() == W10_V6_MANIFEST_SHA256, "W10 v7 predecessor bytes differ")
    _require(predecessor.get("source_commit") == W10_V6_SOURCE_COMMIT, "W10 v7 predecessor source commit differs")
    predecessor_transition = predecessor.get("transition_from_v5")
    _require(isinstance(predecessor_transition, Mapping), "W10 v7 frozen v6 JPEG transition is missing")
    actual_binding = _v7_jpeg_binding(loaded)
    frozen_binding = {field: predecessor_transition.get(field) for field in _W10_V7_JPEG_BINDING_FIELDS}
    _require(actual_binding == frozen_binding, "W10 v7 JPEG carrier identity differs from frozen v6")
    _require(
        loaded.value.get("selection_id") == JPEG_SELECTION_ID
        and loaded.value.get("contract_sha256") == JPEG_SELECTION_CONTRACT_SHA256
        and loaded.value.get("source_epoch") == JPEG_SELECTION_SOURCE_RECORD
        and loaded.provenance.get("raw_sha256") == JPEG_SELECTION_RAW_SHA256
        and loaded.provenance.get("raw_bytes") == JPEG_SELECTION_RAW_BYTES
        and loaded.provenance.get("carrier_path") == JPEG_CARRIER_PATH
        and loaded.provenance.get("carrier_descriptor_path") == JPEG_CARRIER_DESCRIPTOR_PATH,
        "W10 v7 completed JPEG carrier identity differs",
    )
    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V7_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    base["transition_from_v6"] = {
        "path": W10_V6_SOURCE_PATH,
        "manifest_kind": W10_V6_MANIFEST_KIND,
        "manifest_id": W10_V6_MANIFEST_ID,
        "sha256": W10_V6_MANIFEST_SHA256,
        "source_commit": W10_V6_SOURCE_COMMIT,
        "transition_kind": W10_V7_TRANSITION_KIND,
        "repair_reason": W10_V7_REPAIR_REASON,
        "predecessor_papr_constrained_training_runs": 1,
        "jpeg_validation_selection_count": 1,
        "jpeg_validation_selection_closed": True,
        "er12_validation_selection_count": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
        **actual_binding,
    }
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_future_work_state"] = dict(W10_V7_PRE_FUTURE_WORK_STATE)
    base["manifest_id"] = W10_V7_MANIFEST_PREFIX + canonical_sha256(base)
    return base


def assert_v8_freeze_boundary(root: Path) -> Any:
    """Require the honest post-failed-startup/pre-candidate ER-12 frontier."""

    from evaluation.w10_jpeg_carrier import JPEG_SELECTION_PATH, check_jpeg_carrier

    root = Path(root)
    for relative, label in (
        (W10_ER12_SELECTION_SOURCE_PATH, "ER-12 validation selection"),
        (W10_AUTHORITY_SOURCE_PATH, "W10 rehearsal authority"),
        (W10_CLOSEOUT_SOURCE_PATH, "W10 rehearsal closeout"),
        (W10_RUNTIME_SOURCE_PATH, "W10 rehearsal runtime"),
        (G12_FREEZE_MANIFEST_SOURCE_PATH, "G-12 test freeze manifest"),
    ):
        path = root / relative
        _require(
            not path.exists() and not path.is_symlink(),
            f"{label} already exists at the successor-v8 freeze boundary",
        )
    raw_selection = root / JPEG_SELECTION_PATH
    _require(
        not raw_selection.exists() and not raw_selection.is_symlink(),
        "raw JPEG selection JSON must remain absent at the successor-v8 carrier-backed boundary",
    )
    for relative in (
        PAPR_COMPLETION_SOURCE_PATH,
        "results/learned/w10/papr_selected_checkpoint.json",
        "results/learned/w10/papr_training_authorization.json",
    ):
        path = root / relative
        if path.exists() or path.is_symlink():
            _require(path.is_file() and not path.is_symlink(), f"v8 freeze boundary evidence is unsafe: {relative}")
            try:
                evidence = json.loads(path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SourceEpochHold(f"v8 freeze boundary evidence is corrupt: {relative}: {exc}") from None
            _require(evidence.get("test_access") == 0 and evidence.get("test") == "SEALED", f"v8 freeze boundary test access differs: {relative}")
    return check_jpeg_carrier(root)


def build_w10_manifest_v8(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the truthful post-v7-startup-failure source repair over exact v7."""

    root = Path(root)
    output_path = successor_path(root, epoch="v8")
    _require(not output_path.exists() and not output_path.is_symlink(), "W10 v8 manifest already exists at its freeze boundary")
    assert_v8_freeze_boundary(root)
    predecessor_path = successor_path(root, epoch="v7")
    _require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "W10 v8 requires the frozen v7 predecessor")
    predecessor_raw = predecessor_path.read_bytes()
    predecessor = load_w10_manifest(root, live=False, epoch="v7")
    _require(predecessor.get("manifest_kind") == W10_V7_MANIFEST_KIND, "W10 v8 predecessor is not successor-v7")
    _require(predecessor.get("manifest_id") == W10_V7_MANIFEST_ID, "W10 v8 predecessor is not the exact v7 epoch")
    _require(hashlib.sha256(predecessor_raw).hexdigest() == W10_V7_MANIFEST_SHA256, "W10 v8 predecessor bytes differ")
    _require(predecessor.get("source_commit") == W10_V7_SOURCE_COMMIT, "W10 v8 predecessor source commit differs")

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V8_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    base["transition_from_v7"] = {
        "path": W10_V7_SOURCE_PATH,
        "manifest_kind": W10_V7_MANIFEST_KIND,
        "manifest_id": W10_V7_MANIFEST_ID,
        "sha256": W10_V7_MANIFEST_SHA256,
        "source_commit": W10_V7_SOURCE_COMMIT,
        "transition_kind": W10_V8_TRANSITION_KIND,
        "repair_reason": W10_V8_REPAIR_REASON,
        "predecessor_papr_constrained_training_runs": 1,  # literal-ok: one closed historical PAPR lifecycle
        "jpeg_validation_selection_count": 1,  # literal-ok: one closed historical JPEG selection
        "jpeg_validation_selection_closed": True,
        "er12_validation_selection_count": 0,  # literal-ok: no completed ER-12 selection
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,  # literal-ok: W10 rehearsal remains unopened
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,  # literal-ok: sealed test boundary
    }
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_future_work_state"] = dict(W10_V8_PRE_FUTURE_WORK_STATE)
    base["manifest_id"] = W10_V8_MANIFEST_PREFIX + canonical_sha256(base)
    return base


def build_w10_manifest_v9(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build a future successor over failed, already executed v8 science."""

    root = Path(root)
    output = successor_path(root, epoch="v9")
    _require(not output.exists() and not output.is_symlink(), "W10 v9 manifest already exists")
    _require(not (root / W10_CLOSEOUT_SOURCE_PATH).exists(), "W10 closeout already exists")
    _require(not (root / G12_FREEZE_MANIFEST_SOURCE_PATH).exists(), "G12 is already open")
    _require(not (root / "results/learned/w10/w10_continuation_authorization_v9.json").exists(), "W10 v9 authority already exists")
    _require(not (root / "results/learned/w10/w10_continuation_launch_authorization_v9.json").exists(), "W10 v9 launch grant already exists")
    transition = _v9_transition(root)
    base = build_manifest(root, source_commit=source_commit, relevant_config_paths=W10_RELEVANT_CONFIG_PATHS)
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V9_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    base["transition_from_v8"] = transition
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_future_work_state"] = dict(W10_V9_PRE_FUTURE_WORK_STATE)
    base["manifest_id"] = W10_V9_MANIFEST_PREFIX + canonical_sha256(base)
    return base


def source_record(root: Path, manifest: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    if path is None:
        kind = str(manifest.get("manifest_kind"))
        candidates = {
            W10_MANIFEST_KIND: W10_SOURCE_PATH,
            W10_V2_MANIFEST_KIND: W10_V2_SOURCE_PATH,
            W10_V3_MANIFEST_KIND: W10_V3_SOURCE_PATH,
            W10_V4_MANIFEST_KIND: W10_V4_SOURCE_PATH,
            W10_V5_MANIFEST_KIND: W10_V5_SOURCE_PATH,
            W10_V6_MANIFEST_KIND: W10_V6_SOURCE_PATH,
            W10_V7_MANIFEST_KIND: W10_V7_SOURCE_PATH,
            W10_V8_MANIFEST_KIND: W10_V8_SOURCE_PATH,
            W10_V9_MANIFEST_KIND: W10_V9_SOURCE_PATH,
        }
        candidate = root / candidates[kind]
        if not candidate.is_file():
            _require(kind != W10_V9_MANIFEST_KIND, "W10 v9 source record requires its frozen manifest bytes")
            candidate = active_manifest_path(root)
        target = candidate
    else:
        target = path
    return {
        "path": str(target.relative_to(root)),
        "manifest_id": str(manifest["manifest_id"]),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def build_w10_manifest_v2(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the corrected AM-99 successor epoch at the exact clean HEAD it binds."""

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V2_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    v1_path = successor_path(Path(root), epoch="v1")
    if v1_path.is_file() and not v1_path.is_symlink():
        raw = v1_path.read_bytes()
        base["superseded_successor"] = {
            "path": W10_SOURCE_PATH,
            "manifest_kind": W10_MANIFEST_KIND,
            "manifest_id": json.loads(raw)["manifest_id"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_commit": json.loads(raw)["source_commit"],
            "superseded_before_science": True,
            "papr_constrained_training_runs": 0,
            "w10_scientific_units": 0,
            "test_access": 0,
        }
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_science_state"] = {
        "papr_constrained_training_runs": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }
    base["manifest_id"] = W10_V2_MANIFEST_PREFIX + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


def build_w10_manifest_v3(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the repaired successor-v3 epoch over immutable v2 bytes."""

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V3_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    v2_path = successor_path(Path(root), epoch="v2")
    _require(v2_path.is_file() and not v2_path.is_symlink(), "W10 v3 requires the frozen v2 predecessor")
    raw = v2_path.read_bytes()
    predecessor = json.loads(raw)
    base["superseded_successor"] = {
        "path": W10_V2_SOURCE_PATH,
        "manifest_kind": W10_V2_MANIFEST_KIND,
        "manifest_id": predecessor["manifest_id"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": predecessor["source_commit"],
        "superseded_before_science": True,
        "papr_constrained_training_runs": 0,
        "w10_scientific_units": 0,
        "test_access": 0,
    }
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_science_state"] = {
        "papr_constrained_training_runs": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }
    base["manifest_id"] = W10_V3_MANIFEST_PREFIX + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


def build_w10_manifest_v4(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build successor-v4 over the immutable zero-science v3 bytes."""

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V4_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    v3_path = successor_path(Path(root), epoch="v3")
    _require(v3_path.is_file() and not v3_path.is_symlink(), "W10 v4 requires the frozen v3 predecessor")
    raw = v3_path.read_bytes()
    predecessor = json.loads(raw)
    base["superseded_successor"] = {
        "path": W10_V3_SOURCE_PATH,
        "manifest_kind": W10_V3_MANIFEST_KIND,
        "manifest_id": predecessor["manifest_id"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": predecessor["source_commit"],
        "superseded_before_science": True,
        "papr_constrained_training_runs": 0,
        "w10_scientific_units": 0,
        "test_access": 0,
    }
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_science_state"] = {
        "papr_constrained_training_runs": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }
    base["manifest_id"] = W10_V4_MANIFEST_PREFIX + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


def assert_v5_freeze_boundary(root: Path) -> None:
    """Successor-v5 may only be frozen while the post-PAPR frontier is untouched.

    These are freeze-time facts, not live-verification requirements: once the
    JPEG/ER-12 selections and the W10 authority exist they are legitimate work
    under this same epoch, and the historical v5 transition record simply keeps
    saying that none of them existed when it froze.
    """

    root = Path(root)
    for relative, label in (
        (W10_JPEG_SELECTION_SOURCE_PATH, "JPEG-secondary validation selection"),
        (W10_ER12_SELECTION_SOURCE_PATH, "ER-12 validation selection"),
        (W10_AUTHORITY_SOURCE_PATH, "W10 rehearsal authority"),
        (W10_CLOSEOUT_SOURCE_PATH, "W10 rehearsal closeout"),
        (W10_RUNTIME_SOURCE_PATH, "W10 rehearsal runtime"),
        (G12_FREEZE_MANIFEST_SOURCE_PATH, "G-12 test freeze manifest"),
    ):
        path = root / relative
        _require(
            not path.exists() and not path.is_symlink(),
            f"{label} already exists at the successor-v5 freeze boundary",
        )


def build_w10_manifest_v5(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the post-PAPR/pre-selection repair of the closed v4 epoch.

    The predecessor is recorded as ``transition_from_v4``, deliberately not as
    ``superseded_successor``: v4 was *not* superseded before science.  It
    produced the one closed PAPR-constrained training lifecycle, and v5 records
    that truth while binding the exact immutable v4 bytes and the terminal PAPR
    completion evidence.
    """

    root = Path(root)
    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V5_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    predecessor_path = successor_path(root, epoch="v4")
    _require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "W10 v5 requires the frozen v4 predecessor")
    predecessor_raw = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_raw)
    _require(predecessor.get("manifest_kind") == W10_V4_MANIFEST_KIND, "W10 v5 predecessor is not the v4 epoch")
    _require(predecessor.get("manifest_id") == W10_V4_MANIFEST_ID, "W10 v5 predecessor is not the closed v4 epoch")
    _require(hashlib.sha256(predecessor_raw).hexdigest() == W10_V4_MANIFEST_SHA256, "W10 v5 predecessor bytes differ")
    _require(predecessor.get("source_commit") == W10_V4_SOURCE_COMMIT, "W10 v5 predecessor source commit differs")
    completion_path = root / PAPR_COMPLETION_SOURCE_PATH
    _require(completion_path.is_file() and not completion_path.is_symlink(), "W10 v5 requires the closed PAPR completion evidence")
    completion_raw = completion_path.read_bytes()
    completion = json.loads(completion_raw)
    completion_body = dict(completion)
    completion_identifier = completion_body.pop("completion_id", None)
    _require(
        completion_identifier == PAPR_COMPLETION_PREFIX + canonical_sha256(completion_body),
        "W10 v5 PAPR completion ID differs",
    )
    _require(completion_identifier == PAPR_COMPLETION_ID, "W10 v5 PAPR completion is not the closed lifecycle")
    _require(hashlib.sha256(completion_raw).hexdigest() == PAPR_COMPLETION_SHA256, "W10 v5 PAPR completion bytes differ")
    _require(
        completion.get("training_runs") == 1
        and completion.get("test") == "SEALED"
        and completion.get("test_access") == 0,
        "W10 v5 PAPR completion is not a closed validation-only lifecycle",
    )
    assert_v5_freeze_boundary(root)
    base["transition_from_v4"] = {
        "path": W10_V4_SOURCE_PATH,
        "manifest_kind": W10_V4_MANIFEST_KIND,
        "manifest_id": predecessor["manifest_id"],
        "sha256": hashlib.sha256(predecessor_raw).hexdigest(),
        "source_commit": predecessor["source_commit"],
        "transition_kind": W10_V5_TRANSITION_KIND,
        "predecessor_papr_constrained_training_runs": 1,
        "predecessor_papr_completion_path": PAPR_COMPLETION_SOURCE_PATH,
        "predecessor_papr_completion_id": completion["completion_id"],
        "predecessor_papr_completion_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "jpeg_validation_selection_count": 0,
        "er12_validation_selection_count": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
        "repair_reason": W10_V5_REPAIR_REASON,
    }
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_future_work_state"] = dict(W10_V5_PRE_FUTURE_WORK_STATE)
    base["manifest_id"] = W10_V5_MANIFEST_PREFIX + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


def assert_v6_freeze_boundary(root: Path) -> None:
    """Require the post-JPEG/pre-ER-12 frontier before freezing v6."""

    root = Path(root)
    for relative, label in _v6_forbidden_paths(root):
        path = root / relative
        _require(
            not path.exists() and not path.is_symlink(),
            f"{label} already exists at the successor-v6 freeze boundary",
        )
    for relative in (PAPR_COMPLETION_SOURCE_PATH, "results/learned/w10/papr_selected_checkpoint.json", "results/learned/w10/papr_training_authorization.json"):
        path = root / relative
        if path.is_file() and not path.is_symlink():
            try:
                value = json.loads(path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SourceEpochHold(f"v6 freeze boundary evidence is corrupt: {relative}: {exc}") from None
            _require(value.get("test_access") == 0 and value.get("test") == "SEALED", f"v6 freeze boundary test access differs: {relative}")


def build_w10_manifest_v6(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the post-JPEG/pre-ER-12 successor over exact v5 and carrier bytes."""

    from evaluation.w10_jpeg_carrier import (
        JPEG_CARRIER_DESCRIPTOR_PATH,
        JPEG_CARRIER_PATH,
        JPEG_SELECTION_CONTRACT_SHA256,
        JPEG_SELECTION_ID,
        JPEG_SELECTION_RAW_BYTES,
        JPEG_SELECTION_RAW_SHA256,
        JPEG_SELECTION_SOURCE_RECORD,
        check_jpeg_carrier,
    )

    root = Path(root)
    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V6_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    predecessor_path = successor_path(root, epoch="v5")
    _require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "W10 v6 requires the frozen v5 predecessor")
    predecessor_raw = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_raw)
    _require(predecessor.get("manifest_kind") == W10_V5_MANIFEST_KIND, "W10 v6 predecessor is not the v5 epoch")
    _require(predecessor.get("manifest_id") == W10_V5_MANIFEST_ID, "W10 v6 predecessor is not the exact v5 epoch")
    _require(hashlib.sha256(predecessor_raw).hexdigest() == W10_V5_MANIFEST_SHA256, "W10 v6 predecessor bytes differ")
    _require(predecessor.get("source_commit") == W10_V5_SOURCE_COMMIT, "W10 v6 predecessor source commit differs")
    assert_v6_freeze_boundary(root)
    loaded = check_jpeg_carrier(root)
    provenance = loaded.provenance
    predecessor_record = source_record(root, predecessor, path=predecessor_path)
    _require(loaded.value.get("source_epoch") == predecessor_record == JPEG_SELECTION_SOURCE_RECORD, "W10 v6 JPEG historical source differs")
    _require(loaded.value.get("selection_id") == JPEG_SELECTION_ID, "W10 v6 JPEG selection ID differs")
    _require(loaded.value.get("contract_sha256") == JPEG_SELECTION_CONTRACT_SHA256, "W10 v6 JPEG contract differs")
    transition = {
        "path": W10_V5_SOURCE_PATH,
        "manifest_kind": W10_V5_MANIFEST_KIND,
        "manifest_id": predecessor["manifest_id"],
        "sha256": hashlib.sha256(predecessor_raw).hexdigest(),
        "source_commit": predecessor["source_commit"],
        "transition_kind": W10_V6_TRANSITION_KIND,
        "repair_reasons": list(W10_V6_REPAIR_REASONS),
        "predecessor_papr_constrained_training_runs": 1,
        "jpeg_validation_selection_count": 1,
        "jpeg_validation_selection_closed": True,
        "jpeg_validation_selection_id": JPEG_SELECTION_ID,
        "jpeg_validation_selection_contract_sha256": JPEG_SELECTION_CONTRACT_SHA256,
        "jpeg_validation_selection_raw_sha256": JPEG_SELECTION_RAW_SHA256,
        "jpeg_validation_selection_raw_bytes": JPEG_SELECTION_RAW_BYTES,
        "jpeg_validation_selection_source_epoch": JPEG_SELECTION_SOURCE_RECORD,
        "jpeg_validation_selection_carrier_path": JPEG_CARRIER_PATH,
        "jpeg_validation_selection_carrier_sha256": provenance["carrier_sha256"],
        "jpeg_validation_selection_carrier_bytes": provenance["carrier_bytes"],
        "jpeg_validation_selection_carrier_descriptor_path": JPEG_CARRIER_DESCRIPTOR_PATH,
        "jpeg_validation_selection_carrier_descriptor_id": provenance["carrier_descriptor_id"],
        "jpeg_validation_selection_carrier_descriptor_sha256": provenance["carrier_descriptor_sha256"],
        "er12_validation_selection_count": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }
    base["transition_from_v5"] = transition
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_future_work_state"] = dict(W10_V6_PRE_FUTURE_WORK_STATE)
    base["manifest_id"] = W10_V6_MANIFEST_PREFIX + canonical_sha256(base)
    return base


def assert_clean_active_source_closure(root: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    """Live worktree closure against one manifest's own commit and trees."""

    root = Path(root).resolve()
    source_commit = str(authority.get("source_commit", ""))
    _require(len(source_commit) == 40, "active source authority has no full commit")  # literal-ok: SHA-1 commit length
    committed = committed_source_differences(root, source_commit)
    working = working_tree_source_differences(root)
    _require(not committed, f"protected committed source drift: {committed}")
    _require(not any(working.values()), f"protected working-tree source drift: {working}")
    _require(dict(authority.get("tree_hashes", {})) == git_tree_hashes(root, source_commit), "active source Git tree closure differs")
    lock_path = root / "requirements-pascal.lock"
    _require(lock_path.is_file(), "Pascal lock is missing")
    _require(
        hashlib.sha256(lock_path.read_bytes()).hexdigest() == authority.get("requirements_pascal_lock_sha256"),
        "Pascal lock SHA-256 differs from the active authority",
    )
    return {"committed": committed, **working}


def assert_successor_lineage(
    root: Path,
    historical: Mapping[str, Any],
    *,
    successor: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Prove the active successor chain reaches one exact historical epoch.

    Every successor names its predecessor's manifest ID, kind, SHA-256 and
    source commit.  Walking the chain makes the current epoch answerable to the
    historical one instead of merely declaring it, and the predecessor bytes are
    re-hashed from the worktree rather than trusted from the record.
    """

    current = dict(successor) if successor is not None else load_w10_manifest(root, live=True)
    target = str(historical.get("manifest_id", ""))
    _require(bool(target), "historical epoch has no manifest ID")
    seen: set[str] = set()
    for _ in range(len(W10_EPOCHS)):
        if str(current.get("manifest_id")) == target:
            return current
        identifier = str(current.get("manifest_id"))
        _require(identifier not in seen, "active successor lineage contains a cycle")
        seen.add(identifier)
        predecessor = predecessor_binding(current)
        _require(isinstance(predecessor, Mapping), "active successor does not declare its predecessor epoch")
        if str(predecessor.get("manifest_id")) == target:
            epoch = epoch_for_kind(str(predecessor.get("manifest_kind")))
            path = successor_path(root, epoch=epoch)
            _require(path.is_file() and not path.is_symlink(), "historical successor manifest is missing")
            _require(
                predecessor.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest(),
                "historical successor manifest bytes differ",
            )
            _require(predecessor.get("source_commit") == historical.get("source_commit"), "historical successor source commit differs")
            return current
        epoch = epoch_for_kind(str(predecessor.get("manifest_kind")))
        current = load_w10_manifest(root, live=False, epoch=epoch)
    raise SourceEpochHold("active successor lineage does not reach the historical epoch")


def assert_active_epoch_closure(root: Path, historical: Mapping[str, Any]) -> None:
    """Closed-authority closure: historical bytes plus live active closure.

    The historical manifest promised that no protected source would change after
    its commit.  A successor epoch is exactly that permitted change, so the live
    check moves to the active successor while the historical manifest is still
    authenticated against its own commit and never rewritten.  The active
    successor must also prove an authenticated lineage back to the historical
    epoch: a bare path mention is not enough.
    """

    assert_manifest_commit_bytes(root, historical)
    active = active_manifest_path(root)
    if not (active.is_file() and not active.is_symlink()):
        from runtime.source_guard import assert_clean_source_closure  # noqa: PLC0415

        assert_clean_source_closure(root, historical)
        return
    successor = load_w10_manifest(root, live=True)
    if str(historical.get("manifest_id")) == str(successor.get("manifest_id")):
        # The historical epoch is itself the active one; its live closure was
        # just authenticated by ``load_w10_manifest``.
        return
    if str(historical.get("manifest_kind", "")).startswith("W10_PREPARATORY_SOURCE_SUCCESSOR"):
        assert_successor_lineage(root, historical, successor=successor)
        return
    binding = successor.get("historical_w9_downstream_manifest")
    _require(
        isinstance(binding, Mapping) and binding.get("path") == HISTORICAL_SOURCE_PATH,
        "active successor does not name the historical epoch",
    )
    _require(binding.get("manifest_id") == historical.get("manifest_id"), "active successor does not bind the historical manifest ID")
    _require(binding.get("source_commit") == historical.get("source_commit"), "active successor does not bind the historical source commit")


__all__ = [
    "HISTORICAL_SOURCE_COMMIT",
    "HISTORICAL_SOURCE_PATH",
    "W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES",
    "W10_GOVERNS",
    "W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES",
    "W10_MANIFEST_KIND",
    "W10_MANIFEST_KINDS",
    "W10_MANIFEST_PREFIX",
    "W10_RELEVANT_CONFIG_PATHS",
    "W10_SOURCE_PATH",
    "W10_V2_MANIFEST_KIND",
    "W10_V2_MANIFEST_PREFIX",
    "W10_V2_SOURCE_PATH",
    "W10_V3_MANIFEST_KIND",
    "W10_V3_MANIFEST_PREFIX",
    "W10_V3_SOURCE_PATH",
    "W10_V4_MANIFEST_KIND",
    "W10_V4_MANIFEST_ID",
    "W10_V4_MANIFEST_PREFIX",
    "W10_V4_MANIFEST_SHA256",
    "W10_V4_SOURCE_PATH",
    "W10_V4_SOURCE_COMMIT",
    "W10_V5_MANIFEST_KIND",
    "W10_V5_MANIFEST_ID",
    "W10_V5_MANIFEST_SHA256",
    "W10_V5_MANIFEST_PREFIX",
    "W10_V5_PRE_FUTURE_WORK_STATE",
    "W10_V5_REPAIR_REASON",
    "W10_V5_SOURCE_PATH",
    "W10_V5_SOURCE_COMMIT",
    "W10_V5_TRANSITION_KIND",
    "W10_V6_MANIFEST_KIND",
    "W10_V6_MANIFEST_ID",
    "W10_V6_MANIFEST_SHA256",
    "W10_V6_SOURCE_COMMIT",
    "W10_V6_MANIFEST_PREFIX",
    "W10_V6_PRE_FUTURE_WORK_STATE",
    "W10_V6_REPAIR_REASONS",
    "W10_V6_SOURCE_PATH",
    "W10_V6_TRANSITION_KIND",
    "W10_V7_MANIFEST_KIND",
    "W10_V7_MANIFEST_ID",
    "W10_V7_MANIFEST_PREFIX",
    "W10_V7_MANIFEST_SHA256",
    "W10_V7_PRE_FUTURE_WORK_STATE",
    "W10_V7_REPAIR_REASON",
    "W10_V7_SOURCE_COMMIT",
    "W10_V7_SOURCE_PATH",
    "W10_V7_TRANSITION_KIND",
    "W10_V8_MANIFEST_KIND",
    "W10_V8_MANIFEST_PREFIX",
    "W10_V8_PRE_FUTURE_WORK_STATE",
    "W10_V8_REPAIR_REASON",
    "W10_V8_SOURCE_PATH",
    "W10_V8_TRANSITION_KIND",
    "W10_V8_AUTHORITY_ID",
    "W10_V8_AUTHORITY_SHA256",
    "W10_V8_CUSTODY_ID",
    "W10_V8_CUSTODY_PATH",
    "W10_V8_CUSTODY_SHA256",
    "W10_V8_EXECUTION_COMMIT",
    "W10_V8_MANIFEST_ID",
    "W10_V8_MANIFEST_SHA256",
    "W10_V8_PREFIX_DIGEST",
    "W10_V9_MANIFEST_KIND",
    "W10_V9_MANIFEST_PREFIX",
    "W10_V9_PRE_FUTURE_WORK_STATE",
    "W10_V9_SOURCE_PATH",
    "SourceEpochHold",
    "active_manifest_path",
    "assert_active_epoch_closure",
    "assert_clean_active_source_closure",
    "assert_successor_lineage",
    "assert_v6_freeze_boundary",
    "assert_v7_freeze_boundary",
    "assert_v8_freeze_boundary",
    "assert_v5_freeze_boundary",
    "assert_w10_manifest_contract",
    "build_w10_manifest",
    "build_w10_manifest_v2",
    "build_w10_manifest_v3",
    "build_w10_manifest_v4",
    "build_w10_manifest_v5",
    "build_w10_manifest_v6",
    "build_w10_manifest_v7",
    "build_w10_manifest_v8",
    "build_w10_manifest_v9",
    "epoch_for_kind",
    "load_w10_manifest",
    "manifest_prefix",
    "predecessor_binding",
    "source_record",
    "successor_path",
    "v8_continuation_custody",
]
