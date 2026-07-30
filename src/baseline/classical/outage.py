"""Frozen constant-class outage policy, and its keyed sensitivity variant.

An outage is a row the classical arm could not deliver: no legal packetisation
exists, JPEG 2000 could not emit a codestream inside the payload budget, or a
CRC failed after transmission.  Such a row still has to be *scored* — dropping
it would quietly inflate the baseline's accuracy — so the receiver emits a
prediction without ever seeing a reconstruction.

``params.baseline.outage_policy`` fixes that prediction as a single constant
class, chosen on the **validation** split and frozen before test is ever opened.
The two things this module refuses to do are worth stating plainly:

* it never assumes ``1 / n_classes``.  The outage accuracy is the *measured*
  frequency of the selected label in the committed validation manifest, written
  as an explicit ``numerator / denominator``.  For the datasets configured here
  that ratio happens to equal ``1 / n_classes`` exactly, because
  ``data.manifests`` *enforces* an exactly stratified validation split — but it
  is derived from counts, and a manifest whose stratification changed would move
  it.  Hardcoding the theoretical value would silently survive that change;
* it never reselects the class at inference time.  Selection reads labels;
  prediction does not.  The selection result is frozen into a committed artifact
  and the runtime policy only loads and verifies it.

The configured sensitivity variant, ``keyed_uniform_random_label``, draws one
label per row from the central counter-based Philox stream under the declared
``outage_label`` identity, so it is a pure function of its key: row order, batch
size, how many other rows outage, and intervening draws in either arm cannot
move it.  It is a secondary comparison for later G-8 work, never the primary
outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifacts.rng import keyed_generator
from config.params import REPO_ROOT, get
from config.run_config import canonical_sha256
from data.manifests import manifest_path, validate_manifest_bytes

#: The split whose labels may be counted.  ``data.manifests`` spells the
#: validation split ``val``; ``test`` is never read here at any point.
SELECTION_SPLIT = "val"

#: Every W4 evidence artifact repeats these four claims verbatim, in a
#: machine-readable field and in prose, so no reader can mistake bounded
#: plumbing evidence for the BR-4 sweep or a G-8 selection.
EVIDENCE_LABELS = (
    "bounded validation/plumbing integration",
    "not the BR-4 full validation sweep",
    "not a G-8 operating-point selection",
    "not test evidence",
)

#: Implemented policy names.  A change to any of these in the spec must break
#: this module rather than silently reinterpret the frozen artifact.
_POLICY = "frozen_constant_class_selected_on_validation"
_SELECTION = "highest_validation_accuracy_constant_prediction"
_TIE_BREAK = "lowest_class_index"
_SENSITIVITY = "keyed_uniform_random_label"
_RNG_PURPOSE = "outage_label"

OUTAGE_POLICY_SCHEMA_VERSION = 1

#: Non-outage rows carry this in the ``outage_reason`` column.  The schema has
#: no dedicated sentinel, so the repository's canonical "not applicable" for a
#: string column — the empty JSON/CSV value — is used and recorded in the
#: field-semantics table.
NOT_APPLICABLE = None


class OutagePolicyError(RuntimeError):
    """A frozen-outage contract violation, never a link outcome."""


def _require_configured_policy() -> None:
    """Fail closed if the spec no longer describes the implemented policy."""

    actual = {
        "baseline.outage_policy": get("baseline.outage_policy"),
        "baseline.outage_class_selection": get("baseline.outage_class_selection"),
        "baseline.outage_class_tie_break": get("baseline.outage_class_tie_break"),
        "baseline.outage_policy_sensitivity": get(
            "baseline.outage_policy_sensitivity"
        ),
    }
    expected = {
        "baseline.outage_policy": _POLICY,
        "baseline.outage_class_selection": _SELECTION,
        "baseline.outage_class_tie_break": _TIE_BREAK,
        "baseline.outage_policy_sensitivity": _SENSITIVITY,
    }
    for path, value in expected.items():
        if actual[path] != value:
            raise NotImplementedError(
                f"unsupported params.{path}: {actual[path]!r}, "
                f"this module implements {value!r}"
            )


def configured_rng_identity_fields() -> tuple[str, ...]:
    """The declared ``outage_label`` identity, minus the purpose marker.

    ``params.baseline.outage_rng_key`` spells the purpose as the pseudo-field
    ``purpose=outage_label``.  The central ``keyed_generator`` takes the purpose
    as its first argument and rejects it inside the identity, so the marker is
    stripped here rather than duplicated into the key.
    """

    declared = tuple(str(field) for field in get("baseline.outage_rng_key"))
    marker = f"purpose={_RNG_PURPOSE}"
    if marker not in declared:
        raise NotImplementedError(
            f"params.baseline.outage_rng_key does not declare {marker!r}: {declared}"
        )
    fields = tuple(field for field in declared if field != marker)
    central = tuple(get(f"artifacts.rng_identity_fields.{_RNG_PURPOSE}"))
    if set(fields) != set(central):
        raise NotImplementedError(
            "params.baseline.outage_rng_key disagrees with "
            f"params.artifacts.rng_identity_fields.{_RNG_PURPOSE}: "
            f"{sorted(fields)} != {sorted(central)}"
        )
    return fields


def class_count(dataset: str) -> int:
    value = get(f"datasets.{dataset}.classes")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OutagePolicyError(f"params.datasets.{dataset}.classes is not a count")
    return value


@dataclass(frozen=True)
class ValidationLabelCounts:
    """Label frequencies over the *entire* committed validation manifest."""

    dataset: str
    split: str
    manifest_path: str
    manifest_sha256: str
    class_count: int
    class_counts: tuple[int, ...]
    validation_count: int

    def maximum_count(self) -> int:
        return max(self.class_counts)

    def tied_maximum_classes(self) -> tuple[int, ...]:
        maximum = self.maximum_count()
        return tuple(
            label
            for label, count in enumerate(self.class_counts)
            if count == maximum
        )

    def selected_class(self) -> int:
        """Highest validation count; the lowest class index breaks a tie."""

        if get("baseline.outage_class_tie_break") != _TIE_BREAK:
            raise NotImplementedError("unsupported outage tie-break")
        return min(self.tied_maximum_classes())


def count_validation_labels(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> ValidationLabelCounts:
    """Count validation labels across the whole committed manifest.

    This is a label-frequency calculation over canonical manifest bytes: it
    decodes no image, runs no classifier, consults no loader order and reads no
    sample subset.  Rows outside ``SELECTION_SPLIT`` — including every ``test``
    row — are discarded before any counting happens.
    """

    _require_configured_policy()
    path = manifest_path(dataset, repo_root)
    payload = path.read_bytes()
    rows = validate_manifest_bytes(dataset, payload)
    expected_classes = class_count(dataset)

    selection_rows = [row for row in rows if row.split == SELECTION_SPLIT]
    if not selection_rows:
        raise OutagePolicyError(
            f"{dataset}: manifest carries no {SELECTION_SPLIT} rows"
        )
    counts = Counter(row.label for row in selection_rows)
    unexpected = sorted(label for label in counts if label not in range(expected_classes))
    if unexpected:
        raise OutagePolicyError(
            f"{dataset}: validation labels outside the class range: {unexpected}"
        )
    class_counts = tuple(counts.get(label, 0) for label in range(expected_classes))
    total = sum(class_counts)
    if total != len(selection_rows):
        raise OutagePolicyError(f"{dataset}: validation label counts do not reconcile")
    configured_total = int(get(f"datasets.{dataset}.{SELECTION_SPLIT}_images"))
    if total != configured_total:
        raise OutagePolicyError(
            f"{dataset}: counted {total} validation rows, "
            f"params.datasets.{dataset}.{SELECTION_SPLIT}_images is {configured_total}"
        )
    try:
        relative = str(path.relative_to(Path(repo_root)))
    except ValueError:
        relative = str(path)
    return ValidationLabelCounts(
        dataset=dataset,
        split=SELECTION_SPLIT,
        manifest_path=relative,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        class_count=expected_classes,
        class_counts=class_counts,
        validation_count=total,
    )


@dataclass(frozen=True)
class OutagePolicy:
    """The frozen constant prediction, plus the counts that produced it.

    ``predict`` reads none of this beyond ``selected_class``: the counts are
    retained so the artifact can be re-verified, not so inference can re-derive
    the answer.
    """

    dataset: str
    split: str
    manifest_sha256: str
    class_count: int
    selected_class: int
    selected_count: int
    validation_count: int
    tied_maximum_classes: tuple[int, ...]
    class_counts: tuple[int, ...]

    @property
    def measured_validation_accuracy(self) -> float:
        """The *measured* frequency, always ``numerator / denominator``."""

        return self.selected_count / self.validation_count

    def predict(self) -> int:
        """Return the frozen constant prediction for one undelivered row.

        Deliberately argument-free and side-effect-free.  No sample identity,
        no label, no manifest and no system control flow can move it, which is
        what makes the constant-class policy checkable.
        """

        return self.selected_class

    def is_correct(self, true_label: int) -> bool:
        """Binary per-row correctness; an outage row is never fractionally right."""

        return int(true_label) == self.selected_class


def select_outage_policy(
    dataset: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[OutagePolicy, ValidationLabelCounts]:
    """Derive the frozen policy from the committed validation manifest."""

    counts = count_validation_labels(dataset, repo_root)
    selected = counts.selected_class()
    return (
        OutagePolicy(
            dataset=counts.dataset,
            split=counts.split,
            manifest_sha256=counts.manifest_sha256,
            class_count=counts.class_count,
            selected_class=selected,
            selected_count=counts.class_counts[selected],
            validation_count=counts.validation_count,
            tied_maximum_classes=counts.tied_maximum_classes(),
            class_counts=counts.class_counts,
        ),
        counts,
    )


def parameters_sha256() -> str:
    """Identity of the complete scientific parameter snapshot."""

    return canonical_sha256(
        {root: get(root) for root in get("config.fingerprint_parameter_roots")}
    )


def build_outage_policy_record(
    dataset: str,
    *,
    selection_source_commit: str,
    generated_at: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build the committed, deterministic frozen-selection artifact."""

    policy, counts = select_outage_policy(dataset, repo_root)
    return {
        "schema_version": OUTAGE_POLICY_SCHEMA_VERSION,
        "purpose": (
            "frozen constant-class outage prediction for undelivered classical "
            "transport blocks, selected on the validation split only"
        ),
        "status": "frozen",
        "evidence_labels": list(EVIDENCE_LABELS),
        "prominent_declaration": (
            "Bounded validation/plumbing integration. This is not the BR-4 full "
            "validation sweep, not a G-8 operating-point selection, and not test "
            "evidence. The measured validation accuracy below is a label "
            "frequency over the committed validation manifest, not an assumed "
            "1/n_classes."
        ),
        "dataset": policy.dataset,
        "dataset_version": get(
            f"datasets.{dataset}.{get('config.dataset_version_rule')}"
        ),
        "split": policy.split,
        "manifest_path": counts.manifest_path,
        "manifest_sha256": policy.manifest_sha256,
        "selection_policy": get("baseline.outage_policy"),
        "selection_rule": get("baseline.outage_class_selection"),
        "tie_break": get("baseline.outage_class_tie_break"),
        "tie_break_applied": len(policy.tied_maximum_classes) > 1,
        "sensitivity_policy": get("baseline.outage_policy_sensitivity"),
        "sensitivity_rng_identity_fields": list(configured_rng_identity_fields()),
        "class_count": policy.class_count,
        "validation_count": policy.validation_count,
        "class_counts": list(policy.class_counts),
        "maximum_count": counts.maximum_count(),
        "tied_maximum_classes": list(policy.tied_maximum_classes),
        "selected_class": policy.selected_class,
        "selected_count": policy.selected_count,
        "numerator": policy.selected_count,
        "denominator": policy.validation_count,
        "measured_validation_accuracy": policy.measured_validation_accuracy,
        "accuracy_derivation": "selected_count / validation_count",
        "assumed_uniform_accuracy_rejected": True,
        "frozen_before": get("baseline.outage_class_frozen_before"),
        "params_sha256": parameters_sha256(),
        "selection_source_commit": selection_source_commit,
        "generated_at": generated_at,
        "test_split_access": {
            "test_rows_read": False,
            "test_labels_counted": False,
            "test_split_sealed": True,
        },
    }


def _validated_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise OutagePolicyError("outage policy artifact must be an object")
    required = {
        "schema_version",
        "dataset",
        "split",
        "manifest_sha256",
        "selection_policy",
        "selection_rule",
        "tie_break",
        "class_count",
        "validation_count",
        "class_counts",
        "tied_maximum_classes",
        "selected_class",
        "selected_count",
        "numerator",
        "denominator",
        "measured_validation_accuracy",
        "frozen_before",
        "evidence_labels",
    }
    missing = sorted(required - set(record))
    if missing:
        raise OutagePolicyError(f"outage policy artifact is incomplete: {missing}")
    return dict(record)


def policy_from_record(record: Mapping[str, Any]) -> OutagePolicy:
    """Rebuild the runtime policy from artifact bytes, failing closed."""

    _require_configured_policy()
    body = _validated_record(record)
    if body["schema_version"] != OUTAGE_POLICY_SCHEMA_VERSION:
        raise OutagePolicyError(
            f"unsupported outage policy schema_version: {body['schema_version']}"
        )
    if body["selection_policy"] != get("baseline.outage_policy"):
        raise OutagePolicyError("frozen artifact records a different outage policy")
    if body["selection_rule"] != get("baseline.outage_class_selection"):
        raise OutagePolicyError("frozen artifact records a different selection rule")
    if body["tie_break"] != get("baseline.outage_class_tie_break"):
        raise OutagePolicyError("frozen artifact records a different tie-break")
    if body["frozen_before"] != get("baseline.outage_class_frozen_before"):
        raise OutagePolicyError("frozen artifact records a different freeze point")
    if body["split"] != SELECTION_SPLIT:
        raise OutagePolicyError(
            f"frozen artifact was selected on split {body['split']!r}, "
            f"not {SELECTION_SPLIT!r}"
        )
    if tuple(EVIDENCE_LABELS) != tuple(body["evidence_labels"]):
        raise OutagePolicyError("frozen artifact carries the wrong evidence labels")

    dataset = str(body["dataset"])
    counts = tuple(int(value) for value in body["class_counts"])
    expected_classes = class_count(dataset)
    if int(body["class_count"]) != expected_classes or len(counts) != expected_classes:
        raise OutagePolicyError(
            f"frozen artifact class count disagrees with params.datasets.{dataset}.classes"
        )
    selected = int(body["selected_class"])
    if selected not in range(expected_classes):
        raise OutagePolicyError("frozen artifact selected class is out of range")
    selected_count = int(body["selected_count"])
    validation_count = int(body["validation_count"])
    if sum(counts) != validation_count:
        raise OutagePolicyError("frozen artifact class counts do not sum to the total")
    if counts[selected] != selected_count:
        raise OutagePolicyError("frozen artifact selected count disagrees with its counts")
    if int(body["numerator"]) != selected_count or int(body["denominator"]) != validation_count:
        raise OutagePolicyError(
            "frozen artifact numerator/denominator disagree with its counts"
        )
    maximum = max(counts)
    tied = tuple(
        label for label, count in enumerate(counts) if count == maximum
    )
    if tuple(int(value) for value in body["tied_maximum_classes"]) != tied:
        raise OutagePolicyError("frozen artifact tied-maximum classes are wrong")
    if selected_count != maximum:
        raise OutagePolicyError("frozen artifact selected class is not a maximum")
    if selected != min(tied):
        raise OutagePolicyError(
            "frozen artifact selected class does not apply the lowest-index tie-break"
        )
    accuracy = float(body["measured_validation_accuracy"])
    if accuracy != selected_count / validation_count:
        raise OutagePolicyError(
            "frozen artifact accuracy is not selected_count / validation_count"
        )
    return OutagePolicy(
        dataset=dataset,
        split=str(body["split"]),
        manifest_sha256=str(body["manifest_sha256"]),
        class_count=expected_classes,
        selected_class=selected,
        selected_count=selected_count,
        validation_count=validation_count,
        tied_maximum_classes=tied,
        class_counts=counts,
    )


def load_outage_policy(
    path: Path,
    *,
    expected_dataset: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> OutagePolicy:
    """Load and verify the frozen artifact; never reselect the class.

    This is the only runtime entry point.  It reads the committed artifact, not
    the manifest, so ordinary prediction touches no labels at all.
    """

    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutagePolicyError(f"cannot read outage policy artifact {path}: {exc}") from None
    policy = policy_from_record(record)
    if expected_dataset is not None and policy.dataset != expected_dataset:
        raise OutagePolicyError(
            f"frozen outage policy is for {policy.dataset!r}, "
            f"not {expected_dataset!r}"
        )
    if (
        expected_manifest_sha256 is not None
        and policy.manifest_sha256 != expected_manifest_sha256
    ):
        raise OutagePolicyError(
            "frozen outage policy was selected on a different split manifest"
        )
    return policy


def write_json_atomically(path: Path, payload: Mapping[str, Any]) -> str:
    """Write canonical JSON through a temporary file, then rename into place."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(body).hexdigest()


def keyed_uniform_random_label(
    *,
    split_manifest_hash: str,
    stable_sample_id: str,
    channel_seed: int,
    n_classes: int,
) -> int:
    """Draw the configured sensitivity label for exactly one row.

    A fresh counter-based generator is keyed by the complete declared identity
    and one integer is taken from it, so the value is a pure function of that
    key.  Nothing about row order, batch size, the number of outages in the run,
    or draws taken elsewhere can reach it.
    """

    _require_configured_policy()
    fields = configured_rng_identity_fields()
    identity = {
        "split_manifest_hash": split_manifest_hash,
        "stable_sample_id": stable_sample_id,
        "channel_seed": channel_seed,
    }
    if set(fields) != set(identity):
        raise OutagePolicyError(
            f"declared outage RNG identity {sorted(fields)} is not "
            f"{sorted(identity)}"
        )
    if not isinstance(n_classes, int) or isinstance(n_classes, bool) or n_classes <= 0:
        raise OutagePolicyError("n_classes must be a positive integer")
    generator = keyed_generator(_RNG_PURPOSE, identity)
    return int(generator.integers(0, n_classes))


def sensitivity_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_manifest_hash: str,
    channel_seed: int,
    n_classes: int,
) -> tuple[int, ...]:
    """Map the sensitivity policy over rows, one independent key each."""

    return tuple(
        keyed_uniform_random_label(
            split_manifest_hash=split_manifest_hash,
            stable_sample_id=str(row["stable_sample_id"]),
            channel_seed=channel_seed,
            n_classes=n_classes,
        )
        for row in rows
    )
