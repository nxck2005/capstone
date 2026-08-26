"""Resolved, immutable experiment configuration (SR-1, SR-13).

Committed files under ``params.config.dir`` contain choices and sweep axes only.
This module resolves those names through ``params.generated.yaml`` and produces
one fully concrete configuration for one run.  The symbolic and resolved forms
are both retained: the former records what was selected, while the latter
records exactly what that selection meant when the run happened.

This module deliberately imports neither torch nor any project runtime code, so
analysis and demo environments can load archived configurations on the CPU-only
install path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config.params import REPO_ROOT, get


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMap.from_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class FrozenMap(Mapping[str, Any]):
    """A small, recursively immutable mapping with deterministic key order."""

    _items: tuple[tuple[str, Any], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FrozenMap:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("configuration mapping keys must be strings")
        return cls(tuple(sorted((key, _freeze(item)) for key, item in value.items())))

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self)


@dataclass(frozen=True)
class RunConfig:
    """One fully resolved run, retaining its committed symbolic choices."""

    fingerprint_schema_version: int
    experiment: str
    source: str
    choices: FrozenMap
    sweep_axes: FrozenMap
    resolved: FrozenMap
    parameters: FrozenMap

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_schema_version": self.fingerprint_schema_version,
            "experiment": self.experiment,
            "source": self.source,
            "choices": self.choices.to_dict(),
            "sweep_axes": self.sweep_axes.to_dict(),
            "resolved": self.resolved.to_dict(),
            "parameters": self.parameters.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunConfig:
        expected = {
            "fingerprint_schema_version",
            "experiment",
            "source",
            "choices",
            "sweep_axes",
            "resolved",
            "parameters",
        }
        extra = set(value) - expected
        missing = expected - set(value)
        if missing or extra:
            raise ValueError(
                f"run config keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        experiment = value["experiment"]
        source = value["source"]
        fingerprint_schema_version = value["fingerprint_schema_version"]
        if (
            not isinstance(fingerprint_schema_version, int)
            or isinstance(fingerprint_schema_version, bool)
            or fingerprint_schema_version <= 0
        ):
            raise TypeError("fingerprint_schema_version must be a positive integer")
        if not isinstance(experiment, str) or not experiment:
            raise TypeError("experiment must be a non-empty string")
        if not isinstance(source, str) or not source:
            raise TypeError("source must be a non-empty string")
        mappings = (
            value["choices"],
            value["sweep_axes"],
            value["resolved"],
            value["parameters"],
        )
        if not all(isinstance(item, Mapping) for item in mappings):
            raise TypeError(
                "choices, sweep_axes, resolved and parameters must be mappings"
            )
        return cls(
            fingerprint_schema_version=fingerprint_schema_version,
            experiment=experiment,
            source=source,
            choices=FrozenMap.from_mapping(value["choices"]),
            sweep_axes=FrozenMap.from_mapping(value["sweep_axes"]),
            resolved=FrozenMap.from_mapping(value["resolved"]),
            parameters=FrozenMap.from_mapping(value["parameters"]),
        )


_SYMBOLIC_NAMESPACES = {
    "bw_ratio": "bandwidth",
    "train_snr_db": "channel",
    "lambda": "learned_system",
}

_SWEEP_PARAMS = {
    "train_seed": "evaluation.train_seeds",
    "channel_seed": "evaluation.channel_seeds",
    "test_snr_db": "channel.test_snr_grid_db",
}

_REFERENCE_CLASSIFIER_SCHEMA = {
    "experiment",
    "choices",
    "sweep_axes",
}

def _experiment_path(path: str | Path) -> Path:
    config_root = (REPO_ROOT / get("config.dir")).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == Path(get("config.dir")).parts[0]:
            candidate = REPO_ROOT / candidate
        else:
            candidate = config_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(config_root):
        raise ValueError(f"experiment config must live under {config_root}: {candidate}")
    expected_suffix = f".{get('config.file_format')}"
    if candidate.suffix != expected_suffix:
        raise ValueError(f"experiment config must use {expected_suffix}: {candidate}")
    return candidate


def _resolve_choice(key: str, value: Any) -> Any:
    if key == "bw_ratio":
        ratios = get("bandwidth.ratios")
        if value in ratios:
            return value
    namespace = _SYMBOLIC_NAMESPACES.get(key)
    if namespace and isinstance(value, str):
        parameter_path = f"{namespace}.{value}"
        try:
            return get(parameter_path)
        except KeyError as exc:
            raise ValueError(
                f"choice {key!r} has unresolvable value {value!r}; "
                f"attempted parameter path params.{parameter_path}"
            ) from exc
    return value


def _validate_named_choices(resolved: Mapping[str, Any]) -> None:
    dataset = resolved.get("dataset")
    if not isinstance(dataset, str):
        raise ValueError("choices.dataset must name a dataset")
    try:
        dataset_params = get(f"datasets.{dataset}")
    except KeyError:
        raise ValueError(f"unknown dataset choice: {dataset}") from None
    if not isinstance(dataset_params, Mapping) or "image_size" not in dataset_params:
        raise ValueError(f"choices.dataset does not name a dataset entry: {dataset}")

    system = resolved.get("system")
    if system not in get("artifacts.system_values"):
        raise ValueError(f"unknown system choice: {system}")

    channel = resolved.get("channel")
    if channel not in get("channel.models_supported"):
        raise ValueError(f"unknown channel choice: {channel}")

    bw_ratio = resolved.get("bw_ratio")
    if bw_ratio not in get("bandwidth.ratios"):
        raise ValueError(f"unknown bandwidth-ratio choice: {bw_ratio}")

    # The classical arm's per-cell selections. These are optional because the
    # learned arm does not carry them, but a *present* one must name a
    # configured value: a typo would otherwise become a silently distinct run
    # fingerprint rather than an error, which is exactly the class of defect
    # PB_2C exists to close.
    if "modulation" in resolved:
        modulation = resolved["modulation"]
        if modulation not in get("baseline.modulations"):
            raise ValueError(f"unknown modulation choice: {modulation}")
    if "ldpc_rate" in resolved:
        ldpc_rate = resolved["ldpc_rate"]
        if ldpc_rate not in get("baseline.ldpc_rates"):
            raise ValueError(f"unknown LDPC-rate choice: {ldpc_rate}")
    if "encode_axis_px" in resolved:
        axis = resolved["encode_axis_px"]
        # `None` means "let the configured ladder choose", which is a real
        # selection and must stay distinguishable from a pinned axis.
        if axis is not None:
            configured = get("baseline.downsample_axis_px").get(dataset, ())
            if axis not in {int(value) for value in configured}:
                raise ValueError(
                    f"encode_axis_px {axis!r} is not a configured axis for "
                    f"{dataset}: {list(configured)}"
                )


def load_experiment(path: str | Path, **overrides: Any) -> RunConfig:
    """Resolve a committed experiment-choice file into one concrete run.

    Every declared sweep axis must receive an override, and no undeclared
    override is accepted.  This keeps a ``RunConfig`` run-sized rather than
    turning it into a second sweep description.
    """

    source = _experiment_path(path)
    body = yaml.safe_load(source.read_text())
    if not isinstance(body, Mapping):
        raise TypeError(f"experiment config must contain a mapping: {source}")
    expected = {"experiment", "choices", "sweep_axes"}
    missing = expected - set(body)
    extra = set(body) - expected
    if missing or extra:
        raise ValueError(
            f"experiment file keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if not isinstance(body["choices"], Mapping) or not isinstance(body["sweep_axes"], Mapping):
        raise TypeError("choices and sweep_axes must be mappings")

    choices = dict(body["choices"])
    sweep_axes = dict(body["sweep_axes"])
    missing_overrides = set(sweep_axes) - set(overrides)
    extra_overrides = set(overrides) - set(sweep_axes)
    if missing_overrides or extra_overrides:
        raise ValueError(
            "sweep overrides differ: "
            f"missing={sorted(missing_overrides)}, extra={sorted(extra_overrides)}"
        )

    resolved = {key: _resolve_choice(key, value) for key, value in choices.items()}
    for axis, selector in sweep_axes.items():
        param_path = _SWEEP_PARAMS.get(axis)
        if param_path is None:
            raise ValueError(f"unsupported sweep axis: {axis}")
        if selector != param_path.rsplit(".", maxsplit=1)[-1]:
            raise ValueError(
                f"sweep axis {axis} must name {param_path}, got {selector!r}"
            )
        allowed = get(param_path)
        selected = overrides[axis]
        if selected not in allowed:
            raise ValueError(
                f"override {axis}={selected!r} is outside params.{param_path}"
            )
        resolved[axis] = selected

    _validate_named_choices(resolved)
    dataset = resolved["dataset"]
    bw_ratio = resolved["bw_ratio"]
    version_field = get("config.dataset_version_rule")
    resolved["dataset_version"] = get(f"datasets.{dataset}.{version_field}")
    resolved["analysis_version"] = get("config.analysis_version")
    resolved["k"] = get(f"bandwidth.k_symbols.{dataset}.{bw_ratio}")
    resolved["project_id"] = get("project.id")
    resolved["task"] = get("project.task")

    try:
        source_label = str(source.relative_to(REPO_ROOT))
    except ValueError:
        source_label = str(source)
    return RunConfig(
        fingerprint_schema_version=get("config.fingerprint_schema_version"),
        experiment=str(body["experiment"]),
        source=source_label,
        choices=FrozenMap.from_mapping(choices),
        sweep_axes=FrozenMap.from_mapping(sweep_axes),
        resolved=FrozenMap.from_mapping(resolved),
        parameters=FrozenMap.from_mapping(
            {
                root: get(root)
                for root in get("config.fingerprint_parameter_roots")
            }
        ),
    )


def load_reference_classifier_config(
    path: str | Path,
    *,
    dataset: str,
    **overrides: Any,
) -> RunConfig:
    """Resolve the clean classifier recipe without channel-shaped placeholders."""

    if overrides:
        raise ValueError(
            "reference classifier config accepts no overrides: "
            f"extra={sorted(overrides)}"
        )
    source = _experiment_path(path)
    body = yaml.safe_load(source.read_text())
    if not isinstance(body, Mapping):
        raise TypeError(f"classifier config must contain a mapping: {source}")
    missing = _REFERENCE_CLASSIFIER_SCHEMA - set(body)
    extra = set(body) - _REFERENCE_CLASSIFIER_SCHEMA
    if missing or extra:
        raise ValueError(
            f"classifier config keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    choices = body["choices"]
    sweep_axes = body["sweep_axes"]
    if not isinstance(choices, Mapping) or not isinstance(sweep_axes, Mapping):
        raise TypeError("classifier choices and sweep_axes must be mappings")
    if not isinstance(body["experiment"], str) or not body["experiment"]:
        raise TypeError("classifier experiment must be a non-empty string")
    if set(choices) != {"classifier_variant", "train_seed"}:
        raise ValueError("classifier choices must be classifier_variant and train_seed")
    if dict(sweep_axes) != {"dataset": "configured_datasets"}:
        raise ValueError("classifier sweep_axes must be dataset: configured_datasets")
    datasets = get("datasets")
    if dataset not in datasets or not isinstance(datasets[dataset], Mapping) or "loader" not in datasets[dataset]:
        raise ValueError(f"unknown classifier dataset: {dataset}")
    if choices["classifier_variant"] != get("reference_classifier.clean_variant_name"):
        raise ValueError("classifier variant must be the configured clean variant")
    if choices["train_seed"] != "clean_train_seed":
        raise ValueError("classifier train_seed must resolve clean_train_seed")
    resolved = {
        "project_id": get("project.id"),
        "task": get("project.task"),
        "dataset": dataset,
        "dataset_version": get(f"datasets.{dataset}.{get('config.dataset_version_rule')}"),
        "split_manifest_hash": get(f"datasets.{dataset}.manifest_sha256"),
        "classifier_variant": choices["classifier_variant"],
        "train_seed": get("reference_classifier.clean_train_seed"),
        "architecture": get("reference_classifier.arch"),
        "analysis_version": get("config.analysis_version"),
    }
    try:
        source_label = str(source.relative_to(REPO_ROOT))
    except ValueError:
        source_label = str(source)
    return RunConfig(
        fingerprint_schema_version=get("config.fingerprint_schema_version"),
        experiment=str(body["experiment"]),
        source=source_label,
        choices=FrozenMap.from_mapping(dict(choices)),
        sweep_axes=FrozenMap.from_mapping(dict(sweep_axes)),
        resolved=FrozenMap.from_mapping(resolved),
        parameters=FrozenMap.from_mapping(
            {root: get(root) for root in get("config.fingerprint_parameter_roots")}
        ),
    )


def canonical_sha256(value: Any) -> str:
    """Stable SHA-256 over canonical JSON for configuration and artifact IDs."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def config_hash(cfg: RunConfig) -> str:
    """Hash the versioned resolved run and complete scientific parameter snapshot."""

    hash_form = get("config.run_config_hash_form")
    if hash_form != (
        "sha256_over_versioned_resolved_and_parameter_snapshot_canonical_json"
    ):
        raise NotImplementedError(f"unsupported params.config.run_config_hash_form: {hash_form}")
    return canonical_sha256(
        {
            "fingerprint_schema_version": cfg.fingerprint_schema_version,
            "resolved": cfg.resolved.to_dict(),
            "parameters": cfg.parameters.to_dict(),
        }
    )
