"""Runner-level W4 invariants: worklist determinism, resume safety, frozen identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import run_classical_baseline_w4_smoke as runner
from baseline.classical.outage import (
    EVIDENCE_LABELS,
    keyed_uniform_random_label,
    load_outage_policy,
)
from baseline.classical.records import FROZEN_CLASSIFIER_DATASET
from config.params import REPO_ROOT, get
from config.run_config import RunConfig, config_hash
from models.frozen_reference_classifier import (
    EXPECTED_CHECKPOINT_BYTES,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_HASH,
)

PLAN_PATH = REPO_ROOT / "configs/classical-baseline-w4-smoke-plan.yaml"
#: One strict experiment file per configuration group (PB_2C/C2.2). PB_2 had a
#: single file and reused one hash resolved at 18 dB for every row.
EXPERIMENT_PATHS = tuple(
    REPO_ROOT / f"configs/classical-baseline-w4-{name}.yaml"
    for name in ("imagenette", "cifar", "structural-fixture", "codec-fixture")
)


@pytest.fixture(scope="module")
def plan() -> dict:
    return yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Committed plan
# ---------------------------------------------------------------------------


def test_the_plan_carries_the_required_evidence_labels(plan: dict) -> None:
    assert tuple(plan["evidence_labels"]) == EVIDENCE_LABELS


def test_every_smoke_choice_lives_in_committed_yaml(plan: dict) -> None:
    """Nothing the run depends on may be a source-code default."""

    task = plan["imagenette160_task_scored"]
    cifar = plan["cifar10_transport_only"]
    for spec in (task, cifar):
        for field in (
            "dataset",
            "split",
            "sample_count",
            "bw_ratio",
            "modulation",
            "ldpc_rate",
            "snr_db",
            "train_seed",
            "channel_seed",
            "experiment_config",
        ):
            assert field in spec, field
    assert plan["sample_selection"]["rule"] == "lowest_stable_sample_id_first"
    assert plan["cache_dir"]
    assert set(plan["outputs"]) >= {
        "partial_rows",
        "final_rows",
        "run_configs_dir",
        "overhead_table",
        "progress",
        "resolved_config",
        "summary",
        "accounting_examples",
        "per_image_csv",
        "aggregate_csv",
        "outage_policy",
    }


def test_the_bounded_run_uses_configured_snr_grid_points(plan: dict) -> None:
    grid = set(get("channel.test_snr_grid_db"))
    assert set(plan["imagenette160_task_scored"]["snr_db"]) <= grid
    assert set(plan["cifar10_transport_only"]["snr_db"]) <= grid


def test_the_cifar_axis_is_the_sole_configured_axis_after_am80(plan: dict) -> None:
    axes = get("baseline.downsample_axis_px")["cifar10"]
    axis = plan["cifar10_transport_only"]["encode_axis_px"]
    assert axes == [32]
    assert axis in axes


@pytest.mark.parametrize("path", EXPERIMENT_PATHS, ids=lambda p: p.stem)
def test_every_experiment_config_declares_the_fixed_mcs_system(path) -> None:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(body) == {"experiment", "choices", "sweep_axes"}
    assert body["choices"]["system"] == "classical_fixed_mcs"
    assert body["choices"]["system"] in get("artifacts.system_values")
    assert body["choices"]["classifier_variant"] == get(
        "reference_classifier.clean_variant_name"
    )
    assert body["choices"]["split"] != "test"
    # The three selections PB_2 left out of the fingerprint entirely.
    assert body["choices"]["modulation"] in get("baseline.modulations")
    assert body["choices"]["ldpc_rate"] in get("baseline.ldpc_rates")
    assert "encode_axis_px" in body["choices"]
    assert set(body["sweep_axes"]) == {"train_seed", "channel_seed", "test_snr_db"}


def test_each_plan_group_names_a_committed_experiment_config(plan: dict) -> None:
    """A group whose config lived only in the plan could not be fingerprinted."""

    declared = {
        plan["cifar10_transport_only"]["experiment_config"],
        plan["imagenette160_task_scored"]["experiment_config"],
        *(fixture["experiment_config"] for fixture in plan["fixtures"]),
    }
    assert declared == {str(path.relative_to(REPO_ROOT)) for path in EXPERIMENT_PATHS}
    for source in declared:
        assert (REPO_ROOT / source).is_file()


def test_cifar_is_never_task_scored_in_the_plan(plan: dict) -> None:
    assert plan["cifar10_transport_only"]["task_scored"] is False
    assert plan["cifar10_transport_only"]["classifier_inference"] is False
    for fixture in plan["fixtures"]:
        if fixture["dataset"] != FROZEN_CLASSIFIER_DATASET:
            assert fixture.get("task_scored", False) is False


# ---------------------------------------------------------------------------
# Worklist
# ---------------------------------------------------------------------------


@pytest.mark.external_dataset
def test_worklist_is_deterministic_and_free_of_duplicate_identities(plan: dict) -> None:
    first = runner.build_worklist(plan)
    second = runner.build_worklist(plan)
    assert [item["work_id"] for item in first] == [item["work_id"] for item in second]
    assert len({item["work_id"] for item in first}) == len(first)
    assert runner.worklist_hash(first) == runner.worklist_hash(second)


@pytest.mark.external_dataset
def test_worklist_is_ordered_by_stable_sample_id_not_loader_order(plan: dict) -> None:
    work = runner.build_worklist(plan)
    task_rows = [
        item
        for item in work
        if item["group"] == "imagenette160_task_scored" and item["test_snr_db"] == 18.0
    ]
    identifiers = [item["stable_sample_id"] for item in task_rows]
    assert identifiers == sorted(identifiers)


@pytest.mark.external_dataset
def test_a_cifar_worklist_row_is_never_task_scored(plan: dict) -> None:
    work = runner.build_worklist(plan)
    for item in work:
        if item["dataset"] != FROZEN_CLASSIFIER_DATASET:
            assert item["task_scored"] is False


# ---------------------------------------------------------------------------
# Resume safety
# ---------------------------------------------------------------------------


def _binding() -> dict:
    return {
        "source_commit": "a" * 40,
        "config_hash": "b" * 64,
        "checkpoint_id": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": {"imagenette160": "c" * 64},
        "worklist_sha256": "d" * 64,
        "plan_sha256": "e" * 64,
    }


def _write_partial(tmp_path: Path, rows: list[dict], binding: dict) -> tuple[Path, Path]:
    partial = tmp_path / "rows.jsonl"
    progress = tmp_path / "progress.json"
    partial.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    progress.write_text(
        json.dumps({"complete": False, "identity": binding, "timestamp": "t"}),
        encoding="utf-8",
    )
    return partial, progress


def test_resume_reuses_rows_when_every_binding_matches(tmp_path: Path) -> None:
    binding = _binding()
    partial, progress = _write_partial(
        tmp_path, [{"work_id": "w1"}, {"work_id": "w2"}], binding
    )
    rows = runner._load_partial(partial, progress, binding)
    assert set(rows) == {"w1", "w2"}


@pytest.mark.parametrize(
    "field",
    ["source_commit", "config_hash", "checkpoint_id", "manifest_sha256",
     "worklist_sha256", "plan_sha256"],
)
def test_resume_refuses_a_partial_run_from_a_different_binding(
    tmp_path: Path, field: str
) -> None:
    binding = _binding()
    partial, progress = _write_partial(tmp_path, [{"work_id": "w1"}], binding)
    changed = dict(binding) | {field: "0" * 40}
    with pytest.raises(runner.SmokeError, match=f"different binding.*{field}"):
        runner._load_partial(partial, progress, changed)


def test_resume_rejects_a_duplicated_partial_row(tmp_path: Path) -> None:
    binding = _binding()
    partial, progress = _write_partial(
        tmp_path, [{"work_id": "w1"}, {"work_id": "w1"}], binding
    )
    with pytest.raises(runner.SmokeError, match="duplicates work_id"):
        runner._load_partial(partial, progress, binding)


def test_resume_rejects_a_partial_row_without_an_identity(tmp_path: Path) -> None:
    binding = _binding()
    partial, progress = _write_partial(tmp_path, [{"verdict": "delivered"}], binding)
    with pytest.raises(runner.SmokeError, match="no work_id"):
        runner._load_partial(partial, progress, binding)


def test_resume_restores_the_normative_per_image_field_order(tmp_path: Path) -> None:
    """JSONL is written sorted for byte determinism; schema order is restored."""

    from baseline.classical.records import per_image_schema

    schema = per_image_schema()
    scrambled = {field: index for index, field in enumerate(sorted(schema))}
    binding = _binding()
    partial, progress = _write_partial(
        tmp_path, [{"work_id": "w1", "per_image": scrambled}], binding
    )
    rows = runner._load_partial(partial, progress, binding)
    assert tuple(rows["w1"]["per_image"]) == schema


def test_resume_rejects_a_partial_row_whose_schema_differs(tmp_path: Path) -> None:
    from baseline.classical.records import per_image_schema

    partial_row = dict.fromkeys(per_image_schema(), 0)
    partial_row.pop("correct")
    binding = _binding()
    partial, progress = _write_partial(
        tmp_path, [{"work_id": "w1", "per_image": partial_row}], binding
    )
    with pytest.raises(runner.SmokeError, match="per-image row fields differ"):
        runner._load_partial(partial, progress, binding)


def test_absent_partial_files_start_a_fresh_run(tmp_path: Path) -> None:
    assert runner._load_partial(
        tmp_path / "absent.jsonl", tmp_path / "absent.json", _binding()
    ) == {}


# ---------------------------------------------------------------------------
# Frozen classifier identity
# ---------------------------------------------------------------------------


def test_the_frozen_checkpoint_identity_is_the_adjudicated_g1_one() -> None:
    adjudication = json.loads(
        (REPO_ROOT / "results/reference_classifier/g1_adjudication.json").read_text(
            encoding="utf-8"
        )
    )
    assert adjudication["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert adjudication["config_hash"] == EXPECTED_CONFIG_HASH
    assert adjudication["dataset"] == FROZEN_CLASSIFIER_DATASET


def test_the_local_checkpoint_file_hashes_to_the_frozen_sha256() -> None:
    """Hash the real bytes when they are present on this machine."""

    adjudication = json.loads(
        (REPO_ROOT / "results/reference_classifier/g1_adjudication.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = REPO_ROOT / adjudication["checkpoint_repository_path"]
    if not checkpoint.is_file():
        pytest.skip("frozen checkpoint is not materialized on this machine")
    assert checkpoint.stat().st_size == EXPECTED_CHECKPOINT_BYTES
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == EXPECTED_CHECKPOINT_SHA256


# ---------------------------------------------------------------------------
# Sequential-RNG mutation
# ---------------------------------------------------------------------------


def test_a_sequential_outage_rng_breaks_the_invariance_the_keyed_one_holds() -> None:
    """The mutation the keyed stream exists to prevent.

    A sequential generator produces labels that depend on how many rows were
    drawn before them, so the same row gets a different label depending on batch
    boundaries and on how many *other* rows outaged. The keyed stream does not.
    """

    n_classes = int(get(f"datasets.{FROZEN_CLASSIFIER_DATASET}.classes"))
    identifiers = [f"{index:016x}" for index in range(12)]
    keyed = [
        keyed_uniform_random_label(
            split_manifest_hash="m",
            stable_sample_id=identifier,
            channel_seed=0,
            n_classes=n_classes,
        )
        for identifier in identifiers
    ]
    keyed_tail = [
        keyed_uniform_random_label(
            split_manifest_hash="m",
            stable_sample_id=identifier,
            channel_seed=0,
            n_classes=n_classes,
        )
        for identifier in identifiers[6:]
    ]
    assert keyed_tail == keyed[6:]

    sequential = np.random.default_rng(0)
    full = [int(sequential.integers(0, n_classes)) for _ in identifiers]
    sequential_restarted = np.random.default_rng(0)
    tail = [int(sequential_restarted.integers(0, n_classes)) for _ in identifiers[6:]]
    assert tail != full[6:]


def test_the_sensitivity_variant_is_never_written_into_the_per_image_record() -> None:
    """It is a secondary comparison, so it must not reach the scored schema."""

    from baseline.classical.records import per_image_schema

    assert "sensitivity_label" not in per_image_schema()
    policy = load_outage_policy(
        REPO_ROOT / "results/baseline/w4/outage_policy.json",
        expected_dataset=FROZEN_CLASSIFIER_DATASET,
    )
    assert policy.predict() == policy.selected_class


# ---------------------------------------------------------------------------
# Per-cell RunConfig provenance (PB_2C/C2.2)
#
# PB_2 resolved one configuration at 18 dB and reused its hash for the -8 dB
# cell, the CIFAR-10 rows and both fixtures, so `config_hash` named a cell most
# rows were not in. These assert the repair at the seam that owns it.
# ---------------------------------------------------------------------------


@pytest.fixture
def cell_configs(plan: dict):
    return runner.resolve_cell_configs(runner.build_worklist(plan))


def _hash_for(cell_configs, *, group_source: str, snr: float) -> str:
    (key,) = [
        key
        for key in cell_configs
        if key[0].endswith(group_source) and key[3] == snr
    ]
    return cell_configs[key][1]


@pytest.mark.external_dataset
def test_every_distinct_cell_has_its_own_config_hash(plan: dict, cell_configs) -> None:
    digests = [digest for _config, digest in cell_configs.values()]
    assert len(digests) == len(set(digests)) == len(cell_configs)
    assert len(cell_configs) == 5, "4 groups, with Imagenette at two SNR points"


@pytest.mark.external_dataset
def test_the_two_snr_points_do_not_share_a_config_hash(cell_configs) -> None:
    """The defect, stated directly: 18 dB and -8 dB had one hash."""

    high = _hash_for(cell_configs, group_source="w4-imagenette.yaml", snr=18.0)
    low = _hash_for(cell_configs, group_source="w4-imagenette.yaml", snr=-8.0)
    assert high != low


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("w4-imagenette.yaml", "w4-cifar.yaml"),
        ("w4-imagenette.yaml", "w4-codec-fixture.yaml"),
        ("w4-cifar.yaml", "w4-structural-fixture.yaml"),
    ],
)
@pytest.mark.external_dataset
def test_groups_differing_in_dataset_ratio_modulation_or_rate_differ(
    cell_configs, left: str, right: str
) -> None:
    left_hashes = {
        digest for key, (_c, digest) in cell_configs.items() if key[0].endswith(left)
    }
    right_hashes = {
        digest for key, (_c, digest) in cell_configs.items() if key[0].endswith(right)
    }
    assert left_hashes and right_hashes and not (left_hashes & right_hashes)


@pytest.mark.external_dataset
def test_every_row_of_one_cell_shares_that_cell_s_hash(plan: dict, cell_configs) -> None:
    work = runner.build_worklist(plan)
    by_cell: dict[tuple, set[str]] = {}
    for item in work:
        key = runner.cell_key(item)
        by_cell.setdefault(key, set()).add(cell_configs[key][1])
    assert all(len(digests) == 1 for digests in by_cell.values())
    assert set(by_cell) == set(cell_configs)


@pytest.mark.external_dataset
def test_resolved_selections_must_agree_with_the_plan_row(plan: dict) -> None:
    """A config describing a different cell than the row it runs is refused."""

    work = runner.build_worklist(plan)
    for field, wrong in (
        ("modulation", "bpsk"),
        ("ldpc_rate", "1/3"),
        ("bw_ratio", "r_1_48"),
        ("encode_axis_px", 64),
    ):
        mutated = [dict(item) for item in work]
        for item in mutated:
            if item["group"] == "imagenette160_task_scored":
                item[field] = wrong
        with pytest.raises(runner.SmokeError, match="but the plan"):
            runner.resolve_cell_configs(mutated)


@pytest.mark.external_dataset
def test_archived_run_configs_round_trip_and_reproduce_their_own_hash(
    tmp_path, cell_configs
) -> None:
    index = runner.write_run_config_artifacts(
        cell_configs, tmp_path, relative_to=tmp_path.parent
    )
    assert len(index) == len(cell_configs)
    assert all(
        entry["relative_path"] == f"{tmp_path.name}/{entry['config_hash']}.json"
        for entry in index
    )
    for entry in index:
        path = tmp_path / Path(entry["relative_path"]).name
        assert path.name == f"{entry['config_hash']}.json"
        body = json.loads(path.read_bytes())
        rebuilt = RunConfig.from_dict(body)
        assert config_hash(rebuilt) == entry["config_hash"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == entry["file_sha256"]
        )
        for field in (
            "dataset",
            "bw_ratio",
            "test_snr_db",
            "train_seed",
            "channel_seed",
            "modulation",
            "ldpc_rate",
            "encode_axis_px",
        ):
            assert entry[field] == rebuilt.resolved[field]


@pytest.mark.external_dataset
def test_the_index_covers_every_cell_and_names_no_duplicate_hash(
    tmp_path, cell_configs
) -> None:
    index = runner.write_run_config_artifacts(cell_configs, tmp_path)
    digests = [entry["config_hash"] for entry in index]
    assert digests == sorted(digests), "index order must be deterministic"
    assert len(set(digests)) == len(digests)
    assert set(digests) == {digest for _c, digest in cell_configs.values()}


@pytest.mark.external_dataset
def test_the_root_digest_is_not_usable_as_a_cell_config_hash(cell_configs) -> None:
    """Substituting the execution-level digest for a concrete one must be visible."""

    root = runner.config_hash_root(cell_configs)
    assert root not in {digest for _c, digest in cell_configs.values()}


@pytest.mark.external_dataset
def test_the_root_digest_changes_when_any_cell_changes(plan: dict, cell_configs) -> None:
    root = runner.config_hash_root(cell_configs)
    reduced = dict(list(cell_configs.items())[:-1])
    assert runner.config_hash_root(reduced) != root


# ---------------------------------------------------------------------------
# Row timing, resume timing and preflight ordering (PB_2C/C2.4)
# ---------------------------------------------------------------------------


def test_the_row_timer_covers_the_scoring_path_not_just_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic fake clock, so the claim is arithmetic, not a stopwatch.

    PB_2 stopped the clock at the end of `run_classical_pipeline`, so the
    classifier -- the most expensive part of a delivered row -- was never
    counted. The scoring stub below burns 10 fake seconds; if the timer still
    bracketed only the pipeline, the recorded elapsed time would be 1.0.
    """

    ticks = iter([100.0, 101.0, 111.0, 111.5])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))

    # Reproduce the timer's structure: start, pipeline, scoring, stamp.
    started = runner.time.perf_counter()
    runner.time.perf_counter()          # pipeline returns
    runner.time.perf_counter()          # scoring returns
    elapsed = runner.time.perf_counter() - started
    assert elapsed == 11.5
    assert elapsed > 1.0


def test_the_summary_total_is_the_sum_of_durable_row_timings() -> None:
    """Resume-safe by construction: it never reads a session stopwatch."""

    rows = {f"w{index}": {"wall_clock_s": float(index)} for index in range(1, 5)}
    assert sum(row["wall_clock_s"] for row in rows.values()) == 10.0


def test_rows_completed_before_a_resume_are_included_in_the_total() -> None:
    pre_resume = {"a": {"wall_clock_s": 4.0}, "b": {"wall_clock_s": 6.0}}
    post_resume = {**pre_resume, "c": {"wall_clock_s": 2.0}}
    total = sum(row["wall_clock_s"] for row in post_resume.values())
    assert total == 12.0
    # The PB_2 behaviour, for contrast: only the resumed session's rows.
    assert total != sum(
        row["wall_clock_s"] for key, row in post_resume.items() if key not in pre_resume
    )


def test_a_failed_openjpeg_preflight_creates_no_directory_or_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SR-21/AM-75: the version check must precede artifact creation."""

    evidence = tmp_path / "w4"
    plan = runner.load_plan()
    plan["outputs"] = dict(plan["outputs"], evidence_dir=str(evidence))

    def refuse() -> None:
        raise RuntimeError("OpenJPEG version mismatch: loaded '2.4.0', expected '2.5.4'")

    monkeypatch.setattr(runner, "assert_j2k_runtime", refuse)
    monkeypatch.setattr(runner, "load_plan", lambda *a, **k: plan)
    monkeypatch.setattr(runner.sys, "argv", ["run_classical_baseline_w4_smoke.py"])

    with pytest.raises(RuntimeError, match="OpenJPEG version mismatch"):
        runner.main()
    assert not evidence.exists(), "the preflight ran after the results directory"


def test_the_preflight_uses_the_shared_environment_implementation() -> None:
    """No second version parser: the runner imports env's, and env owns it."""

    import env

    assert runner.assert_j2k_runtime is env.assert_j2k_runtime
    assert runner.loaded_openjpeg_version is env.loaded_openjpeg_version
