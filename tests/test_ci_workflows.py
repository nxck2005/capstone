from __future__ import annotations

import re
from pathlib import Path

import yaml


WORKFLOW_ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^[0-9a-f]{40}$")  # literal-ok: GitHub SHA length


def _load(name: str) -> tuple[dict, str]:
    path = WORKFLOW_ROOT / name
    source = path.read_text(encoding="utf-8")
    value = yaml.load(source, Loader=yaml.BaseLoader)
    assert isinstance(value, dict)
    return value, source


def _steps(value: dict) -> list[dict]:
    steps: list[dict] = []
    for job in value.get("jobs", {}).values():
        steps.extend(job.get("steps", []))
    return steps


def test_only_two_conservative_workflows_exist():
    assert sorted(path.name for path in WORKFLOW_ROOT.glob("*.yml")) == [
        "ci.yml",
        "weekly-clean-install.yml",
    ]


def test_workflows_are_read_only_and_pin_github_actions():
    for name in ("ci.yml", "weekly-clean-install.yml"):
        value, source = _load(name)
        assert value["permissions"] == {"contents": "read"}
        assert "pull_request_target" not in source
        assert "ubuntu-latest" not in source
        assert "runs-on: ubuntu-24.04" in source
        for step in _steps(value):
            uses = step.get("uses")
            if not uses:
                continue
            action, sha = uses.split("@", 1)
            assert action.startswith("actions/")
            assert PINNED_ACTION.fullmatch(sha), uses
            assert f"{action}@" in source


def test_ci_has_one_lane_and_bounded_timeouts():
    value, source = _load("ci.yml")
    assert value["jobs"]["verify"]["timeout-minutes"] == "30"
    assert "software" in source and "evidence" in source
    assert "run_one_unit" not in source
    assert "run_g8_bler_characterization" not in source
    assert "run_g8_bler_characterization_v2" not in source


def test_weekly_trigger_is_weekly_or_manual_only():
    value, _source = _load("weekly-clean-install.yml")
    triggers = value["on"]
    assert "workflow_dispatch" in triggers
    assert triggers["schedule"] == [{"cron": "17 3 * * 0"}]


def test_workflows_never_fetch_project_datasets_or_run_science():
    for name in ("ci.yml", "weekly-clean-install.yml"):
        source = (WORKFLOW_ROOT / name).read_text(encoding="utf-8")
        assert "fetch_datasets.py" not in source
        assert "run_one_unit" not in source
        assert "run_g8_bler_characterization" not in source
        assert "train_reference_classifier" not in source
