from __future__ import annotations

import ci_classify_changes as classifier


def test_only_allowlisted_evidence_paths_use_evidence_lane():
    assert classifier.classify_paths(
        [
            "results/baseline/g8/work_units/aa/unit.attempt-1.result.json",
            "results/baseline/g8/campaign_state.json",
            "instructions/RESUME.md",
            "worklogs/w4-classical-baseline-progress.md",
        ]
    ) == "evidence"


def test_empty_changes_fail_closed_to_software():
    assert classifier.classify_paths([]) == "software"


def test_unknown_and_code_paths_fail_closed_to_software():
    for paths in (
        ["README.md"],
        ["results/baseline/g8/work_units/a.json", "src/new.py"],
        ["tools/ci_classify_changes.py"],
        [".github/workflows/ci.yml"],
    ):
        assert classifier.classify_paths(paths) == "software"


def test_path_normalisation_does_not_expand_allowlist():
    assert classifier.classify_paths(["./instructions/RESUME.md"]) == "evidence"
    assert classifier.classify_paths(["results/baseline/g8/work_units"]) == "software"
