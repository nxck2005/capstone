"""Streamlit demo: published test-curve parity and a headless page run (not scientific evaluation)."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from demo_streamlit.results import CLOSEOUT_CSV, CURVES_CSV, DIFFERENCES_CSV, ROOT, load


def test_curves_match_the_g12_closeout() -> None:
    results = load()
    assert len(results.grid) == 21 and results.grid[0] == -8.0 and results.grid[-1] == 18.0
    learned = results.point("learned", -8.0)
    assert learned.cells == 3 and learned.coverage == 1.0
    assert learned.ci_low < learned.accuracy < learned.ci_high
    # Below -4 dB the digital link delivers nothing and scores only the outage class.
    assert results.point("classical_adaptive", -8.0).coverage == 0.0
    assert results.point("er9_digital_low_rate", -8.0).cells == 1


def _copy_tables(tmp_path: Path) -> Path:
    for name in (CURVES_CSV, DIFFERENCES_CSV, CLOSEOUT_CSV):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    return tmp_path


def _rewrite(path: Path, edit) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    edit(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(("table", "field"), [(CURVES_CSV, "accuracy"), (CURVES_CSV, "coverage"),
                                             (DIFFERENCES_CSV, "difference")])
def test_a_tampered_exported_value_is_refused(tmp_path: Path, table: str, field: str) -> None:
    root = _copy_tables(tmp_path)

    def edit(rows: list[dict[str, str]]) -> None:
        key = "comparison" if table == DIFFERENCES_CSV else "system"
        target = "learned-classical_adaptive" if table == DIFFERENCES_CSV else "classical_adaptive"
        row = next(r for r in rows if r["bw_ratio"] == "r_1_6" and r[key] == target
                   and r.get("classifier_variant", "artifact_finetuned") == "artifact_finetuned")
        row[field] = str(float(row[field]) + 0.001)

    _rewrite(root / table, edit)
    with pytest.raises(RuntimeError):
        load(root)


def test_a_non_test_closeout_row_is_refused(tmp_path: Path) -> None:
    root = _copy_tables(tmp_path)
    _rewrite(root / CLOSEOUT_CSV, lambda rows: rows[0].update(split="val"))
    with pytest.raises(RuntimeError, match="not a full test-split row"):
        load(root)


def test_page_renders_headlessly() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    page = testing.AppTest.from_file(str(ROOT / "demo_streamlit/app.py"), default_timeout=180)
    page.run()
    assert not page.exception
    assert [h.value for h in page.header][:2] == ["1. Single-image transmission", "2. Measured accuracy on the test split"]
    page.select_slider[0].set_value(18.0).run()
    assert not page.exception
    assert any("Measured values at +18 dB" in m.value for m in page.markdown)
