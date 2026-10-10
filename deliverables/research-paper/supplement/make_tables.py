"""Write the supplement's full-grid test-result tables from the committed G-12 outputs.

Reads presentation-results/data/g12_test_curves.csv and g12_test_differences.csv
(written by tools/export_g12_tables.py), results/g12/results.csv and
results/g12/analysis.json (the committed G-12 closeout), and, for the encode
size of each frozen operating point, results/learned/w10/w10_continuation_units_v11.json.
Writes tables.tex next to this script. No value is typed by hand.

    python deliverables/research-paper/supplement/make_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "presentation-results" / "data"
RESULTS = ROOT / "results" / "g12" / "results.csv"
ANALYSIS = ROOT / "results" / "g12" / "analysis.json"
UNITS = ROOT / "results" / "learned" / "w10" / "w10_continuation_units_v11.json"
OUT = Path(__file__).resolve().parent / "tables.tex"

SNRS = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18]

PRIMARY_ROWS = [
    ("learned", "r_1_6", "DJSCC (3)"),
    ("learned_snr_randomised", "r_1_6", "DJSCC, SNR-rand."),
    ("learned_papr_constrained", "r_1_6", "DJSCC, PAPR-capped"),
    ("semantic_recon_ablation", "r_1_6", "DJSCC recon.\\ + clean"),
    ("classical_adaptive", "r_1_6", "J2K adaptive (3)"),
    ("classical_fixed_mod", "r_1_6", "J2K QPSK only"),
    ("classical_fixed_mcs", "r_1_6", "J2K fixed MCS (3)"),
    ("classical_jpeg_secondary", "r_1_6", "JPEG adaptive"),
    ("er9_digital", "r_1_6", "Task-aware digital (3)"),
    ("er9_digital_low_rate", "r_1_6", "Task-aware, rate 1/5"),
    ("label_transmission_bound", "r_1_6", "Label transmission"),
    ("learned", "r_1_24", "DJSCC ($1/24$)"),
    ("classical_adaptive", "r_1_24", "J2K adaptive ($1/24$)"),
]

DELIVERY_ROWS = [
    ("classical_adaptive", "r_1_6", "J2K adaptive (3)"),
    ("classical_fixed_mod", "r_1_6", "J2K QPSK only"),
    ("classical_fixed_mcs", "r_1_6", "J2K fixed MCS (3)"),
    ("classical_jpeg_secondary", "r_1_6", "JPEG adaptive"),
    ("er9_digital", "r_1_6", "Task-aware digital (3)"),
    ("er9_digital_low_rate", "r_1_6", "Task-aware, rate 1/5"),
    ("label_transmission_bound", "r_1_6", "Label transmission"),
    ("classical_adaptive", "r_1_24", "J2K adaptive ($1/24$)"),
]

SCORER_ROWS = [
    ("classical_adaptive", "r_1_6", "J2K adaptive (3)"),
    ("classical_fixed_mod", "r_1_6", "J2K QPSK only"),
    ("classical_fixed_mcs", "r_1_6", "J2K fixed MCS (3)"),
    ("classical_jpeg_secondary", "r_1_6", "JPEG adaptive"),
    ("classical_adaptive", "r_1_24", "J2K adaptive ($1/24$)"),
]

SEED_ROWS = [
    ("learned", "r_1_6", "DJSCC"),
    ("classical_adaptive", "r_1_6", "J2K adaptive"),
    ("classical_fixed_mcs", "r_1_6", "J2K fixed MCS"),
    ("er9_digital", "r_1_6", "Task-aware digital"),
]

PRIMARY_SCORER = {
    "learned": "own_task_head", "learned_snr_randomised": "own_task_head",
    "learned_papr_constrained": "own_task_head", "semantic_recon_ablation": "clean",
    "er9_digital": "own_task_head", "er9_digital_low_rate": "own_task_head",
    "label_transmission_bound": "predicted_label",
}

MOD_NAMES = {"bpsk": "BPSK", "qpsk": "QPSK", "qam16": "16-QAM"}


def load_curves() -> dict[tuple[str, str, str], dict[int, dict[str, str]]]:
    table: dict[tuple[str, str, str], dict[int, dict[str, str]]] = {}
    with open(DATA / "g12_test_curves.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["system"], row["bw_ratio"], row["classifier_variant"])
            table.setdefault(key, {})[int(float(row["snr_db"]))] = row
    for rows in table.values():
        if sorted(rows) != SNRS:
            raise ValueError("a test curve does not cover the SNR grid")
    return table


def curve(curves, system: str, ratio: str, scorer: str | None = None) -> dict[int, dict[str, str]]:
    return curves[(system, ratio, scorer or PRIMARY_SCORER.get(system, "artifact_finetuned"))]


def pct(value: str | float) -> str:
    return f"{100 * float(value):.1f}"


def signed(value: float, digits: int = 1) -> str:
    """A signed number with a typeset minus sign."""
    return f"{value:+.{digits}f}".replace("-", "$-$")


def grid_table(caption: str, label: str, rows: list[tuple[str, list[str]]]) -> str:
    cols = "l" + "r" * len(SNRS)
    head = " & ".join(["System"] + [f"${s}$" for s in SNRS])
    body = "\n".join(" & ".join([name] + cells) + r" \\" for name, cells in rows)
    return (
        "\\begin{table*}[!htbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.2pt}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n\\toprule\n"
        f"& \\multicolumn{{{len(SNRS)}}}{{c}}{{SNR $E_s/N_0$ (dB)}} \\\\\n"
        f"\\cmidrule(l){{2-{len(SNRS) + 1}}}\n{head} \\\\\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def operating_point_table(curves, results, units, system: str, ratio: str, caption: str, label: str) -> str:
    """Frozen operating points (checked against the test rows) with test accuracy and delivery."""

    frozen = {int(u["snr_db"]): u["binding"] for u in units if u["system"] == system and u["bw_ratio"] == ratio}
    tested = {int(float(r["test_snr_db"])): r for r in results
              if r["system"] == system and r["bw_ratio"] == ratio and r["train_seed"] == "0"
              and r["classifier_variant"] == "artifact_finetuned"}
    if sorted(frozen) != SNRS or sorted(tested) != SNRS:
        raise ValueError((system, ratio))
    has_quality = any(b.get("quality") is not None for b in frozen.values())
    measured = curve(curves, system, ratio)
    lines = []
    for snr in SNRS:
        b, r = frozen[snr], tested[snr]
        if (b["ldpc_rate"], b["modulation"]) != (r["ldpc_rate"], r["modulation"]):
            raise ValueError(f"test operating point differs from the frozen one: {system} {ratio} {snr}")
        cells = [f"${snr}$", str(b.get("encode_axis_px"))]
        if has_quality:
            cells.append(str(b.get("quality")))
        cells += [b["ldpc_rate"], MOD_NAMES[b["modulation"]], pct(measured[snr]["accuracy"]), pct(measured[snr]["coverage"])]
        lines.append(" & ".join(cells) + r" \\")
    head = ["SNR (dB)", "Image (px)"] + (["JPEG $Q$"] if has_quality else []) + \
        ["LDPC", "Mod.", "Acc.\\ (\\%)", "Deliv.\\ (\\%)"]
    cols = "r" * len(head)
    return (
        "\\begin{table}[!htbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n\\toprule\n"
        + " & ".join(head) + " \\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def difference_table() -> str:
    rows: dict[str, dict[int, dict[str, str]]] = {}
    with open(DATA / "g12_test_differences.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["comparison"], {})[int(float(row["snr_db"]))] = row
    names = {"learned-classical_adaptive": "J2K adaptive", "learned-er9_digital": "Task-aware digital"}
    lines = []
    for comparison, name in names.items():
        table = rows[comparison]
        for snr in SNRS:
            r = table[snr]
            lines.append((name if snr == SNRS[0] else "", snr, signed(100 * float(r["difference"])),
                          f"[{signed(100 * float(r['ci_low']))}, {signed(100 * float(r['ci_high']))}]"))
    half = len(SNRS)
    body = "\n".join(
        f"${a[1]}$ & {a[2]} & {a[3]} & {b[2]} & {b[3]} \\\\"
        for a, b in zip(lines[:half], lines[half:])
    )
    return (
        "\\begin{table}[!htbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Paired test-accuracy difference, DJSCC minus each digital system, at $r=1/6$, "
        "in points, averaged over three seed pairs within each image, with 95\\% image-bootstrap intervals "
        "(10,000 resamples, shared with the hypothesis tests).}\n\\label{tab:s-diff}\n"
        "\\begin{tabular}{@{}rrrrr@{}}\n\\toprule\n"
        "& \\multicolumn{2}{c}{vs.\\ J2K adaptive} & \\multicolumn{2}{c}{vs.\\ task-aware digital} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
        "SNR (dB) & $\\Delta$ & 95\\% CI & $\\Delta$ & 95\\% CI \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def hypothesis_point_table(analysis) -> str:
    """Per-SNR studentized statistics behind the H1 and H4 run rules."""

    lines = []
    h1, h4 = analysis["H1"], analysis["H4"]
    if h1["region_snr_db"] != h4["region_snr_db"]:
        raise ValueError("H1 and H4 regions differ")
    for i, snr in enumerate(h1["region_snr_db"]):
        cells = [f"${int(snr)}$"]
        for h in (h1, h4):
            cells += [signed(100 * h["per_point_mean"][i]), signed(h["per_point_t"][i]).lstrip("+"), "\\checkmark" if h["qualified"][i] else ""]
        lines.append(" & ".join(cells) + r" \\")
    return (
        "\\begin{table}[!htbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Point statistics for H1 (DJSCC vs.\\ adaptive JPEG~2000) and H4 (DJSCC vs.\\ task-aware digital) "
        "over the region at or below 7~dB: mean paired difference (points), studentized statistic $t$, and whether "
        "the point qualifies ($t>1.96$). "
        f"H1: longest run {h1['r_obs']}, calibrated $p={h1['calibrated_p']:.4f}$. "
        f"H4: longest run {h4['r_obs']}, calibrated $p={h4['calibrated_p']:.4f}$.}}\n\\label{{tab:s-points}}\n"
        "\\begin{tabular}{@{}rrrcrrc@{}}\n\\toprule\n"
        "& \\multicolumn{3}{c}{H1} & \\multicolumn{3}{c}{H4} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
        "SNR (dB) & $\\Delta$ & $t$ & Q & $\\Delta$ & $t$ & Q \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def main() -> None:
    curves = load_curves()
    with open(RESULTS, newline="") as handle:
        results = list(csv.DictReader(handle))
    analysis = json.loads(ANALYSIS.read_text())
    if analysis["split"] != "test":
        raise ValueError("the analysis is not the test closeout")
    units = json.loads(UNITS.read_text())["units"]

    acc_rows = [(name, [pct(curve(curves, s, r)[x]["accuracy"]) for x in SNRS]) for s, r, name in PRIMARY_ROWS]
    delivery_rows = [(name, [pct(curve(curves, s, r)[x]["coverage"]) for x in SNRS]) for s, r, name in DELIVERY_ROWS]
    scorer_rows = []
    for system, ratio, name in SCORER_ROWS:
        for variant, tag in (("artifact_finetuned", "fine-tuned"), ("clean", "clean")):
            scorer_rows.append((f"{name}, {tag}", [pct(curve(curves, system, ratio, variant)[x]["accuracy"]) for x in SNRS]))
    seed_rows = []
    for system, ratio, name in SEED_ROWS:
        rows = curve(curves, system, ratio)
        for cell in range(3):
            seed_rows.append((f"{name}, seed {cell}", [pct(rows[x][f"accuracy_cell{cell}"]) for x in SNRS]))

    parts = [
        "% Generated by make_tables.py from the committed G-12 test results. Do not edit.\n",
        grid_table("Test top-1 accuracy (\\%) of every system variant at every measured SNR "
                   "(3925 images per entry, primary scorer). ``(3)'' marks the mean over three seed pairs; "
                   "all other rows are the first seed pair.", "tab:s-acc", acc_rows),
        grid_table("Test top-1 accuracy (\\%) of each seed pair for the systems evaluated in all three. "
                   "For the digital codecs the seed pairs differ only in the channel noise.", "tab:s-seeds", seed_rows),
        grid_table("Fraction of packets delivered (\\%) on the test split for the digital systems. "
                   "DJSCC variants always produce an output and are omitted.", "tab:s-deliv", delivery_rows),
        grid_table("Test accuracy (\\%) of the same digital outputs scored by the fine-tuned and the "
                   "clean ResNet-18.", "tab:s-scorer", scorer_rows),
        difference_table(),
        hypothesis_point_table(analysis),
        operating_point_table(curves, results, units, "classical_adaptive", "r_1_6",
                              "Operating points of the adaptive JPEG~2000 baseline at $r=1/6$, selected on validation "
                              "data, with test accuracy and delivered fraction.", "tab:s-op6"),
        operating_point_table(curves, results, units, "classical_adaptive", "r_1_24",
                              "Operating points of the adaptive JPEG~2000 baseline at $r=1/24$, selected on validation "
                              "data, with test accuracy and delivered fraction.", "tab:s-op24"),
        operating_point_table(curves, results, units, "classical_jpeg_secondary", "r_1_6",
                              "Operating points of the baseline JPEG system at $r=1/6$, selected on validation data, "
                              "with test accuracy and delivered fraction.", "tab:s-opjpeg"),
        operating_point_table(curves, results, units, "classical_fixed_mod", "r_1_6",
                              "Operating points of the QPSK-only JPEG~2000 baseline at $r=1/6$, selected on validation "
                              "data, with test accuracy and delivered fraction.", "tab:s-opqpsk"),
    ]
    OUT.write_text("\n".join(parts))


if __name__ == "__main__":
    main()
