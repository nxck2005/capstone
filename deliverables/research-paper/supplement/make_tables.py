"""Write the supplement's full-grid result tables from the published W10 CSVs.

Reads presentation-results/data/w10_primary_252.csv, w10_scorers_357.csv and
results/learned/w10/w10_continuation_units_v11.json (for operating points), and
writes tables.tex next to this script. No value is typed by hand.

    python deliverables/research-paper/supplement/make_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "presentation-results" / "data"
UNITS = ROOT / "results" / "learned" / "w10" / "w10_continuation_units_v11.json"
OUT = Path(__file__).resolve().parent / "tables.tex"

SNRS = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18]

PRIMARY_ROWS = [
    ("learned", "r_1_6", "DJSCC"),
    ("learned_snr_randomised", "r_1_6", "DJSCC, SNR-rand."),
    ("learned_papr_constrained", "r_1_6", "DJSCC, PAPR-capped"),
    ("semantic_recon_ablation", "r_1_6", "DJSCC recon.\\ + clean"),
    ("classical_adaptive", "r_1_6", "J2K adaptive"),
    ("classical_fixed_mod", "r_1_6", "J2K QPSK only"),
    ("classical_fixed_mcs", "r_1_6", "J2K fixed MCS"),
    ("classical_jpeg_secondary", "r_1_6", "JPEG adaptive"),
    ("er9_digital", "r_1_6", "Task-aware digital"),
    ("label_transmission_bound", "r_1_6", "Label transmission"),
    ("learned", "r_1_24", "DJSCC ($1/24$)"),
    ("classical_adaptive", "r_1_24", "J2K adaptive ($1/24$)"),
]

DELIVERY_ROWS = [
    ("classical_adaptive", "r_1_6", "J2K adaptive"),
    ("classical_fixed_mod", "r_1_6", "J2K QPSK only"),
    ("classical_fixed_mcs", "r_1_6", "J2K fixed MCS"),
    ("classical_jpeg_secondary", "r_1_6", "JPEG adaptive"),
    ("er9_digital", "r_1_6", "Task-aware digital"),
    ("label_transmission_bound", "r_1_6", "Label transmission"),
    ("classical_adaptive", "r_1_24", "J2K adaptive ($1/24$)"),
]

SCORER_ROWS = [
    ("classical_adaptive", "r_1_6"),
    ("classical_fixed_mod", "r_1_6"),
    ("classical_fixed_mcs", "r_1_6"),
    ("classical_jpeg_secondary", "r_1_6"),
    ("classical_adaptive", "r_1_24"),
]
SCORER_NAMES = {
    ("classical_adaptive", "r_1_6"): "J2K adaptive",
    ("classical_fixed_mod", "r_1_6"): "J2K QPSK only",
    ("classical_fixed_mcs", "r_1_6"): "J2K fixed MCS",
    ("classical_jpeg_secondary", "r_1_6"): "JPEG adaptive",
    ("classical_adaptive", "r_1_24"): "J2K adaptive ($1/24$)",
}

MOD_NAMES = {"bpsk": "BPSK", "qpsk": "QPSK", "qam16": "16-QAM"}


def load(name: str, variant: str) -> dict[tuple[str, str, str], dict[int, dict[str, str]]]:
    table: dict[tuple[str, str, str], dict[int, dict[str, str]]] = {}
    with open(DATA / name, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["system"], row["bw_ratio"], row[variant])
            table.setdefault(key, {})[int(float(row["snr_db"]))] = row
    return table


def only(table, system: str, ratio: str, variant: str | None = None) -> dict[int, dict[str, str]]:
    matches = [v for (s, r, c), v in table.items()
               if s == system and r == ratio and (variant is None or c == variant)]
    if len(matches) != 1 or sorted(matches[0]) != SNRS:
        raise ValueError((system, ratio, variant))
    return matches[0]


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


def operating_point_table(units, system: str, ratio: str, caption: str, label: str) -> str:
    rows = sorted((u for u in units if u["system"] == system and u["bw_ratio"] == ratio),
                  key=lambda u: u["snr_db"])
    if [int(u["snr_db"]) for u in rows] != SNRS:
        raise ValueError((system, ratio))
    has_quality = any(u["binding"].get("quality") is not None for u in rows)
    lines = []
    for u in rows:
        b = u["binding"]
        cells = [f"${int(u['snr_db'])}$", str(b.get("encode_axis_px"))]
        if has_quality:
            cells.append(str(b.get("quality")))
        cells += [b["ldpc_rate"], MOD_NAMES[b["modulation"]],
                  f"{u['n_correct'] / 10:.1f}", f"{100 * u['coverage_rate']:.1f}"]
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


def main() -> None:
    primary = load("w10_primary_252.csv", "primary_scorer")
    scorers = load("w10_scorers_357.csv", "classifier_variant")
    units = json.loads(UNITS.read_text())["units"]

    acc_rows = []
    for system, ratio, name in PRIMARY_ROWS:
        rows = only(primary, system, ratio)
        acc_rows.append((name, [f"{int(rows[s]['n_correct']) / 10:.1f}" for s in SNRS]))

    delivery_rows = []
    for system, ratio, name in DELIVERY_ROWS:
        rows = only(primary, system, ratio)
        delivery_rows.append((name, [f"{100 * float(rows[s]['coverage_rate']):.1f}" for s in SNRS]))

    scorer_rows = []
    for system, ratio in SCORER_ROWS:
        for variant, tag in (("artifact_finetuned", "fine-tuned"), ("clean", "clean")):
            rows = only(scorers, system, ratio, variant)
            scorer_rows.append((f"{SCORER_NAMES[(system, ratio)]}, {tag}",
                                [f"{int(rows[s]['n_correct']) / 10:.1f}" for s in SNRS]))

    parts = [
        "% Generated by make_tables.py from the published W10 validation data. Do not edit.\n",
        grid_table("Validation top-1 accuracy (\\%) of every system variant at every measured SNR "
                   "(1000 images per entry, primary scorer).", "tab:s-acc", acc_rows),
        grid_table("Fraction of packets delivered (\\%) for the digital systems. "
                   "DJSCC variants always produce an output and are omitted.", "tab:s-deliv",
                   delivery_rows),
        grid_table("Accuracy (\\%) of the same digital outputs scored by the fine-tuned and the "
                   "clean ResNet-18.", "tab:s-scorer", scorer_rows),
        operating_point_table(units, "classical_adaptive", "r_1_24",
                              "Operating points of the adaptive JPEG~2000 baseline at $r=1/24$.",
                              "tab:s-op24"),
        operating_point_table(units, "classical_jpeg_secondary", "r_1_6",
                              "Operating points of the baseline JPEG system at $r=1/6$.",
                              "tab:s-opjpeg"),
        operating_point_table(units, "classical_fixed_mod", "r_1_6",
                              "Operating points of the QPSK-only JPEG~2000 baseline at $r=1/6$.",
                              "tab:s-opqpsk"),
    ]
    OUT.write_text("\n".join(parts))


if __name__ == "__main__":
    main()
