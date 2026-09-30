"""Build the research paper's figures from the published W10 validation tables.

Reads only presentation-results/data/w10_primary_252.csv and
w10_scorers_357.csv. Every plotted value is a published n_correct/1000 or
coverage_rate at a measured SNR; lines connect measured points only.

    python deliverables/research-paper/figures/make_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "presentation-results" / "data"
OUT = Path(__file__).resolve().parent

COL_W = 3.5    # IEEE single column, inches
PAGE_W = 7.16  # IEEE two-column text width

BLUE, ORANGE, VIOLET, AQUA = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(name: str, variant_key: str) -> dict[tuple[str, str, str], dict[int, dict[str, float]]]:
    table: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    with open(DATA / name, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["system"], row["bw_ratio"], row[variant_key])
            table[key][int(float(row["snr_db"]))] = row
    return table


PRIMARY = load("w10_primary_252.csv", "primary_scorer")
SCORERS = load("w10_scorers_357.csv", "classifier_variant")


def series(system: str, ratio: str = "r_1_6", field: str = "n_correct",
           scorer: str | None = None) -> tuple[list[int], list[float]]:
    source = SCORERS if scorer else PRIMARY
    matches = [v for (s, r, c), v in source.items()
               if s == system and r == ratio and (scorer is None or c == scorer)]
    if len(matches) != 1:
        raise KeyError((system, ratio, scorer))
    rows = matches[0]
    snrs = sorted(rows)
    if len(snrs) != 21:
        raise ValueError(f"{system} {ratio}: expected 21 SNR points, found {len(snrs)}")
    if field == "n_correct":
        values = [int(rows[s]["n_correct"]) / int(rows[s]["n_total"]) for s in snrs]
    else:
        values = [float(rows[s][field]) for s in snrs]
    return snrs, values


def accuracy_axes(ax, *, ylim=(0.0, 1.0), floor=True, xlim=(-8.6, 18.6)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Top-1 accuracy")
    if floor:
        ax.axhline(0.10, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)


def plot(ax, system, label, color, marker, *, ratio="r_1_6", scorer=None, ls="-", field="n_correct"):
    x, y = series(system, ratio, field, scorer)
    ax.plot(x, y, color=color, marker=marker, ls=ls, label=label, zorder=3,
            markeredgecolor="white", markeredgewidth=0.4)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)


def fig_headline() -> None:
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(COL_W, 3.3), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08})
    plot(top, "learned", "DJSCC", BLUE, "o")
    plot(top, "classical_adaptive", "JPEG 2000 + LDPC (adaptive)", ORANGE, "s")
    plot(top, "er9_digital", "Task-aware digital", VIOLET, "^")
    accuracy_axes(top)
    top.text(18.4, 0.115, "outage fallback", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    top.legend(loc="lower right", bbox_to_anchor=(1.0, 0.16))

    plot(bottom, "classical_adaptive", "JPEG 2000 + LDPC", ORANGE, "s", field="coverage_rate")
    plot(bottom, "er9_digital", "Task-aware digital", VIOLET, "^", field="coverage_rate", ls="--")
    bottom.set_ylim(-0.05, 1.08)
    bottom.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    bottom.set_ylabel("Delivered")
    bottom.set_xlabel(r"SNR, $E_s/N_0$ (dB)")
    bottom.xaxis.set_major_locator(MultipleLocator(2))
    save(fig, "fig_headline")


def fig_bandwidth() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, ratio, title in zip(axes, ("r_1_6", "r_1_24"),
                                (r"(a) $r=1/6$, $k=12{,}800$", r"(b) $r=1/24$, $k=3{,}200$")):
        plot(ax, "learned", "DJSCC", BLUE, "o", ratio=ratio)
        plot(ax, "classical_adaptive", "JPEG 2000 + LDPC (adaptive)", ORANGE, "s", ratio=ratio)
        accuracy_axes(ax)
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"SNR, $E_s/N_0$ (dB)")
    axes[1].set_ylabel("")
    axes[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
    save(fig, "fig_bandwidth")


def fig_training() -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    plot(ax, "learned", r"Fixed 7 dB training", BLUE, "o")
    plot(ax, "learned_snr_randomised", "SNR-randomised training", AQUA, "D")
    plot(ax, "learned_papr_constrained", "PAPR-capped (3 dB)", VIOLET, "v", ls="--")
    accuracy_axes(ax, ylim=(0.70, 0.86), floor=False)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.set_xlabel(r"SNR, $E_s/N_0$ (dB)")
    ax.legend(loc="lower right")
    save(fig, "fig_training")


def fig_scorer() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    ax = axes[0]
    plot(ax, "classical_adaptive", "Fine-tuned on JPEG 2000 artifacts", ORANGE, "s", scorer="artifact_finetuned")
    plot(ax, "classical_adaptive", "Clean-trained ResNet-18", ORANGE, "o", scorer="clean", ls="--")
    plot(ax, "learned", "DJSCC (own task head)", BLUE, "o")
    ax.set_title("(a) Same JPEG 2000 + LDPC outputs, two classifiers", loc="left")
    ax = axes[1]
    plot(ax, "learned", "DJSCC task head", BLUE, "o")
    plot(ax, "semantic_recon_ablation", "DJSCC reconstruction + clean ResNet-18", BLUE, "s",
         scorer="clean", ls="--")
    ax.set_title("(b) Same DJSCC transmissions, two receivers", loc="left")
    for ax in axes:
        accuracy_axes(ax)
        ax.set_xlabel(r"SNR, $E_s/N_0$ (dB)")
        ax.legend(loc="lower right")
    axes[1].set_ylabel("")
    save(fig, "fig_scorer")


def fig_controls() -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    plot(ax, "classical_adaptive", "Adaptive (reference)", ORANGE, "s")
    plot(ax, "classical_fixed_mod", "QPSK only", BLUE, "o", ls="--")
    plot(ax, "classical_fixed_mcs", "Fixed 16-QAM, rate 1/2", VIOLET, "^", ls="-.")
    plot(ax, "classical_jpeg_secondary", "Baseline JPEG codec", AQUA, "D", ls=":")
    accuracy_axes(ax)
    ax.set_xlabel(r"SNR, $E_s/N_0$ (dB)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "fig_controls")


if __name__ == "__main__":
    fig_headline()
    fig_bandwidth()
    fig_training()
    fig_scorer()
    fig_controls()
