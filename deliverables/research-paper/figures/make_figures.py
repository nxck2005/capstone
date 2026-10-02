"""Build the research paper's figures from the G-12 test-split tables.

Reads only presentation-results/data/g12_test_curves.csv and
g12_test_differences.csv, written by tools/export_g12_tables.py from the
committed G-12 closeout. Every plotted value is a measured accuracy or delivered
fraction on the 3925 test images, averaged over the seed cells run (three for
the main systems, one otherwise); shaded bands are 95% image-bootstrap
intervals. Lines connect measured points only.

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


def load() -> dict[tuple[str, str, str], dict[float, dict[str, str]]]:
    table: dict[tuple[str, str, str], dict[float, dict[str, str]]] = defaultdict(dict)
    with open(DATA / "g12_test_curves.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            table[(row["system"], row["bw_ratio"], row["classifier_variant"])][float(row["snr_db"])] = row
    return table


CURVES = load()


def series(system: str, ratio: str = "r_1_6", field: str = "accuracy",
           scorer: str | None = None) -> tuple[list[float], list[float], list[float], list[float], int]:
    matches = [v for (s, r, c), v in CURVES.items()
               if s == system and r == ratio
               and (c == scorer if scorer else next(iter(v.values()))["primary_scorer"] == "True")]
    if len(matches) != 1:
        raise KeyError((system, ratio, scorer))
    rows = matches[0]
    snrs = sorted(rows)
    if len(snrs) != 21:
        raise ValueError(f"{system} {ratio}: expected 21 SNR points, found {len(snrs)}")
    values = [float(rows[s][field]) for s in snrs]
    low = [float(rows[s]["ci_low"]) for s in snrs]
    high = [float(rows[s]["ci_high"]) for s in snrs]
    return snrs, values, low, high, int(rows[snrs[0]]["cells"])


def accuracy_axes(ax, *, ylim=(0.0, 1.0), floor=True, xlim=(-8.6, 18.6)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Top-1 accuracy")
    if floor:
        ax.axhline(0.10, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)


def plot(ax, system, label, color, marker, *, ratio="r_1_6", scorer=None, ls="-", field="accuracy", band=True):
    x, y, low, high, cells = series(system, ratio, field, scorer)
    if band and field == "accuracy" and cells > 1:
        ax.fill_between(x, low, high, color=color, alpha=0.18, lw=0, zorder=2)
    ax.plot(x, y, color=color, marker=marker, ls=ls, label=label, zorder=3,
            markeredgecolor="white", markeredgewidth=0.4)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)


def fig_headline() -> None:
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(COL_W, 3.3), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08})
    plot(top, "learned", "DJSCC", BLUE, "o")
    plot(top, "classical_adaptive", "JPEG 2000 + LDPC (adaptive)", ORANGE, "s")
    plot(top, "er9_digital", "Task-aware digital", VIOLET, "^")
    plot(top, "er9_digital_low_rate", "Task-aware digital, rate 1/5", VIOLET, "v", ls=":")
    accuracy_axes(top)
    top.text(18.4, 0.115, "outage fallback", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    top.legend(loc="lower right", bbox_to_anchor=(1.0, 0.16))

    plot(bottom, "classical_adaptive", "JPEG 2000 + LDPC", ORANGE, "s", field="coverage")
    plot(bottom, "er9_digital", "Task-aware digital", VIOLET, "^", field="coverage", ls="--")
    plot(bottom, "er9_digital_low_rate", "Task-aware digital, rate 1/5", VIOLET, "v", field="coverage", ls=":")
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
    # The two variants were trained once, in seed cell 0, so DJSCC is shown from that cell too.
    plot(ax, "learned", r"Fixed 7 dB training", BLUE, "o", field="accuracy_cell0")
    plot(ax, "learned_snr_randomised", "SNR-randomised training", AQUA, "D")
    plot(ax, "learned_papr_constrained", "PAPR-capped (3 dB)", VIOLET, "v", ls="--")
    accuracy_axes(ax, ylim=(0.70, 0.86), floor=False)
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
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
    plot(ax, "learned", "DJSCC task head", BLUE, "o", field="accuracy_cell0")
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


def fig_differences() -> None:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(DATA / "g12_test_differences.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["comparison"]].append(row)
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for comparison, label, color, marker in (
        ("learned-classical_adaptive", "DJSCC $-$ JPEG 2000 + LDPC", ORANGE, "s"),
        ("learned-er9_digital", "DJSCC $-$ task-aware digital", VIOLET, "^"),
    ):
        # Below -4 dB both digital systems deliver nothing; the 66-70 point gaps are stated in the text.
        table = sorted((r for r in rows[comparison] if float(r["snr_db"]) >= -4), key=lambda r: float(r["snr_db"]))
        x = [float(r["snr_db"]) for r in table]
        ax.fill_between(x, [100 * float(r["ci_low"]) for r in table], [100 * float(r["ci_high"]) for r in table],
                        color=color, alpha=0.2, lw=0, zorder=2)
        ax.plot(x, [100 * float(r["difference"]) for r in table], color=color, marker=marker, label=label,
                zorder=3, markeredgecolor="white", markeredgewidth=0.4)
    ax.axhline(0.0, color=MUTED, lw=0.7, zorder=1)
    ax.set_xlim(-4.6, 18.6)
    ax.set_ylim(-6, 3)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylabel("DJSCC minus other (points)")
    ax.set_xlabel(r"SNR, $E_s/N_0$ (dB)")
    ax.legend(loc="upper left")
    save(fig, "fig_differences")


if __name__ == "__main__":
    fig_headline()
    fig_bandwidth()
    fig_training()
    fig_scorer()
    fig_controls()
    fig_differences()
