"""The demo's accuracy figure, drawn by the paper's own figure module.

`deliverables/research-paper/figures/make_figures.py` sets the paper's
matplotlib style when imported and owns the plotting helpers, so the figure on
screen is the paper's headline figure plus a marker at the selected SNR (DR-4).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deliverables/research-paper/figures"))
import make_figures as paper  # noqa: E402  (sets the paper's rcParams on import)
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402


def headline(snr_db: float, *, controls: bool) -> Figure:
    """Test accuracy (top) and delivered fraction (bottom) at r = 1/6, marked at `snr_db`."""
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 3.7), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08})
    paper.plot(top, "learned", "DJSCC", paper.BLUE, "o")
    paper.plot(top, "classical_adaptive", "JPEG 2000 + LDPC (adaptive)", paper.ORANGE, "s")
    if controls:
        paper.plot(top, "er9_digital", "Task-aware digital", paper.VIOLET, "^")
        paper.plot(top, "er9_digital_low_rate", "Task-aware digital, rate 1/5", paper.VIOLET, "v", ls=":")
    paper.accuracy_axes(top)
    # Opaque backgrounds let the SNR marker pass behind the labels instead of through them.
    opaque = {"facecolor": "white", "edgecolor": "none", "pad": 1.0}
    top.text(18.4, 0.115, "outage fallback", ha="right", va="bottom", fontsize=6.5, color=paper.MUTED,
             bbox=opaque, zorder=5)
    top.legend(loc="lower right", bbox_to_anchor=(1.0, 0.16), frameon=True, facecolor="white",
               edgecolor="none", framealpha=1.0).set_zorder(5)

    paper.plot(bottom, "classical_adaptive", "JPEG 2000 + LDPC", paper.ORANGE, "s", field="coverage")
    if controls:
        paper.plot(bottom, "er9_digital", "Task-aware digital", paper.VIOLET, "^", field="coverage", ls="--")
        paper.plot(bottom, "er9_digital_low_rate", "Task-aware digital, rate 1/5", paper.VIOLET, "v",
                   field="coverage", ls=":")
    bottom.set_ylim(-0.05, 1.08)
    bottom.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    bottom.set_ylabel("Delivered")
    bottom.set_xlabel(r"SNR, $E_s/N_0$ (dB)")
    bottom.xaxis.set_major_locator(MultipleLocator(2))

    for ax in (top, bottom):
        ax.axvline(snr_db, color=paper.INK, lw=0.8, ls=(0, (4, 2)), zorder=4)
    label = f"{snr_db:+g} dB".replace("-", "\N{MINUS SIGN}")
    left = snr_db < 15
    top.annotate(label, (snr_db, 1.0), xytext=(3 if left else -3, -2), textcoords="offset points",
                 ha="left" if left else "right", va="top", fontsize=7, color=paper.INK)
    return fig
