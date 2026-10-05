"""Streamlit companion to the paper: one image through both links, beside the test-split curves.

Start it with ./run-streamlit-demo.sh from the repository root.

Live outputs come from the frozen r = 1/6 checkpoints through the existing demo
backend's verified inference classes (demo/backend/app.py); the figure and table
are the published G-12 test results, checked against the closeout at start-up.
Nothing here trains a model or changes scientific evidence.
"""

from __future__ import annotations

import base64
import html
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_streamlit.figure import headline  # noqa: E402
from demo_streamlit.results import TEST_IMAGES, TestResults, load  # noqa: E402

OUTAGES = {
    "decode_failure": "decoding failed",
    "codec_infeasibility": "no codec setting fits the channel budget",
    "structural_infeasibility": "the packet structure does not fit the channel budget",
}
SYSTEMS = (
    ("learned", "DJSCC"),
    ("classical_adaptive", "JPEG 2000 + LDPC (adaptive)"),
    ("er9_digital", "Task-aware digital"),
    ("er9_digital_low_rate", "Task-aware digital, rate 1/5"),
)


def signed(value: float) -> str:
    return f"{value:+g}".replace("-", "\N{MINUS SIGN}")


def points(value: float) -> str:
    """A proportion as signed percentage points, one decimal."""
    return f"{100 * value:+.1f}".replace("-", "\N{MINUS SIGN}")


def percent(value: float) -> str:
    return f"{100 * value:.1f}"


@st.cache_resource(show_spinner="Loading frozen checkpoints and published results…")
def evidence():
    from demo.backend.app import Evidence

    return Evidence()


@st.cache_resource(show_spinner=False)
def test_results() -> TestResults:
    return load()


def data_url_bytes(url: str | None) -> bytes | None:
    return base64.b64decode(url.split(",", 1)[1]) if url else None


def run_learned(image_id: str, snr: int) -> dict:
    live = evidence().live
    if not live.available:
        return {"status": "unavailable", "detail": "The frozen DJSCC checkpoint is missing or fails its SHA-256 check."}
    try:
        result = live.infer(evidence().product(image_id), snr)
    except Exception as exc:  # fail closed: show no prediction rather than a substitute
        return {"status": "unavailable", "detail": f"Inference failed ({type(exc).__name__}); no prediction was produced."}
    from demo.backend.app import CLASS_NAMES

    return {"status": "delivered", "predicted_label": CLASS_NAMES[result["label_index"]],
            "confidence": result["confidence"], "image": data_url_bytes(result["reconstruction_png_data_url"])}


def run_classical(image_id: str, snr: int) -> dict:
    classical = evidence().classical
    if not classical.available:
        return {"status": "unavailable",
                "detail": "The frozen artifact classifier or the OpenJPEG 2.5.4 codec is unavailable on this machine."}
    example = evidence().examples[image_id]
    try:
        result = classical.infer(evidence().product(image_id), label=example["label_index"], snr_db=snr)
    except Exception as exc:
        return {"status": "unavailable", "detail": f"Inference failed ({type(exc).__name__}); no prediction was produced."}
    return {**result, "image": data_url_bytes(result["image_url"])}


def setting(result: dict) -> str:
    """The per-SNR digital configuration, from the backend's detail line."""
    match = re.match(r"Frozen adaptive (\S+) / LDPC (\S+) / axis (\d+)\.", result.get("detail") or "")
    if result["status"] == "unavailable" or not match:
        return ""
    modulation, rate, axis = match.groups()
    modulation = re.sub(r"^QAM(\d+)$", r"\1-QAM", modulation.upper())
    return (f"Selected for this SNR: {html.escape(modulation)}, LDPC rate {html.escape(rate)}, "
            f"image coded at {axis}&times;{axis} px.")


def receiver(title: str, result: dict, truth: str, note: str) -> None:
    st.markdown(f'<p class="panel-title">{title}</p>', unsafe_allow_html=True)
    status = result["status"]
    if status == "delivered":
        st.image(result["image"], width="stretch")
        correct = result["predicted_label"] == truth
        st.markdown(
            f'<p class="readout">Predicted: <b>{html.escape(result["predicted_label"])}</b> '
            f'(softmax {result["confidence"]:.2f}) &middot; '
            f'<span class="{"ok" if correct else "bad"}">{"correct" if correct else "incorrect"}</span></p>',
            unsafe_allow_html=True)
    elif status in OUTAGES:
        st.markdown(f'<div class="empty">No image delivered<br><small>{OUTAGES[status]}</small></div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<p class="readout">Outage rule: <b>{html.escape(result["predicted_label"])}</b> &middot; '
            f'confidence n/a</p><p class="note">The fixed class assigned whenever nothing arrives; '
            f'it is not a prediction from this image.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">Unavailable</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="note">{html.escape(result["detail"])}</p>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="note">{note}</p>', unsafe_allow_html=True)


def table(results: TestResults, snr: float, controls: bool) -> str:
    rows = []
    for system, name in SYSTEMS:
        if not controls and system.startswith("er9"):
            continue
        p = results.point(system, snr)
        interval = f"[{percent(p.ci_low)}, {percent(p.ci_high)}]" if p.cells > 1 else "&mdash;"
        delivered = "&mdash;" if system == "learned" else percent(p.coverage)
        rows.append(f"<tr><td>{name}</td><td>{percent(p.accuracy)}</td><td>{interval}</td>"
                    f"<td>{delivered}</td><td>{p.cells}</td></tr>")
    for comparison, name in (("learned-classical_adaptive", "DJSCC &minus; JPEG 2000 + LDPC"),
                             ("learned-er9_digital", "DJSCC &minus; task-aware digital")):
        if not controls and comparison.endswith("er9_digital"):
            continue
        d = results.difference(comparison, snr)
        rows.append(f'<tr class="diff"><td>{name} (paired)</td><td>{points(d.difference)}</td>'
                    f"<td>[{points(d.ci_low)}, {points(d.ci_high)}]</td>"
                    f"<td></td><td>{d.cells}</td></tr>")
    return ('<table class="booktabs"><thead><tr><th>System</th><th>Top-1 (%)</th><th>95% CI</th>'
            "<th>Delivered (%)</th><th>Seeds</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


STYLE = """
<style>
[data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stToolbar"], footer {display: none;}
.block-container {max-width: 1040px; padding-top: 2.2rem; padding-bottom: 3rem;}
h1 {font-size: 1.85rem !important; font-weight: 600 !important; line-height: 1.25 !important;
    text-align: center; padding-bottom: 0.2rem !important;}
h2 {font-size: 1.15rem !important; font-weight: 600 !important; padding-top: 1.4rem !important;
    border-bottom: 1px solid #d9d8d4; padding-bottom: 0.25rem !important; margin-bottom: 0.8rem !important;}
.byline {text-align: center; color: #52514e; font-size: 0.95rem; margin-bottom: 1.2rem;}
.abstract {font-size: 0.95rem; line-height: 1.55; margin: 0 2.5rem 0.4rem; text-align: justify;}
.abstract b {font-variant: small-caps;}
.panel-title {font-weight: 600; margin-bottom: 0.3rem;}
.readout {font-size: 0.92rem; margin: 0.35rem 0 0.1rem;}
.note, .caption {font-size: 0.84rem; color: #52514e; line-height: 1.45; margin-top: 0.2rem;}
.caption {text-align: justify; margin: 0.4rem 0 1rem;}
.caption b {color: #0b0b0b;}
.ok {color: #1b7a52;} .bad {color: #b3261e;}
.empty {aspect-ratio: 1 / 1; border: 1px dashed #b9b8b3; display: flex; flex-direction: column;
        align-items: center; justify-content: center; color: #52514e; text-align: center;
        background: repeating-linear-gradient(45deg, #faf9f7, #faf9f7 6px, #f2f1ee 6px, #f2f1ee 12px);}
table.booktabs {border-collapse: collapse; margin: 0.6rem auto 0.4rem; font-size: 0.92rem;
                font-variant-numeric: tabular-nums;}
table.booktabs thead tr {border-top: 1.5px solid #0b0b0b; border-bottom: 0.8px solid #0b0b0b;}
table.booktabs tbody tr:last-child {border-bottom: 1.5px solid #0b0b0b;}
table.booktabs th, table.booktabs td {padding: 0.28rem 0.9rem; border: none; text-align: right;}
table.booktabs th:first-child, table.booktabs td:first-child {text-align: left;}
table.booktabs tr.diff:first-of-type td {border-top: 0.5px solid #b9b8b3;}
table.booktabs tr.diff td {font-style: italic;}
.table-caption {text-align: center; font-size: 0.84rem; color: #52514e; margin-top: 0.4rem;}
.refs {font-size: 0.8rem; color: #52514e; line-height: 1.5;}
.refs code {font-size: 0.76rem; color: #3a3936; background: #f6f5f2;}
[data-testid="stImage"] img {border-radius: 0 !important;}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="DJSCC versus adaptive digital transmission", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    results = test_results()
    examples = evidence().examples

    st.title("Deep Joint Source–Channel Coding versus Adaptive Digital Transmission for Image Classification")
    st.markdown('<p class="byline">Interactive companion to the capstone paper &middot; Imagenette-160 '
                "&middot; simulated AWGN channel &middot; bandwidth ratio <i>r</i> = 1/6 "
                "(<i>k</i> = 12,800 channel uses)</p>", unsafe_allow_html=True)
    st.markdown(
        '<p class="abstract"><b>Overview.</b> Both links carry the same image over the same simulated noisy '
        "channel with the same number of channel uses. The digital link compresses the image with JPEG 2000, "
        "protects the bits with 5G NR LDPC codes and re-selects its modulation and code rate at every SNR; a "
        "ResNet-18 classifies whatever is decoded. The DJSCC link maps the image directly to channel symbols "
        "with a neural encoder trained end-to-end, once, at 7 dB, and its decoder predicts the class. Choose "
        "an image and a channel SNR: Section 1 transmits that image through both links on this computer, and "
        "Section 2 places the operating point on the measured test-split curves.</p>",
        unsafe_allow_html=True)

    grid = results.grid
    linked = st.query_params.get("snr")
    try:
        initial = float(linked) if linked is not None and float(linked) in grid else grid[0]
    except ValueError:
        initial = grid[0]
    left, right = st.columns([3, 2], gap="large")
    with left:
        snr = st.select_slider("Channel SNR, Eₛ/N₀ (dB)", options=grid, value=initial,
                               format_func=lambda v: signed(v))
    with right:
        image_id = st.radio("Source image (training split)", options=list(examples),
                            format_func=lambda i: examples[i]["label"], horizontal=True)
    controls = st.checkbox("Include the task-aware digital control (ER-9) in Section 2", value=False)

    st.header("1. Single-image transmission")
    truth = examples[image_id]["label"]
    with st.spinner(f"Transmitting at {signed(snr)} dB…"):
        learned = run_learned(image_id, int(snr))
        classical = run_classical(image_id, int(snr))
    a, b, c = st.columns(3, gap="medium")
    with a:
        st.markdown('<p class="panel-title">(a) Transmitted image</p>', unsafe_allow_html=True)
        st.image(evidence().image_png(image_id), width="stretch")
        st.markdown(f'<p class="readout">Ground truth: <b>{html.escape(truth)}</b></p>', unsafe_allow_html=True)
    with b:
        receiver("(b) DJSCC receiver", learned, truth,
                 "Reconstruction shown for illustration; the class comes from the task head." if learned["status"] == "delivered" else "")
    with c:
        receiver("(c) JPEG 2000 + LDPC receiver", classical, truth, setting(classical))
    st.markdown(
        f'<p class="caption"><b>Fig. 1.</b> One transmission of the selected image at {signed(snr)} dB, computed '
        "on this machine with the frozen <i>r</i> = 1/6 checkpoints and deterministic keyed channel noise. "
        "The inputs are training-split examples, so these outputs illustrate the mechanism and are not "
        "evidence of held-out accuracy. Softmax values are uncalibrated. The digital link uses the "
        "configuration selected on validation data for this SNR; DJSCC was trained once and is not adapted.</p>",
        unsafe_allow_html=True)

    st.header("2. Measured accuracy on the test split")
    figure = headline(snr, controls=controls)
    st.pyplot(figure, width="stretch")
    import matplotlib.pyplot as plt

    plt.close(figure)  # Streamlit keeps no reference; without this every rerun leaks a figure.
    st.markdown(
        f'<p class="caption"><b>Fig. 2.</b> Test accuracy (top) and fraction of packets delivered (bottom) at '
        f"<i>r</i> = 1/6 on the {TEST_IMAGES:,} test images. Systems with three seed pairs show their mean with "
        "shaded 95% image-bootstrap intervals"
        + ("; the rate-1/5 task-aware variant is one seed pair" if controls else "") + ". DJSCC is scored "
        "by its own task head, JPEG 2000 by the ResNet-18 fine-tuned on JPEG 2000 artifacts. The digital baseline "
        "is re-tuned at every SNR while DJSCC is trained once at 7 dB and frozen. The dotted line is the outage "
        "rule and the dashed vertical line marks the selected SNR. Lines join measured points only.</p>",
        unsafe_allow_html=True)
    st.markdown(f'<p class="table-caption"><span style="font-variant: small-caps">Table I.</span> '
                f"Measured values at {signed(snr)} dB</p>", unsafe_allow_html=True)
    st.markdown(table(results, snr, controls), unsafe_allow_html=True)
    st.markdown(
        '<p class="caption">Paired rows give the per-image accuracy difference in percentage points, averaged '
        "over seed pairs, with a 95% image-bootstrap interval. DJSCC always produces an output, so its delivered "
        "fraction is not defined.</p>", unsafe_allow_html=True)

    st.header("Data and provenance")
    st.markdown(
        '<p class="refs">Curves and Table I: <code>presentation-results/data/g12_test_curves.csv</code> and '
        "<code>g12_test_differences.csv</code>, the tables behind the paper's figures; every plotted point is "
        "checked against the committed G-12 closeout <code>results/g12/results.csv</code> when this page starts. "
        "Fig. 2 is drawn by the paper's figure module, <code>deliverables/research-paper/figures/make_figures.py</code>. "
        "Live outputs: DJSCC checkpoint SHA-256 "
        f"<code>{evidence().live.checkpoint_id[:16]}…</code>, artifact classifier SHA-256 "
        f"<code>{evidence().classical.checkpoint_id[:16]}…</code>, both verified before loading. "
        "No model is trained and no reported metric is recomputed.</p>", unsafe_allow_html=True)


main()
