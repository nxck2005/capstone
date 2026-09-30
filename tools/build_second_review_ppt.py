#!/usr/bin/env python3
"""Build the editable Second Review deck, its PDF proof and slide previews.

Same scene description drives three outputs: a PowerPoint of native editable
shapes, a PDF rendered with PIL, and per-slide PNGs plus a contact sheet.

The results slides embed the committed W10 validation figures rather than
redrawing them, so the plotted curves and the deck cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import shutil

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "review-2"
PREVIEWS = OUT / "previews"
FIGDIR = ROOT / "presentation-results" / "figures" / "png"

PPTX_PATH = OUT / "semantic-communication-second-review.pptx"
PDF_PATH = OUT / "semantic-communication-second-review.pdf"
CONTACT_PATH = OUT / "semantic-communication-second-review-contact-sheet.png"

SW, SH = 13.333, 7.5
PX_W, PX_H = 1600, 900

IVORY = "F7F5EF"
PAPER = "FFFEFA"
INK = "18212B"
MUTED = "5B6570"
FAINT = "A8AFB6"
LINE = "D7D2C8"
NAVY = "203A57"
BURGUNDY = "8A3346"
GREEN = "496B5A"
AMBER = "A46A28"
PALE_NAVY = "E8EDF2"
PALE_RED = "F2E5E8"
PALE_GREEN = "E8EFEA"
PALE_AMBER = "F4EBDD"
WHITE = "FFFFFF"

TITLE_FONT = "Georgia"
BODY_FONT = "Arial"
MATH_FONT = "Cambria Math"
MONO_FONT = "Cascadia Mono"

FONT_FILES = {
    "serif": "/usr/share/fonts/Adwaito/AdwaitaSans-Regular.ttf",
    "serif_bold": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "sans": "/usr/share/fonts/Adwaito/AdwaitaSans-Regular.ttf",
    "sans_bold": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "mono": "/usr/share/fonts/Adwaito/AdwaitaMono-Regular.ttf",
}

TOTAL = 18


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def px(v: float, axis: str) -> int:
    return round(v * (PX_W / SW if axis == "x" else PX_H / SH))


def color(value: str) -> tuple[int, int, int]:
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


@dataclass
class Element:
    kind: str
    x: float
    y: float
    w: float = 0
    h: float = 0
    text: str = ""
    size: float = 16
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float = 1
    text_color: str = INK
    font: str = "sans"
    bold: bool = False
    italic: bool = False
    align: str = "left"
    valign: str = "top"
    radius: float = 0.08
    opacity: int = 255
    margin: float = 0.05
    rotation: float = 0
    src: str = ""


@dataclass
class SlideScene:
    number: int
    section: str
    title: str
    criteria: tuple[str, ...]
    citation: str = ""
    elements: list[Element] = field(default_factory=list)

    def text(self, x, y, w, h, text, **kwargs):
        self.elements.append(Element("text", x, y, w, h, text=text, **kwargs))

    def rect(self, x, y, w, h, **kwargs):
        self.elements.append(Element("rect", x, y, w, h, **kwargs))

    def line(self, x, y, w, h, **kwargs):
        self.elements.append(Element("line", x, y, w, h, **kwargs))

    def circle(self, x, y, w, h, **kwargs):
        self.elements.append(Element("circle", x, y, w, h, **kwargs))

    def arrow(self, x, y, w, h, **kwargs):
        self.elements.append(Element("arrow", x, y, w, h, **kwargs))

    def image(self, x, y, w, path, **kwargs):
        """Place a figure scaled to width w, keeping its own aspect ratio."""
        h = kwargs.pop("h", None)
        if h is None:
            with Image.open(path) as im:
                h = w * im.height / im.width
        self.elements.append(Element("image", x, y, w, h, src=str(path), **kwargs))


def add_header(scene: SlideScene) -> None:
    scene.text(0.58, 0.25, 5.4, 0.24, "CAPSTONE · SECOND REVIEW · 29 SEP–3 OCT 2026",
               size=8.5, text_color=MUTED, bold=True, font="sans")
    scene.text(11.65, 0.23, 1.05, 0.25, f"§ {scene.number:02d}", size=10,
               text_color=BURGUNDY, bold=True, align="right", font="mono")
    scene.text(0.58, 0.62, 11.9, 0.52, scene.title, size=27, bold=True,
               font="serif", text_color=INK, valign="mid")
    scene.line(0.58, 1.20, 12.15, 0, stroke=BURGUNDY, stroke_width=1.3)


def add_footer(scene: SlideScene) -> None:
    scene.line(0.58, 7.05, 12.15, 0, stroke=LINE, stroke_width=0.7)
    x = 0.58
    for criterion in scene.criteria:
        w = 0.28 + 0.064 * len(criterion)
        scene.rect(x, 7.13, w, 0.22, fill=PALE_NAVY, stroke=None, radius=0.05)
        scene.text(x + 0.07, 7.16, w - 0.14, 0.13, criterion.upper(), size=6.8,
                   text_color=NAVY, bold=True, font="sans", valign="mid")
        x += w + 0.10
    if scene.citation:
        scene.text(4.25, 7.13, 7.75, 0.20, scene.citation, size=6.4,
                   text_color=MUTED, font="sans", align="right", valign="mid")
    scene.text(12.18, 7.12, 0.52, 0.20, f"{scene.number:02d}/{TOTAL}", size=7.2,
               text_color=MUTED, font="mono", align="right", valign="mid")


def bullet(scene: SlideScene, x: float, y: float, w: float, text_value: str,
           *, size: float = 14, accent: str = BURGUNDY, lines: float = 0.52,
           text_color: str = INK) -> None:
    scene.rect(x, y + 0.13, 0.07, 0.07, fill=accent, stroke=None, radius=0)
    scene.text(x + 0.18, y, w - 0.18, lines, text_value, size=size,
               text_color=text_color, font="sans", valign="top")


def label(scene: SlideScene, x, y, w, text_value, *, fill=PALE_NAVY, ink=NAVY):
    scene.rect(x, y, w, 0.27, fill=fill, stroke=None, radius=0.04)
    scene.text(x + 0.08, y + 0.04, w - 0.16, 0.16, text_value.upper(), size=7.3,
               text_color=ink, bold=True, font="sans", valign="mid")


def card(scene: SlideScene, x, y, w, h, heading, body, *, accent=NAVY,
         body_size=12.5, fill=PAPER, tag=None):
    scene.rect(x, y, w, h, fill=fill, stroke=LINE, stroke_width=0.8, radius=0.08)
    scene.rect(x, y, 0.06, h, fill=accent, stroke=None, radius=0)
    if tag:
        label(scene, x + 0.24, y + 0.20, min(1.12, 0.35 + 0.075 * len(tag)), tag,
              fill=PALE_NAVY if accent == NAVY else PALE_RED,
              ink=accent)
        head_y = y + 0.60
    else:
        head_y = y + 0.23
    scene.text(x + 0.24, head_y, w - 0.46, 0.35, heading, size=15.5,
               bold=True, font="serif", text_color=INK)
    scene.text(x + 0.24, head_y + 0.48, w - 0.46, h - (head_y - y) - 0.65,
               body, size=body_size, font="sans", text_color=MUTED)


def stat(scene: SlideScene, x, y, w, h, value, caption, *, accent=NAVY, fill=PAPER,
         value_size=27):
    scene.rect(x, y, w, h, fill=fill, stroke=LINE, stroke_width=0.8, radius=0.08)
    scene.rect(x, y, w, 0.055, fill=accent, stroke=None, radius=0)
    scene.text(x + 0.16, y + 0.30, w - 0.32, 0.52, value, size=value_size, bold=True,
               font="serif", text_color=accent, align="center")
    scene.text(x + 0.16, y + 0.94, w - 0.32, h - 1.06, caption, size=9.4,
               font="sans", text_color=MUTED, align="center")


def table(scene: SlideScene, x, y, w, *, headers, rows, row_labels,
          label_w=1.94, row_h=0.40, head_h=0.34, accent=NAVY,
          highlight=None):
    """A light rules-only numeric table. highlight: row indices to tint."""
    ncols = len(headers)
    col_w = (w - label_w) / ncols
    highlight = highlight or {}

    scene.rect(x, y, w, head_h, fill=PALE_NAVY, stroke=None, radius=0.04)
    scene.text(x + 0.10, y + 0.07, label_w - 0.16, 0.20, "SYSTEM", size=7.2,
               bold=True, font="sans", text_color=NAVY)
    for j, head in enumerate(headers):
        cx = x + label_w + j * col_w
        scene.text(cx, y + 0.07, col_w, 0.20, head, size=7.6, bold=True,
                   font="mono", text_color=NAVY, align="center")

    for i, (rlab, values) in enumerate(zip(row_labels, rows)):
        ry = y + head_h + i * row_h
        if i in highlight:
            scene.rect(x, ry, w, row_h, fill=highlight[i][1], stroke=None, radius=0.03)
        scene.line(x, ry + row_h, w, 0, stroke=LINE, stroke_width=0.6)
        scene.text(x + 0.10, ry + 0.11, label_w - 0.16, 0.22, rlab, size=8.6,
                   font="sans", text_color=INK)
        ink = highlight[i][0] if i in highlight else MUTED
        for j, val in enumerate(values):
            cx = x + label_w + j * col_w
            scene.text(cx, ry + 0.11, col_w, 0.22, val, size=9.2,
                       font="mono", text_color=ink, align="center",
                       bold=(i in highlight))


def figure(scene: SlideScene, path_name: str, *, y=1.52, max_h=4.62, x=None,
           caption=None, cap_size=7.9):
    """Fit a committed figure into the content area, centred."""
    path = FIGDIR / path_name
    with Image.open(path) as im:
        ar = im.height / im.width
    width = min(11.55, max_h / ar)
    fx = x if x is not None else (SW - width) / 2
    scene.image(fx, y, width, path)
    if caption:
        scene.text(0.90, y + max_h * 0 + width * ar + 0.10, 11.55, 0.30, caption,
                   size=cap_size, font="sans", text_color=MUTED, align="center")
    return fx, y, width, width * ar


def build_scenes() -> list[SlideScene]:
    slides: list[SlideScene] = []

    # 1 — title
    s = SlideScene(1, "Thesis", "Task-Oriented Deep Joint Source–Channel Coding",
                   ("Results", "Presentation"),
                   "Validation evidence only · the test split has never been opened")
    add_header(s)
    s.text(0.72, 1.62, 11.86, 0.44,
           "What a camera should send when the receiver only needs the answer",
           size=19.0, text_color=NAVY, font="serif", italic=True, align="center")

    story = [
        (1.12, "EDGE CAMERA", "sees an image", NAVY, PALE_NAVY),
        (5.11, "SHORT, NOISY LINK", "cannot carry every pixel reliably", BURGUNDY, PALE_RED),
        (9.10, "RECEIVER CLASSIFIER", "needs one label back", GREEN, PALE_GREEN),
    ]
    for x, heading, body, accent, pale in story:
        s.rect(x, 2.42, 3.10, 1.30, fill=pale, stroke=None, radius=0.08)
        s.text(x + 0.22, 2.74, 2.66, 0.23, heading, size=10.5, bold=True,
               font="sans", text_color=accent, align="center")
        s.text(x + 0.22, 3.18, 2.66, 0.24, body, size=11.2,
               font="serif", text_color=INK, align="center")
    s.arrow(4.32, 3.07, 0.64, 0, stroke=FAINT, stroke_width=1.2)
    s.arrow(8.31, 3.07, 0.64, 0, stroke=FAINT, stroke_width=1.2)

    s.rect(1.05, 4.28, 11.23, 1.32, fill=PAPER, stroke=BURGUNDY,
           stroke_width=1.1, radius=0.08)
    label(s, 1.34, 4.52, 0.88, "Thesis", fill=PALE_RED, ink=BURGUNDY)
    s.text(1.33, 4.94, 10.66, 0.42,
           "Train the sender and the receiver together so the classification decision survives a short, noisy link.",
           size=16.2, font="serif", text_color=INK, align="center", valign="mid")

    s.rect(1.05, 5.86, 11.23, 0.92, fill=PALE_AMBER, stroke=None, radius=0.07)
    s.text(1.31, 6.02, 10.71, 0.62,
           "Second Review, 29 September – 3 October 2026.\nEverything shown here was measured on the validation split. No test data has been read.",
           size=11.4, font="sans", text_color=AMBER, align="center")
    add_footer(s); slides.append(s)

    # 2 — what happened since the first review
    s = SlideScene(2, "Progress", "What changed between the two reviews",
                   ("Timeline", "Results"),
                   "First Review 18–22 Aug 2026 · Second Review 29 Sep–3 Oct 2026")
    add_header(s)
    s.text(0.82, 1.44, 11.68, 0.34,
           "In August we had a working classical chain and no trained learned model. Both exist now.",
           size=13.6, font="serif", text_color=MUTED, align="center")

    rows = [
        ("Aug", "Classical baseline", "3,213 measured BLER identities, 153 BLER curves, 288,000-row validation sweep, per-SNR adaptive selection frozen", GREEN),
        ("Aug", "Learned training", "Checkpoint/resume loop, multi-seed headline models, three frozen seed cells at 1/6", GREEN),
        ("Sep", "Attribution control", "ER-9 task-aware digital features over the same LDPC chain — the control for H4", GREEN),
        ("Sep", "Robustness", "SNR-randomised training variant, compared against fixed-SNR training", GREEN),
        ("Sep", "Crossover decision", "G-10 closed on frozen validation curves: crossover observed between −5 and −4 dB", GREEN),
        ("Sep", "Peak-power constraint", "One PAPR-constrained training run, capped at 3 dB in the symbol domain", GREEN),
        ("Oct", "Validation rehearsal", "252 measurement units across 12 arms and 21 SNRs, 1,000 images each", BURGUNDY),
        ("Nov", "Test campaign", "Still sealed. Opens at G-12 once the freeze manifest is signed", FAINT),
    ]
    s.rect(0.82, 1.96, 11.68, 4.66, fill=PAPER, stroke=LINE, radius=0.08)
    for i, (when, what, detail, accent) in enumerate(rows):
        y = 2.16 + i * 0.555
        s.text(1.06, y + 0.04, 0.72, 0.22, when, size=8.6, bold=True,
               font="mono", text_color=accent)
        s.text(1.86, y, 2.42, 0.24, what, size=11.0, bold=True,
               font="serif", text_color=INK)
        s.text(4.34, y + 0.02, 7.86, 0.44, detail, size=10.0,
               font="sans", text_color=MUTED)
        if i < len(rows) - 1:
            s.line(1.06, y + 0.47, 11.14, 0, stroke=LINE, stroke_width=0.5)
    s.text(0.82, 6.74, 11.68, 0.22,
           "Everything above ran on the validation split. The test split stays sealed until gate G-12.",
           size=9.6, italic=True, font="sans", text_color=BURGUNDY, align="center")
    add_footer(s); slides.append(s)

    # 3 — objectives in completion terms (PR-8a)
    s = SlideScene(3, "Objectives", "Objectives, stated as completion criteria",
                   ("Objectives Met", "Methodology"),
                   "SPEC §2 wording · objectives are not restated at this review")
    add_header(s)
    s.text(0.82, 1.42, 11.68, 0.30,
           "The rubric asks whether objectives were met, so they are written as things we did, not results we hoped for.",
           size=12.4, font="sans", text_color=MUTED, align="center")

    left = [
        ("Build", "A learned encoder and decoder trained end to end for image classification over a simulated AWGN link."),
        ("Bandwidth-match", "Every system gets the same channel-use budget at each SNR and ratio."),
        ("Bit-account", "Filler, CRC, code-block segmentation and rate matching are charged before any source byte."),
        ("Evaluate", "At the operating point chosen by a classical-only rule, so the selection never saw the learned curve."),
        ("Report", "Paired per-image outcomes, including the outage decision, on every row."),
    ]
    s.rect(0.82, 1.92, 7.48, 4.62, fill=PAPER, stroke=LINE, radius=0.08)
    label(s, 1.08, 2.16, 1.02, "We set out to")
    for i, (heading, body) in enumerate(left):
        y = 2.66 + i * 0.755
        s.circle(1.11, y + 0.02, 0.28, 0.28, fill=NAVY, stroke=None)
        s.text(1.11, y + 0.08, 0.28, 0.14, str(i + 1), size=7.6, bold=True,
               font="sans", text_color=WHITE, align="center")
        s.text(1.52, y, 1.72, 0.24, heading, size=11.4, bold=True,
               font="serif", text_color=INK)
        s.text(3.22, y - 0.02, 4.82, 0.56, body, size=10.2,
               font="sans", text_color=MUTED)
        if i < len(left) - 1:
            s.line(1.52, y + 0.60, 6.52, 0, stroke=LINE, stroke_width=0.5)

    s.rect(8.64, 1.92, 3.86, 4.62, fill=PALE_GREEN, stroke=None, radius=0.08)
    label(s, 8.92, 2.16, 1.42, "Not the objective", fill=WHITE, ink=GREEN)
    s.text(8.92, 2.72, 3.30, 1.60,
           "“Show that the learned system beats the classical one.”",
           size=15.0, bold=True, font="serif", text_color=INK, align="center")
    s.text(8.92, 4.46, 3.30, 1.72,
           "That is a result, not an objective. A crossover and learned dominance are both complete outcomes under SPEC §2, and the rubric scores the second review on objectives we set — not on which way the curve went.",
           size=10.4, font="sans", text_color=GREEN, align="center")
    s.text(0.82, 6.68, 11.68, 0.22,
           "G-10 found a crossover, so DEC-16's fallback clause never fired and the objectives were not rewritten.",
           size=9.8, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 4 — the evidence base
    s = SlideScene(4, "Evidence base", "What was measured for this review",
                   ("Results", "Analytical Skills"),
                   "AM-98 rehearsal scope · 252 units = 12 arms × 21 SNRs")
    add_header(s)
    tiles = [
        ("252", "measurement units, each one row of the published aggregate", NAVY),
        ("12", "system arms, including five controls", BURGUNDY),
        ("21", "SNR points, −8 dB to +18 dB on the measured grid", GREEN),
        ("1,000", "validation images behind every point", AMBER),
        ("0", "test images read, at any point, ever", BURGUNDY),
    ]
    for i, (value, caption, accent) in enumerate(tiles):
        stat(s, 0.82 + i * 2.36, 1.50, 2.18, 1.72, value, caption, accent=accent)

    s.rect(0.82, 3.50, 5.78, 3.10, fill=PAPER, stroke=LINE, radius=0.08)
    label(s, 1.08, 3.74, 1.20, "Bandwidth")
    s.text(1.08, 4.22, 5.26, 0.50,
           "1/6 ratio — 12,800 channel uses per image\n1/24 ratio — 3,200 channel uses per image",
           size=11.2, font="sans", text_color=INK)
    s.text(1.08, 5.00, 5.26, 1.34,
           "These are matched channel-use budgets within a ratio. They do not match transmitted information, and the learned and digital arms carry different representations.",
           size=10.4, font="sans", text_color=MUTED)

    s.rect(6.84, 3.50, 5.68, 3.10, fill=PAPER, stroke=LINE, radius=0.08)
    label(s, 7.10, 3.74, 1.46, "Scorers")
    s.text(7.10, 4.22, 5.16, 2.12,
           "Learned arms are scored by their own task head. The classical arms are scored by the artifact-finetuned classifier; the same transmitted outputs also carry a clean-classifier stream in the published scorer CSV.\n\nSo an end-to-end learned-versus-classical gap combines communication, representation and scorer. We treat it as a whole-system comparison and say so wherever it appears.",
           size=10.4, font="sans", text_color=MUTED)
    s.rect(0.82, 6.70, 11.70, 0.30, fill=PALE_RED, stroke=None, radius=0.04)
    s.text(1.00, 6.76, 11.34, 0.18,
           "One frozen seed cell. These are observed point differences, not confidence intervals.",
           size=9.4, bold=True, font="sans", text_color=BURGUNDY, align="center")
    add_footer(s); slides.append(s)

    # 5 — headline figure
    s = SlideScene(5, "Headline", "Learned DJSCC against the adaptive classical chain",
                   ("Results",),
                   "Figure 01 · 1/6 bandwidth · all 21 measured SNRs")
    add_header(s)
    s.text(0.82, 1.36, 11.68, 0.30,
           "The learned system answers at every SNR. The digital chain is silent until −4 dB, then overtakes it.",
           size=13.2, bold=True, font="serif", text_color=NAVY, align="center")
    fx, fy, fw, fh = figure(s, "01_headline_full_snr.png", y=1.76, max_h=4.52)
    s.rect(0.90, 6.44, 11.55, 0.50, fill=PALE_NAVY, stroke=None, radius=0.05)
    s.text(1.06, 6.53, 11.23, 0.34,
           "At −8 dB: learned 72.8%, digital 10.0% (no delivery). At −4 dB: digital 83.4%, learned 79.4%. At +18 dB: digital 89.3%, learned 83.4%.",
           size=9.6, font="sans", text_color=NAVY, align="center")
    add_footer(s); slides.append(s)

    # 6 — the close-up
    s = SlideScene(6, "Low SNR", "Close-up: where the digital chain gives up",
                   ("Results", "Analytical Skills"),
                   "Figure 02 · −8 to +2 dB · measured points only")
    add_header(s)
    s.text(0.82, 1.36, 11.68, 0.30,
           "Between −5 and −4 dB the digital chain starts delivering and its accuracy jumps 73 points.",
           size=13.2, bold=True, font="serif", text_color=NAVY, align="center")
    fx, fy, fw, fh = figure(s, "02_low_snr_closeup.png", y=1.74, max_h=3.94)
    notes = [
        ("−8 dB", "Learned 72.8%. Digital coverage is 0%, so its score is the 10.0% outage fallback.", BURGUNDY),
        ("−4 dB", "Digital coverage reaches 100% and accuracy 83.4%, above the learned 79.4%.", GREEN),
        ("Above", "Coverage dips to 97.7% at −2 dB and 96.2% at +9 dB, so the digital curve is not monotone.", AMBER),
    ]
    for i, (tag, body, accent) in enumerate(notes):
        x = 0.90 + i * 3.87
        s.rect(x, 5.86, 3.72, 0.98, fill=PAPER, stroke=LINE, radius=0.06)
        s.rect(x, 5.86, 0.055, 0.98, fill=accent, stroke=None, radius=0)
        s.text(x + 0.20, 5.99, 0.90, 0.22, tag, size=10.2, bold=True,
               font="serif", text_color=accent)
        s.text(x + 0.20, 6.28, 3.34, 0.48, body, size=9.2,
               font="sans", text_color=MUTED)
    s.text(0.82, 6.90, 11.68, 0.18,
           "No SNR between −5 and −4 dB was measured, so we quote a bracket and not a threshold.",
           size=8.8, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 7 — bandwidth
    s = SlideScene(7, "Bandwidth", "Cutting the budget fourfold",
                   ("Results", "Analytical Skills"),
                   "Figure 03 · 1/24 and 1/6 · compare within each panel")
    add_header(s)
    s.text(0.82, 1.34, 11.68, 0.30,
           "At 1/24 the digital arm loses far more at low SNR than the learned arm does.",
           size=13.2, bold=True, font="serif", text_color=NAVY, align="center")
    fx, fy, fw, fh = figure(s, "03_bandwidth_efficiency.png", y=1.72, max_h=3.28)
    cards = [
        ("−8 dB, 1/6 → 1/24", "Learned drops 72.8% → 49.4%, a 23.4 point loss.", BURGUNDY),
        ("0 dB, 1/24", "Learned 78.7% against digital 69.4%.", GREEN),
        ("First overtaking", "Digital passes learned at the measured +4 dB point, then learned retakes the lead at +9 dB where digital coverage dips.", AMBER),
    ]
    for i, (heading, body, accent) in enumerate(cards):
        card(s, 0.90 + i * 3.87, 5.22, 3.72, 1.62, heading, body, accent=accent, body_size=10.0)
    s.text(0.82, 6.92, 11.68, 0.18,
           "The +9 dB recross means we should not claim one clean crossover at this ratio.",
           size=8.8, italic=True, font="sans", text_color=BURGUNDY, align="center")
    add_footer(s); slides.append(s)

    # 8 — SNR-randomised training
    s = SlideScene(8, "Robustness", "Training across SNRs instead of at one",
                   ("Results", "Analytical Skills"),
                   "Figure 04 · 1/6 · fixed-SNR versus SNR-randomised checkpoints")
    add_header(s)
    s.text(0.82, 1.36, 11.68, 0.30,
           "Drawing the training SNR uniformly lifts the whole weak-channel end of the curve.",
           size=13.2, bold=True, font="serif", text_color=NAVY, align="center")
    fx, fy, fw, fh = figure(s, "04_training_robustness.png", y=1.76, max_h=4.10)
    table(s, 1.90, 6.02, 9.54,
          headers=["−8 dB", "−4 dB", "0 dB", "+9 dB", "+18 dB"],
          rows=[["77.0", "82.4", "83.0", "84.1", "83.9"],
                ["72.8", "79.4", "81.2", "83.8", "83.4"]],
          row_labels=["SNR-randomised", "Fixed-SNR"],
          highlight={0: (GREEN, PALE_GREEN)})
    s.text(0.82, 6.92, 11.68, 0.18,
           "Two separately trained checkpoints in one seed cell. This is not an estimate of seed-to-seed training variance.",
           size=8.8, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 9 — ER-9 control
    s = SlideScene(9, "Attribution", "The control that keeps the credit honest",
                   ("Results", "Analytical Skills"),
                   "Figure 05 · ER-9 task-aware digital at the matched 1/6 budget")
    add_header(s)
    s.text(0.82, 1.36, 11.68, 0.30,
           "If we had only compared against JPEG, the low-SNR story would have been too easy to believe.",
           size=13.2, bold=True, font="serif", text_color=NAVY, align="center")
    fx, fy, fw, fh = figure(s, "05_task_aware_digital.png", y=1.76, max_h=3.86)
    s.rect(0.82, 5.82, 5.78, 1.10, fill=PAPER, stroke=LINE, radius=0.07)
    s.rect(0.82, 5.82, 0.055, 1.10, fill=BURGUNDY, stroke=None, radius=0)
    s.text(1.04, 5.94, 5.34, 0.24, "What ER-9 does", size=12.4, bold=True,
           font="serif", text_color=INK)
    s.text(1.04, 6.26, 5.34, 0.56,
           "Sends learned task features over the same LDPC and QPSK chain at the same 12,800 channel uses. It is the control that separates task-aware representation from joint coding.",
           size=9.8, font="sans", text_color=MUTED)
    s.rect(6.84, 5.82, 5.68, 1.10, fill=PALE_AMBER, stroke=None, radius=0.07)
    s.text(7.06, 5.94, 5.24, 0.24, "What it settles", size=12.4, bold=True,
           font="serif", text_color=AMBER)
    s.text(7.06, 6.26, 5.24, 0.56,
           "It holds 82.0% at every SNR from −4 dB up. Digital transmission of task features is strong once it delivers, so the low-SNR gap is about outage behaviour as well as representation. It does not isolate the LDPC code.",
           size=9.8, font="sans", text_color=INK)
    add_footer(s); slides.append(s)

    # 10 — PAPR
    s = SlideScene(10, "Peak power", "Holding the crest factor under a hard cap",
                   ("Results", "Analytical Skills"),
                   "Figure 06 · symbol-domain PAPR · 3.0 dB cap, 0.0001 dB tolerance")
    add_header(s)
    s.text(0.82, 1.36, 11.68, 0.30,
           "A PA-friendly transmitter is a real design constraint, so we trained under one.",
           size=13.2, bold=True, font="serif", text_color=NAVY, align="center")
    fx, fy, fw, fh = figure(s, "06_papr_tradeoff.png", y=1.76, max_h=3.16)
    tiles = [
        ("3.000002 dB", "max PAPR, constrained run — inside the frozen cap", GREEN),
        ("20.059284 dB", "max PAPR, unconstrained learned symbols", BURGUNDY),
        ("+3.1 pp", "accuracy gain at −8 dB, constrained versus unconstrained", NAVY),
        ("−0.2 pp", "at +18 dB — the cap costs nothing up here", AMBER),
    ]
    for i, (value, caption, accent) in enumerate(tiles):
        stat(s, 0.82 + i * 2.94, 5.18, 2.76, 1.42, value, caption, accent=accent,
             value_size=21)
    s.text(0.82, 6.72, 11.68, 0.30,
           "This is PAPR measured on channel symbols. It says nothing about amplifier or over-the-air compliance, and the constrained run is a separately trained checkpoint, so the accuracy differences are not a clean causal penalty.",
           size=9.0, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 11 — secondary controls
    s = SlideScene(11, "Controls", "Four controls, including one that went against us",
                   ("Results", "Analytical Skills"),
                   "Figure 07 · selection, codec, transmitted label, reconstruction pathway")
    add_header(s)
    s.text(0.82, 1.30, 11.68, 0.28,
           "Each panel takes one design choice away from the classical chain and shows what it was worth.",
           size=12.6, bold=True, font="serif", text_color=NAVY, align="center")
    fx, fy, fw, fh = figure(s, "07_secondary_controls.png", y=1.62, max_h=3.26)
    notes = [
        ("Fixed MCS", "Outage floor until +6 dB, then 89.0%. Per-SNR selection buys real reach.", NAVY),
        ("Fixed modulation", "Delivers later than adaptive selection, plateaus near 88.7%.", NAVY),
        ("JPEG secondary", "88.6% at +18 dB, and a genuine 0% coverage / 10.0% point at +9 dB. Kept in.", AMBER),
        ("Predicted label", "82.0% once it delivers. A transmitted label, not the oracle true label.", NAVY),
    ]
    for i, (heading, body, accent) in enumerate(notes):
        x = 0.82 + (i % 2) * 6.00
        y = 5.10 + (i // 2) * 0.90
        s.rect(x, y, 5.70, 0.80, fill=PAPER, stroke=LINE, radius=0.06)
        s.rect(x, y, 0.06, 0.80, fill=accent, stroke=None, radius=0)
        s.text(x + 0.22, y + 0.10, 5.26, 0.24, heading, size=11.6, bold=True,
               font="serif", text_color=INK)
        s.text(x + 0.22, y + 0.40, 5.26, 0.34, body, size=9.0,
               font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 12 — delivery coverage
    s = SlideScene(12, "Mechanism", "Why the digital curve breaks where it does",
                   ("Analytical Skills", "Results"),
                   "Figure 08 · delivery coverage explains the accuracy steps")
    add_header(s)
    s.text(0.82, 1.32, 7.00, 0.30,
           "Accuracy steps track delivery steps, one for one.",
           size=13.2, bold=True, font="serif", text_color=NAVY)
    fx, fy, fw, fh = figure(s, "08_delivery_coverage.png", y=1.72, max_h=4.20, x=0.86)
    s.rect(6.60, 1.72, 5.92, 4.20, fill=PAPER, stroke=LINE, radius=0.08)
    label(s, 6.86, 1.96, 1.32, "Reading it")
    body = [
        ("The 10.0% floor is a policy, not a delivery", "When the digital chain cannot deliver, every image falls back to the single class selected on validation. On an exactly stratified split that is 100 of 1,000 images, so 10.0%." , BURGUNDY),
        ("Outage is part of the result", "Coverage and accuracy are reported side by side on every row, so a reader can separate “the link failed” from “the classifier was wrong.”", NAVY),
        ("Learned output availability is contractual", "The learned contract returns a label at every SNR, which is why its curve has no floor to sit on.", GREEN),
    ]
    for i, (heading, text_v, accent) in enumerate(body):
        y = 2.42 + i * 1.16
        s.rect(6.86, y + 0.03, 0.055, 0.30, fill=accent, stroke=None, radius=0)
        s.text(7.06, y, 5.24, 0.26, heading, size=11.4, bold=True,
               font="serif", text_color=INK)
        s.text(7.06, y + 0.32, 5.24, 0.76, text_v, size=9.6,
               font="sans", text_color=MUTED)
    s.text(0.82, 6.34, 5.60, 0.60,
           "No points were inserted between measured SNRs, and no curve was smoothed to hide a step.",
           size=9.4, italic=True, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 13 — numbers table
    s = SlideScene(13, "Tables", "The numbers behind the figures",
                   ("Results", "Analytical Skills"),
                   "End-to-end top-1 accuracy, n_correct / 1000, validation split")
    add_header(s)
    s.text(0.82, 1.34, 11.68, 0.28,
           "End-to-end accuracy on the frozen validation split, counted per image including outage decisions.",
           size=11.8, font="sans", text_color=MUTED, align="center")

    s.text(0.82, 1.78, 5.78, 0.24, "1/6 BANDWIDTH  ·  12,800 CHANNEL USES", size=8.4,
           bold=True, font="mono", text_color=BURGUNDY)
    table(s, 0.82, 2.08, 5.78,
          headers=["−8", "−5", "−4", "0", "+9", "+18"],
          rows=[["72.8", "77.5", "79.4", "81.2", "83.8", "83.4"],
                ["10.0", "10.0", "83.4", "87.3", "86.5", "89.3"],
                ["77.0", "82.4", "82.4", "83.0", "84.1", "83.9"],
                ["10.0", "10.0", "82.0", "82.0", "82.0", "82.0"],
                ["75.9", "80.1", "80.5", "82.3", "83.0", "83.2"]],
          row_labels=["Learned DJSCC", "Adaptive classical", "SNR-randomised",
                      "ER-9 digital", "PAPR-constrained"],
          label_w=1.92, row_h=0.40,
          highlight={0: (NAVY, PALE_NAVY), 1: (BURGUNDY, PALE_RED)})

    s.text(6.84, 1.78, 5.68, 0.24, "1/24 BANDWIDTH  ·  3,200 CHANNEL USES", size=8.4,
           bold=True, font="mono", text_color=NAVY)
    table(s, 6.84, 2.08, 5.68,
          headers=["−8", "−4", "−2", "0", "+4", "+18"],
          rows=[["49.4", "69.3", "76.2", "78.7", "82.0", "82.8"],
                ["10.0", "10.0", "31.0", "69.4", "83.4", "87.7"]],
          row_labels=["Learned DJSCC", "Adaptive classical"],
          label_w=1.92, row_h=0.40,
          highlight={0: (NAVY, PALE_NAVY), 1: (BURGUNDY, PALE_RED)})

    s.rect(6.84, 3.30, 5.68, 1.06, fill=PALE_GREEN, stroke=None, radius=0.06)
    s.text(7.06, 3.42, 5.24, 0.24, "G-10, the crossover decision", size=12.0,
           bold=True, font="serif", text_color=GREEN)
    s.text(7.06, 3.74, 5.24, 0.52,
           "The measured sign of the learned-minus-classical gap flips between −5 dB and −4 dB at 1/6. That is an expected-direction crossover and it closes gate G-10.",
           size=9.6, font="sans", text_color=INK)

    s.rect(0.82, 4.64, 11.70, 1.86, fill=PAPER, stroke=LINE, radius=0.08)
    label(s, 1.08, 4.88, 1.62, "What the table is not")
    s.text(1.08, 5.36, 11.18, 0.94,
           "These are 21 SNR points on one grid, not 21 independent replications. Every cell comes from the same 1,000-image validation set at one frozen seed cell, so the smoothness of a curve carries no statistical information. A one- or two-point difference between rows should be described as an observed difference and nothing more. The bulk per-image rows that would support paired uncertainty intervals are not in this checkout, so we have not computed any.",
           size=10.2, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 14 — limits
    s = SlideScene(14, "Limits", "What this evidence cannot tell you",
                   ("Analytical Skills", "Methodology"),
                   "Stated plainly so the third review is not a surprise")
    add_header(s)
    items = [
        ("One seed cell, validation only", "Every number here is descriptive. No population interval, no significance claim, no replication count.", BURGUNDY),
        ("Different scorers on the two sides", "The learned arms use their own task head, the classical arms the artifact-finetuned classifier. The gap is a system-level quantity, not a controlled test of channel coding.", BURGUNDY),
        ("Not equal optimisation effort", "The classical selection rules and artifact scorer were frozen from validation work; the learned controls are separate training runs. Specific and fair, but not matched compute.", AMBER),
        ("No waveform or amplifier data", "PAPR is measured on channel symbols. Nothing here establishes RF behaviour, and the project stays simulation-first.", AMBER),
        ("Discrete, frozen choices produce steps", "Outage fallback, 1,000 images and frozen operating points give abrupt changes and a few non-monotone points. We did not smooth them.", NAVY),
    ]
    for i, (heading, body, accent) in enumerate(items):
        y = 1.50 + i * 1.06
        s.rect(0.82, y, 11.70, 0.94, fill=PAPER, stroke=LINE, radius=0.06)
        s.rect(0.82, y, 0.06, 0.94, fill=accent, stroke=None, radius=0)
        s.text(1.10, y + 0.14, 3.62, 0.26, heading, size=12.4, bold=True,
               font="serif", text_color=INK)
        s.text(4.84, y + 0.12, 7.42, 0.68, body, size=10.0,
               font="sans", text_color=MUTED)
    s.text(0.82, 6.86, 11.70, 0.20,
           "SPEC §2 treats both a crossover and learned dominance as complete Tier 1 outcomes. Neither result undoes the other.",
           size=9.2, italic=True, font="sans", text_color=BURGUNDY, align="center")
    add_footer(s); slides.append(s)

    # 15 — SPEC §17 distillation (PR-8c)
    s = SlideScene(15, "Course correction", "Four times we changed the approach because of what we found",
                   ("Methodology",),
                   "Required by PR-8(c): a one-slide distillation of SPEC §17")
    add_header(s)
    s.text(0.82, 1.34, 11.68, 0.28,
           "The rubric asks what we decided based on the results obtained. These are the four that count.",
           size=12.4, font="sans", text_color=MUTED, align="center")
    corrections = [
        ("AM-34 · DEC-16", "The baseline was capped at QPSK, which made a crossover arithmetically impossible. We added 16-QAM and made modulation adaptive per SNR, so the baseline transmits more as the link improves.", "The rule we adopted: every lever strengthens the baseline or is preregistered. We never weaken the learned system."),
        ("AM-52", "Our SNR grid was sampled every 2 dB exactly where the BPSK waterfall sits, so the classical cliff could have been smeared or missed. We added −7, −5 and −3 dB before the curve mattered.", "Dense where the effect lives, not where sampling was convenient."),
        ("AM-58", "The packetisation checker reported zero failures while violating four of its own rules, including byte alignment and exact code-block division. We fixed the checker, not the results.", "All 215 configurations stayed feasible and every scientific outcome was unchanged."),
        ("AM-60", "Test access was set to release at G-10, which sits in week 9. That opened the test split about three weeks before the freeze manifest existed.", "The release moved to G-12 in week 11. It has stayed sealed since."),
    ]
    for i, (tag, what, why) in enumerate(corrections):
        x = 0.82 + (i % 2) * 6.00
        y = 1.78 + (i // 2) * 2.52
        s.rect(x, y, 5.70, 2.32, fill=PAPER, stroke=LINE, radius=0.07)
        s.rect(x, y, 5.70, 0.055, fill=BURGUNDY, stroke=None, radius=0)
        s.text(x + 0.24, y + 0.22, 5.22, 0.24, tag, size=9.0, bold=True,
               font="mono", text_color=BURGUNDY)
        s.text(x + 0.24, y + 0.58, 5.22, 0.92, what, size=10.4,
               font="sans", text_color=INK)
        s.line(x + 0.24, y + 1.48, 5.22, 0, stroke=LINE, stroke_width=0.6)
        s.text(x + 0.24, y + 1.60, 5.22, 0.46, why, size=9.2, italic=True,
               font="sans", text_color=GREEN)
    s.rect(0.82, 6.60, 11.70, 0.34, fill=PALE_GREEN, stroke=None, radius=0.05)
    s.text(1.00, 6.67, 11.34, 0.20,
           "Twenty-two further amendments are recorded in SPEC §17, including three checks that were wrong and had to be repaired.",
           size=9.4, bold=True, font="sans", text_color=GREEN, align="center")
    add_footer(s); slides.append(s)

    # 16 — originality / novelty
    s = SlideScene(16, "Originality", "What we claim as new, and what we do not",
                   ("Originality", "Novelty"),
                   "SPEC PR-7 · DEC-13 · claims resolvable to the 30-source review")
    add_header(s)
    s.text(0.82, 1.32, 11.68, 0.28,
           "Two claims, both tied to the experimental design rather than to the idea of semantic communication.",
           size=12.4, font="sans", text_color=MUTED, align="center")

    claims = [
        ("ER-9 attribution decomposition", BURGUNDY,
         "A task-aware digital arm sends learned features over the same LDPC and modulation chain at the same channel-use budget. It separates task-aware representation from joint coding inside one experiment.",
         "Distinguished from: work that reports a learned-versus-classical gap without such a control."),
        ("BR-11 format-overhead-controlled baseline", NAVY,
         "Codec container, framing, CRC, segmentation and rate-matching overhead are measured, charged and published, so the classical arm is tuned on the payload it can actually use.",
         "Distinguished from: pipelines that quote a nominal ratio and leave the overhead unaccounted."),
    ]
    for i, (heading, accent, body, against) in enumerate(claims):
        y = 1.78 + i * 1.72
        s.rect(0.82, y, 7.48, 1.52, fill=PAPER, stroke=LINE, radius=0.07)
        s.rect(0.82, y, 0.06, 1.52, fill=accent, stroke=None, radius=0)
        s.text(1.08, y + 0.16, 6.98, 0.28, heading, size=14.0, bold=True,
               font="serif", text_color=INK)
        s.text(1.08, y + 0.54, 6.98, 0.56, body, size=10.0,
               font="sans", text_color=MUTED)
        s.text(1.08, y + 1.14, 6.98, 0.28, against, size=8.8, italic=True,
               font="sans", text_color=accent)

    s.rect(8.64, 1.78, 3.88, 3.44, fill=PALE_RED, stroke=None, radius=0.07)
    label(s, 8.92, 2.02, 1.86, "Not claimed as new", fill=WHITE, ink=BURGUNDY)
    nots = [
        "Graceful degradation under noise",
        "Avoiding the digital cliff",
        "A task-aware digital semantic system",
        "DeepJSCC as a method",
    ]
    for i, item in enumerate(nots):
        y = 2.56 + i * 0.50
        s.rect(8.92, y + 0.09, 0.09, 0.09, fill=BURGUNDY, stroke=None, radius=0)
        s.text(9.14, y, 3.16, 0.30, item, size=10.4, font="sans", text_color=INK)
    s.text(8.92, 4.62, 3.36, 0.50,
           "All four have substantial prior art. Claiming any of them would cost more than it earns.",
           size=9.6, italic=True, font="sans", text_color=BURGUNDY, align="center")

    s.rect(0.82, 5.44, 11.70, 1.34, fill=PAPER, stroke=LINE, radius=0.07)
    label(s, 1.08, 5.66, 1.50, "Negative check")
    s.text(1.08, 6.06, 11.18, 0.60,
           "Each claim above is checked against the 30 references in docs/literature-review.md, and each negative assertion in the novelty statement must resolve to that list rather than to this specification. The full PR-7 statement ships with the third review package.",
           size=10.2, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 17 — timeline
    s = SlideScene(17, "Timeline", "Where the remaining weeks went",
                   ("Timeline",),
                   "Second Review 29 Sep–3 Oct · Final Review 17–21 Nov · report due 20 Nov")
    add_header(s)
    phases = [
        ("W10", "29 Sep – 3 Oct", "Second Review", "Validation rehearsal complete; package and figures frozen", GREEN, 1.00),
        ("W11", "5 – 11 Oct", "Test campaign", "One guarded opening at G-12 after the freeze manifest is signed", BURGUNDY, 0.00),
        ("W12", "12 – 18 Oct", "Tier 1 close", "Every hypothesis decided; ER-1 to ER-4, ER-9, ER-10 frozen", BURGUNDY, 0.00),
        ("W13", "19 – 25 Oct", "Demo", "SNR slider, both pipelines side by side, frozen plot", BURGUNDY, 0.00),
        ("W14", "26 Oct – 1 Nov", "Poster + optional SDR", "Poster draft; hardware replay only if Tier 1 leaves room", MUTED, 0.00),
        ("W15", "2 – 8 Nov", "Internal freeze", "Report complete in the prescribed format", MUTED, 0.00),
        ("W16", "9 – 15 Nov", "Contingency", "Report completion and audit only. No new experiments", FAINT, 0.00),
        ("W17", "16 – 22 Nov", "Final Review", "Final review 17–21 Nov; report and supporting material due 20 Nov", BURGUNDY, 0.00),
    ]
    s.line(1.72, 1.86, 0, 4.72, stroke=LINE, stroke_width=1.2)
    for i, (week, dates, name, detail, accent, done) in enumerate(phases):
        y = 1.62 + i * 0.60
        s.text(0.82, y + 0.06, 0.62, 0.22, week, size=9.0, bold=True,
               font="mono", text_color=accent, align="right")
        s.circle(1.62, y + 0.08, 0.20, 0.20, fill=accent if accent != FAINT else PAPER,
                 stroke=accent, stroke_width=1.4)
        s.text(1.98, y, 1.72, 0.24, dates, size=9.4, font="mono", text_color=MUTED)
        s.text(3.76, y - 0.02, 2.30, 0.26, name, size=11.6, bold=True,
               font="serif", text_color=INK)
        s.text(6.14, y, 5.30, 0.28, detail, size=9.8, font="sans", text_color=MUTED)
        s.text(11.62, y, 0.90, 0.24, "done" if done else "planned", size=8.4,
               bold=True, font="sans",
               text_color=GREEN if done else FAINT, align="right")
    s.rect(0.82, 6.52, 11.70, 0.42, fill=PALE_AMBER, stroke=None, radius=0.05)
    s.text(1.00, 6.60, 11.34, 0.26,
           "The report deadline falls inside Final Review week, which is why W15 carries an internal freeze and W16 holds no experimental scope.",
           size=9.4, font="sans", text_color=AMBER, align="center")
    add_footer(s); slides.append(s)

    # 18 — close
    s = SlideScene(18, "Close", "Where this leaves the project",
                   ("Results", "Methodology"),
                   "Questions welcome — the detail is in the repository, not on this slide")
    add_header(s)
    s.rect(0.82, 1.50, 11.70, 1.44, fill=PAPER, stroke=BURGUNDY,
           stroke_width=1.1, radius=0.08)
    s.text(1.14, 1.74, 11.06, 0.94,
           "A learned system trained for the task keeps answering when the link is too poor for the digital chain to deliver anything. Once the digital chain delivers, it is the more accurate system, and the curves cross between −5 and −4 dB.",
           size=15.4, font="serif", text_color=INK, align="center", valign="mid")

    nexts = [
        ("Next", "One guarded test campaign at G-12, once the freeze manifest is signed. Test access is 0 today.", BURGUNDY),
        ("Then", "Every hypothesis decided on the test split, and Tier 1 frozen at G-5.", NAVY),
        ("Later", "Tier 2 SDR replay and the live demo are stretch. The project does not depend on them.", MUTED),
    ]
    for i, (heading, body, accent) in enumerate(nexts):
        card(s, 0.82 + i * 3.94, 3.20, 3.72, 1.44, heading, body,
             accent=accent, body_size=10.2)

    s.rect(0.82, 4.90, 11.70, 1.06, fill=PALE_NAVY, stroke=None, radius=0.07)
    label(s, 1.08, 5.10, 1.72, "Evidence trail")
    s.text(1.08, 5.50, 11.18, 0.34,
           "252-unit aggregate and per-image manifest in results/learned/w10/ · figures and CSVs in presentation-results/ · requirements and amendments in spec/SPEC.md §17",
           size=10.0, font="mono", text_color=NAVY, align="center")
    s.rect(0.82, 6.16, 11.70, 0.62, fill=PALE_RED, stroke=None, radius=0.06)
    s.text(1.06, 6.28, 11.22, 0.40,
           "The test split has not been opened. test_access = 0 across every gate, every run and every artifact in this project.",
           size=11.0, bold=True, font="sans", text_color=BURGUNDY, align="center")
    add_footer(s); slides.append(s)

    return slides


def ppt_font(el: Element) -> str:
    return {"serif": TITLE_FONT, "sans": BODY_FONT, "math": MATH_FONT, "mono": MONO_FONT}.get(el.font, BODY_FONT)


def add_to_ppt(slide, el: Element) -> None:
    if el.kind == "text":
        box = slide.shapes.add_textbox(Inches(el.x), Inches(el.y), Inches(el.w), Inches(el.h))
        tf = box.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = Inches(el.margin)
        tf.margin_right = Inches(el.margin)
        tf.margin_top = Inches(el.margin)
        tf.margin_bottom = Inches(el.margin)
        tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "mid": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}.get(el.valign, MSO_ANCHOR.TOP)
        p = tf.paragraphs[0]
        p.text = el.text
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}.get(el.align, PP_ALIGN.LEFT)
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        p.line_spacing = 1.0
        for run in p.runs:
            run.font.name = ppt_font(el)
            run.font.size = Pt(el.size)
            run.font.bold = el.bold
            run.font.italic = el.italic
            run.font.color.rgb = rgb(el.text_color)
        box.rotation = el.rotation
    elif el.kind == "image":
        slide.shapes.add_picture(el.src, Inches(el.x), Inches(el.y),
                                 Inches(el.w), Inches(el.h))
    elif el.kind in {"rect", "circle"}:
        shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if el.kind == "rect" and el.radius else MSO_SHAPE.RECTANGLE
        if el.kind == "circle":
            shape_type = MSO_SHAPE.OVAL
        shape = slide.shapes.add_shape(shape_type, Inches(el.x), Inches(el.y), Inches(el.w), Inches(el.h))
        if el.fill:
            shape.fill.solid(); shape.fill.fore_color.rgb = rgb(el.fill)
        else:
            shape.fill.background()
        if el.stroke:
            shape.line.color.rgb = rgb(el.stroke); shape.line.width = Pt(el.stroke_width)
        else:
            shape.line.fill.background()
    elif el.kind in {"line", "arrow"}:
        line = slide.shapes.add_connector(1, Inches(el.x), Inches(el.y), Inches(el.x + el.w), Inches(el.y + el.h))
        line.line.color.rgb = rgb(el.stroke or INK)
        line.line.width = Pt(el.stroke_width)
        if el.kind == "arrow":
            angle = math.atan2(el.h, el.w) if el.w or el.h else 0
            tip_x, tip_y = el.x + el.w, el.y + el.h
            tri = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                                         Inches(tip_x - 0.07), Inches(tip_y - 0.055),
                                         Inches(0.14), Inches(0.11))
            tri.rotation = math.degrees(angle) + 90
            tri.fill.solid(); tri.fill.fore_color.rgb = rgb(el.stroke or INK)
            tri.line.fill.background()


def build_pptx(scenes: list[SlideScene]) -> None:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    prs.core_properties.title = "Task-Oriented Deep Joint Source–Channel Coding — Second Review"
    prs.core_properties.subject = "Capstone Second Review, 29 September – 3 October 2026"
    prs.core_properties.author = "Capstone project team"
    prs.core_properties.comments = "W10 validation evidence only; test split sealed. Editable native shapes and text."
    blank = prs.slide_layouts[6]
    for scene in scenes:
        slide = prs.slides.add_slide(blank)
        bg = slide.background.fill
        bg.solid(); bg.fore_color.rgb = rgb(IVORY)
        for el in scene.elements:
            add_to_ppt(slide, el)
    prs.save(PPTX_PATH)


def pil_font(el: Element) -> ImageFont.FreeTypeFont:
    if el.font in {"serif", "math"}:
        key = "serif_bold" if el.bold else "serif"
    elif el.font == "mono":
        key = "mono"
    else:
        key = "sans_bold" if el.bold else "sans"
    return ImageFont.truetype(FONT_FILES[key], max(8, round(el.size * PX_H / 540)))


def wrap_for_draw(draw: ImageDraw.ImageDraw, text_value: str, font, max_width: int) -> str:
    wrapped: list[str] = []
    for raw in text_value.split("\n"):
        if not raw:
            wrapped.append("")
            continue
        words = raw.split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                line = candidate
            else:
                if line:
                    wrapped.append(line)
                line = word
        if line:
            wrapped.append(line)
    return "\n".join(wrapped)


def render_element(draw: ImageDraw.ImageDraw, el: Element, canvas: Image.Image) -> None:
    x, y, w, h = px(el.x, "x"), px(el.y, "y"), px(el.w, "x"), px(el.h, "y")
    if el.kind == "rect":
        r = max(0, px(el.radius, "y"))
        draw.rounded_rectangle((x, y, x + w, y + h), radius=r,
                               fill=color(el.fill) if el.fill else None,
                               outline=color(el.stroke) if el.stroke else None,
                               width=max(1, round(el.stroke_width * 1.5)))
    elif el.kind == "circle":
        draw.ellipse((x, y, x + w, y + h), fill=color(el.fill) if el.fill else None,
                     outline=color(el.stroke) if el.stroke else None,
                     width=max(1, round(el.stroke_width * 1.5)))
    elif el.kind == "image":
        with Image.open(el.src) as im:
            im = im.convert("RGB")
            resized = im.resize((max(1, w), max(1, h)), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x, y))
    elif el.kind in {"line", "arrow"}:
        x2, y2 = px(el.x + el.w, "x"), px(el.y + el.h, "y")
        draw.line((x, y, x2, y2), fill=color(el.stroke or INK), width=max(1, round(el.stroke_width * 2)))
        if el.kind == "arrow":
            angle = math.atan2(y2 - y, x2 - x)
            length = 11
            pts = [(x2, y2),
                   (x2 - length * math.cos(angle - 0.48), y2 - length * math.sin(angle - 0.48)),
                   (x2 - length * math.cos(angle + 0.48), y2 - length * math.sin(angle + 0.48))]
            draw.polygon(pts, fill=color(el.stroke or INK))
    elif el.kind == "text":
        font = pil_font(el)
        margin_x, margin_y = px(el.margin, "x"), px(el.margin, "y")
        max_width = max(4, w - 2 * margin_x)
        rendered = wrap_for_draw(draw, el.text, font, max_width)
        bbox = draw.multiline_textbbox((0, 0), rendered, font=font, spacing=2, align=el.align)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x + margin_x
        if el.align == "center": tx = x + (w - tw) / 2
        elif el.align == "right": tx = x + w - margin_x - tw
        ty = y + margin_y
        if el.valign == "mid": ty = y + (h - th) / 2
        elif el.valign == "bottom": ty = y + h - margin_y - th
        draw.multiline_text((tx, ty), rendered, font=font, fill=color(el.text_color),
                            spacing=2, align=el.align)


def render_previews(scenes: list[SlideScene]) -> list[Path]:
    if PREVIEWS.exists():
        shutil.rmtree(PREVIEWS)
    PREVIEWS.mkdir(parents=True)
    paths: list[Path] = []
    images: list[Image.Image] = []
    for scene in scenes:
        img = Image.new("RGB", (PX_W, PX_H), color(IVORY))
        draw = ImageDraw.Draw(img)
        for el in scene.elements:
            render_element(draw, el, img)
        path = PREVIEWS / f"slide-{scene.number:02d}.png"
        img.save(path, quality=95)
        paths.append(path)
        images.append(img)
    images[0].save(PDF_PATH, "PDF", resolution=144.0, save_all=True,
                   append_images=images[1:])
    cols, rows = 3, 6
    thumb_w, thumb_h = 533, 300
    sheet = Image.new("RGB", (thumb_w * cols, thumb_h * rows), color("D6D3CC"))
    for i, img in enumerate(images):
        thumb = img.resize((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((i % cols) * thumb_w + 4, (i // cols) * thumb_h + 4))
    sheet.save(CONTACT_PATH, quality=95)
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes()
    assert len(scenes) == TOTAL, f"expected {TOTAL} slides, built {len(scenes)}"
    build_pptx(scenes)
    paths = render_previews(scenes)
    print(f"wrote {PPTX_PATH}")
    print(f"wrote {PDF_PATH}")
    print(f"wrote {CONTACT_PATH}")
    print(f"wrote {len(paths)} slide previews under {PREVIEWS}")


if __name__ == "__main__":
    main()