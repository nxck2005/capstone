#!/usr/bin/env python3
"""Build the editable Second Review deck, its PDF proof and slide previews.

Same scene description drives three outputs: a PowerPoint of native editable
shapes, a PDF rendered with PIL, and per-slide PNGs plus a contact sheet.

Visual style follows deliverables/review-1/semantic-communication-first-review.pptx:
white ground, Arial throughout, black and grey type, and no colour accents.
Semantic emphasis comes from weight, rules and label chips instead.

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

IVORY = "FFFFFF"
PAPER = "FFFFFF"
INK = "111111"
MUTED = "555555"
FAINT = "999999"
LINE = "D9D9D9"
NAVY = "111111"
BURGUNDY = "111111"
GREEN = "333333"
AMBER = "555555"
PALE_NAVY = "F3F3F3"
PALE_RED = "F3F3F3"
PALE_GREEN = "F3F3F3"
PALE_AMBER = "F3F3F3"
WHITE = "FFFFFF"

TITLE_FONT = "Arial"
BODY_FONT = "Arial"
MATH_FONT = "Arial"
# The plain First Review deck sets every run in Arial and uses no monospace at
# all, so mono maps to Arial too rather than introducing a second typeface.
MONO_FONT = "Arial"

FONT_FILES = {
    "serif": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "serif_bold": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "sans": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "sans_bold": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "mono": "/usr/share/fonts/Adwaito/AdwaitaMono-Regular.ttf",
}

TOTAL = 19


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
               size=9.5, text_color=MUTED, bold=True, font="sans")
    scene.text(11.65, 0.23, 1.05, 0.25, f"§ {scene.number:02d}", size=11,
               text_color=BURGUNDY, bold=True, align="right", font="mono")
    scene.text(0.58, 0.62, 11.9, 0.52, scene.title, size=27, bold=True,
               font="serif", text_color=INK, valign="mid")
    scene.line(0.58, 1.20, 12.15, 0, stroke=BURGUNDY, stroke_width=1.3)


def add_footer(scene: SlideScene) -> None:
    scene.line(0.58, 7.05, 12.15, 0, stroke=LINE, stroke_width=0.7)
    x = 0.58
    for criterion in scene.criteria:
        w = 0.30 + 0.075 * len(criterion)
        scene.rect(x, 7.13, w, 0.24, fill=PALE_NAVY, stroke=None, radius=0.05)
        scene.text(x + 0.08, 7.16, w - 0.16, 0.16, criterion.upper(), size=7.6,
                   text_color=NAVY, bold=True, font="sans", valign="mid")
        x += w + 0.10
    if scene.citation:
        scene.text(4.25, 7.14, 7.75, 0.22, scene.citation, size=7.2,
                   text_color=MUTED, font="sans", align="right", valign="mid")
    scene.text(11.62, 7.13, 1.10, 0.22, f"{scene.number} / {TOTAL}", size=8.6,
               text_color=MUTED, font="mono", align="right", valign="mid")


def label(scene: SlideScene, x, y, w, text_value, *, fill=PALE_NAVY, ink=NAVY):
    scene.rect(x, y, w, 0.28, fill=fill, stroke=None, radius=0.04)
    scene.text(x + 0.09, y + 0.045, w - 0.18, 0.19, text_value.upper(), size=8.2,
               text_color=ink, bold=True, font="sans", valign="mid")


def gloss(scene: SlideScene, x, y, w, text_value, *, accent=NAVY):
    """A small 'plain words / technical name' chip."""
    scene.rect(x, y, w, 0.30, fill=PALE_NAVY, stroke=None, radius=0.04)
    scene.text(x + 0.10, y + 0.055, w - 0.20, 0.20, text_value, size=8.6,
               font="mono", text_color=accent, valign="mid")


def card(scene: SlideScene, x, y, w, h, heading, body, *, accent=NAVY,
         body_size=12.5, fill=PAPER, tag=None):
    scene.rect(x, y, w, h, fill=fill, stroke=LINE, stroke_width=0.9, radius=0.08)
    scene.rect(x, y, 0.07, h, fill=accent, stroke=None, radius=0)
    head_y = y + 0.22 if not tag else y + 0.58
    if tag:
        label(scene, x + 0.24, y + 0.20, min(1.16, 0.38 + 0.082 * len(tag)), tag)
    scene.text(x + 0.24, head_y, w - 0.46, 0.36, heading, size=16,
               bold=True, font="serif", text_color=INK)
    scene.text(x + 0.24, head_y + 0.44, w - 0.46, h - (head_y - y) - 0.58,
               body, size=body_size, font="sans", text_color=MUTED)


def stat(scene: SlideScene, x, y, w, h, value, caption, *, accent=NAVY, fill=PAPER,
         value_size=30):
    scene.rect(x, y, w, h, fill=fill, stroke=LINE, stroke_width=0.9, radius=0.08)
    scene.rect(x, y, w, 0.06, fill=accent, stroke=None, radius=0)
    scene.text(x + 0.12, y + 0.28, w - 0.24, 0.60, value, size=value_size, bold=True,
               font="serif", text_color=INK, align="center")
    scene.text(x + 0.12, y + 0.98, w - 0.24, h - 1.08, caption, size=10.5,
               font="sans", text_color=MUTED, align="center")


def table(scene: SlideScene, x, y, w, *, headers, rows, row_labels,
          label_w=2.10, row_h=0.44, head_h=0.38, accent=NAVY, highlight=None):
    """A light rules-only numeric table. highlight: row indices to tint."""
    ncols = len(headers)
    col_w = (w - label_w) / ncols
    highlight = highlight or {}

    scene.rect(x, y, w, head_h, fill=PALE_NAVY, stroke=None, radius=0.04)
    scene.text(x + 0.10, y + 0.05, label_w - 0.16, 0.26, "SYSTEM", size=8.4,
               bold=True, font="sans", text_color=NAVY)
    for j, head in enumerate(headers):
        cx = x + label_w + j * col_w
        scene.text(cx, y + 0.05, col_w, 0.26, head, size=10, bold=True,
                   font="mono", text_color=NAVY, align="center")

    for i, (rlab, values) in enumerate(zip(row_labels, rows)):
        ry = y + head_h + i * row_h
        if i in highlight:
            scene.rect(x, ry, w, row_h, fill=highlight[i][1], stroke=None, radius=0.03)
        scene.line(x, ry + row_h, w, 0, stroke=LINE, stroke_width=0.6)
        scene.text(x + 0.10, ry + 0.09, label_w - 0.16, 0.28, rlab, size=11.5,
                   font="sans", text_color=INK)
        ink = highlight[i][0] if i in highlight else MUTED
        for j, val in enumerate(values):
            cx = x + label_w + j * col_w
            scene.text(cx, ry + 0.09, col_w, 0.28, val, size=11.5,
                       font="mono", text_color=ink, align="center",
                       bold=(i in highlight))


def figure(scene: SlideScene, path_name: str, *, y=1.52, max_h=4.60, x=None):
    """Fit a committed figure into the content area."""
    path = FIGDIR / path_name
    with Image.open(path) as im:
        ar = im.height / im.width
    width = min(11.55, max_h / ar)
    fx = x if x is not None else (SW - width) / 2
    scene.image(fx, y, width, path)
    return fx, y, width, width * ar


def build_scenes() -> list[SlideScene]:
    slides: list[SlideScene] = []

    # 1 — title
    s = SlideScene(1, "Thesis", "Task-Oriented Deep Joint Source–Channel Coding",
                   ("Results", "Presentation"),
                   "Held-back pictures only · the exam set has never been opened")
    add_header(s)
    s.text(0.72, 1.58, 11.86, 0.46,
           "What a camera should send when the receiver only needs the answer",
           size=20, text_color=NAVY, font="serif", italic=True, align="center")

    story = [
        (1.12, "EDGE CAMERA", "sees an image"),
        (5.11, "SHORT, NOISY LINK", "cannot carry every picture reliably"),
        (9.10, "RECEIVER CLASSIFIER", "needs one label back"),
    ]
    for x, heading, body in story:
        s.rect(x, 2.40, 3.10, 1.28, fill=PALE_NAVY, stroke=LINE, stroke_width=0.9, radius=0.08)
        s.text(x + 0.22, 2.70, 2.66, 0.24, heading, size=11.5, bold=True,
               font="sans", text_color=INK, align="center")
        s.text(x + 0.22, 3.14, 2.66, 0.30, body, size=12,
               font="sans", text_color=MUTED, align="center")
    s.arrow(4.32, 3.04, 0.64, 0, stroke=FAINT, stroke_width=1.2)
    s.arrow(8.31, 3.04, 0.64, 0, stroke=FAINT, stroke_width=1.2)

    s.rect(1.05, 4.26, 11.23, 1.30, fill=PAPER, stroke=BURGUNDY,
           stroke_width=1.2, radius=0.08)
    label(s, 1.34, 4.50, 0.92, "Thesis")
    s.text(1.33, 4.92, 10.66, 0.46,
           "Train the sender and the receiver together so the classification decision survives a short, noisy link.",
           size=17, font="serif", text_color=INK, align="center", valign="mid")

    s.rect(1.05, 5.82, 11.23, 0.94, fill=PALE_AMBER, stroke=LINE, stroke_width=0.9, radius=0.07)
    s.text(1.31, 5.98, 10.71, 0.64,
           "Second Review, 29 September – 3 October 2026.\nEverything here was measured on held-back pictures. The exam set has not been opened.",
           size=12, font="sans", text_color=INK, align="center")
    add_footer(s); slides.append(s)

    # 2 — progress since the first review
    s = SlideScene(2, "Progress", "What has changed since the first review",
                   ("Timeline", "Results"),
                   "First review 18–22 Aug 2026 · Second review 29 Sep–3 Oct 2026")
    add_header(s)
    s.text(0.82, 1.40, 11.68, 0.34,
           "In August we had a working digital system and no trained learned model. We have both now.",
           size=14, font="serif", text_color=MUTED, align="center")

    rows = [
        ("Aug", "The digital system", "Measured how badly it loses pictures at every setting, tested it on all 288,000 held-back combinations, and froze the settings it should use at each signal strength"),
        ("Aug", "The learned system", "Built the training loop, trained models at several seeds, and froze three of them at half rate"),
        ("Sep", "The fair comparison", "Built the control that sends learned features digitally, so we are not just comparing against a normal compressed picture"),
        ("Sep", "Weak-link training", "Trained a second version with the signal strength picked at random, and compared it against the first"),
        ("Sep", "The crossing question", "Decided it on held-back pictures before seeing the answer: the two lines cross between −5 and −4 dB"),
        ("Sep", "Power limits", "Trained one model under a hard limit on how tall the signal spikes may get"),
        ("Oct", "The measurements", "252 measurements across twelve versions of the system and twenty-one signal strengths, 1,000 held-back pictures each"),
        ("Nov", "The exam set", "Still locked. It opens in week 11 once the locking document is signed"),
    ]
    s.rect(0.82, 1.94, 11.68, 4.62, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.08)
    for i, (when, what, detail) in enumerate(rows):
        y = 2.12 + i * 0.555
        s.text(1.06, y + 0.04, 0.74, 0.24, when, size=10, bold=True,
               font="mono", text_color=MUTED)
        s.text(1.86, y, 2.30, 0.26, what, size=12.5, bold=True,
               font="sans", text_color=INK)
        s.text(4.24, y + 0.02, 8.00, 0.44, detail, size=11,
               font="sans", text_color=MUTED)
        if i < len(rows) - 1:
            s.line(1.06, y + 0.47, 11.14, 0, stroke=LINE, stroke_width=0.5)
    s.text(0.82, 6.72, 11.68, 0.24,
           "Everything above ran on held-back pictures. The exam set stays locked until gate G-12.",
           size=10.5, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 3 — objectives in completion terms (PR-8a)
    s = SlideScene(3, "Objectives", "Objectives, stated as completion criteria",
                   ("Objectives Met", "Methodology"),
                   "Objectives are not restated at this review")
    add_header(s)
    s.text(0.82, 1.38, 11.68, 0.32,
           "The rubric asks whether we finished what we set out to do, so we wrote it as things we did, not results we hoped for.",
           size=12.5, font="sans", text_color=MUTED, align="center")

    left = [
        ("Build it", "Train a learned sender and receiver together, so the picture goes through the same kind of noise a radio adds."),
        ("Same radio time", "At every signal strength and rate, both systems get the same number of chances to use the radio."),
        ("Charge the overhead", "Checksums, padding and error-correction bytes come out of the budget before any picture data."),
        ("Pick the setting fairly", "The operating point is chosen using the digital system alone, before anyone sees the learned curve."),
        ("Report per picture", "Every row records what happened to one picture, including the ones that never arrived."),
    ]
    s.rect(0.82, 1.90, 7.48, 4.60, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.08)
    label(s, 1.08, 2.14, 1.30, "What we set out to do")
    for i, (heading, body) in enumerate(left):
        y = 2.62 + i * 0.76
        s.circle(1.10, y + 0.03, 0.30, 0.30, fill=NAVY, stroke=None)
        s.text(1.10, y + 0.09, 0.30, 0.16, str(i + 1), size=9, bold=True,
               font="sans", text_color=WHITE, align="center")
        s.text(1.54, y, 1.86, 0.26, heading, size=12.5, bold=True,
               font="sans", text_color=INK)
        s.text(3.46, y - 0.02, 4.58, 0.62, body, size=11,
               font="sans", text_color=MUTED)
        if i < len(left) - 1:
            s.line(1.54, y + 0.62, 6.50, 0, stroke=LINE, stroke_width=0.5)

    s.rect(8.64, 1.90, 3.86, 4.60, fill=PALE_GREEN, stroke=LINE, stroke_width=0.9, radius=0.08)
    label(s, 8.92, 2.14, 1.78, "What we did not aim for")
    s.text(8.92, 2.68, 3.30, 1.44,
           "“Show that the learned system beats the digital one.”",
           size=16, bold=True, font="serif", text_color=INK, align="center")
    s.text(8.92, 4.28, 3.30, 1.92,
           "That is a result, not an objective. Our specification treats both a crossing and the learned system winning everywhere as complete, successful outcomes. The rubric scores this review on what we set out to do, not on which way the curve went.",
           size=11, font="sans", text_color=INK, align="center")
    s.text(0.82, 6.64, 11.68, 0.24,
           "We found a crossing, so the fallback we had planned for never happened and the objectives were not rewritten.",
           size=10.5, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 4 — plain-language on-ramp before any results
    s = SlideScene(4, "How to read these", "Before the numbers, one picture",
                   ("Results", "Analytical Skills"),
                   "Nothing technical on this slide. The next fourteen depend on it.")
    add_header(s)
    s.text(0.82, 1.38, 11.68, 0.34,
           "Every chart that follows runs from a very weak radio signal on the left to a clean one on the right.",
           size=13.5, font="serif", text_color=MUTED, align="center")

    words = [
        ("How strong the signal is", "SNR, in decibels", "the left-to-right axis"),
        ("How many pictures got through", "delivery coverage", "the flat 10% line"),
        ("How many we labelled correctly", "top-1 accuracy", "the up-and-down axis"),
        ("How many times we use the radio", "channel-use budget", "12,800 or 3,200 per picture"),
    ]
    s.rect(0.82, 1.88, 7.30, 4.26, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.08)
    label(s, 1.08, 2.12, 1.90, "Plain words / our term")
    for i, (plain, term, where) in enumerate(words):
        y = 2.70 + i * 0.80
        s.rect(1.08, y + 0.08, 0.08, 0.34, fill=NAVY, stroke=None, radius=0)
        s.text(1.32, y + 0.04, 2.80, 0.30, plain, size=13, bold=True,
               font="sans", text_color=INK)
        s.text(4.20, y + 0.06, 1.92, 0.26, term, size=11, font="mono", text_color=NAVY)
        s.text(6.18, y + 0.06, 1.72, 0.40, where, size=10.5,
               font="sans", text_color=MUTED)
        if i < len(words) - 1:
            s.line(1.08, y + 0.62, 6.80, 0, stroke=LINE, stroke_width=0.5)
    s.text(1.08, 5.78, 6.80, 0.28,
           "Every point on every chart is the same 1,000 pictures, at that one signal strength.",
           size=11, italic=True, font="sans", text_color=MUTED)

    s.rect(8.46, 1.88, 4.06, 4.26, fill=PALE_AMBER, stroke=LINE, stroke_width=0.9, radius=0.08)
    label(s, 8.74, 2.12, 1.70, "One example")
    s.text(8.74, 2.60, 3.50, 0.28, "At the weakest signal we measured:",
           size=11.5, bold=True, font="sans", text_color=INK)
    s.text(8.74, 2.98, 3.50, 1.36,
           "The learned system labelled 728 of the 1,000 pictures correctly.\n\nThe digital system got almost nothing through, so every picture fell back to one fixed answer — and that answer happens to be right 100 times out of 1,000. So 10%.",
           size=11, font="sans", text_color=INK)
    s.rect(8.74, 4.62, 3.50, 1.22, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.06)
    s.text(8.92, 4.82, 3.14, 0.90,
           "That flat 10% is not a bad prediction. It is 1,000 pictures that never arrived.",
           size=12, bold=True, font="sans", text_color=INK)
    s.text(0.82, 6.66, 11.68, 0.24,
           "These were 1,000 held-back pictures. The exam set stays locked until week 11.",
           size=10.5, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 5 — what was measured
    s = SlideScene(5, "Evidence base", "What we measured for this review",
                   ("Results", "Analytical Skills"),
                   "252 measurements = 12 versions of the system × 21 signal strengths")
    add_header(s)
    s.text(0.82, 1.40, 11.68, 0.32,
           "We built twelve versions of the system and tested each one at twenty-one signal strengths.",
           size=13.5, font="serif", text_color=MUTED, align="center")
    tiles = [
        ("252", "individual measurements"),
        ("12", "versions of the system"),
        ("21", "signal strengths"),
        ("1,000", "held-back pictures each"),
        ("0", "exam pictures opened"),
    ]
    for i, (value, caption) in enumerate(tiles):
        stat(s, 0.82 + i * 2.36, 1.86, 2.18, 1.70, value, caption,
             accent=NAVY, value_size=30)

    s.rect(0.82, 3.82, 5.78, 2.72, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.08)
    label(s, 1.08, 4.06, 1.40, "Radio time")
    s.text(1.08, 4.54, 5.26, 0.56,
           "Half rate — 12,800 uses of the radio per picture\nQuarter rate — 3,200 uses per picture",
           size=12, font="sans", text_color=INK)
    s.text(1.08, 5.30, 5.26, 1.06,
           "Within one rate, both systems get the same amount of radio time. It does not mean they send the same information, and the two sides carry different kinds of picture.",
           size=11, font="sans", text_color=MUTED)

    s.rect(6.84, 3.82, 5.68, 2.72, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.08)
    label(s, 7.10, 4.06, 1.70, "Who marks it")
    s.text(7.10, 4.54, 5.16, 1.86,
           "The learned systems are marked by the classifier built into them. The digital systems are marked by one we trained on compressed-looking images.\n\nSo we compare whole systems: the link, the way the picture is represented, and the marker.",
           size=11, font="sans", text_color=MUTED)
    s.rect(0.82, 6.66, 11.70, 0.34, fill=PALE_RED, stroke=LINE, stroke_width=0.9, radius=0.04)
    s.text(1.00, 6.72, 11.34, 0.22,
           "One training run. These are observed differences, not confidence intervals.",
           size=10.5, bold=True, font="sans", text_color=INK, align="center")
    add_footer(s); slides.append(s)

    # 6 — headline figure
    s = SlideScene(6, "Headline", "Where the digital system gives up, and where it takes over",
                   ("Results",),
                   "Half rate · all 21 measured signal strengths · held-back pictures")
    add_header(s)
    s.text(0.82, 1.34, 11.68, 0.32,
           "The learned system keeps answering the whole way across. The digital system stays silent, then overtakes it.",
           size=13.5, bold=True, font="serif", text_color=INK, align="center")
    figure(s, "01_headline_full_snr.png", y=1.76, max_h=4.48)
    s.rect(0.90, 6.40, 11.55, 0.52, fill=PALE_NAVY, stroke=LINE, stroke_width=0.9, radius=0.05)
    s.text(1.06, 6.50, 11.23, 0.34,
           "Weakest signal: learned right on 728 of 1,000, digital got nothing through. At −4 dB the digital system starts working and goes ahead. At +18 dB it is well ahead.",
           size=10.5, font="sans", text_color=INK, align="center")
    add_footer(s); slides.append(s)

    # 7 — the close-up
    s = SlideScene(7, "Low SNR", "Close-up: the moment the digital system starts working",
                   ("Results", "Analytical Skills"),
                   "−8 to +2 dB · measured points only, nothing in between")
    add_header(s)
    s.text(0.82, 1.34, 11.68, 0.32,
           "Between −5 and −4 dB the digital system begins getting pictures through, and its score jumps 73 points.",
           size=13.5, bold=True, font="serif", text_color=INK, align="center")
    figure(s, "02_low_snr_closeup.png", y=1.74, max_h=3.68)
    notes = [
        ("At −8 dB", "Learned right on 728 of 1,000. Digital got nothing through, so it scores 10%."),
        ("At −4 dB", "Digital now gets everything through and reaches 83.4%, ahead of learned's 79.4%."),
        ("After that", "It dips at −2 dB and again at +9 dB, so its line is not perfectly smooth."),
    ]
    for i, (tag, body) in enumerate(notes):
        x = 0.90 + i * 3.87
        s.rect(x, 5.60, 3.72, 1.02, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.06)
        s.rect(x, 5.60, 0.07, 1.02, fill=NAVY, stroke=None, radius=0)
        s.text(x + 0.22, 5.72, 3.30, 0.26, tag, size=12, bold=True,
               font="sans", text_color=INK)
        s.text(x + 0.22, 6.02, 3.30, 0.52, body, size=10.5,
               font="sans", text_color=MUTED)
    s.text(0.82, 6.74, 11.68, 0.22,
           "We never measured a strength between −5 and −4 dB, so this is a gap between two points, not a known threshold.",
           size=10, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 8 — bandwidth
    s = SlideScene(8, "Bandwidth", "What happens when we quarter the radio time",
                   ("Results", "Analytical Skills"),
                   "Quarter rate against half rate · compare within each panel")
    add_header(s)
    s.text(0.82, 1.32, 11.68, 0.32,
           "On a quarter of the radio time, the digital system loses far more at the weak end than the learned one does.",
           size=13.5, bold=True, font="serif", text_color=INK, align="center")
    figure(s, "03_bandwidth_efficiency.png", y=1.70, max_h=3.16)
    cards = [
        ("Weak signal, half → quarter", "Learned drops from 72.8% to 49.4%. A 23 point loss."),
        ("At 0 dB, quarter rate", "Learned 78.7%, digital 69.4%. Learned is still ahead."),
        ("Catches up, then slips", "Digital passes learned at +4 dB, loses the lead at +9 dB, then goes ahead again."),
    ]
    for i, (heading, body) in enumerate(cards):
        card(s, 0.90 + i * 3.87, 5.10, 3.72, 1.52, heading, body, body_size=11)
    s.text(0.82, 6.76, 11.68, 0.22,
           "Because it crosses and then uncrosses, we should not claim one clean crossover at this rate.",
           size=10, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 9 — randomised training
    s = SlideScene(9, "Robustness", "Training across signal strengths instead of one",
                   ("Results", "Analytical Skills"),
                   "Two separately trained versions of the learned system")
    add_header(s)
    s.text(0.82, 1.34, 11.68, 0.32,
           "Picking the training signal strength at random each time lifts the whole weak end of the line.",
           size=13.5, bold=True, font="serif", text_color=INK, align="center")
    figure(s, "04_training_robustness.png", y=1.72, max_h=3.56)
    table(s, 1.90, 5.44, 9.54,
          headers=["−8 dB", "−4 dB", "0 dB", "+9 dB", "+18 dB"],
          rows=[["77.0", "82.4", "83.0", "84.1", "83.9"],
                ["72.8", "79.4", "81.2", "83.8", "83.4"]],
          row_labels=["Random each time", "One fixed strength"],
          row_h=0.40,
          highlight={0: (INK, PALE_GREEN)})
    s.text(0.82, 6.70, 11.68, 0.22,
           "Two separately trained models from one run. This is not a measure of how much results vary between runs.",
           size=10, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 10 — task-aware digital control
    s = SlideScene(10, "Attribution", "The control that keeps the comparison honest",
                   ("Results", "Analytical Skills"),
                   "Same radio time, same error-correction code, same modulation")
    add_header(s)
    s.text(0.82, 1.34, 11.68, 0.32,
           "If we had only compared against a normal compressed image, the weak-signal story would be too easy to believe.",
           size=13.5, bold=True, font="serif", text_color=INK, align="center")
    figure(s, "05_task_aware_digital.png", y=1.72, max_h=3.68)
    s.rect(0.82, 5.66, 5.78, 1.20, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.07)
    s.rect(0.82, 5.66, 0.07, 1.20, fill=NAVY, stroke=None, radius=0)
    s.text(1.04, 5.78, 5.34, 0.26, "What this arm does", size=13, bold=True,
           font="sans", text_color=INK)
    s.text(1.04, 6.10, 5.34, 0.68,
           "Instead of a JPEG image it sends the learned features themselves, through the same error-correction code and radio settings, using the same 12,800 uses. It separates “knows the task” from “codes both at once”.",
           size=10.5, font="sans", text_color=MUTED)
    s.rect(6.84, 5.66, 5.68, 1.20, fill=PALE_AMBER, stroke=LINE, stroke_width=0.9, radius=0.07)
    s.text(7.06, 5.78, 5.24, 0.26, "What it settles", size=13, bold=True,
           font="sans", text_color=INK)
    s.text(7.06, 6.10, 5.24, 0.68,
           "It holds 82.0% at every strength from −4 dB up. Sending learned features digitally is strong once pictures get through, so the gap is about failed deliveries as much as representation. It does not isolate the code.",
           size=10.5, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 11 — peak power
    s = SlideScene(11, "Peak power", "Keeping the signal peaks from getting too tall",
                   ("Results", "Analytical Skills"),
                   "Peak-to-average power ratio, measured on the numbers we send")
    add_header(s)
    s.text(0.82, 1.34, 11.68, 0.32,
           "A real transmitter cannot send a signal with tall spikes, so we trained a second model under that limit.",
           size=13.5, bold=True, font="serif", text_color=INK, align="center")
    figure(s, "06_papr_tradeoff.png", y=1.72, max_h=2.98)
    gloss(s, 0.82, 4.84, 6.60,
          "peak-to-average ratio = how much taller the tallest spike is than the average")
    tiles = [
        ("3.000002 dB", "tallest spike, limited model"),
        ("20.059284 dB", "tallest spike, ordinary model"),
        ("+3.1 pp", "better at −8 dB"),
        ("−0.2 pp", "worse at +18 dB"),
    ]
    for i, (value, caption) in enumerate(tiles):
        stat(s, 0.82 + i * 2.94, 5.26, 2.76, 1.32, value, caption,
             accent=(NAVY if i < 2 else FAINT), value_size=21)
    s.text(0.82, 6.72, 11.68, 0.26,
           "This measures the numbers we put on the radio, not a real amplifier. The limited model is a separate training run, so the gap is not a clean penalty.",
           size=10, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 12 — secondary controls
    s = SlideScene(12, "Controls", "Four controls, including one that went against us",
                   ("Results", "Analytical Skills"),
                   "Settings, image format, sending the answer, rebuilding the picture")
    add_header(s)
    s.text(0.82, 1.30, 11.68, 0.30,
           "Each panel takes one design choice away from the digital system and shows what it was worth.",
           size=13, bold=True, font="serif", text_color=INK, align="center")
    figure(s, "07_secondary_controls.png", y=1.66, max_h=3.10)
    notes = [
        ("Fixed settings", "No delivery until +6 dB, then 89.0%. Choosing per strength buys real reach."),
        ("Fixed modulation", "Starts working later, and settles near 88.7%."),
        ("JPEG instead of JPEG 2000", "88.6% at +18 dB, and a real point at +9 dB where nothing arrived. Kept in."),
        ("Sending the answer itself", "82.0% once it arrives. A label the sender guessed, not the true answer."),
    ]
    for i, (heading, body) in enumerate(notes):
        x = 0.82 + (i % 2) * 6.00
        y = 4.94 + (i // 2) * 1.00
        s.rect(x, y, 5.70, 0.88, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.06)
        s.rect(x, y, 0.07, 0.88, fill=(FAINT if i == 2 else NAVY), stroke=None, radius=0)
        s.text(x + 0.22, y + 0.10, 5.26, 0.26, heading, size=12, bold=True,
               font="sans", text_color=INK)
        s.text(x + 0.22, y + 0.42, 5.26, 0.38, body, size=10.5,
               font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 13 — delivery coverage
    s = SlideScene(13, "Mechanism", "Why the digital line breaks where it does",
                   ("Analytical Skills", "Results"),
                   "When pictures stop arriving, the score drops to match")
    add_header(s)
    s.text(0.82, 1.32, 7.00, 0.32,
           "The score only moves when the pictures start or stop arriving.",
           size=13.5, bold=True, font="serif", text_color=INK)
    figure(s, "08_delivery_coverage.png", y=1.72, max_h=4.10, x=0.86)
    s.rect(6.60, 1.72, 5.92, 4.10, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.08)
    label(s, 6.86, 1.96, 1.40, "Reading it")
    body = [
        ("The flat 10% is a rule, not a delivery", "When nothing gets through, every picture takes the same fallback answer. On an evenly split set that is right 100 times in 1,000."),
        ("Failed deliveries are part of the result", "We report how many pictures arrived next to how many we got right, so you can tell a broken link from a wrong guess."),
        ("The learned system always answers", "Its rules promise a label at every strength, which is why its line has no floor to sit on."),
    ]
    for i, (heading, text_v) in enumerate(body):
        y = 2.38 + i * 1.14
        s.rect(6.86, y + 0.02, 0.07, 0.30, fill=NAVY, stroke=None, radius=0)
        s.text(7.06, y, 5.24, 0.28, heading, size=12, bold=True,
               font="sans", text_color=INK)
        s.text(7.06, y + 0.34, 5.24, 0.72, text_v, size=10.5,
               font="sans", text_color=MUTED)
    s.text(0.82, 6.32, 5.60, 0.62,
           "We did not add points between the ones we measured, and we did not smooth any curve to hide a step.",
           size=10.5, italic=True, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 14 — numbers table
    s = SlideScene(14, "Tables", "The numbers behind the curves",
                   ("Results", "Analytical Skills"),
                   "Pictures labelled correctly out of 1,000 · held-back set")
    add_header(s)
    s.text(0.82, 1.32, 11.68, 0.30,
           "How many of the 1,000 held-back pictures we labelled correctly, one picture at a time.",
           size=12.5, font="sans", text_color=MUTED, align="center")

    s.text(0.82, 1.76, 5.78, 0.26, "HALF RATE · 12,800 USES", size=9.6,
           bold=True, font="mono", text_color=INK)
    table(s, 0.82, 2.06, 5.78,
          headers=["−8", "−5", "−4", "0", "+9", "+18"],
          rows=[["72.8", "77.5", "79.4", "81.2", "83.8", "83.4"],
                ["10.0", "10.0", "83.4", "87.3", "86.5", "89.3"],
                ["77.0", "82.4", "82.4", "83.0", "84.1", "83.9"],
                ["10.0", "10.0", "82.0", "82.0", "82.0", "82.0"],
                ["75.9", "80.1", "80.5", "82.3", "83.0", "83.2"]],
          row_labels=["Learned, one strength", "Digital, picks settings",
                      "Learned, random", "Digital, learned feats",
                      "Learned, power-limited"],
          label_w=2.06, row_h=0.42,
          highlight={0: (INK, PALE_NAVY), 1: (INK, PALE_RED)})

    s.text(6.84, 1.76, 5.68, 0.26, "QUARTER RATE · 3,200 USES", size=9.6,
           bold=True, font="mono", text_color=INK)
    table(s, 6.84, 2.06, 5.68,
          headers=["−8", "−4", "−2", "0", "+4", "+18"],
          rows=[["49.4", "69.3", "76.2", "78.7", "82.0", "82.8"],
                ["10.0", "10.0", "31.0", "69.4", "83.4", "87.7"]],
          row_labels=["Learned, one strength", "Digital, picks settings"],
          label_w=2.06, row_h=0.42,
          highlight={0: (INK, PALE_NAVY), 1: (INK, PALE_RED)})

    s.rect(6.84, 3.30, 5.68, 1.14, fill=PALE_GREEN, stroke=LINE, stroke_width=0.9, radius=0.06)
    s.text(7.06, 3.42, 5.24, 0.28, "Where the two lines cross", size=13, bold=True,
           font="sans", text_color=INK)
    s.text(7.06, 3.76, 5.24, 0.58,
           "At half rate, which system is ahead flips between −5 dB and −4 dB. We decided the crossing question on exactly that, before seeing the answer.",
           size=10.5, font="sans", text_color=MUTED)

    s.rect(0.82, 4.62, 11.70, 1.90, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.08)
    label(s, 1.08, 4.86, 1.80, "What this table is not")
    s.text(1.08, 5.32, 11.18, 1.02,
           "Twenty-one signal strengths is not twenty-one independent experiments. Every cell comes from the same 1,000 held-back pictures and the same single training run, so a smooth-looking line tells you nothing statistically. A one or two point gap between rows is an observed difference and nothing more. The full per-picture records that would let us compute proper uncertainty ranges are not in this checkout, so we have not computed any.",
           size=11, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 15 — limits
    s = SlideScene(15, "Limits", "What this evidence cannot tell you",
                   ("Analytical Skills", "Methodology"),
                   "Stated plainly now, so the final review is not a surprise")
    add_header(s)
    items = [
        ("One training run, held-back pictures only", "Everything here is a description. No ranges, no significance, no repeated trials."),
        ("A different grader on each side", "The learned systems are marked by their own classifier, the digital systems by one trained on compressed images. So this measures whole systems."),
        ("Not the same amount of effort", "The digital system's settings and grader were frozen from earlier work; the learned versions are separate training runs."),
        ("No real amplifier involved", "We measured how tall the spikes are in the numbers we send. Nothing here shows what a real transmitter would do."),
        ("Frozen choices make sharp steps", "The fallback rule, 1,000 pictures and fixed settings produce jumps and a few bumpy points. We did not smooth them."),
    ]
    for i, (heading, body) in enumerate(items):
        y = 1.46 + i * 1.02
        s.rect(0.82, y, 11.70, 0.90, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.06)
        s.rect(0.82, y, 0.07, 0.90, fill=(FAINT if i >= 3 else NAVY), stroke=None, radius=0)
        s.text(1.10, y + 0.11, 3.70, 0.30, heading, size=13, bold=True,
               font="sans", text_color=INK)
        s.text(4.90, y + 0.09, 7.36, 0.72, body, size=11,
               font="sans", text_color=MUTED)
    s.text(0.82, 6.72, 11.70, 0.24,
           "Our specification treats both a crossing and the learned system winning everywhere as complete, successful outcomes. Neither result undoes the other.",
           size=10.5, italic=True, font="sans", text_color=MUTED, align="center")
    add_footer(s); slides.append(s)

    # 16 — SPEC §17 distillation (PR-8c)
    s = SlideScene(16, "Course correction", "Four times we changed the approach because of what we found",
                   ("Methodology",),
                   "Required by PR-8(c): a one-slide distillation of SPEC §17")
    add_header(s)
    s.text(0.82, 1.32, 11.68, 0.30,
           "The rubric asks what we decided based on the results obtained. These are the four that count.",
           size=12.5, font="sans", text_color=MUTED, align="center")
    corrections = [
        ("AM-34", "We had pinned the digital system to the weakest radio setting, which made the lines crossing impossible by arithmetic. We let it pick a stronger setting as the signal improves.",
         "The rule we adopted: every change strengthens the digital system or is decided in advance. We never weaken the learned system."),
        ("AM-52", "Our signal strengths were spaced 2 dB apart exactly where the sharpest drop happens, so we could have missed it entirely. We added three more strengths before it mattered.",
         "Put the detail where the effect is, not where it is convenient to sample."),
        ("AM-58", "The script that checks our packet sizes reported no problems while breaking four of its own rules. We fixed the script, not the results.",
         "All 215 configurations stayed feasible and every scientific outcome came out the same."),
        ("AM-60", "The exam set was set to unlock in week 9, about three weeks before the manifest that locks it existed. We moved that to week 11.",
         "They have stayed locked ever since. Nothing has been read."),
    ]
    for i, (tag, what, why) in enumerate(corrections):
        x = 0.82 + (i % 2) * 6.00
        y = 1.74 + (i // 2) * 2.54
        s.rect(x, y, 5.70, 2.34, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.07)
        s.rect(x, y, 5.70, 0.06, fill=NAVY, stroke=None, radius=0)
        s.text(x + 0.24, y + 0.22, 5.22, 0.26, tag, size=10.5, bold=True,
               font="mono", text_color=NAVY)
        s.text(x + 0.24, y + 0.58, 5.22, 0.96, what, size=11.5,
               font="sans", text_color=INK)
        s.line(x + 0.24, y + 1.60, 5.22, 0, stroke=LINE, stroke_width=0.6)
        s.text(x + 0.24, y + 1.72, 5.22, 0.50, why, size=10.5, italic=True,
               font="sans", text_color=MUTED)
    s.rect(0.82, 6.60, 11.70, 0.36, fill=PALE_GREEN, stroke=LINE, stroke_width=0.9, radius=0.05)
    s.text(1.00, 6.67, 11.34, 0.22,
           "Twenty-two further changes are written down in SPEC §17, including three checks that turned out to be wrong and had to be repaired.",
           size=10.5, bold=True, font="sans", text_color=INK, align="center")
    add_footer(s); slides.append(s)

    # 17 — originality / novelty
    s = SlideScene(17, "Originality", "What we claim as new, and what we do not",
                   ("Originality", "Novelty"),
                   "Two claims · both about how the experiment is built, not the idea itself")
    add_header(s)
    s.text(0.82, 1.30, 11.68, 0.30,
           "Both claims are about how the experiment is built, not about the idea of sending meaning instead of pixels.",
           size=12.5, font="sans", text_color=MUTED, align="center")

    claims = [
        ("The task-aware digital control",
         "A digital arm sends learned features through the same error-correction code and radio settings, using the same amount of radio time. It tells “knows the task” apart from “codes both at once” inside one experiment.",
         "Different from: studies that report a learned-versus-digital gap without such a control."),
        ("Charging the overhead to the digital system",
         "The picture file, framing, checksums and rate-matching bytes are all measured and counted, so the digital system is tuned on the payload it can genuinely use.",
         "Different from: pipelines that quote a nominal rate and leave the overhead uncounted."),
    ]
    for i, (heading, body, against) in enumerate(claims):
        y = 1.74 + i * 1.76
        s.rect(0.82, y, 7.48, 1.58, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.07)
        s.rect(0.82, y, 0.07, 1.58, fill=NAVY, stroke=None, radius=0)
        s.text(1.08, y + 0.14, 6.98, 0.30, heading, size=15, bold=True,
               font="sans", text_color=INK)
        s.text(1.08, y + 0.52, 6.98, 0.62, body, size=11,
               font="sans", text_color=MUTED)
        s.text(1.08, y + 1.18, 6.98, 0.30, against, size=10.5, italic=True,
               font="sans", text_color=MUTED)

    s.rect(8.64, 1.74, 3.88, 3.52, fill=PALE_RED, stroke=LINE, stroke_width=0.9, radius=0.07)
    label(s, 8.92, 1.98, 1.94, "Not claimed as new")
    nots = [
        "Getting steadily worse as the link worsens",
        "Avoiding the sudden drop-off",
        "Sending task-aware features digitally",
        "Training a neural encoder and decoder",
    ]
    for i, item in enumerate(nots):
        y = 2.54 + i * 0.52
        s.rect(8.92, y + 0.10, 0.09, 0.09, fill=NAVY, stroke=None, radius=0)
        s.text(9.14, y, 3.16, 0.32, item, size=11.5, font="sans", text_color=INK)
    s.text(8.92, 4.66, 3.36, 0.52,
           "People have already done all four of these. Claiming them would cost more than it earns.",
           size=10.5, italic=True, font="sans", text_color=MUTED, align="center")

    s.rect(0.82, 5.42, 11.70, 1.38, fill=PAPER, stroke=LINE, stroke_width=0.9, radius=0.07)
    label(s, 1.08, 5.64, 1.58, "Negative check")
    s.text(1.08, 6.04, 11.18, 0.64,
           "Both claims above are checked against the 30 references in docs/literature-review.md, and any claim that something has never been reported has to trace back to that list rather than to our own specification. The full written statement ships with the final review package.",
           size=11, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 18 — timeline
    s = SlideScene(18, "Timeline", "Where the remaining weeks went",
                   ("Timeline",),
                   "Second Review 29 Sep–3 Oct · Final Review 17–21 Nov · report due 20 Nov")
    add_header(s)
    phases = [
        ("W10", "29 Sep – 3 Oct", "Second Review", "Held-back measurements complete; package and figures frozen", 1.00),
        ("W11", "5 – 11 Oct", "Open the exam set", "One guarded opening once the locking manifest is signed", 0.00),
        ("W12", "12 – 18 Oct", "Decide everything", "Every hypothesis settled on the exam set; Tier 1 frozen", 0.00),
        ("W13", "19 – 25 Oct", "Demo", "Signal-strength slider, both systems side by side, frozen plot", 0.00),
        ("W14", "26 Oct – 1 Nov", "Poster + hardware", "Poster draft; real radio only if Tier 1 leaves room", 0.00),
        ("W15", "2 – 8 Nov", "Internal freeze", "Report finished in the required format", 0.00),
        ("W16", "9 – 15 Nov", "Spare week", "Report completion and checking only. No new experiments", 0.00),
        ("W17", "16 – 22 Nov", "Final Review", "Final review 17–21 Nov; report and material due 20 Nov", 0.00),
    ]
    s.line(1.72, 1.84, 0, 4.66, stroke=LINE, stroke_width=1.2)
    for i, (week, dates, name, detail, done) in enumerate(phases):
        y = 1.60 + i * 0.60
        s.text(0.82, y + 0.06, 0.62, 0.24, week, size=10, bold=True,
               font="mono", text_color=(INK if done else MUTED), align="right")
        s.circle(1.62, y + 0.08, 0.20, 0.20, fill=(INK if done else PAPER),
                 stroke=INK, stroke_width=1.4)
        s.text(1.98, y, 1.72, 0.26, dates, size=10.5, font="mono", text_color=MUTED)
        s.text(3.76, y - 0.02, 2.44, 0.28, name, size=12.5, bold=True,
               font="sans", text_color=INK)
        s.text(6.28, y, 5.16, 0.30, detail, size=10.5, font="sans", text_color=MUTED)
        s.text(11.62, y, 0.90, 0.26, "done" if done else "planned", size=9.5,
               bold=True, font="sans", text_color=(INK if done else FAINT), align="right")
    s.rect(0.82, 6.50, 11.70, 0.44, fill=PALE_AMBER, stroke=LINE, stroke_width=0.9, radius=0.05)
    s.text(1.00, 6.59, 11.34, 0.28,
           "The report is due inside Final Review week, which is why W15 is an internal freeze and W16 holds no experimental work.",
           size=10.5, font="sans", text_color=INK, align="center")
    add_footer(s); slides.append(s)

    # 19 — close
    s = SlideScene(19, "Close", "Where this leaves the project",
                   ("Results", "Methodology"),
                   "Questions welcome — the detail is in the repository, not on this slide")
    add_header(s)
    s.rect(0.82, 1.48, 11.70, 1.42, fill=PAPER, stroke=BURGUNDY,
           stroke_width=1.2, radius=0.08)
    s.text(1.14, 1.72, 11.06, 0.94,
           "The learned system keeps answering when the link is too weak for the digital one to get a single picture through. Once the digital system starts delivering, it is the more accurate of the two. They cross between −5 and −4 dB.",
           size=16, font="serif", text_color=INK, align="center", valign="mid")

    nexts = [
        ("Next", "One guarded run on the exam set once the locking manifest is signed. So far, zero exam pictures opened."),
        ("Then", "Every hypothesis settled on the exam set, and the first tier frozen."),
        ("Later", "Real-radio replay and the live demo are stretch goals. The project does not depend on them."),
    ]
    for i, (heading, body) in enumerate(nexts):
        card(s, 0.82 + i * 3.94, 3.18, 3.72, 1.48, heading, body, body_size=11)

    s.rect(0.82, 4.88, 11.70, 1.08, fill=PALE_NAVY, stroke=LINE, stroke_width=0.9, radius=0.07)
    label(s, 1.08, 5.08, 1.80, "Evidence trail")
    s.text(1.08, 5.48, 11.18, 0.36,
           "252-measurement record and per-picture manifest in results/learned/w10/ · figures and data in presentation-results/ · requirements and changes in spec/SPEC.md §17",
           size=10.5, font="mono", text_color=INK, align="center")
    s.rect(0.82, 6.14, 11.70, 0.64, fill=PALE_RED, stroke=LINE, stroke_width=0.9, radius=0.06)
    s.text(1.06, 6.26, 11.22, 0.42,
           "The exam set has never been opened. It stays that way through every gate, run and record in this project.",
           size=12, bold=True, font="sans", text_color=INK, align="center")
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
    cols, rows = 3, 7
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