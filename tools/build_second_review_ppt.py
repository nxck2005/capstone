#!/usr/bin/env python3
"""Build the editable Second Review deck, its PDF proof and slide previews.

Visual idiom copied from
deliverables/review-1/semantic-communication-first-review.pptx — the plain First
Review deck, which is the reference for this project:

  * no header banner, no section marker. A title and one hairline rule.
  * a footer that is plain text: "Rubric: ..." left, a citation right, page
    number far right. No pills or chips.
  * sharp-cornered white boxes with a thin grey border.
  * real tables: every row is a bordered cell with vertical separators.
  * term/definition rows for explanatory text, with no boxes at all.
  * one typeface, two greys and black. No colour, no rounding, no circles.

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
EDGE = "777777"
WHITE = "FFFFFF"

BODY_FONT = "Arial"
TITLE_FONT = "Arial"
MATH_FONT = "Arial"
MONO_FONT = "Arial"

FONT_FILES = {
    "serif": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "serif_bold": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "sans": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "sans_bold": "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "mono": "/usr/share/fonts/AdwaitaMono-Regular.ttf",
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
    radius: float = 0
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
    scene.text(0.62, 0.46, 12.05, 0.56, scene.title, size=27, bold=True,
               font="sans", text_color=INK, valign="mid")
    scene.line(0.62, 1.18, 12.05, 0, stroke=LINE, stroke_width=0.9)


def add_footer(scene: SlideScene) -> None:
    scene.line(0.62, 7.02, 12.05, 0, stroke=LINE, stroke_width=0.7)
    scene.text(0.62, 7.12, 3.35, 0.18,
               "Rubric: " + ", ".join(scene.criteria), size=8.5, text_color=MUTED)
    if scene.citation:
        scene.text(3.45, 7.12, 8.55, 0.18, scene.citation, size=8.5,
                   text_color=MUTED, font="sans", align="right")
    scene.text(11.72, 7.11, 0.92, 0.18, f"{scene.number} / {TOTAL}", size=8.5,
               text_color=MUTED, font="sans", align="right")


def box(scene: SlideScene, x, y, w, h, *, fill=PAPER, stroke=EDGE, weight=0.9):
    """Sharp-cornered white box with a thin grey border."""
    scene.rect(x, y, w, h, fill=fill, stroke=stroke, stroke_width=weight, radius=0)


def note(scene: SlideScene, x, y, w, text_value, *, size=10.5, italic=True,
         bold=False, color_=MUTED, align="left"):
    scene.text(x, y, w, 0.30, text_value, size=size, italic=italic, bold=bold,
               font="sans", text_color=color_, align=align)


def lead(scene: SlideScene, text_value, *, size=14, bold=True, y=1.34, color_=INK):
    scene.text(0.62, y, 12.05, 0.34, text_value, size=size, bold=bold,
               font="sans", text_color=color_, align="center")


def defs(scene: SlideScene, x, y, w, rows, *, label_w=2.30, pitch=0.60,
         size=11.5, rule=True):
    """Term/definition rows: bold label left, plain text right, hairline between."""
    for i, (term, body) in enumerate(rows):
        ry = y + i * pitch
        scene.text(x, ry, label_w - 0.10, 0.30, term, size=size + 0.5, bold=True,
                   font="sans", text_color=INK)
        scene.text(x + label_w, ry - 0.02, w - label_w, pitch - 0.06, body,
                   size=size, font="sans", text_color=MUTED)
        if rule and i < len(rows) - 1:
            scene.line(x, ry + pitch - 0.14, w, 0, stroke=LINE, stroke_width=0.5)
    return y + len(rows) * pitch


def grid(scene: SlideScene, x, y, cols, headers, rows, *, head_h=0.44,
         row_h=0.48, size=11.5, head_size=10.5, aligns=None, tint=None):
    """A real bordered table: one boxed cell band per row, vertical separators."""
    widths = [c[1] for c in cols]
    total_w = sum(widths)
    aligns = aligns or (["left"] + ["center"] * (len(cols) - 1))
    tint = tint or {}
    nrows = len(rows) + 1
    total_h = head_h + len(rows) * row_h

    scene.rect(x, y, total_w, head_h, fill="F3F3F3", stroke=LINE,
               stroke_width=0.9, radius=0)
    for i, ((_, cw), head) in enumerate(zip(cols, headers)):
        cx = x + sum(widths[:i])
        scene.text(cx + 0.14, y + 0.11, cw - 0.24, head_h - 0.18, head,
                   size=head_size, bold=True, font="sans", text_color=INK,
                   align=aligns[i], valign="mid")
    for r, values in enumerate(rows):
        ry = y + head_h + r * row_h
        scene.rect(x, ry, total_w, row_h, fill=tint.get(r, WHITE), stroke=LINE,
                   stroke_width=0.9, radius=0)
        for i, ((_, cw), val) in enumerate(zip(cols, values)):
            cx = x + sum(widths[:i])
            scene.text(cx + 0.14, ry + 0.11, cw - 0.24, row_h - 0.18, val,
                       size=size, font="sans", text_color=INK if r in tint else MUTED,
                       bold=(r in tint), align=aligns[i], valign="mid")
    for i in range(1, len(cols)):
        cx = x + sum(widths[:i])
        scene.line(cx, y, 0, total_h, stroke=LINE, stroke_width=0.7)
    return total_h


def figure(scene: SlideScene, path_name: str, *, y=1.70, max_h=4.40, x=None):
    path = FIGDIR / path_name
    with Image.open(path) as im:
        ar = im.height / im.width
    width = min(12.05, max_h / ar)
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
    s.text(0.62, 1.52, 12.05, 0.44,
           "What a camera should send when the receiver only needs the answer",
           size=20, text_color=INK, italic=True, align="center")

    story = [(1.10, 2.70, "Edge camera", "sees an image"),
             (5.10, 3.30, "Short, noisy link", "cannot carry every picture reliably"),
             (9.30, 2.40, "Receiver classifier", "needs one label back")]
    for x, w, head, body in story:
        box(s, x, 2.42, w, 1.34)
        s.text(x + 0.12, 2.78, w - 0.24, 0.30, head, size=14, bold=True,
               font="sans", text_color=INK, align="center")
        s.text(x + 0.12, 3.28, w - 0.24, 0.30, body, size=11.5,
               font="sans", text_color=MUTED, align="center")
    s.arrow(3.86, 3.09, 1.12, 0, stroke=MUTED, stroke_width=1.0)
    s.arrow(8.46, 3.09, 0.72, 0, stroke=MUTED, stroke_width=1.0)

    box(s, 0.62, 4.30, 12.05, 1.30)
    s.text(0.86, 4.58, 11.57, 0.44,
           "Train the sender and the receiver together so the classification decision survives a short, noisy link.",
           size=17, font="sans", text_color=INK, align="center", valign="mid")
    s.text(0.86, 5.06, 11.57, 0.34, "The thesis",
           size=10.5, bold=True, font="sans", text_color=MUTED, align="center")

    s.line(0.62, 5.94, 12.05, 0, stroke=LINE, stroke_width=0.7)
    s.text(0.62, 6.16, 12.05, 0.34,
           "Second Review, 29 September – 3 October 2026. Everything here was measured on held-back pictures; the exam set has not been opened.",
           size=11.5, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 2 — progress since the first review
    s = SlideScene(2, "Progress", "What has changed since the first review",
                   ("Timeline", "Results"),
                   "First review 18–22 Aug 2026 · Second review 29 Sep–3 Oct 2026")
    add_header(s)
    lead(s, "In August we had a working digital system and no trained learned model. We have both now.",
         size=14, bold=False, color_=MUTED)

    defs(s, 0.72, 1.96, 11.75, [
        ("Aug · Digital system",
         "Measured how badly it loses pictures at every setting, tested it on all 288,000 held-back combinations, and froze the settings it should use at each signal strength."),
        ("Aug · Learned system",
         "Built the training loop, trained models at several seeds, and froze three of them at half rate."),
        ("Sep · Fair comparison",
         "Built the control that sends learned features digitally, so we are not just comparing against a normal compressed picture."),
        ("Sep · Weak-link training",
         "Trained a second version with the signal strength picked at random, and compared it against the first."),
        ("Sep · Crossing question",
         "Decided it on held-back pictures before seeing the answer: the two lines cross between −5 and −4 dB."),
        ("Sep · Power limits",
         "Trained one model under a hard limit on how tall the signal spikes may get."),
        ("Oct · Measurements",
         "252 measurements across twelve versions of the system and twenty-one signal strengths, 1,000 held-back pictures each."),
        ("Nov · Exam set",
         "Still locked. It opens in week 11 once the locking document is signed."),
    ], label_w=2.55, pitch=0.60, size=11)

    s.line(0.72, 6.58, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.64, 11.75,
         "Everything above ran on held-back pictures. The exam set stays locked until gate G-12.")
    add_footer(s); slides.append(s)

    # 3 — objectives in completion terms (PR-8a)
    s = SlideScene(3, "Objectives", "Objectives, stated as completion criteria",
                   ("Objectives Met", "Methodology"),
                   "Objectives are not restated at this review")
    add_header(s)
    lead(s, "The rubric asks whether we finished what we set out to do, so we wrote it as things we did, not results we hoped for.",
         size=12.5, bold=False, color_=MUTED)

    s.text(0.72, 1.86, 6.90, 0.28, "What we set out to do", size=13, bold=True,
           font="sans", text_color=INK)
    left = [
        ("1.  Build it", "Train a learned sender and receiver together, so the picture goes through the same kind of noise a radio adds."),
        ("2.  Same radio time", "At every signal strength and rate, both systems get the same number of chances to use the radio."),
        ("3.  Charge the overhead", "Checksums, padding and error-correction bytes come out of the budget before any picture data."),
        ("4.  Pick the setting fairly", "The operating point is chosen using the digital system alone, before anyone sees the learned curve."),
        ("5.  Report per picture", "Every row records what happened to one picture, including the ones that never arrived."),
    ]
    for i, (term, body) in enumerate(left):
        ry = 2.30 + i * 0.78
        s.text(0.72, ry, 2.30, 0.30, term, size=12, bold=True, font="sans", text_color=INK)
        s.text(3.10, ry - 0.02, 4.52, 0.66, body, size=11, font="sans", text_color=MUTED)
        if i < len(left) - 1:
            s.line(0.72, ry + 0.62, 6.90, 0, stroke=LINE, stroke_width=0.5)

    box(s, 7.94, 1.86, 4.68, 4.20)
    s.text(8.18, 2.06, 4.20, 0.28, "What we did not aim for", size=13, bold=True,
           font="sans", text_color=INK)
    s.text(8.18, 2.50, 4.20, 1.20,
           "“Show that the learned system beats the digital one.”",
           size=16, bold=True, font="sans", text_color=INK, align="center")
    s.text(8.18, 3.86, 4.20, 1.96,
           "That is a result, not an objective. Our specification treats both a crossing and the learned system winning everywhere as complete, successful outcomes. The rubric scores this review on what we set out to do, not on which way the curve went.",
           size=11, font="sans", text_color=MUTED)

    s.line(0.72, 6.42, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.48, 11.75,
         "We found a crossing, so the fallback we had planned for never happened and the objectives were not rewritten.")
    add_footer(s); slides.append(s)

    # 4 — plain-language on-ramp before any results
    s = SlideScene(4, "How to read these", "Before the numbers, one picture",
                   ("Results", "Analytical Skills"),
                   "Nothing technical on this slide. The next fourteen depend on it.")
    add_header(s)
    lead(s, "Every chart that follows runs from a very weak radio signal on the left to a clean one on the right.",
         size=13.5, bold=False, color_=MUTED)

    grid(s, 0.72, 1.84,
         [("Plain words", 3.60), ("Our term", 2.40), ("Where you see it", 2.55)],
         ["Plain words", "Our term", "Where you see it"],
         [["How strong the signal is", "SNR, in decibels", "the left-to-right axis"],
          ["How many pictures got through", "delivery coverage", "the flat 10% line"],
          ["How many we labelled correctly", "top-1 accuracy", "the up-and-down axis"],
          ["How many times we use the radio", "channel-use budget", "12,800 or 3,200 per picture"]],
         head_h=0.40, row_h=0.54, size=11.5)

    s.text(0.72, 4.34, 8.55, 0.28, "Every point on every chart is the same 1,000 pictures, at that one signal strength.",
           size=11.5, bold=True, font="sans", text_color=INK)

    box(s, 9.72, 1.84, 2.90, 4.42)
    s.text(9.90, 2.02, 2.54, 0.28, "One example", size=12.5, bold=True,
           font="sans", text_color=INK)
    s.text(9.90, 2.36, 2.54, 0.26, "At the weakest signal we measured:",
           size=10.5, font="sans", text_color=MUTED)
    s.text(9.90, 2.72, 2.54, 1.86,
           "The learned system labelled 728 of the 1,000 pictures correctly.\n\nThe digital system got almost nothing through, so every picture fell back to one fixed answer — and that answer happens to be right 100 times out of 1,000. So 10%.",
           size=10.5, font="sans", text_color=MUTED)
    s.line(9.90, 4.72, 2.54, 0, stroke=LINE, stroke_width=0.7)
    s.text(9.90, 4.86, 2.54, 1.22,
           "That flat 10% is not a bad prediction. It is 1,000 pictures that never arrived.",
           size=11.5, bold=True, font="sans", text_color=INK)

    s.line(0.72, 6.42, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.48, 11.75,
         "These were 1,000 held-back pictures. The exam set stays locked until week 11.")
    add_footer(s); slides.append(s)

    # 5 — what was measured
    s = SlideScene(5, "Evidence base", "What we measured for this review",
                   ("Results", "Analytical Skills"),
                   "252 measurements = 12 versions of the system × 21 signal strengths")
    add_header(s)
    lead(s, "We built twelve versions of the system and tested each one at twenty-one signal strengths.",
         size=13.5, bold=False, color_=MUTED)

    grid(s, 0.72, 1.82,
         [("Item", 3.40), ("Figure", 1.70), ("What it is", 6.65)],
         ["Item", "Figure", "What it is"],
         [["Measurements", "252", "One row each in the published record"],
          ["Versions of the system", "12", "Five of them are controls rather than headline systems"],
          ["Signal strengths", "21", "From very weak to very clean"],
          ["Pictures per point", "1,000", "The same held-back pictures every time"],
          ["Exam pictures opened", "0", "The exam set has never been read"]],
         head_h=0.42, row_h=0.46, size=11.5)

    s.line(0.72, 4.58, 11.75, 0, stroke=LINE, stroke_width=0.7)
    s.text(0.72, 4.72, 11.75, 0.28, "Two things that govern how every later chart should be read",
           size=13, bold=True, font="sans", text_color=INK)
    defs(s, 0.72, 5.16, 11.75, [
        ("Radio time",
         "Half rate uses the radio 12,800 times per picture; quarter rate uses it 3,200 times. Within one rate both systems get the same amount of radio time. It does not mean they send the same information."),
        ("Who marks it",
         "The learned systems are marked by the classifier built into them. The digital systems are marked by one we trained on compressed-looking images. So we compare whole systems: the link, the representation, and the marker."),
    ], label_w=2.00, pitch=0.70, size=11)

    s.line(0.72, 6.58, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.64, 11.75,
         "One training run. These are observed differences, not confidence intervals.")
    add_footer(s); slides.append(s)

    # 6 — headline figure
    s = SlideScene(6, "Headline", "Where the digital system gives up, and where it takes over",
                   ("Results",),
                   "Half rate · all 21 measured signal strengths · held-back pictures")
    add_header(s)
    lead(s, "The learned system keeps answering the whole way across. The digital system stays silent, then overtakes it.")
    figure(s, "01_headline_full_snr.png", y=1.80, max_h=4.36)
    s.line(0.72, 6.34, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.42, 11.75,
         "At the weakest signal the learned system is right on 728 of 1,000 and the digital system got nothing through. At −4 dB the digital system starts delivering and goes ahead. At +18 dB it is well ahead.")
    add_footer(s); slides.append(s)

    # 7 — the close-up
    s = SlideScene(7, "Low SNR", "Close-up: the moment the digital system starts working",
                   ("Results", "Analytical Skills"),
                   "−8 to +2 dB · measured points only, nothing in between")
    add_header(s)
    lead(s, "Between −5 and −4 dB the digital system begins getting pictures through, and its score jumps 73 points.")
    figure(s, "02_low_snr_closeup.png", y=1.80, max_h=3.30)
    defs(s, 0.72, 5.30, 11.75, [
        ("At −8 dB", "The learned system is right on 728 of 1,000. The digital system got nothing through, so it scores 10%."),
        ("At −4 dB", "The digital system now gets everything through and reaches 83.4%, ahead of the learned system's 79.4%."),
        ("After that", "It dips at −2 dB and again at +9 dB, so its line is not perfectly smooth."),
    ], label_w=1.50, pitch=0.44, size=11.5)
    s.line(0.72, 6.58, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.64, 11.75,
         "We never measured a strength between −5 and −4 dB, so this is a gap between two points, not a known threshold.")
    add_footer(s); slides.append(s)

    # 8 — bandwidth
    s = SlideScene(8, "Bandwidth", "What happens when we quarter the radio time",
                   ("Results", "Analytical Skills"),
                   "Quarter rate against half rate · compare within each panel")
    add_header(s)
    lead(s, "On a quarter of the radio time, the digital system loses far more at the weak end than the learned one does.")
    figure(s, "03_bandwidth_efficiency.png", y=1.80, max_h=2.92)
    defs(s, 0.72, 5.00, 11.75, [
        ("Weak signal", "Going from half to quarter rate drops the learned system from 72.8% to 49.4%. That is a 23 point loss."),
        ("At 0 dB, quarter rate", "Learned 78.7% against digital 69.4%. The learned system is still ahead."),
        ("Catches up, then slips", "Digital passes learned at the +4 dB point, loses the lead at +9 dB, then goes ahead again."),
    ], label_w=2.10, pitch=0.50, size=11.5)
    s.line(0.72, 6.58, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.64, 11.75,
         "Because it crosses and then uncrosses, we should not claim one clean crossover at this rate.")
    add_footer(s); slides.append(s)

    # 9 — randomised training
    s = SlideScene(9, "Robustness", "Training across signal strengths instead of one",
                   ("Results", "Analytical Skills"),
                   "Two separately trained versions of the learned system")
    add_header(s)
    lead(s, "Picking the training signal strength at random each time lifts the whole weak end of the line.")
    figure(s, "04_training_robustness.png", y=1.80, max_h=3.16)
    grid(s, 2.20, 5.14,
         [("Training", 2.90), ("−8 dB", 1.10), ("−4 dB", 1.10), ("0 dB", 1.10),
          ("+9 dB", 1.10), ("+18 dB", 1.10)],
         ["Training", "−8 dB", "−4 dB", "0 dB", "+9 dB", "+18 dB"],
         [["Random each time", "77.0", "82.4", "83.0", "84.1", "83.9"],
          ["One fixed strength", "72.8", "79.4", "81.2", "83.8", "83.4"]],
         head_h=0.42, row_h=0.46, size=11.5, tint={0: "F3F3F3"})
    s.line(0.72, 6.56, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.62, 11.75,
         "Two separately trained models from one run. This is not a measure of how much results vary between runs.")
    add_footer(s); slides.append(s)

    # 10 — task-aware digital control
    s = SlideScene(10, "Attribution", "The control that keeps the comparison honest",
                   ("Results", "Analytical Skills"),
                   "Same radio time, same error-correction code, same modulation")
    add_header(s)
    lead(s, "If we had only compared against a normal compressed image, the weak-signal story would be too easy to believe.")
    figure(s, "05_task_aware_digital.png", y=1.80, max_h=3.34)
    defs(s, 0.72, 5.32, 11.75, [
        ("What this arm does",
         "Instead of a JPEG image it sends the learned features themselves, through the same error-correction code and radio settings, using the same 12,800 uses of the radio. It separates “knows the task” from “codes both at once”."),
        ("What it settles",
         "It holds 82.0% at every strength from −4 dB up. Sending learned features digitally is strong once pictures get through, so the weak-signal gap is about failed deliveries as much as representation. It does not isolate the error-correction code."),
    ], label_w=2.10, pitch=0.74, size=11.5)
    add_footer(s); slides.append(s)

    # 11 — peak power
    s = SlideScene(11, "Peak power", "Keeping the signal peaks from getting too tall",
                   ("Results", "Analytical Skills"),
                   "Peak-to-average power ratio, measured on the numbers we send")
    add_header(s)
    lead(s, "A real transmitter cannot send a signal with tall spikes, so we trained a second model under that limit.")
    figure(s, "06_papr_tradeoff.png", y=1.80, max_h=2.70)
    note(s, 0.72, 4.62, 11.75,
         "“Peak-to-average power ratio” = how much taller the tallest spike is than the average.",
         size=11, italic=True)
    grid(s, 1.60, 5.02,
         [("Model", 3.40), ("Tallest spike", 1.90), ("Accuracy at −8 dB", 1.90),
          ("Accuracy at +18 dB", 1.90)],
         ["Model", "Tallest spike", "At −8 dB", "At +18 dB"],
         [["Power-limited", "3.000002 dB", "75.9%", "83.2%"],
          ["Ordinary learned", "20.059284 dB", "72.8%", "83.4%"]],
         head_h=0.42, row_h=0.46, size=11.5, tint={0: "F3F3F3"})
    s.line(0.72, 6.62, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.68, 11.75,
         "This measures the numbers we put on the radio, not a real amplifier. The limited model is a separate training run, so the gap is not a clean penalty for the limit.")
    add_footer(s); slides.append(s)

    # 12 — secondary controls
    s = SlideScene(12, "Controls", "Four controls, including one that went against us",
                   ("Results", "Analytical Skills"),
                   "Settings, image format, sending the answer, rebuilding the picture")
    add_header(s)
    lead(s, "Each panel takes one design choice away from the digital system and shows what it was worth.",
         size=13)
    figure(s, "07_secondary_controls.png", y=1.76, max_h=2.72)
    defs(s, 0.72, 4.72, 11.75, [
        ("Fixed settings", "No delivery until +6 dB, then 89.0%. Choosing settings per signal strength buys real reach."),
        ("Fixed modulation", "Starts working later than adaptive selection, and settles near 88.7%."),
        ("JPEG, not JPEG 2000", "88.6% at +18 dB, and a real point at +9 dB where nothing arrived. We kept it in."),
        ("Sending the answer", "82.0% once it arrives. That is a label the sender guessed, not the true answer."),
    ], label_w=2.10, pitch=0.50, size=11.5)
    add_footer(s); slides.append(s)

    # 13 — delivery coverage
    s = SlideScene(13, "Mechanism", "Why the digital line breaks where it does",
                   ("Analytical Skills", "Results"),
                   "When pictures stop arriving, the score drops to match")
    add_header(s)
    s.text(0.72, 1.40, 6.20, 0.32,
           "The score only moves when the pictures start or stop arriving.",
           size=13, bold=True, font="sans", text_color=INK)
    figure(s, "08_delivery_coverage.png", y=1.84, max_h=4.10, x=0.86)
    defs(s, 7.36, 1.90, 5.26, [
        ("The 10% is a rule",
         "When nothing gets through, every picture takes the same fallback answer. On an evenly split set that is right 100 times in 1,000."),
        ("Outages are part of it",
         "We report how many pictures arrived next to how many we got right, so a broken link is distinguishable from a wrong guess."),
        ("Learned always answers",
         "Its rules promise a label at every strength, which is why its line has no floor to sit on."),
    ], label_w=1.70, pitch=1.26, size=11)
    s.line(0.72, 6.44, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.50, 11.75,
         "We did not add points between the ones we measured, and we did not smooth any curve to hide a step.")
    add_footer(s); slides.append(s)

    # 14 — numbers table
    s = SlideScene(14, "Tables", "The numbers behind the curves",
                   ("Results", "Analytical Skills"),
                   "Pictures labelled correctly out of 1,000 · held-back set")
    add_header(s)
    lead(s, "How many of the 1,000 held-back pictures we labelled correctly, one picture at a time.",
         size=12.5, bold=False, color_=MUTED)

    s.text(0.72, 1.76, 5.80, 0.24, "Half rate — 12,800 uses of the radio", size=11,
           bold=True, font="sans", text_color=INK)
    grid(s, 0.62, 2.06,
         [("System", 2.00), ("−8", 0.60), ("−5", 0.60), ("−4", 0.60), ("0", 0.60),
          ("+9", 0.60), ("+18", 0.64)],
         ["System", "−8", "−5", "−4", "0", "+9", "+18"],
         [["Learned, one strength", "72.8", "77.5", "79.4", "81.2", "83.8", "83.4"],
          ["Digital, picks settings", "10.0", "10.0", "83.4", "87.3", "86.5", "89.3"],
          ["Learned, random", "77.0", "82.4", "82.4", "83.0", "84.1", "83.9"],
          ["Digital, learned feats", "10.0", "10.0", "82.0", "82.0", "82.0", "82.0"],
          ["Learned, power-limited", "75.9", "80.1", "80.5", "82.3", "83.0", "83.2"]],
         head_h=0.40, row_h=0.44, size=11, head_size=10, tint={0: "F3F3F3", 1: "F3F3F3"})

    s.text(6.72, 1.76, 5.64, 0.24, "Quarter rate — 3,200 uses of the radio", size=11,
           bold=True, font="sans", text_color=INK)
    grid(s, 6.72, 2.06,
         [("System", 2.00), ("−8", 0.60), ("−4", 0.60), ("−2", 0.60), ("0", 0.60),
          ("+4", 0.60), ("+18", 0.64)],
         ["System", "−8", "−4", "−2", "0", "+4", "+18"],
         [["Learned, one strength", "49.4", "69.3", "76.2", "78.7", "82.0", "82.8"],
          ["Digital, picks settings", "10.0", "10.0", "31.0", "69.4", "83.4", "87.7"]],
         head_h=0.40, row_h=0.44, size=11, head_size=10, tint={0: "F3F3F3", 1: "F3F3F3"})

    s.text(6.72, 3.34, 5.64, 0.28, "Where the two lines cross", size=12, bold=True,
           font="sans", text_color=INK)
    s.text(6.72, 3.66, 5.64, 0.70,
           "At half rate, which system is ahead flips between −5 dB and −4 dB. We decided the crossing question on exactly that, before seeing the answer.",
           size=11, font="sans", text_color=MUTED)

    s.line(0.72, 4.62, 11.75, 0, stroke=LINE, stroke_width=0.7)
    s.text(0.72, 4.76, 11.75, 0.28, "What this table is not", size=13, bold=True,
           font="sans", text_color=INK)
    s.text(0.72, 5.12, 11.75, 1.20,
           "Twenty-one signal strengths is not twenty-one independent experiments. Every cell comes from the same 1,000 held-back pictures and the same single training run, so a smooth-looking line tells you nothing statistically. A one or two point gap between rows is an observed difference and nothing more. The full per-picture records that would let us compute proper uncertainty ranges are not in this checkout, so we have not computed any.",
           size=11, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 15 — limits
    s = SlideScene(15, "Limits", "What this evidence cannot tell you",
                   ("Analytical Skills", "Methodology"),
                   "Stated plainly now, so the final review is not a surprise")
    add_header(s)
    defs(s, 0.72, 1.60, 11.75, [
        ("One training run",
         "Held-back pictures only. Everything here is a description. No ranges, no significance, no repeated trials."),
        ("A different grader each side",
         "The learned systems are marked by their own classifier, the digital systems by one trained on compressed images. So this measures whole systems, not error-correction coding on its own."),
        ("Not the same effort",
         "The digital system's settings and grader were frozen from earlier work. The learned versions are separate training runs. Specific and fair, but not matched compute."),
        ("No real amplifier",
         "We measured how tall the spikes are in the numbers we send. Nothing here shows what a real transmitter would do. The project stays on a simulated channel."),
        ("Frozen choices are bumpy",
         "The fallback rule, 1,000 pictures and fixed operating points produce jumps and a few non-monotone points. We did not smooth them."),
    ], label_w=2.55, pitch=0.94, size=11.5)
    s.line(0.72, 6.42, 11.75, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.72, 6.48, 11.75,
         "Our specification treats both a crossing and the learned system winning everywhere as complete, successful outcomes. Neither result undoes the other.")
    add_footer(s); slides.append(s)

    # 16 — SPEC §17 distillation (PR-8c)
    s = SlideScene(16, "Course correction", "Four times we changed the approach because of what we found",
                   ("Methodology",),
                   "Required by PR-8(c): a one-slide distillation of SPEC §17")
    add_header(s)
    lead(s, "The rubric asks what we decided based on the results obtained. These are the four that count.",
         size=12.5, bold=False, color_=MUTED)
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
        x = 0.62 + (i % 2) * 6.14
        y = 1.78 + (i // 2) * 2.42
        box(s, x, y, 5.91, 2.24)
        s.text(x + 0.20, y + 0.16, 5.51, 0.26, tag, size=10.5, bold=True,
               font="sans", text_color=INK)
        s.text(x + 0.20, y + 0.48, 5.51, 1.06, what, size=11.5,
               font="sans", text_color=MUTED)
        s.line(x + 0.20, y + 1.60, 5.51, 0, stroke=LINE, stroke_width=0.6)
        s.text(x + 0.20, y + 1.70, 5.51, 0.46, why, size=10.5, italic=True,
               font="sans", text_color=MUTED)
    s.line(0.62, 6.52, 12.05, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.62, 6.58, 12.05,
         "Twenty-two further changes are written down in SPEC §17, including three checks that turned out to be wrong and had to be repaired.")
    add_footer(s); slides.append(s)

    # 17 — originality / novelty
    s = SlideScene(17, "Originality", "What we claim as new, and what we do not",
                   ("Originality", "Novelty"),
                   "Two claims · both about how the experiment is built, not the idea itself")
    add_header(s)
    lead(s, "Both claims are about how the experiment is built, not about the idea of sending meaning instead of pixels.",
         size=12.5, bold=False, color_=MUTED)
    claims = [
        ("The task-aware digital control",
         "A digital arm sends learned features through the same error-correction code and radio settings, using the same amount of radio time. It tells “knows the task” apart from “codes both at once” inside one experiment.",
         "Different from studies that report a learned-versus-digital gap without such a control."),
        ("Charging the overhead to the digital system",
         "The picture file, framing, checksums and rate-matching bytes are all measured and counted, so the digital system is tuned on the payload it can genuinely use.",
         "Different from pipelines that quote a nominal rate and leave the overhead uncounted."),
    ]
    for i, (heading, body, against) in enumerate(claims):
        y = 1.78 + i * 1.72
        box(s, 0.62, y, 7.62, 1.54)
        s.text(0.84, y + 0.14, 7.18, 0.30, heading, size=14, bold=True,
               font="sans", text_color=INK)
        s.text(0.84, y + 0.50, 7.18, 0.66, body, size=11, font="sans", text_color=MUTED)
        s.line(0.84, y + 1.18, 7.18, 0, stroke=LINE, stroke_width=0.6)
        s.text(0.84, y + 1.26, 7.18, 0.24, against, size=10.5, italic=True,
               font="sans", text_color=MUTED)

    s.text(8.62, 1.78, 4.00, 0.28, "Not claimed as new", size=13, bold=True,
           font="sans", text_color=INK)
    nots = ["Getting steadily worse as the link worsens",
            "Avoiding the sudden drop-off",
            "Sending task-aware features digitally",
            "Training a neural encoder and decoder"]
    for i, item in enumerate(nots):
        ry = 2.20 + i * 0.44
        s.text(8.62, ry, 0.22, 0.26, "—", size=11, font="sans", text_color=MUTED)
        s.text(8.90, ry, 3.72, 0.26, item, size=11.5, font="sans", text_color=INK)
    s.text(8.62, 4.06, 4.00, 0.62,
           "People have already done all four of these. Claiming them would cost more than it earns.",
           size=11, italic=True, font="sans", text_color=MUTED)

    s.line(0.62, 5.30, 12.05, 0, stroke=LINE, stroke_width=0.7)
    s.text(0.62, 5.44, 12.05, 0.28, "Negative check", size=13, bold=True,
           font="sans", text_color=INK)
    s.text(0.62, 5.80, 12.05, 0.70,
           "Both claims above are checked against the 30 references in docs/literature-review.md, and any claim that something has never been reported has to trace back to that list rather than to our own specification. The full written statement ships with the final review package.",
           size=11, font="sans", text_color=MUTED)
    add_footer(s); slides.append(s)

    # 18 — timeline
    s = SlideScene(18, "Timeline", "Where the remaining weeks went",
                   ("Timeline",),
                   "Second Review 29 Sep–3 Oct · Final Review 17–21 Nov · report due 20 Nov")
    add_header(s)
    phases = [
        ("W10", "29 Sep – 3 Oct", "Second Review", "Held-back measurements complete; package and figures frozen", True),
        ("W11", "5 – 11 Oct", "Open the exam set", "One guarded opening once the locking manifest is signed", False),
        ("W12", "12 – 18 Oct", "Decide everything", "Every hypothesis settled on the exam set; first tier frozen", False),
        ("W13", "19 – 25 Oct", "Demo", "Signal-strength slider, both systems side by side, frozen plot", False),
        ("W14", "26 Oct – 1 Nov", "Poster and hardware", "Poster draft; real radio only if the first tier leaves room", False),
        ("W15", "2 – 8 Nov", "Internal freeze", "Report finished in the required format", False),
        ("W16", "9 – 15 Nov", "Spare week", "Report completion and checking only. No new experiments", False),
        ("W17", "16 – 22 Nov", "Final Review", "Final review 17–21 Nov; report and material due 20 Nov", False),
    ]
    for i, (week, dates, name, detail, done) in enumerate(phases):
        y = 1.50 + i * 0.60
        s.text(0.62, y + 0.06, 0.62, 0.24, week, size=11, bold=True, font="sans",
               text_color=INK)
        s.text(1.42, y + 0.06, 1.72, 0.24, dates, size=11, font="sans", text_color=MUTED)
        s.text(3.30, y, 2.60, 0.28, name, size=12, bold=True, font="sans", text_color=INK)
        s.text(6.06, y + 0.02, 5.20, 0.28, detail, size=11, font="sans", text_color=MUTED)
        s.text(11.42, y + 0.04, 1.20, 0.24, "done" if done else "planned", size=10.5,
               bold=True, font="sans", text_color=INK if done else FAINT, align="right")
        s.line(0.62, y + 0.44, 12.05, 0, stroke=LINE, stroke_width=0.5)
    s.line(0.62, 6.46, 12.05, 0, stroke=LINE, stroke_width=0.7)
    note(s, 0.62, 6.52, 12.05,
         "The report is due inside Final Review week, which is why W15 is an internal freeze and W16 holds no experimental work.")
    add_footer(s); slides.append(s)

    # 19 — close
    s = SlideScene(19, "Close", "Where this leaves the project",
                   ("Results", "Methodology"),
                   "Questions welcome — the detail is in the repository, not on this slide")
    add_header(s)
    box(s, 0.62, 1.56, 12.05, 1.34)
    s.text(0.94, 1.80, 11.41, 0.90,
           "The learned system keeps answering when the link is too weak for the digital one to get a single picture through. Once the digital system starts delivering, it is the more accurate of the two. They cross between −5 and −4 dB.",
           size=16, font="sans", text_color=INK, align="center", valign="mid")

    defs(s, 0.62, 3.24, 12.05, [
        ("Next", "One guarded run on the exam set once the locking manifest is signed. So far, zero exam pictures opened."),
        ("Then", "Every hypothesis settled on the exam set, and the first tier frozen."),
        ("Later", "Real-radio replay and the live demo are stretch goals. The project does not depend on them."),
    ], label_w=1.20, pitch=0.56, size=12)

    s.line(0.62, 5.06, 12.05, 0, stroke=LINE, stroke_width=0.7)
    s.text(0.62, 5.20, 12.05, 0.28, "Evidence trail", size=13, bold=True,
           font="sans", text_color=INK)
    s.text(0.62, 5.54, 12.05, 0.34,
           "results/learned/w10/ holds the 252-measurement record and the per-picture manifest. Figures and data are in presentation-results/. Requirements and changes are in spec/SPEC.md §17.",
           size=11, font="sans", text_color=MUTED)
    s.line(0.62, 6.06, 12.05, 0, stroke=LINE, stroke_width=0.7)
    s.text(0.62, 6.20, 12.05, 0.34,
           "The exam set has never been opened. It stays that way through every gate, run and record in this project.",
           size=12, bold=True, font="sans", text_color=INK)
    add_footer(s); slides.append(s)

    return slides


def ppt_font(el: Element) -> str:
    return {"serif": TITLE_FONT, "sans": BODY_FONT, "math": MATH_FONT, "mono": MONO_FONT}.get(el.font, BODY_FONT)


def add_to_ppt(slide, el: Element) -> None:
    if el.kind == "text":
        box_ = slide.shapes.add_textbox(Inches(el.x), Inches(el.y), Inches(el.w), Inches(el.h))
        tf = box_.text_frame
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
        box_.rotation = el.rotation
    elif el.kind == "image":
        slide.shapes.add_picture(el.src, Inches(el.x), Inches(el.y),
                                 Inches(el.w), Inches(el.h))
    elif el.kind == "rect":
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(el.x), Inches(el.y),
                                       Inches(el.w), Inches(el.h))
        if el.fill:
            shape.fill.solid(); shape.fill.fore_color.rgb = rgb(el.fill)
        else:
            shape.fill.background()
        if el.stroke:
            shape.line.color.rgb = rgb(el.stroke); shape.line.width = Pt(el.stroke_width)
        else:
            shape.line.fill.background()
    elif el.kind == "line":
        line = slide.shapes.add_connector(1, Inches(el.x), Inches(el.y), Inches(el.x + el.w), Inches(el.y + el.h))
        line.line.color.rgb = rgb(el.stroke or INK)
        line.line.width = Pt(el.stroke_width)
    elif el.kind == "arrow":
        line = slide.shapes.add_connector(1, Inches(el.x), Inches(el.y), Inches(el.x + el.w), Inches(el.y + el.h))
        line.line.color.rgb = rgb(el.stroke or INK)
        line.line.width = Pt(el.stroke_width)
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
        draw.rectangle((x, y, x + w, y + h),
                       fill=color(el.fill) if el.fill else None,
                       outline=color(el.stroke) if el.stroke else None,
                       width=max(1, round(el.stroke_width * 1.5)))
    elif el.kind == "image":
        with Image.open(el.src) as im:
            im = im.convert("RGB")
            canvas.paste(im.resize((max(1, w), max(1, h)), Image.Resampling.LANCZOS), (x, y))
    elif el.kind == "line":
        x2, y2 = px(el.x + el.w, "x"), px(el.y + el.h, "y")
        if w == 0 and h == 0:
            w = 1
        if h == 0 and w > 0:
            draw.line((x, y, x2, y2), fill=color(el.stroke or INK), width=max(1, round(el.stroke_width * 2)))
        elif w == 0 and h > 0:
            draw.line((x, y, x2, y2), fill=color(el.stroke or INK), width=max(1, round(el.stroke_width * 2)))
        else:
            draw.line((x, y, x2, y2), fill=color(el.stroke or INK), width=max(1, round(el.stroke_width * 2)))
    elif el.kind == "arrow":
        x2, y2 = px(el.x + el.w, "x"), px(el.y + el.h, "y")
        draw.line((x, y, x2, y2), fill=color(el.stroke or INK), width=max(1, round(el.stroke_width * 2)))
        angle = math.atan2(y2 - y, x2 - x)
        length = 11
        draw.polygon([(x2, y2),
                      (x2 - length * math.cos(angle - 0.48), y2 - length * math.sin(angle - 0.48)),
                      (x2 - length * math.cos(angle + 0.48), y2 - length * math.sin(angle + 0.48))],
                     fill=color(el.stroke or INK))
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