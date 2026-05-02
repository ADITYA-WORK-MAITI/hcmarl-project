"""HC-MARL Major Project Report — generator.

Mimics the cosmetic of REPORT/EVALUATION 1 REPORT.pdf:
- Letter (8.5 x 11), 1-inch margins, Times-Roman 11/justified, line-leading 14.
- Centered chapter heads, bold; numbered '1. ABSTRACT', '2. INTRODUCTION', ...
- Tables with thin grid lines; small caption above ('Table 1: ...').
- Equations rendered via mathtext PNG, embedded inline-display, centered.
- Page numbers bottom-center, starting at '1' on Abstract.

Math-tone rules from REPORT/AI to Human College Report Writing.pdf are
encoded in the prose itself, not in the harness:
- no em dashes / semicolons; only commas, periods, parentheses
- no banned vocab (delve, robust, pivotal, intricate, seamless, ...
- no formulaic transitions (Furthermore, Moreover, Additionally, ...)
- mixed sentence lengths; specific file/function references where possible
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, NextPageTemplate,
    Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    KeepTogether, FrameBreak,
)
from reportlab.platypus.flowables import HRFlowable

ROOT = Path(__file__).parent
EQ = ROOT / "build_assets" / "eqs"
OUT = ROOT / "FINAL_REPORT.pdf"
RESULTS4 = ROOT.parent / "Results 4"

PAGE_W, PAGE_H = LETTER
MARGIN = 1.0 * inch
BODY_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=11, leading=14, alignment=TA_JUSTIFY, spaceBefore=0,
    spaceAfter=8, firstLineIndent=0,
)

BODY_INDENT = ParagraphStyle(
    "BodyIndent", parent=BODY, firstLineIndent=18,
)

CHAPTER_HEAD = ParagraphStyle(
    "ChapterHead", parent=styles["Heading1"], fontName="Times-Bold",
    fontSize=13, leading=16, alignment=TA_CENTER, spaceBefore=0,
    spaceAfter=14, textColor=colors.black,
)

SECTION_HEAD = ParagraphStyle(
    "SectionHead", parent=styles["Heading2"], fontName="Times-Bold",
    fontSize=11.5, leading=14, alignment=TA_LEFT, spaceBefore=10,
    spaceAfter=6, textColor=colors.black,
)

SUBSECTION_HEAD = ParagraphStyle(
    "SubHead", parent=styles["Heading3"], fontName="Times-Bold",
    fontSize=11, leading=14, alignment=TA_LEFT, spaceBefore=6,
    spaceAfter=4, textColor=colors.black, italic=False,
)

CAPTION = ParagraphStyle(
    "Caption", parent=BODY, fontName="Times-Roman", fontSize=10,
    leading=12, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10,
    italic=True,
)

TABLE_TITLE = ParagraphStyle(
    "TableTitle", parent=BODY, fontName="Times-Bold", fontSize=11,
    leading=14, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6,
)

REF_STYLE = ParagraphStyle(
    "Ref", parent=BODY, fontName="Times-Roman", fontSize=10.5,
    leading=13, alignment=TA_JUSTIFY, leftIndent=20, firstLineIndent=-20,
    spaceBefore=2, spaceAfter=2,
)

COVER_LARGE = ParagraphStyle(
    "CoverLarge", parent=BODY, fontName="Times-Bold", fontSize=22,
    leading=28, alignment=TA_CENTER, spaceBefore=8, spaceAfter=10,
)

COVER_MED = ParagraphStyle(
    "CoverMed", parent=BODY, fontName="Times-Bold", fontSize=14,
    leading=18, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6,
)

COVER_SMALL = ParagraphStyle(
    "CoverSmall", parent=BODY, fontName="Times-Roman", fontSize=12,
    leading=16, alignment=TA_CENTER, spaceBefore=2, spaceAfter=2,
)

COVER_HEADER = ParagraphStyle(
    "CoverHeader", parent=BODY, fontName="Times-Bold", fontSize=12.5,
    leading=15, alignment=TA_CENTER, spaceBefore=2, spaceAfter=2,
)


# ---------------------------------------------------------------------------
# Equation flowable (centered, scaled to fit body width if too wide)
# ---------------------------------------------------------------------------

def eq_flowable(name: str, target_height_pt: float = 16.0,
                max_width_in: float = 5.5,
                render_dpi: float = 600.0) -> Image:
    """Embed an equation PNG at its natural rendered size.

    Equations are rendered by build_eq_assets.py at a fixed font size and
    a fixed DPI, so the natural display size is (pixels * 72 / dpi). This
    gives every equation the same visual character height regardless of
    whether it contains fractions, brackets, or other tall structures.
    target_height_pt is retained for signature compatibility but ignored;
    max_width_in still clamps very wide equations to the body width.
    """
    png = EQ / f"{name}.png"
    if not png.exists():
        raise FileNotFoundError(png)
    from PIL import Image as PILImage
    with PILImage.open(png) as im:
        w, h = im.size
    width_pt = w * 72.0 / render_dpi
    height_pt = h * 72.0 / render_dpi
    max_w_pt = max_width_in * inch
    if width_pt > max_w_pt:
        scale = max_w_pt / width_pt
        width_pt *= scale
        height_pt *= scale
    img = Image(str(png), width=width_pt, height=height_pt)
    img.hAlign = "CENTER"
    return img


def display_eq(name: str, height: float = 24.0) -> list:
    """Return a list of flowables: spacer, centered eq, label, spacer."""
    return [
        Spacer(1, 4),
        eq_flowable(name, target_height_pt=height),
        Spacer(1, 4),
    ]


def fig_flowable(path: Path, caption: str, width_in: float = 5.5) -> list:
    """Embed a figure PNG centered with a caption underneath. The label MUST
    appear under every figure (hard constraint)."""
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    aspect = w / h
    width_pt = width_in * inch
    height_pt = width_pt / aspect
    img = Image(str(path), width=width_pt, height=height_pt)
    img.hAlign = "CENTER"
    return [
        Spacer(1, 6),
        img,
        Paragraph(caption, CAPTION),
    ]


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------

def _draw_page_number(canvas, doc):
    """Bottom-center page number, starting at 1 on Abstract."""
    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    page_num = canvas.getPageNumber()
    # cover and INDEX are pages 1 and 2 in PDF order, but show no page
    # number; arabic '1' begins at page 3 (Abstract).
    if page_num >= 3:
        text = str(page_num - 2)
        canvas.drawCentredString(PAGE_W / 2.0, 0.5 * inch, text)
    canvas.restoreState()


def _draw_cover_rule(canvas, doc):
    """Horizontal rule near the bottom of the cover page (matches EVAL 1)."""
    canvas.saveState()
    canvas.setLineWidth(0.7)
    canvas.line(MARGIN, 0.85 * inch, PAGE_W - MARGIN, 0.85 * inch)
    canvas.restoreState()


def make_doc() -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="HC-MARL Major Project Report",
        author="Aditya Maiti",
        subject="Human-Centric Multi-Agent Reinforcement Learning",
    )
    body_frame = Frame(MARGIN, MARGIN, BODY_W, PAGE_H - 2 * MARGIN,
                       id="body", showBoundary=0)
    cover_frame = Frame(MARGIN, MARGIN, BODY_W, PAGE_H - 2 * MARGIN,
                        id="cover", showBoundary=0)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame],
                     onPage=_draw_cover_rule),
        PageTemplate(id="Body", frames=[body_frame],
                     onPage=_draw_page_number),
    ])
    return doc


# ---------------------------------------------------------------------------
# Cover page (matches EVAL 1 layout)
# ---------------------------------------------------------------------------

def cover() -> list:
    out = [
        Spacer(1, 30),
        p("GURU GOBIND SINGH INDRAPRASTHA UNIVERSITY", COVER_HEADER),
        p("NEW DELHI", COVER_HEADER),
        Spacer(1, 28),
        p("UNIVERSITY SCHOOL OF AUTOMATION AND ROBOTICS", COVER_HEADER),
        p("EAST DELHI CAMPUS, SURAJMAL VIHAR, DELHI- 110032", COVER_HEADER),
        Spacer(1, 36),
        p("Major Project Report", COVER_LARGE),
        Spacer(1, 24),
        p("on", COVER_SMALL),
        Spacer(1, 14),
        p("Human Centric Multi Agent Reinforcement Learning", COVER_MED),
        p("(HC-MARL)", COVER_MED),
        Spacer(1, 36),
        p("Submitted in partial fulfillment of", COVER_SMALL),
        p("B.Tech (8<sup>th</sup> Semester)", COVER_SMALL),
        Spacer(1, 14),
        p("<b>In</b>", COVER_SMALL),
        p("AI-ML", COVER_SMALL),
        Spacer(1, 36),
        p("Name: Aditya Maiti", COVER_SMALL),
        p("Enrollment Number: 03819051622", COVER_SMALL),
        Spacer(1, 26),
        p("<b>Under the supervision of</b>", COVER_SMALL),
        Spacer(1, 18),
        p("<b>Dr. Amrit Pal Singh</b>", COVER_SMALL),
        p("<b>Assistant Professor, GGSIPU</b>", COVER_SMALL),
        NextPageTemplate("Body"),
        PageBreak(),
    ]
    return out


# ---------------------------------------------------------------------------
# INDEX (page 2)
# ---------------------------------------------------------------------------

def index_page() -> list:
    rows = [
        ["S. No.", "Title", "Page No."],
        ["1", "Abstract", "1"],
        ["2", "Chapter 1: Introduction", "2"],
        ["3", "Chapter 2: Literature Review", "5"],
        ["4", "Chapter 3: Methodology", "7"],
        ["5", "Chapter 4: Results and Discussions", "12"],
        ["6", "Conclusion", "21"],
        ["7", "References", "22"],
    ]
    tbl = Table(rows, colWidths=[0.9 * inch, 4.0 * inch, 1.1 * inch])
    tbl.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 0.7, colors.black),
        ("INNERGRID",  (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME",   (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",   (0, 0), (-1, -1), 11.5),
        ("ALIGN",      (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",      (2, 0), (2, -1),  "CENTER"),
        ("ALIGN",      (1, 0), (1, 0),   "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    return [
        Spacer(1, 16),
        p("<b>INDEX</b>", ParagraphStyle(
            "IndexHead", parent=BODY, fontName="Times-Bold",
            fontSize=14, leading=18, alignment=TA_CENTER, spaceAfter=22)),
        tbl,
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# 1. ABSTRACT (page 1, arabic-numbered)
# ---------------------------------------------------------------------------

def abstract() -> list:
    return [
        p("1. ABSTRACT", CHAPTER_HEAD),
        p(
            "Warehouse workers absorb cumulative musculoskeletal damage when "
            "fatigue is allowed to accumulate without an active monitor in the "
            "loop. The harm is invisible until it is too late, and no single "
            "framework today models that fatigue in real time, learns the "
            "personal limit at which a given worker breaks, enforces that "
            "limit with formal guarantees, and still divides labour across "
            "the team in a way that is provably fair. This report presents "
            "<b>HC-MARL</b> (Human-Centric Multi-Agent Reinforcement Learning), "
            "a four-component closed-loop framework that closes that gap.",
            BODY,
        ),
        p(
            "The first component is a physiologically calibrated 3CC-r "
            "ordinary differential equation system that tracks each worker's "
            "muscle pool across three compartments, resting (M<sub>R</sub>), "
            "active (M<sub>A</sub>), and fatigued (M<sub>F</sub>), with "
            "joint-specific rate constants taken from Frey-Law et al. 2012 "
            "and a switched reperfusion multiplier introduced by Looft et al. "
            "2018 that accelerates recovery during rest. The second component "
            "is Multi-Modal Inverse Constrained RL (MMICRL), which discovers "
            "latent worker types z and per-type fatigue thresholds "
            "&Theta;<sub>max</sub>(z) by maximising a weighted entropy "
            "objective &lambda;<sub>1</sub>H[&pi;(&tau;)] - "
            "&lambda;<sub>2</sub>H[&pi;(&tau;|z)] over unlabelled "
            "demonstrations. The third component is an Exponential Control "
            "Barrier Function (ECBF) that solves a dual-barrier quadratic "
            "program at every timestep, the first barrier holding the fatigue "
            "ceiling M<sub>F</sub> &le; &Theta;<sub>max</sub>, the second "
            "holding the resting floor M<sub>R</sub> &ge; 0. A "
            "Nagumo-invariance argument carries the safety guarantee across "
            "the work-to-rest mode switch. The fourth component is a Nash "
            "Social Welfare allocator with a divergent disagreement utility "
            "D<sub>i</sub> = &kappa; (M<sub>F</sub>)<sup>2</sup> / (1 - "
            "M<sub>F</sub>) that pushes the bargaining surplus toward "
            "negative infinity as a worker approaches burnout, which makes "
            "overloading mathematically impossible.",
            BODY,
        ),
        p(
            "The pipeline runs as a closed loop. At every round it observes "
            "the physiological state of every worker, allocates tasks via "
            "NSWF, translates the assignment into a per-muscle load demand, "
            "asks the MAPPO actor for a nominal neural drive, projects that "
            "command through the dual-barrier QP, and integrates the 3CC-r "
            "ODEs forward. End-to-end training was carried out on an L4 GPU "
            "across four experiments. Empirical results, reported in Chapter "
            "4, show HC-MARL beats the strongest cooperative MARL baselines "
            "by roughly 6.5x in shaped reward at the same safety level, and "
            "isolates the contribution of every component through a four-rung "
            "leave-one-out ablation.",
            BODY,
        ),
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# 2. CHAPTER 1: INTRODUCTION  (pages 2-5)
# ---------------------------------------------------------------------------

def chapter1() -> list:
    out = [
        p("2. CHAPTER 1: INTRODUCTION", CHAPTER_HEAD),
        # ----- BACKGROUND -----
        p("2.1 Background", SECTION_HEAD),
        p(
            "Warehouses run on bodies. A modern fulfilment centre can hold "
            "tens of thousands of stock-keeping units across hundreds of "
            "thousands of square feet, and the floor labour that moves those "
            "items from rack to packing line carries the overwhelming share "
            "of upper-limb and lower-back loading reported in occupational "
            "safety surveys. The Bureau of Labor Statistics tracks "
            "warehousing as one of the higher-incidence sectors for "
            "musculoskeletal injury year after year, and the proximate cause "
            "is almost always the same: cumulative fatigue that drifts past "
            "the worker's individual recovery margin without anyone noticing "
            "in time to intervene. The quantitative tools that connect a "
            "task profile to a worker's internal state have existed in the "
            "biomechanics literature for two decades, beginning with the "
            "three-compartment muscle model of Liu, Brown and Yue (2002) and "
            "its physiologically corrected form by Xia and Frey-Law (2008), "
            "but those tools have not made it into the multi-agent control "
            "stack that schedules the actual shifts.",
            BODY,
        ),
        p(
            "The scientific obstacle is that fatigue is invisible, "
            "individual, and nonlinear. It is invisible because it lives in "
            "metabolic state variables, not in observable kinematics. It is "
            "individual because the fatigue rate F and recovery rate R "
            "differ by joint and by person. Frey-Law and Avin (2010) "
            "documented this explicitly across 194 publications, "
            "establishing a fatigue-resistance hierarchy of "
            "ankle &gt; trunk &gt; grip &gt; elbow &gt; knee &gt; shoulder. "
            "It is nonlinear because the rate of recovery during rest is "
            "not the same as the rate of recovery during continued activity. "
            "Looft, Herkert and Frey-Law (2018) closed the last gap with the "
            "reperfusion multiplier r, producing the 3CC-r model in which "
            "the effective recovery rate switches from R to R&middot;r the "
            "moment the neural drive C(t) drops to zero.",
            BODY,
        ),
        p(
            "Modelling fatigue, however, is not the same as controlling it. "
            "A useful warehouse system has to do four things at once. It "
            "needs to learn a per-worker safety threshold from naturalistic "
            "behaviour, because no operations engineer hand-tunes "
            "&Theta;<sub>max</sub> per person. It needs to project the "
            "policy's nominal command through a real-time filter that "
            "guarantees the threshold is never crossed, including during "
            "the rest phase where the equation that governs M<sub>F</sub> "
            "can transiently increase whenever M<sub>A</sub> / M<sub>F</sub> "
            "exceeds R&middot;r / F. It needs to allocate tasks while "
            "remembering that pulling one worker off a heavy lift moves "
            "that load onto someone else, so the assignment problem is "
            "coupled across workers. And it needs to do all of this fairly, "
            "because a system that maximises throughput by quietly "
            "overworking the most resilient employee has not solved the "
            "human-centric problem at all. HC-MARL was built around that "
            "set of joint requirements.",
            BODY,
        ),
        # ----- OBJECTIVES -----
        p("2.2 Objectives of the Project", SECTION_HEAD),
        p(
            "The project sets out a closed list of objectives that map "
            "one-to-one onto the four mechanisms in the framework. Each "
            "objective is something the implementation has to satisfy, not "
            "something it would be nice to have.",
            BODY,
        ),
        objectives_table(),
        # ----- PROBLEM STATEMENT MATHEMATICALLY DEFINED -----
        p("2.3 Problem Statement: Mathematically Defined", SECTION_HEAD),
        p(
            "Consider N human workers and M tasks scheduled over a finite "
            "horizon at a fixed sampling interval &Delta;t. The "
            "physiological state of worker i at time t is a three-vector",
            BODY,
        ),
        *display_eq("eq_state_vector", height=24),
        p(
            "constrained to lie on the unit simplex by the conservation "
            "law",
            BODY,
        ),
        *display_eq("eq_conservation", height=24),
        p(
            "across every t. The dynamics evolve according to the three "
            "coupled ordinary differential equations",
            BODY,
        ),
        *display_eq("eq_dMA", height=24),
        *display_eq("eq_dMF", height=24),
        *display_eq("eq_dMR", height=24),
        p(
            "where C(t) &isin; [0, 1] is the voluntary neural drive (the "
            "control input chosen by the policy and filtered by ECBF), "
            "F is the joint-specific fatigue rate constant, and "
            "R<sub>eff</sub>(t) is the effective recovery rate governed by "
            "a reperfusion switch:",
            BODY,
        ),
        *display_eq("eq_Reff", height=27),
        p(
            "with r &asymp; 15 for the shoulder, ankle, knee, elbow and "
            "trunk groups and r &asymp; 30 for the hand grip, per Looft "
            "et al. (2018) Table 2. The full safe-set on the simplex is",
            BODY,
        ),
        *display_eq("eq_safety_set", height=24),
        p(
            "where &Theta;<sub>max</sub> is the per-worker fatigue ceiling "
            "discovered by MMICRL. The control problem decomposes into four "
            "sub-problems that have to be solved jointly, because a "
            "solution to any one of them in isolation is known to fail.",
            BODY,
        ),
        problem_subproblems(),
        # ----- PROBLEM STATEMENT -----
        p("2.4 Problem Statement", SECTION_HEAD),
        p(
            "In plain words, the problem this project takes on is to build a "
            "controller for a team of human warehouse workers that knows "
            "the personal fatigue limit of each individual without being "
            "told, never lets that limit be crossed even during the rest-"
            "recovery phase where the equations can briefly fight the "
            "controller, allocates tasks fairly so no single worker is "
            "silently exploited for aggregate throughput, and still "
            "completes as much productive work per shift as a system "
            "without any of those constraints in place. The controller "
            "has to scale to N workers and M tasks, has to handle the "
            "work-to-rest mode switch correctly, has to produce an audit "
            "trail a safety regulator can read end to end, and has to "
            "rest on a codebase small enough to be reviewed line by "
            "line.",
            BODY,
        ),
        p(
            "Each of the four sub-problems P1 through P4 has been "
            "addressed in isolation in the published literature. Section "
            "3.1 walks through the physiological-modelling line of work "
            "that produces P1's underlying state, Section 3.2 tracks the "
            "constraint-learning line that targets P1's inverse problem, "
            "Section 3.3 follows the control-barrier line that handles "
            "P2, and Section 3.4 covers the fair-allocation line that "
            "addresses P3 and P4 together. None of them, taken alone, "
            "solves the joint problem. The integration is what HC-MARL "
            "delivers, and the rest of this report develops the design, "
            "verifies it against the codebase line by line, and reports "
            "the empirical evidence collected across four end-to-end "
            "experiments. Every numeric claim made in subsequent "
            "chapters is traceable to a file in <i>hcmarl/</i> or to a "
            "result artefact under <i>Results 4/</i>, and the "
            "experimental protocol that produced those artefacts is "
            "summarised in Chapter 4 with explicit references to the "
            "scripts that ran each experiment and the seeds that were "
            "used.",
            BODY,
        ),
        PageBreak(),
    ]
    return out


def objectives_table() -> Table:
    cell = ParagraphStyle("TblCell", parent=BODY, fontSize=10, leading=12,
                          alignment=TA_LEFT, spaceBefore=0, spaceAfter=0)
    cell_b = ParagraphStyle("TblCellB", parent=cell, fontName="Times-Bold")
    cell_c = ParagraphStyle("TblCellC", parent=cell, alignment=TA_CENTER)
    rows = [
        [Paragraph("<b>No.</b>", cell_c),
         Paragraph("<b>Objective</b>", cell_b),
         Paragraph("<b>Mechanism in HC-MARL</b>", cell_b)],
        [Paragraph("O1", cell_c),
         Paragraph("Discover the latent worker type z and the per-type "
                   "fatigue ceiling &Theta;<sub>max</sub>(z) from "
                   "unlabelled mixed demonstrations.", cell),
         Paragraph("MMICRL with a CFDE flow density estimator.", cell)],
        [Paragraph("O2", cell_c),
         Paragraph("Enforce M<sub>F</sub> &le; &Theta;<sub>max</sub> "
                   "and M<sub>R</sub> &ge; 0 at every step including "
                   "the work-rest mode switch.", cell),
         Paragraph("Dual-barrier ECBF-QP solved per timestep.", cell)],
        [Paragraph("O3", cell_c),
         Paragraph("Allocate tasks fairly so a worker close to burnout "
                   "cannot be assigned more load.", cell),
         Paragraph("Nash Social Welfare with divergent disagreement "
                   "utility.", cell)],
        [Paragraph("O4", cell_c),
         Paragraph("Maximise aggregate task completion subject to O1, "
                   "O2, O3 across an 8-hour shift.", cell),
         Paragraph("MAPPO actor-critic with shared parameters and PPO "
                   "clipping.", cell)],
        [Paragraph("O5", cell_c),
         Paragraph("Produce reproducible empirical evidence that every "
                   "component carries its weight.", cell),
         Paragraph("Four experiments (EXP0 to EXP3), 10 / 5 / 5 / 5 "
                   "seeds, IQM with 95% bootstrap CI.", cell)],
    ]
    tbl = Table(rows, colWidths=[0.55 * inch, 3.05 * inch,
                                 2.4 * inch])
    tbl.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 0.7, colors.black),
        ("INNERGRID",  (0, 0), (-1, -1), 0.4, colors.black),
        ("FONTNAME",   (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("LEADING",    (0, 0), (-1, -1), 12),
        ("ALIGN",      (0, 0), (0, -1),  "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0),  colors.HexColor("#EEEEEE")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    return tbl


def problem_subproblems() -> Table:
    cell = ParagraphStyle("TblCellP", parent=BODY, fontSize=10, leading=12,
                          alignment=TA_LEFT, spaceBefore=0, spaceAfter=0)
    cell_b = ParagraphStyle("TblCellPB", parent=cell, alignment=TA_CENTER,
                            fontName="Times-Bold")
    rows = [
        [Paragraph("P1", cell_b),
         Paragraph(
             "<b>Personalised constraint discovery.</b> Recover latent "
             "types z and per-type fatigue thresholds "
             "&Theta;<sub>max</sub>(z) by maximising "
             "&lambda;<sub>1</sub>&middot;H[&pi;(&tau;)] - "
             "&lambda;<sub>2</sub>&middot;H[&pi;(&tau;|z)] over "
             "unlabelled mixed demonstrations.", cell)],
        [Paragraph("P2", cell_b),
         Paragraph(
             "<b>Formal safety.</b> Hold M<sub>F</sub>(t) &le; "
             "&Theta;<sub>max</sub> and M<sub>R</sub>(t) &ge; 0 for "
             "every worker, every muscle group, every timestep, "
             "including across the work-rest mode switch where "
             "dM<sub>F</sub>/dt &gt; 0 whenever "
             "M<sub>A</sub> / M<sub>F</sub> &gt; R&middot;r / F.", cell)],
        [Paragraph("P3", cell_b),
         Paragraph(
             "<b>Fair allocation.</b> Assign tasks by maximising "
             "&Sigma;<sub>i</sub> ln(U(i, j*(i)) - D<sub>i</sub>) where "
             "D<sub>i</sub> = &kappa;&middot;(M<sub>F</sub>)<sup>2</sup>"
             " / (1 - M<sub>F</sub>) drives the surplus to negative "
             "infinity as the worker approaches burnout.", cell)],
        [Paragraph("P4", cell_b),
         Paragraph(
             "<b>Productivity.</b> Maximise aggregate task completion "
             "subject to P1, P2, P3.", cell)],
    ]
    tbl = Table(rows, colWidths=[0.55 * inch, 5.45 * inch])
    tbl.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 0.7, colors.black),
        ("INNERGRID",  (0, 0), (-1, -1), 0.4, colors.black),
        ("FONTNAME",   (0, 0), (-1, -1), "Times-Roman"),
        ("FONTNAME",   (0, 0), (0, -1),  "Times-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("LEADING",    (0, 0), (-1, -1), 12),
        ("ALIGN",      (0, 0), (0, -1),  "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# 3. CHAPTER 2: LITERATURE REVIEW (pages 6-9)
# ---------------------------------------------------------------------------

def chapter2() -> list:
    return [
        p("3. CHAPTER 2: LITERATURE REVIEW", CHAPTER_HEAD),
        p(
            "Four lines of work are relevant to HC-MARL. None of them, "
            "taken alone, addresses the joint problem the framework targets, "
            "and the gap left by their non-intersection is precisely the "
            "research opening this report fills.",
            BODY,
        ),
        # A
        p("3.1 Physiological Fatigue Modelling", SECTION_HEAD),
        p(
            "Liu, Brown and Yue (2002) split the motor unit pool into "
            "three dynamic compartments, resting, active, and fatigued, "
            "and wrote three coupled mass-action ordinary differential "
            "equations governed by a conservation law. This was the first "
            "dynamic model of muscle fatigue. Earlier work in ergonomics "
            "leaned on static empirical endurance curves consolidated by "
            "Frey-Law and Avin (2010) in their 194-publication "
            "meta-analysis, and could not predict the time course of "
            "recovery between bouts. The original Liu formulation, however, "
            "routed fatigued motor units back to the active pool directly, "
            "which is physiologically wrong. Xia and Frey-Law (2008) "
            "corrected that route, established the recovery sequence as "
            "Fatigued → Resting → Active, and introduced the "
            "voluntary neural drive C(t) as an explicit control input. They "
            "validated the corrected model against historical isometric "
            "endurance data tabulated in the Frey-Law and Avin (2010) "
            "meta-analysis.",
            BODY,
        ),
        p(
            "Frey-Law, Looft and Heitsman (2012) ran roughly nine million "
            "Monte Carlo simulations to calibrate the F and R rate "
            "constants for five major joints, moving the model from a "
            "theoretical sketch to a numerically grounded predictor. "
            "Frey-Law and Avin (2010) followed with a meta-analysis of 194 "
            "publications that established the joint-specific endurance "
            "hierarchy, ankle on top, shoulder at the bottom, with grip, "
            "trunk, elbow, and knee in between. Looft, Herkert and "
            "Frey-Law (2018) added the reperfusion multiplier r, where "
            "rest-phase recovery switches from R to R&middot;r, with r "
            "&asymp; 15 across most joints and r &asymp; 30 for hand grip. "
            "The full 3CC-r model is the version used in this work. Looft "
            "and Frey-Law (2020) independently validated r &asymp; 15 for "
            "shoulder flexion in a separate cohort, which closes the "
            "external-validity loop on the parameter the framework relies "
            "on most heavily.",
            BODY,
        ),
        p(
            "<i>Gap.</i> Across that twenty-year programme, the 3CC-r "
            "model has not been embedded inside a multi-agent control "
            "architecture. No prior system uses its dynamics as the "
            "real-time state for a reinforcement-learning policy.",
            BODY,
        ),
        # B
        p("3.2 Constraint Learning", SECTION_HEAD),
        p(
            "Ziebart et al. (2008) introduced Maximum Entropy Inverse "
            "Reinforcement Learning, recovering reward functions from "
            "demonstrations under an entropy maximisation criterion. Malik "
            "et al. (2021) shifted the question from rewards to "
            "constraints with Inverse Constrained RL (ICRL) inside the "
            "Constrained MDP framework of Altman (1999), asking what "
            "constraints does this agent respect rather than what is "
            "rewarded. Achiam et al. (2017) developed Constrained Policy "
            "Optimisation (CPO) for training policies that satisfy "
            "constraints during training, not just at deployment. Both "
            "ICRL and CPO assume homogeneous agents. Li, Song and Ermon "
            "(2017) broke that assumption with InfoGAIL, using a latent "
            "variable z to separate behavioural modes from mixed "
            "demonstrations. Qiao et al. (NeurIPS 2023) put the two ideas "
            "together as MMICRL, simultaneously discovering agent types "
            "and learning per-type safety constraints through a weighted "
            "entropy objective with flow-based density estimation.",
            BODY,
        ),
        p(
            "<i>Gap.</i> MMICRL has not been applied to physiological "
            "safety limits. No method in the published record learns "
            "individual fatigue thresholds from unlabelled worker "
            "demonstrations.",
            BODY,
        ),
        # C
        p("3.3 Safety Guarantees", SECTION_HEAD),
        p(
            "Nagumo (1942) proved the subtangentiality condition for set "
            "invariance, which remains the analytical anchor for every "
            "modern barrier-function argument. Blanchini (1999) wrote the "
            "definitive Automatica survey connecting set invariance to "
            "Lyapunov stability. Prajna and Jadbabaie (2004) introduced "
            "barrier certificates for safety verification of hybrid "
            "systems. Ames et al. (2017) translated those certificates "
            "into real-time control through the CBF-QP, where the safe "
            "input is the projection of the nominal input onto the half-"
            "space defined by an active barrier constraint. Ames et al. "
            "(2019) consolidated the framework into a single reference. "
            "Standard CBFs require the safety constraint to have "
            "relative degree one with respect to the control input. "
            "Nguyen and Sreenath (2016) solved the relative-degree-two "
            "case with Exponential CBFs by constructing composite "
            "barriers, and Xiao and Belta (2019) generalised to "
            "high-order CBFs for arbitrary relative degree.",
            BODY,
        ),
        p(
            "<i>Gap.</i> CBFs have been deployed for position, velocity, "
            "and collision constraints in robotics and autonomous driving. "
            "They have not been applied to physiological fatigue dynamics, "
            "which is a relative-degree-two system with a switched "
            "recovery mode. The dual-barrier construction in HC-MARL is "
            "the first such application in the published record.",
            BODY,
        ),
        # D
        p("3.4 Fair Task Allocation", SECTION_HEAD),
        p(
            "Nash (1950) wrote down the four axioms a fair division has "
            "to satisfy and showed the unique solution maximises the "
            "product of surpluses over a fixed disagreement point. Nash "
            "(1953) extended the result to two-person cooperative games "
            "with an endogenous, state-dependent disagreement point. "
            "Kaneko and Nakamura (1979) generalised the axiomatisation to "
            "N players, establishing the Nash Social Welfare Function as "
            "the unique allocation rule satisfying the multi-player "
            "extension. Binmore, Shaked and Sutton (1989) gave the "
            "outside-option principle a sharp empirical and theoretical "
            "treatment. Navon et al. (ICML 2022) applied the log-transform "
            "of the Nash product to multi-task learning, where the "
            "log-objective produces a numerically stable gradient even "
            "when one task's loss is much larger than another's.",
            BODY,
        ),
        p(
            "<i>Gap.</i> No prior work uses a physiological-state-dependent "
            "divergent disagreement utility, where D &rarr; +&infin; as "
            "fatigue approaches the burnout boundary. That construction "
            "is what makes overloading a fatigued worker mathematically "
            "impossible inside HC-MARL, rather than merely costly.",
            BODY,
        ),
        # closing
        p("3.5 The Synthesis Gap", SECTION_HEAD),
        p(
            "Physiological fatigue models, constraint learning frameworks, "
            "control barrier safety theory, and Nash bargaining fairness "
            "exist as four separate research literatures. Each has "
            "produced strong results in isolation, and each has been the "
            "subject of book-length treatments. The integration of all "
            "four into a single closed-loop human-centric multi-agent "
            "reinforcement-learning framework has not been attempted in "
            "the published record. That synthesis is the contribution "
            "of HC-MARL, and it is what the rest of this report develops, "
            "implements, and tests.",
            BODY,
        ),
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# 4. CHAPTER 3: METHODOLOGY (pages 10-14)
# ---------------------------------------------------------------------------

def chapter3() -> list:
    DIAG = ROOT / "build_assets" / "diagrams"
    return [
        p("4. CHAPTER 3: METHODOLOGY", CHAPTER_HEAD),
        # ----- PROPOSED DESIGN FLOW -----
        p("4.1 Proposed Design Flow", SECTION_HEAD),
        *fig_flowable(
            DIAG / "FIG4_HC_MARL_4.png",
            "Diagram 1: HC-MARL closed-loop design. Offline MMICRL "
            "produces &Theta;<sub>max</sub>; online MAPPO + NSWF + ECBF "
            "run per step.",
            width_in=4.5),
        *fig_flowable(
            DIAG / "FIG6_MATH_LOOP_6.png",
            "Diagram 2: HC-MARL inner loop expressed as the per-step "
            "math. PPO + GAE closes the outer loop after 480 steps.",
            width_in=4.5),
        PageBreak(),
        *fig_flowable(
            DIAG / "FIG1_MAPPO_1.png",
            "Diagram 3: MAPPO baseline (Yu 2022). Centralised critic, "
            "shared-weight actors.",
            width_in=4.0),
        *fig_flowable(
            DIAG / "FIG2_PS_IPPO_2.png",
            "Diagram 4: PS-IPPO baseline (parameter-shared actor and "
            "decentralised critic on local obs; Yu et al. 2022, "
            "Section 4.3). Decentralised critic reads local obs "
            "(red arrow).",
            width_in=4.0),
        *fig_flowable(
            DIAG / "FIG3_MAPPO_LAG_3.png",
            "Diagram 5: MAPPO-Lagrangian baseline (Stooke 2020). Cost "
            "critic + PID-driven &lambda; add a soft safety penalty.",
            width_in=4.0),
        PageBreak(),
        # ----- PROPOSED METHODOLOGY -----
        p("4.2 Proposed Methodology", SECTION_HEAD),
        p(
            "The proposed methodology is a closed-loop pipeline with one "
            "offline phase and one online phase. The offline phase runs "
            "MMICRL on a corpus of unlabelled worker demonstrations to "
            "discover latent worker types z and to learn per-type fatigue "
            "thresholds &Theta;<sub>max</sub>(z). The online phase repeats "
            "six steps every round of the simulated shift, with the updated "
            "physiological state of each worker feeding back into the next "
            "round.",
            BODY,
        ),
        p(
            "Step one observes the physiological state of every worker, "
            "i.e. the resting, active, and fatigued fractions per muscle "
            "group. Step two allocates tasks via the Nash Social Welfare "
            "Function with a divergent disagreement utility, so the "
            "bargaining surplus collapses to negative infinity for any "
            "worker close to the threshold. Step three translates the "
            "assigned task into a per-muscle load demand T<sub>L,g</sub> "
            "via the task profile table. Step four feeds the load demand "
            "and the current state through the MAPPO actor and produces a "
            "nominal neural drive C<sub>nom</sub>. Step five projects "
            "C<sub>nom</sub> through the dual-barrier ECBF quadratic "
            "program to obtain a safe command C*. Step six integrates the "
            "3CC-r ordinary differential equations forward by &Delta;t = "
            "1 minute under the reperfusion switch, returning the new "
            "state to step one.",
            BODY,
        ),
        p(
            "Two structural choices in this pipeline carry most of its "
            "behaviour. The first is that MMICRL feeds &Theta;<sub>max</sub>"
            "(z) into the ECBF, not into the policy, so the safety "
            "guarantee comes from a deterministic projection rather than "
            "from a learned penalty. The second is that the disagreement "
            "utility D<sub>i</sub> grows without bound as M<sub>F</sub> "
            "approaches one, so the allocator cannot find any feasible "
            "assignment that overloads a near-burnout worker, regardless "
            "of how attractive that worker's task-completion utility "
            "looks. Together these two choices make the safety property a "
            "structural fact of the pipeline rather than something the "
            "learning algorithm has to discover.",
            BODY,
        ),
        # ----- ALGORITHM FORMULATED -----
        p("4.3 Algorithm Formulated", SECTION_HEAD),
        p(
            "The full closed-loop control algorithm is summarised below. "
            "It runs once per simulated minute and returns the next-step "
            "physiological state and the executed task allocation.",
            BODY,
        ),
        algorithm_box(),
        # ----- MATHEMATICAL MODELLING -----
        p("4.4 Mathematical Modelling", SECTION_HEAD),
        p("4.4.1 3CC-r Fatigue Dynamics", SUBSECTION_HEAD),
        p(
            "The state vector x<sub>i</sub>(t) of worker i was already "
            "introduced in Section 2.3. The dynamics are the three "
            "coupled ordinary differential equations of Equations 2 to 4, "
            "with the reperfusion-switched recovery rate of Equation 5. "
            "At a steady state where M<sub>R</sub> = 0, the maximum "
            "sustainable neural drive and the corresponding endurance "
            "limit follow:",
            BODY,
        ),
        *display_eq("eq_steady", height=24),
        p(
            "Substituting the calibrated rate constants from Frey-Law "
            "et al. (2012) Table 1, the shoulder produces "
            "&delta;<sub>max</sub> &asymp; 8.4 % under the F = 0.01820, "
            "R = 0.00168 isometric pair, while the ankle reaches "
            "&delta;<sub>max</sub> &asymp; 9.0 % under F = 0.00589, "
            "R = 0.00058. The fatigue-resistance hierarchy of Frey-Law "
            "and Avin (2010) emerges naturally from these algebraic "
            "expressions and is what the warehouse policy ultimately has "
            "to respect.",
            BODY,
        ),
        p("4.4.2 MMICRL Constraint Learning", SUBSECTION_HEAD),
        p(
            "Given a dataset of unlabelled demonstrations {&tau;} drawn "
            "from workers of unknown types, MMICRL maximises a weighted "
            "entropy objective:",
            BODY,
        ),
        *display_eq("eq_mmicrl_obj", height=24),
        p(
            "which decomposes into",
            BODY,
        ),
        *display_eq("eq_mmicrl_mi", height=24),
        p(
            "where I(&tau;; z) is the mutual information between "
            "trajectories and worker type. Setting "
            "&lambda;<sub>1</sub> = &lambda;<sub>2</sub> = &lambda; gives "
            "pure mutual information maximisation. A Conditional Flow-"
            "based Density Estimator (Masked Autoregressive Flow with "
            "MADE blocks) recovers per-type density estimates whose "
            "90th-percentile fatigue value at each muscle group becomes "
            "the raw &Theta;<sub>max</sub>(z) before the rescale-into-"
            "feasibility step. The implementation lives in <i>hcmarl/"
            "mmicrl.py</i>.",
            BODY,
        ),
        p("4.4.3 ECBF Safety Filter", SUBSECTION_HEAD),
        p(
            "The primary barrier protects the fatigue ceiling:",
            BODY,
        ),
        *display_eq("eq_h", height=29),
        p(
            "Differentiating once gives the time derivative",
            BODY,
        ),
        *display_eq("eq_h_dot", height=24),
        p(
            "in which the control input C(t) does not appear, because "
            "C(t) feeds the M<sub>A</sub> equation, not the M<sub>F</sub> "
            "equation, in one differentiation step. C(t) appears only "
            "after a second differentiation, which makes the system "
            "relative-degree two:",
            BODY,
        ),
        *display_eq("eq_relative_degree", height=24),
        p(
            "An Exponential Control Barrier Function handles this case by "
            "constructing the composite barriers",
            BODY,
        ),
        *display_eq("eq_psi", height=24),
        p(
            "and enforcing the condition",
            BODY,
        ),
        *display_eq("eq_psi_dot", height=29),
        p(
            "with the ECBF gains taken as &alpha;<sub>1</sub> = "
            "&alpha;<sub>2</sub> = 0.05 in <i>config/hcmarl_full_config."
            "yaml</i>. A secondary barrier prevents depletion of the "
            "resting pool:",
            BODY,
        ),
        *display_eq("eq_h2", height=29),
        p(
            "which is relative-degree one and is enforced by a standard "
            "CBF constraint with gain &alpha;<sub>3</sub> = 0.1. The two "
            "constraints combine into a single quadratic program that "
            "minimally adjusts the policy's nominal command:",
            BODY,
        ),
        *display_eq("eq_qp", height=24),
        p(
            "The design requirement on the learned threshold is",
            BODY,
        ),
        *display_eq("eq_feasibility", height=24),
        p(
            "which guarantees the safe set is non-empty even during "
            "rest-phase reperfusion. Safety carries across the work-rest "
            "mode switch by a Nagumo invariance argument, because the "
            "work phase (R<sub>eff</sub> = R) is the worst case and the "
            "rest phase (R<sub>eff</sub> = R&middot;r > R) only "
            "strengthens the barrier. The QP is solved by OSQP through "
            "CVXPY in <i>hcmarl/ecbf_filter.py</i>, with a slack-augmented "
            "fallback that activates whenever the strict QP is infeasible "
            "and writes the slack value into the diagnostics record.",
            BODY,
        ),
        p("4.4.4 Nash Bargaining with Divergent Disagreement", SUBSECTION_HEAD),
        p(
            "Each worker's disagreement utility is the divergent function",
            BODY,
        ),
        *display_eq("eq_disagreement", height=27),
        p(
            "with &kappa; &gt; 0. When M<sub>F</sub> &asymp; 0 the worker "
            "is fresh and has low bargaining power (D<sub>i</sub> "
            "&asymp; 0). As M<sub>F</sub> &rarr; 1 the disagreement "
            "utility diverges to positive infinity, which makes the "
            "log-surplus in the NSWF objective collapse to negative "
            "infinity. The allocator solves",
            BODY,
        ),
        *display_eq("eq_nswf", height=29),
        p(
            "by reducing the assignment problem to the Hungarian algorithm "
            "on the cost matrix C[i, j] = -ln(U(i,j) - D<sub>i</sub>), as "
            "implemented in <i>hcmarl/nswf_allocator.py</i> via "
            "<i>scipy.optimize.linear_sum_assignment</i>. If no productive "
            "assignment yields a positive surplus for a given worker, "
            "the allocator routes that worker to the rest column instead, "
            "which is the mandatory rest condition formalised in "
            "Definition 6.1 of the project mathematical model.",
            BODY,
        ),
        # ----- ENVIRONMENT -----
        p("4.5 Environment", SECTION_HEAD),
        p(
            "The simulation environment is a custom PettingZoo parallel "
            "warehouse env, implemented in <i>hcmarl/envs/"
            "pettingzoo_wrapper.py</i> and wrapped for the training script "
            "by <i>hcmarl/warehouse_env.py</i>. Each agent is one human "
            "worker with a six-muscle physiological state vector, "
            "exposing per-muscle (M<sub>R</sub>, M<sub>A</sub>, "
            "M<sub>F</sub>) along with the current task assignment. The "
            "task action space spans six options: heavy lift, light "
            "sort, carry, overhead reach, push cart, and rest. Each "
            "task carries a per-muscle load profile T<sub>L,g</sub> drawn "
            "from the warehouse EMG literature (Skals 2021, Skovlund "
            "2022a/b, Kao 2015, Byström and Fransson-Hall 1994), with "
            "heavy lift and carry calibrated against the NIOSH RLE "
            "Lifting Index of Waters et al. (1993). All sources are "
            "documented in <i>config/hcmarl_full_config.yaml</i>.",
            BODY,
        ),
        *env_table(),
        p(
            "Reward shaping in <i>hcmarl/envs/reward_functions.py</i> "
            "combines the NSWF surplus, a peak-fatigue penalty, an "
            "ECBF-intervention penalty, and a task-completion bonus. "
            "Episodes run for max_steps = 480 minutes (an 8-hour shift) "
            "at &Delta;t = 1 minute. Workers default to N = 6 in the "
            "headline configuration, with task allocation re-solved every "
            "step. The worker calibration data (Path G profiles) come "
            "from the WSD4FEDSRM dataset and live in <i>config/"
            "pathg_profiles.json</i>.",
            BODY,
        ),
        p(
            "The training script <i>scripts/train.py</i> drives the "
            "MAPPO actor-critic update on the parallel PettingZoo env, "
            "with the headline hyperparameters lr_actor = 0.0003, "
            "lr_critic = 0.001, &gamma; = 0.99, GAE &lambda; = 0.95, PPO "
            "clip &epsilon; = 0.2, entropy coefficient annealed linearly "
            "from 0.05 to 0.01 across the full 2,000,160 steps, and a "
            "max gradient norm of 0.5. Mini-batches of 256 steps are "
            "drawn from the rollout buffer per epoch, with ten epochs "
            "per update. Determinism is enforced via "
            "<i>cudnn.deterministic = True</i>, which costs roughly 1.5x "
            "wall-clock time but produces bit-exact reproducibility "
            "across reruns at the same seed. Each seed runs as a "
            "separate process under <i>scripts/run_baselines.py</i> for "
            "the EXP1 grid and <i>scripts/run_ablations.py</i> for the "
            "EXP2 leave-one-out grid, both of which read the seed list "
            "and the per-method hyperparameters from <i>config/"
            "experiment_matrix.yaml</i> rather than from hard-coded "
            "constants.",
            BODY,
        ),
        p(
            "The pre-flight gate in EXP0 checks the env one more time "
            "before any GPU run. <i>scripts/experiment_0_runner.py</i> "
            "executes nine self-contained validations covering the "
            "pytest suite, the constants ledger, the 3CC-r conservation "
            "invariant, the Path G calibration profiles, the NSWF "
            "allocator across five seed scenarios, an MMICRL fit on "
            "Path G demonstrations, an ECBF state sweep on a 126-state "
            "grid, and smoke forward-passes for every method and "
            "ablation. The gate's job is to assert that the codebase is "
            "numerically intact and ready to be put on a GPU, not to "
            "evaluate a trained policy. The actual scientific weight is "
            "carried by the three training experiments documented in "
            "Chapter 4.",
            BODY,
        ),
        PageBreak(),
    ]


def algorithm_box() -> Table:
    text = (
        "<font name=\"Times-Bold\">Algorithm 1.</font> HC-MARL closed-loop "
        "round at simulated time <i>t</i>.<br/><br/>"
        "<b>Inputs:</b> physiological state x<sub>i</sub>(t) for each "
        "worker i, learned thresholds &Theta;<sub>max</sub>(z<sub>i</sub>), "
        "task profile table {T<sub>L,g</sub>}, MAPPO actor &pi;<sub>"
        "&theta;</sub>.<br/><br/>"
        "<b>Step 1.</b> Read M<sub>R</sub>, M<sub>A</sub>, M<sub>F</sub> "
        "for every worker and every muscle group from the env state.<br/>"
        "<b>Step 2.</b> Compute D<sub>i</sub> = "
        "&kappa;&middot;(M<sub>F</sub><sup>(i)</sup>)<sup>2</sup> / "
        "(1 - M<sub>F</sub><sup>(i)</sup>) for every worker. Solve the "
        "Hungarian assignment "
        "max<sub>&tau;</sub> &Sigma;<sub>i</sub> ln(U(i, j) - D<sub>i</sub>). "
        "Workers without a positive-surplus task are routed to rest.<br/>"
        "<b>Step 3.</b> Convert each worker's assigned task j into a "
        "per-muscle load demand T<sub>L,g</sub><sup>(i)</sup>(t) by "
        "lookup in the task profile table.<br/>"
        "<b>Step 4.</b> Forward the augmented observation "
        "(state, T<sub>L</sub>) through the MAPPO actor to get the "
        "nominal neural drive C<sub>nom</sub>(t).<br/>"
        "<b>Step 5.</b> Solve the dual-barrier QP "
        "min<sub>C</sub> ||C - C<sub>nom</sub>||<sup>2</sup> "
        "subject to the ECBF constraint and the resting-floor CBF "
        "constraint. If strictly infeasible, activate the slack-"
        "augmented relaxation and record slack &gt; 0 in diagnostics. "
        "Return the safe command C*.<br/>"
        "<b>Step 6.</b> Integrate the 3CC-r ODEs forward by &Delta;t = 1 "
        "minute using <i>scipy.integrate.solve_ivp</i> RK45 with "
        "R<sub>eff</sub>(t) chosen by the reperfusion switch.<br/><br/>"
        "<b>Outputs:</b> updated state x<sub>i</sub>(t + &Delta;t), "
        "executed assignment &tau;<sub>ij*</sub>, ECBF diagnostics "
        "(h, h<sub>2</sub>, slack values, was_clipped flag)."
    )
    tbl = Table([[Paragraph(text, BODY)]], colWidths=[BODY_W])
    tbl.setStyle(TableStyle([
        ("BOX",          (0, 0), (-1, -1), 0.7, colors.black),
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#F8F8F8")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    return tbl


def env_table() -> list:
    title = p("Table 1: Warehouse environment configuration.", TABLE_TITLE)
    cell = ParagraphStyle("TCell", parent=BODY, fontSize=9.5, leading=11.5,
                          alignment=TA_LEFT, spaceBefore=0, spaceAfter=0)
    cell_b = ParagraphStyle("TCellB", parent=cell, fontName="Times-Bold")

    def row(a, b, c):
        return [Paragraph(a, cell), Paragraph(b, cell), Paragraph(c, cell)]

    rows = [
        [Paragraph("<b>Parameter</b>", cell_b),
         Paragraph("<b>Value</b>", cell_b),
         Paragraph("<b>Source / file</b>", cell_b)],
        row("Number of workers (N)",
            "6 default, scaling N &isin; {3, 4, 6, 8, 12}",
            "config/hcmarl_full_config.yaml"),
        row("Episode length",
            "480 minutes (8-hour shift)",
            "config/hcmarl_full_config.yaml"),
        row("Sampling interval &Delta;t",
            "1.0 minute per step",
            "config/hcmarl_full_config.yaml"),
        row("Muscle groups",
            "shoulder, ankle, knee, elbow, trunk, grip",
            "hcmarl/three_cc_r.py"),
        row("Task action space",
            "heavy_lift, light_sort, carry, overhead_reach, "
            "push_cart, rest",
            "config/hcmarl_full_config.yaml"),
        row("Per-muscle (F, R)",
            "Frey-Law et al. 2012 Table 1 (isometric)",
            "hcmarl/three_cc_r.py L82-L87"),
        row("Reperfusion multiplier r",
            "15 (shoulder, ankle, knee, elbow, trunk); 30 (grip)",
            "Looft et al. 2018 Table 2"),
        row("&Theta;<sub>max</sub> ceilings (per-muscle)",
            "shoulder 0.70, ankle 0.80, knee 0.60, elbow 0.45, "
            "trunk 0.65, grip 0.45",
            "config/hcmarl_full_config.yaml"),
        row("ECBF gains",
            "&alpha;<sub>1</sub> = &alpha;<sub>2</sub> = 0.05, "
            "&alpha;<sub>3</sub> = 0.1",
            "config/hcmarl_full_config.yaml ecbf:"),
        row("NSWF &kappa; (disagreement scale)",
            "1.0",
            "config/hcmarl_full_config.yaml"),
        row("Total training steps",
            "2,000,160 per seed",
            "config/hcmarl_full_config.yaml training:"),
        row("MMICRL k_range / n_iter",
            "[1, 5] / 150",
            "config/hcmarl_full_config.yaml mmicrl:"),
    ]
    tbl = Table(rows, colWidths=[1.7 * inch, 2.5 * inch, 1.8 * inch])
    tbl.setStyle(TableStyle([
        ("BOX",          (0, 0), (-1, -1), 0.7, colors.black),
        ("INNERGRID",    (0, 0), (-1, -1), 0.4, colors.black),
        ("FONTNAME",     (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9.5),
        ("LEADING",      (0, 0), (-1, -1), 11.5),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#EEEEEE")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    return [title, tbl]


# ---------------------------------------------------------------------------
# 5. CHAPTER 4: RESULTS AND DISCUSSIONS (pages 15-20)
# ---------------------------------------------------------------------------

def chapter4() -> list:
    out = [
        p("5. CHAPTER 4: RESULTS AND DISCUSSIONS", CHAPTER_HEAD),
        p(
            "Empirical evaluation runs across four experiments: EXP0 is a "
            "static-property pre-flight gate (560 pytest cases, constants "
            "ledger, smoke forward-pass); EXP1 trains HC-MARL against three "
            "cooperative MARL baselines (MAPPO, PS-IPPO, MAPPO-Lag) for ten "
            "seeds at 2,000,160 steps each on an L4 GPU; EXP2 runs four "
            "leave-one-out ablations (no_ecbf, no_nswf, no_divergent, "
            "no_reperfusion) at five seeds each; EXP3 isolates MMICRL's "
            "type-discovery validity on synthetic K = 3 and K = 1 "
            "populations, then demonstrates the paired-policy effect of "
            "MMICRL-on versus MMICRL-off on synthetic K = 3 data. "
            "Every figure here was produced by <i>scripts/"
            "analyze_results_4.py</i> and <i>scripts/visualize_results_4."
            "py</i> at 300 dpi. Confidence intervals are 95 % stratified "
            "bootstrap from <i>hcmarl/aggregation.py</i>, ten thousand "
            "resamples, following the Agarwal et al. (2021) recipe.",
            BODY,
        ),
        # ----- EXP0 / pre-flight -----
        p("5.1 Pre-Flight Numerical Gate (EXP0)", SECTION_HEAD),
        p(
            "EXP0 answers a different question than the learning-curve "
            "experiments. It asks whether the codebase is numerically "
            "intact and ready to be put on a GPU, not how well the trained "
            "policy performs. The gate exposes nine static checks "
            "covering the test suite, the constants ledger, the 3CC-r "
            "conservation invariant, the Path G calibration profiles, the "
            "NSWF allocator on a five-seed scenario set, an MMICRL fit on "
            "Path G demonstrations, an ECBF state sweep, and smoke "
            "forward-passes for every method and ablation.",
            BODY,
        ),
        *fig_flowable(
            RESULTS4 / "Result 0 Analysis" / "fig_01_test_suite_breakdown.png",
            "Figure 1: pytest suite breakdown across all 31 test files. "
            "560 passed, 2 skipped, 0 failed; integration suites "
            "(test_phase3, test_all_methods, test_round8_s4) dominate "
            "wall time, as expected.",
            width_in=5.2),
        *fig_flowable(
            RESULTS4 / "Result 0 Analysis" / "fig_07_constants_provenance.png",
            "Figure 2: Constants provenance ledger. 32 of 38 constants "
            "are PRIMARY-source PDF-verified (Frey-Law 2012, Frey-Law and "
            "Avin 2010, Looft 2018, Waters 1993); 6 are honestly "
            "labelled DESIGN choices.",
            width_in=5.2),
        *fig_flowable(
            RESULTS4 / "Result 0 Analysis" / "fig_06_three_cc_r_smoke_curves.png",
            "Figure 3: 3CC-r smoke trajectory under the headline config "
            "vs the no_reperfusion ablation. The reperfusion-disabled "
            "rung crosses the shoulder &Theta;<sub>max</sub> = 0.7 line "
            "before the 100-step mark, which confirms the rest-recovery "
            "term is doing real work in the env.",
            width_in=5.2),
        p(
            "The reperfusion ablation already separates from the rest of "
            "the configurations at smoke time (mean reward "
            "&minus;17.093 vs &minus;7.261 baseline, peak M<sub>F</sub> "
            "0.7317 vs 0.5947). Every other separation, however, requires "
            "actual training, which is what the next three experiments "
            "deliver.",
            BODY,
        ),
        # ----- EXP1 / baselines -----
        p("5.2 HC-MARL vs Cooperative MARL Baselines (EXP1)", SECTION_HEAD),
        p(
            "EXP1 trains the four methods at ten seeds each. The reward "
            "ladder at the final 500-episode window puts HC-MARL at IQM "
            "&minus;1389.2, MAPPO at &minus;9062.3, PS-IPPO at &minus;"
            "8868.1, and MAPPO-Lag at &minus;9039.9, a separation of "
            "roughly 6.5x in absolute reward. The probability of "
            "improvement of HC-MARL over each baseline is 1.000 across "
            "every seed pair, which is the strongest possible separation "
            "the bootstrap procedure can return.",
            BODY,
        ),
        *fig_flowable(
            RESULTS4 / "Result 1 Analysis" / "fig_01_learning_curves.png",
            "Figure 4: Learning curves across the four methods, ten seeds "
            "per arm, IQM and 95 % bootstrap CI ribbons. HC-MARL crashes "
            "to a tight plateau near &minus;1380 by episode 500. Three "
            "baselines plateau near &minus;9000.",
            width_in=5.5),
        *fig_flowable(
            RESULTS4 / "Result 1 Analysis" / "fig_02_iqm_bars.png",
            "Figure 5: Final-window IQM bars (final reward, best reward, "
            "safety rate, violation rate). All four methods plateau above "
            "0.95 safety; the headline gap lives in the reward channel.",
            width_in=5.2),
        *fig_flowable(
            RESULTS4 / "Result 1 Analysis" / "fig_04_pairwise_poi.png",
            "Figure 6: Pairwise probability of improvement on the "
            "final-window reward. Every cell in the HC-MARL row reads "
            "1.00, showing every HC-MARL seed beats every baseline seed.",
            width_in=4.6),
        p(
            "The story underneath the headline number is mechanistic. "
            "All four methods converge to roughly the same task-completion "
            "count, around 2,580 to 2,620 tasks per shift, with HC-MARL "
            "at 2,602. They are doing similar amounts of work. The "
            "reward difference therefore lives in the shaping channels "
            "that are not raw task count: peak-fatigue penalty, divergent "
            "disagreement utility, NSWF surplus, and ECBF intervention "
            "rate. HC-MARL operates at a peak M<sub>F</sub> of 0.695, "
            "while the three baselines sit closer to 0.563. HC-MARL is "
            "deliberately running workers harder under tighter safety "
            "control, while the baselines stay safe by underutilising "
            "the agents. Tightening the operating point and still "
            "respecting the ECBF barrier is what produces the gap.",
            BODY,
        ),
        *fig_flowable(
            RESULTS4 / "Result 1 Analysis" / "fig_05_metric_heatmap.png",
            "Figure 7: Metric heatmap, all metrics z-scored per column. "
            "Green is better. HC-MARL dominates the reward, surplus, and "
            "tasks-completed columns; baselines dominate only on the "
            "intervention-rate column, which is consistent with "
            "underutilisation.",
            width_in=5.2),
        *fig_flowable(
            RESULTS4 / "Result 1 Analysis" / "fig_03_per_seed_strip.png",
            "Figure 8: Per-seed strip plot. HC-MARL's ten seeds form a "
            "tight cluster from &minus;1410 to &minus;1363. PS-IPPO "
            "shows bimodal behaviour, with four seeds escaping the "
            "&minus;9000 attractor.",
            width_in=5.2),
        # ----- EXP2 / ablations -----
        p("5.3 Component-Level Attribution (EXP2)", SECTION_HEAD),
        p(
            "EXP2 turns each HC-MARL component off, one at a time, "
            "keeping every other component active. Five seeds per rung. "
            "The HC-MARL full anchor from EXP1 (IQM = &minus;1389.2) "
            "sits as the dashed line in the learning-curve panel. The "
            "delta column carries the attribution.",
            BODY,
        ),
        *ablation_table(),
        *fig_flowable(
            RESULTS4 / "Result 2 Analysis" / "fig_01_learning_curves.png",
            "Figure 9: Ablation learning curves, five seeds per rung, "
            "with the HC-MARL full anchor from EXP1 plotted as a dashed "
            "reference. The reperfusion-disabled rung collapses to a "
            "different reward order of magnitude.",
            width_in=5.5),
        *fig_flowable(
            RESULTS4 / "Result 2 Analysis" / "fig_02_ablation_reward_bar.png",
            "Figure 10: Ranked ablation reward delta against HC-MARL "
            "full. Removing the reperfusion term costs the policy "
            "roughly 52,000 reward units; the other three components "
            "each cost between 6,000 and 7,800 units.",
            width_in=5.0),
        p(
            "Reading these numbers component by component: the "
            "no_reperfusion rung is catastrophic by design, because "
            "disabling the rest-recovery term in the ODE breaks the "
            "physiology that the rest of the framework is built on. "
            "Safety rate falls to 0.202 and peak M<sub>F</sub> "
            "saturates at 0.818. The fact that this rung still completes "
            "2,332 tasks confirms the policy will keep pushing in the "
            "absence of working recovery dynamics, which is exactly the "
            "warehouse-injury failure mode the framework is built to "
            "prevent. Removing ECBF (no_ecbf) drops task completion to "
            "1,458, the cleanest reward-equals-work-times-safety "
            "signature in the dataset: without the safety filter, the "
            "policy cannot trust itself to push hard, so it under-"
            "utilises capacity to stay safe. Removing NSWF or removing "
            "the divergent disagreement utility costs roughly 6,000 to "
            "7,800 reward units each, and the no_divergent rung shows a "
            "wide 95 % CI of about 3,500 units, which suggests the "
            "disagreement utility is what stabilises credit assignment "
            "across seeds.",
            BODY,
        ),
        *fig_flowable(
            RESULTS4 / "Result 2 Analysis" / "fig_03_metric_heatmap.png",
            "Figure 11: Ablation metric heatmap. The no_ecbf rung shows "
            "an unexpectedly high safety score, which is the "
            "underutilisation tell: it stays safe by completing fewer "
            "tasks rather than by managing the fatigue trajectory.",
            width_in=5.2),
        *fig_flowable(
            RESULTS4 / "Combined" / "fig_cross_experiment_reward_ladder.png",
            "Figure 12: Cross-experiment reward ladder, EXP1 baselines "
            "and EXP2 ablations on a single chart. HC-MARL full sits at "
            "the top, no_reperfusion sits at the bottom, the three "
            "baselines and the three remove-one ablations cluster in "
            "the middle band.",
            width_in=5.5),
        # ----- EXP3 -----
        p("5.4 MMICRL Type-Discovery Validity (EXP3)", SECTION_HEAD),
        p(
            "EXP3 separates two questions about MMICRL. The first is "
            "whether the algorithm recovers ground-truth latent types "
            "when those types exist in the data. The second is whether "
            "running MMICRL changes downstream policy performance "
            "compared with running with a flat configured threshold. "
            "Part 1 of EXP3 answers the first question on synthetic 3CC-r "
            "demonstrations with three known F values per shoulder. "
            "Part 2 demonstrates the second question through a paired "
            "MMICRL-on versus MMICRL-off policy comparison on synthetic "
            "K = 3 data.",
            BODY,
        ),
        *fig_flowable(
            RESULTS4 / "Result 3 Analysis" / "fig_part1_kselection.png",
            "Figure 13: MMICRL K-discovery on the synthetic K = 3 regime. "
            "The confusion matrix is fully diagonal, with ARI = 1.000 and "
            "MI = 1.0986 (the theoretical log(3) ceiling). On the "
            "homogeneous K = 1 regime (not shown) MMICRL correctly "
            "collapses, returning K = 1 and MI = 0.000.",
            width_in=5.2),
        p(
            "On the K = 3 synthetic regime MMICRL recovers the "
            "ground-truth clustering exactly, with ARI = 1.000 and "
            "mutual information at the theoretical log(3) = 1.0986 "
            "ceiling. On the K = 1 homogeneous regime MMICRL correctly "
            "returns K = 1 with zero mutual information, which is the "
            "honest negative control. Together these two regimes "
            "establish that MMICRL is operationally correct: when types "
            "exist it finds them, when they do not it does not "
            "hallucinate them.",
            BODY,
        ),
        p(
            "On the real WSD4FEDSRM single-shoulder calibration data, "
            "the EXP1 production runs report MI collapse on every seed "
            "(mutual_information &lt; 0.01, mi_collapsed = true). The "
            "framework is built for that case: when MI collapses, the "
            "rescale-into-feasibility helper in <i>hcmarl/utils.py</i> "
            "pulls every per-type threshold to the configured floor, "
            "which means the downstream policy receives a stable "
            "&Theta;<sub>max</sub> regardless of how MMICRL clustered. "
            "MMICRL is included in the framework not because it currently "
            "improves real-data performance, but because it is ready for "
            "the multi-muscle calibration where occupational worker "
            "populations have been shown to carry separable types.",
            BODY,
        ),
        *fig_flowable(
            RESULTS4 / "Result 3 Analysis" / "fig_part2_iqm_bars_clean.png",
            "Figure 14: IQM bars for the MMICRL-on vs MMICRL-off "
            "paired comparison on synthetic K = 3 data. HCMARL with "
            "MMICRL reaches IQM -1149 versus -2497 without, with safety "
            "rate 0.97 versus 0.94.",
            width_in=5.0),
        # ----- DISCUSSION -----
        p("5.5 Discussion", SECTION_HEAD),
        p(
            "Three threads tie the four experiments together. First, the "
            "headline reward gap in EXP1 is real, large, and "
            "decomposable. The probability of improvement is 1.00 across "
            "every seed pair, the per-seed cluster for HC-MARL is tight, "
            "and the gap survives every reasonable robustness check. "
            "Second, the gap is not an accident of any single component. "
            "EXP2 attributes a meaningful share of the reward to each of "
            "the four mechanisms, with reperfusion as the largest "
            "contributor by an order of magnitude and the other three "
            "(ECBF, NSWF, divergent disagreement) clustering at a similar "
            "scale. Third, MMICRL is honestly characterised. It works "
            "as advertised on synthetic data where types exist, it "
            "correctly degenerates on real data where they do not, and "
            "the framework's behaviour is invariant to that degeneracy "
            "because of the rescale-into-feasibility safeguard.",
            BODY,
        ),
        p(
            "The honest limitations are visible in the results. The "
            "five-seed ablation grid in EXP2 is half the seed count "
            "of EXP1, and the no_divergent rung in particular shows "
            "wide CI that a ten-seed re-run would tighten. The "
            "single-muscle Path G calibration cannot exhibit cleanly "
            "separable types, which is why MMICRL collapses on real "
            "data; multi-muscle calibration is left as the natural "
            "next step.",
            BODY,
        ),
        PageBreak(),
    ]
    return out


def ablation_table() -> list:
    title = p("Table 2: Ablation reward ladder, final 500-episode window.",
              TABLE_TITLE)
    cell = ParagraphStyle("AblCell", parent=BODY, fontSize=9, leading=11,
                          alignment=TA_CENTER, spaceBefore=0, spaceAfter=0)
    cell_l = ParagraphStyle("AblCellL", parent=cell, alignment=TA_LEFT)
    cell_b = ParagraphStyle("AblCellB", parent=cell, fontName="Times-Bold")

    def row(label, n, iqm, ci, delta, safe, peak):
        return [Paragraph(label, cell_l), Paragraph(n, cell),
                Paragraph(iqm, cell), Paragraph(ci, cell),
                Paragraph(delta, cell), Paragraph(safe, cell),
                Paragraph(peak, cell)]

    rows = [
        [Paragraph("<b>Rung</b>", cell_b),
         Paragraph("<b>n</b>", cell_b),
         Paragraph("<b>Final reward IQM</b>", cell_b),
         Paragraph("<b>95% CI</b>", cell_b),
         Paragraph("<b>&Delta; vs HC-MARL full</b>", cell_b),
         Paragraph("<b>Safety rate</b>", cell_b),
         Paragraph("<b>Peak M<sub>F</sub></b>", cell_b)],
        row("HC-MARL full (EXP1 anchor)", "10", "&minus;1389.2",
            "[&minus;1401.2, &minus;1378.0]", "&mdash;",
            "0.965", "0.695"),
        row("no_nswf", "5", "&minus;7579.5",
            "[&minus;8269.7, &minus;6968.6]", "&minus;6190.3",
            "0.825", "0.706"),
        row("no_ecbf", "5", "&minus;9075.6",
            "[&minus;9086.3, &minus;9048.6]", "&minus;7686.4",
            "0.968", "0.563"),
        row("no_divergent", "5", "&minus;9187.4",
            "[&minus;11582.4, &minus;8085.3]", "&minus;7798.2",
            "0.802", "0.726"),
        row("no_reperfusion", "5", "&minus;53775.0",
            "[&minus;60267.4, &minus;50514.2]", "&minus;52385.8",
            "0.202", "0.818"),
    ]
    tbl = Table(rows,
                colWidths=[1.3 * inch, 0.32 * inch, 0.95 * inch,
                           1.20 * inch, 0.85 * inch, 0.65 * inch,
                           0.55 * inch])
    tbl.setStyle(TableStyle([
        ("BOX",          (0, 0), (-1, -1), 0.7, colors.black),
        ("INNERGRID",    (0, 0), (-1, -1), 0.4, colors.black),
        ("FONTNAME",     (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("LEADING",      (0, 0), (-1, -1), 11),
        ("ALIGN",        (1, 1), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#EEEEEE")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    return [title, tbl]


# ---------------------------------------------------------------------------
# 6. CONCLUSION (page 21)
# ---------------------------------------------------------------------------

def conclusion() -> list:
    return [
        p("6. CONCLUSION", CHAPTER_HEAD),
        p(
            "HC-MARL is a closed-loop human-centric multi-agent "
            "reinforcement-learning framework that addresses the four "
            "warehouse-safety sub-problems jointly. A physiologically "
            "calibrated 3CC-r ODE system tracks per-worker fatigue in "
            "real time. A multi-modal inverse constrained learner "
            "discovers latent worker types and per-type fatigue "
            "thresholds without supervision. A dual-barrier exponential "
            "control barrier function enforces those thresholds with a "
            "Nagumo invariance argument that carries across the "
            "work-rest mode switch. A Nash Social Welfare allocator with "
            "a divergent disagreement utility makes overloading a "
            "fatigued worker mathematically impossible rather than "
            "merely costly.",
            BODY,
        ),
        p(
            "Empirical results across four experiments support the "
            "design. The pre-flight gate (EXP0) confirms numerical "
            "integrity at 560 of 562 pytest cases passing, the 3CC-r "
            "conservation invariant exact to machine precision, and "
            "all 38 numerical constants traced to a primary source or "
            "honestly labelled as a design choice. The headline "
            "experiment (EXP1) shows HC-MARL at IQM &minus;1389.2 "
            "against three baselines clustered near &minus;9000, with "
            "probability of improvement equal to 1.00 across every "
            "seed pair. The ablation (EXP2) attributes the gap "
            "component by component, with the reperfusion term carrying "
            "the largest single contribution and the other three "
            "components contributing at a comparable scale. The "
            "constraint-learning experiment (EXP3 Part 1) confirms "
            "MMICRL recovers ground-truth latent types on synthetic "
            "K = 3 data with ARI = 1.000 and correctly collapses on "
            "K = 1 homogeneous data.",
            BODY,
        ),
        p(
            "The honest limitations are visible in the same results. "
            "MMICRL collapses on real WSD4FEDSRM single-shoulder "
            "calibration data, and the framework absorbs that collapse "
            "through the rescale-into-feasibility helper in "
            "<i>hcmarl/utils.py</i> rather than papering over it. "
            "The five-seed ablation grid is half the seed count of "
            "the headline experiment, and the no_divergent rung in "
            "particular shows wide confidence intervals that a "
            "ten-seed re-run would tighten.",
            BODY,
        ),
        p(
            "The natural next step is multi-muscle calibration on a "
            "real warehouse cohort, where occupational worker "
            "populations have a stronger prior of separable types than "
            "the single-shoulder Path G slice. The framework is "
            "already structured around that extension. A second "
            "extension is real-time deployment with sensor-derived "
            "EMG estimates substituted for the simulated 3CC-r state, "
            "for which the dual-barrier QP latency budget on commodity "
            "hardware is the only open engineering question. Both "
            "extensions are scoped against the existing codebase rather "
            "than a re-design, which is the operational definition of "
            "research that has finished one cycle and is ready to start "
            "another.",
            BODY,
        ),
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# 7. REFERENCES (pages 22-24)
# ---------------------------------------------------------------------------

def references() -> list:
    refs = [
        "Liu JZ, Brown RW, Yue GH. A dynamical model of muscle "
        "activation, fatigue, and recovery. <i>Biophysical Journal</i>, "
        "82(5):2344–2359, 2002.",

        "Xia T, Frey-Law LA. A theoretical approach for modeling "
        "peripheral muscle fatigue and recovery. <i>Journal of "
        "Biomechanics</i>, 41(14):3046–3052, 2008.",

        "Frey-Law LA, Looft JM, Heitsman J. A three-compartment muscle "
        "fatigue model accurately predicts joint-specific maximum "
        "endurance times. <i>Journal of Biomechanics</i>, "
        "45(10):1803–1808, 2012.",

        "Looft JM, Herkert N, Frey-Law L. Modification of a "
        "three-compartment muscle fatigue model to predict peak torque "
        "decline during intermittent tasks. <i>Journal of "
        "Biomechanics</i>, 77:16–25, 2018.",

        "Looft JM, Frey-Law LA. Adapting a fatigue model for shoulder "
        "flexion fatigue. <i>Journal of Biomechanics</i>, 106:109762, "
        "2020.",

        "Frey-Law LA, Avin KG. Endurance time is joint-specific: a "
        "modelling and meta-analysis investigation. <i>Ergonomics</i>, "
        "53(1):109–129, 2010.",

        "Ziebart BD, Maas AL, Bagnell JA, Dey AK. Maximum entropy "
        "inverse reinforcement learning. <i>Proceedings of the 23rd "
        "AAAI Conference on Artificial Intelligence</i>, "
        "pp. 1433–1438, 2008.",

        "Qiao G, Liu G, Poupart P, Xu Z. Multi-modal inverse "
        "constrained reinforcement learning from a mixture of "
        "demonstrations. <i>Advances in Neural Information Processing "
        "Systems 36</i> (NeurIPS 2023), 2023.",

        "Malik S, Anwar U, Aghasi A, Ahmed A. Inverse constrained "
        "reinforcement learning. <i>Proceedings of the 38th "
        "International Conference on Machine Learning</i>, PMLR "
        "139:7390–7399, 2021.",

        "Li Y, Song J, Ermon S. InfoGAIL: interpretable imitation "
        "learning from visual demonstrations. <i>Advances in Neural "
        "Information Processing Systems 30</i> (NeurIPS 2017), "
        "pp. 3812–3822, 2017.",

        "Achiam J, Held D, Tamar A, Abbeel P. Constrained policy "
        "optimization. <i>Proceedings of the 34th International "
        "Conference on Machine Learning</i>, PMLR 70:22–31, 2017.",

        "Nguyen Q, Sreenath K. Exponential control barrier functions "
        "for enforcing high relative-degree safety-critical "
        "constraints. <i>2016 American Control Conference (ACC)</i>, "
        "pp. 322–328, 2016.",

        "Xiao W, Belta C. Control barrier functions for systems with "
        "high relative degree. <i>2019 IEEE 58th Conference on "
        "Decision and Control (CDC)</i>, pp. 474–479, 2019.",

        "Ames AD, Xu X, Grizzle JW, Tabuada P. Control barrier "
        "function based quadratic programs for safety critical "
        "systems. <i>IEEE Transactions on Automatic Control</i>, "
        "62(8):3861–3876, 2017.",

        "Ames AD, Coogan S, Egerstedt M, Notomista G, Sreenath K, "
        "Tabuada P. Control barrier functions: theory and applications. "
        "<i>2019 18th European Control Conference (ECC)</i>, "
        "pp. 3420–3431, 2019.",

        "Prajna S, Jadbabaie A. Safety verification of hybrid systems "
        "using barrier certificates. <i>Hybrid Systems: Computation "
        "and Control (HSCC 2004)</i>, LNCS 2993, pp. 477–492, 2004.",

        "Nash JF. The bargaining problem. <i>Econometrica</i>, "
        "18(2):155–162, 1950.",

        "Nash JF. Two-person cooperative games. <i>Econometrica</i>, "
        "21(1):128–140, 1953.",

        "Navon A, Shamsian A, Achituve I, Maron H, Kawaguchi K, "
        "Chechik G, Fetaya E. Multi-task learning as a bargaining "
        "game. <i>Proceedings of the 39th International Conference "
        "on Machine Learning</i>, PMLR 162:16428–16446, 2022.",

        "Binmore K, Shaked A, Sutton J. An outside option experiment. "
        "<i>Quarterly Journal of Economics</i>, 104(4):753–770, "
        "1989.",

        "Kaneko M, Nakamura K. The Nash social welfare function. "
        "<i>Econometrica</i>, 47(2):423–435, 1979.",

        "Nagumo M. Über die Lage der Integralkurven gewö"
        "hnlicher Differentialgleichungen. <i>Proceedings of the "
        "Physico-Mathematical Society of Japan</i>, "
        "24:551–559, 1942.",

        "Blanchini F. Set invariance in control. <i>Automatica</i>, "
        "35(11):1747–1767, 1999.",

        "Altman E. <i>Constrained Markov Decision Processes</i>. "
        "Chapman and Hall / CRC, 1999.",

        "Boyd S, Vandenberghe L. <i>Convex Optimization</i>. "
        "Cambridge University Press, 2004.",

        "Stellato B, Banjac G, Goulart P, Bemporad A, Boyd S. OSQP: "
        "an operator splitting solver for quadratic programs. "
        "<i>Mathematical Programming Computation</i>, "
        "12(4):637–672, 2020.",

        "Waters TR, Putz-Anderson V, Garg A, Fine LJ. Revised NIOSH "
        "equation for the design and evaluation of manual lifting "
        "tasks. <i>Ergonomics</i>, 36(7):749–776, 1993.",

        "Agarwal R, Schwarzer M, Castro PS, Courville AC, Bellemare "
        "MG. Deep reinforcement learning at the edge of the "
        "statistical precipice. <i>Advances in Neural Information "
        "Processing Systems 34</i> (NeurIPS 2021), 2021.",
    ]
    flow = [p("7. REFERENCES", CHAPTER_HEAD)]
    for i, ref in enumerate(refs, start=1):
        flow.append(p(f"[{i}] {ref}", REF_STYLE))
    return flow


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main() -> None:
    doc = make_doc()
    story: list = []
    story += cover()
    story += index_page()
    story += abstract()
    story += chapter1()
    story += chapter2()
    story += chapter3()
    story += chapter4()
    story += conclusion()
    story += references()
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
