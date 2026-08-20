# viz/court.py
"""
WNBA half-court drawing for matplotlib.

Coordinate system matches stats.wnba.com shot data (all values in tenths of feet):
  - Basket center at (0, 0)
  - Y increases away from basket toward half court
  - Baseline at Y = -47
  - Key: 16ft wide (±80), 19ft long, FT line at Y = 137.5
  - FT circle radius: 60 (6ft)
  - Restricted area radius: 40 (4ft)

THE THREE-POINT LINE IS NOT A CONSTANT. The WNBA moved it in 2004 and again in
2013, so the arc is resolved per season from data/zones.py — the same table the
shot classifier uses, so the drawn line and the zone boundaries can never drift
apart:

    1997-2003   19'9"       pure circle, no straight corner
    2004-2012   20'6.25"    pure circle, no straight corner
    2013-       22'1.75"    22ft straight corners

Everything else on the court (lane, FT circle, restricted area) is era-invariant.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle, Rectangle
from matplotlib.lines import Line2D

from data.zones import CourtGeometry, geometry_for_season

# ── Brand colors ───────────────────────────────────────────────────────────
COURT_COLOR = "#1a1a2e"
LINE_COLOR = "#cccccc"
LINE_WIDTH = 1.5
MADE_COLOR = "#e8c547"
MISSED_COLOR = "#e05c5c"
ACCENT_COLOR = "#4fc3f7"

# ── Confirmed dimensions (tenths of feet) ──────────────────────────────────
BASELINE_Y = -47
BACKBOARD_Y = -40  # 4ft behind basket center
KEY_HALF_W = 80  # key is 16ft wide total
FT_Y = 137.5  # 13ft 9in from basket center
FT_RADIUS = 60  # 6ft radius
RA_RADIUS = 40  # 4ft restricted area
SIDELINE_X = 250

# Season used when a caller does not name one. Drawing the modern arc over a
# 1998 chart is the exact bug this module's season parameter exists to prevent,
# so callers that know the season should always pass it.
DEFAULT_SEASON = 2013


def three_point_theta(geom: CourtGeometry) -> float:
    """Angle (degrees) at which the arc stops, measured from the right baseline.

    With straight corners (2013+) the arc stops where it meets the corner line.
    Without them the arc runs all the way down into the baseline, which is BELOW
    the hoop's Y — hence a negative angle, not zero.
    """
    if geom.has_corners:
        return float(np.degrees(np.arctan2(geom.y_meet, geom.corner_x)))
    return float(-np.degrees(np.arcsin(abs(BASELINE_Y) / geom.arc_r)))


def draw_wnba_court(
    ax=None,
    season=DEFAULT_SEASON,
    color=LINE_COLOR,
    lw=LINE_WIDTH,
    court_color=COURT_COLOR,
):
    """
    Draw a WNBA half-court for `season`. All lines verified against official
    dimensions and cross-checked against stats.wnba.com shot coordinate data;
    the three-point line comes from the season's era in data/zones.py.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 9))

    geom = geometry_for_season(season)

    ax.set_facecolor(court_color)

    # ── Basket ─────────────────────────────────────────────────────────────
    basket = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False, zorder=5)
    ax.add_patch(basket)

    # ── Backboard ──────────────────────────────────────────────────────────
    ax.add_line(
        Line2D(
            [-30, 30],
            [BACKBOARD_Y, BACKBOARD_Y],
            linewidth=lw * 2,
            color=color,
            zorder=5,
        )
    )

    # ── Restricted area (4ft radius, half-circle above basket) ────────────
    ax.add_patch(
        Arc(
            (0, 0),
            width=RA_RADIUS * 2,
            height=RA_RADIUS * 2,
            angle=0,
            theta1=0,
            theta2=180,
            linewidth=lw,
            color=color,
            zorder=5,
        )
    )

    # ── Key (paint) — 16ft wide, from baseline to FT line ─────────────────
    ax.add_patch(
        Rectangle(
            (-KEY_HALF_W, BASELINE_Y),
            KEY_HALF_W * 2,
            FT_Y - BASELINE_Y,
            linewidth=lw,
            color=color,
            fill=False,
            zorder=5,
        )
    )

    # ── Free throw circle ──────────────────────────────────────────────────
    # Solid top half, dashed bottom half — center at FT line
    ax.add_patch(
        Arc(
            (0, FT_Y),
            width=FT_RADIUS * 2,
            height=FT_RADIUS * 2,
            angle=0,
            theta1=0,
            theta2=180,
            linewidth=lw,
            color=color,
            zorder=5,
        )
    )
    ax.add_patch(
        Arc(
            (0, FT_Y),
            width=FT_RADIUS * 2,
            height=FT_RADIUS * 2,
            angle=0,
            theta1=180,
            theta2=360,
            linewidth=lw,
            color=color,
            linestyle="dashed",
            zorder=5,
        )
    )

    # ── Three-point line (era-dependent) ───────────────────────────────────
    theta = three_point_theta(geom)

    # Straight corner segments exist only from 2013 on. Before that the line is
    # a pure circle and drawing corners would invent geometry that never existed.
    if geom.has_corners:
        for sx in (-1, 1):
            ax.add_line(
                Line2D(
                    [sx * geom.corner_x, sx * geom.corner_x],
                    [BASELINE_Y, geom.y_meet],
                    linewidth=lw,
                    color=color,
                    zorder=5,
                )
            )

    ax.add_patch(
        Arc(
            (0, 0),
            width=geom.arc_r * 2,
            height=geom.arc_r * 2,
            angle=0,
            theta1=theta,  # right end: corner junction, or into the baseline
            theta2=180 - theta,  # left end, mirrored
            linewidth=lw,
            color=color,
            zorder=5,
        )
    )

    # ── Baseline ───────────────────────────────────────────────────────────
    ax.add_line(
        Line2D(
            [-SIDELINE_X, SIDELINE_X],
            [BASELINE_Y, BASELINE_Y],
            linewidth=lw * 2,
            color=color,
            zorder=5,
        )
    )

    # ── Sidelines ──────────────────────────────────────────────────────────
    ax.add_line(
        Line2D(
            [-SIDELINE_X, -SIDELINE_X],
            [BASELINE_Y, 422],
            linewidth=lw,
            color=color,
            zorder=5,
        )
    )
    ax.add_line(
        Line2D(
            [SIDELINE_X, SIDELINE_X],
            [BASELINE_Y, 422],
            linewidth=lw,
            color=color,
            zorder=5,
        )
    )

    # ── Axis settings ──────────────────────────────────────────────────────
    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 430)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def add_court_title(
    ax, title, subtitle=None, title_color="white", subtitle_color="#aaaaaa"
):
    ax.text(
        0,
        448,
        title,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color=title_color,
        fontfamily="monospace",
    )
    if subtitle:
        ax.text(
            0,
            436,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=9,
            color=subtitle_color,
            fontfamily="monospace",
        )


def add_watermark(ax, text="@yourhandle", color="#555555"):
    ax.text(
        0.99,
        0.01,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=color,
        fontfamily="monospace",
        alpha=0.8,
    )
