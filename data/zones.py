# data/zones.py
"""
zones.py — THE WNBA shot-zone classifier. One copy, imported by everything.

Ported from nba_charts/pipeline/zones.py with one structural change: WNBA court
geometry is NOT constant. The league moved its three-point line twice, so the
arc radius has to be a function of the season rather than a module constant.
Measured directly off shotchartdetail:

    1997-2003   ~19.7 ft   19' 9"        arc meets the baseline, no true corner
    2004-2012   ~20.5 ft   20' 6.25"     still effectively a pure circle
    2013-       ~22.1 ft   22' 1.75"     22 ft straight corners (FIBA distance)

Classifying a 1998 shot against the modern 22'1.75" arc puts the line ~2.4 ft
beyond where those threes were actually taken, which turns real threes into
mid-range attempts. Always resolve geometry from the season.

ZONE NAMES ARE STABLE ACROSS ERAS. The boundaries move; the labels do not, so
"3PT - Left Center" means the same basketball thing in 1998 and 2024 and
cross-era comparisons stay meaningful.

COORDINATES (nba_api shotchartdetail)
-------------------------------------
    LOC_X : lateral, tenths of a foot, hoop at 0, positive = viewer's right
    LOC_Y : toward halfcourt, tenths of a foot, hoop at 0 (slightly negative
            behind the baseline)

14-wedge scheme: a restricted-area disk, then concentric rings (close mid /
mid / three) sliced into angular wedges.

    rings   RA < 4ft - close 4-11ft - mid 11ft-arc - three beyond arc
    angle   0deg = right baseline, 90deg = straight out, 180deg = left baseline
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Court constants in tenths of a foot, matching LOC_X / LOC_Y directly.
RA_R = 40.0        # restricted area radius (4 ft)
CLOSE_R = 110.0    # close-mid outer radius (11 ft)
HEAVE_R = 400.0    # 40 ft+ — excluded as heaves, not real shot selection

ZONE_NAMES = [
    "Restricted Area",                                              # 0
    "Close Mid - Right", "Close Mid - Center", "Close Mid - Left",  # 1-3
    "Mid - Right", "Mid - Right Center", "Mid - Center",            # 4-6
    "Mid - Left Center", "Mid - Left",                              # 7-8
    "3PT - Right", "3PT - Right Center", "3PT - Center",            # 9-11
    "3PT - Left Center", "3PT - Left",                              # 12-13
]
ZONE_INDEX = {name: i for i, name in enumerate(ZONE_NAMES)}
THREE_POINT_ZONES = {"3PT - Right", "3PT - Right Center", "3PT - Center",
                     "3PT - Left Center", "3PT - Left"}


@dataclass(frozen=True)
class CourtGeometry:
    """Three-point geometry for one era, in tenths of a foot."""
    arc_r: float          # radius of the above-the-break arc
    corner_x: float       # |x| of the straight corner segment
    first_season: int
    label: str

    @property
    def y_meet(self) -> float:
        """Height where the corner line meets the arc.

        When the corner sits at (or outside) the arc radius there is no straight
        segment at all — the line is a pure circle — so the junction collapses
        to the baseline and the corner test never fires on its own.
        """
        if self.corner_x >= self.arc_r:
            return 0.0
        return float(np.sqrt(self.arc_r ** 2 - self.corner_x ** 2))

    @property
    def has_corners(self) -> bool:
        return self.corner_x < self.arc_r


# Ordered newest-last. Distances are the league's published rule values;
# `estimate_arc` below re-derives them from data to check this table.
ERAS = [
    CourtGeometry(arc_r=197.5, corner_x=197.5, first_season=1997,
                  label="19'9\""),
    CourtGeometry(arc_r=205.2, corner_x=205.2, first_season=2004,
                  label="20'6.25\""),
    CourtGeometry(arc_r=221.5, corner_x=220.0, first_season=2013,
                  label="22'1.75\""),
]


def geometry_for_season(season) -> CourtGeometry:
    """Court geometry in force for a given season (int or str year)."""
    year = int(str(season)[:4])
    chosen = ERAS[0]
    for era in ERAS:
        if year >= era.first_season:
            chosen = era
    return chosen


def classify(loc_x, loc_y, geom: CourtGeometry) -> np.ndarray:
    """
    Vectorised: arrays of LOC_X/LOC_Y -> array of zone-name strings.

    Shots that are heaves or otherwise unclassifiable come back as "Heave" /
    "Unknown" so callers drop them explicitly rather than silently binning a
    60-footer into the mid-range.
    """
    x = np.asarray(loc_x, dtype=float)
    y = np.asarray(loc_y, dtype=float)

    dist = np.sqrt(x ** 2 + y ** 2)
    angle = np.degrees(np.arctan2(y, x))
    # Behind-the-baseline shots come back negative; unwrap so the left corner
    # stays near 180 instead of jumping to -180.
    angle = np.where(angle < -90, angle + 360, angle)

    is_heave = dist >= HEAVE_R
    is_rim = ~is_heave & (dist < RA_R)
    beyond_arc = dist >= geom.arc_r
    if geom.has_corners:
        beyond_arc = beyond_arc | ((np.abs(x) >= geom.corner_x) & (y <= geom.y_meet))
    is_three = ~is_heave & ~is_rim & beyond_arc
    is_close = ~is_heave & ~is_rim & ~is_three & (dist < CLOSE_R)
    is_mid = ~is_heave & ~is_rim & ~is_three & ~is_close

    out = np.full(x.shape, "Unknown", dtype=object)
    out[is_heave] = "Heave"
    out[is_rim] = "Restricted Area"

    out[is_close & (angle < 60)] = "Close Mid - Right"
    out[is_close & (angle >= 60) & (angle < 120)] = "Close Mid - Center"
    out[is_close & (angle >= 120)] = "Close Mid - Left"

    out[is_mid & (angle < 36)] = "Mid - Right"
    out[is_mid & (angle >= 36) & (angle < 72)] = "Mid - Right Center"
    out[is_mid & (angle >= 72) & (angle < 108)] = "Mid - Center"
    out[is_mid & (angle >= 108) & (angle < 144)] = "Mid - Left Center"
    out[is_mid & (angle >= 144)] = "Mid - Left"

    out[is_three & (angle < 36)] = "3PT - Right"
    out[is_three & (angle >= 36) & (angle < 72)] = "3PT - Right Center"
    out[is_three & (angle >= 72) & (angle < 108)] = "3PT - Center"
    out[is_three & (angle >= 108) & (angle < 144)] = "3PT - Left Center"
    out[is_three & (angle >= 144)] = "3PT - Left"
    return out


def zone_records(made, zones) -> list[list[int]]:
    """[[zone_index, makes, attempts], ...] — compact form (zone names live once
    in ZONE_NAMES rather than being repeated on every row)."""
    made = np.asarray(made, dtype=float)
    zones = np.asarray(zones, dtype=object)
    out = []
    for name, idx in ZONE_INDEX.items():
        m = zones == name
        att = int(m.sum())
        if att:
            out.append([idx, int(made[m].sum()), att])
    return out


def baselines(made, zones) -> list[float | None]:
    """League-average FG% per zone index — drives relative-to-average colours.

    Baselines must be computed WITHIN a season (or at least within an era):
    a 19'9" three and a 22'1.75" three are different shots, so pooling them
    produces a baseline that describes neither.
    """
    made = np.asarray(made, dtype=float)
    zones = np.asarray(zones, dtype=object)
    out = []
    for name in ZONE_NAMES:
        m = zones == name
        att = int(m.sum())
        out.append(round(float(made[m].sum()) / att, 4) if att else None)
    return out


# ── Calibration ───────────────────────────────────────────────────────────────

def estimate_arc(shots: pd.DataFrame) -> float | None:
    """
    Estimate the three-point arc radius (tenths of a foot) from shot data.

    Uses the boundary between labelled 2s and 3s rather than a raw minimum,
    which a single mis-coordinated row would wreck: the arc sits between the
    farthest above-the-break two and the nearest above-the-break three, so the
    5th percentile of three-point radius is a stable proxy. Lets `ERAS` be
    checked against reality instead of trusted.
    """
    df = shots.dropna(subset=["LOC_X", "LOC_Y", "SHOT_TYPE"])
    if df.empty:
        return None
    is_three = df["SHOT_TYPE"].astype(str).str.startswith("3PT")
    # Above the break only — corner threes are nearer the hoop by design.
    above = df["LOC_Y"].astype(float) > 100.0
    sel = df[is_three & above]
    if len(sel) < 50:
        return None
    r = np.sqrt(sel["LOC_X"].astype(float) ** 2 + sel["LOC_Y"].astype(float) ** 2)
    return float(np.percentile(r, 1))


def check_geometry(shots: pd.DataFrame, season, tol: float = 12.0) -> dict:
    """Compare the ERAS table against what the season's shots actually show."""
    geom = geometry_for_season(season)
    measured = estimate_arc(shots)
    ok = measured is not None and abs(measured - geom.arc_r) <= tol
    return {"season": str(season), "era": geom.label, "table_arc": geom.arc_r,
            "measured_arc": None if measured is None else round(measured, 1),
            "ok": ok}


# ── Browser mirror invariant ──────────────────────────────────────────────────

def label_position(name: str, geom: CourtGeometry) -> tuple[float, float]:
    """
    Where the FG% label for a zone is drawn.

    MUST stay identical to ZoneChart.labelFor() in site/js/zone-chart.js. The
    NBA original hardcodes 14 label positions with a warning that each must
    classify into its own zone; with three arcs that is unmaintainable by hand,
    so both sides derive the position from the geometry and `verify_labels`
    below asserts the invariant.
    """
    def at(r: float, a: float) -> tuple[float, float]:
        rad = np.radians(a)
        return (r * np.cos(rad), r * np.sin(rad))

    ring_mid = RA_R + (CLOSE_R - RA_R) * 0.5
    mid_mid = CLOSE_R + (geom.arc_r - CLOSE_R) * 0.5
    corner_x = (geom.corner_x + 18) if geom.has_corners else (geom.arc_r + 14)

    return {
        "Restricted Area":     (0.0, 5.0),
        "Close Mid - Right":   at(ring_mid, 30),
        "Close Mid - Center":  at(ring_mid, 90),
        "Close Mid - Left":    at(ring_mid, 150),
        "Mid - Right":         at(mid_mid, 18),
        "Mid - Right Center":  at(mid_mid, 54),
        "Mid - Center":        at(mid_mid, 90),
        "Mid - Left Center":   at(mid_mid, 126),
        "Mid - Left":          at(mid_mid, 162),
        "3PT - Right":         (corner_x, 6.0),
        "3PT - Right Center":  at(geom.arc_r + 26, 54),
        "3PT - Center":        at(geom.arc_r + 30, 90),
        "3PT - Left Center":   at(geom.arc_r + 26, 126),
        "3PT - Left":          (-corner_x, 6.0),
    }[name]


def verify_labels(geom: CourtGeometry | None = None) -> list[dict]:
    """Every zone label must classify into the zone it names. Returns offenders."""
    bad = []
    for era in ([geom] if geom else ERAS):
        for name in ZONE_NAMES:
            x, y = label_position(name, era)
            got = classify([x], [y], era)[0]
            if got != name:
                bad.append({"era": era.label, "zone": name, "got": got,
                            "at": (round(float(x)), round(float(y)))})
    return bad
