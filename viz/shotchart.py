# viz/shotchart.py
"""
Shot chart visualization functions.
All charts use the court drawn by viz/court.py and data from data/cache.py.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from viz.court import (
    DEFAULT_SEASON,
    draw_wnba_court,
    add_court_title,
    add_watermark,
    MADE_COLOR,
    MISSED_COLOR,
    COURT_COLOR,
    ACCENT_COLOR,
)


def _filter_halfcourt(df: pd.DataFrame) -> pd.DataFrame:
    """Remove backcourt heaves and shots below baseline."""
    return df[(df["LOC_Y"] >= 0) & (df["LOC_Y"] <= 420)].copy()


def _resolve_season(df: pd.DataFrame, season) -> int | str:
    """Which court to draw under these shots.

    An explicit `season` wins. Otherwise fall back to the SEASON column that
    data.shots.prepare() stamps on every row — that is the season the shots were
    classified in, so the drawn arc matches the zones by construction. Charts
    that mix seasons have no single correct line; those must pass one.
    """
    if season is not None:
        return season
    for col in ("season", "SEASON"):
        if col in df.columns:
            vals = df[col].dropna().unique()
            if len(vals) == 1:
                return vals[0]
            if len(vals) > 1:
                raise ValueError(
                    f"shots span {len(vals)} seasons ({sorted(map(str, vals))[:4]}…) — "
                    "the three-point line moved in 2004 and 2013, so pass season= "
                    "explicitly to choose which court to draw."
                )
    return DEFAULT_SEASON


def shot_scatter(
    df: pd.DataFrame,
    title: str,
    subtitle: str = "",
    watermark: str = "@yourhandle",
    season=None,
    made_color: str = MADE_COLOR,
    missed_color: str = MISSED_COLOR,
    alpha: float = 0.65,
    size: float = 14,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Scatter shot chart — every shot as a dot, color-coded made/missed.

    `season` picks the three-point line; inferred from the frame when omitted.
    """
    season = _resolve_season(df, season)
    df = _filter_halfcourt(df)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(COURT_COLOR)
    draw_wnba_court(ax, season=season)

    made = df[df["SHOT_MADE_FLAG"] == 1]
    missed = df[df["SHOT_MADE_FLAG"] == 0]

    ax.scatter(
        missed["LOC_X"],
        missed["LOC_Y"],
        c=missed_color,
        s=size,
        alpha=alpha,
        marker="x",
        linewidths=0.9,
        zorder=4,
        label="Missed",
    )
    ax.scatter(
        made["LOC_X"],
        made["LOC_Y"],
        c=made_color,
        s=size,
        alpha=alpha,
        marker="o",
        linewidths=0,
        zorder=4,
        label="Made",
    )

    # Stats block — top left
    total = len(df)
    made_n = len(made)
    pct = made_n / total * 100 if total > 0 else 0
    ax.text(
        -255,
        430,
        f"{made_n}/{total}  |  {pct:.1f} FG%",
        color="white",
        fontsize=9,
        fontfamily="monospace",
        va="top",
    )

    # Legend
    legend_elements = [
        plt.scatter([], [], c=made_color, marker="o", s=50, label="Made"),
        plt.scatter(
            [], [], c=missed_color, marker="x", s=50, linewidths=1.2, label="Missed"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.15,
        labelcolor="white",
        fontsize=9,
        facecolor=COURT_COLOR,
        edgecolor="#444444",
    )

    add_court_title(ax, title, subtitle)
    add_watermark(ax, watermark)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COURT_COLOR)
        print(f"Saved → {save_path}")
    return fig


def shot_hexmap(
    df: pd.DataFrame,
    title: str,
    subtitle: str = "",
    watermark: str = "@yourhandle",
    season=None,
    min_shots: int = 3,
    gridsize: int = 18,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Hexbin shot chart — the Goldsberry signature.
    Hex SIZE  = volume of shots from that zone.
    Hex COLOR = efficiency (FG%) vs scale (red=cold, green=hot).

    `season` picks the three-point line; inferred from the frame when omitted.
    """
    season = _resolve_season(df, season)
    df = _filter_halfcourt(df)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(COURT_COLOR)
    draw_wnba_court(ax, season=season)

    x = df["LOC_X"].values
    y = df["LOC_Y"].values
    made = df["SHOT_MADE_FLAG"].values

    extent = [-250, 250, -47.5, 420]

    # Two invisible hexbins: one for counts, one for efficiency
    hb_count = ax.hexbin(
        x, y, gridsize=gridsize, extent=extent, mincnt=min_shots, cmap="Blues", alpha=0
    )
    hb_eff = ax.hexbin(
        x,
        y,
        C=made,
        gridsize=gridsize,
        extent=extent,
        reduce_C_function=np.mean,
        mincnt=min_shots,
        cmap="RdYlGn",
        vmin=0.25,
        vmax=0.65,
        alpha=0,
    )

    offsets = hb_count.get_offsets()
    counts = hb_count.get_array()
    efficiencies = hb_eff.get_array()

    # Normalize counts → hex radius scale
    if len(counts) > 0 and counts.max() > counts.min():
        count_norm = (counts - counts.min()) / (counts.max() - counts.min())
    else:
        count_norm = np.ones(len(counts))
    radii = 0.35 + count_norm * 1.0  # relative scale 0.35–1.35

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.25, vmax=0.65)

    # Base hex radius in coordinate units — tuned to gridsize=18
    base_radius = 500 / gridsize * 0.58

    for offset, eff, r in zip(offsets, efficiencies, radii):
        if np.isnan(eff):
            continue
        color = cmap(norm(eff))
        hex_patch = mpatches.RegularPolygon(
            (offset[0], offset[1]),
            numVertices=6,
            radius=r * base_radius,
            orientation=np.radians(30),  # pointy-top orientation
            facecolor=color,
            edgecolor=COURT_COLOR,
            linewidth=0.4,
            alpha=0.88,
            zorder=4,
        )
        ax.add_patch(hex_patch)

    hb_count.remove()
    hb_eff.remove()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.035,
        pad=0.02,
        aspect=28,
        ticks=[0.25, 0.35, 0.45, 0.55, 0.65],
    )
    cbar.set_label(
        "Field Goal %", color="white", fontsize=9, fontfamily="monospace", labelpad=4
    )
    cbar.ax.set_xticklabels(
        ["25%", "35%", "45%", "55%", "65%"],
        color="white",
        fontsize=8,
        fontfamily="monospace",
    )
    cbar.ax.xaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("#555555")

    # Stats block
    total = len(df)
    made_n = int(df["SHOT_MADE_FLAG"].sum())
    pct = made_n / total * 100 if total > 0 else 0
    ax.text(
        -255,
        430,
        f"{made_n}/{total}  |  {pct:.1f} FG%",
        color="white",
        fontsize=9,
        fontfamily="monospace",
        va="top",
    )

    add_court_title(ax, title, subtitle)
    add_watermark(ax, watermark)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COURT_COLOR)
        print(f"Saved → {save_path}")
    return fig
