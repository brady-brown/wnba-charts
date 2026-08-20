"""
WNBA Player Archetypes — 2022-2025
Uses HDBSCAN (density-based, natural cluster shapes) on a combination of
per-100-possession box stats + advanced rate stats.
No height/weight. Min-minutes filter applied per season.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import euclidean_distances
from data.cache import get_player_stats

# ── config ─────────────────────────────────────────────────────────────────────
SEASONS = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
MIN_MINUTES_PG = 12   # per-game minutes (from PerGame df)
MIN_GP = 10

# Per-100 box stats kept for style signals only (not volume/performance)
BOX_FEATURES_100 = [
    "FG3A",   # 3PT attempt rate — how a player scores
    "BLK",    # rim protection role
    "STL",    # disruption / guard-vs-big signal
]

# Advanced rate stats — all role/style descriptors, not outcome quality
ADV_FEATURES = [
    "USG_PCT",    # share of offense running through player
    "AST_PCT",    # playmaking role
    "OREB_PCT",   # offensive rebounding role
    "DREB_PCT",   # defensive rebounding role
    "TS_PCT",     # shooting profile (3PT-heavy vs. paint vs. FT-reliant)
    "E_TOV_PCT",  # turnover tendency (ball-handlers vs. finishers)
    "AST_TO",     # playmaking decisiveness
]

# ── data loading ───────────────────────────────────────────────────────────────

def load_season(season: str) -> pd.DataFrame:
    pg_base, p100_base = get_player_stats(season, measure_type="Base")
    pg_adv, _p100_adv = get_player_stats(season, measure_type="Advanced")

    # per-game MIN for filtering, plus GP
    pg_filter = pg_base[["PLAYER_ID", "MIN", "GP"]].rename(columns={"MIN": "MIN_PG"})

    # per-100 box stats (count stats normalized to 100 possessions)
    box_cols = [c for c in BOX_FEATURES_100 if c in p100_base.columns]
    box = p100_base[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"] + box_cols].copy()

    # advanced rate stats from per-game (rates are pace-neutral already)
    adv_cols = [c for c in ADV_FEATURES if c in pg_adv.columns]
    adv = pg_adv[["PLAYER_ID"] + adv_cols].copy()

    df = box.merge(pg_filter, on="PLAYER_ID", how="left")
    df = df.merge(adv, on="PLAYER_ID", how="left")
    df["SEASON"] = season
    return df


def load_all() -> pd.DataFrame:
    frames = []
    for s in SEASONS:
        print(f"  loading {s}...")
        frames.append(load_season(s))
    return pd.concat(frames, ignore_index=True)


# ── feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame):
    all_features = [c for c in BOX_FEATURES_100 + ADV_FEATURES if c in df.columns]

    # Normalize within each season so clustering reflects role within era,
    # not era-level trends (e.g., 3PT volume has risen sharply since 2016).
    normed = df[all_features].copy().astype(float)
    for feat in all_features:
        for season, grp in df.groupby("SEASON"):
            idx = grp.index
            col = normed.loc[idx, feat]
            med = col.median()
            std = col.std()
            normed.loc[idx, feat] = (col - med) / std if std > 0 else 0.0

    imp = SimpleImputer(strategy="constant", fill_value=0.0)
    X = imp.fit_transform(normed.values)
    return X, all_features


# ── clustering ─────────────────────────────────────────────────────────────────

def run_hdbscan(X: np.ndarray, min_cluster_size: int = 12,
                cluster_selection_epsilon: float = 0.0) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        cluster_selection_method="eom",
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
    )
    return clusterer.fit_predict(X)


# ── soft archetype assignment ──────────────────────────────────────────────────

SOFT_TEMP = 0.3   # lower = sharper; 0.3 matches the NBA reference script

def compute_soft_probs(X: np.ndarray, labels: np.ndarray,
                       cluster_ids: list) -> np.ndarray:
    """Softmax over euclidean distances to each cluster centroid (in feature space)."""
    centroids = np.array([X[labels == c].mean(axis=0) for c in cluster_ids])
    dists = euclidean_distances(X, centroids)           # (n_players, n_clusters)
    logits = -dists / SOFT_TEMP
    logits -= logits.max(axis=1, keepdims=True)         # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs                                        # (n_players, n_clusters)


def fmt_archetype_str(probs_row: np.ndarray, cluster_ids: list,
                      name_map: dict, threshold: float = 0.20) -> str:
    """Return 'Primary (pct) / Secondary (pct)' string for a player."""
    ranked = sorted(enumerate(probs_row), key=lambda x: -x[1])
    parts = [ranked[0]]
    for idx, val in ranked[1:]:
        if val >= threshold:
            parts.append((idx, val))
        if len(parts) == 3:
            break
    return " / ".join(
        f"{name_map.get(cluster_ids[i], f'Cluster{cluster_ids[i]}')} ({v:.0%})"
        for i, v in parts
    )


# ── archetype naming ───────────────────────────────────────────────────────────

def describe_cluster(df: pd.DataFrame, label: int, all_features: list,
                     global_med: pd.Series, global_std: pd.Series) -> dict:
    sub = df[df["cluster"] == label]
    n = len(sub)
    med = sub[all_features].median()
    z = (med - global_med) / global_std   # how this cluster compares to league

    stats = {col: med.get(col, np.nan) for col in all_features}
    stats["n"] = n
    stats["min_pg"] = sub["MIN_PG"].median()

    name = _auto_name_from_z(stats, z)
    # rank members by USG_PCT * TS_PCT as a neutral proxy for impact without pure volume
    impact = (sub.groupby("PLAYER_NAME")
                 .apply(lambda g: (g["USG_PCT"] * g["TS_PCT"]).mean())
                 .sort_values(ascending=False))
    top_players = impact.head(8).index.tolist()

    return {"name": name, "stats": stats, "n": n, "z": z, "members": top_players}


def _z(z: pd.Series, col: str) -> float:
    return z.get(col, 0.0)


def _auto_name_from_z(stats: dict, z: pd.Series) -> str:
    """
    Name a cluster by its z-score profile vs league median.
    Only uses style/role features — no PTS, REB, AST volume.
    Positive z = above average, negative z = below average.
    """
    z_usg      = _z(z, "USG_PCT")
    z_ast_pct  = _z(z, "AST_PCT")
    z_oreb_pct = _z(z, "OREB_PCT")
    z_dreb_pct = _z(z, "DREB_PCT")
    z_ts       = _z(z, "TS_PCT")
    z_tov_pct  = _z(z, "E_TOV_PCT")
    z_ast_to   = _z(z, "AST_TO")
    z_fg3a     = _z(z, "FG3A")
    z_blk      = _z(z, "BLK")
    z_stl      = _z(z, "STL")

    # ─── Frontcourt / Bigs ─────────────────────────────────────────────────────
    # Cluster 0: A'ja Wilson, Griner — BLK≥2.5, USG≥1.5, dominant rebounder
    if z_blk > 2.0 and z_usg > 1.4:
        return "Dominant Two-Way Big"

    # Cluster 5: Breanna Stewart, Elena Delle Donne, Napheesa — versatile big, efficient
    if z_usg > 0.7 and z_blk > 0.8 and z_oreb_pct > 0.6 and z_ts > 0.5:
        return "Versatile Two-Way Forward"

    # Cluster 6: Sylvia Fowles, Teaira McCowan — classic interior big, massive BLK+OREB
    if z_blk > 1.4 and z_oreb_pct > 1.5:
        return "Traditional Rim Protector"

    # Cluster 11: Tina Charles, Satou Sabally — high usage big, below-avg efficiency
    if z_usg > 0.9 and z_dreb_pct > 1.0 and z_ts < 0:
        return "High-Usage Frontcourt"

    # Cluster 4: glass cleaner — dominant rebounding, very low 3PT, low usage
    if z_oreb_pct > 1.0 and z_fg3a < -1.0 and z_usg < 0:
        return "Glass Cleaner / Rebounder"

    # Cluster 2: turnover-prone big/forward — elevated turnovers, weak AST/TO, some BLK
    if z_tov_pct > 1.0 and z_ast_to < -0.8 and z_dreb_pct > 0.2:
        return "Turnover-Prone Frontcourt"

    # General big fallback
    if z_oreb_pct > 0.7 or z_dreb_pct > 0.9 or z_blk > 0.8:
        if z_fg3a < 0:
            return "Interior Frontcourt"
        return "Frontcourt Contributor"

    # ─── Playmakers / Guards ─────────────────────────────────────────────────
    # Cluster 3: Chelsea Gray, Courtney Vandersloot, Sue Bird — elite AST_PCT + AST/TO
    if z_ast_pct > 1.0 and z_ast_to > 0.8:
        return "Playmaking Guard"

    if z_ast_pct > 0.8 and z_tov_pct > 0.6:
        return "Combo Guard / Facilitator"

    # ─── Wings / Perimeter ───────────────────────────────────────────────────
    # Cluster 1: Taurasi, Loyd, Sabrina, Plum — high 3PT vol + high usage + low TOV
    if z_fg3a > 0.9 and z_usg > 0.5 and z_tov_pct < 0:
        return "High-Volume Guard / Shot Creator"

    # Cluster 12: Kahleah Copper, DeWanna Bonner, Satou — high usage wing, some 3PT
    if z_usg > 0.7 and z_oreb_pct > 0 and z_fg3a > 0.4:
        return "Wing Primary Option"

    if z_usg > 0.7 and z_fg3a < 0:
        return "High-Usage Athletic Wing"

    # Cluster 7: Brittney Sykes, Rae Burrell — disruptive defensive wing, inefficient
    if z_stl > 0.7 and z_ts < -0.4:
        return "Defensive Perimeter Wing"

    # Cluster 9: Kaleena Mosqueda-Lewis, Kia Nurse — 3PT volume but very inefficient
    if z_fg3a > 0.6 and z_ts < -1.0:
        return "3PT Volume Shooter"

    # Cluster 8: Alysha Clark, Leonie Fiebich — very low usage, efficient glue
    if z_usg < -0.8 and z_ts > 0:
        return "Efficient Role Player"

    # Cluster 10: Maddy Siegrist, Morgan Tuck — moderate everything, below avg
    if z_stl < -0.5 and z_ts < -0.2:
        return "Secondary Wing / Developing"

    if z_usg < -0.4:
        return "Role Wing"

    return "Multi-Role Contributor"


# ── UMAP reduction ─────────────────────────────────────────────────────────────

def run_umap(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    reducer = umap.UMAP(n_components=n_components, n_neighbors=15,
                        min_dist=0.1, random_state=random_state)
    return reducer.fit_transform(X)


# ── plotting ───────────────────────────────────────────────────────────────────

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
]

DARK_BG = "#0d0d0d"
PANEL_BG = "#111111"


def _color_map(labels):
    unique = sorted(set(labels))
    cmap = {}
    pi = 0
    for lbl in unique:
        if lbl == -1:
            cmap[lbl] = "#555555"
        else:
            cmap[lbl] = PALETTE[pi % len(PALETTE)]
            pi += 1
    return cmap


def plot_umap_overview(df: pd.DataFrame, embedding: np.ndarray,
                       archetype_map: dict, out: str):
    color_map = _color_map(df["cluster"].values)

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # noise first (behind)
    mask_noise = df["cluster"] == -1
    ax.scatter(embedding[mask_noise, 0], embedding[mask_noise, 1],
               c="#333333", alpha=0.3, s=12, linewidths=0, zorder=1)

    # clusters
    for lbl in sorted(c for c in set(df["cluster"]) if c != -1):
        mask = df["cluster"] == lbl
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color_map[lbl], alpha=0.7, s=22, linewidths=0, zorder=2)

    # centroid labels
    for lbl, info in archetype_map.items():
        if lbl == -1:
            continue
        mask = df["cluster"] == lbl
        cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
        ax.text(cx, cy, info["name"], fontsize=7.5, color=color_map[lbl],
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=DARK_BG,
                          alpha=0.75, edgecolor="none"), zorder=3)

    ax.set_title(f"WNBA Player Archetypes — {SEASONS[0]}-{SEASONS[-1]} (HDBSCAN + UMAP)",
                 color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_xlabel("UMAP 1", color="#888888")
    ax.set_ylabel("UMAP 2", color="#888888")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  saved: {out}")


def plot_season_scatter(df: pd.DataFrame, embedding: np.ndarray,
                        archetype_map: dict, out: str):
    color_map = _color_map(df["cluster"].values)
    ncols = 5
    nrows = (len(SEASONS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    fig.patch.set_facecolor(DARK_BG)

    for ax, season in zip(axes.flat, SEASONS):
        ax.set_facecolor(PANEL_BG)
        mask_s = df["SEASON"] == season
        # dim background
        ax.scatter(embedding[~mask_s, 0], embedding[~mask_s, 1],
                   c="#2a2a2a", alpha=0.4, s=8, linewidths=0)
        # highlighted season
        lbl_arr = df.loc[mask_s, "cluster"].values
        colors = [color_map[l] for l in lbl_arr]
        ax.scatter(embedding[mask_s, 0], embedding[mask_s, 1],
                   c=colors, alpha=0.85, s=28, linewidths=0)
        ax.set_title(f"{season}", color="white", fontsize=13, pad=6)
        ax.tick_params(colors="#444444", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    fig.suptitle(f"WNBA Archetypes by Season — {SEASONS[0]}-{SEASONS[-1]} (same UMAP embedding)",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  saved: {out}")


def plot_archetype_profiles(df: pd.DataFrame, archetype_map: dict,
                             all_features: list, out: str):
    display_stats = ["FG3A", "BLK", "STL",
                     "USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT",
                     "TS_PCT", "E_TOV_PCT", "AST_TO"]
    display_stats = [c for c in display_stats if c in df.columns]

    valid_clusters = sorted(lbl for lbl in archetype_map if lbl != -1)
    n = len(valid_clusters)
    if n == 0:
        return

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
    fig.patch.set_facecolor(DARK_BG)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    color_map = _color_map([lbl for lbl in archetype_map])
    global_med = df[display_stats].median()
    global_std = df[display_stats].std().replace(0, 1)

    for ax, lbl in zip(axes_flat, valid_clusters):
        ax.set_facecolor(PANEL_BG)
        info = archetype_map[lbl]
        sub = df[df["cluster"] == lbl]
        med = sub[display_stats].median()
        z = (med - global_med) / global_std

        color = color_map[lbl]
        bar_colors = [color if v >= 0 else "#888888" for v in z.values]
        ax.barh(display_stats, z.values, color=bar_colors, alpha=0.8)
        ax.axvline(0, color="#555555", lw=0.8, linestyle="--")
        ax.set_title(f'{info["name"]}\n(n={info["n"]})',
                     color="white", fontsize=8.5, pad=4)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_xlabel("z-score vs league", color="#888888", fontsize=7)
        # annotate actual medians
        for i, (stat, val) in enumerate(med.items()):
            ax.text(z.values[i] + (0.05 if z.values[i] >= 0 else -0.05),
                    i, f"{val:.2f}", va="center", fontsize=5.5,
                    color="white", ha="left" if z.values[i] >= 0 else "right")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Archetype Stat Profiles (z-score vs league median)",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  saved: {out}")


def plot_player_trajectory(df: pd.DataFrame, archetype_map: dict,
                            out: str, top_n: int = 35):
    counts = df.groupby("PLAYER_NAME")["SEASON"].nunique()
    multi = counts[counts >= 3].index.tolist()

    impact_rank = (df[df["PLAYER_NAME"].isin(multi)]
                   .groupby("PLAYER_NAME")
                   .apply(lambda g: (g["USG_PCT"] * g["TS_PCT"]).mean())
                   .sort_values(ascending=False))
    players = impact_rank.head(top_n).index.tolist()

    name_map = {lbl: info["name"] for lbl, info in archetype_map.items()}
    name_map[-1] = "Unclassified"

    all_archetypes = sorted(set(name_map.values()))
    y_map = {a: i for i, a in enumerate(all_archetypes)}

    fig, ax = plt.subplots(figsize=(14, max(10, top_n * 0.45)))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    cmap = plt.cm.tab20
    player_colors = {p: cmap(i / max(len(players), 1)) for i, p in enumerate(players)}

    for player in players:
        sub = df[df["PLAYER_NAME"] == player].sort_values("SEASON")
        xs = sub["SEASON"].tolist()
        ys = [y_map.get(name_map.get(c, "Unclassified"), 0) for c in sub["cluster"].tolist()]
        color = player_colors[player]
        ax.plot(xs, ys, "-o", color=color, alpha=0.75, markersize=5, lw=1.5)
        ax.text(xs[-1], ys[-1], f"  {player.split()[-1]}", fontsize=6,
                color=color, va="center")

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()), color="white", fontsize=7)
    ax.set_xticks(SEASONS)
    ax.tick_params(axis="x", colors="white", labelsize=9)
    ax.set_xlabel("Season", color="#888888")
    ax.set_title(f"Archetype Trajectory — Top {top_n} Players (3+ seasons, by PIE)",
                 color="white", fontsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(axis="y", color="#222222", lw=0.5)
    ax.grid(axis="x", color="#222222", lw=0.5)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  saved: {out}")


def plot_heatmap(df: pd.DataFrame, archetype_map: dict, all_features: list, out: str):
    """Heatmap of z-scores: archetypes × key stats."""
    display_stats = ["FG3A", "BLK", "STL",
                     "USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT",
                     "TS_PCT", "E_TOV_PCT", "AST_TO"]
    display_stats = [c for c in display_stats if c in df.columns]

    valid_clusters = sorted(lbl for lbl in archetype_map if lbl != -1)
    global_med = df[display_stats].median()
    global_std = df[display_stats].std().replace(0, 1)

    rows = []
    row_labels = []
    for lbl in valid_clusters:
        sub = df[df["cluster"] == lbl]
        med = sub[display_stats].median()
        z = (med - global_med) / global_std
        rows.append(z.values)
        row_labels.append(archetype_map[lbl]["name"])

    Z = np.array(rows)

    fig, ax = plt.subplots(figsize=(16, max(6, len(valid_clusters) * 0.7)))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    im = ax.imshow(Z, cmap="RdYlGn", aspect="auto", vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(len(display_stats)))
    ax.set_xticklabels(display_stats, rotation=45, ha="right", color="white", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color="white", fontsize=8)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            text_color = "black" if abs(val) < 1.5 else "white"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=6.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("z-score vs league", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Archetype Stat Heatmap (z-scores vs league median)",
                 color="white", fontsize=13, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_all()

    # filter on per-game minutes
    df = df[(df["MIN_PG"] >= MIN_MINUTES_PG) & (df["GP"] >= MIN_GP)].copy()
    df = df.reset_index(drop=True)
    print(f"  {len(df)} player-seasons after filter (MIN_PG>={MIN_MINUTES_PG}, GP>={MIN_GP})")

    print("Building feature matrix...")
    X, all_features = build_feature_matrix(df)

    # 8-d UMAP (nn=20) + min_cluster_size=25 → ~13 natural clusters at ~14% noise
    print("UMAP 8-d reduction for clustering...")
    X_clust = umap.UMAP(n_components=8, n_neighbors=15, min_dist=0.0,
                         random_state=42).fit_transform(X)

    min_cs = 20
    print(f"Running HDBSCAN (min_cluster_size={min_cs})...")
    labels = run_hdbscan(X_clust, min_cluster_size=min_cs)
    df["cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  {n_clusters} clusters found, {n_noise} noise ({n_noise/len(labels)*100:.1f}%)")

    print("Naming archetypes...")
    feat_cols = [c for c in all_features if c in df.columns]
    global_med = df[feat_cols].median()
    global_std = df[feat_cols].std().replace(0, 1)

    archetype_map = {}
    for lbl in sorted(set(labels)):
        archetype_map[lbl] = describe_cluster(df, lbl, feat_cols, global_med, global_std)

    name_map = {lbl: info["name"] for lbl, info in archetype_map.items()}

    # Soft assignments — computed in original feature space against named clusters only
    cluster_ids = sorted(lbl for lbl in archetype_map if lbl != -1)
    probs = compute_soft_probs(X, labels, cluster_ids)   # (n_players, n_clusters)

    # Primary archetype: argmax of soft probs (rescues noise points from "Unclassified")
    primary_idx = probs.argmax(axis=1)
    df["archetype"] = [name_map[cluster_ids[i]] for i in primary_idx]

    # For hard-clustered players keep their assigned label; noise gets soft label + marker
    hard_mask = labels != -1
    df.loc[hard_mask, "archetype"] = df.loc[hard_mask, "cluster"].map(name_map)

    df["archetype_str"] = [
        fmt_archetype_str(probs[i], cluster_ids, name_map)
        for i in range(len(df))
    ]

    # ── summary ────────────────────────────────────────────────────────────────
    print("\n── Archetypes ──────────────────────────────────────────────────────")
    for lbl, info in sorted(archetype_map.items(), key=lambda x: -x[1]["n"]):
        s = info["stats"]
        label_str = "NOISE" if lbl == -1 else f"#{lbl:2d}"
        print(f"  {label_str}  {info['name']:<42}  n={info['n']:3d}  "
              f"USG={s.get('USG_PCT',0):.2f}  AST%={s.get('AST_PCT',0):.2f}  "
              f"OREB%={s.get('OREB_PCT',0):.2f}  DREB%={s.get('DREB_PCT',0):.2f}  "
              f"TS={s.get('TS_PCT',0):.3f}  FG3A={s.get('FG3A',0):.1f}  "
              f"BLK={s.get('BLK',0):.1f}  STL={s.get('STL',0):.1f}")

    print("\n── Top members per archetype (by PIE) ──────────────────────────────")
    for lbl, info in sorted(archetype_map.items(), key=lambda x: -x[1]["n"]):
        if lbl == -1:
            continue
        print(f"  #{lbl}  {info['name']}")
        print(f"       {', '.join(info['members'])}")

    print("\n── Top players per archetype (career avg USG×TS, ≥2 seasons) ────────")
    for lbl, info in sorted(archetype_map.items(), key=lambda x: -x[1]["n"]):
        if lbl == -1:
            continue
        sub = df[df["cluster"] == lbl].copy()
        # Require ≥2 seasons in archetype; rank by avg USG×TS as a neutral style-impact proxy
        player_seasons = sub.groupby("PLAYER_NAME").agg(
            seasons=("SEASON", "nunique"),
            avg_usg=("USG_PCT", "mean"),
            avg_ts=("TS_PCT", "mean"),
            season_list=("SEASON", lambda x: ", ".join(sorted(x.unique()))),
        )
        player_seasons["impact"] = player_seasons["avg_usg"] * player_seasons["avg_ts"]
        top = player_seasons[player_seasons["seasons"] >= 2].sort_values("impact", ascending=False).head(8)
        if len(top) == 0:
            top = player_seasons.sort_values("impact", ascending=False).head(5)
        print(f"\n  {info['name']}  (n={info['n']} player-seasons)")
        for name, row in top.iterrows():
            print(f"    {name:<28}  seasons in archetype: {row['season_list']}")

    print("\n── Notable player trajectories ─────────────────────────────────────")
    star_players = [
        "A'ja Wilson", "Breanna Stewart", "Sabrina Ionescu", "Napheesa Collier",
        "Alyssa Thomas", "Kelsey Plum", "Jewell Loyd", "Caitlin Clark",
        "Aari McDonald", "Allisha Gray", "Chelsea Gray", "Diana Taurasi",
        "Elena Delle Donne", "Sylvia Fowles", "Maya Moore", "Brittney Griner",
        "Candace Parker", "Sue Bird", "Jonquel Jones", "Skylar Diggins",
    ]
    for player in star_players:
        sub = df[df["PLAYER_NAME"] == player].sort_values("SEASON")
        if len(sub) == 0:
            continue
        print(f"\n  {player}:")
        for _, r in sub.iterrows():
            print(f"    {r['SEASON']}  {r['archetype_str']}")

    # save
    out_csv = "archetypes_results.csv"
    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "SEASON", "GP", "MIN_PG",
            "cluster", "archetype", "archetype_str",
            "USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT", "TS_PCT",
            "E_TOV_PCT", "AST_TO", "FG3A", "BLK", "STL"]
    df[[c for c in keep if c in df.columns]].to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    # ── plots ──────────────────────────────────────────────────────────────────
    print("\nUMAP 2-d for visualization...")
    embedding = run_umap(X)

    print("Plotting...")
    plot_umap_overview(df, embedding, archetype_map, "archetypes_umap_overview.png")
    plot_season_scatter(df, embedding, archetype_map, "archetypes_by_season.png")
    plot_archetype_profiles(df, archetype_map, feat_cols, "archetypes_profiles.png")
    plot_heatmap(df, archetype_map, feat_cols, "archetypes_heatmap.png")
    plot_player_trajectory(df, archetype_map, "archetypes_trajectory.png", top_n=35)

    print("\nDone.")
    return df, archetype_map, embedding


if __name__ == "__main__":
    df, archetype_map, embedding = main()
