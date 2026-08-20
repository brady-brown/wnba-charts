# data/evaluation.py
"""
Meta-metric evaluation for WNBA RAPM and BPM, following Franks et al. (2016).

Three tests:
  discrimination — split-half reliability: fraction of variance that is signal vs. noise.
  stability      — year-to-year persistence, normalized by discrimination.
  independence   — redundancy across metrics (correlation matrix + PCA).

All three take pre-built rapm DataFrames so nothing is re-fetched from the API.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data.rapm import _build_design_matrix, _fit_weighted_ridge


# ── Internal helpers ──────────────────────────────────────────────────────────


def _fit_rapm_half(stints_half: pd.DataFrame, alpha: float) -> dict[int, dict]:
    """
    Fit RAPM on a half-split stints DataFrame.
    Returns {player_id: {'ORAPM': float, 'DRAPM': float, 'POSS': float}}.
    """
    if stints_half.empty:
        return {}
    X, y, w, all_players, _ = _build_design_matrix(stints_half)
    n_players = len(all_players)
    coef, _ = _fit_weighted_ridge(X, y, w, alpha)
    o_coef = coef[:n_players]
    d_coef = coef[n_players:]

    poss: dict[int, float] = {pid: 0.0 for pid in all_players}
    for stint in stints_half.itertuples(index=False):
        sp = float(stint.home_poss) + float(stint.away_poss)
        for pid in stint.home_lineup | stint.away_lineup:
            if pid in poss:
                poss[pid] += sp

    return {
        pid: {
            "ORAPM": float(o_coef[i]),
            "DRAPM": float(-d_coef[i]),
            "POSS": poss[pid],
        }
        for i, pid in enumerate(all_players)
    }


def _spearman_brown(r: float) -> float:
    """Correct a split-half r to full-length reliability: ρ = 2r / (1 + r)."""
    return 2.0 * r / (1.0 + r) if r > -1.0 else np.nan


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ── 1. Discrimination ─────────────────────────────────────────────────────────


def rapm_discrimination(
    stints_df: pd.DataFrame,
    alpha: float,
    n_splits: int = 20,
    min_poss_per_half: float = 250.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Estimate split-half reliability of RAPM, ORAPM, and DRAPM.

    For each random 50/50 split of stints, fits weighted ridge on each half
    (alpha held fixed), correlates per-player estimates across the two halves
    for players with enough possessions in both, then applies Spearman-Brown
    to estimate full-season reliability.

    Reliability ranges from 0 (pure noise) to 1 (perfect signal). NBA RAPM
    typically lands around 0.6–0.75 for a single season; WNBA will be lower
    due to the shorter season and smaller player pool.

    Parameters
    ----------
    stints_df        : full-season stint DataFrame (as returned by build_stints,
                       after garbage-time / min_poss filtering if desired)
    alpha            : ridge alpha used for each half-fit — hold at the value
                       chosen for the full-data fit so alpha doesn't confound splits
    n_splits         : random splits to average over (20 is sufficient)
    min_poss_per_half: possessions required in *each* half to include a player;
                       250 ≈ ~10 min/game over a full half-season
    seed             : RNG seed for reproducibility

    Returns
    -------
    DataFrame with columns:
        metric, r_raw (mean half-half r), reliability (Spearman-Brown corrected),
        n_players_avg, n_splits_used
    """
    rng = np.random.default_rng(seed)
    n = len(stints_df)
    arr = stints_df.reset_index(drop=True)

    rs: dict[str, list[float]] = {"ORAPM": [], "DRAPM": [], "RAPM": []}
    ns: dict[str, list[int]]   = {"ORAPM": [], "DRAPM": [], "RAPM": []}

    for _ in range(n_splits):
        perm = rng.permutation(n)
        mid = n // 2
        ra = _fit_rapm_half(arr.iloc[perm[:mid]], alpha)
        rb = _fit_rapm_half(arr.iloc[perm[mid:]], alpha)

        common = {
            pid for pid in set(ra) & set(rb)
            if ra[pid]["POSS"] >= min_poss_per_half
            and rb[pid]["POSS"] >= min_poss_per_half
        }
        if len(common) < 10:
            continue

        pids = sorted(common)
        for metric in ("ORAPM", "DRAPM", "RAPM"):
            if metric == "RAPM":
                va = np.array([ra[p]["ORAPM"] + ra[p]["DRAPM"] for p in pids])
                vb = np.array([rb[p]["ORAPM"] + rb[p]["DRAPM"] for p in pids])
            else:
                va = np.array([ra[p][metric] for p in pids])
                vb = np.array([rb[p][metric] for p in pids])

            r = _corr(va, vb)
            if np.isfinite(r):
                rs[metric].append(r)
                ns[metric].append(len(pids))

    rows = []
    for metric in ("ORAPM", "DRAPM", "RAPM"):
        if not rs[metric]:
            continue
        r_mean = float(np.mean(rs[metric]))
        rows.append({
            "metric": metric,
            "r_raw": round(r_mean, 3),
            "reliability": round(_spearman_brown(r_mean), 3),
            "n_players_avg": round(float(np.mean(ns[metric])), 1),
            "n_splits_used": len(rs[metric]),
        })
    return pd.DataFrame(rows)


# ── 2. Stability ──────────────────────────────────────────────────────────────


def stability(
    rapm_by_season: dict[str, pd.DataFrame],
    discrimination_by_season: dict[str, pd.DataFrame] | None = None,
    min_poss: float = 500.0,
) -> pd.DataFrame:
    """
    Compute year-to-year stability of RAPM, ORAPM, and DRAPM.

    For each consecutive season pair, correlates metric values for players
    appearing in both seasons above the possession threshold. The attenuation-
    corrected stability divides by sqrt(rho_a * rho_b) — if both seasons had
    perfect reliability this equals the raw r; in practice it adjusts for the
    fact that noisy measurements mechanically depress cross-season correlations.

    A stability_corrected near 1.0 means the reliable portion of the metric
    is a persistent player attribute. Near 0 means it's mostly transient context.

    Parameters
    ----------
    rapm_by_season         : {season_str: rapm_df} — one per season
    discrimination_by_season : {season_str: discrimination_df from rapm_discrimination};
                               if None, attenuation correction is skipped
    min_poss               : possession threshold applied per season

    Returns
    -------
    DataFrame with columns:
        season_a, season_b, metric, n_players,
        r_across, reliability_a, reliability_b, stability_corrected
    """
    seasons = sorted(rapm_by_season)
    metrics = ["ORAPM", "DRAPM", "RAPM"]
    rows = []

    for sa, sb in zip(seasons[:-1], seasons[1:]):
        dfa = rapm_by_season[sa]
        dfb = rapm_by_season[sb]
        dfa = dfa[dfa["POSS"] >= min_poss][["PLAYER_ID"] + metrics]
        dfb = dfb[dfb["POSS"] >= min_poss][["PLAYER_ID"] + metrics]

        merged = dfa.merge(dfb, on="PLAYER_ID", suffixes=("_a", "_b"))
        if len(merged) < 10:
            continue

        disc_a = discrimination_by_season.get(sa) if discrimination_by_season else None
        disc_b = discrimination_by_season.get(sb) if discrimination_by_season else None

        for metric in metrics:
            va = merged[f"{metric}_a"].values
            vb = merged[f"{metric}_b"].values
            r = _corr(va, vb)
            if not np.isfinite(r):
                continue

            rho_a = rho_b = np.nan
            if disc_a is not None:
                row = disc_a[disc_a["metric"] == metric]
                if not row.empty:
                    rho_a = float(row["reliability"].iloc[0])
            if disc_b is not None:
                row = disc_b[disc_b["metric"] == metric]
                if not row.empty:
                    rho_b = float(row["reliability"].iloc[0])

            if np.isfinite(rho_a) and np.isfinite(rho_b) and rho_a > 0 and rho_b > 0:
                stab_corr = round(r / np.sqrt(rho_a * rho_b), 3)
            else:
                stab_corr = np.nan

            rows.append({
                "season_a": sa,
                "season_b": sb,
                "metric": metric,
                "n_players": len(merged),
                "r_across": round(r, 3),
                "reliability_a": round(rho_a, 3) if np.isfinite(rho_a) else np.nan,
                "reliability_b": round(rho_b, 3) if np.isfinite(rho_b) else np.nan,
                "stability_corrected": stab_corr,
            })

    return pd.DataFrame(rows)


# ── 3. Independence ───────────────────────────────────────────────────────────


def independence(
    rapm_df: pd.DataFrame,
    min_poss: float = 500.0,
    extra_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correlation matrix + PCA over evaluation metrics to quantify redundancy.

    High pairwise correlations mean multiple columns encode the same information.
    The PCA scree shows the effective dimensionality: how many truly independent
    axes of player value exist in your metric suite.

    Parameters
    ----------
    rapm_df    : output of compute_rapm, optionally with BPM columns (OBPM, DBPM, BPM)
    min_poss   : possession threshold for inclusion
    extra_cols : additional columns to include beyond the default set

    Returns
    -------
    (corr_df, pca_df):
        corr_df  — Pearson correlation matrix, rounded to 3 decimal places
        pca_df   — component, explained_variance, cumulative_variance
    """
    BASE = [
        "RAPM", "ORAPM", "DRAPM",
        "ON_OFF",
        "USG_PCT", "TS_PCT",
        "PTS_100", "AST_100", "DREB_100", "STL_100", "BLK_100", "TOV_100",
    ]
    cols = BASE + (extra_cols or [])
    available = [c for c in cols if c in rapm_df.columns]

    df = rapm_df[rapm_df["POSS"] >= min_poss][available].dropna()
    if len(df) < 10:
        raise ValueError(f"Only {len(df)} players with complete data — lower min_poss.")

    corr_df = df.corr(method="pearson").round(3)

    X = StandardScaler().fit_transform(df.values)
    pca = PCA()
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame({
        "component": np.arange(1, len(cum_var) + 1),
        "explained_variance": pca.explained_variance_ratio_.round(3),
        "cumulative_variance": cum_var.round(3),
    })

    print(f"Independence: {len(df)} players, {len(available)} metrics")
    n_90 = int((cum_var >= 0.90).argmax()) + 1
    print(f"  {n_90} component(s) explain ≥90% of variance "
          f"(of {len(available)} total metrics)")

    return corr_df, pca_df


# ── Summary printer ───────────────────────────────────────────────────────────


def print_discrimination(disc_df: pd.DataFrame) -> None:
    print("\nRApm Discrimination (split-half reliability)")
    print(f"  {'Metric':<10} {'r_raw':>7} {'reliability':>12} {'n_players':>10}")
    print("  " + "-" * 44)
    for _, row in disc_df.iterrows():
        bar = "█" * int(row["reliability"] * 20)
        print(f"  {row['metric']:<10} {row['r_raw']:>7.3f} {row['reliability']:>12.3f}"
              f"  {row['n_players_avg']:>6.0f}   {bar}")
    print()
    print("  Interpretation: 0 = pure noise, 1 = perfect signal.")
    print("  NBA single-season RAPM typically ~0.60–0.75.")
    print("  DRAPM expected to be lower than ORAPM.")


def print_stability(stab_df: pd.DataFrame) -> None:
    print("\nYear-over-Year Stability")
    print(f"  {'Pair':<14} {'Metric':<8} {'N':>5} {'r_across':>9} {'corrected':>10}")
    print("  " + "-" * 52)
    for _, row in stab_df.iterrows():
        pair = f"{row['season_a']}→{row['season_b']}"
        corr = f"{row['stability_corrected']:.3f}" if pd.notna(row['stability_corrected']) else "  n/a"
        print(f"  {pair:<14} {row['metric']:<8} {row['n_players']:>5} "
              f"{row['r_across']:>9.3f} {corr:>10}")
