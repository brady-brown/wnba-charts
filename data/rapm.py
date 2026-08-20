# data/rapm.py
"""
Compute split RAPM (Regularized Adjusted Plus-Minus) for WNBA players.

Model: weighted ridge regression with separate offensive and defensive columns
per player. Each stint contributes two rows (once per team on offense).
Target: ORTG (points per 100 possessions).
Result: ORAPM, DRAPM, RAPM — all in points per 100 possessions above/below average.
Final output also includes per-game and per-100-possession box stats.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from data.stints import build_stints
from data.cache import get_player_stats

# ── Box stat helpers ──────────────────────────────────────────────────────────

_PG_COLS = ["GP", "MIN", "FGA", "FTA", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT"]
_P100_COLS = ["PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV"]


def _combine_stats(dfs: list[pd.DataFrame], stat_cols: list[str]) -> pd.DataFrame:
    """
    Merge stats across multiple seasons.
    Numeric columns are GP-weighted averages; GP itself is summed.
    """
    keep = ["PLAYER_ID", "PLAYER_NAME", "GP"] + [c for c in stat_cols if c != "GP"]
    combined = pd.concat([df[keep] for df in dfs if not df.empty], ignore_index=True)

    out_rows = []
    for pid, grp in combined.groupby("PLAYER_ID", sort=False):
        total_gp = grp["GP"].sum()
        row: dict = {
            "PLAYER_ID": pid,
            "PLAYER_NAME": grp["PLAYER_NAME"].iloc[0],
            "GP": total_gp,
        }
        for col in stat_cols:
            if col == "GP":
                continue
            row[col] = (grp[col] * grp["GP"]).sum() / total_gp if total_gp > 0 else 0.0
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _fetch_box_stats(seasons, season_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull and combine per-game + per-100 stats across all requested seasons."""
    if isinstance(seasons, str):
        seasons = [seasons]

    pg_frames, p100_frames = [], []
    for season in seasons:
        pg, p100 = get_player_stats(season, season_type=season_type)
        pg_frames.append(pg)
        p100_frames.append(p100)

    pg_combined = _combine_stats(pg_frames, _PG_COLS)
    p100_combined = _combine_stats(p100_frames, _P100_COLS)

    # Most recent season's team abbreviation per player
    team_lookup = pd.concat(pg_frames, ignore_index=True)[
        ["PLAYER_ID", "TEAM_ABBREVIATION"]
    ].drop_duplicates("PLAYER_ID", keep="last")
    pg_combined = pg_combined.merge(team_lookup, on="PLAYER_ID", how="left")

    # Rename to avoid column clashes when merging
    pg_combined = pg_combined.rename(columns={c: f"{c}_PG" for c in _PG_COLS})
    p100_combined = p100_combined.rename(columns={c: f"{c}_100" for c in _P100_COLS})

    return pg_combined, p100_combined


# ── Raw on/off splits ───────────────────────────────────────────────────────


def _compute_on_off(stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the raw on-court / off-court splits that the ridge is fit on.

    For each player we accumulate the team-offense and opponent-offense
    points and possessions both while the player is ON the court and while
    their team plays without them (OFF). Off-court is computed relative to
    each player's primary team — the team they logged the most on-court
    possessions with — so mid-season trades are handled only approximately.

    Returns a DataFrame keyed by PLAYER_ID with columns:
      ON_ORTG, ON_DRTG, ON_NET, OFF_ORTG, OFF_DRTG, OFF_NET, ON_OFF,
      ON_POSS_O, ON_POSS_D, OFF_POSS_O, OFF_POSS_D
    """
    # Primary team = team with the most on-court possessions for each player.
    player_team_poss: dict = defaultdict(lambda: defaultdict(float))
    for s in stints_df.itertuples(index=False):
        stint_poss = float(s.home_poss) + float(s.away_poss)
        for pid in s.home_lineup:
            player_team_poss[pid][s.home_team_id] += stint_poss
        for pid in s.away_lineup:
            player_team_poss[pid][s.away_team_id] += stint_poss
    primary_team = {
        pid: max(tp, key=tp.get) for pid, tp in player_team_poss.items()
    }
    team_roster: dict = defaultdict(set)
    for pid, t in primary_team.items():
        team_roster[t].add(pid)

    on_o_pts: dict = defaultdict(float); on_o_poss: dict = defaultdict(float)
    on_d_pts: dict = defaultdict(float); on_d_poss: dict = defaultdict(float)
    off_o_pts: dict = defaultdict(float); off_o_poss: dict = defaultdict(float)
    off_d_pts: dict = defaultdict(float); off_d_poss: dict = defaultdict(float)

    for s in stints_df.itertuples(index=False):
        hp, ap = float(s.home_poss), float(s.away_poss)
        hpts, apts = float(s.home_pts), float(s.away_pts)
        h_lu, a_lu = s.home_lineup, s.away_lineup

        # On-court: each player gets their own team's offense and the
        # opponent's offense (their defense).
        for pid in h_lu:
            on_o_pts[pid] += hpts; on_o_poss[pid] += hp
            on_d_pts[pid] += apts; on_d_poss[pid] += ap
        for pid in a_lu:
            on_o_pts[pid] += apts; on_o_poss[pid] += ap
            on_d_pts[pid] += hpts; on_d_poss[pid] += hp

        # Off-court: rostered teammates not in the lineup for this stint.
        for pid in team_roster[s.home_team_id] - h_lu:
            off_o_pts[pid] += hpts; off_o_poss[pid] += hp
            off_d_pts[pid] += apts; off_d_poss[pid] += ap
        for pid in team_roster[s.away_team_id] - a_lu:
            off_o_pts[pid] += apts; off_o_poss[pid] += ap
            off_d_pts[pid] += hpts; off_d_poss[pid] += hp

    def rtg(pts: float, poss: float) -> float:
        return pts / poss * 100 if poss > 0 else np.nan

    rows = []
    for pid in primary_team:
        on_ortg = rtg(on_o_pts[pid], on_o_poss[pid])
        on_drtg = rtg(on_d_pts[pid], on_d_poss[pid])
        off_ortg = rtg(off_o_pts[pid], off_o_poss[pid])
        off_drtg = rtg(off_d_pts[pid], off_d_poss[pid])
        on_net = on_ortg - on_drtg
        off_net = off_ortg - off_drtg
        rows.append({
            "PLAYER_ID": pid,
            "ON_ORTG": on_ortg,
            "ON_DRTG": on_drtg,
            "ON_NET": on_net,
            "OFF_ORTG": off_ortg,
            "OFF_DRTG": off_drtg,
            "OFF_NET": off_net,
            "ON_OFF": on_net - off_net,
            "ON_POSS_O": on_o_poss[pid],
            "ON_POSS_D": on_d_poss[pid],
            "OFF_POSS_O": off_o_poss[pid],
            "OFF_POSS_D": off_d_poss[pid],
        })

    out = pd.DataFrame(rows)
    rtg_cols = ["ON_ORTG", "ON_DRTG", "ON_NET", "OFF_ORTG", "OFF_DRTG",
                "OFF_NET", "ON_OFF"]
    poss_cols = ["ON_POSS_O", "ON_POSS_D", "OFF_POSS_O", "OFF_POSS_D"]
    out[rtg_cols] = out[rtg_cols].round(1)
    out[poss_cols] = out[poss_cols].round(1)
    return out


# ── Design matrix ─────────────────────────────────────────────────────────────


def _build_design_matrix(stints_df: pd.DataFrame):
    """
    Build the weighted-ridge design matrix from stints.

    Each stint contributes two rows (one per offensive team). The first
    n_players columns are offensive indicators, the second n_players are
    defensive. Target y is the stint ORTG; weight w is possessions.

    Returns (X, y, w, all_players, p_idx).
    """
    all_players = sorted(
        {
            pid
            for lineup in stints_df["home_lineup"].tolist()
            + stints_df["away_lineup"].tolist()
            for pid in lineup
        }
    )
    n_players = len(all_players)
    p_idx = {pid: i for i, pid in enumerate(all_players)}

    n_rows = len(stints_df) * 2
    n_cols = n_players * 2

    X = np.zeros((n_rows, n_cols), dtype=np.float32)
    y = np.zeros(n_rows, dtype=np.float32)
    w = np.zeros(n_rows, dtype=np.float32)

    for i, stint in enumerate(stints_df.itertuples(index=False)):
        home_poss = max(float(stint.home_poss), 0.01)
        away_poss = max(float(stint.away_poss), 0.01)
        home_ortg = float(stint.home_pts) / home_poss * 100
        away_ortg = float(stint.away_pts) / away_poss * 100

        r_home = i * 2
        r_away = i * 2 + 1

        for pid in stint.home_lineup:
            X[r_home, p_idx[pid]] = 1.0
            X[r_away, n_players + p_idx[pid]] = 1.0
        for pid in stint.away_lineup:
            X[r_away, p_idx[pid]] = 1.0
            X[r_home, n_players + p_idx[pid]] = 1.0

        y[r_home], w[r_home] = home_ortg, home_poss
        y[r_away], w[r_away] = away_ortg, away_poss

    return X, y, w, all_players, p_idx


def _fit_weighted_ridge(X, y, w, alpha: float) -> np.ndarray:
    """Weighted ridge: center y by its weighted mean, scale rows by √w, fit."""
    y_mean = np.average(y, weights=w)
    sqrt_w = np.sqrt(w)
    model = Ridge(alpha=alpha, fit_intercept=False, max_iter=10_000)
    model.fit(X * sqrt_w[:, np.newaxis], (y - y_mean) * sqrt_w)
    return model.coef_, y_mean


# ── Alpha tuning ──────────────────────────────────────────────────────────────


def find_best_alpha(
    stints_df: pd.DataFrame,
    alphas=None,
    cv: int = 5,
    min_stint_poss: float = 0.5,
    random_state: int = 0,
    verbose: bool = True,
) -> tuple[float, pd.DataFrame]:
    """
    Pick the ridge alpha that best predicts held-out stints via K-fold CV.

    Folds are split at the *stint* level (both rows of a stint stay together)
    so the offensive/defensive halves of a possession can't leak across the
    train/test boundary. For each alpha we fit weighted ridge on the training
    stints and score possession-weighted RMSE of predicted vs actual ORTG on
    the held-out stints, averaged over folds.

    Parameters
    ----------
    stints_df      : stint DataFrame (as returned by build_stints)
    alphas         : iterable of alphas to try; default is a log grid
                     from 100 to ~31,600
    cv             : number of folds
    min_stint_poss : drop stints below this many combined possessions first
    random_state   : seed for the fold shuffle

    Returns
    -------
    best_alpha : the alpha with the lowest mean CV RMSE
    scores     : DataFrame with columns [alpha, cv_rmse] for every alpha tried
    """
    if alphas is None:
        alphas = np.logspace(2, 4.5, 12)
    alphas = [float(a) for a in alphas]

    total_poss = stints_df["home_poss"] + stints_df["away_poss"]
    stints_df = stints_df[total_poss >= min_stint_poss]
    n_stints = len(stints_df)
    if n_stints < cv:
        raise ValueError(f"Need at least {cv} stints to run {cv}-fold CV.")

    X, y, w, _, _ = _build_design_matrix(stints_df)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    stint_idx = np.arange(n_stints)

    if verbose:
        print(f"Tuning alpha over {len(alphas)} values, {cv}-fold CV "
              f"on {n_stints:,} stints ...")

    rows = []
    for alpha in alphas:
        fold_rmses = []
        for train_st, test_st in kf.split(stint_idx):
            train_rows = np.concatenate([train_st * 2, train_st * 2 + 1])
            test_rows = np.concatenate([test_st * 2, test_st * 2 + 1])

            coef, y_mean = _fit_weighted_ridge(
                X[train_rows], y[train_rows], w[train_rows], alpha
            )
            pred = X[test_rows] @ coef + y_mean
            err = pred - y[test_rows]
            rmse = float(np.sqrt(np.average(err ** 2, weights=w[test_rows])))
            fold_rmses.append(rmse)

        mean_rmse = float(np.mean(fold_rmses))
        rows.append({"alpha": alpha, "cv_rmse": mean_rmse})
        if verbose:
            print(f"  alpha={alpha:>10.1f}   CV weighted RMSE={mean_rmse:.4f}")

    scores = pd.DataFrame(rows)
    best_alpha = float(scores.loc[scores["cv_rmse"].idxmin(), "alpha"])
    if verbose:
        print(f"Best alpha: {best_alpha:.1f}")
    return best_alpha, scores


# ── Main function ─────────────────────────────────────────────────────────────


def compute_rapm(
    seasons,
    season_type: str = "Regular Season",
    ridge_alpha: float | str = 4000.0,
    min_stint_poss: float = 0.5,
    stints_df: pd.DataFrame | None = None,
    name_lookup: dict | None = None,
    cv: int = 5,
    filter_garbage_time: bool = False,
    garbage_margin: int = 20,
    garbage_window: float = 300.0,
) -> pd.DataFrame:
    """
    Compute split ORAPM / DRAPM / RAPM for all players in the given season(s),
    then join per-game and per-100 box stats.

    Parameters
    ----------
    seasons          : str or list of str, e.g. "2024" or ["2023", "2024"]
    season_type      : "Regular Season" | "Playoffs"
    ridge_alpha      : L2 regularization strength (higher = more shrinkage).
                       Pass "auto" to pick it by cross-validation via
                       find_best_alpha.
    min_stint_poss   : drop stints with fewer combined possessions than this
    stints_df            : pre-built stint DataFrame (skip data fetch if provided)
    name_lookup          : dict of PERSON_ID → name (returned by build_stints)
    cv                   : number of folds used when ridge_alpha="auto"
    filter_garbage_time  : drop stints that occur in garbage time before fitting
    garbage_margin       : point margin threshold for garbage time (default 20)
    garbage_window       : seconds from end of regulation to consider (default 300 = last 5 min)

    Returns
    -------
    DataFrame sorted by RAPM descending with columns:
      PLAYER_NAME, ORAPM, DRAPM, RAPM, POSS,
      ON_ORTG, ON_DRTG, ON_NET, OFF_ORTG, OFF_DRTG, OFF_NET, ON_OFF,
      ON_POSS_O, ON_POSS_D, OFF_POSS_O, OFF_POSS_D,
      GP_PG, MIN_PG, PTS_PG, REB_PG, AST_PG, STL_PG, BLK_PG, TOV_PG,
      FG_PCT_PG, FG3_PCT_PG,
      PTS_100, REB_100, AST_100, STL_100, BLK_100, TOV_100

    The ON_* / OFF_* / ON_OFF columns are the raw, un-regularized on/off
    splits aggregated from the stints — the same data the ridge is fit on,
    before adjustment. ORAPM/DRAPM/RAPM are the regularized, opponent- and
    teammate-adjusted versions of those splits.
    """
    if stints_df is None:
        print(f"Fetching stints for: {seasons}")
        stints_df, name_lookup = build_stints(seasons, season_type=season_type)

    if stints_df.empty:
        raise ValueError("No stint data — check season IDs and API connectivity.")

    if name_lookup is None:
        name_lookup = {}

    # Drop very-short stints (noise)
    total_poss = stints_df["home_poss"] + stints_df["away_poss"]
    stints_df = stints_df[total_poss >= min_stint_poss].copy()

    if filter_garbage_time and "score_margin" in stints_df.columns:
        reg_end = 4 * 600  # 2400 s — end of regulation (QUARTER_SEC = 600)
        garbage_start = reg_end - garbage_window
        is_garbage = (
            (stints_df["t_start"] >= garbage_start)
            & (stints_df["score_margin"].abs() >= garbage_margin)
        )
        n_dropped = is_garbage.sum()
        stints_df = stints_df[~is_garbage].copy()
        print(f"Garbage time filter: dropped {n_dropped:,} stints "
              f"(margin≥{garbage_margin}, last {garbage_window/60:.0f} min of regulation)")

    print(
        f"Using {len(stints_df):,} stints after filtering (min_poss={min_stint_poss})"
    )

    # ── Pick alpha by cross-validation if requested ──────────────────────────
    if isinstance(ridge_alpha, str):
        if ridge_alpha != "auto":
            raise ValueError("ridge_alpha must be a number or \"auto\".")
        ridge_alpha, _ = find_best_alpha(
            stints_df, cv=cv, min_stint_poss=min_stint_poss
        )

    # ── Build design matrix ──────────────────────────────────────────────────
    X, y, w, all_players, p_idx = _build_design_matrix(stints_df)
    n_players = len(all_players)
    print(
        f"Design matrix: {X.shape[0]:,} rows × {X.shape[1]:,} cols "
        f"({n_players} players)"
    )

    # ── Weighted ridge regression ────────────────────────────────────────────
    print(f"Fitting ridge regression (alpha={ridge_alpha}) ...")
    coef, _ = _fit_weighted_ridge(X, y, w, ridge_alpha)
    o_coef = coef[:n_players]
    d_coef = coef[n_players:]

    # ── Possession totals per player ─────────────────────────────────────────
    player_poss = {pid: 0.0 for pid in all_players}
    for stint in stints_df.itertuples(index=False):
        stint_poss = float(stint.home_poss) + float(stint.away_poss)
        for pid in stint.home_lineup | stint.away_lineup:
            player_poss[pid] += stint_poss

    # ── RAPM results ─────────────────────────────────────────────────────────
    results = pd.DataFrame(
        {
            "PLAYER_ID": all_players,
            "PLAYER_NAME": [name_lookup.get(pid, str(pid)) for pid in all_players],
            "ORAPM": o_coef.round(2),
            "DRAPM": (-d_coef).round(2),
            "POSS": [round(player_poss[pid], 1) for pid in all_players],
        }
    )
    results["RAPM"] = (results["ORAPM"] + results["DRAPM"]).round(2)

    # ── Join raw on/off splits (the un-regularized inputs) ─────────────────────
    on_off_df = _compute_on_off(stints_df)
    results = results.merge(on_off_df, on="PLAYER_ID", how="left")

    # ── Join box stats ────────────────────────────────────────────────────────
    print("Fetching box stats ...")
    try:
        pg_df, p100_df = _fetch_box_stats(seasons, season_type)
        results = results.merge(
            pg_df.drop(columns=["PLAYER_NAME"]), on="PLAYER_ID", how="left"
        )
        results = results.merge(
            p100_df.drop(columns=["PLAYER_NAME"]), on="PLAYER_ID", how="left"
        )

        # Usage Rate: % of team offensive possessions used while on court
        # USG% = 100 * (FGA + 0.44*FTA + TOV) * GP / ON_POSS_O
        if all(c in results.columns for c in ("FGA_PG", "FTA_PG", "TOV_PG", "GP_PG", "ON_POSS_O")):
            player_uses = (
                (results["FGA_PG"] + 0.44 * results["FTA_PG"] + results["TOV_PG"])
                * results["GP_PG"]
            )
            results["USG_PCT"] = (player_uses / results["ON_POSS_O"] * 100).round(1)

        # Round box stat columns
        pg_num = [f"{c}_PG" for c in _PG_COLS if c not in ("GP", "FG_PCT", "FG3_PCT")]
        p100_num = [f"{c}_100" for c in _P100_COLS]
        for col in pg_num + p100_num:
            if col in results.columns:
                results[col] = results[col].round(1)
        if "GP_PG" in results.columns:
            results["GP_PG"] = results["GP_PG"].round(0).astype("Int64")
        for pct_col in ("FG_PCT_PG", "FG3_PCT_PG"):
            if pct_col in results.columns:
                results[pct_col] = (results[pct_col] * 100).round(1)

    except Exception as e:
        print(f"  [warn] box stats unavailable: {e}")
    # Guarded: the box-stat join above is best-effort, and the endpoint has no
    # data for some early seasons. Without this check a missing join turns into
    # a KeyError here and takes down the whole run.
    if all(c in results.columns for c in ("STL_100", "BLK_100")):
        results["STOCKS_100"] = results["STL_100"] + results["BLK_100"]

    col_order = [
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "ORAPM",
        "DRAPM",
        "RAPM",
        "POSS",
        "USG_PCT",
        "ON_ORTG",
        "ON_DRTG",
        "ON_NET",
        "OFF_ORTG",
        "OFF_DRTG",
        "OFF_NET",
        "ON_OFF",
        "ON_POSS_O",
        "ON_POSS_D",
        "OFF_POSS_O",
        "OFF_POSS_D",
        "GP_PG",
        "MIN_PG",
        "PTS_PG",
        "REB_PG",
        "AST_PG",
        "STL_PG",
        "BLK_PG",
        "TOV_PG",
        "FGA_PG",
        "FTA_PG",
        "FG_PCT_PG",
        "FG3_PCT_PG",
        "PTS_100",
        "REB_100",
        "OREB_100",
        "DREB_100",
        "AST_100",
        "STL_100",
        "BLK_100",
        "TOV_100",
        "STOCKS_100",
        "PLAYER_ID",
    ]
    results = results[[c for c in col_order if c in results.columns]]

    return results.sort_values("RAPM", ascending=False).reset_index(drop=True)
