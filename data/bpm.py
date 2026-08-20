# data/bpm.py
"""
WNBA Box Plus-Minus (BPM) — calibrated against RAPM.

Fit ridge regressions from per-100 box stats to ORAPM / DRAPM targets,
then apply to all players with a box score (including low-minute players
whose RAPM is too shrunk to be informative).

Separate OBPM / DBPM models, summed to BPM. All values in points per 100
possessions above/below league average (same scale as RAPM).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data.rapm import compute_rapm

# Features used for each sub-model. These must all be columns in the RAPM output.
OBPM_FEATURES = ["AST_100", "TOV_100", "USG_PCT", "TS_PCT"]
DBPM_FEATURES = ["STL_100", "BLK_100", "DREB_100", "OREB_100"]

MIN_POSS_TRAIN = 500  # below this, RAPM is too shrunk to be a reliable training target


# ── Helpers ───────────────────────────────────────────────────────────────────


def _add_ts_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TS_PCT from PTS_PG, FGA_PG, FTA_PG (in-place on a copy)."""
    df = df.copy()
    denom = 2.0 * (df["FGA_PG"] + 0.44 * df["FTA_PG"])
    df["TS_PCT"] = np.where(denom > 0, df["PTS_PG"] / denom * 100, np.nan)
    return df


# ── Model fitting ─────────────────────────────────────────────────────────────


def fit_bpm(
    rapm_df: pd.DataFrame,
    min_poss: float = MIN_POSS_TRAIN,
    alpha: float = 10.0,
) -> tuple[dict, dict]:
    """
    Fit OBPM and DBPM ridge models using RAPM as the regression target.

    Players with fewer than `min_poss` possessions are excluded from training
    because their RAPM is heavily regularized toward zero and would corrupt the fit.
    Remaining players are weighted by possessions so stable estimates matter more.

    Parameters
    ----------
    rapm_df  : output of compute_rapm (needs ORAPM, DRAPM, POSS, and feature cols)
    min_poss : possession threshold for training inclusion
    alpha    : ridge regularization for the BPM regression itself

    Returns
    -------
    (o_model_info, d_model_info) — dicts with keys:
        features, coef, intercept, scaler
    """
    df = _add_ts_pct(rapm_df)
    train = df[df["POSS"] >= min_poss].copy()

    # Only use features that actually exist in the DataFrame
    o_feats = [f for f in OBPM_FEATURES if f in train.columns]
    d_feats = [f for f in DBPM_FEATURES if f in train.columns]
    missing = set(OBPM_FEATURES + DBPM_FEATURES) - set(o_feats + d_feats)
    if missing:
        print(f"  [bpm] missing features (re-run compute_rapm to add them): {sorted(missing)}")

    all_feats = list(dict.fromkeys(o_feats + d_feats))
    train = train.dropna(subset=all_feats + ["ORAPM", "DRAPM"])

    if len(train) < 10:
        raise ValueError(
            f"Only {len(train)} players meet min_poss={min_poss} with complete stats — "
            "need at least 10 to fit BPM."
        )

    poss_weights = train["POSS"].values

    def _fit(features: list[str], target: str) -> dict:
        X = train[features].values.astype(float)
        y = train[target].values.astype(float)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = Ridge(alpha=alpha)
        model.fit(X_s, y, sample_weight=poss_weights)
        return {
            "features": features,
            "coef": model.coef_,
            "intercept": float(model.intercept_),
            "scaler": scaler,
        }

    o_info = _fit(o_feats, "ORAPM")
    d_info = _fit(d_feats, "DRAPM")

    print(f"BPM fit on {len(train)} players (min_poss={min_poss})")
    print("  OBPM features:", ", ".join(
        f"{f}={c:+.3f}" for f, c in zip(o_info["features"], o_info["coef"])
    ))
    print("  DBPM features:", ", ".join(
        f"{f}={c:+.3f}" for f, c in zip(d_info["features"], d_info["coef"])
    ))

    return o_info, d_info


# ── Model application ─────────────────────────────────────────────────────────


def apply_bpm(
    rapm_df: pd.DataFrame,
    o_info: dict,
    d_info: dict,
) -> pd.DataFrame:
    """
    Apply fitted BPM models to a rapm DataFrame.

    Returns the same DataFrame with OBPM, DBPM, BPM, and TS_PCT columns added.
    Players missing any required feature get NaN.
    """
    df = _add_ts_pct(rapm_df)

    def _predict(model_info: dict) -> np.ndarray:
        feats = model_info["features"]
        mask = df[feats].notna().all(axis=1)
        preds = np.full(len(df), np.nan)
        if mask.any():
            X = model_info["scaler"].transform(df.loc[mask, feats].values.astype(float))
            preds[mask.values] = X @ model_info["coef"] + model_info["intercept"]
        return preds

    df["OBPM"] = np.round(_predict(o_info), 2)
    df["DBPM"] = np.round(_predict(d_info), 2)
    df["BPM"] = np.round(df["OBPM"] + df["DBPM"], 2)
    df["TS_PCT"] = df["TS_PCT"].round(1)
    return df


# ── Public API ────────────────────────────────────────────────────────────────


def compute_bpm(
    seasons,
    season_type: str = "Regular Season",
    training_rapm: pd.DataFrame | None = None,
    ridge_alpha: float | str = 4000.0,
    min_poss_train: float = MIN_POSS_TRAIN,
    bpm_alpha: float = 10.0,
    filter_garbage_time: bool = True,
    stints_df: pd.DataFrame | None = None,
    name_lookup: dict | None = None,
) -> pd.DataFrame:
    """
    Compute BPM for all players in `seasons`.

    Fits OBPM/DBPM models using `training_rapm` as the regression target, then
    applies them to the RAPM computed for `seasons`. If `training_rapm` is not
    provided, uses the same seasons' RAPM for both fitting and scoring (in-sample).

    Passing multi-season RAPM as `training_rapm` gives more stable coefficients
    because the training targets are less noisy.

    Parameters
    ----------
    seasons          : str or list of str, e.g. "2025" or ["2023","2024","2025"]
    season_type      : "Regular Season" | "Playoffs"
    training_rapm    : pre-computed RAPM DataFrame to use as regression target.
                       If None, RAPM is computed from `seasons` and used in-sample.
    ridge_alpha      : RAPM ridge alpha (only used when computing RAPM internally)
    min_poss_train   : minimum possessions for a player to enter the BPM training set
    bpm_alpha        : ridge alpha for the BPM box-score regression
    filter_garbage_time : passed to compute_rapm
    stints_df        : pre-built stints (passed through to compute_rapm)
    name_lookup      : player name lookup (passed through to compute_rapm)

    Returns
    -------
    DataFrame sorted by BPM descending — same columns as compute_rapm output plus:
      OBPM, DBPM, BPM  (points per 100 above/below average)
      TS_PCT           (true shooting %, computed from box stats)
    """
    rapm_df = compute_rapm(
        seasons,
        season_type=season_type,
        ridge_alpha=ridge_alpha,
        filter_garbage_time=filter_garbage_time,
        stints_df=stints_df,
        name_lookup=name_lookup,
    )

    train_df = training_rapm if training_rapm is not None else rapm_df
    o_info, d_info = fit_bpm(train_df, min_poss=min_poss_train, alpha=bpm_alpha)

    result = apply_bpm(rapm_df, o_info, d_info)

    # Put BPM columns right after RAPM in the output
    cols = list(result.columns)
    rapm_idx = cols.index("RAPM") + 1
    for col in ["BPM", "DBPM", "OBPM", "TS_PCT"]:
        if col in cols:
            cols.remove(col)
            cols.insert(rapm_idx, col)

    return result[cols].sort_values("BPM", ascending=False).reset_index(drop=True)
