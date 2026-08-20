# data/cache.py
"""
Local cache for WNBA API data.
Saves CSV files to data/cache/ so we never hit the API twice for the same request.
"""

import os
import hashlib
import pandas as pd
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(name: str, **kwargs) -> str:
    param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return f"{name}_{hash_suffix}.csv"


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key


def load_from_cache(name: str, **kwargs) -> pd.DataFrame | None:
    key = _cache_key(name, **kwargs)
    path = _cache_path(key)
    if path.exists():
        print(f"[cache] HIT — loading {key}")
        return pd.read_csv(path)
    return None


def save_to_cache(df: pd.DataFrame, name: str, **kwargs) -> None:
    key = _cache_key(name, **kwargs)
    path = _cache_path(key)
    df.to_csv(path, index=False)
    print(f"[cache] SAVED — {key} ({len(df)} rows)")


def get_shot_chart(
    season: str,
    player_id: int = 0,
    team_id: int = 0,
    season_type: str = "Regular Season",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull WNBA shot chart data, using local cache if available.

    Parameters
    ----------
    season        : e.g. "2024", "2025", "2026"
    player_id     : 0 = all players, specific int = one player
    team_id       : 0 = all teams
    season_type   : "Regular Season" | "Playoffs"
    force_refresh : ignore cache and re-pull from API
    """
    from nba_api.stats.endpoints import shotchartdetail

    kwargs = dict(
        season=season, player_id=player_id, team_id=team_id, season_type=season_type
    )

    if not force_refresh:
        cached = load_from_cache("shot_chart", **kwargs)
        if cached is not None:
            return cached

    print(f"[api] Fetching shot chart — season={season}, player_id={player_id} ...")
    time.sleep(0.6)

    sc = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=player_id,
        league_id="10",
        season_nullable=season,
        season_type_all_star=season_type,
        context_measure_simple="FGA",
    )
    df = sc.get_data_frames()[0]
    save_to_cache(df, "shot_chart", **kwargs)
    return df


def get_player_index(season: str = "2024", force_refresh: bool = False) -> pd.DataFrame:
    """
    Pull the full WNBA player list with IDs, teams, and active status.
    """
    from nba_api.stats.endpoints import commonallplayers

    kwargs = dict(season=season)

    if not force_refresh:
        cached = load_from_cache("player_index", **kwargs)
        if cached is not None:
            return cached

    print(f"[api] Fetching player index — season={season} ...")
    time.sleep(0.6)

    players = commonallplayers.CommonAllPlayers(
        league_id="10",
        season=season,
        is_only_current_season=0,
    )
    df = players.get_data_frames()[0]
    save_to_cache(df, "player_index", **kwargs)
    return df


def get_player_stats(
    season: str,
    season_type: str = "Regular Season",
    measure_type: str = "Base",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull per-game and per-100-possession stats for all WNBA players in a season.
    Returns (per_game_df, per_100_df).

    measure_type: "Base" for box stats, "Advanced" for USG_PCT, PACE, etc.
    """
    pg = get_player_feed(season, season_type, "PerGame", measure_type, force_refresh)
    p100 = get_player_feed(season, season_type, "Per100Possessions", measure_type,
                           force_refresh)
    return pg, p100


def get_player_feed(
    season: str,
    season_type: str = "Regular Season",
    per_mode: str = "Totals",
    measure_type: str = "Base",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    One LeagueDashPlayerStats frame, cached on the full parameter set.

    Split out of get_player_stats() so a caller can ask for a single mode. The
    site's own box score is derived from play-by-play; this feed exists to be
    the benchmark it is checked against (data/verify_box.py), and that check
    needs "Totals" — per-game figures are published to one decimal, so
    multiplying them back by GP reintroduces up to half a unit of error per
    player and manufactures a discrepancy that isn't in the data.
    """
    from nba_api.stats.endpoints import leaguedashplayerstats

    kwargs = dict(season=season, season_type=season_type, per_mode=per_mode,
                  measure_type=measure_type)
    if not force_refresh:
        cached = load_from_cache("player_stats", **kwargs)
        if cached is not None:
            return cached

    print(f"[api] Fetching player stats — season={season}, mode={per_mode}, "
          f"measure={measure_type} ...")
    time.sleep(0.6)
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        league_id_nullable="10",
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed=per_mode,
        measure_type_detailed_defense=measure_type,
    )
    df = stats.get_data_frames()[0]
    save_to_cache(df, "player_stats", **kwargs)
    return df


def get_game_ids(
    seasons,
    season_type: str = "Regular Season",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull all WNBA game IDs for one or more seasons.

    Parameters
    ----------
    seasons     : str or list of str, e.g. "2024" or ["2023", "2024"]
    season_type : "Regular Season" | "Playoffs"
    """
    from nba_api.stats.endpoints import leaguegamefinder

    if isinstance(seasons, str):
        seasons = [seasons]

    all_frames = []
    for season in seasons:
        kwargs = dict(season=season, season_type=season_type)
        if not force_refresh:
            cached = load_from_cache("game_ids", **kwargs)
            if cached is not None:
                all_frames.append(cached)
                continue

        print(f"[api] Fetching game IDs — season={season} ...")
        time.sleep(0.6)
        finder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable="10",
            season_nullable=season,
            season_type_nullable=season_type,
        )
        df = (
            finder.get_data_frames()[0][["GAME_ID", "GAME_DATE", "MATCHUP"]]
            .drop_duplicates("GAME_ID")
            .reset_index(drop=True)
        )
        save_to_cache(df, "game_ids", **kwargs)
        all_frames.append(df)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def get_pbp(game_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """Pull play-by-play events for a single game."""
    from nba_api.stats.endpoints import playbyplayv2

    kwargs = dict(game_id=game_id)
    if not force_refresh:
        cached = load_from_cache("pbp", **kwargs)
        if cached is not None:
            return cached

    print(f"[api] Fetching PBP — game_id={game_id} ...")
    time.sleep(0.6)
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
    df = pbp.get_data_frames()[0]
    save_to_cache(df, "pbp", **kwargs)
    return df


def get_box_score(game_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Pull the official player box score for a single game.

    Used to validate PBP-reconstructed lineups in seasons GameRotation does not
    cover (see data/verify_minutes.py): the MIN column is an independent record
    of how long each player was on the floor.
    """
    from nba_api.stats.endpoints import boxscoretraditionalv2

    kwargs = dict(game_id=game_id)
    if not force_refresh:
        cached = load_from_cache("box_score", **kwargs)
        if cached is not None:
            return cached

    print(f"[api] Fetching box score — game_id={game_id} ...")
    time.sleep(0.6)
    box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    df = box.get_data_frames()[0]
    save_to_cache(df, "box_score", **kwargs)
    return df


def _label_home_away(df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """
    Stamp HOME_AWAY on a rotation frame using the play-by-play as the authority.

    GameRotation returns two frames and does NOT say which is which. Checked
    against the MATCHUP string on 200 games, frame [0] is the AWAY team and
    frame [1] is the HOME team — the reverse of what this module assumed
    originally, so every cached rotation CSV carries an inverted label.

    Rather than trust frame order (or a stale cache), re-derive the home team
    from the PBP, where a substitution's text lands in HOMEDESCRIPTION or
    VISITORDESCRIPTION according to which bench made it. That repairs old cache
    files on read without re-pulling anything.
    """
    from data.pbp_rotation import _home_away_from_pbp

    try:
        pbp = get_pbp(game_id)
        home_id, away_id = _home_away_from_pbp(pbp)
    except Exception:
        home_id = away_id = None

    if home_id is not None and away_id is not None:
        df = df.copy()
        df["HOME_AWAY"] = df["TEAM_ID"].map({home_id: "home", away_id: "away"})
    return df


def get_game_rotation(game_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """Pull player on-court intervals (GameRotation) for a single game."""
    from nba_api.stats.endpoints import gamerotation

    kwargs = dict(game_id=game_id)
    if not force_refresh:
        cached = load_from_cache("rotation", **kwargs)
        if cached is not None:
            return _label_home_away(cached, game_id)

    print(f"[api] Fetching rotation — game_id={game_id} ...")
    time.sleep(0.6)
    rot = gamerotation.GameRotation(game_id=game_id, league_id="10")
    # get_data_frames() returns [AwayTeam, HomeTeam].
    away_df = rot.get_data_frames()[0].copy()
    home_df = rot.get_data_frames()[1].copy()
    home_df["HOME_AWAY"] = "home"
    away_df["HOME_AWAY"] = "away"
    df = pd.concat([home_df, away_df], ignore_index=True)
    save_to_cache(df, "rotation", **kwargs)
    return _label_home_away(df, game_id)


def get_team_ratings(
    season: str,
    season_type: str = "Regular Season",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull offensive rating, defensive rating, and net rating for all WNBA teams.
    Uses the Advanced measure type from LeagueDashTeamStats.
    """
    from nba_api.stats.endpoints import leaguedashteamstats

    kwargs = dict(season=season, season_type=season_type)
    if not force_refresh:
        cached = load_from_cache("team_ratings", **kwargs)
        if cached is not None:
            return cached

    print(f"[api] Fetching team ratings — season={season} ...")
    time.sleep(0.6)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        league_id_nullable="10",
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    )
    df = stats.get_data_frames()[0]
    cols = ["TEAM_ID", "TEAM_NAME", "GP", "W", "L", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
    df = df[[c for c in cols if c in df.columns]].copy()
    save_to_cache(df, "team_ratings", **kwargs)
    return df


def get_player_game_log(
    player_id: int, season: str, force_refresh: bool = False
) -> pd.DataFrame:
    """
    Pull game-by-game stats for a single WNBA player.
    """
    from nba_api.stats.endpoints import playergamelog

    kwargs = dict(player_id=player_id, season=season)

    if not force_refresh:
        cached = load_from_cache("player_game_log", **kwargs)
        if cached is not None:
            return cached

    print(f"[api] Fetching game log — player_id={player_id}, season={season} ...")
    time.sleep(0.6)

    log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        league_id="10",
    )
    df = log.get_data_frames()[0]
    save_to_cache(df, "player_game_log", **kwargs)
    return df
