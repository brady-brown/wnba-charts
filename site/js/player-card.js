/* ════════════════════════════════════════════════════════════════════════
   player-card.js — THE player modal. Every page uses this one.

   Ported from nba_charts/site/js/player-card.js. Content is deliberately
   quick stats + shot charts. RAPM and on/off are NOT here: when you click a
   name you want to know how she scores, not re-read the impact table you are
   already looking at.

   Injects its own DOM + CSS on first use, so a page only needs:
       <script src="js/common.js"></script>
       <script src="js/zone-chart.js"></script>
       <script src="js/court.js"></script>
       <script src="js/player-card.js"></script>
       PlayerCard.open(playerId, { season, scope, name, teamName });

   Differences from the NBA original, all forced by this project's layout:
     * shot payloads live at data/{season}/shots/{slug}{scopeSuffix}.json,
       not .../shots/{scope}/{slug}.json
     * field names follow data/build_site.py (pts/reb/ast, not ppg/rpg/apg)
     * no bio row — there is no roster feed here, and inventing one would be
       worse than omitting it
     * court geometry is read from the shot payload, because the WNBA
       three-point line moved in 2004 and 2013

   Requires: common.js (loadJSON/ensurePlotly/slugify), zone-chart.js, court.js.
   Percentile badges appear only if percentile.js is also loaded.
   ════════════════════════════════════════════════════════════════════════ */
const PlayerCard = (function () {
  "use strict";

  const PS = {};        // `${season}|${scope}`        -> {id: row}
  const SHOTS = {};     // `${season}|${scope}|${slug}`-> payload
  const IDX = {};       // `${season}|${scope}`        -> shots-index
  let built = false;

  const CSS = `
  .pcx-overlay { position:fixed; inset:0; background:rgba(15,23,42,.55);
    display:none; align-items:flex-start; justify-content:center; z-index:1000;
    overflow-y:auto; padding:28px 14px; }
  .pcx-overlay.open { display:flex; }
  .pcx { background:var(--card,#fff); border:1px solid var(--line,#E2E8F0);
    border-radius:14px; width:min(920px,100%); box-shadow:0 20px 60px rgba(15,23,42,.28);
    padding:18px 20px 22px; position:relative; }
  .pcx-close { position:absolute; top:12px; right:14px; cursor:pointer;
    color:var(--muted,#64748B); font-size:1.05rem; line-height:1; padding:4px; }
  .pcx-close:hover { color:var(--ink,#0F172A); }
  .pcx-nm { font-size:1.35rem; font-weight:800; letter-spacing:-.01em; }
  .pcx-tm { color:var(--muted,#64748B); font-size:.85rem; margin-top:1px; }
  .pcx-chips { margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; }
  .pcx-chip { font-size:.7rem; font-weight:700; padding:3px 8px; border-radius:999px;
    background:var(--card2,#F1F5F9); border:1px solid var(--line,#E2E8F0);
    color:var(--muted,#64748B); }
  .pcx-stats { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }
  @media (max-width:700px){ .pcx-stats { grid-template-columns:1fr; } }
  .pcx-h { font-size:.7rem; font-weight:800; text-transform:uppercase;
    letter-spacing:.07em; color:var(--muted,#64748B); margin-bottom:6px; }
  .pcx-r { display:flex; justify-content:space-between; align-items:center;
    padding:4px 0; border-bottom:1px solid var(--line2,#EEF2F6); font-size:.83rem; }
  .pcx-k { color:var(--muted,#64748B); }
  .pcx-v { font-weight:700; font-variant-numeric:tabular-nums;
    display:flex; align-items:center; gap:6px; }
  .pcx-pct { font-size:.65rem; font-weight:800; padding:1px 5px; border-radius:5px; }
  .pcx-charts { margin-top:18px; }
  .pcx-cg { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:700px){ .pcx-cg { grid-template-columns:1fr; } }
  .pcx-ct { font-size:.7rem; font-weight:800; text-transform:uppercase;
    letter-spacing:.07em; color:var(--muted,#64748B); margin-bottom:6px; }
  .pcx-note { color:var(--muted,#64748B); font-size:.74rem; line-height:1.5; }
  .pcx-warn { color:#b45309; font-size:.72rem; line-height:1.5; margin-top:6px; }
  .pcx .zone-card { border:0; box-shadow:none; padding:0; }
  .pcx .zone-area { position:relative; }
  .pcx .zone-area svg { width:100%; height:auto; display:block; }
  .pcx .zone-tip { display:none; position:absolute; z-index:10; pointer-events:none;
    background:#0F172A; color:#fff; padding:8px 10px; border-radius:8px;
    font-size:.74rem; line-height:1.45; min-width:150px; }
  .pcx .zone-legend { margin-top:8px; }
  .pcx .zone-legend .cap { color:var(--muted,#64748B); font-size:.7rem; margin-bottom:4px; }
  .pcx .zone-legend .bar { height:8px; border-radius:999px;
    background:linear-gradient(90deg,rgb(122,0,0),rgb(192,57,43),rgb(232,216,184),rgb(39,174,96),rgb(10,74,37)); }
  .pcx .zone-legend .lab { display:flex; justify-content:space-between;
    color:var(--dim,#94A3B8); font-size:.66rem; margin-top:3px; }`;

  function build() {
    if (built) return;
    const st = document.createElement("style");
    st.textContent = CSS;
    document.head.appendChild(st);

    const el = document.createElement("div");
    el.className = "pcx-overlay";
    el.id = "pcx-overlay";
    el.innerHTML = `<div class="pcx">
      <div class="pcx-close" id="pcx-close">✕</div>
      <div class="pcx-nm" id="pcx-nm"></div>
      <div class="pcx-tm" id="pcx-tm"></div>
      <div class="pcx-chips" id="pcx-chips"></div>
      <div class="pcx-stats">
        <div><div class="pcx-h">Box score</div><div id="pcx-box"></div></div>
        <div><div class="pcx-h">Shooting &amp; usage</div><div id="pcx-adv"></div></div>
      </div>
      <div class="pcx-charts">
        <div id="pcx-note" class="pcx-note"></div>
        <div class="pcx-cg" id="pcx-cg" style="display:none">
          <div><div class="pcx-ct" id="pcx-zt">Zone efficiency</div><div id="pcx-zone"></div></div>
          <div><div class="pcx-ct">Shot density</div><div id="pcx-dens"></div>
            <div id="pcx-dens-note" class="pcx-warn"></div></div>
        </div>
      </div></div>`;
    document.body.appendChild(el);
    el.addEventListener("click", e => { if (e.target === el) close(); });
    document.getElementById("pcx-close").addEventListener("click", close);
    document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
    built = true;
  }

  function close() {
    const el = document.getElementById("pcx-overlay");
    if (el) el.classList.remove("open");
    document.body.style.overflow = "";
    // Purge the Plotly instance rather than leaving it attached — reopening the
    // card rebuilds it, and orphaned plots leak both memory and event handlers.
    const d = document.getElementById("pcx-dens");
    if (d && window.Plotly) Plotly.purge(d);
    const z = document.getElementById("pcx-zone");
    if (z) z.innerHTML = "";
  }

  const n1 = v => (v == null ? "—" : Number(v).toFixed(1));
  const nInt = v => (v == null ? "—" : Number(v).toLocaleString());
  const pct1 = v => (v == null ? "—" : Number(v).toFixed(1) + "%");

  // ── Percentiles ─────────────────────────────────────────────────────────
  // Ranked against the FULL scope pool, not the filtered view, so the card
  // can never disagree with the row that was clicked.
  const LOWER_BETTER = new Set(["tov", "tov100", "tovp", "pf"]);
  const PCT_MIN_GP = 10, PCT_MIN_MPG = 12;
  const POOLS = {};
  // Every key the card badges. Keep in step with the columns build_site.py
  // exports — a key that isn't here simply gets no percentile badge.
  const PCT_KEYS = ["pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov",
                    "pf", "min",
                    "fg", "fg3", "ft", "efg", "ts", "par3", "ftr",
                    "usg", "tovp", "astp", "orbp", "drbp", "trbp", "stlp", "blkp",
                    "pts100", "reb100", "ast100",
                    "stl100", "blk100", "tov100", "stocks100"];

  function pools(season, scope, rows) {
    const key = `${season}|${scope}`;
    if (!POOLS[key] && typeof Percentile !== "undefined") {
      POOLS[key] = Percentile.buildPools(rows, PCT_KEYS,
        r => (r.gp ?? 0) >= PCT_MIN_GP && (r.min ?? 0) >= PCT_MIN_MPG);
    }
    return POOLS[key] || {};
  }
  let _pool = {};
  function badge(key, val) {
    if (typeof Percentile === "undefined" || !_pool[key] || val == null) return "";
    const p = Percentile.rank(_pool[key], val, !LOWER_BETTER.has(key));
    if (p == null) return "";
    const { bg, fg } = Percentile.color(p);
    return `<span class="pcx-pct" style="background:${bg};color:${fg === "inherit" ? "var(--muted)" : fg}">${Percentile.label(p)}</span>`;
  }

  const row = (label, shown, key, raw) =>
    `<div class="pcx-r"><span class="pcx-k">${label}</span>` +
    `<span class="pcx-v">${badge(key, raw)}${shown}</span></div>`;

  // ── Data ────────────────────────────────────────────────────────────────
  async function ensurePlayers(season, scope) {
    const key = `${season}|${scope}`;
    if (!PS[key]) {
      const sfx = (typeof scopeSuffix === "function") ? scopeSuffix(scope) : "";
      const d = await loadJSON(`data/${season}/player-stats${sfx}.json`);
      PS[key] = d && d.players
        ? Object.fromEntries(d.players.map(p => [p.id, p]))
        : {};
      POOLS[key] = null;
      if (d && d.players) pools(season, scope, d.players);
    }
    return PS[key];
  }

  // Shots are exported for "reg" and "playoffs" only — build_shots() is never
  // run for "all", so data/{season}/shots/{slug}-all.json does not exist. Left
  // unhandled that 404s and the card renders an empty shell on the All Games
  // tab, which reads as "this player has no shots" rather than "this scope has
  // no payload". Fall back to the regular season and say so instead.
  // SHOT_SCOPES is the set that has data; keep it in step with the scopes
  // build_shots() is actually run for.
  const SHOT_SCOPES = new Set(["reg", "playoffs"]);
  let _shotScope = "reg";   // the scope the last ensureShots() actually served

  async function ensureShots(season, scope, teamName, slug) {
    const s = slug || slugify(teamName || "");
    const eff = SHOT_SCOPES.has(scope) ? scope : "reg";
    const sfx = (typeof scopeSuffix === "function") ? scopeSuffix(eff) : "";
    const key = `${season}|${eff}|${s}`;
    if (!(key in SHOTS)) {
      SHOTS[key] = await loadJSON(`data/${season}/shots/${s}${sfx}.json`);
    }
    const ik = `${season}|${eff}`;
    if (!IDX[ik]) {
      IDX[ik] = await loadJSON(`data/${season}/shots-index${sfx}.json`);
    }
    _shotScope = eff;
    return SHOTS[key];
  }

  /** Open the card. opts: { season, scope='reg', name?, teamName?, slug? } */
  async function open(playerId, opts = {}) {
    build();
    const season = opts.season, scope = opts.scope || "reg";
    const overlay = document.getElementById("pcx-overlay");
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";

    document.getElementById("pcx-nm").textContent = opts.name || "…";
    document.getElementById("pcx-tm").textContent = opts.teamName || "";
    document.getElementById("pcx-chips").innerHTML = "";
    document.getElementById("pcx-box").innerHTML = "";
    document.getElementById("pcx-adv").innerHTML = "";
    document.getElementById("pcx-note").textContent = "Loading…";
    document.getElementById("pcx-cg").style.display = "none";
    document.getElementById("pcx-dens-note").textContent = "";

    const players = await ensurePlayers(season, scope);
    const p = players[playerId];
    _pool = pools(season, scope, Object.values(players));

    if (!p) {
      // A player can be missing from a scope she did not appear in — say so
      // instead of rendering an empty shell.
      document.getElementById("pcx-note").textContent =
        "This player has no data in the selected scope.";
      return;
    }

    document.getElementById("pcx-nm").textContent = p.n;
    document.getElementById("pcx-tm").textContent =
      [p.t, season, (typeof SCOPES !== "undefined"
        ? (SCOPES.find(s => s.key === scope) || {}).label : "")].filter(Boolean).join(" · ");

    document.getElementById("pcx-chips").innerHTML = [
      p.conf ? `<span class="pcx-chip">${p.conf}</span>` : "",
      p.gp != null ? `<span class="pcx-chip">${p.gp} GP</span>` : "",
      p.poss != null ? `<span class="pcx-chip">${Number(p.poss).toLocaleString()} poss</span>` : "",
    ].join("");

    document.getElementById("pcx-box").innerHTML = [
      row("Minutes", n1(p.min), "min", p.min),
      row("Points", n1(p.pts), "pts", p.pts),
      row("Rebounds", n1(p.reb), "reb", p.reb),
      row("· Offensive", n1(p.oreb), "oreb", p.oreb),
      row("· Defensive", n1(p.dreb), "dreb", p.dreb),
      row("Assists", n1(p.ast), "ast", p.ast),
      row("Steals", n1(p.stl), "stl", p.stl),
      row("Blocks", n1(p.blk), "blk", p.blk),
      row("Turnovers", n1(p.tov), "tov", p.tov),
      row("Fouls", n1(p.pf), "pf", p.pf),
    ].join("");

    // Shooting first, then the on-floor rates. These are PBP-derived against
    // the exact lineup rather than a minutes-share estimate — see the note on
    // the Player Stats page. row() tolerates a null value, so a season without
    // an advanced table simply drops the line rather than printing "NaN".
    document.getElementById("pcx-adv").innerHTML = [
      row("TS%", pct1(p.ts), "ts", p.ts),
      row("eFG%", pct1(p.efg), "efg", p.efg),
      row("FG%", pct1(p.fg), "fg", p.fg),
      row("3P%", pct1(p.fg3), "fg3", p.fg3),
      row("FT%", pct1(p.ft), "ft", p.ft),
      row("3PAr", pct1(p.par3), "par3", p.par3),
      row("FTr", pct1(p.ftr), "ftr", p.ftr),
      row("Usage", pct1(p.usg), "usg", p.usg),
      row("TOV%", pct1(p.tovp), "tovp", p.tovp),
      row("AST%", pct1(p.astp), "astp", p.astp),
      row("ORB%", pct1(p.orbp), "orbp", p.orbp),
      row("DRB%", pct1(p.drbp), "drbp", p.drbp),
      row("REB%", pct1(p.trbp), "trbp", p.trbp),
      row("STL%", pct1(p.stlp), "stlp", p.stlp),
      row("BLK%", pct1(p.blkp), "blkp", p.blkp),
      row("Points / 100", n1(p.pts100), "pts100", p.pts100),
      row("Rebounds / 100", n1(p.reb100), "reb100", p.reb100),
      row("Assists / 100", n1(p.ast100), "ast100", p.ast100),
      row("Stocks / 100", n1(p.stocks100), "stocks100", p.stocks100),
    ].join("");

    // ── Shot charts ───────────────────────────────────────────────────────
    const payload = await ensureShots(season, scope, p.t, p.slug);
    const note = document.getElementById("pcx-note");
    if (!payload || !payload.players) {
      note.textContent = "No shot data for this team and scope.";
      return;
    }
    // Join on PLAYER ID, never on name. The two feeds disagree on diacritics —
    // LeagueDashPlayerStats returns "Dorka Juhász" while shotchartdetail returns
    // "Dorka Juhasz" — so a name join silently drops every accented player
    // (5 of 157 in 2024 alone, and rising as the league internationalises).
    let pi = payload.player_ids ? payload.player_ids.indexOf(playerId) : -1;
    if (pi < 0) pi = payload.players.indexOf(p.n);   // pre-id payload fallback
    if (pi < 0) {
      note.textContent = "No shots recorded for this player in this scope.";
      return;
    }
    // player_zones is keyed by the SHOT feed's spelling, so index back through
    // the payload's own player list rather than reusing the stats-file name.
    const shotName = payload.players[pi];

    // Geometry from the payload, never from the season number — the arc moved
    // twice and this must match the classifier that produced the zone rollups.
    const zgeom = ZoneChart.geometry(payload.arc_r, payload.corner_x);
    const idx = IDX[`${season}|${_shotScope}`] || {};

    let zoneData = payload.player_zones ? payload.player_zones[shotName] : null;
    if (!zoneData) {
      zoneData = ZoneChart.computeZones(payload.shots, zgeom,
        (_g, playerIdx) => playerIdx === pi);
    }

    // Say which scope the shots are from whenever it isn't the one selected,
    // so a fallback chart is never mistaken for the scope you asked for.
    note.textContent = _shotScope === scope ? "" :
      `Shot charts are regular season only — there is no ` +
      `${(SCOPES.find(s => s.key === scope) || {}).label || scope} shot payload.`;
    document.getElementById("pcx-cg").style.display = "";
    document.getElementById("pcx-zt").textContent =
      `Zone efficiency · 3PT line ${payload.era} · ` +
      `${(SCOPES.find(s => s.key === _shotScope) || {}).label || _shotScope}`;

    ZoneChart.render(document.getElementById("pcx-zone"), zoneData, {
      geom: zgeom,
      zones: idx.zone_names,
      baselines: idx.league_baselines,
      minAtt: 4,
      league: "WNBA",
      compact: true,
    });

    const cgeom = Court.geometry(payload.arc_r, payload.corner_x);
    const { xs, ys } = Court.xyFor(payload.shots, pi);
    await Court.renderDensity(document.getElementById("pcx-dens"), xs, ys, cgeom, 300);

    // Through 2010 the feed logged rim attempts at (0,0) with no location.
    const share = Court.sentinelShare(xs, ys);
    document.getElementById("pcx-dens-note").textContent = share > 0.02
      ? `⚠ ${(share * 100).toFixed(0)}% of these shots have no recorded coordinate and sit on the rim — the feed did not capture locations for rim attempts until 2011.`
      : "";
  }

  return { open, close };
})();
