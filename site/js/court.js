/* ════════════════════════════════════════════════════════════════════════
   court.js — WNBA half-court geometry + the shot-density renderer.

   Ported from nba_charts/site/js/court.js. One copy, shared by every page that
   draws a court.

   Geometry is in FEET with the hoop at the origin, matching shotchartdetail's
   LOC_X/LOC_Y (tenths of a foot, hoop-centred) after a /10. It MUST agree with
   data/zones.py — and because the WNBA three-point line moved twice, the arc
   is a PARAMETER, not a constant:

       1997-2003   19' 9"      no straight corner (arc meets the baseline)
       2004-2012   20' 6.25"   no straight corner
       2013-       22' 1.75"   22 ft straight corners

   Call traces()/layout() with the geometry from the payload, exactly as
   zone-chart.js does, so the drawn court always matches the season shown.

   API:
     Court.geometry(arcR, cornerX)            → {r3, corner, junc, hasCorners} in FEET
     Court.traces(geom, lineColor)            → Plotly line traces for the court
     Court.layout(height)                     → Plotly layout for a half-court
     Court.renderDensity(el, xs, ys, geom)    → density heatmap (lazy Plotly)
     Court.xyFor(shots, playerIdx)            → one player's coords out of the
                                                flat [game,x,y,made,player] array
     Court.sentinelShare(xs, ys)              → fraction with no recorded coordinate
   ════════════════════════════════════════════════════════════════════════ */
const Court = (function () {
  "use strict";

  const BASE = -5.25;          // baseline, relative to the hoop

  // Lane geometry is era-INVARIANT, unlike the arc. Verified by measuring the
  // outer edge of the feed's "In The Paint (Non-RA)" zone season by season:
  // 8.0 ft half-width in every season from 1997 through 2024, so one constant
  // is correct for the whole history. (That measures the coordinate space the
  // shots are classified in, which is exactly what this court has to match.)
  const LANE_HALF = 8, FT_LINE = 13.75, FT_R = 6, RA_R = 4, RIM_R = 0.75;

  /** Payload arc/corner (tenths of a foot) → court geometry in FEET. */
  function geometry(arcR, cornerX) {
    const r3 = (Number(arcR) || 221.5) / 10;
    let corner = Number(cornerX) / 10;
    if (!isFinite(corner) || corner <= 0) corner = r3;
    const hasCorners = corner < r3;
    return { r3, corner: hasCorners ? corner : r3,
             junc: hasCorners ? Math.sqrt(r3 * r3 - corner * corner) : 0,
             hasCorners };
  }

  function arcPts(cx, cy, r, t0, t1, n = 80) {
    const x = [], y = [];
    for (let i = 0; i < n; i++) {
      const t = (t0 + (t1 - t0) * i / (n - 1)) * Math.PI / 180;
      x.push(cx + r * Math.cos(t)); y.push(cy + r * Math.sin(t));
    }
    return { x, y };
  }

  function traces(geom, lc = "rgba(15,23,42,0.35)") {
    const g = geom || geometry();
    const S = [], ln = (x0, y0, x1, y1) => S.push({ x: [x0, x1], y: [y0, y1] });
    ln(-25, BASE, 25, BASE); ln(-25, BASE, -25, 32); ln(25, BASE, 25, 32);
    ln(-3, -4.25, 3, -4.25);                        // backboard
    S.push(arcPts(0, 0, RIM_R, 0, 360));            // rim
    S.push(arcPts(0, 0, RA_R, 0, 180));             // restricted area
    ln(-LANE_HALF, BASE, -LANE_HALF, FT_LINE);
    ln(LANE_HALF, BASE, LANE_HALF, FT_LINE);
    ln(-LANE_HALF, FT_LINE, LANE_HALF, FT_LINE);    // paint
    S.push(arcPts(0, FT_LINE, FT_R, 0, 180));       // free-throw circle

    // Straight corner segments exist only once the line has them (2013+); on
    // the earlier courts the arc runs all the way DOWN INTO the baseline, which
    // sits below the hoop — so the arc ends at a negative angle, not at 0.
    // Stopping at 0 leaves the 19'9" and 20'6.25" lines floating a hoop's-depth
    // above the baseline with a visible gap at each corner.
    const ang = g.hasCorners
      ? Math.atan2(g.junc, g.corner) * 180 / Math.PI
      : -Math.asin(Math.min(1, Math.abs(BASE) / g.r3)) * 180 / Math.PI;
    if (g.hasCorners) {
      ln(-g.corner, BASE, -g.corner, g.junc);
      ln(g.corner, BASE, g.corner, g.junc);
    }
    S.push(arcPts(0, 0, g.r3, ang, 180 - ang));
    return S.map(s => ({ type: "scatter", x: s.x, y: s.y, mode: "lines",
                         line: { color: lc, width: 1.4 }, hoverinfo: "skip",
                         showlegend: false }));
  }

  function layout(height = 360) {
    return {
      paper_bgcolor: "#FFFFFF", plot_bgcolor: "#FFFFFF",
      height, margin: { l: 6, r: 6, t: 6, b: 6 },
      showlegend: false, dragmode: false,
      xaxis: { range: [-26, 26], visible: false, scaleanchor: "y", scaleratio: 1, constrain: "domain" },
      yaxis: { range: [-6, 32], visible: false, constrain: "domain" },
    };
  }

  /** Pull one player's shot coords out of the flat [game,x,y,made,player]
      array, converting tenths of a foot to feet. playerIdx < 0 = everyone. */
  function xyFor(shots, playerIdx = -1) {
    const xs = [], ys = [];
    for (let i = 0; i < shots.length; i += 5) {
      if (playerIdx >= 0 && shots[i + 4] !== playerIdx) continue;
      xs.push(shots[i + 1] / 10); ys.push(shots[i + 2] / 10);
    }
    return { xs, ys };
  }

  /* ── NO-COORDINATE SHOTS, and why the scale is clipped ──────────────────
     Through 2010 the WNBA feed recorded rim attempts (layups, tip-ins) with
     SHOT_DISTANCE 0 and LOC_X/LOC_Y of exactly (0,0) — it knew the shot was at
     the rim but never captured where. Measured across the cache: 30.0% of all
     1997 shots, 25.8% in 2002, 18.5% in 2010, then 0.00% from 2011 on. The
     cutover is abrupt and total.

     The damage is to the colour ramp, not the data. Plotly scales z linearly
     from 0 to the largest bin, so when one bin holds a third of a season's
     shots every other bin lands near the bottom of the ramp and the court
     reads as a single dot at the rim. Anchoring the ramp to a high percentile
     of the *occupied* bins keeps the rim saturated — it genuinely is the
     hottest spot — while giving the paint, mid-range and arc a real gradient.

     We do not invent coordinates for the (0,0) shots. Callers that want to
     caveat an early season ask sentinelShare() how many there were. */
  const XBINS = { start: -25, end: 25, size: 2 };
  const YBINS = { start: -5, end: 32, size: 2 };
  const CLIP = 0.98;           // percentile of occupied bins that saturates

  /** Count into the same grid Plotly will use, then take the CLIP percentile.
      Approximate by design — this only picks a colour ceiling. */
  function densityCeiling(xs, ys) {
    const nx = Math.ceil((XBINS.end - XBINS.start) / XBINS.size);
    const ny = Math.ceil((YBINS.end - YBINS.start) / YBINS.size);
    const counts = new Int32Array(nx * ny);
    for (let i = 0; i < xs.length; i++) {
      const ix = Math.floor((xs[i] - XBINS.start) / XBINS.size);
      const iy = Math.floor((ys[i] - YBINS.start) / YBINS.size);
      if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) counts[iy * nx + ix]++;
    }
    const occupied = Array.from(counts).filter(c => c > 0).sort((a, b) => a - b);
    if (!occupied.length) return 1;
    return Math.max(1, occupied[Math.min(occupied.length - 1,
                                         Math.floor(occupied.length * CLIP))]);
  }

  /** Fraction of shots sitting on the (0,0) "no coordinate captured" sentinel. */
  function sentinelShare(xs, ys) {
    if (!xs.length) return 0;
    let n = 0;
    for (let i = 0; i < xs.length; i++) if (xs[i] === 0 && ys[i] === 0) n++;
    return n / xs.length;
  }

  /** Density heatmap. Plotly is lazy-loaded (4.6MB) — only on first render. */
  async function renderDensity(el, xs, ys, geom, height = 360) {
    if (!xs.length) { el.innerHTML = ""; return false; }
    let Plotly;
    try { Plotly = await ensurePlotly(); } catch { return false; }
    const zmax = densityCeiling(xs, ys);
    Plotly.react(el, [
      // Plotly's built-in YlOrRd runs dark-red -> pale-yellow as z goes 0 -> 1,
      // the reverse of the usual heat-map reading. Without reversescale the
      // empty floor renders as the hottest colour and the rim as the coldest.
      { type: "histogram2dcontour", x: xs, y: ys, colorscale: "YlOrRd", reversescale: true,
        showscale: false, hoverinfo: "skip",
        // Explicit contour levels rather than ncontours: the bands have to line
        // up with the clipped ceiling, and anything above it folds into the top.
        contours: { coloring: "fill", showlines: false,
                    start: 0, end: zmax, size: zmax / 18 },
        xbins: XBINS, ybins: YBINS, zauto: false, zmin: 0, zmax },
      ...traces(geom, "rgba(15,23,42,0.35)"),
    ], layout(height), { displaylogo: false, staticPlot: true });
    return true;
  }

  return { geometry, traces, layout, renderDensity, xyFor, arcPts, sentinelShare };
})();

if (typeof module !== "undefined" && module.exports) module.exports = Court;
