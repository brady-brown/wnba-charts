/* ════════════════════════════════════════════════════════════════════════
   zone-chart.js — shared 14-wedge hot-zone shot chart, ERA-AWARE.

   Ported from nba_charts/site/js/zone-chart.js. The NBA version hardcodes one
   court (23.75ft arc / 22ft corners) at module scope. The WNBA moved its
   three-point line twice, so geometry is a PARAMETER here:

       1997-2003   19' 9"      arc 197.5, no straight corner
       2004-2012   20' 6.25"   arc 205.2, no straight corner
       2013-       22' 1.75"   arc 221.5, corner 220.0

   ⚠ Boundaries MUST stay identical to data/zones.py classify(). That file is
   the source of truth; this is the browser mirror. Change an arc in one and
   change it in the other.

   The arc for a season is READ FROM THE PAYLOAD (arc_r / corner_x written by
   data/shots.py compact_payload), not guessed from the season number, so the
   page cannot drift from the classifier that produced the numbers.

   Label positions are COMPUTED from the geometry rather than hand-tuned. The
   NBA file carries a warning that every label must classify into its own zone;
   with three different arcs that is impossible to maintain by hand, so
   labelFor() derives each one and verifyLabels() checks the invariant.

   API:
     ZoneChart.geometry(arcR, cornerX)             → geometry object
     ZoneChart.classifyZone(x, y, geom)            → 0..13 or -1  (tenths of a foot)
     ZoneChart.computeZones(shots, geom, keepFn)   → [[zi, makes, atts], …]
     ZoneChart.render(el, zoneData, opts)          → draws SVG
        opts = { geom, zones, baselines, title, subtitle, minAtt, compact, league }
     ZoneChart.verifyLabels(geom)                  → [] when every label is in its zone
   ════════════════════════════════════════════════════════════════════════ */
const ZoneChart = (function () {
  "use strict";

  const ZONE_NAMES = [
    "Restricted Area",
    "Close Mid - Right", "Close Mid - Center", "Close Mid - Left",
    "Mid - Right", "Mid - Right Center", "Mid - Center",
    "Mid - Left Center", "Mid - Left",
    "3PT - Right", "3PT - Right Center", "3PT - Center",
    "3PT - Left Center", "3PT - Left",
  ];

  // Ring radii are era-invariant; only the arc moves.
  const R_RA = 40, R_CLOSE = 110, HEAVE = 400, Y_BASE = -52.5, EDGE = 500;

  // Default: the modern WNBA court, used when a payload omits its geometry.
  function geometry(arcR, cornerX) {
    const R3 = Number(arcR) || 221.5;
    // A corner at or beyond the arc means the line is a pure circle meeting the
    // baseline — the pre-2013 courts. Y_MEET collapses to 0 and the corner test
    // never fires on its own, which is exactly the degenerate case we want.
    let CX = Number(cornerX);
    if (!isFinite(CX) || CX <= 0) CX = R3;
    const hasCorners = CX < R3;
    const YMEET = hasCorners ? Math.sqrt(R3 * R3 - CX * CX) : 0;
    return { R3, CX: hasCorners ? CX : R3, YMEET, hasCorners,
             R_RA, R_CLOSE, Y_MAX: Math.round(R3 + 48) };
  }

  // ── classification (mirrors data/zones.py classify) ──────────────────────
  function classifyZone(x, y, g) {
    g = g || geometry();
    const dist = Math.hypot(x, y);
    if (dist >= HEAVE) return -1;
    let ang = Math.atan2(y, x) * 180 / Math.PI;
    if (ang < -90) ang += 360;
    if (dist < g.R_RA) return 0;

    let is3 = dist >= g.R3;
    if (g.hasCorners) is3 = is3 || (Math.abs(x) >= g.CX && y <= g.YMEET);
    if (is3) {
      if (ang < 36) return 9;
      if (ang < 72) return 10;
      if (ang < 108) return 11;
      if (ang < 144) return 12;
      return 13;
    }
    if (dist < g.R_CLOSE) { if (ang < 60) return 1; if (ang < 120) return 2; return 3; }
    if (ang < 36) return 4;
    if (ang < 72) return 5;
    if (ang < 108) return 6;
    if (ang < 144) return 7;
    return 8;
  }

  // Flat [game, x, y, made, player] quintuples → [[zoneIndex, makes, atts], …]
  function computeZones(shots, g, keepFn) {
    const makes = new Array(14).fill(0), atts = new Array(14).fill(0);
    for (let i = 0; i < shots.length; i += 5) {
      if (keepFn && !keepFn(shots[i], shots[i + 4])) continue;
      const zi = classifyZone(shots[i + 1], shots[i + 2], g);
      if (zi < 0) continue;
      atts[zi]++;
      if (shots[i + 3]) makes[zi]++;
    }
    const out = [];
    for (let zi = 0; zi < 14; zi++) if (atts[zi] > 0) out.push([zi, makes[zi], atts[zi]]);
    return out;
  }

  // ── colour: FG% relative to the zone's league baseline ───────────────────
  // Baselines are passed in per season (data/shots.py league_baselines). The
  // fallbacks below only apply when a payload ships none — WNBA-scaled, not
  // the NBA numbers the parent file used.
  function familyBaseline(name) {
    if (name === "Restricted Area") return 0.60;
    if (name.startsWith("Close Mid")) return 0.40;
    if (name.startsWith("3PT")) return 0.33;
    return 0.39;
  }
  function baselineFor(name, zones, baselines) {
    if (zones && baselines) {
      const i = zones.indexOf(name);
      if (i >= 0 && baselines[i] != null && !isNaN(baselines[i])) return baselines[i];
    }
    return familyBaseline(name);
  }
  const SPAN = 0.15;
  function colorRel(pct, base) {
    const d = Math.max(-SPAN, Math.min(SPAN, pct - base));
    const t = (d + SPAN) / (2 * SPAN);
    const stops = [[0, [122, 0, 0]], [0.25, [192, 57, 43]], [0.5, [232, 216, 184]],
                   [0.75, [39, 174, 96]], [1, [10, 74, 37]]];
    for (let i = 0; i < stops.length - 1; i++) {
      if (t <= stops[i + 1][0]) {
        const f = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
        const a = stops[i][1], b = stops[i + 1][1];
        return `rgb(${Math.round(a[0] + f * (b[0] - a[0]))},${Math.round(a[1] + f * (b[1] - a[1]))},${Math.round(a[2] + f * (b[2] - a[2]))})`;
      }
    }
    return `rgb(${stops[stops.length - 1][1].join(",")})`;
  }

  // ── SVG helpers (tenths of a foot, hoop at the polar origin) ─────────────
  const NS = "http://www.w3.org/2000/svg";
  const PAD = 10, Y_MIN = -58;
  const polar = (r, d) => [r * Math.cos(d * Math.PI / 180), r * Math.sin(d * Math.PI / 180)];

  function frame(g) {
    const VB_W = 500 + 2 * PAD, VB_H = (g.Y_MAX - Y_MIN) + 2 * PAD;
    const cx = x => x + 250 + PAD, cy = y => (g.Y_MAX - y) + PAD;
    const pt = (x, y) => `${cx(x).toFixed(2)},${cy(y).toFixed(2)}`;
    const arc = (r, a1, a2) => {
      const sweep = a2 > a1 ? 0 : 1, large = Math.abs(a2 - a1) > 180 ? 1 : 0;
      const p = polar(r, a2);
      return `A${r.toFixed(2)},${r.toFixed(2)} 0 ${large},${sweep} ${pt(p[0], p[1])}`;
    };
    return { VB_W, VB_H, cx, cy, pt, arc,
             Mv: (x, y) => `M${pt(x, y)}`, Lv: (x, y) => `L${pt(x, y)}` };
  }

  // Wedge path builders, parameterised by geometry.
  function buildZGEO(g, f) {
    const { Mv, Lv, arc } = f;
    const R3 = g.R3, CX = g.CX, YM = g.YMEET;
    const cornerAng = Math.atan2(YM, CX) * 180 / Math.PI;

    return {
      "Restricted Area": () => Mv(-R_RA, Y_BASE) + Lv(-R_RA, 0) + arc(R_RA, 180, 0) + Lv(R_RA, Y_BASE) + " Z",
      "Close Mid - Right": () => { const p = polar(R_RA, 60); return Mv(R_RA, Y_BASE) + Lv(R_CLOSE, Y_BASE) + Lv(R_CLOSE, 0) + arc(R_CLOSE, 0, 60) + Lv(p[0], p[1]) + arc(R_RA, 60, 0) + " Z"; },
      "Close Mid - Center": () => { const p1 = polar(R_CLOSE, 60), p2 = polar(R_RA, 120), s = polar(R_RA, 60); return Mv(s[0], s[1]) + Lv(p1[0], p1[1]) + arc(R_CLOSE, 60, 120) + Lv(p2[0], p2[1]) + arc(R_RA, 120, 60) + " Z"; },
      "Close Mid - Left": () => { const p1 = polar(R_CLOSE, 120), p2 = polar(R_RA, 120); return Mv(p2[0], p2[1]) + Lv(p1[0], p1[1]) + arc(R_CLOSE, 120, 180) + Lv(-R_CLOSE, Y_BASE) + Lv(-R_RA, Y_BASE) + Lv(-R_RA, 0) + arc(R_RA, 180, 120) + " Z"; },
      "Mid - Right": () => { const p36 = polar(R_CLOSE, 36); return Mv(R_CLOSE, Y_BASE) + Lv(CX, Y_BASE) + Lv(CX, YM) + arc(R3, cornerAng, 36) + Lv(p36[0], p36[1]) + arc(R_CLOSE, 36, 0) + " Z"; },
      "Mid - Right Center": () => { const p1 = polar(R_CLOSE, 36), p2 = polar(R3, 36), p72 = polar(R_CLOSE, 72); return Mv(p1[0], p1[1]) + Lv(p2[0], p2[1]) + arc(R3, 36, 72) + Lv(p72[0], p72[1]) + arc(R_CLOSE, 72, 36) + " Z"; },
      "Mid - Center": () => { const p1 = polar(R_CLOSE, 72), p2 = polar(R3, 72), p108 = polar(R_CLOSE, 108); return Mv(p1[0], p1[1]) + Lv(p2[0], p2[1]) + arc(R3, 72, 108) + Lv(p108[0], p108[1]) + arc(R_CLOSE, 108, 72) + " Z"; },
      "Mid - Left Center": () => { const p1 = polar(R_CLOSE, 108), p2 = polar(R3, 108), p144 = polar(R_CLOSE, 144); return Mv(p1[0], p1[1]) + Lv(p2[0], p2[1]) + arc(R3, 108, 144) + Lv(p144[0], p144[1]) + arc(R_CLOSE, 144, 108) + " Z"; },
      "Mid - Left": () => { const p1 = polar(R_CLOSE, 144), p2 = polar(R3, 144); return Mv(p1[0], p1[1]) + Lv(p2[0], p2[1]) + arc(R3, 144, 180 - cornerAng) + Lv(-CX, YM) + Lv(-CX, Y_BASE) + Lv(-R_CLOSE, Y_BASE) + Lv(-R_CLOSE, 0) + arc(R_CLOSE, 180, 144) + " Z"; },
      "3PT - Right": () => { const o36 = polar(EDGE, 36), i36 = polar(R3, 36); return Mv(CX, Y_BASE) + Lv(EDGE, Y_BASE) + Lv(EDGE, EDGE) + Lv(o36[0], o36[1]) + Lv(i36[0], i36[1]) + arc(R3, 36, cornerAng) + Lv(CX, YM) + " Z"; },
      "3PT - Right Center": () => { const p1 = polar(R3, 36), o = polar(EDGE, 36), p2 = polar(R3, 72); return Mv(p1[0], p1[1]) + Lv(o[0], o[1]) + arc(EDGE, 36, 72) + Lv(p2[0], p2[1]) + arc(R3, 72, 36) + " Z"; },
      "3PT - Center": () => { const p1 = polar(R3, 72), o = polar(EDGE, 72), p2 = polar(R3, 108); return Mv(p1[0], p1[1]) + Lv(o[0], o[1]) + arc(EDGE, 72, 108) + Lv(p2[0], p2[1]) + arc(R3, 108, 72) + " Z"; },
      "3PT - Left Center": () => { const p1 = polar(R3, 108), o = polar(EDGE, 108), p2 = polar(R3, 144); return Mv(p1[0], p1[1]) + Lv(o[0], o[1]) + arc(EDGE, 108, 144) + Lv(p2[0], p2[1]) + arc(R3, 144, 108) + " Z"; },
      "3PT - Left": () => { const o = polar(EDGE, 144); return Mv(-CX, Y_BASE) + Lv(-CX, YM) + arc(R3, 180 - cornerAng, 144) + Lv(o[0], o[1]) + Lv(-EDGE, EDGE) + Lv(-EDGE, Y_BASE) + " Z"; },
    };
  }

  // ── labels, derived from geometry ────────────────────────────────────────
  // The NBA file hardcodes these and warns they must classify into their own
  // zone. Three arcs make that unmaintainable by hand, so each label is placed
  // at the angular midpoint and a radius inside its ring.
  function labelFor(name, g) {
    const mid = (a, b) => (a + b) / 2;
    const at = (r, a) => polar(r, a);
    const rMid = (r1, r2) => r1 + (r2 - r1) * 0.5;
    const R3 = g.R3;

    if (name === "Restricted Area") return [0, 5];
    if (name === "Close Mid - Right")  return at(rMid(R_RA, R_CLOSE), 30);
    if (name === "Close Mid - Center") return at(rMid(R_RA, R_CLOSE), 90);
    if (name === "Close Mid - Left")   return at(rMid(R_RA, R_CLOSE), 150);
    if (name === "Mid - Right")        return at(rMid(R_CLOSE, R3), 18);
    if (name === "Mid - Right Center") return at(rMid(R_CLOSE, R3), mid(36, 72));
    if (name === "Mid - Center")       return at(rMid(R_CLOSE, R3), 90);
    if (name === "Mid - Left Center")  return at(rMid(R_CLOSE, R3), mid(108, 144));
    if (name === "Mid - Left")         return at(rMid(R_CLOSE, R3), 162);
    // Threes sit just outside the arc; the corners hug the baseline.
    if (name === "3PT - Right")        return [g.hasCorners ? (g.CX + 18) : (R3 + 14), 6];
    if (name === "3PT - Left")         return [-(g.hasCorners ? (g.CX + 18) : (R3 + 14)), 6];
    if (name === "3PT - Right Center") return at(R3 + 26, mid(36, 72));
    if (name === "3PT - Center")       return at(R3 + 30, 90);
    if (name === "3PT - Left Center")  return at(R3 + 26, mid(108, 144));
    return [0, 0];
  }

  /** Dev check: every label must land in the zone it names. Returns offenders. */
  function verifyLabels(g) {
    g = g || geometry();
    const bad = [];
    ZONE_NAMES.forEach((name, i) => {
      const [x, y] = labelFor(name, g);
      const got = classifyZone(x, y, g);
      if (got !== i) bad.push({ zone: name, expected: i, got, at: [Math.round(x), Math.round(y)] });
    });
    return bad;
  }

  function courtLines(g, f) {
    const s = "rgba(15,23,42,0.30)", w = 1.2;
    const { cx, cy } = f;
    const tm = Math.atan2(g.YMEET, g.CX) * 180 / Math.PI;
    const a = polar(g.R3, tm), b = polar(g.R3, 180 - tm);
    const out = [
      `<line x1="${cx(-250)}" y1="${cy(Y_BASE)}" x2="${cx(250)}" y2="${cy(Y_BASE)}" stroke="${s}" stroke-width="${w}"/>`,
      `<line x1="${cx(-250)}" y1="${cy(Y_BASE)}" x2="${cx(-250)}" y2="${cy(g.Y_MAX - 4)}" stroke="${s}" stroke-width="${w}"/>`,
      `<line x1="${cx(250)}" y1="${cy(Y_BASE)}" x2="${cx(250)}" y2="${cy(g.Y_MAX - 4)}" stroke="${s}" stroke-width="${w}"/>`,
      `<path d="M${cx(a[0]).toFixed(2)},${cy(a[1]).toFixed(2)} A${g.R3},${g.R3} 0 0,0 ${cx(b[0]).toFixed(2)},${cy(b[1]).toFixed(2)}" stroke="${s}" stroke-width="${w}" fill="none"/>`,
    ];
    // Straight corner segments only exist on the modern court.
    if (g.hasCorners) {
      out.push(`<line x1="${cx(-g.CX)}" y1="${cy(Y_BASE)}" x2="${cx(-g.CX)}" y2="${cy(g.YMEET)}" stroke="${s}" stroke-width="${w}"/>`);
      out.push(`<line x1="${cx(g.CX)}" y1="${cy(Y_BASE)}" x2="${cx(g.CX)}" y2="${cy(g.YMEET)}" stroke="${s}" stroke-width="${w}"/>`);
    }
    return out.join("");
  }

  // ── render ───────────────────────────────────────────────────────────────
  function render(container, zoneData, opts) {
    opts = opts || {};
    const g = opts.geom || geometry();
    const f = frame(g);
    const zones = opts.zones || ZONE_NAMES;
    const baselines = opts.baselines || null;
    const minAtt = opts.minAtt || 5;
    const league = opts.league || "WNBA";
    const ZGEO = buildZGEO(g, f);

    const stat = {};
    for (const [zi, m, a] of zoneData) stat[zones[zi]] = { m, a, pct: a > 0 ? m / a : 0 };

    let polys = "", labels = "";
    for (const [name, fn] of Object.entries(ZGEO)) {
      const z = stat[name], base = baselineFor(name, zones, baselines);
      let fill = "#F1F5F9", low = false, stroke = "rgba(255,255,255,0.95)", sw = 1.4;
      if (z && z.a > 0) {
        if (z.a < minAtt) { fill = "url(#zhatch)"; low = true; stroke = colorRel(z.pct, base); sw = 2; }
        else fill = colorRel(z.pct, base);
      }
      polys += `<path class="zone-poly" d="${fn()}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" data-zone="${name}"></path>`;
      if (z && z.a > 0) {
        const lp = labelFor(name, g);
        let col = "#fff";
        if (!low) {
          const mm = fill.match(/\d+/g);
          if (mm) col = ((0.299 * mm[0] + 0.587 * mm[1] + 0.114 * mm[2]) / 255) > 0.6 ? "#15181f" : "#fff";
        }
        labels += `<text x="${f.cx(lp[0]).toFixed(2)}" y="${f.cy(lp[1]).toFixed(2)}" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="700" fill="${col}" pointer-events="none">${(z.pct * 100).toFixed(0)}%</text>`;
      }
    }

    const totA = zoneData.reduce((s, [, , a]) => s + a, 0);
    const head = opts.title
      ? `<div class="zone-head"><div class="zone-title">${opts.title}</div><div class="zone-sub">${opts.subtitle || ""}</div></div>`
      : "";
    container.innerHTML = `
      <div class="zone-card${opts.compact ? " zone-compact" : ""}">
        ${head}
        <div class="zone-area">
          <svg viewBox="0 0 ${f.VB_W} ${f.VB_H}" xmlns="${NS}">
            <defs><pattern id="zhatch" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(45)">
              <rect width="7" height="7" fill="#E9EDF3"/><line x1="0" y1="0" x2="0" y2="7" stroke="rgba(15,23,42,0.22)" stroke-width="2"/>
            </pattern></defs>
            <rect x="0" y="0" width="${f.VB_W}" height="${f.VB_H}" fill="#FFFFFF"/>
            ${polys}${labels}${courtLines(g, f)}
          </svg>
          <div class="zone-tip"></div>
        </div>
        <div class="zone-legend">
          <div class="cap">FG% vs ${league} average · ${totA} FGA · hatched = under ${minAtt} FGA</div>
          <div class="bar"></div>
          <div class="lab"><span>−15%</span><span>−7%</span><span>avg</span><span>+7%</span><span>+15%</span></div>
        </div>
      </div>`;

    const area = container.querySelector(".zone-area");
    const tip = container.querySelector(".zone-tip");
    area.querySelectorAll(".zone-poly").forEach(path => {
      const name = path.getAttribute("data-zone"), z = stat[name];
      if (!z || z.a === 0) return;
      const base = baselineFor(name, zones, baselines), rel = (z.pct - base) * 100;
      path.addEventListener("mousemove", e => {
        const r = area.getBoundingClientRect();
        tip.innerHTML =
          `<div class="t">${name}</div>` +
          `<div class="s">${z.m}/${z.a} FG</div>` +
          `<div class="p" style="color:${colorRel(z.pct, base)}">${(z.pct * 100).toFixed(1)}% FG</div>` +
          `<div class="r" style="color:${rel >= 0 ? "#7fd8a0" : "#e08a82"}">${rel >= 0 ? "+" : ""}${rel.toFixed(1)}% vs ${league} avg (${(base * 100).toFixed(0)}%)</div>` +
          (z.a < minAtt ? `<div class="r" style="color:#b45309">⚠ Low sample</div>` : "");
        tip.style.left = Math.min(e.clientX - r.left + 12, r.width - 150) + "px";
        tip.style.top = Math.max(e.clientY - r.top - 12, 0) + "px";
        tip.style.display = "block";
      });
      path.addEventListener("mouseleave", () => tip.style.display = "none");
    });
  }

  return { geometry, classifyZone, computeZones, render, verifyLabels,
           labelFor, ZONE_NAMES };
})();

if (typeof module !== "undefined" && module.exports) module.exports = ZoneChart;
