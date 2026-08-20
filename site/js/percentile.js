/* ════════════════════════════════════════════════════════════════════════
   percentile.js — shared percentile machinery for every table on the site.

   Ranks a row's stat against the FULL qualified pool for the current scope, NOT
   the filtered view. That matters: if you filter to guards, a guard's TS%
   percentile shouldn't jump — it still means "vs everyone who qualifies". Same
   convention Cleaning the Glass uses.

   The pool definition (who qualifies, which keys) is page-specific and stays on
   the page. This module owns the reusable parts:

     Percentile.buildPools(rows, keys, qualifies) → { key: ascending values }
     Percentile.rank(pool, val, higherBetter)     → 0–100 (100 = best) or null
     Percentile.color(p)                          → { bg, fg }
     Percentile.label(p)                          → "94th"

   Colour is tuned for the LIGHT theme: the tint fades to nothing at the median
   and only deepens toward the extremes, so a full table doesn't turn into a
   traffic light. (Hog Charts' version returns dark-theme values — don't copy
   those in.)
   ════════════════════════════════════════════════════════════════════════ */
const Percentile = (function () {
  "use strict";

  // first index i with arr[i] >= x  (count of values strictly < x)
  function lowerBound(arr, x) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
    return lo;
  }
  // first index i with arr[i] > x
  function upperBound(arr, x) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
    return lo;
  }

  /** Ascending value array per key, from rows passing qualifies(row). */
  function buildPools(rows, keys, qualifies) {
    const q = qualifies || (() => true);
    const out = {};
    keys.forEach(k => {
      const vals = [];
      for (const r of (rows || [])) {
        if (!q(r)) continue;
        const x = r[k];
        if (x != null && !isNaN(x)) vals.push(+x);
      }
      vals.sort((a, b) => a - b);
      out[k] = vals;
    });
    return out;
  }

  /** 0–100 where 100 = best. higherBetter=false flips it for stats where low
      is good (turnovers, defensive rating, opponent shooting). */
  function rank(arr, val, higherBetter = true) {
    if (!arr || arr.length < 2 || val == null || isNaN(val)) return null;
    const n = arr.length;
    const below = lowerBound(arr, val);
    const above = n - upperBound(arr, val);
    return 100 * (higherBetter ? below : above) / (n - 1);
  }

  /** Diverging tint: red (0) → transparent (50) → green (100).
      Alpha scales with distance from the median so mid values stay clean. */
  function color(p) {
    if (p == null) return { bg: "transparent", fg: "inherit" };
    const hue = (p / 100) * 130;                 // 0 = red, 130 = green
    const a = Math.abs(p - 50) / 50 * 0.30;      // 0 at median → .30 at extremes
    return {
      bg: `hsla(${hue.toFixed(0)},68%,45%,${a.toFixed(3)})`,
      fg: Math.abs(p - 50) > 34 ? `hsl(${hue.toFixed(0)},85%,24%)` : "inherit",
    };
  }

  const SUF = p => {
    const n = Math.round(p);
    if (n % 100 >= 11 && n % 100 <= 13) return "th";
    return ({ 1: "st", 2: "nd", 3: "rd" })[n % 10] || "th";
  };
  /** "94th" — for the percentile display mode and the player card. */
  function label(p) {
    if (p == null) return "—";
    const n = Math.round(p);
    return `${n}${SUF(n)}`;
  }

  return { buildPools, rank, color, label };
})();
