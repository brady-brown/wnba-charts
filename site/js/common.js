// Shared utilities for the WNBA Charts site.
// Ported from nba_charts/site/js/common.js — the differences are the season key
// format (single year, not "2024-25"), the nav, and the footer attribution.

// Team/player name → URL slug. MUST stay identical to the Python slugify() in
// data/build_common.py — per-team JSON filenames depend on it.
function slugify(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// WNBA season keys are single years ("2024") and need no transformation. The
// NBA site's seasonLabel exists for the same reason and does the same nothing —
// kept so page code stays portable between the two sites.
function seasonLabel(s) { return String(s); }

// Scope keys → labels. Mirrors SCOPES in data/build_common.py; the filename
// suffix convention ("" / "-playoffs" / "-all") must match scope_filename().
const SCOPES = [
  { key: "reg",      label: "Regular Season", suffix: "" },
  { key: "playoffs", label: "Playoffs",       suffix: "-playoffs" },
  { key: "all",      label: "All Games",      suffix: "-all" },
];
const scopeSuffix = (key) => (SCOPES.find(s => s.key === key) || SCOPES[0]).suffix;

// fetch() wrapper — returns parsed JSON or null on error
async function loadJSON(path) {
  try {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  } catch (e) {
    console.error("loadJSON failed:", path, e);
    return null;
  }
}

// Lazy-load Plotly (4.6MB) on first use instead of eagerly in every <head>.
// Only the shot-density chart needs it, and only once that view is opened.
// Returns a promise resolving to window.Plotly; caches so it loads once.
let _plotlyPromise = null;
function ensurePlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (_plotlyPromise) return _plotlyPromise;
  _plotlyPromise = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    s.charset = "utf-8";
    s.onload = () => resolve(window.Plotly);
    s.onerror = () => { _plotlyPromise = null; reject(new Error("Plotly failed to load")); };
    document.head.appendChild(s);
  });
  return _plotlyPromise;
}

// Show an error banner inside any element
function showError(el, msg) {
  if (typeof el === "string") el = document.getElementById(el);
  if (!el) return;
  el.innerHTML = `<div style="color:var(--t1);padding:24px;font-weight:700">${msg}</div>`;
}

// Sign-aware formatting: "+3.2" / "-1.0"
function fmt(v, decimals = 1) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (!isFinite(n)) return "—";
  const s = Math.abs(n).toFixed(decimals);
  return n >= 0 ? `+${s}` : `-${s}`;
}
function fmtPlain(v, decimals = 1) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return isFinite(n) ? n.toFixed(decimals) : "—";
}
function fmtInt(v) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return isFinite(n) ? n.toLocaleString() : "—";
}

// ── Site navigation (single source of truth) ────────────────────────────────
// Adding a page means editing this array only — each page ships an empty <nav>
// that populateNav() fills.
const NAV_LINKS = [
  { href: "index.html",        label: "Home" },
  { href: "player-stats.html", label: "Player Stats" },
  { href: "shot-charts.html",  label: "Shot Charts" },
];

function populateNav() {
  const nav = document.querySelector("nav");
  if (!nav || nav.querySelector("a[data-page]")) return;   // absent or already built
  const brand = nav.querySelector(".nav-brand");
  const frag = document.createDocumentFragment();
  NAV_LINKS.forEach(link => {
    const a = document.createElement("a");
    a.href = link.href;
    a.dataset.page = link.page || link.href;
    a.textContent = link.label;
    frag.appendChild(a);
  });
  if (brand && brand.nextSibling) nav.insertBefore(frag, brand.nextSibling);
  else nav.appendChild(frag);
}

function markActiveNav() {
  const page = location.pathname.replace(/\/$/, "").split("/").pop() || "index.html";
  document.querySelectorAll("nav a[data-page]").forEach(a => {
    if (a.dataset.page.split(/\s+/).includes(page)) a.classList.add("active");
  });
}

// Responsive nav with a hamburger toggle on phones.
function buildResponsiveNav() {
  const nav = document.querySelector("nav");
  if (!nav || nav.querySelector(".nav-toggle")) return;
  const links = [...nav.querySelectorAll("a:not(.nav-brand)")];
  if (!links.length) return;

  const panel = document.createElement("div");
  panel.className = "nav-links";
  links.forEach(a => panel.appendChild(a));

  const toggle = document.createElement("button");
  toggle.className = "nav-toggle";
  toggle.type = "button";
  toggle.setAttribute("aria-label", "Toggle menu");
  toggle.setAttribute("aria-expanded", "false");
  toggle.innerHTML = "<span></span><span></span><span></span>";

  nav.appendChild(toggle);
  nav.appendChild(panel);

  const setOpen = (open) => {
    nav.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
  };
  toggle.addEventListener("click", () => setOpen(!nav.classList.contains("open")));
  panel.addEventListener("click", e => { if (e.target.closest("a")) setOpen(false); });
  window.matchMedia("(min-width: 701px)").addEventListener("change", e => {
    if (e.matches) setOpen(false);
  });
}

// ── Season picker ───────────────────────────────────────────────────────────
// Every page needs the same dropdown off the same seasons.json, so it lives
// here rather than being re-implemented per page.
async function loadSeasons() {
  const seasons = await loadJSON("data/seasons.json");
  return Array.isArray(seasons) ? seasons : [];
}

function fillSeasonSelect(select, seasons, selected) {
  if (!select) return;
  select.innerHTML = seasons
    .map(s => `<option value="${s}"${s === selected ? " selected" : ""}>${seasonLabel(s)}</option>`)
    .join("");
}

// ── Site footer (single source of truth) ────────────────────────────────────
// Injected on every page so the disclaimer/attribution stays in one place.
const FOOTER_HTML = `
  <strong>WNBA Charts</strong> is an independent, non-commercial project. It is not
  affiliated with, endorsed by, or sponsored by the Women's National Basketball
  Association or any of its teams. Raw play-by-play data is sourced from
  <a href="https://www.wnba.com" rel="nofollow noopener" target="_blank">WNBA.com</a>;
  every rating, metric, and chart on this site is computed from those facts by our
  own analytics engine. &ldquo;WNBA&rdquo; and all team names are trademarks of their
  respective owners.`;

function populateFooter() {
  if (document.querySelector("footer.site-footer")) return;
  const st = document.createElement("style");
  st.textContent = `
    footer.site-footer { max-width: 1100px; margin: 48px auto 0; padding: 20px 20px 32px;
      border-top: 1px solid var(--line, #e5e7eb); color: var(--muted, #6b7280);
      font-size: .74rem; line-height: 1.55; }
    footer.site-footer a { color: inherit; text-decoration: underline; }`;
  document.head.appendChild(st);
  const f = document.createElement("footer");
  f.className = "site-footer";
  f.innerHTML = FOOTER_HTML;
  document.body.appendChild(f);
}

document.addEventListener("DOMContentLoaded", () => {
  populateNav();          // must run before the responsive wrap + active mark
  buildResponsiveNav();
  markActiveNav();
  populateFooter();
});
