// FinSight results page: the graded track record from GET /results (no AI involved).
// All text is inserted with textContent, never parsed as HTML.
"use strict";

const API = document.documentElement.dataset.api || "http://localhost:8000";
const $ = (id) => document.getElementById(id);

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "style") Object.assign(el.style, v);  // CSSOM, which the CSP allows (style attributes it refuses)
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
  return el;
}

const pct = (rate) => (rate * 100).toFixed(1) + "%";
const num = (n) => n.toLocaleString("en-US");
const day = (iso) => new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
const when = (iso) => new Date(iso).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
const TF_NAMES = { "1H": "1 hour", "4H": "4 hours", "1D": "1 day", "1W": "1 week" };

function panel(cls, title, intro, ...body) {
  return h("section", { class: `panel ${cls}` }, h("h2", { class: "sec" }, title), intro && h("p", { class: "rs-intro" }, intro), ...body);
}

// One row of a bar list. Every bar shares the same 0..`max` scale, and starts at zero.
function bar(item, max, cls = "") {
  const share = item.rate == null ? 0 : item.rate / max;
  return h("li", { class: `rs-bar ${cls} ${item.trusted ? "" : "few"}`.trim() },
    h("span", { class: "rs-name" }, item.name),
    h("span", { class: "rs-track", "aria-hidden": "true" }, h("span", { class: "rs-fill", style: { width: `${(share * 100).toFixed(1)}%` } })),
    h("span", { class: "rs-val" }, item.trusted ? pct(item.rate) : "too few yet",
      h("small", {}, `${num(item.hits)} of ${num(item.n)}`)));
}

function barList(items, max, highlight) {
  return h("ol", { class: "rs-bars" }, items.map((it) => bar(it, max, it === highlight ? "lead" : "")));
}

const scaleMax = (items) => Math.min(1, Math.ceil((Math.max(...items.map((i) => i.rate || 0)) + 0.05) * 10) / 10);

function headline(r) {
  const trend = r.baselines.find((b) => b.name === "Follow the recent trend");
  const diff = r.finsight.hits - trend.hits, close = Math.max(3, r.n * 0.01);
  const vs = `${num(r.finsight.hits)} correct views against ${num(trend.hits)}`;
  if (Math.abs(diff) <= close) return `About level with simply following the recent trend (${vs}). So far FinSight adds nothing the price chart doesn't already say.`;
  return diff > 0 ? `Ahead of simply following the recent trend (${vs}).`
                  : `Behind simply following the recent trend (${vs}).`;
}

function scoreboard(r) {
  const finsight = { name: "FinSight", ...r.finsight };
  const items = [finsight, ...r.baselines].sort((a, b) => b.rate - a.rate);
  return panel("rs-board", "Against simple rules",
    `The same ${num(r.n)} views, graded the same way. If FinSight can't beat these, it isn't adding anything.`,
    barList(items, scaleMax(items), finsight),
    h("p", { class: "rs-verdict" }, headline(r)));
}

function buckets(r) {
  const verdict = {
    true: "Yes: views with a higher score have been right more often.",
    false: "Not yet. Higher scores haven't been right more often, so read the score as how much the evidence agrees, not as how likely the view is to be right.",
    null: `Too early to tell: each score level needs at least ${r.min_sample} graded views.`,
  }[String(r.score_is_meaningful)];
  return panel("rs-buckets", "Does a higher score mean a better view?",
    "Each view comes with an evidence score: Low, Medium or High.",
    barList(r.buckets, scaleMax(r.buckets)),
    h("p", { class: "rs-verdict" }, verdict));
}

function table(title, intro, items, label = (x) => x) {
  return panel("rs-split", title, intro, h("ul", { class: "rs-rows" }, items.map((it) =>
    h("li", { class: it.trusted ? "" : "few" },
      h("span", {}, label(it.name)),
      h("span", { class: "rs-val" }, it.trusted ? pct(it.rate) : "too few yet", h("small", {}, `${num(it.hits)} of ${num(it.n)}`))))));
}

function regime(r) {
  const { up, down, flat } = r.regime, total = up + down + flat;
  const seg = (n, cls, label) => n > 0 ? h("span", { class: `rs-seg ${cls}`, style: { flex: String(n) }, title: `${label}: ${num(n)}` }) : null;
  let note = "Prices rose and fell about equally often in this period.";
  if (down > up * 1.5) note = "Prices mostly fell in this period, so Bearish views had an easier time. A rising market could look very different.";
  else if (down > up * 1.2) note = "Prices fell more often than they rose in this period, which favours Bearish views. A rising market could look different.";
  else if (up > down * 1.5) note = "Prices mostly rose in this period, so Bullish views had an easier time. A falling market could look very different.";
  else if (up > down * 1.2) note = "Prices rose more often than they fell in this period, which favours Bullish views. A falling market could look different.";
  return panel("rs-regime", "What the market did",
    "After each view: did the price clearly rise, clearly fall, or go nowhere much?",
    h("div", { class: "rs-stack", role: "img", "aria-label": `Rose ${num(up)}, went nowhere ${num(flat)}, fell ${num(down)} of ${num(total)}` },
      seg(up, "up", "Rose"), seg(flat, "flat", "Went nowhere"), seg(down, "down", "Fell")),
    h("ul", { class: "rs-legend" },
      h("li", { class: "up" }, `Rose ${Math.round(up / total * 100)}%`),
      h("li", { class: "flat" }, `Went nowhere ${Math.round(flat / total * 100)}%`),
      h("li", { class: "down" }, `Fell ${Math.round(down / total * 100)}%`)),
    h("p", { class: "rs-verdict" }, note));
}

function latest(r) {
  return panel("rs-latest", "Latest graded views",
    "The most recent views whose time is up. These are past market views, not trades to copy.",
    h("ol", { class: "rs-calls" }, r.latest.map((c) =>
      h("li", {},
        h("span", { class: "rs-when" }, when(c.at) + " UTC"),
        h("span", {}, `${c.asset} · ${c.timeframe}`),
        h("span", { class: `rs-bias ${c.bias}` }, c.bias, h("small", {}, ` ${c.bucket}`)),
        h("span", { class: `rs-mark ${c.correct ? "right" : "wrong"}` }, c.correct ? "Right" : "Wrong")))));
}

function method(r) {
  return panel("rs-method", "How a view is graded", null,
    h("ul", { class: "rs-notes" },
      h("li", {}, "When the hour, 4 hours, day or week is over, the server compares the price then with the price when the view was given."),
      h("li", {}, `A Bullish or Bearish view is right if the price moved that way by more than a quarter of its usual range (${r.threshold_atr} ATR). A Neutral view is right if the price stayed inside that range.`),
      h("li", {}, "The simple rules are graded on exactly the same moments. “Follow the recent trend” says Bullish when the short average is above the long one, Bearish when it's below."),
      h("li", {}, `Groups with fewer than ${r.min_sample} graded views say “too few yet”: with so few, a percentage is mostly luck.`),
      h("li", {}, "Most views come from an hourly sampler that asks FinSight about each market on a fixed schedule, so the record isn't only the questions people chose to ask.")));
}

function render(r) {
  const out = $("results");
  if (!r.n) {
    $("asOf").textContent = "No graded views yet.";
    out.replaceChildren(h("section", { class: "panel notice" }, h("h2", {}, "Nothing graded yet"),
      h("p", {}, "Views are graded once their timeframe is over. Check back in an hour.")));
    return;
  }
  $("asOf").replaceChildren(h("b", {}, num(r.n)), " graded views · ", `${day(r.first)} to ${day(r.last)}`,
    h("br"), h("small", {}, `updated ${when(r.generated)} UTC · refreshed every 15 minutes`));
  out.replaceChildren(
    scoreboard(r), buckets(r), regime(r),
    table("By view", "How often each kind of view was right.", r.bias),
    table("By market", null, r.assets),
    table("By timeframe", null, r.timeframes, (tf) => TF_NAMES[tf] || tf),
    latest(r), method(r));
}

async function load() {
  const ctrl = new AbortController(), t = setTimeout(() => ctrl.abort(), 20000);
  try {
    const res = await fetch(API + "/results", { signal: ctrl.signal });
    if (!res.ok) throw new Error(String(res.status));
    render(await res.json());
  } catch {
    $("asOf").textContent = "Results are unavailable right now.";
    $("results").replaceChildren(h("section", { class: "panel notice error" }, h("h2", {}, "Couldn't load the results"),
      h("p", {}, "The server didn't answer. Reload the page in a minute.")));
  } finally {
    clearTimeout(t);
  }
}

window.addEventListener("resize", () => drawBg());
drawBg(true);
load();
