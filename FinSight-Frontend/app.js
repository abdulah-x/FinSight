// FinSight: talks to the agent API (POST /analyze, POST /resume).
// Everything the model or a data source wrote is inserted as text (textContent),
// never parsed as HTML, so it can't inject markup or scripts.
"use strict";

const API = document.documentElement.dataset.api || "http://localhost:8000";
const ASSETS = { "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "GC=F": "Gold" };
const PICKS = { "BTC-USD": "BTC", "ETH-USD": "ETH", "GC=F": "Gold" };
const TFS = ["1H", "4H", "1D", "1W"];
const SOURCE_NAMES = { price_action: "price action", macro: "macro", calendar: "calendar", fear_greed: "fear & greed",
                       funding: "funding", news: "news", social: "social" };
const REQUEST_TIMEOUT_MS = 60000;

const state = { sessionId: null, pending: false, current: null, history: [], busy: false, asset: null, tf: null };
const $ = (id) => document.getElementById(id);

// ---------- DOM helper ----------
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
  return el;
}
const fmtNet = (n) => (n > 0 ? "+" : n < 0 ? "−" : "") + Math.abs(n).toFixed(2);
const tone = (v) => (v == null ? "" : Math.abs(v) < 0.05 ? "flat" : v > 0 ? "up" : "down");
// colour a group by the engine's own label, so "neutral" is never drawn red after rounding
const labelTone = (label) => (/bullish/.test(label) ? "up" : /bearish/.test(label) ? "down" : "flat");
const assetName = (a) => ASSETS[a] || a;
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function countUp(el, target, ms = 700) {
  if (reducedMotion()) { el.textContent = String(target); return; }
  const start = performance.now();
  const frame = (now) => {
    const t = Math.min(1, (now - start) / ms);
    el.textContent = String(Math.round(target * (1 - (1 - t) ** 3)));
    if (t < 1) requestAnimationFrame(frame);
  };
  el.textContent = "0";
  requestAnimationFrame(frame);
}

// ---------- background: faint dot grid and one long price path ----------
function drawBg(animate = false) {
  const c = $("bgArt"), dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = window.innerWidth, ht = window.innerHeight;
  c.width = w * dpr; c.height = ht * dpr;
  const x = c.getContext("2d"); x.scale(dpr, dpr); x.clearRect(0, 0, w, ht);
  const css = getComputedStyle(document.documentElement);
  const muted = css.getPropertyValue("--muted").trim(), accent = css.getPropertyValue("--accent").trim();
  x.globalAlpha = 0.22; x.fillStyle = muted;
  for (let gx = 12; gx < w; gx += 28) for (let gy = 12; gy < ht; gy += 28) x.fillRect(gx, gy, 1.2, 1.2);
  const glow = x.createRadialGradient(w * 0.82, ht * 0.12, 0, w * 0.82, ht * 0.12, Math.max(w, ht) * 0.5);
  glow.addColorStop(0, accent); glow.addColorStop(1, "transparent");
  x.globalAlpha = 0.09; x.fillStyle = glow; x.fillRect(0, 0, w, ht);
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const pts = []; let y = ht * 0.72;
  for (let px = 0; px <= w; px += 6) { y += (rnd() - 0.48) * 14; y = Math.max(ht * 0.45, Math.min(ht * 0.92, y)); pts.push([px, y]); }
  const candleWicks = Array.from({ length: 16 }, () => rnd() * 10);
  const dots = x.getImageData(0, 0, c.width, c.height);

  const paint = (share) => {  // draw the price path up to `share` (0..1) of its length
    x.putImageData(dots, 0, 0);
    const upto = pts.slice(0, Math.max(2, Math.ceil(pts.length * share)));
    const [endX] = upto[upto.length - 1];
    x.globalAlpha = 0.07; x.fillStyle = accent; x.beginPath(); x.moveTo(0, ht);
    upto.forEach(([px, py]) => x.lineTo(px, py)); x.lineTo(endX, ht); x.closePath(); x.fill();
    x.globalAlpha = 0.22; x.strokeStyle = accent; x.lineWidth = 1.2; x.beginPath();
    upto.forEach(([px, py], i) => (i ? x.lineTo(px, py) : x.moveTo(px, py))); x.stroke();
    // faint candles riding the line: one per 4 points, body from open to close, wick past both
    x.globalAlpha = 0.16; x.fillStyle = accent; x.strokeStyle = accent; x.lineWidth = 1;
    for (let i = 4; i < upto.length; i += 4) {
      const [cx, close] = upto[i], open = upto[i - 4][1], wick = 6 + candleWicks[i % candleWicks.length];
      x.beginPath(); x.moveTo(cx, Math.min(open, close) - wick); x.lineTo(cx, Math.max(open, close) + wick); x.stroke();
      x.fillRect(cx - 4, Math.min(open, close), 8, Math.max(2, Math.abs(close - open)));
    }
  };
  if (!animate || reducedMotion()) return paint(1);
  const start = performance.now(), DRAW_MS = 1600;
  const frame = (now) => {
    const t = Math.min(1, (now - start) / DRAW_MS);
    paint(1 - (1 - t) ** 3);  // ease out: fast start, gentle finish
    if (t < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// ---------- markets strip: live prices from GET /markets (no AI involved) ----------
const SVG_NS = "http://www.w3.org/2000/svg";
const MARKETS_REFRESH_MS = 60000;
const fmtPrice = (v) => v.toLocaleString("en-US", { maximumFractionDigits: v >= 10000 ? 0 : v >= 100 ? 1 : 2 });

function sparkline(values, up) {
  const w = 96, ht = 28, min = Math.min(...values), span = (Math.max(...values) - min) || 1;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${ht}`);
  svg.setAttribute("class", `spark ${up ? "up" : "down"}`);
  svg.setAttribute("aria-hidden", "true");
  const line = document.createElementNS(SVG_NS, "polyline");
  line.setAttribute("points", values.map((v, i) => `${(i / (values.length - 1)) * w},${ht - 3 - ((v - min) / span) * (ht - 6)}`).join(" "));
  svg.append(line);
  return svg;
}

function renderMarkets(data) {
  $("mkList").replaceChildren(...data.assets.map((q) => {
    const name = assetName(q.symbol);
    if (q.error) return h("li", {}, h("div", { class: "mk-row off" }, h("span", { class: "mk-name" }, name), h("span", { class: "mk-err" }, "price unavailable")));
    const up = q.change_pct >= 0;
    return h("li", {}, h("button", { type: "button", class: "mk-row", title: `Fill in a ${name} task`,
      onclick: () => {
        state.asset = q.symbol; state.tf = state.tf || "4H"; renderPicks();
        $("task").value = `${name} ${state.tf} bias`; $("task").focus();
      } },
      h("span", { class: "mk-name" }, name, PICKS[q.symbol] !== name && h("small", {}, PICKS[q.symbol])),
      q.spark.length > 1 ? sparkline(q.spark, up) : h("span"),
      h("span", { class: "mk-price" }, fmtPrice(q.last),
        h("small", { class: up ? "up" : "down" }, `${up ? "▲" : "▼"} ${Math.abs(q.change_pct).toFixed(2)}%`))));
  }));
  const ev = data.next_event;
  $("mkEvent").hidden = !ev;
  if (ev) $("mkEvent").replaceChildren(h("span", { class: "mk-dot", "aria-hidden": "true" }), "Next high-impact event: ",
    h("b", {}, ev.event), ` in ${ev.hours_away < 1 ? "under 1h" : Math.round(ev.hours_away) + "h"}`);
  $("mkTime").textContent = "24h · updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

async function loadMarkets() {
  const ctrl = new AbortController(), t = setTimeout(() => ctrl.abort(), 15000);
  try {
    const res = await fetch(API + "/markets", { signal: ctrl.signal });
    if (!res.ok) throw new Error(String(res.status));
    renderMarkets(await res.json());
  } catch {
    if (!$("mkList").children.length) $("mkList").replaceChildren(h("li", { class: "empty-note" }, "Prices are unavailable right now."));
    $("mkTime").textContent = "offline";
  } finally {
    clearTimeout(t);
  }
}

// ---------- quick picks: they only fill in the task bar ----------
function renderPicks() {
  const chips = (id, values, label, key) => $(id).replaceChildren(...values.map((v) => h("button", {
    type: "button", class: "pick", "aria-pressed": String(state[key] === v),
    onclick: () => {
      state[key] = v; renderPicks();
      if (state.asset && state.tf) $("task").value = `${assetName(state.asset)} ${state.tf} bias`;
      $("task").focus();
    },
  }, label(v))));
  chips("assetPicks", Object.keys(PICKS), (a) => PICKS[a], "asset");
  chips("tfPicks", TFS, (t) => t, "tf");
}

// ---------- the writer's text: drop the code-written header and breakdown ----------
function explanation(answer) {
  let body = (answer || "").split("\nHow the score was built:")[0];
  const firstBreak = body.indexOf("\n");
  body = firstBreak >= 0 ? body.slice(firstBreak + 1) : "";
  return body.trim().split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
}

function blocks(paragraphs) {
  const clean = (s) => s.replace(/\*\*(.+?)\*\*/g, "$1").replace(/^#+\s*/, "");
  const bullet = /^\s*[-*•]\s+/;
  // within a block, runs of "- item" lines become a list and other lines a paragraph
  return paragraphs.flatMap((p) => {
    const out = [];
    for (const line of p.split("\n")) {
      const last = out[out.length - 1];
      if (bullet.test(line)) {
        const item = h("li", {}, clean(line.replace(bullet, "")));
        if (last && last.tagName === "UL") last.append(item); else out.push(h("ul", {}, item));
      } else if (last && last.tagName === "P") {
        last.append(" " + clean(line));
      } else {
        out.push(h("p", {}, clean(line)));
      }
    }
    return out;
  });
}

// ---------- run log ----------
// Each step keeps its own <li> and is updated in place, so its entrance animation
// plays once when it appears (the log repaints every 100ms while the agent works).
function renderSteps(steps, seconds) {
  const list = $("steps");
  if ([...list.children].some((li, i) => li !== steps[i]?.el)) list.replaceChildren();  // a new run
  for (const s of steps) {
    if (!s.el) {
      s.el = h("li", {}, h("span", { class: "dot", "aria-hidden": "true" }), h("span", { class: "st-title" }),
        h("span", { class: "st-time" }), h("span", { class: "st-sub" }));
      if (s.quick) s.el.append(h("span", { class: "quick" }, TFS.map((t) => h("button", { type: "button", onclick: () => resume(t) }, t))));
    }
    const [, title, time, sub] = s.el.children;
    s.el.className = `st ${s.status} enter`;
    title.textContent = s.title;
    time.textContent = s.time || (s.status === "now" && s.t0 ? elapsed(s.t0) : "");
    sub.textContent = s.sub || "";
    sub.hidden = !s.sub;
    if (s.el.parentNode !== list) list.append(s.el);
  }
  $("runTotal").textContent = seconds != null ? `${seconds.toFixed(1)}s` : "";
}

function renderHistory() {
  const past = state.history.filter((r) => r !== state.current);
  $("past").replaceChildren(...(past.length
    ? past.slice().reverse().map((r) => h("li", {}, h("button", { type: "button", onclick: () => reopen(r) },
        h("span", {}, `${assetName(r.analysis.asset)} ${r.analysis.timeframe}`),
        h("span", { class: tone(r.analysis.bias === "Bullish" ? 1 : r.analysis.bias === "Bearish" ? -1 : 0) }, `${r.analysis.bias} ${r.analysis.score}`))))
    : [h("li", { class: "empty-note" }, "Runs you make appear here.")]));
}

// ---------- result area ----------
function biasCard(run, reveal = false) {
  const a = run.analysis;
  const score = h("b", {}, String(a.score));
  if (reveal) countUp(score, a.score);
  const marker = h("div", { class: "marker", title: `net evidence ${fmtNet(a.net)}` });
  requestAnimationFrame(() => requestAnimationFrame(() => { marker.style.left = `${(a.net + 1) * 50}%`; }));
  const events = a.event_penalty < 1 && (a.breakdown.find((l) => l.startsWith("Events:")) || "").replace(/^Events:\s*/, "").replace(/\s*\(score x[\d.]+\)$/, "");

  return h("article", { class: reveal ? "card reveal" : "card" },
    h("div", { class: "card-top" },
      h("div", { class: "eyebrow" }, `${assetName(a.asset).toUpperCase()} · ${a.timeframe}`,
        run.cached && h("span", { class: "chip", title: "Same asset and timeframe were analysed recently, so the stored result is shown" }, "Recent result"),
        run.reopened && h("span", { class: "chip" }, "Earlier run"),
        events && h("span", { class: "chip warn" }, `${events} · score ×${a.event_penalty}`)),
      h("div", { class: "verdict" },
        h("h2", { class: `bias ${a.bias}` }, a.bias),
        h("div", { class: "score" }, score, h("small", {}, " /100"), h("span", { class: "bucket" }, `${a.bucket} evidence`))),
      h("div", { class: "meter", role: "img", "aria-label": `Net evidence ${fmtNet(a.net)} on a scale from −1 bearish to +1 bullish; neutral between −0.15 and +0.15` },
        h("div", { class: "track" }, h("div", { class: "band" }), marker),
        h("div", { class: "scale", "aria-hidden": "true" }, h("span", {}, "Bearish −1"), h("span", {}, "neutral zone ±0.15"), h("span", {}, "+1 Bullish")))),
    h("div", { class: "cols" },
      h("div", { class: "prose" }, h("h3", { class: "sec" }, "What the agent found"),
        run.prose.length ? blocks(run.prose) : h("p", {}, "No explanation was written for this run.")),
      h("div", { class: "groups" }, h("h3", { class: "sec" }, `How the score was built · coverage ${Math.round(a.coverage * 100)}%`),
        (a.rows || []).map((g, i) => {
          // bars grow outward from the centre line: right for bullish, left for bearish
          const fill = h("div", { class: `g-fill ${labelTone(g.label)} ${g.value >= 0 ? "from-left" : "from-right"}` });
          if (g.value != null) { fill.style[g.value >= 0 ? "left" : "right"] = "50%"; fill.style.width = `${Math.abs(g.value) * 50}%`; }
          const row = h("div", { class: "g" + (g.value == null ? " missing" : "") },
            h("div", { class: "g-name" }, g.group, h("small", {}, `${Math.round(g.weight * 100)}%`)),
            h("div", { class: "g-bar", "aria-hidden": "true" }, g.value != null && fill),
            h("div", { class: `g-word ${g.value == null ? "" : labelTone(g.label)}` }, g.label),
            h("div", { class: "g-detail" }, g.detail || "source unavailable this run"));
          row.style.setProperty("--i", i);  // stagger: one row after another
          return row;
        }))),
    run.qa.map((x) => h("div", { class: "qa" }, h("div", { class: "q" }, x.q), h("div", { class: "a" }, blocks(x.a)))),
    h("div", { class: "foot" }, "The evidence score measures how strongly the evidence agrees, scaled by coverage and upcoming events. It is not yet calibrated against outcomes. Not financial advice."));
}

function notice(title, text, kind = "") {
  $("result").replaceChildren(h("div", { class: `card notice ${kind}` }, h("h2", {}, title), h("p", {}, text)));
}

function showCurrent(reveal = false) {
  if (state.current) $("result").replaceChildren(biasCard(state.current, reveal));
  renderHistory();
}

// ---------- live run log: one step per event from the agent ----------
const run = { steps: [], start: 0, gathered: false };
const elapsed = (t0) => `${((performance.now() - t0) / 1000).toFixed(1)}s`;
const sourceList = (names) => names.map((n) => SOURCE_NAMES[n] || n).join(", ");

function paintSteps() { renderSteps(run.steps, run.start ? (performance.now() - run.start) / 1000 : null); }
function openStep(title, sub) { run.steps.push({ title, sub, status: "now", t0: performance.now() }); paintSteps(); }
function closeStep(title, sub, status = "done") {
  const step = run.steps.findLast((s) => s.status === "now");
  if (!step) return;
  Object.assign(step, { status, time: elapsed(step.t0) });
  if (title !== undefined) step.title = title;
  if (sub !== undefined) step.sub = sub;
  paintSteps();
}

function onEvent(e, prompt, path) {
  switch (e.event) {
    case "checked":
      if (e.category !== "ok") { closeStep("Checked the request", "declined", "stop"); break; }
      closeStep("Checked the request", e.follow_up ? "a question about the last result"
        : e.asset ? `${assetName(e.asset)}, ${e.timeframe || "timeframe not given"}` : undefined);
      if (e.cached) { openStep("Reusing a recent result"); closeStep(undefined, "same asset and timeframe, no new evidence needed"); }
      else if (e.follow_up) openStep("Answering from the conversation", "no new evidence");
      else openStep("Planning the evidence", "choosing the sources that matter");
      break;
    case "planned":
      closeStep("Planned the evidence", sourceList(e.sources));
      openStep(`Gathering ${e.sources.length} sources`, "in parallel");
      break;
    case "asking":
      closeStep("Decided to ask you", "the timeframe is missing");
      break;
    case "gathered": {
      run.gathered = true;
      const missing = e.sources.filter((s) => !s.ok).map((s) => s.name);
      closeStep(`Gathered ${e.sources.length} sources`,
        sourceList(e.sources.map((s) => s.name)) + (missing.length ? ` · unavailable: ${sourceList(missing)}` : ""));
      openStep("Reviewing the evidence", "deciding if it has enough");
      break;
    }
    case "agent_done":
      if (run.gathered) { closeStep("Reviewed the evidence", "enough to score"); openStep("Scoring and writing", "the engine scores, the model explains"); }
      break;
    case "result":
      finish(e.response, prompt);
      return true;
    case "error":
      failed({ status: e.status, data: { detail: e.detail }, retryAfter: e.retry_after }, path);
      return true;
  }
  return false;
}

// ---------- talking to the API (server-sent events over a POST) ----------
let ticker = null;
function setBusy(busy) {
  state.busy = busy;
  const btn = $("runBtn");
  btn.disabled = busy;
  btn.classList.toggle("busy", busy);
  btn.setAttribute("aria-busy", String(busy));
  $("runLabel").textContent = busy ? "Running…" : "Run agent";
  document.querySelectorAll(".pick, .quick button").forEach((b) => { b.disabled = busy; });
  document.querySelector("#result .card")?.classList.toggle("stale", busy);
  clearInterval(ticker);
  if (busy) ticker = setInterval(paintSteps, 100);  // keeps the open step's timer moving
}

async function send(path, body, prompt) {
  if (state.busy) return;
  Object.assign(run, { steps: [], start: performance.now(), gathered: false });
  setBusy(true);
  openStep("Checking the request", `“${prompt}”`);
  const ctrl = new AbortController(), timeout = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
  let done = false;
  try {
    const res = await fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: ctrl.signal });
    if (!res.ok) {
      let data = null;
      try { data = await res.json(); } catch { /* empty or non-JSON body */ }
      failed({ status: res.status, data, retryAfter: res.headers.get("Retry-After") }, path);
      done = true;
      return;
    }
    const reader = res.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done: ended } = await reader.read();
      if (ended) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) done = onEvent(JSON.parse(line.slice(6)), prompt, path) || done;
        }
      }
    }
  } catch {
    // network error or timeout: reported below
  } finally {
    clearTimeout(timeout);
    if (!done) failed({ status: 0 }, path);
    setBusy(false);
  }
}

function analyze(text) {
  send("/analyze/stream", state.sessionId ? { query: text, session_id: state.sessionId } : { query: text }, text);
}

function resume(answer) {
  if (!state.pending || !state.sessionId) return;
  send("/resume/stream", { session_id: state.sessionId, answer }, answer);
}

function finish(data, prompt) {
  state.sessionId = data.session_id;
  state.pending = false;
  const total = (performance.now() - run.start) / 1000;
  const freeze = () => renderSteps(run.steps, total);

  if (data.status === "refused") {
    freeze();
    notice("Request declined", data.answer);
    return;
  }

  if (data.status === "needs_input") {
    state.pending = true;
    closeStep();
    run.steps.push({ title: "Needs your input", sub: data.question, status: "ask", quick: true });
    freeze();
    notice("Waiting for your answer", `${data.question}\nPick a timeframe in the run log, or type your answer in the task bar.`);
    $("task").placeholder = "Type your answer, e.g. 4H";
    return;
  }

  $("task").placeholder = "Give the agent a task, e.g. “Gold daily bias” or “why is it bearish?”";
  const a = data.analysis;

  if (!a) {  // a question about the previous answer, or a reply with no new analysis
    closeStep("Answered from the conversation", "no new evidence and no new score");
    freeze();
    if (state.current) { state.current.qa.push({ q: prompt, a: data.answer.split(/\n\s*\n/) }); showCurrent(); }
    else notice("The agent says", data.answer);
    return;
  }

  if (!data.cached) {
    const open = run.steps.findLast((s) => s.status === "now");
    if (data.stop_reason) {  // a cap stopped the loop: say so before the scoring step
      closeStep();
      run.steps.push({ title: "Stopped early", sub: data.stop_reason, status: "stop" });
      openStep("Scoring and writing");
    } else if (!open || open.title !== "Scoring and writing") {
      closeStep();
      openStep("Scoring and writing");
    }
    closeStep("Scored and wrote the explanation", `net ${fmtNet(a.net)} · coverage ${Math.round(a.coverage * 100)}%`);
  }
  freeze();

  state.asset = PICKS[a.asset] ? a.asset : null;
  state.tf = a.timeframe;
  renderPicks();
  state.current = { analysis: a, prose: explanation(data.answer), cached: data.cached, qa: [] };
  state.history.push(state.current);
  showCurrent(true);
}

function failed(res, path) {
  const why = {
    0: ["Can't reach the agent", `The FinSight API at ${API} didn't respond. Check that the backend is running, then try again.`],
    409: ["That question has expired", "The agent is no longer waiting for an answer. Give it the task again."],
    422: ["Request not accepted", "Keep the task under 500 characters."],
    429: ["Too many requests", `The limit is 10 requests a minute. Try again in ${res.retryAfter || 60} seconds.`],
    503: ["The agent is busy", "The AI provider is at its limit right now. Try again in a minute."],
  }[res.status] || ["Something went wrong", (res.data && res.data.detail) || `The API answered with status ${res.status}.`];
  if (res.status === 409 || path.startsWith("/analyze")) state.pending = false;
  if (run.steps.some((s) => s.status === "now")) closeStep("Run failed", why[0], "fail");
  else run.steps.push({ title: "Run failed", sub: why[0], status: "fail" });
  renderSteps(run.steps, (performance.now() - run.start) / 1000);
  notice(why[0], why[1], "error");
}

function reopen(run) {
  state.current = { ...run, reopened: true, cached: false };
  renderSteps([{ title: "Reopened an earlier run", sub: `${assetName(run.analysis.asset)} ${run.analysis.timeframe}`, status: "done" }], null);
  $("result").replaceChildren(biasCard(state.current));
  state.current = run;
  renderHistory();
}

// ---------- start ----------
$("taskForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("task"), text = input.value.trim();
  if (!text || state.busy) return;
  input.value = "";
  if (state.pending) resume(text); else analyze(text);
});
window.addEventListener("resize", drawBg);

drawBg(true);
loadMarkets();
setInterval(loadMarkets, MARKETS_REFRESH_MS);
renderPicks();
renderSteps([{ title: "Ready", sub: "give the agent a task above", status: "wait" }], null);
renderHistory();
notice("No analysis yet", "Pick an asset and a timeframe, or type a task like “Gold daily bias”. The agent checks the request, gathers the evidence it needs, scores it and explains the result.");
