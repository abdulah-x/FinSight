// FinSight: the page background (faint dot grid and one long price path), shared by
// every page. Wrapped so its names can't clash with the page script; exposes drawBg().
"use strict";

(() => {
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- background: faint dot grid and one long price path ----------
  function drawBg(animate = false) {
    const c = document.getElementById("bgArt"), dpr = Math.min(window.devicePixelRatio || 1, 2);
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

  window.drawBg = drawBg;
})();
