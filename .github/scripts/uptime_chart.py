"""Draw the README uptime chart from the outside check's log. Standard library only.

    python uptime_chart.py uptime.csv uptime.svg

uptime.csv has one line per check: ISO time (UTC), response time in ms (empty if the
site didn't answer), 1 if the check passed else 0.
"""

import csv
import statistics
import sys
from datetime import datetime, timedelta, timezone

DAYS = 7
GAP = timedelta(hours=3)        # GitHub sometimes skips scheduled runs; break the line, don't count it as down
W, H = 820, 250
LEFT, RIGHT, TOP, BOTTOM = 56, 20, 64, 34


def load(path: str) -> list[tuple[datetime, float | None, bool]]:
    rows = []
    with open(path, newline="") as f:
        for when, ms, ok in csv.reader(f):
            rows.append((datetime.fromisoformat(when), float(ms) if ms else None, ok == "1"))
    return rows


def nice_top(value: float) -> int:
    for step in (200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000, 20000):
        if value <= step:
            return step
    return int(value)


def draw(rows, now: datetime) -> str:
    start = now - timedelta(days=DAYS)
    rows = [r for r in rows if r[0] >= start]
    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM

    def x(t: datetime) -> float:
        return LEFT + (t - start) / (now - start) * plot_w

    answered = [ms for _, ms, ok in rows if ok and ms is not None]
    top = nice_top(max(answered, default=500) * 1.15)

    def y(ms: float) -> float:
        return TOP + plot_h - min(ms, top) / top * plot_h

    # Line segments, broken at failures and at gaps in the schedule.
    segments, current, last_t = [], [], None
    for t, ms, ok in rows:
        if not ok or ms is None or (last_t and t - last_t > GAP):
            if len(current) > 1:
                segments.append(current)
            current = []
        if ok and ms is not None:
            current.append((x(t), y(ms)))
        last_t = t
    if len(current) > 1:
        segments.append(current)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           'role="img" aria-labelledby="t">',
           """<style>
  .ink{fill:#1f1d1a} .muted{fill:#6b665d} .grid{stroke:#e6e2da}
  .line{stroke:#a07c00} .dot{fill:#a07c00} .fail{fill:#c4392f}
  @media (prefers-color-scheme: dark){
    .ink{fill:#efece4} .muted{fill:#9d988e} .grid{stroke:#262a31}
    .line{stroke:#ebc934} .dot{fill:#ebc934} .fail{fill:#f0857a}
  }
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
</style>""",
           ]  # no background: the page shows through

    if rows:
        up = sum(ok for _, _, ok in rows) / len(rows) * 100
        up_text = f"{up:.2f}".rstrip("0").rstrip(".")
        median = f"{statistics.median(answered):.0f} ms median" if answered else "no answers"
        title = f"{up_text}% up · {len(rows)} checks · {median}"
    else:
        title = "collecting data…"
    out.append(f'<title id="t">FinSight uptime, last {DAYS} days: {title}</title>')
    out.append(f'<text class="ink" x="{LEFT}" y="30" font-size="17" font-weight="600">FinSight uptime</text>')
    out.append(f'<text class="muted" x="{LEFT}" y="50" font-size="12.5">last {DAYS} days · {title}</text>')
    out.append(f'<text class="muted" x="{W - RIGHT}" y="30" font-size="11" text-anchor="end">'
               f'updated {now:%d %b %H:%M} UTC · checked from outside, via GitHub</text>')

    for i in range(5):  # horizontal grid with ms labels
        ms = top * i / 4
        gy = y(ms)
        out.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{gy:.1f}" y2="{gy:.1f}" stroke-width="1"/>')
        out.append(f'<text class="muted" x="{LEFT - 8}" y="{gy + 4:.1f}" font-size="10.5" text-anchor="end">'
                   f'{ms:.0f}</text>')
    out.append(f'<text class="muted" x="{LEFT - 8}" y="{TOP - 8}" font-size="10.5" text-anchor="end">ms</text>')

    day = (start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    while day < now:  # day ticks
        out.append(f'<text class="muted" x="{x(day):.1f}" y="{H - 12}" font-size="10.5" text-anchor="middle">'
                   f'{day:%a %d}</text>')
        out.append(f'<line class="grid" x1="{x(day):.1f}" x2="{x(day):.1f}" y1="{TOP}" y2="{TOP + plot_h}" '
                   'stroke-width="1" stroke-dasharray="2 4"/>')
        day += timedelta(days=1)

    base = TOP + plot_h
    for seg in segments:
        pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in seg)
        out.append(f'<polyline class="line" points="{pts}" fill="none" stroke-width="1.6" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')

    for t, ms, ok in rows:  # every answered check is a dot, so isolated checks still show
        if ok and ms is not None:
            out.append(f'<circle class="dot" cx="{x(t):.1f}" cy="{y(ms):.1f}" r="2.6"><title>'
                       f'{ms:.0f} ms at {t:%d %b %H:%M} UTC</title></circle>')

    for t, _, ok in rows:  # failed checks: red dots on the baseline
        if not ok:
            out.append(f'<circle class="fail" cx="{x(t):.1f}" cy="{base:.1f}" r="4"><title>'
                       f'check failed {t:%d %b %H:%M} UTC</title></circle>')

    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    src, dest = sys.argv[1], sys.argv[2]
    try:
        data = load(src)
    except FileNotFoundError:
        data = []
    now = datetime.now(timezone.utc)
    keep = [r for r in data if r[0] >= now - timedelta(days=DAYS + 1)]  # the log never grows past ~8 days
    with open(src, "w", newline="") as f:
        csv.writer(f).writerows((t.isoformat(), "" if ms is None else f"{ms:.0f}", int(ok)) for t, ms, ok in keep)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(draw(keep, now))
