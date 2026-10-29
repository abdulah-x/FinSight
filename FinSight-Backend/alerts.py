"""Pipeline alerts to Discord: every ERROR the backend logs (collector, resolver,
scheduled jobs, unexpected API errors, LLM provider failures) is posted to a channel.

Optional: set DISCORD_WEBHOOK_URL in .env (Discord: channel settings -> Integrations ->
Webhooks -> New Webhook -> Copy Webhook URL). Without it, nothing is sent.

Sending happens on a background thread, so a slow or broken Discord never slows or
breaks a request. The same kind of error is sent at most once per cooldown.
"""

import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone

import requests

COOLDOWN_SECONDS = 600         # same logger + same message template: once per 10 minutes
MAX_CHARS = 1800               # Discord's limit is 2,000 characters per message
WEBHOOK_PREFIXES = ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")
WATCHED_LOGGERS = ("finsight", "apscheduler")  # finsight.* covers the resolver and collector


class DiscordHandler(logging.Handler):
    def __init__(self, url: str, cooldown: float = COOLDOWN_SECONDS):
        super().__init__(level=logging.ERROR)
        self.url, self.cooldown = url, cooldown
        self._last_sent: dict[tuple, float] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        threading.Thread(target=self._send_forever, daemon=True, name="discord-alerts").start()

    def emit(self, record: logging.LogRecord) -> None:
        key = (record.name, str(record.msg))  # the message template: repeats of one failure send once
        now = time.time()
        if now - self._last_sent.get(key, 0) < self.cooldown:
            return
        self._last_sent[key] = now
        detail = record.getMessage()
        if record.exc_info:
            detail += "\n" + "".join(logging.Formatter().formatException(record.exc_info).splitlines(True)[-12:])
        self.send(f"**FinSight error** · `{record.name}`", detail)

    def send(self, title: str, detail: str = "") -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        body = f"{title} · {stamp}"
        if detail:
            body += "\n```\n" + detail[-MAX_CHARS:].replace("```", "'''") + "\n```"
        try:
            self._queue.put_nowait(body)
        except queue.Full:
            pass  # an error storm: drop rather than pile up memory

    def _send_forever(self) -> None:
        while True:
            body = self._queue.get()
            try:
                requests.post(self.url, json={"content": body, "username": "FinSight alerts"}, timeout=5)
            except Exception:
                pass  # never log from here: that would feed this handler and loop


_handler: DiscordHandler | None = None


def install() -> DiscordHandler | None:
    """Attach the Discord handler if DISCORD_WEBHOOK_URL is set. Safe to call twice."""
    global _handler
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if _handler or not url:
        return _handler
    if not url.startswith(WEBHOOK_PREFIXES):  # never ship logs to some other host by mistake
        logging.getLogger("finsight").warning("DISCORD_WEBHOOK_URL is not a Discord webhook URL; alerts are off")
        return None
    _handler = DiscordHandler(url)
    for name in WATCHED_LOGGERS:
        logging.getLogger(name).addHandler(_handler)
    return _handler


def notify(title: str, detail: str = "") -> None:
    """A one-off message (e.g. "backend started"), if alerts are on."""
    if _handler:
        _handler.send(title, detail)
