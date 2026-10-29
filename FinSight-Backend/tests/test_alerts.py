"""Discord alerts, sent to a real local HTTP server standing in for the webhook."""

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import alerts


@pytest.fixture
def webhook():
    """A tiny real server on localhost that records every JSON body posted to it."""
    received = []

    class Hook(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):  # keep test output quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), Hook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/hook", received
    server.shutdown()


def wait_for(received, count, secs=3):
    end = time.time() + secs
    while len(received) < count and time.time() < end:
        time.sleep(0.02)
    return received


def log_with(handler):
    logger = logging.getLogger(f"finsight.test{id(handler)}")
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def test_errors_are_posted_with_the_traceback(webhook):
    url, received = webhook
    logger = log_with(alerts.DiscordHandler(url))
    try:
        raise ConnectionError("yahoo down")
    except ConnectionError:
        logger.exception("could not fetch exit prices for %s", "GC=F")
    [message] = wait_for(received, 1)
    assert message["username"] == "FinSight alerts"
    assert "**FinSight error**" in message["content"]
    assert "could not fetch exit prices for GC=F" in message["content"] and "ConnectionError: yahoo down" in message["content"]


def test_the_same_failure_is_sent_once_per_cooldown(webhook):
    url, received = webhook
    logger = log_with(alerts.DiscordHandler(url, cooldown=60))
    for symbol in ("BTC-USD", "ETH-USD", "GC=F"):
        logger.error("quote failed for %s", symbol)
    logger.error("a different failure")
    wait_for(received, 2)
    time.sleep(0.3)  # give any extra (wrong) sends time to arrive
    assert [m["content"].split("\n")[2] for m in received] == ["quote failed for BTC-USD", "a different failure"]  # line 1 is the ``` fence


def test_warnings_and_info_are_not_sent(webhook):
    url, received = webhook
    logger = log_with(alerts.DiscordHandler(url))
    logger.setLevel(logging.INFO)
    logger.warning("job was missed by 4s")
    logger.info("resolved 3 predictions")
    time.sleep(0.3)
    assert received == []


def test_long_messages_fit_discords_limit(webhook):
    url, received = webhook
    logger = log_with(alerts.DiscordHandler(url))
    logger.error("x" * 5000)
    [message] = wait_for(received, 1)
    assert len(message["content"]) <= 2000


@pytest.mark.parametrize("url", ["", "https://evil.example/collect", "http://discord.com/api/webhooks/1/x"])
def test_install_only_accepts_a_real_discord_webhook(monkeypatch, url):
    monkeypatch.setattr(alerts, "_handler", None)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", url)
    assert alerts.install() is None
