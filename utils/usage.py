"""Claude Code usage meter.

Reads the current session + weekly limit usage that Claude Code's ``/usage``
command reports, so the floating dot can show it. There is no local cache of
these percentages (they come from the server), and no read-only CLI
subcommand, so we invoke the CLI in print mode:

    claude -p "/usage" --no-session-persistence

``--no-session-persistence`` keeps each poll from cluttering the user's
session history / contributing-factors stats. The call hits the network and
takes several seconds, so it must always run off the UI thread.

The parsed output looks like::

    Current session: 38% used · resets Jul 8 at 8:49pm (...)
    Current week (all models): 59% used · resets Jul 10 at 2:59pm (...)
    Current week (Fable): 15% used · resets ...
"""

from __future__ import annotations

import datetime
import json
import queue
import re
import shutil
import subprocess
import threading
import time

_SESSION_RE = re.compile(r"current session:\s*(\d+)\s*%", re.IGNORECASE)
# Prefer the "(all models)" weekly line; fall back to the first weekly line.
_WEEK_ALL_RE = re.compile(r"current week\s*\(all models\):\s*(\d+)\s*%", re.IGNORECASE)
_WEEK_ANY_RE = re.compile(r"current week[^\n]*?:\s*(\d+)\s*%", re.IGNORECASE)


def parse_usage(text: str):
    """Extract ``(session_pct, week_pct)`` from ``/usage`` output.

    Each element is an int 0-100, or ``None`` if that line wasn't found.
    """
    text = text or ""
    s = _SESSION_RE.search(text)
    w = _WEEK_ALL_RE.search(text) or _WEEK_ANY_RE.search(text)
    session = int(s.group(1)) if s else None
    week = int(w.group(1)) if w else None
    return session, week


# The "current session" line is the 5-hour rolling window; grab its reset time.
_SESSION_LINE_RE = re.compile(r"current session:[^\n]*", re.IGNORECASE)
_CLOCK_RE = re.compile(r"(\d{1,2}:\d{2}\s*[ap]\.?m\.?)", re.IGNORECASE)


def parse_session_reset(text: str):
    """Return when the 5-hour session window resets, e.g. ``"8:49pm"`` (or None)."""
    text = text or ""
    m = _SESSION_LINE_RE.search(text)
    if not m:
        return None
    line = m.group(0)
    clock = _CLOCK_RE.search(line)
    if clock:
        return re.sub(r"\s+", "", clock.group(1)).lower()  # "8:49 pm" -> "8:49pm"
    tail = re.search(r"resets\s+(.+?)(?:\s*\(|$)", line, re.IGNORECASE)
    return tail.group(1).strip() if tail else None


# A bare clock time, 12- or 24-hour: "8:49pm", "8pm", "20:49".
_CLOCK_ONLY_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$")


def clock_to_utc(clock: str):
    """Convert a local wall-clock string like ``"8:49pm"`` to UTC ``"HH:MM"``.

    ``parse_session_reset`` reports the reset in the machine's local time,
    which is awkward to compare against anything that speaks UTC.  Returns
    ``None`` when the string isn't a plain clock time — that line sometimes
    carries a date or a phrase instead — so the caller can hide the UTC
    reading rather than show something wrong.

    Today's local date supplies the UTC offset, so the result stays right
    across DST changes.  Only the time of day comes back; a late-evening reset
    may well land on tomorrow's UTC date.
    """
    text = (clock or "").strip().lower().replace(".", "").replace(" ", "")
    m = _CLOCK_ONLY_RE.match(text)
    if not m:
        return None
    hour, minute, suffix = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if minute > 59:
        return None
    if suffix:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if suffix == "pm" else 0)
    elif hour > 23:
        return None
    local = datetime.datetime.combine(
        datetime.date.today(), datetime.time(hour, minute)
    )
    return local.astimezone(datetime.timezone.utc).strftime("%H:%M")


def claude_executable():
    """Absolute path to the ``claude`` CLI, or ``None`` if not on PATH."""
    return shutil.which("claude")


def claude_available() -> bool:
    return claude_executable() is not None


def fetch_usage(timeout: float = 30.0):
    """Run the CLI once and return ``(session_pct, week_pct, session_reset)``.

    Returns ``(None, None, None)`` if the CLI is missing, times out, or errors —
    the caller treats that as "no data yet" and keeps the last known value.
    """
    exe = claude_executable()
    if not exe:
        return (None, None, None)
    try:
        proc = subprocess.run(
            [exe, "-p", "/usage", "--no-session-persistence"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return (None, None, None)

    out = proc.stdout
    session, week = parse_usage(out)
    reset = parse_session_reset(out)
    if session is None and week is None and reset is None:
        # Some builds route the panel to stderr; try that too.
        out = proc.stderr
        session, week = parse_usage(out)
        reset = parse_session_reset(out)
    return (session, week, reset)


# --------------------------------------------------------------------------- #
# ccusage cost meter (today + this month, USD-equivalent)                      #
# --------------------------------------------------------------------------- #
#
# `bunx ccusage daily --json` reports per-day rows with a `period` (YYYY-MM-DD)
# and a `totalCost`. Today's cost is the row whose period is today; the month
# cost is the sum of the rows in the current calendar month so far (from the
# 1st through today, inclusive) — month-to-date, NOT a rolling 30-day window.


def parse_ccusage(json_text: str, today=None):
    """Extract ``(today_cost, month_cost)`` (floats, USD) from daily JSON.

    ``month_cost`` sums the current calendar month so far (from the 1st through
    today, inclusive). ``today`` overrides the current date (``date`` or ISO
    string, for tests). Returns ``(None, None)`` if the JSON can't be parsed; a
    valid payload with no row for today yields ``0.0`` (you simply haven't spent
    anything today).
    """
    if today is None:
        today = datetime.date.today()
    elif isinstance(today, str):
        today = datetime.date.fromisoformat(today)
    start = today.replace(day=1)  # first of the current month
    today_str = today.isoformat()

    try:
        rows = json.loads(json_text).get("daily") or []
    except Exception:
        return (None, None)

    def in_month(period) -> bool:
        try:
            d = datetime.date.fromisoformat(str(period)[:10])
        except Exception:
            return False
        return start <= d <= today

    today_cost = sum(r.get("totalCost", 0.0) for r in rows if r.get("period") == today_str)
    month_cost = sum(r.get("totalCost", 0.0) for r in rows if in_month(r.get("period")))
    return (float(today_cost), float(month_cost))


def bunx_available() -> bool:
    return shutil.which("bunx") is not None


def fetch_ccusage(timeout: float = 90.0):
    """Run ``bunx ccusage daily --json`` and return ``(today_cost, month_cost)``.

    Returns ``(None, None)`` if ``bunx`` is missing, times out, or errors.
    """
    exe = shutil.which("bunx")
    if not exe:
        return (None, None)
    try:
        proc = subprocess.run(
            [exe, "ccusage", "daily", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return (None, None)
    return parse_ccusage(proc.stdout)


# --------------------------------------------------------------------------- #
# Codex 7-day limit meter                                                      #
# --------------------------------------------------------------------------- #
#
# Codex has no `usage` subcommand, but its app-server — the same backend the
# interactive ``/status`` command reads — exposes the live plan rate limits
# over JSON-RPC on stdio. We spawn it, `initialize`, and call
# ``account/rateLimits/read``. This reuses the stored ChatGPT login, costs no
# tokens (there is no model turn) and writes no session. The reply looks like::
#
#     {"rateLimitsByLimitId": {"codex": {"primary":
#         {"usedPercent": 33, "windowDurationMins": 10080, "resetsAt": ...},
#      "secondary": null}}}
#
# On this plan Codex reports a single 7-day (10080-minute) window, so we pick
# the widest window and surface its used-percent.


def codex_available() -> bool:
    return shutil.which("codex") is not None


def _pick_weekly_window(snapshot):
    """From a rate-limit snapshot, return the widest window dict (or None).

    The app-server splits metered windows into ``primary``/``secondary``; the
    7-day (10080-min) window is the widest, so picking by ``windowDurationMins``
    yields the weekly limit no matter which slot it occupies.
    """
    if not isinstance(snapshot, dict):
        return None
    weekly = None
    for w in (snapshot.get("primary"), snapshot.get("secondary")):
        if not isinstance(w, dict):
            continue
        if weekly is None or (w.get("windowDurationMins") or 0) > (weekly.get("windowDurationMins") or 0):
            weekly = w
    return weekly


def parse_codex_usage(response):
    """Extract ``(week_pct, reset)`` from an ``account/rateLimits/read`` result.

    Prefers the ``codex`` bucket of ``rateLimitsByLimitId`` and falls back to
    the flat ``rateLimits`` snapshot. ``week_pct`` is the 7-day used-percent
    (int 0-100) and ``reset`` a short local date like ``"Jul 19"``. Returns
    ``(None, None)`` when no usable window is present.
    """
    if not isinstance(response, dict):
        return (None, None)
    by_id = response.get("rateLimitsByLimitId") or {}
    snapshot = by_id.get("codex") or response.get("rateLimits")
    window = _pick_weekly_window(snapshot)
    if not window or window.get("usedPercent") is None:
        return (None, None)
    return (int(round(window["usedPercent"])), _fmt_codex_reset(window.get("resetsAt")))


def _fmt_codex_reset(epoch):
    """Format a unix ``resetsAt`` as a short local date, e.g. ``"Jul 19"``."""
    try:
        dt = datetime.datetime.fromtimestamp(float(epoch))
    except Exception:
        return None
    return dt.strftime("%b %d").replace(" 0", " ")  # "Jul 07" -> "Jul 7"


def fetch_codex_usage(timeout: float = 20.0):
    """Return ``(week_pct, reset)`` for Codex's 7-day limit via the app-server.

    Drives ``codex app-server`` over JSON-RPC (``initialize`` then
    ``account/rateLimits/read``). The server is long-lived and never closes its
    stdout, so we read line-by-line off a helper thread until the id=2 reply
    arrives (or ``timeout``), then tear it down. Returns ``(None, None)`` if
    ``codex`` is missing, not logged in, or the call errors/times out.
    """
    exe = shutil.which("codex")
    if not exe:
        return (None, None)

    try:
        proc = subprocess.Popen(
            [exe, "app-server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
    except Exception:
        return (None, None)

    lines = queue.Queue()

    def _reader():
        try:
            for line in proc.stdout:
                lines.put(line)
        except Exception:
            pass
        finally:
            lines.put(None)  # sentinel: stdout closed

    threading.Thread(target=_reader, daemon=True).start()

    try:
        for msg in (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"clientInfo": {"name": "yiliVoice", "version": "1.0"}}},
            {"jsonrpc": "2.0", "method": "initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "account/rateLimits/read", "params": {}},
        ):
            proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return (None, None)
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                return (None, None)
            if line is None:  # stdout closed before a reply
                return (None, None)
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("id") == 2:
                result = obj.get("result")
                return parse_codex_usage(result) if isinstance(result, dict) else (None, None)
    except Exception:
        return (None, None)
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
