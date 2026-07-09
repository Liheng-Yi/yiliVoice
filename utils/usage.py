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
import re
import shutil
import subprocess

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
# and a `totalCost`. Today's cost is the row whose period is today; the window
# cost is the sum of the rows in the trailing 30 days (ending today,
# inclusive) — a rolling window, NOT the calendar month.


def parse_ccusage(json_text: str, today=None, window_days: int = 30):
    """Extract ``(today_cost, window_cost)`` (floats, USD) from daily JSON.

    ``window_cost`` sums the last ``window_days`` days ending today (inclusive).
    ``today`` overrides the current date (``date`` or ISO string, for tests).
    Returns ``(None, None)`` if the JSON can't be parsed; a valid payload with
    no row for today yields ``0.0`` (you simply haven't spent anything today).
    """
    if today is None:
        today = datetime.date.today()
    elif isinstance(today, str):
        today = datetime.date.fromisoformat(today)
    start = today - datetime.timedelta(days=window_days - 1)
    today_str = today.isoformat()

    try:
        rows = json.loads(json_text).get("daily") or []
    except Exception:
        return (None, None)

    def in_window(period) -> bool:
        try:
            d = datetime.date.fromisoformat(str(period)[:10])
        except Exception:
            return False
        return start <= d <= today

    today_cost = sum(r.get("totalCost", 0.0) for r in rows if r.get("period") == today_str)
    window_cost = sum(r.get("totalCost", 0.0) for r in rows if in_window(r.get("period")))
    return (float(today_cost), float(window_cost))


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
