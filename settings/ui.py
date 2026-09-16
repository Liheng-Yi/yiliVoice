"""Minimal floating status dot, built on PySide6 (Qt).

The whole status UI is a single coloured dot that floats on top of other
windows (like the old Windows indicator). Its colour reflects the app state:

    loading   amber      ready   violet
    recording green      idle    red

Interactions:
    * left-click  → open the Settings window (debug_callback)
    * drag        → move the dot
    * right-click → menu (Settings / Quit)

Public API (kept compatible with the app):
    create_overlay_window(debug_callback=None, hotkey_label="…", on_close=None)
        -> (qt_app, window, window)
    update_indicator(qt_app, window, state)   # ready/recording/idle/loading
"""

import sys
import time

from PySide6 import QtCore, QtGui, QtWidgets

from utils.usage import clock_to_utc


# --- colours --------------------------------------------------------------- #
BG = "#0f1115"
PANEL_BG = "#0b0d11"
FG = "#e5e7eb"
MUTED = "#9ca3af"

# state -> (dot colour, label text)
STATE_INFO = {
    "loading":   ("#fbbf24", "Loading model…"),
    "ready":     ("#a78bfa", "Ready"),
    "recording": ("#4ade80", "Recording…"),
    "idle":      ("#f87171", "Idle (auto-paused)"),
}

# Application-wide dark stylesheet (used by the Settings window + menus).
APP_QSS = f"""
QWidget {{
    background-color: {BG};
    color: {FG};
    font-family: -apple-system, "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QPushButton {{
    background-color: #1f2937;
    color: {FG};
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background-color: #374151; }}
QPushButton:pressed {{ background-color: #4b5563; }}
QPlainTextEdit, QTextEdit {{
    background-color: {PANEL_BG};
    color: #cbd5e1;
    border: 1px solid #1f2937;
    border-radius: 8px;
}}
QComboBox {{
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: #1f2937;
    selection-background-color: #4f46e5;
    border: 1px solid #374151;
}}
QMenu {{ background-color: #151923; border: 1px solid #374151; }}
QMenu::item:selected {{ background-color: #4f46e5; }}
QTabWidget::pane {{ border: 1px solid #1f2937; border-radius: 8px; }}
QTabBar::tab {{
    background: #1f2937;
    color: #cbd5e1;
    padding: 7px 14px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ background: #6366f1; color: white; }}
QSlider::groove:horizontal {{
    height: 5px; background: #374151; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: #6366f1; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #a5b4fc; width: 14px; margin: -5px 0; border-radius: 7px;
}}
QScrollBar:vertical {{ background: {PANEL_BG}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #374151; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


class StatusWindow(QtWidgets.QWidget):
    """A single always-on-top, draggable status dot.

    With a meter enabled, the window grows a small translucent panel below the
    dot showing the Claude limit bars and/or today + this-month spend.
    """

    SIZE = 30  # dot-only window is SIZE×SIZE; the dot is inset a few px

    # Layout when the meter panel is shown (below the dot).
    USAGE_W = 96
    DOT_D = 26                      # dot diameter in panel mode
    _PAD = 5
    _ROW_H = 15
    _HEAD_LINE_H = 11               # one line of the header stack beside the dot
    _CMD_H = 17                     # "Commands ▾" footer row height
    # Gutter left of every bar for its one-letter tag. Applied to all bar rows
    # (not just the labelled ones) so every bar keeps the same length.
    _BAR_LABEL_W = 9
    # Hover legend: a second panel that slides out to the LEFT naming what each
    # row measures. _LEGEND_GAP is the transparent seam between the two panels.
    _LEGEND_W = 112
    _LEGEND_GAP = 4
    # Drag-to-zoom: how wide the grab strip on the right/bottom edge is (in
    # widget pixels, so it stays the same physical size at any zoom).
    _RESIZE_EDGE = 6
    MIN_SCALE = 1.0
    MAX_SCALE = 3.0
    _BACKDROP_ALPHA = 140           # panel background opacity (0-255); lower = more see-through

    # Usage-bar fill colour by level: calm under 70%, warning, then alarm.
    _USAGE_OK = "#4ade80"
    _USAGE_WARN = "#fbbf24"
    _USAGE_ALARM = "#f87171"
    # Codex's 7-day bar is always blue, marking it as the non-Claude limit.
    _USAGE_CODEX = "#60a5fa"

    def __init__(self, hotkey_label="the hotkey", debug_callback=None, on_close=None,
                 show_usage=False, show_cost=False, show_codex=False,
                 usage_click_callback=None, macros=None, macro_callback=None,
                 show_dot=True, scale=1.0, on_scale=None):
        super().__init__()
        # Everything below is authored in "design pixels"; paintEvent applies
        # this factor so bars, text and the dot all grow together. Widget
        # coordinates are design * _scale — see _to_design() for hit testing.
        self._scale = min(max(float(scale or 1.0), self.MIN_SCALE), self.MAX_SCALE)
        self._on_scale = on_scale
        self._resize_mode = None     # None | 'right' | 'bottom' | 'corner'
        self._resize_origin = None   # (global QPoint, scale) at drag start
        # The coloured dot reports recording state, so it is only meaningful
        # while speech-to-text is on; with it off the dot would sit on one
        # colour forever. Kept regardless when there is no meter panel, since
        # hiding both would leave an invisible window.
        self._show_dot = bool(show_dot)
        self.hotkey_label = hotkey_label
        self._debug_callback = debug_callback
        self._on_close = on_close
        self._tick_cb = None
        self.state = "loading"

        # Meter panel state. It shows up to three groups of rows:
        #   * usage — Claude session + weekly + Fable ("F") bars, from /usage
        #   * codex — Codex 7-day limit bar (always blue), from rollout logs
        #   * cost  — today + this-month spend (from ccusage)
        self.show_usage = show_usage
        self.show_cost = show_cost
        self.show_codex = show_codex
        self._usage_click_cb = usage_click_callback
        # Typed-command macros: [(label, typed text, hotkey hint), ...].
        # Shown in the "Commands ▾" footer row (panel mode) and the
        # right-click menu.
        self.macros = list(macros or [])
        self._macro_cb = macro_callback
        self.usage_session = None    # int 0-100 or None (not fetched)
        self.usage_week = None
        self.session_reset = None    # 5-hour window reset time, e.g. "8:49pm"
        self.session_reset_utc = None  # the same instant in UTC, e.g. "03:49"
        self.usage_fable = None       # Fable weekly limit %, int 0-100 or None
        self.usage_codex = None       # Codex 7-day limit %, int 0-100 or None
        self.codex_reset = None       # Codex 7-day reset date, e.g. "Jul 17"
        self.cost_today = None        # float USD or None
        self.cost_month = None        # this month so far (month-to-date) USD
        self._press_local = None     # widget-local press point (dot vs panel)
        # Hover legend. While open the window is _LEGEND_W wider and sits that
        # much further left, so the meter itself stays put on screen;
        # _legend_shift is how far it actually moved (less, when clamped at the
        # screen edge) so collapsing can put it back exactly.
        self._legend_open = False
        self._legend_shift = 0

        # Refresh countdown (footer). ``_refresh_deadline`` is a time.monotonic()
        # value; ``note_refreshed`` sets it at the start of each poll's wait.
        self._refresh_interval = None
        self._refresh_deadline = None

        n_rows = ((3 if show_usage else 0) + (1 if show_codex else 0)
                  + (2 if show_cost else 0))
        self._n_rows = n_rows
        self._has_panel = n_rows > 0
        self._has_cmd_row = self._has_panel and bool(self.macros)

        # The header sits right of the dot and stacks one line per reading:
        # reset time, that time in UTC, refresh countdown — or the countdown
        # alone in cost-only mode. It never shrinks below the dot itself.
        self._head_lines = 3 if show_usage else 1
        self._recompute_header()

        self._drag_offset = None
        self._press_pos = None
        self._press_win_pos = None
        self._moved = False
        self._system_move = False

        # Position persistence. moveEvent fires for both manual and native
        # (startSystemMove) drags, so we debounce it and save the resting spot.
        self._on_move = None
        self._user_moving = False       # only persist positions the user set
        self._persist_enabled = False   # armed after the initial placement
        self._last_saved_pos = None
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._emit_move)

        self.setWindowTitle("yiliVoice")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")  # override app QSS bg
        self._apply_size()
        self.setCursor(QtCore.Qt.OpenHandCursor)  # signal the dot is draggable
        self.setMouseTracking(True)  # so the edge can show a resize cursor
        self._refresh_tooltip()

        # Tick the refresh countdown once a second (panel only).
        if self._has_panel:
            self._countdown_timer = QtCore.QTimer(self)
            self._countdown_timer.setInterval(1000)
            self._countdown_timer.timeout.connect(self.update)  # repaint footer
            self._countdown_timer.start()

    def _dot_visible(self):
        """Whether the dot is actually painted (always, in dot-only mode)."""
        return self._show_dot or not self._has_panel

    def _recompute_header(self):
        """Size the header band and the first row's y for the current dot state."""
        floor = self.DOT_D if self._dot_visible() else 0
        self._header_h = max(floor, self._head_lines * self._HEAD_LINE_H)
        self._rows_top = self._PAD + self._header_h + 6

    def _panel_height(self):
        """Panel height in design pixels (before _scale)."""
        h = self._rows_top + self._n_rows * self._ROW_H + self._PAD
        if self._has_cmd_row:
            h += self._CMD_H
        return h

    def _design_size(self):
        """(width, height) in design pixels, legend included when open."""
        if not self._has_panel:
            return (self.SIZE, self.SIZE)
        w = self.USAGE_W + (self._LEGEND_W if self._legend_open else 0)
        return (w, self._panel_height())

    def _apply_size(self):
        """Resize the widget to the current design size times _scale."""
        w, h = self._design_size()
        self.setFixedSize(round(w * self._scale), round(h * self._scale))

    def _to_design(self, pt):
        """Widget point -> design point (undo the zoom)."""
        s = self._scale or 1.0
        return QtCore.QPoint(round(pt.x() / s), round(pt.y() / s))

    def set_scale(self, scale):
        """Zoom the whole overlay; 1.0 is the authored size."""
        scale = min(max(float(scale), self.MIN_SCALE), self.MAX_SCALE)
        if abs(scale - self._scale) < 1e-3:
            return
        self._scale = scale
        self._apply_size()
        self.update()

    def set_dot_visible(self, visible: bool):
        """Show or hide the status dot, reflowing the panel around it."""
        visible = bool(visible)
        if visible == self._show_dot:
            return
        self._show_dot = visible
        self._recompute_header()
        self._apply_size()
        self.update()

    def _dot_rect(self):
        """(x, y, diameter) of the status dot within the window.

        In panel mode the dot sits top-left, leaving the top-right of the
        header for the reset time + refresh countdown (stacked).
        """
        if self._has_panel:
            return (self._PAD, self._PAD, self.DOT_D)
        return (0, 0, self.SIZE)

    def _row_specs(self):
        """Ordered meter rows.

        ``("bar", pct, color, label)`` draws a progress bar + percentage with a
        one-letter tag in the left gutter (``label=None`` leaves the gutter
        blank; ``color=None`` picks the level-based traffic-light fill);
        ``("text", label, value)`` draws a left label + right value.
        """
        rows = []
        if self.show_usage:
            rows.append(("bar", self.usage_session, None, None))
            rows.append(("bar", self.usage_week, None, None))
            rows.append(("bar", self.usage_fable, None, "F"))
        if self.show_codex:
            rows.append(("bar", self.usage_codex, self._USAGE_CODEX, None))
        if self.show_cost:
            rows.append(("text", "Today", self._fmt_cost(self.cost_today)))
            rows.append(("text", "Month", self._fmt_cost(self.cost_month)))
        return rows

    def _legend_specs(self):
        """One caption per meter row, in the same order as ``_row_specs``."""
        rows = []
        if self.show_usage:
            rows.append("Claude 5h window")
            rows.append("Claude week (all)")
            rows.append("Fable week")
        if self.show_codex:
            rows.append("Codex 7 days")
        if self.show_cost:
            rows.append("Spend today")
            rows.append("Spend this month")
        return rows

    def _x0(self):
        """Left offset of the meter inside the window (0 unless the legend is open)."""
        return self._LEGEND_W if self._legend_open else 0

    def _set_legend(self, open_):
        """Slide the legend out to the left (or back in), keeping the meter put.

        The window grows leftward instead of rightward so the meter never
        jumps under the cursor. Near the left screen edge there may not be room
        for the full slide, so the move is clamped and the real distance kept
        in ``_legend_shift``.
        """
        if not self._has_panel or open_ == self._legend_open:
            return
        pos = self.pos()
        # The legend is _LEGEND_W design px, so on screen it is that times the
        # zoom — otherwise a zoomed panel would slide the wrong distance.
        slide = round(self._LEGEND_W * self._scale)
        if open_:
            target_x = pos.x() - slide
            scr = self.screen()
            if scr is not None:
                target_x = max(target_x, scr.availableGeometry().left())
            self._legend_shift = pos.x() - target_x
            self._legend_open = True
            self._apply_size()
            self.move(target_x, pos.y())
        else:
            shift = self._legend_shift
            self._legend_open = False
            self._legend_shift = 0
            self._apply_size()
            self.move(pos.x() + shift, pos.y())
        self.update()

    def _edge_at(self, pos):
        """Which resize edge *pos* (widget px) is on, or None.

        Only the right and bottom edges grab: the legend slides out of the
        left edge, and the window is anchored by its top-left, so growing
        down-right is the one direction that doesn't fight either of those.
        """
        if not self._has_panel:
            return None
        e = self._RESIZE_EDGE
        on_r = pos.x() >= self.width() - e
        on_b = pos.y() >= self.height() - e
        if on_r and on_b:
            return "corner"
        if on_r:
            return "right"
        if on_b:
            return "bottom"
        return None

    _EDGE_CURSORS = {
        "right": QtCore.Qt.SizeHorCursor,
        "bottom": QtCore.Qt.SizeVerCursor,
        "corner": QtCore.Qt.SizeFDiagCursor,
    }

    def _update_cursor(self, pos):
        edge = self._edge_at(pos)
        self.setCursor(self._EDGE_CURSORS.get(edge, QtCore.Qt.OpenHandCursor))

    def _resize_to(self, gpos):
        """Zoom from the drag so far (design size is the reference, not the
        current size, so the factor can't drift as the window grows)."""
        start_pos, start_scale = self._resize_origin
        dw, dh = self._design_size()
        dx = gpos.x() - start_pos.x()
        dy = gpos.y() - start_pos.y()
        cands = []
        if self._resize_mode in ("right", "corner"):
            cands.append((dw * start_scale + dx) / dw)
        if self._resize_mode in ("bottom", "corner"):
            cands.append((dh * start_scale + dy) / dh)
        if cands:
            self.set_scale(max(cands))

    def enterEvent(self, e):
        super().enterEvent(e)
        self._set_legend(True)

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self._set_legend(False)

    # -- public API used by the app ------------------------------------- #

    def set_tick_callback(self, cb):
        """Register a callback run on the GUI thread every UI tick."""
        self._tick_cb = cb

    def set_move_callback(self, cb):
        """Register ``cb(x, y)`` to persist the dot's position after a drag."""
        self._on_move = cb

    def enable_persist(self):
        """Start persisting user drags (call after the initial placement so the
        startup ``move()`` isn't mistaken for a user action)."""
        self._last_saved_pos = (self.pos().x(), self.pos().y())
        self._persist_enabled = True

    def moveEvent(self, e):
        super().moveEvent(e)
        # Debounce: coalesce the stream of positions during a drag and save the
        # final resting spot ~400 ms after motion stops.
        if self._persist_enabled and self._user_moving and self._on_move:
            self._save_timer.start()

    def _emit_move(self):
        self._user_moving = False
        if not self._on_move:
            return
        # While the legend is out the window sits _legend_shift px further
        # left; save where the meter alone would be, or the dot would creep
        # left by that much on every relaunch.
        pos = (self.pos().x() + self._legend_shift, self.pos().y())
        if pos == self._last_saved_pos:
            return
        self._last_saved_pos = pos
        try:
            self._on_move(*pos)
        except Exception as exc:
            sys.__stderr__.write(f"position save error: {exc}\n")

    def set_state(self, state: str) -> None:
        if state not in STATE_INFO:
            return
        self.state = state
        self._refresh_tooltip()
        self.update()  # trigger repaint

    def set_usage(self, session, week, fable=None, session_reset=None) -> None:
        """Update session/weekly/Fable limit % (ints) and the 5-hour reset (str).

        The UTC reading is derived here rather than at paint time — the header
        repaints once a second for the countdown, and the conversion only
        changes when a poll brings a new reset time.
        """
        self.usage_session = session
        self.usage_week = week
        self.usage_fable = fable
        self.session_reset = session_reset
        self.session_reset_utc = clock_to_utc(session_reset)
        self._refresh_tooltip()
        self.update()

    def set_codex_usage(self, week, week_reset=None) -> None:
        """Update Codex's 7-day limit % (int) and its reset date (str)."""
        self.usage_codex = week
        self.codex_reset = week_reset
        self._refresh_tooltip()
        self.update()

    def set_cost(self, today, month) -> None:
        """Update today / this-month spend in USD (floats or None)."""
        self.cost_today = today
        self.cost_month = month
        self._refresh_tooltip()
        self.update()

    def note_refreshed(self, interval_seconds) -> None:
        """Restart the refresh countdown; called at the start of each poll wait."""
        self._refresh_interval = interval_seconds
        self._refresh_deadline = time.monotonic() + interval_seconds
        self.update()

    def _countdown_text(self):
        """"m:ss" until the next poll, or a refreshing marker while it runs."""
        if self._refresh_deadline is None:
            return "↻ --:--"
        remaining = self._refresh_deadline - time.monotonic()
        if remaining <= 0:
            return "↻ …"  # ↻ …  (poll in flight)
        m, s = divmod(int(remaining + 0.5), 60)
        return f"↻ {m}:{s:02d}"

    @staticmethod
    def _fmt_cost(v):
        if v is None:
            return "…"
        if v >= 1000:
            return f"${v / 1000:.2f}k"  # 4252.0 -> "$4.25k"
        return f"${v:.2f}"

    def append_log(self, s: str) -> None:
        """No on-screen log in dot mode — output goes to the terminal."""
        return

    def _refresh_tooltip(self):
        # In panel mode the hover legend already names every row, and the
        # header carries the reset time — a tooltip saying it again just
        # covers the screen under the meter. Dot-only mode has neither, so
        # there the tooltip stays as the one place the readings are shown.
        if self._has_panel:
            self.setToolTip("")
            return

        _, text = STATE_INFO[self.state]
        if self.state in ("ready", "idle"):
            text = f"{text} — {self.hotkey_label}"
        lines = [f"yiliVoice — {text}"]

        def pct(v):
            return f"{v}%" if v is not None else "…"

        if self.show_usage:
            lines.append(
                f"Claude limit — session {pct(self.usage_session)} · "
                f"week {pct(self.usage_week)} · Fable {pct(self.usage_fable)}"
            )
            utc = f" (UTC {self.session_reset_utc})" if self.session_reset_utc else ""
            lines.append(
                f"5-hour window resets at {self.session_reset or '…'}{utc}"
            )
        if self.show_codex:
            reset = f" · resets {self.codex_reset}" if self.codex_reset else ""
            lines.append(f"Codex 7-day limit — {pct(self.usage_codex)}{reset}")
        if self.show_cost:
            lines.append(
                f"Spend — today {self._fmt_cost(self.cost_today)} · "
                f"this month {self._fmt_cost(self.cost_month)}"
            )
        if self._has_panel:
            hint = "click dot: settings · click meter: refresh · right-click: menu"
            if self._has_cmd_row:
                hint = ("click dot: settings · Commands ▾: type a command · "
                        "click meter: refresh · right-click: menu")
            lines.append(hint)
        else:
            lines.append("click: settings   ·   right-click: menu")
        self.setToolTip("\n".join(lines))

    @classmethod
    def _usage_color(cls, pct):
        if pct is None:
            return QtGui.QColor(cls._USAGE_WARN)
        if pct >= 90:
            return QtGui.QColor(cls._USAGE_ALARM)
        if pct >= 70:
            return QtGui.QColor(cls._USAGE_WARN)
        return QtGui.QColor(cls._USAGE_OK)

    # -- painting ------------------------------------------------------- #

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setPen(QtCore.Qt.NoPen)
        # One scale for the whole panel: every constant below stays in design
        # pixels, and fonts scale with it because Qt scales the pen and glyphs.
        if self._scale != 1.0:
            p.scale(self._scale, self._scale)

        if self._legend_open:
            self._paint_legend(p)

        # Everything below is drawn in meter coordinates; when the legend is
        # out, the whole meter is simply translated right past it.
        p.save()
        p.translate(self._x0(), 0)

        if self._has_panel:
            # Rounded translucent backdrop so the rows read on any wallpaper.
            p.setBrush(QtGui.QColor(11, 13, 17, self._BACKDROP_ALPHA))
            p.drawRoundedRect(
                QtCore.QRect(0, 0, self.USAGE_W, self._panel_height()), 10, 10)

        if self._dot_visible():
            dx, dy, d = self._dot_rect()
            # subtle dark halo so the dot stays visible on any background
            p.setBrush(QtGui.QColor(0, 0, 0, 70))
            p.drawEllipse(dx, dy, d, d)
            # coloured status dot
            m = 5
            p.setBrush(QtGui.QColor(STATE_INFO[self.state][0]))
            p.drawEllipse(dx + m, dy + m, d - 2 * m, d - 2 * m)

        if self._has_panel:
            self._paint_header(p)
            self._paint_panel(p)
            if self._has_cmd_row:
                self._paint_cmd_row(p)
        p.restore()
        p.end()

    def _paint_legend(self, p):
        """Captions naming each meter row, right-aligned against the meter."""
        w = self._LEGEND_W - self._LEGEND_GAP
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(11, 13, 17, self._BACKDROP_ALPHA))
        p.drawRoundedRect(QtCore.QRect(0, 0, w, self._panel_height()), 10, 10)

        font = QtGui.QFont()
        font.setPixelSize(9)
        p.setFont(font)
        p.setPen(QtGui.QColor(MUTED))
        text_w = w - 2 * self._PAD
        for i, caption in enumerate(self._legend_specs()):
            y = self._rows_top + i * self._ROW_H
            p.drawText(QtCore.QRect(self._PAD, y, text_w, self._ROW_H),
                       QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, caption)

    def _paint_header(self, p):
        """Right-aligned header stack next to the dot: the 5-hour reset time,
        that same time in UTC, then the refresh countdown."""
        w = self.USAGE_W - 2 * self._PAD
        line = self._HEAD_LINE_H
        right = QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight
        font = QtGui.QFont()
        y = self._PAD

        if self.show_usage:
            # 5-hour reset in local time, then the UTC equivalent under it.
            font.setPixelSize(10)
            p.setFont(font)
            p.setPen(QtGui.QColor(FG))
            p.drawText(QtCore.QRect(self._PAD, y, w, line), right,
                       self.session_reset or "…")
            y += line
            font.setPixelSize(9)
            p.setFont(font)
            p.setPen(QtGui.QColor(MUTED))
            p.drawText(QtCore.QRect(self._PAD, y, w, line), right,
                       f"UTC: {self.session_reset_utc or '--:--'}")
            y += line
            countdown_rect = QtCore.QRect(self._PAD, y, w, line)
        else:
            # No reset time (cost-only) — center the countdown in the header.
            countdown_rect = QtCore.QRect(self._PAD, self._PAD, w, self._header_h)

        font.setPixelSize(9)
        p.setFont(font)
        p.setPen(QtGui.QColor(MUTED))
        p.drawText(countdown_rect, right, self._countdown_text())

    def _paint_panel(self, p):
        font = QtGui.QFont()
        font.setPixelSize(10)
        p.setFont(font)
        for i, spec in enumerate(self._row_specs()):
            row_top = self._rows_top + i * self._ROW_H
            if spec[0] == "bar":
                self._paint_bar_row(p, row_top, spec[1], spec[2], spec[3])
            else:
                self._paint_text_row(p, row_top, spec[1], spec[2])

    def _paint_bar_row(self, p, row_top, pct, color=None, label=None):
        """A progress bar + percentage, with an optional one-letter tag.

        ``color`` forces a fixed fill (Codex's blue); ``None`` picks the
        level-based traffic-light colour (Claude's session/weekly bars).
        ``label`` is drawn in the left gutter — that gutter is reserved on
        every bar row, labelled or not, so all the bars stay the same length
        and line up with each other.
        """
        val_x = self.USAGE_W - self._PAD - 30
        gutter_x = self._PAD + 1
        bar_x = gutter_x + self._BAR_LABEL_W
        bar_w = val_x - bar_x - 6
        bar_h = 6
        bar_y = row_top + (self._ROW_H - bar_h) // 2

        if label:
            p.setPen(QtGui.QColor(MUTED))
            p.drawText(QtCore.QRect(gutter_x, row_top, self._BAR_LABEL_W, self._ROW_H),
                       QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, label)

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(55, 65, 81))  # #374151 track
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)
        if pct is not None:
            fill = int(bar_w * max(0, min(100, pct)) / 100)
            if fill > 0:
                p.setBrush(QtGui.QColor(color) if color else self._usage_color(pct))
                p.drawRoundedRect(bar_x, bar_y, fill, bar_h, 3, 3)

        p.setPen(QtGui.QColor(FG))
        text = f"{pct}%" if pct is not None else "…"
        p.drawText(QtCore.QRect(val_x, row_top, 30, self._ROW_H),
                   QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, text)

    def _cmd_row_top(self):
        """y where the "Commands ▾" footer row starts."""
        return self._rows_top + self._n_rows * self._ROW_H

    def _paint_cmd_row(self, p):
        """Dropdown-style footer button that opens the typed-commands menu."""
        r = QtCore.QRect(self._PAD, self._cmd_row_top() + 2,
                         self.USAGE_W - 2 * self._PAD, self._CMD_H - 4)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(55, 65, 81, 130))  # #374151, translucent
        p.drawRoundedRect(r, 5, 5)
        font = QtGui.QFont()
        font.setPixelSize(9)
        p.setFont(font)
        p.setPen(QtGui.QColor(FG))
        p.drawText(r, QtCore.Qt.AlignCenter, "Commands ▾")

    def _paint_text_row(self, p, row_top, label, value):
        p.setPen(QtGui.QColor(MUTED))
        p.drawText(QtCore.QRect(self._PAD + 1, row_top, 40, self._ROW_H),
                   QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, label)
        val_x = self._PAD + 30
        p.setPen(QtGui.QColor(FG))
        p.drawText(QtCore.QRect(val_x, row_top, self.USAGE_W - val_x - self._PAD, self._ROW_H),
                   QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, value)

    # -- mouse: click → settings, drag → move, right-click → menu ------- #

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            edge = self._edge_at(e.position().toPoint())
            if edge is not None:
                # Zoom drag, not a move drag: claim the press and leave the
                # move/click state untouched so releasing can't be read as a
                # click on whatever sits under the edge.
                self._resize_mode = edge
                self._resize_origin = (e.globalPosition().toPoint(), self._scale)
                e.accept()
                return
            self._press_pos = e.globalPosition().toPoint()
            # In meter coordinates, so the dot / Commands hit tests below don't
            # have to know whether the legend is out.
            self._press_local = (self._to_design(e.position().toPoint())
                                 - QtCore.QPoint(self._x0(), 0))
            self._press_win_pos = self.pos()
            self._drag_offset = self._press_pos - self.frameGeometry().topLeft()
            self._moved = False
            self._system_move = False
            self._user_moving = True  # so moveEvent knows this drag is user-driven
            e.accept()

    def mouseMoveEvent(self, e):
        if self._resize_mode is not None:
            self._resize_to(e.globalPosition().toPoint())
            e.accept()
            return
        if not (e.buttons() & QtCore.Qt.LeftButton):
            self._update_cursor(e.position().toPoint())  # hover feedback
            return
        if self._drag_offset is None:
            return
        gp = e.globalPosition().toPoint()
        # Wait until the pointer clearly moves so a small wobble on click still
        # counts as a click (which opens Settings).
        if not self._moved and (gp - self._press_pos).manhattanLength() > 4:
            self._moved = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            # Hand the drag to the native window manager: manual move() on a
            # frameless, always-on-top, translucent overlay is laggy/unreliable
            # on macOS, whereas startSystemMove tracks the cursor smoothly.
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                self._system_move = True
                self._drag_offset = None  # the OS owns the drag now
                e.accept()
                return
        if not self._system_move and self._drag_offset is not None:
            self.move(gp - self._drag_offset)  # fallback: manual move
        e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and self._resize_mode is not None:
            self._resize_mode = None
            self._resize_origin = None
            self._update_cursor(e.position().toPoint())
            if self._on_scale:
                try:
                    self._on_scale(self._scale)
                except Exception as exc:
                    print(f"Could not save the overlay size: {exc}")
            e.accept()
            return
        if e.button() == QtCore.Qt.LeftButton:
            # Click vs drag: use the movement flag, but also compare the net
            # window displacement, since after a native move mouseMoveEvent may
            # not have fired on this widget.
            displaced = (
                self._press_win_pos is not None
                and (self.pos() - self._press_win_pos).manhattanLength() > 4
            )
            was_click = not (self._moved or displaced)
            local = self._press_local
            self._drag_offset = None
            self._press_pos = None
            self._press_local = None
            self._press_win_pos = None
            self._moved = False
            self._system_move = False
            self.setCursor(QtCore.Qt.OpenHandCursor)
            if was_click:
                # Clicking the dot opens Settings; the "Commands ▾" footer
                # opens the typed-commands menu; anywhere else on the meter
                # (header reset, bars, spend) refreshes it.
                # The dot's corner still opens Settings when the dot isn't
                # drawn — otherwise a speech-off user's only way back to the
                # toggle is the right-click menu.
                dx, dy, d = self._dot_rect()
                on_dot = local is not None and QtCore.QRect(dx, dy, d, d).contains(local)
                on_cmds = (self._has_cmd_row and local is not None
                           and local.y() >= self._cmd_row_top())
                if on_cmds:
                    self._show_macro_menu()
                elif self._has_panel and not on_dot and self._usage_click_cb:
                    self._usage_click_cb()
                elif self._debug_callback:
                    self._debug_callback()
            e.accept()

    def _macro_menu_entries(self, menu):
        """Fill *menu* with one action per macro; returns {action: typed text}.

        The menu shows the short label; picking it types the (possibly much
        longer) text.
        """
        entries = {}
        for label, text, hint in self.macros:
            # "\t" puts the hotkey hint in the menu's shortcut column.
            act = menu.addAction(f"{label}\t{hint}" if hint else label)
            entries[act] = text
        return entries

    def _show_macro_menu(self):
        """Pop the typed-commands menu (from the "Commands ▾" footer)."""
        if not self.macros:
            return
        menu = QtWidgets.QMenu(self)
        entries = self._macro_menu_entries(menu)
        chosen = menu.exec(QtGui.QCursor.pos())
        if chosen in entries and self._macro_cb:
            self._macro_cb(entries[chosen])

    def contextMenuEvent(self, e):
        menu = QtWidgets.QMenu(self)
        act_settings = menu.addAction("Settings")
        entries = {}
        if self.macros:
            entries = self._macro_menu_entries(menu.addMenu("Type Command"))
        menu.addSeparator()
        act_quit = menu.addAction("Quit yiliVoice")
        chosen = menu.exec(e.globalPos())
        if chosen == act_settings and self._debug_callback:
            self._debug_callback()
        elif chosen in entries and self._macro_cb:
            self._macro_cb(entries[chosen])
        elif chosen == act_quit:
            self._quit()

    # -- lifecycle ------------------------------------------------------ #

    def _quit(self):
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                pass
        QtWidgets.QApplication.quit()

    def closeEvent(self, event):
        self._quit()
        event.accept()


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("yiliVoice")
    app.setStyleSheet(APP_QSS)
    # The dot is a frameless overlay; closing the Settings window must NOT quit
    # the app. We quit explicitly (dot right-click → Quit, or Ctrl+C).
    app.setQuitOnLastWindowClosed(False)
    return app


_DEFAULT_MARGIN = (40, 60)  # from the primary screen's top-left


def _default_pos(size):
    """Top-left-ish spot on the primary screen (available area)."""
    screen = QtGui.QGuiApplication.primaryScreen()
    if screen is None:
        return _DEFAULT_MARGIN
    g = screen.availableGeometry()
    return (g.left() + _DEFAULT_MARGIN[0], g.top() + _DEFAULT_MARGIN[1])


def _resolve_start_pos(saved_x, saved_y, w, h):
    """Return a valid on-screen start position for the overlay (``w``×``h``).

    Multi-monitor safe: the saved coordinates live in the virtual desktop that
    spans every monitor.  If they still land on a connected screen we keep them
    (clamped so the whole window is visible even after a resolution change); if
    that monitor is gone — unplugged, or the layout changed so the spot is now
    in dead space — we fall back to the primary screen's default corner.
    """
    if saved_x is None or saved_y is None:
        return _default_pos(w)

    rect = QtCore.QRect(int(saved_x), int(saved_y), w, h)
    screens = QtGui.QGuiApplication.screens()

    # Pick the screen the window overlaps most; none => it's in dead space.
    best, best_area = None, 0
    for s in screens:
        inter = s.geometry().intersected(rect)
        area = inter.width() * inter.height()
        if area > best_area:
            best, best_area = s, area
    if best is None or best_area <= 0:
        return _default_pos(w)

    # Clamp fully onto that screen (handles a shrunk/rotated display).
    g = best.geometry()
    x = min(max(int(saved_x), g.left()), g.left() + g.width() - w)
    y = min(max(int(saved_y), g.top()), g.top() + g.height() - h)
    return (x, y)


def create_overlay_window(debug_callback=None, hotkey_label="the hotkey",
                          on_close=None, initial_pos=None, on_move=None,
                          show_usage=False, show_cost=False, show_codex=False,
                          show_dot=True, scale=1.0, on_scale=None,
                          usage_click_callback=None, macros=None,
                          macro_callback=None):
    """Create the QApplication (if needed) and the floating status dot.

    ``initial_pos`` is a saved ``(x, y)`` (or ``None``); it is validated
    against the current monitor layout before use.  ``on_move(x, y)`` is
    called (debounced) whenever the user drags the dot, to persist its spot.
    ``show_usage`` adds the Claude limit bars below the dot, ``show_codex`` the
    blue Codex 7-day limit bar, and ``show_cost`` the ccusage spend rows;
    ``usage_click_callback`` is invoked when the user clicks the meter.
    ``macros`` is ``[(label, text, hotkey_hint), ...]`` for the "Commands ▾"
    footer (also in the right-click menu); the menu shows ``label`` and picking
    one calls ``macro_callback(text)``.

    Returns ``(qt_app, window, window)``.
    """
    qt_app = _ensure_app()

    window = StatusWindow(
        hotkey_label=hotkey_label,
        debug_callback=debug_callback,
        on_close=on_close,
        show_usage=show_usage,
        show_cost=show_cost,
        show_codex=show_codex,
        show_dot=show_dot,
        scale=scale,
        on_scale=on_scale,
        usage_click_callback=usage_click_callback,
        macros=macros,
        macro_callback=macro_callback,
    )
    saved_x, saved_y = (initial_pos or (None, None))
    start_x, start_y = _resolve_start_pos(
        saved_x, saved_y, window.width(), window.height()
    )
    window.move(start_x, start_y)
    window.show()
    window.raise_()
    # Arm persistence only after the startup placement so it isn't saved back.
    if on_move is not None:
        window.set_move_callback(on_move)
    window.enable_persist()

    # A GUI-thread timer runs the app's per-tick work (indicator updates,
    # deferred hotkey start) and keeps Python servicing Ctrl+C during exec().
    def _pump():
        if window._tick_cb is not None:
            try:
                window._tick_cb()
            except Exception as exc:
                sys.__stderr__.write(f"UI tick error: {exc}\n")

    timer = QtCore.QTimer(window)
    timer.timeout.connect(_pump)
    timer.start(40)
    window._pump_timer = timer

    def teardown():
        try:
            timer.stop()
        except Exception:
            pass

    window.teardown = teardown

    return qt_app, window, window


def update_indicator(qt_app, window, state):
    """Update the dot to a named state (ready/recording/idle/loading)."""
    if hasattr(window, "set_state"):
        window.set_state(state)
