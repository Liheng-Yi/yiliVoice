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

from PySide6 import QtCore, QtGui, QtWidgets


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
    dot showing the Claude limit bars and/or today + last-30-day spend.
    """

    SIZE = 30  # dot-only window is SIZE×SIZE; the dot is inset a few px

    # Layout when the meter panel is shown (below the dot).
    USAGE_W = 108
    DOT_D = 26                      # dot diameter in panel mode
    _PAD = 5
    _ROW_H = 15
    _ROWS_TOP = _PAD + DOT_D + 6    # y where the first meter row starts
    _BACKDROP_ALPHA = 140           # panel background opacity (0-255); lower = more see-through

    # Usage-bar fill colour by level: calm under 70%, warning, then alarm.
    _USAGE_OK = "#4ade80"
    _USAGE_WARN = "#fbbf24"
    _USAGE_ALARM = "#f87171"

    def __init__(self, hotkey_label="the hotkey", debug_callback=None, on_close=None,
                 show_usage=False, show_cost=False, usage_click_callback=None):
        super().__init__()
        self.hotkey_label = hotkey_label
        self._debug_callback = debug_callback
        self._on_close = on_close
        self._tick_cb = None
        self.state = "loading"

        # Meter panel state. It shows up to two groups of rows:
        #   * usage — session + weekly limit bars (no letters), from /usage
        #   * cost  — today + last-30-day spend (from ccusage)
        self.show_usage = show_usage
        self.show_cost = show_cost
        self._usage_click_cb = usage_click_callback
        self.usage_session = None    # int 0-100 or None (not fetched)
        self.usage_week = None
        self.cost_today = None        # float USD or None
        self.cost_month = None        # rolling last-30-days USD
        self._press_local = None     # widget-local press point (dot vs panel)

        n_rows = (2 if show_usage else 0) + (2 if show_cost else 0)
        self._n_rows = n_rows
        self._has_panel = n_rows > 0

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
        if self._has_panel:
            height = self._ROWS_TOP + self._n_rows * self._ROW_H + self._PAD
            self.setFixedSize(self.USAGE_W, height)
        else:
            self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(QtCore.Qt.OpenHandCursor)  # signal the dot is draggable
        self._refresh_tooltip()

    def _dot_rect(self):
        """(x, y, diameter) of the status dot within the window."""
        if self._has_panel:
            return ((self.USAGE_W - self.DOT_D) // 2, self._PAD, self.DOT_D)
        return (0, 0, self.SIZE)

    def _row_specs(self):
        """Ordered meter rows.

        ``("bar", pct)`` draws a label-less progress bar + percentage;
        ``("text", label, value)`` draws a left label + right value.
        """
        rows = []
        if self.show_usage:
            rows.append(("bar", self.usage_session))
            rows.append(("bar", self.usage_week))
        if self.show_cost:
            rows.append(("text", "Today", self._fmt_cost(self.cost_today)))
            rows.append(("text", "30d", self._fmt_cost(self.cost_month)))
        return rows

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
        pos = (self.pos().x(), self.pos().y())
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

    def set_usage(self, session, week) -> None:
        """Update the session / weekly limit percentages (ints or None)."""
        self.usage_session = session
        self.usage_week = week
        self._refresh_tooltip()
        self.update()

    def set_cost(self, today, month) -> None:
        """Update today / last-30-day spend in USD (floats or None)."""
        self.cost_today = today
        self.cost_month = month
        self._refresh_tooltip()
        self.update()

    @staticmethod
    def _fmt_cost(v):
        return "…" if v is None else f"${v:,.2f}"

    def append_log(self, s: str) -> None:
        """No on-screen log in dot mode — output goes to the terminal."""
        return

    def _refresh_tooltip(self):
        _, text = STATE_INFO[self.state]
        if self.state in ("ready", "idle"):
            text = f"{text} — {self.hotkey_label}"
        lines = [f"yiliVoice — {text}"]
        if self.show_usage:
            def pct(v):
                return f"{v}%" if v is not None else "…"
            lines.append(
                f"Claude limit — session {pct(self.usage_session)} · "
                f"week {pct(self.usage_week)}"
            )
        if self.show_cost:
            lines.append(
                f"Spend — today {self._fmt_cost(self.cost_today)} · "
                f"last 30 days {self._fmt_cost(self.cost_month)}"
            )
        if self._has_panel:
            lines.append("click dot: settings · click meter: refresh · right-click: menu")
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

        if self._has_panel:
            # Rounded translucent backdrop so the rows read on any wallpaper.
            p.setBrush(QtGui.QColor(11, 13, 17, self._BACKDROP_ALPHA))
            p.drawRoundedRect(self.rect(), 10, 10)

        dx, dy, d = self._dot_rect()
        # subtle dark halo so the dot stays visible on any background
        p.setBrush(QtGui.QColor(0, 0, 0, 70))
        p.drawEllipse(dx, dy, d, d)
        # coloured status dot
        m = 5
        p.setBrush(QtGui.QColor(STATE_INFO[self.state][0]))
        p.drawEllipse(dx + m, dy + m, d - 2 * m, d - 2 * m)

        if self._has_panel:
            self._paint_panel(p)
        p.end()

    def _paint_panel(self, p):
        font = QtGui.QFont()
        font.setPixelSize(10)
        p.setFont(font)
        for i, spec in enumerate(self._row_specs()):
            row_top = self._ROWS_TOP + i * self._ROW_H
            if spec[0] == "bar":
                self._paint_bar_row(p, row_top, spec[1])
            else:
                self._paint_text_row(p, row_top, spec[1], spec[2])

    def _paint_bar_row(self, p, row_top, pct):
        """A label-less progress bar (session/weekly limit) + percentage."""
        val_x = self.USAGE_W - self._PAD - 30
        bar_x = self._PAD + 1
        bar_w = val_x - bar_x - 6
        bar_h = 6
        bar_y = row_top + (self._ROW_H - bar_h) // 2

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(55, 65, 81))  # #374151 track
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)
        if pct is not None:
            fill = int(bar_w * max(0, min(100, pct)) / 100)
            if fill > 0:
                p.setBrush(self._usage_color(pct))
                p.drawRoundedRect(bar_x, bar_y, fill, bar_h, 3, 3)

        p.setPen(QtGui.QColor(FG))
        text = f"{pct}%" if pct is not None else "…"
        p.drawText(QtCore.QRect(val_x, row_top, 30, self._ROW_H),
                   QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, text)

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
            self._press_pos = e.globalPosition().toPoint()
            self._press_local = e.position().toPoint()
            self._press_win_pos = self.pos()
            self._drag_offset = self._press_pos - self.frameGeometry().topLeft()
            self._moved = False
            self._system_move = False
            self._user_moving = True  # so moveEvent knows this drag is user-driven
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_offset is None or not (e.buttons() & QtCore.Qt.LeftButton):
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
                # Clicking the dot opens Settings; clicking anywhere else on
                # the meter (header reset, bars, spend) refreshes it.
                dx, dy, d = self._dot_rect()
                on_dot = local is not None and QtCore.QRect(dx, dy, d, d).contains(local)
                if self._has_panel and not on_dot and self._usage_click_cb:
                    self._usage_click_cb()
                elif self._debug_callback:
                    self._debug_callback()
            e.accept()

    def contextMenuEvent(self, e):
        menu = QtWidgets.QMenu(self)
        act_settings = menu.addAction("Settings")
        menu.addSeparator()
        act_quit = menu.addAction("Quit yiliVoice")
        chosen = menu.exec(e.globalPos())
        if chosen == act_settings and self._debug_callback:
            self._debug_callback()
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
                          show_usage=False, show_cost=False, usage_click_callback=None):
    """Create the QApplication (if needed) and the floating status dot.

    ``initial_pos`` is a saved ``(x, y)`` (or ``None``); it is validated
    against the current monitor layout before use.  ``on_move(x, y)`` is
    called (debounced) whenever the user drags the dot, to persist its spot.
    ``show_usage`` adds the Claude limit bars below the dot and ``show_cost``
    the ccusage spend rows; ``usage_click_callback`` is invoked when the user
    clicks the meter.

    Returns ``(qt_app, window, window)``.
    """
    qt_app = _ensure_app()

    window = StatusWindow(
        hotkey_label=hotkey_label,
        debug_callback=debug_callback,
        on_close=on_close,
        show_usage=show_usage,
        show_cost=show_cost,
        usage_click_callback=usage_click_callback,
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
