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
    """A single always-on-top, draggable status dot."""

    SIZE = 30  # window is SIZE×SIZE; the dot is inset a few px

    def __init__(self, hotkey_label="the hotkey", debug_callback=None, on_close=None):
        super().__init__()
        self.hotkey_label = hotkey_label
        self._debug_callback = debug_callback
        self._on_close = on_close
        self._tick_cb = None
        self.state = "loading"

        self._drag_offset = None
        self._press_pos = None
        self._moved = False

        self.setWindowTitle("yiliVoice")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")  # override app QSS bg
        self.setFixedSize(self.SIZE, self.SIZE)
        self._refresh_tooltip()

    # -- public API used by the app ------------------------------------- #

    def set_tick_callback(self, cb):
        """Register a callback run on the GUI thread every UI tick."""
        self._tick_cb = cb

    def set_state(self, state: str) -> None:
        if state not in STATE_INFO:
            return
        self.state = state
        self._refresh_tooltip()
        self.update()  # trigger repaint

    def append_log(self, s: str) -> None:
        """No on-screen log in dot mode — output goes to the terminal."""
        return

    def _refresh_tooltip(self):
        _, text = STATE_INFO[self.state]
        if self.state in ("ready", "idle"):
            text = f"{text} — {self.hotkey_label}"
        self.setToolTip(
            f"yiliVoice — {text}\nclick: settings   ·   right-click: menu"
        )

    # -- painting ------------------------------------------------------- #

    def paintEvent(self, event):
        color = QtGui.QColor(STATE_INFO[self.state][0])
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        d = self.SIZE
        p.setPen(QtCore.Qt.NoPen)
        # subtle dark halo so the dot stays visible on any background
        p.setBrush(QtGui.QColor(0, 0, 0, 90))
        p.drawEllipse(2, 2, d - 4, d - 4)
        # coloured status dot
        m = 5
        p.setBrush(color)
        p.drawEllipse(m, m, d - 2 * m, d - 2 * m)
        p.end()

    # -- mouse: click → settings, drag → move, right-click → menu ------- #

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._press_pos = e.globalPosition().toPoint()
            self._drag_offset = self._press_pos - self.frameGeometry().topLeft()
            self._moved = False
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_offset is not None and (e.buttons() & QtCore.Qt.LeftButton):
            gp = e.globalPosition().toPoint()
            if (gp - self._press_pos).manhattanLength() > 4:
                self._moved = True
            self.move(gp - self._drag_offset)
            e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            was_click = not self._moved
            self._drag_offset = None
            self._press_pos = None
            self._moved = False
            if was_click and self._debug_callback:
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


def create_overlay_window(debug_callback=None, hotkey_label="the hotkey", on_close=None):
    """Create the QApplication (if needed) and the floating status dot.

    Returns ``(qt_app, window, window)``.
    """
    qt_app = _ensure_app()

    window = StatusWindow(
        hotkey_label=hotkey_label,
        debug_callback=debug_callback,
        on_close=on_close,
    )
    window.move(40, 60)  # near the top-left; drag to taste
    window.show()
    window.raise_()

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
