"""Main status + log window, built on PySide6 (Qt).

Qt renders reliably on macOS — unlike Apple's deprecated system Tcl/Tk 8.5,
which left plain ``tk`` widgets unpainted (only native buttons showed). This
module replaces the old Tk/customtkinter UI.

Public API (kept compatible with the app):
    create_overlay_window(debug_callback=None, hotkey_label="…", on_close=None)
        -> (qt_app, window, window)
    update_indicator(qt_app, window, state)   # state: ready/recording/idle/loading

``window`` is returned twice so the app's historical ``(canvas, indicator)``
pair both point at the same :class:`StatusWindow`.
"""

import sys
import queue as _queue

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

# Application-wide dark stylesheet (shared by the status + settings windows).
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


class _LogTee:
    """Write-through stdout/stderr wrapper that also feeds a queue."""

    def __init__(self, original, q):
        self._orig = original
        self._q = q

    def write(self, s):
        if self._orig:
            try:
                self._orig.write(s)
            except Exception:
                pass
        try:
            self._q.put(s)
        except Exception:
            pass

    def flush(self):
        if self._orig:
            try:
                self._orig.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class StatusWindow(QtWidgets.QWidget):
    """Status header (coloured dot + text) + a live, scrolling log."""

    def __init__(self, hotkey_label="the hotkey", debug_callback=None, on_close=None):
        super().__init__()
        self.hotkey_label = hotkey_label
        self._on_close = on_close
        self._tick_cb = None
        self.state = "loading"

        self.setWindowTitle("yiliVoice")
        self.setMinimumSize(440, 320)
        self.resize(600, 460)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        # ---- header: coloured dot + status text ----------------------- #
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)
        self.dot = QtWidgets.QLabel("●")
        self.status_label = QtWidgets.QLabel("Loading model…")
        header.addWidget(self.dot)
        header.addWidget(self.status_label)
        header.addStretch(1)
        layout.addLayout(header)

        # ---- hotkey hint ---------------------------------------------- #
        self.hint = QtWidgets.QLabel(
            f"Press {hotkey_label} to start / stop recording."
        )
        self.hint.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.hint)

        # ---- buttons -------------------------------------------------- #
        btns = QtWidgets.QHBoxLayout()
        if debug_callback:
            settings_btn = QtWidgets.QPushButton("Settings")
            settings_btn.clicked.connect(lambda: debug_callback())
            btns.addWidget(settings_btn)
        clear_btn = QtWidgets.QPushButton("Clear log")
        clear_btn.clicked.connect(self.clear_log)
        btns.addWidget(clear_btn)
        btns.addStretch(1)
        layout.addLayout(btns)

        # ---- log ------------------------------------------------------ #
        log_caption = QtWidgets.QLabel("Log")
        log_caption.setStyleSheet(f"color: {MUTED}; font-weight: 600;")
        layout.addWidget(log_caption)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)  # cap memory; drop oldest lines
        mono = QtGui.QFont("Menlo")
        mono.setStyleHint(QtGui.QFont.Monospace)
        mono.setPointSize(11)
        self.log.setFont(mono)
        layout.addWidget(self.log, 1)

        self._apply_state_styles()

    # -- public API used by the app ------------------------------------- #

    def set_tick_callback(self, cb):
        """Register a callback run on the GUI thread every UI tick."""
        self._tick_cb = cb

    def set_state(self, state: str) -> None:
        if state not in STATE_INFO:
            return
        self.state = state
        self._apply_state_styles()

    def _apply_state_styles(self) -> None:
        color, text = STATE_INFO[self.state]
        if self.state in ("ready", "idle"):
            text = f"{text} — press {self.hotkey_label}"
        self.dot.setStyleSheet(f"color: {color}; font-size: 20px;")
        text_color = FG if self.state == "loading" else color
        self.status_label.setStyleSheet(
            f"color: {text_color}; font-size: 15px; font-weight: 700;"
        )
        self.status_label.setText(text)

    def append_log(self, s: str) -> None:
        """Append text to the log. Must be called on the GUI thread."""
        if not s:
            return
        bar = self.log.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.insertPlainText(s)
        if at_bottom:
            self.log.moveCursor(QtGui.QTextCursor.End)
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def clear_log(self) -> None:
        self.log.clear()

    # -- lifecycle ------------------------------------------------------ #

    def closeEvent(self, event):
        """Window close asks the app to shut down, then quits the Qt loop."""
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                pass
        QtWidgets.QApplication.quit()
        event.accept()


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("yiliVoice")
    app.setStyleSheet(APP_QSS)
    return app


def create_overlay_window(debug_callback=None, hotkey_label="the hotkey", on_close=None):
    """Create the QApplication (if needed) and the main status/log window.

    Returns ``(qt_app, window, window)``.
    """
    qt_app = _ensure_app()

    window = StatusWindow(
        hotkey_label=hotkey_label,
        debug_callback=debug_callback,
        on_close=on_close,
    )
    window.move(20, 40)  # top-left, clear of the menu bar
    window.show()
    window.raise_()
    window.activateWindow()

    # Mirror stdout/stderr into the on-screen log (keeps the terminal too).
    log_q = _queue.Queue()
    tee_out = _LogTee(sys.__stdout__, log_q)
    tee_err = _LogTee(sys.__stderr__, log_q)
    sys.stdout = tee_out
    sys.stderr = tee_err

    # A single GUI-thread timer drains the log queue and runs the app's
    # per-tick work (indicator updates, deferred hotkey start). Keeping a
    # short interval also lets Python service Ctrl+C while Qt's loop runs.
    def _pump():
        try:
            while True:
                window.append_log(log_q.get_nowait())
        except _queue.Empty:
            pass
        if window._tick_cb is not None:
            try:
                window._tick_cb()
            except Exception as exc:
                # Never let a tick error kill the timer.
                sys.__stderr__.write(f"UI tick error: {exc}\n")

    timer = QtCore.QTimer(window)
    timer.timeout.connect(_pump)
    timer.start(40)
    window._pump_timer = timer

    def teardown():
        """Stop the pump and restore stdio. Idempotent."""
        try:
            timer.stop()
        except Exception:
            pass
        if sys.stdout is tee_out:
            sys.stdout = sys.__stdout__
        if sys.stderr is tee_err:
            sys.stderr = sys.__stderr__

    window.teardown = teardown

    return qt_app, window, window


def update_indicator(qt_app, window, state):
    """Update the status window to a named state (ready/recording/idle/loading).

    Safe to call from the GUI-thread tick (which is how the app drives it).
    """
    if hasattr(window, "set_state"):
        window.set_state(state)
