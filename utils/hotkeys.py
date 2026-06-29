"""Cross-platform global hotkeys.

Two backends behind one callback-based interface:

  * ``keyboard`` (Windows): the original library.  Works without admin
    rights on Windows.
  * ``pynput`` (macOS/Linux): triggers on the *real* OS event stream and,
    crucially on macOS, works **without sudo** — it only needs the
    Accessibility / Input-Monitoring permission granted once.

Logical actions (``toggle_recording`` etc.) are mapped to OS-appropriate
key combos by the :class:`~settings.platform_profile.PlatformProfile`, so
the app registers callbacks by action name and never hard-codes a key.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time


_macos_layout_patched = False


def prewarm_macos_key_layout():
    """Cache the macOS keyboard layout on the main thread, once.

    pynput translates each key event to a character via
    ``keycode_context()`` → ``TISCopyCurrentKeyboardInputSource`` /
    ``TISGetInputSourceProperty``. On macOS 14+ those HIToolbox/TSM calls
    assert they run on the main thread (``dispatch_assert_queue``), but pynput
    calls them from its background listener thread — aborting the whole
    process with SIGTRAP ("trace trap"). The translation itself
    (``UCKeyTranslate``) is a pure function over the layout bytes and is
    thread-safe, so we fetch the layout once here (on the main thread) and
    patch ``keycode_context`` to hand that cached value to every later
    translation, keeping TIS off the listener thread entirely.

    Must be called from the main thread, before the listener starts. A no-op
    off macOS and after the first successful call. (If the user switches
    keyboard layout mid-session the cache is stale, but the app's hotkeys are
    ASCII so this is harmless.)
    """
    global _macos_layout_patched
    if _macos_layout_patched or sys.platform != "darwin":
        return
    try:
        from pynput._util import darwin as _pd
        from pynput.keyboard import _darwin as _kd

        with _pd.keycode_context() as ctx:
            cached = ctx  # (keyboard_type: int, layout_data: bytes)

        @contextlib.contextmanager
        def _cached_keycode_context():
            yield cached

        # Patch both bindings: the per-event path imports the name into
        # keyboard._darwin, while get_unicode_to_keycode_map uses the one in
        # _util.darwin.
        _pd.keycode_context = _cached_keycode_context
        _kd.keycode_context = _cached_keycode_context
        _macos_layout_patched = True
    except Exception as exc:
        print(f"[Hotkeys] macOS key-layout pre-warm skipped: {exc}")


class _DoubleTapTracker:
    """State machine: ``feed()`` returns True on the 2nd hit within *window*."""

    def __init__(self, window: float = 0.35):
        self.window = window
        self.last = None

    def feed(self, now: float) -> bool:
        if self.last is not None and (now - self.last) <= self.window:
            self.last = None  # reset so a 3rd tap doesn't immediately re-fire
            return True
        self.last = now
        return False


def macos_accessibility_trusted(prompt: bool = False):
    """Whether this process may monitor global input on macOS.

    Returns True/False, or None if it can't be determined. With
    ``prompt=True`` and no trust yet, macOS shows its Accessibility dialog
    (which also registers the app in the permission list so the user can
    enable it).
    """
    import sys
    if sys.platform != "darwin":
        return True
    try:
        if prompt:
            from ApplicationServices import AXIsProcessTrustedWithOptions
            try:
                from ApplicationServices import kAXTrustedCheckOptionPrompt as KEY
            except Exception:
                KEY = "AXTrustedCheckOptionPrompt"
            return bool(AXIsProcessTrustedWithOptions({KEY: True}))
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:
        return None


def macos_input_monitoring_trusted(prompt: bool = False):
    """Whether this process may *listen* to global key events on macOS.

    This is the **Input Monitoring** permission (System Settings → Privacy &
    Security → Input Monitoring), which pynput's keyboard listener needs to
    receive key events. It is SEPARATE from Accessibility (which covers typing
    output / posting events): a process can have Accessibility yet receive no
    key events because Input Monitoring is off — the listener then sits silent.

    Returns True/False, or None if it can't be determined. With ``prompt=True``
    and no grant yet, macOS registers this app in the Input Monitoring list and
    shows its prompt.
    """
    if sys.platform != "darwin":
        return True
    try:
        import ctypes
        iokit = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/IOKit.framework/IOKit"
        )
        LISTEN_EVENT = 1   # kIOHIDRequestTypeListenEvent
        GRANTED = 0        # kIOHIDAccessTypeGranted
        iokit.IOHIDCheckAccess.restype = ctypes.c_int
        iokit.IOHIDCheckAccess.argtypes = [ctypes.c_int]
        status = iokit.IOHIDCheckAccess(LISTEN_EVENT)
        if status != GRANTED and prompt:
            iokit.IOHIDRequestAccess.restype = ctypes.c_bool
            iokit.IOHIDRequestAccess.argtypes = [ctypes.c_int]
            iokit.IOHIDRequestAccess(LISTEN_EVENT)
        return status == GRANTED
    except Exception:
        return None


def _make_key_predicate(key_name: str):
    """Return a predicate matching a pynput key event for *key_name*."""
    from pynput import keyboard

    specials = {
        "enter": keyboard.Key.enter,
        "return": keyboard.Key.enter,
        "space": keyboard.Key.space,
        "tab": keyboard.Key.tab,
        "esc": keyboard.Key.esc,
        "escape": keyboard.Key.esc,
    }
    if key_name in specials:
        target = specials[key_name]
        return lambda k: k == target
    return lambda k: getattr(k, "char", None) == key_name


class HotkeyManager:
    """Base class: callback registry + debounce.  Subclasses do the binding."""

    backend_name = "none"

    def __init__(self, profile, debounce: float = 0.4):
        self.profile = profile
        self.bindings = dict(profile.hotkeys)        # action -> combo string
        self.labels = dict(profile.hotkey_labels)    # action -> human label
        self.debounce = debounce
        self._callbacks = {}                         # action -> fn
        self._last_fire = {}                         # action -> monotonic ts
        self._on_activity = None
        self._started = False

    # -- registration ----------------------------------------------------- #

    def register(self, action: str, callback) -> None:
        if action not in self.bindings:
            print(f"[Hotkeys] No binding defined for action '{action}'")
            return
        self._callbacks[action] = callback

    def set_activity_callback(self, fn) -> None:
        """Called on *any* hotkey press (used to wake from idle)."""
        self._on_activity = fn

    def describe(self) -> list[tuple[str, str, str]]:
        """Return ``[(action, label, combo), ...]`` for registered actions."""
        return [
            (action, self.labels.get(action, action), self.bindings.get(action, ""))
            for action in self._callbacks
        ]

    # -- dispatch --------------------------------------------------------- #

    def _fire(self, action: str) -> None:
        now = time.monotonic()
        if now - self._last_fire.get(action, 0.0) < self.debounce:
            return
        self._last_fire[action] = now

        if self._on_activity:
            try:
                self._on_activity()
            except Exception as exc:
                print(f"[Hotkeys] activity callback error: {exc}")

        cb = self._callbacks.get(action)
        if cb:
            try:
                cb()
            except Exception as exc:
                print(f"[Hotkeys] error running '{action}': {exc}")

    # -- lifecycle (overridden) ------------------------------------------ #

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass


class PynputHotkeys(HotkeyManager):
    """pynput backend supporting both modifier combos and double-tap keys.

    A binding of the form ``"double:<key>"`` (e.g. ``"double:enter"``) fires
    when its key is pressed twice within ``double_tap_window`` seconds.  All
    other bindings are normal pynput hotkey combos (``"<ctrl>+<alt>+9"``).
    """

    backend_name = "pynput"

    def __init__(self, profile, debounce: float = 0.4, double_tap_window: float = 0.35):
        super().__init__(profile, debounce)
        self.double_tap_window = double_tap_window
        self._listeners = []

    def start(self) -> None:
        if self._started:
            return
        from pynput import keyboard

        # macOS: cache the keyboard layout on the main thread so pynput's
        # listener thread never calls the main-thread-only TIS APIs (SIGTRAP).
        prewarm_macos_key_layout()

        combo_map = {}
        double_specs = []  # (action, key_name)
        for action in self._callbacks:
            binding = self.bindings.get(action)
            if not binding:
                continue
            if binding.startswith("double:"):
                double_specs.append((action, binding.split(":", 1)[1].strip().lower()))
            else:
                combo_map[binding] = (lambda a=action: self._fire(a))

        if not combo_map and not double_specs:
            print("[Hotkeys] No hotkeys registered.")
            return

        self._listeners = []

        if combo_map:
            gl = keyboard.GlobalHotKeys(combo_map)
            gl.daemon = True
            gl.start()
            self._listeners.append(gl)

        for action, key_name in double_specs:
            predicate = _make_key_predicate(key_name)
            tracker = _DoubleTapTracker(self.double_tap_window)

            def on_press(key, action=action, predicate=predicate, tracker=tracker):
                try:
                    if predicate(key) and tracker.feed(time.monotonic()):
                        self._fire(action)
                except Exception as exc:
                    print(f"[Hotkeys] double-tap error: {exc}")

            lis = keyboard.Listener(on_press=on_press)
            lis.daemon = True
            lis.start()
            self._listeners.append(lis)

        # Optional raw-key logging for diagnosing "hotkey does nothing":
        # run with YILIVOICE_DEBUG_KEYS=1. If no lines appear when you type,
        # events aren't reaching us (a permission problem); if they appear but
        # the combo never fires, it's a key-matching problem.
        if os.environ.get("YILIVOICE_DEBUG_KEYS"):
            def _debug_press(key):
                try:
                    print(f"[Hotkeys][debug] key down: {key!r}")
                except Exception:
                    pass
            dbg = keyboard.Listener(on_press=_debug_press)
            dbg.daemon = True
            dbg.start()
            self._listeners.append(dbg)
            print("[Hotkeys][debug] raw key logging ON (YILIVOICE_DEBUG_KEYS).")

        self._started = True
        print(
            f"[Hotkeys] pynput listening "
            f"(combos={len(combo_map)}, double-tap={len(double_specs)}, no sudo needed)."
        )

    def stop(self) -> None:
        for lis in self._listeners:
            try:
                lis.stop()
            except Exception:
                pass
        self._listeners = []
        self._started = False


class KeyboardHotkeys(HotkeyManager):
    backend_name = "keyboard"

    def __init__(self, profile, debounce: float = 0.4):
        super().__init__(profile, debounce)
        self._keyboard = None

    def start(self) -> None:
        if self._started:
            return
        import keyboard

        self._keyboard = keyboard
        for action, cb in self._callbacks.items():
            combo = self.bindings.get(action)
            if not combo:
                continue
            keyboard.add_hotkey(combo, lambda a=action: self._fire(a))
        self._started = True
        print(f"[Hotkeys] keyboard library listening ({len(self._callbacks)} bindings).")

    def stop(self) -> None:
        if self._keyboard:
            try:
                self._keyboard.unhook_all_hotkeys()
            except Exception:
                pass
        self._started = False


class NullHotkeys(HotkeyManager):
    """Fallback when no hotkey backend can be loaded."""

    backend_name = "none"

    def start(self) -> None:
        print(
            "[Hotkeys] No global-hotkey backend available — hotkeys disabled.\n"
            "          Install 'pynput' (macOS/Linux) or 'keyboard' (Windows)."
        )

    def stop(self) -> None:
        pass


def create_hotkey_manager(profile, debounce: float = 0.4) -> HotkeyManager:
    """Pick the hotkey backend the profile asks for, with safe fallback."""
    backend = profile.hotkey_backend
    try:
        if backend == "pynput":
            import pynput  # noqa: F401  (availability probe)
            return PynputHotkeys(profile, debounce)
        if backend == "keyboard":
            import keyboard  # noqa: F401
            return KeyboardHotkeys(profile, debounce)
    except Exception as exc:
        print(f"[Hotkeys] '{backend}' backend unavailable ({exc}).")
    return NullHotkeys(profile, debounce)
