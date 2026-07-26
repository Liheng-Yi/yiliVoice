"""Fast typed-text output.

``pyautogui.write()`` posts one key event per character with a mandatory
sleep between them, so a full sentence visibly "types out" over seconds.
This module picks the fastest backend available on the host:

  * ``pynput`` ``Controller.type()`` (macOS/Linux): posts key events with no
    per-character sleep — a whole sentence lands near-instantly.  On macOS it
    needs the same Accessibility permission pyautogui did.
  * ``keyboard.write()`` (Windows): same idea, no per-character sleeps.
  * ``pyautogui`` with ``interval=0`` as the last-resort fallback.

Construct :class:`TextTyper` on the **main thread**: on macOS the pynput
backend caches the keyboard layout via
:func:`utils.hotkeys.prewarm_macos_key_layout`, whose TIS calls must run on
the main thread (see that function's docstring).  ``type()`` itself is then
safe to call from any worker thread.
"""

from __future__ import annotations

import time

from .hotkeys import prewarm_macos_key_layout


class TextTyper:
    """Types text into the focused window using the fastest available backend."""

    def __init__(self):
        self.backend_name = "none"
        self._release_modifiers = lambda: None  # set per-backend below
        self._type = self._make_backend()

    def _make_backend(self):
        try:
            from pynput.keyboard import Controller, Key

            prewarm_macos_key_layout()  # main-thread TIS cache; no-op off macOS
            controller = Controller()
            self.backend_name = "pynput"
            self._release_modifiers = self._pynput_modifier_releaser(controller, Key)
            return controller.type
        except Exception:
            pass

        try:
            import keyboard

            self.backend_name = "keyboard"
            self._release_modifiers = lambda: self._keyboard_release_modifiers(keyboard)
            return keyboard.write
        except Exception:
            pass

        import pyautogui

        self.backend_name = "pyautogui"
        return lambda text: pyautogui.write(text, interval=0)

    @staticmethod
    def _pynput_modifier_releaser(controller, Key):
        """Return a fn that posts key-up for every modifier (both L/R sides).

        Releasing a key the controller never pressed just posts an up event —
        harmless if it wasn't down, and it clears the flag if it was.
        """
        mods = [
            Key.ctrl, Key.ctrl_l, Key.ctrl_r,
            Key.cmd, Key.cmd_l, Key.cmd_r,
            Key.alt, Key.alt_l, Key.alt_r,
            Key.shift, Key.shift_l, Key.shift_r,
        ]

        def release():
            for m in mods:
                try:
                    controller.release(m)
                except Exception:
                    pass

        return release

    @staticmethod
    def _keyboard_release_modifiers(keyboard):
        for name in ("ctrl", "alt", "shift", "windows", "cmd"):
            try:
                keyboard.release(name)
            except Exception:
                pass

    def type(self, text: str) -> None:
        if text:
            self._type(text)

    def type_isolated(self, text: str, pre_delay: float = 0.3) -> None:
        """Type *text* after releasing any modifier keys still held down.

        For hotkey-triggered macros: when the hotkey fires, the user is usually
        still holding the trigger combo (e.g. Cmd+Ctrl+1). Typing right then
        would emit each character as a modifier chord — Cmd+V, Ctrl+R, Ctrl+W —
        clobbering the focused app. We release the common modifiers and pause
        briefly so the physical keys are up, then type normally.

        Call from a worker thread: the pause would otherwise block the caller
        (a hotkey listener thread).
        """
        if not text:
            return
        try:
            self._release_modifiers()
        except Exception:
            pass
        if pre_delay:
            time.sleep(pre_delay)
        self._type(text)


def stable_prefix(text: str, holdback_words: int) -> str:
    """Return *text* with its last ``holdback_words`` words dropped.

    Used to decide how much of a streaming hypothesis is safe to type: the
    model revises the trailing few words as more audio arrives, so we only
    commit the part behind that moving edge.  Returns ``""`` when the text is
    at or below the hold-back length (nothing is stable enough yet).
    """
    text = (text or "").strip()
    if not text:
        return ""
    words = text.split(" ")
    if len(words) <= holdback_words:
        return ""
    return " ".join(words[: len(words) - holdback_words])


class IncrementalTyper:
    """Types a growing (and partially revised) transcript, append-only.

    Streaming ASR hands us an ever-changing hypothesis.  We commit it *by word
    index*: once the first N words have been typed, we only ever append words
    N+1 onward and never re-touch the earlier ones.  We never backspace,
    because we can't safely delete characters from whatever app is focused.

    The trade-off: if the model revises a word we already committed (rare —
    e.g. a tense change on the final pass), that word simply stays as first
    typed.  Diffing the raw strings instead would let a single early revision
    re-append the whole tail and garble the output, so index-commit is the
    safe choice — a rare benign one-word imperfection versus duplication.

    * ``update(hyp)`` commits the stabilized prefix (hold-back applied) — the
      steady stream of words that appears while you talk.
    * ``flush_tail(hyp)`` commits every word (no hold-back); call it on a
      pause so the trailing words appear.
    * ``finalize(hyp)`` flushes, adds a separating space, and resets for the
      next recording session.

    Because it types Parakeet's text verbatim, capitalization and punctuation
    are preserved (unlike the batch path's ``clean_sentence``).
    """

    # Commit words only once they are this many back from the moving tail.
    # 5 was the smallest margin that eliminated deep tail revisions across the
    # test clips (parakeet-tdt-0.6b-v2, 256/256 context); smaller garbles, and
    # larger just delays the on-screen text without improving accuracy.
    def __init__(self, typer, holdback_words: int = 5):
        self.typer = typer
        self.holdback = holdback_words
        self.committed = 0       # words already typed this session
        self._started = False    # any text typed yet (controls the join space)

    def reset(self) -> None:
        self.committed = 0
        self._started = False

    def update(self, hypothesis: str) -> None:
        self._commit_upto(hypothesis, self.holdback)

    def flush_tail(self, hypothesis: str) -> None:
        self._commit_upto(hypothesis, 0)

    def finalize(self, hypothesis: str) -> None:
        self._commit_upto(hypothesis, 0)
        if self._started:
            self.typer.type(" ")
        self.reset()

    def _commit_upto(self, hypothesis: str, holdback: int) -> None:
        text = (hypothesis or "").strip()
        words = text.split(" ") if text else []
        stable = len(words) - holdback
        if stable <= self.committed:
            return
        new_words = words[self.committed:stable]
        if not new_words:
            return
        chunk = " ".join(new_words)
        if self._started:
            chunk = " " + chunk
        self.typer.type(chunk)
        self.committed = stable
        self._started = True
