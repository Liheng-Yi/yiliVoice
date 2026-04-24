"""Smooth animated floating status overlay.

Uses customtkinter for a modern look and a lightweight hand-rolled
animation on a tk.Canvas for a butter-smooth breathing/pulse effect.

Public API (kept backwards compatible):
    create_overlay_window(debug_callback=None) -> (root, canvas, indicator)
    update_indicator(canvas, indicator, is_recording, idle=False)
"""

import math
import tkinter as tk

import customtkinter as ctk


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class OverlayIndicator:
    """Animated status indicator rendered on a tk.Canvas.

    States:
        ready     - soft violet, very gentle breathing
        recording - vibrant green, stronger pulse
        idle      - muted red, almost still
    """

    SIZE = 44
    CHROMA_BG = "#010203"

    PALETTE = {
        "ready":     {"core": "#a78bfa", "glow": "#7c3aed"},
        "recording": {"core": "#4ade80", "glow": "#16a34a"},
        "idle":      {"core": "#f87171", "glow": "#991b1b"},
    }

    PULSE_AMPLITUDE = {
        "ready":     0.06,
        "recording": 0.22,
        "idle":      0.03,
    }

    PULSE_SPEED = {
        "ready":     0.035,
        "recording": 0.09,
        "idle":      0.02,
    }

    def __init__(self, root, debug_callback=None):
        self.root = root
        self.state = "ready"
        self.phase = 0.0
        self._running = True

        self.canvas = tk.Canvas(
            root,
            width=self.SIZE,
            height=self.SIZE,
            bg=self.CHROMA_BG,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()

        # Layered circles for a soft glow falloff (drawn back-to-front)
        self._layers = [
            self.canvas.create_oval(0, 0, 0, 0, fill="", outline="")
            for _ in range(4)
        ]

        if debug_callback:
            self.canvas.bind("<Button-1>", lambda _e: debug_callback())

        self._animate()

    def set_state(self, is_recording: bool, idle: bool = False) -> None:
        if idle:
            self.state = "idle"
        elif is_recording:
            self.state = "recording"
        else:
            self.state = "ready"

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _lerp(c1: str, c2: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _animate(self) -> None:
        if not self._running:
            return

        try:
            self.phase = (self.phase + self.PULSE_SPEED[self.state]) % (2 * math.pi)
            pulse = (math.sin(self.phase) + 1.0) / 2.0
            scale = 1.0 + self.PULSE_AMPLITUDE[self.state] * pulse

            palette = self.PALETTE[self.state]
            core = palette["core"]
            glow = palette["glow"]

            center = self.SIZE / 2
            base_radii = [16.0, 12.0, 8.0, 5.0]
            fade_steps = [0.82, 0.55, 0.2, 0.0]

            for layer_id, base_r, fade in zip(self._layers, base_radii, fade_steps):
                r = base_r * scale
                color = self._lerp(glow if fade > 0.3 else core, self.CHROMA_BG, fade)
                if fade == 0.0:
                    color = core
                self.canvas.coords(
                    layer_id,
                    center - r, center - r,
                    center + r, center + r,
                )
                self.canvas.itemconfig(layer_id, fill=color)

        except tk.TclError:
            self._running = False
            return

        self.root.after(30, self._animate)


def create_overlay_window(debug_callback=None):
    """Create the floating status overlay and return (root, canvas, indicator)."""
    root = ctk.CTk()
    root.title("Voice Status")
    root.configure(fg_color=OverlayIndicator.CHROMA_BG)
    root.attributes('-topmost', True)
    root.overrideredirect(True)

    try:
        root.attributes('-transparentcolor', OverlayIndicator.CHROMA_BG)
    except tk.TclError:
        pass

    indicator = OverlayIndicator(root, debug_callback)

    screen_width = root.winfo_screenwidth()
    size = OverlayIndicator.SIZE
    root.geometry(f'{size}x{size}+{(screen_width // 2) - size // 2}+0')

    def keep_on_top():
        try:
            root.lift()
            root.attributes('-topmost', True)
        except tk.TclError:
            return
        root.after(1500, keep_on_top)

    keep_on_top()

    return root, indicator.canvas, indicator


def update_indicator(canvas, indicator, is_recording, idle=False):
    """Update the overlay state (thread-safe via tkinter main loop)."""
    if isinstance(indicator, OverlayIndicator):
        indicator.set_state(is_recording, idle)
    else:
        color = 'red' if idle else ('green' if is_recording else '#a78bfa')
        try:
            canvas.itemconfig(indicator, fill=color)
        except tk.TclError:
            pass
