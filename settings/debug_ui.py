"""Settings / control window, built on PySide6 (Qt).

Replaces the customtkinter version (which sat on the deprecated macOS system
Tk and rendered blank). Public API is unchanged:

    DebugUI(app).toggle_debug_window()
    DebugUI(app).cleanup()
"""

import json
import os
from datetime import datetime

import pyaudio
import speech_recognition as sr
from PySide6 import QtCore, QtGui, QtWidgets

from utils.voice_converter import VoiceConverter, SOUNDDEVICE_AVAILABLE
from .platform_profile import (
    VALID_OVERRIDES,
    OS_DISPLAY_NAMES,
    PROFILE_AUTO,
)


PANEL_SETTINGS_EDIT = "#3d3418"


def _mono_font(size=11):
    f = QtGui.QFont("Menlo")
    f.setStyleHint(QtGui.QFont.Monospace)
    f.setPointSize(size)
    return f


class _DebugWindow(QtWidgets.QWidget):
    """Top-level settings window that reports its own close."""

    def __init__(self, on_close):
        super().__init__()
        self._on_close = on_close
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

    def closeEvent(self, event):
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                pass
        event.accept()


class DebugUI:
    """Settings / control panel for the yiliVoice application."""

    def __init__(self, app_instance):
        self.app = app_instance
        self.debug_window = None

        self.mic_info_text = None
        self.settings_text = None
        self.mic_selection_combo = None
        self.settings_edit_mode = False
        self.edit_button = None
        self._filler_text = None

        # Voice changer tab widgets
        self._vc_status_label = None
        self._vc_toggle_btn = None
        self._vc_input_combo = None
        self._vc_output_combo = None
        self._vc_pitch_slider = None
        self._vc_pitch_label = None
        self._vc_gain_slider = None
        self._vc_gain_label = None

        # Volume threshold slider widgets
        self._vol_thresh_slider = None
        self._vol_thresh_label = None

        # System / platform tab widgets
        self._system_info_text = None
        self._platform_combo = None
        self._vc_hint_label = None

    # ------------------------------------------------------------------ #
    # Small widget helpers                                                #
    # ------------------------------------------------------------------ #

    def _heading(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("font-size: 14px; font-weight: 700;")
        return lbl

    def _label(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("font-weight: 600;")
        return lbl

    def _hint(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: #6b7280; font-size: 11px;")
        lbl.setWordWrap(True)
        return lbl

    def _info_box(self, bg="#111827", height=None):
        box = QtWidgets.QPlainTextEdit()
        box.setReadOnly(True)
        box.setFont(_mono_font(11))
        box.setStyleSheet(f"QPlainTextEdit {{ background-color: {bg}; }}")
        if height is not None:
            box.setFixedHeight(height)
        return box

    @staticmethod
    def _set_combo(combo, names, match_index=None, default_first=False):
        """Populate a combo and select the item whose label is ``[index] …``."""
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        chosen = -1
        if match_index is not None:
            for i, n in enumerate(names):
                if n.startswith(f"[{match_index}]"):
                    chosen = i
                    break
        if chosen < 0 and default_first and names:
            chosen = 0
        if chosen >= 0:
            combo.setCurrentIndex(chosen)
        combo.blockSignals(False)

    @staticmethod
    def _parse_device_index(combo_value):
        try:
            return int(combo_value.split("]")[0][1:])
        except (ValueError, IndexError, AttributeError):
            return None

    # ------------------------------------------------------------------ #
    # Microphone discovery (pure logic, unchanged behaviour)              #
    # ------------------------------------------------------------------ #

    def get_available_microphones(self):
        """Return a filtered ``[(display_name, device_index), ...]`` list."""
        try:
            mic_list = []
            mic_names = sr.Microphone.list_microphone_names()

            input_keywords = ['microphone', 'mic', 'input', 'headset', 'webcam', 'broadcast']
            output_keywords = ['speaker', 'output', 'headphone', 'earphone']

            for i, name in enumerate(mic_names):
                name_lower = name.lower()

                if any(keyword in name_lower for keyword in output_keywords):
                    continue

                if 'sound mapper' in name_lower and any(
                    keyword in other_name.lower()
                    for other_name in mic_names
                    for keyword in input_keywords
                    if other_name != name
                ):
                    continue

                if any(keyword in name_lower for keyword in input_keywords) or 'sound mapper' in name_lower:
                    mic_list.append((f"[{i}] {name}", i))

            if not mic_list:
                for i, name in enumerate(mic_names):
                    mic_list.append((f"[{i}] {name}", i))

            return mic_list
        except Exception as e:
            print(f"Error getting available microphones: {e}")
            return [("Error getting microphones", None)]

    def on_microphone_selection_change(self, text):
        if not text or text.startswith("Error"):
            return
        try:
            device_index = int(text.split(']')[0][1:])
            print(f"User selected microphone device index: {device_index}")

            self.app.config.selected_microphone_index = device_index

            if self.app.recording_event.is_set():
                print("Switching microphone during recording...")
                self.app.toggle_recording()
                QtCore.QTimer.singleShot(500, self.app.toggle_recording)

            self.update_debug_info()
        except (ValueError, IndexError) as e:
            print(f"Error parsing microphone selection: {e}")

    def refresh_microphone_list(self):
        if not self.mic_selection_combo:
            return
        try:
            names = [m[0] for m in self.get_available_microphones()]
            current = getattr(self.app.config, 'selected_microphone_index', None)
            self._set_combo(self.mic_selection_combo, names,
                            match_index=current, default_first=True)
        except Exception as e:
            print(f"Error refreshing microphone list: {e}")

    def get_default_microphone_info(self):
        try:
            pa = pyaudio.PyAudio()
            try:
                default_input_info = pa.get_default_input_device_info()
                default_device_index = default_input_info['index']
                default_device_name = default_input_info['name']

                mic_names = sr.Microphone.list_microphone_names()

                for sr_index, sr_name in enumerate(mic_names):
                    if sr_name.strip() == default_device_name.strip():
                        return {'index': sr_index, 'name': default_device_name}

                return {
                    'index': default_device_index,
                    'name': f"{default_device_name} (Default)",
                }
            finally:
                pa.terminate()

        except Exception as pa_error:
            print(f"PyAudio default device detection failed: {pa_error}")

            try:
                temp_mic = sr.Microphone(sample_rate=16000)
                device_index = getattr(temp_mic, 'device_index', None)
                mic_names = sr.Microphone.list_microphone_names()

                if device_index is not None and isinstance(device_index, int):
                    if device_index < len(mic_names):
                        return {'index': device_index, 'name': mic_names[device_index]}

                if mic_names:
                    return {'index': 0, 'name': f"{mic_names[0]} (System Default)"}

                return {'index': 'none', 'name': 'No Microphone Detected'}
            except Exception as e:
                print(f"Error getting default microphone info: {e}")
                return {'index': 'error', 'name': f'Error: {str(e)}'}

    def get_microphone_info(self):
        mic_info = {
            'current_device': None,
            'sample_rate': 16000,
            'energy_threshold': self.app.recorder.energy_threshold if self.app.recorder else None,
            'dynamic_energy_threshold': self.app.recorder.dynamic_energy_threshold if self.app.recorder else None,
        }

        try:
            selected_index = self.app.config.selected_microphone_index

            if self.app.source and self.app.recording_event.is_set():
                device_index = getattr(self.app.source, 'device_index', None)

                if device_index is not None:
                    mic_names = sr.Microphone.list_microphone_names()
                    device_name = mic_names[device_index] if device_index < len(mic_names) else 'Unknown Device'
                    mic_info['current_device'] = {
                        'index': device_index,
                        'name': f"{device_name} (Recording)",
                    }
                else:
                    default_info = self.get_default_microphone_info()
                    mic_info['current_device'] = {
                        'index': default_info['index'],
                        'name': f"{default_info['name']} (Recording)",
                    }
            else:
                if selected_index is not None:
                    try:
                        mic_names = sr.Microphone.list_microphone_names()
                        if selected_index < len(mic_names):
                            mic_info['current_device'] = {
                                'index': selected_index,
                                'name': f"{mic_names[selected_index]} (Selected)",
                            }
                        else:
                            mic_info['current_device'] = {
                                'index': selected_index,
                                'name': f"Device {selected_index} (Invalid)",
                            }
                    except Exception as e:
                        print(f"Error getting selected microphone: {e}")
                        mic_info['current_device'] = self.get_default_microphone_info()
                else:
                    mic_info['current_device'] = self.get_default_microphone_info()

        except Exception as e:
            print(f"Error getting microphone info: {e}")
            mic_info['error'] = str(e)

        return mic_info

    # ------------------------------------------------------------------ #
    # Window lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def toggle_debug_window(self):
        if self.debug_window is not None and self.debug_window.isVisible():
            self.close_debug_window()
        else:
            self.create_debug_window()

    def _on_window_closed(self):
        self.debug_window = None

    def close_debug_window(self):
        win = self.debug_window
        self.debug_window = None
        if win is not None:
            try:
                win.close()
            except Exception as e:
                print(f"Error closing debug window: {e}")

    def create_debug_window(self):
        if self.debug_window is not None:
            self.debug_window.raise_()
            self.debug_window.activateWindow()
            return

        win = _DebugWindow(on_close=self._on_window_closed)
        self.debug_window = win
        win.setWindowTitle("yiliVoice · Debug")
        win.resize(680, 620)
        win.setMinimumSize(QtCore.QSize(620, 520))
        win.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)

        layout = QtWidgets.QVBoxLayout(win)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        # ---- header -------------------------------------------------- #
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("yiliVoice Control Panel")
        title.setStyleSheet("font-size: 17px; font-weight: 700;")
        sub = QtWidgets.QLabel("live settings · no restart needed")
        sub.setStyleSheet("color: #6b7280;")
        header.addWidget(title)
        header.addWidget(sub)
        header.addStretch(1)
        layout.addLayout(header)

        # ---- tabs ---------------------------------------------------- #
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_microphone_tab(), "Microphone")
        tabs.addTab(self._build_system_tab(), "System")
        tabs.addTab(self._build_filters_tab(), "Filters")
        tabs.addTab(self._build_voice_changer_tab(), "Voice Changer")
        tabs.addTab(self._build_settings_tab(), "Settings")
        layout.addWidget(tabs, 1)

        # ---- footer -------------------------------------------------- #
        footer = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_debug_info)
        save_btn = QtWidgets.QPushButton("Save Settings")
        save_btn.setStyleSheet("background-color: #16a34a; border: none; color: white;")
        save_btn.clicked.connect(self.save_debug_settings)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close_debug_window)
        footer.addWidget(refresh_btn)
        footer.addWidget(save_btn)
        footer.addStretch(1)
        footer.addWidget(close_btn)
        layout.addLayout(footer)

        self.update_debug_info()
        win.show()
        win.raise_()
        win.activateWindow()

    # ------------------------------------------------------------------ #
    # Microphone tab                                                      #
    # ------------------------------------------------------------------ #

    def _build_microphone_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)

        v.addWidget(self._heading("Current Microphone"))
        self.mic_info_text = self._info_box("#1f2937", height=170)
        v.addWidget(self.mic_info_text)

        v.addWidget(self._label("Select Microphone"))
        self.mic_selection_combo = QtWidgets.QComboBox()
        self.mic_selection_combo.textActivated.connect(self.on_microphone_selection_change)
        v.addWidget(self.mic_selection_combo)
        self.refresh_microphone_list()

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._label("Volume Threshold"))
        row.addStretch(1)
        self._vol_thresh_label = QtWidgets.QLabel(f"{self.app.config.volume_threshold:.4f}")
        self._vol_thresh_label.setStyleSheet("color: #93c5fd;")
        row.addWidget(self._vol_thresh_label)
        v.addLayout(row)

        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setRange(0, 500)  # 0.0000 – 0.0500 in 0.0001 steps
        s.setValue(int(round(self.app.config.volume_threshold * 10000)))
        s.valueChanged.connect(self._on_vol_thresh_change)
        self._vol_thresh_slider = s
        v.addWidget(s)

        v.addWidget(self._hint(
            "Lower = more sensitive (picks up quiet speech).  "
            "Higher = ignores background noise."
        ))
        v.addStretch(1)
        return w

    def _on_vol_thresh_change(self, value):
        val = round(value / 10000.0, 4)
        self._vol_thresh_label.setText(f"{val:.4f}")
        self.app.config.volume_threshold = val

    # ------------------------------------------------------------------ #
    # System / Platform tab                                               #
    # ------------------------------------------------------------------ #

    def _build_system_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)

        v.addWidget(self._heading("Platform & Acceleration"))
        self._system_info_text = self._info_box("#111827", height=230)
        v.addWidget(self._system_info_text)

        v.addWidget(self._label("Platform Profile"))
        self._platform_combo = QtWidgets.QComboBox()
        self._platform_combo.addItems([OS_DISPLAY_NAMES[k] for k in VALID_OVERRIDES])
        current = OS_DISPLAY_NAMES.get(
            getattr(self.app.config, "platform_override", PROFILE_AUTO), "Auto-detect"
        )
        idx = self._platform_combo.findText(current)
        if idx >= 0:
            self._platform_combo.setCurrentIndex(idx)
        self._platform_combo.textActivated.connect(self._apply_platform_override)
        self._platform_combo.setMaximumWidth(240)
        v.addWidget(self._platform_combo, alignment=QtCore.Qt.AlignLeft)

        v.addWidget(self._hint(
            "Auto-detect chooses the right setup for your OS. The accelerator "
            "(Apple GPU / NVIDIA CUDA / CPU) always follows your real hardware. "
            "Switching the engine takes effect after a restart; cable + hotkey "
            "settings update live. Click “Save Settings” to remember your choice."
        ))
        v.addStretch(1)
        self._refresh_system_tab()
        return w

    def _refresh_system_tab(self):
        if self._system_info_text is None:
            return
        p = getattr(self.app, "profile", None)
        if p is None:
            return

        backend = getattr(self.app, "backend", None)
        backend_label = backend.device_label if backend else p.backend
        model_ref = backend.model_ref if backend else "(loading…)"
        hk = getattr(self.app, "hotkeys", None)
        hk_name = hk.backend_name if hk else p.hotkey_backend

        pin_note = "auto" if p.override == PROFILE_AUTO else "pinned"
        lines = [
            f"Detected OS         {OS_DISPLAY_NAMES.get(p.detected_os, p.detected_os)}",
            f"Active Profile      {OS_DISPLAY_NAMES.get(p.effective_os, p.effective_os)}  ({pin_note})",
            f"Apple Silicon       {p.is_apple_silicon}",
            "",
            f"Accelerator         {p.accelerator_label}",
            f"Engine              {backend_label}",
            f"Model               {model_ref}",
            "",
            f"Hotkey Backend      {hk_name}",
        ]
        for action in ("toggle_recording", "toggle_voice_changer", "toggle_vc_routing"):
            label = action.replace("toggle_", "").replace("_", " ").title()
            combo = p.hotkey_labels.get(action, "—")
            lines.append(f"  {label:<18}{combo}")
        if p.permission_note:
            lines += ["", "Permissions", f"  {p.permission_note}"]

        self._system_info_text.setPlainText("\n".join(lines))

    def _apply_platform_override(self, display=None):
        display = display or self._platform_combo.currentText()
        reverse = {v: k for k, v in OS_DISPLAY_NAMES.items()}
        override = reverse.get(display, PROFILE_AUTO)

        profile = self.app.config.apply_profile(override)
        self.app.profile = profile

        if getattr(self.app, "voice_converter", None):
            self.app.voice_converter.virtual_cable_keywords = list(
                profile.virtual_cable_keywords
            )

        try:
            if getattr(self.app, "hotkeys", None):
                self.app.hotkeys.stop()
            if hasattr(self.app, "setup_hotkeys"):
                self.app.setup_hotkeys()
        except Exception as exc:
            print(f"[Debug UI] Hotkey restart failed: {exc}")

        if self._vc_hint_label is not None:
            self._vc_hint_label.setText(profile.virtual_cable_setup)

        print(f"[Debug UI] Platform profile → '{override}'.  {profile.summary()}")
        self._refresh_system_tab()
        self.update_debug_info()

    # ------------------------------------------------------------------ #
    # Voice Changer tab                                                   #
    # ------------------------------------------------------------------ #

    def _build_voice_changer_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)

        if not SOUNDDEVICE_AVAILABLE:
            msg = QtWidgets.QLabel("sounddevice not installed.\n\nRun:  pip install sounddevice")
            msg.setStyleSheet("color: #f87171; font-size: 13px;")
            msg.setAlignment(QtCore.Qt.AlignCenter)
            v.addWidget(msg, 1)
            return w

        top = QtWidgets.QHBoxLayout()
        self._vc_status_label = QtWidgets.QLabel("OFF")
        self._vc_status_label.setStyleSheet("color: #f87171; font-size: 16px; font-weight: 700;")
        top.addWidget(self._vc_status_label)
        top.addStretch(1)
        self._vc_toggle_btn = QtWidgets.QPushButton("Start Voice Changer")
        self._vc_toggle_btn.setStyleSheet("background-color: #16a34a; border: none; color: white;")
        self._vc_toggle_btn.clicked.connect(self._on_vc_toggle)
        top.addWidget(self._vc_toggle_btn)
        v.addLayout(top)

        v.addWidget(self._label("Input  (your microphone)"))
        self._vc_input_combo = QtWidgets.QComboBox()
        self._vc_input_combo.textActivated.connect(self._on_vc_input_change)
        v.addWidget(self._vc_input_combo)

        v.addWidget(self._label("Output  (virtual cable → game mic)"))
        self._vc_output_combo = QtWidgets.QComboBox()
        self._vc_output_combo.textActivated.connect(self._on_vc_output_change)
        v.addWidget(self._vc_output_combo)

        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self._label("Pitch  (semitones)"))
        prow.addStretch(1)
        self._vc_pitch_label = QtWidgets.QLabel("+7")
        self._vc_pitch_label.setStyleSheet("color: #93c5fd;")
        prow.addWidget(self._vc_pitch_label)
        v.addLayout(prow)
        ps = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        ps.setRange(-12, 12)
        ps.setValue(7)
        ps.valueChanged.connect(self._on_vc_pitch_change)
        self._vc_pitch_slider = ps
        v.addWidget(ps)

        grow = QtWidgets.QHBoxLayout()
        grow.addWidget(self._label("Gain"))
        grow.addStretch(1)
        self._vc_gain_label = QtWidgets.QLabel("1.0")
        self._vc_gain_label.setStyleSheet("color: #93c5fd;")
        grow.addWidget(self._vc_gain_label)
        v.addLayout(grow)
        gs = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        gs.setRange(0, 20)  # 0.0 – 2.0
        gs.setValue(10)
        gs.valueChanged.connect(self._on_vc_gain_change)
        self._vc_gain_slider = gs
        v.addWidget(gs)

        self._vc_hint_label = self._hint(self.app.profile.virtual_cable_setup)
        v.addWidget(self._vc_hint_label)
        v.addStretch(1)

        self._populate_vc_devices()
        return w

    def _populate_vc_devices(self):
        vc = self.app.voice_converter
        if not vc:
            return

        inputs = VoiceConverter.get_input_devices()
        outputs = VoiceConverter.get_output_devices()

        in_names = [f"[{i}] {n}" for i, n in inputs]

        speaker_kw = ("speaker", "headphone", "earphone", "realtek", "hd audio")
        cable_outputs = [
            (i, n) for i, n in outputs
            if not any(kw in n.lower() for kw in speaker_kw)
        ]
        if not cable_outputs:
            cable_outputs = outputs
        out_names = [f"[{i}] {n}" for i, n in cable_outputs]

        self._set_combo(self._vc_input_combo, in_names,
                        match_index=vc.input_device, default_first=True)

        if vc.output_device is not None:
            self._set_combo(self._vc_output_combo, out_names, match_index=vc.output_device)
        else:
            cables = VoiceConverter.find_virtual_cables()
            match = cables[0][0] if cables else None
            self._set_combo(self._vc_output_combo, out_names, match_index=match)

    def _on_vc_toggle(self):
        vc = self.app.voice_converter
        if not vc:
            return
        is_on = vc.toggle()
        self._vc_status_label.setText("ON" if is_on else "OFF")
        self._vc_status_label.setStyleSheet(
            f"color: {'#4ade80' if is_on else '#f87171'}; font-size: 16px; font-weight: 700;"
        )
        self._vc_toggle_btn.setText("Stop Voice Changer" if is_on else "Start Voice Changer")
        self._vc_toggle_btn.setStyleSheet(
            f"background-color: {'#dc2626' if is_on else '#16a34a'}; border: none; color: white;"
        )

    def _on_vc_input_change(self, text):
        vc = self.app.voice_converter
        if not vc:
            return
        idx = self._parse_device_index(text)
        if idx is not None:
            vc.set_devices(input_device=idx)

    def _on_vc_output_change(self, text):
        vc = self.app.voice_converter
        if not vc:
            return
        idx = self._parse_device_index(text)
        if idx is not None:
            vc.set_devices(output_device=idx)

    def _on_vc_pitch_change(self, value):
        vc = self.app.voice_converter
        self._vc_pitch_label.setText(f"{value:+d}")
        if vc:
            vc.set_pitch(value)

    def _on_vc_gain_change(self, value):
        vc = self.app.voice_converter
        gain = round(value / 10.0, 1)
        self._vc_gain_label.setText(f"{gain:.1f}")
        if vc:
            vc.set_gain(gain)

    # ------------------------------------------------------------------ #
    # Filters tab                                                         #
    # ------------------------------------------------------------------ #

    def _build_filters_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)

        head = QtWidgets.QHBoxLayout()
        head.addWidget(self._label("Active Filters   (edit settings/filters.json to change)"))
        head.addStretch(1)
        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(self._reload_filters_tab)
        head.addWidget(reload_btn)
        v.addLayout(head)

        v.addWidget(QtWidgets.QLabel("Filler / Stopping Words   (stripped inline from output)"))
        self._filler_text = self._info_box("#1e3a5f")
        v.addWidget(self._filler_text, 1)

        self._populate_filters_tab()
        return w

    def _populate_filters_tab(self):
        config = self.app.config
        filler = sorted(getattr(config, "filler_words", set()))
        filler_text = "  •  ".join(filler) if filler else "(none loaded)"
        self._filler_text.setPlainText(filler_text)

    def _reload_filters_tab(self):
        try:
            from settings.config import (
                _load_filters,
                _build_filler_strip_pattern,
            )

            filters = _load_filters()
            raw_filler = filters["filler_words"]

            self.app.config.filler_words = {w.lower().strip() for w in raw_filler}
            self.app.config._filler_strip_pattern = _build_filler_strip_pattern(raw_filler)

            self._populate_filters_tab()
            print("[Debug UI] Filters reloaded from filters.json")
        except Exception as exc:
            print(f"[Debug UI] Error reloading filters: {exc}")

    # ------------------------------------------------------------------ #
    # Settings tab                                                        #
    # ------------------------------------------------------------------ #

    def _build_settings_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)

        head = QtWidgets.QHBoxLayout()
        head.addWidget(self._heading("Current Configuration"))
        head.addStretch(1)
        self.edit_button = QtWidgets.QPushButton("Edit")
        self.edit_button.clicked.connect(self.toggle_settings_edit_mode)
        head.addWidget(self.edit_button)
        v.addLayout(head)

        self.settings_text = self._info_box("#111827")
        v.addWidget(self.settings_text, 1)
        self.settings_edit_mode = False
        return w

    def toggle_settings_edit_mode(self):
        if not self.settings_text:
            return
        self.settings_edit_mode = not self.settings_edit_mode
        if self.settings_edit_mode:
            self.settings_text.setReadOnly(False)
            self.settings_text.setStyleSheet(
                f"QPlainTextEdit {{ background-color: {PANEL_SETTINGS_EDIT}; }}"
            )
            self.edit_button.setText("Done")
            self.edit_button.setStyleSheet("background-color: #16a34a; border: none; color: white;")
        else:
            self.settings_text.setReadOnly(True)
            self.settings_text.setStyleSheet("QPlainTextEdit { background-color: #111827; }")
            self.edit_button.setText("Edit")
            self.edit_button.setStyleSheet("")

    # ------------------------------------------------------------------ #
    # Live updates                                                        #
    # ------------------------------------------------------------------ #

    def update_debug_info(self):
        if self.debug_window is None:
            return
        try:
            self.refresh_microphone_list()
            mic_info = self.get_microphone_info()

            is_recording = self.app.recording_event.is_set()
            device_status = "Active Recording Device" if is_recording else "Default Device (Ready)"
            current_info = (
                f"Status              {device_status}\n"
                f"Device              {mic_info['current_device']['name'] if mic_info['current_device'] else 'None'}\n"
                f"Index               {mic_info['current_device']['index'] if mic_info['current_device'] else 'None'}\n"
                f"Sample Rate         {mic_info['sample_rate']} Hz\n"
                f"Energy Threshold    {mic_info['energy_threshold']}\n"
                f"Dynamic Energy      {mic_info['dynamic_energy_threshold']}\n"
                f"Recording Status    {'Active' if is_recording else 'Inactive'}\n"
            )
            if self.mic_info_text is not None:
                self.mic_info_text.setPlainText(current_info)

            if not self.settings_edit_mode and self.settings_text is not None:
                settings_info = (
                    f"Model                  {self.app.config.model}\n"
                    f"Non-English            {self.app.config.non_english}\n"
                    f"Energy Threshold       {self.app.config.energy_threshold}\n"
                    f"Record Timeout         {self.app.config.record_timeout}s\n"
                    f"Phrase Timeout         {self.app.config.phrase_timeout}s\n"
                    f"Volume Threshold       {self.app.config.volume_threshold}\n"
                    f"No Speech Threshold    {self.app.config.no_speech_threshold}\n"
                    f"Trailing Silence       {self.app.config.trailing_silence}s\n"
                    f"Threshold Adjustment   {self.app.config.threshold_adjustment}\n"
                    f"Inactivity Timeout     {self.app.config.inactivity_timeout}s\n"
                    f"Platform               {OS_DISPLAY_NAMES.get(self.app.profile.effective_os, self.app.profile.effective_os)}\n"
                    f"Accelerator            {self.app.profile.accelerator_label}\n"
                    f"Engine                 {getattr(self.app.backend, 'name', self.app.profile.backend)}\n"
                )
                self.settings_text.setPlainText(settings_info)

            self._refresh_system_tab()

        except Exception as e:
            print(f"Error updating debug info: {e}")

    def save_debug_settings(self):
        try:
            success = self.app.config.save_to_file()

            if success:
                mic_info = self.get_microphone_info()
                settings_dir = "./settings"
                mic_file = os.path.join(settings_dir, "microphone_info.json")
                with open(mic_file, 'w') as f:
                    json.dump(mic_info, f, indent=2)

                self._append_settings_note(
                    f"\n\n[OK]  Settings saved to ./settings at {datetime.now().strftime('%H:%M:%S')}"
                )
            else:
                self._append_settings_note("\n\n[ERR]  Error saving settings")

        except Exception as e:
            print(f"Error saving settings: {e}")
            self._append_settings_note(f"\n\n[ERR]  Error saving settings: {e}")

    def _append_settings_note(self, note: str):
        if self.settings_text is None:
            return
        self.settings_text.moveCursor(QtGui.QTextCursor.End)
        self.settings_text.insertPlainText(note)
        self.settings_text.moveCursor(QtGui.QTextCursor.End)

    def cleanup(self):
        win = self.debug_window
        self.debug_window = None
        if win is not None:
            try:
                win.close()
            except Exception:
                pass
