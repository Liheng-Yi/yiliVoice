"""Modern debug / control window built on customtkinter.

Keeps the public API unchanged (``DebugUI(app).toggle_debug_window()``)
so ``yiliVoice.py`` does not need to be modified.
"""

import json
import os
from datetime import datetime

import customtkinter as ctk
import pyaudio
import speech_recognition as sr
import torch

from utils.voice_converter import VoiceConverter, SOUNDDEVICE_AVAILABLE


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# Accent colours used across the window (subtle dark-mode panels)
PANEL_MIC = "#1f2937"
PANEL_FILLER = "#1e3a5f"
PANEL_SETTINGS = "#111827"
PANEL_SETTINGS_EDIT = "#3d3418"

HEADING_FONT = ("Segoe UI", 14, "bold")
LABEL_FONT = ("Segoe UI", 11, "bold")
BODY_FONT = ("Segoe UI", 11)
HINT_FONT = ("Segoe UI", 9)
MONO_FONT = ("Consolas", 11)


class DebugUI:
    """Modern debug UI for the yiliVoice application."""

    def __init__(self, app_instance):
        self.app = app_instance
        self.debug_window = None
        self.mic_info_text = None
        self.settings_text = None
        self.mic_selection_var = None
        self.mic_selection_combo = None
        self.settings_edit_mode = False
        self.edit_button = None
        self._filler_text = None

        # Voice changer tab widgets
        self._vc_status_label = None
        self._vc_toggle_btn = None
        self._vc_input_combo = None
        self._vc_output_combo = None
        self._vc_pitch_var = None
        self._vc_pitch_label = None
        self._vc_gain_var = None
        self._vc_gain_label = None

        # Volume threshold slider widgets
        self._vol_thresh_var = None
        self._vol_thresh_label = None

    # ------------------------------------------------------------------ #
    # Microphone discovery                                                #
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

    def on_microphone_selection_change(self, selection=None):
        if not self.mic_selection_combo:
            return

        selection = selection if selection is not None else self.mic_selection_combo.get()
        if not selection or selection.startswith("Error"):
            return

        try:
            device_index = int(selection.split(']')[0][1:])
            print(f"User selected microphone device index: {device_index}")

            self.app.config.selected_microphone_index = device_index

            if self.app.recording_event.is_set():
                print("Switching microphone during recording...")
                self.app.toggle_recording()
                self.app.root.after(500, self.app.toggle_recording)

            self.update_debug_info()

        except (ValueError, IndexError) as e:
            print(f"Error parsing microphone selection: {e}")

    def refresh_microphone_list(self):
        if not self.mic_selection_combo:
            return

        try:
            available_mics = self.get_available_microphones()
            mic_names = [mic[0] for mic in available_mics]

            self.mic_selection_combo.configure(values=mic_names)

            current_index = getattr(self.app.config, 'selected_microphone_index', None)
            if current_index is not None:
                for name in mic_names:
                    if name.startswith(f"[{current_index}]"):
                        self.mic_selection_var.set(name)
                        break
            elif mic_names:
                self.mic_selection_var.set(mic_names[0])

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
        if self.debug_window and self.debug_window.winfo_exists():
            self.close_debug_window()
        else:
            self.create_debug_window()

    def close_debug_window(self):
        if self.debug_window:
            try:
                self.debug_window.destroy()
            except Exception as e:
                print(f"Error closing debug window: {e}")
            finally:
                self.debug_window = None

    def create_debug_window(self):
        if self.debug_window and self.debug_window.winfo_exists():
            self.debug_window.lift()
            return

        self.debug_window = ctk.CTkToplevel(self.app.root)
        self.debug_window.title("yiliVoice · Debug")
        self.debug_window.geometry("680x620")
        self.debug_window.minsize(620, 520)
        self.debug_window.attributes('-topmost', True)
        self.debug_window.configure(fg_color="#0f1115")

        # ---- header -------------------------------------------------- #
        header = ctk.CTkFrame(self.debug_window, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 6))

        ctk.CTkLabel(
            header, text="yiliVoice Control Panel",
            font=("Segoe UI", 18, "bold"),
        ).pack(side="left")

        ctk.CTkLabel(
            header, text="live settings · no restart needed",
            font=HINT_FONT, text_color="#6b7280",
        ).pack(side="left", padx=(10, 0), pady=(6, 0))

        # ---- tabs ---------------------------------------------------- #
        tabview = ctk.CTkTabview(
            self.debug_window,
            fg_color="#151923",
            segmented_button_fg_color="#1f2937",
            segmented_button_selected_color="#6366f1",
            segmented_button_selected_hover_color="#4f46e5",
        )
        tabview.pack(fill="both", expand=True, padx=16, pady=(4, 8))

        mic_tab = tabview.add("Microphone")
        filters_tab = tabview.add("Filters")
        vc_tab = tabview.add("Voice Changer")
        settings_tab = tabview.add("Settings")

        self._build_microphone_tab(mic_tab)
        self._build_filters_tab(filters_tab)
        self._build_voice_changer_tab(vc_tab)
        self._build_settings_tab(settings_tab)

        # ---- footer -------------------------------------------------- #
        footer = ctk.CTkFrame(self.debug_window, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkButton(
            footer, text="Refresh", width=96,
            command=self.update_debug_info,
        ).pack(side="left")

        ctk.CTkButton(
            footer, text="Save Settings", width=120,
            fg_color="#16a34a", hover_color="#15803d",
            command=self.save_debug_settings,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            footer, text="Close", width=90,
            fg_color="#374151", hover_color="#4b5563",
            command=self.close_debug_window,
        ).pack(side="right")

        self.update_debug_info()

        self.debug_window.protocol("WM_DELETE_WINDOW", self.close_debug_window)

    # ------------------------------------------------------------------ #
    # Microphone tab                                                      #
    # ------------------------------------------------------------------ #

    def _build_microphone_tab(self, parent):
        ctk.CTkLabel(
            parent, text="Current Microphone",
            font=HEADING_FONT, anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 6))

        self.mic_info_text = ctk.CTkTextbox(
            parent, height=170, fg_color=PANEL_MIC,
            font=MONO_FONT, wrap="word", corner_radius=10,
        )
        self.mic_info_text.pack(fill="x", padx=12, pady=(0, 12))
        self.mic_info_text.configure(state="disabled")

        ctk.CTkLabel(
            parent, text="Select Microphone",
            font=LABEL_FONT, anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 4))

        self.mic_selection_var = ctk.StringVar()
        self.mic_selection_combo = ctk.CTkComboBox(
            parent, variable=self.mic_selection_var,
            values=[], state="readonly",
            command=self.on_microphone_selection_change,
            width=560, height=32,
            fg_color="#1f2937", button_color="#4f46e5",
            button_hover_color="#4338ca", border_color="#374151",
        )
        self.mic_selection_combo.pack(fill="x", padx=12)

        self.refresh_microphone_list()

        # --- Volume threshold slider ------------------------------- #
        vol_frame = ctk.CTkFrame(parent, fg_color="transparent")
        vol_frame.pack(fill="x", padx=12, pady=(16, 2))

        ctk.CTkLabel(vol_frame, text="Volume Threshold", font=LABEL_FONT).pack(side="left")
        self._vol_thresh_label = ctk.CTkLabel(
            vol_frame, text=f"{self.app.config.volume_threshold:.4f}",
            font=BODY_FONT, text_color="#93c5fd",
        )
        self._vol_thresh_label.pack(side="right")

        self._vol_thresh_var = ctk.DoubleVar(value=self.app.config.volume_threshold)
        ctk.CTkSlider(
            parent, from_=0.0, to=0.05,
            variable=self._vol_thresh_var,
            command=self._on_vol_thresh_change,
            progress_color="#6366f1", button_color="#a5b4fc",
            button_hover_color="#818cf8",
        ).pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(
            parent,
            text="Lower = more sensitive (picks up quiet speech).  "
                 "Higher = ignores background noise.",
            font=HINT_FONT, text_color="#6b7280", anchor="w", justify="left",
        ).pack(fill="x", padx=12, pady=(0, 12))

    # ------------------------------------------------------------------ #
    # Voice Changer tab                                                   #
    # ------------------------------------------------------------------ #

    def _build_voice_changer_tab(self, parent):
        if not SOUNDDEVICE_AVAILABLE:
            ctk.CTkLabel(
                parent,
                text="sounddevice not installed.\n\nRun:  pip install sounddevice",
                font=("Segoe UI", 13), text_color="#f87171",
                justify="center",
            ).pack(expand=True, pady=60)
            return

        # --- status + toggle ------------------------------------------ #
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=(12, 8))

        self._vc_status_label = ctk.CTkLabel(
            top, text="OFF",
            font=("Segoe UI", 16, "bold"),
            text_color="#f87171",
        )
        self._vc_status_label.pack(side="left")

        self._vc_toggle_btn = ctk.CTkButton(
            top, text="Start Voice Changer", width=170, height=34,
            fg_color="#16a34a", hover_color="#15803d",
            command=self._on_vc_toggle,
        )
        self._vc_toggle_btn.pack(side="right")

        # --- input device --------------------------------------------- #
        ctk.CTkLabel(
            parent, text="Input  (your microphone)",
            font=LABEL_FONT, anchor="w",
        ).pack(fill="x", padx=12, pady=(6, 2))

        self._vc_input_combo = ctk.CTkComboBox(
            parent, values=[], state="readonly",
            command=self._on_vc_input_change, height=32,
            fg_color="#1f2937", button_color="#4f46e5",
            button_hover_color="#4338ca", border_color="#374151",
        )
        self._vc_input_combo.pack(fill="x", padx=12)

        # --- output device -------------------------------------------- #
        ctk.CTkLabel(
            parent, text="Output  (virtual cable → game mic)",
            font=LABEL_FONT, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        self._vc_output_combo = ctk.CTkComboBox(
            parent, values=[], state="readonly",
            command=self._on_vc_output_change, height=32,
            fg_color="#1f2937", button_color="#4f46e5",
            button_hover_color="#4338ca", border_color="#374151",
        )
        self._vc_output_combo.pack(fill="x", padx=12)

        # --- pitch slider --------------------------------------------- #
        pitch_frame = ctk.CTkFrame(parent, fg_color="transparent")
        pitch_frame.pack(fill="x", padx=12, pady=(14, 0))

        ctk.CTkLabel(pitch_frame, text="Pitch  (semitones)", font=LABEL_FONT).pack(side="left")
        self._vc_pitch_label = ctk.CTkLabel(
            pitch_frame, text="+7", font=BODY_FONT, text_color="#93c5fd",
        )
        self._vc_pitch_label.pack(side="right")

        self._vc_pitch_var = ctk.DoubleVar(value=7.0)
        ctk.CTkSlider(
            parent, from_=-12, to=12,
            variable=self._vc_pitch_var,
            command=self._on_vc_pitch_change,
            progress_color="#6366f1", button_color="#a5b4fc",
            button_hover_color="#818cf8",
        ).pack(fill="x", padx=12, pady=(2, 4))

        # --- gain slider ---------------------------------------------- #
        gain_frame = ctk.CTkFrame(parent, fg_color="transparent")
        gain_frame.pack(fill="x", padx=12, pady=(6, 0))

        ctk.CTkLabel(gain_frame, text="Gain", font=LABEL_FONT).pack(side="left")
        self._vc_gain_label = ctk.CTkLabel(
            gain_frame, text="1.0", font=BODY_FONT, text_color="#93c5fd",
        )
        self._vc_gain_label.pack(side="right")

        self._vc_gain_var = ctk.DoubleVar(value=1.0)
        ctk.CTkSlider(
            parent, from_=0.0, to=2.0,
            variable=self._vc_gain_var,
            command=self._on_vc_gain_change,
            progress_color="#6366f1", button_color="#a5b4fc",
            button_hover_color="#818cf8",
        ).pack(fill="x", padx=12, pady=(2, 4))

        # --- setup hint ----------------------------------------------- #
        hint = (
            "Setup:  Install VB-CABLE  (https://vb-audio.com/Cable/)\n"
            "1. Select your real mic as Input above\n"
            "2. Select CABLE Input as Output above\n"
            "3. In your game, set microphone to CABLE Output"
        )
        ctk.CTkLabel(
            parent, text=hint, font=HINT_FONT,
            text_color="#6b7280", justify="left", anchor="w",
        ).pack(fill="x", padx=12, pady=(16, 12))

        self._populate_vc_devices()

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

        self._vc_input_combo.configure(values=in_names)
        self._vc_output_combo.configure(values=out_names)

        if vc.input_device is not None:
            for name in in_names:
                if name.startswith(f"[{vc.input_device}]"):
                    self._vc_input_combo.set(name)
                    break
        elif in_names:
            self._vc_input_combo.set(in_names[0])

        if vc.output_device is not None:
            for name in out_names:
                if name.startswith(f"[{vc.output_device}]"):
                    self._vc_output_combo.set(name)
                    break
        else:
            cables = VoiceConverter.find_virtual_cables()
            if cables:
                for name in out_names:
                    if name.startswith(f"[{cables[0][0]}]"):
                        self._vc_output_combo.set(name)
                        break

    def _parse_device_index(self, combo_value):
        try:
            return int(combo_value.split("]")[0][1:])
        except (ValueError, IndexError):
            return None

    def _on_vc_toggle(self):
        vc = self.app.voice_converter
        if not vc:
            return
        is_on = vc.toggle()
        self._vc_status_label.configure(
            text="ON" if is_on else "OFF",
            text_color="#4ade80" if is_on else "#f87171",
        )
        self._vc_toggle_btn.configure(
            text="Stop Voice Changer" if is_on else "Start Voice Changer",
            fg_color="#dc2626" if is_on else "#16a34a",
            hover_color="#b91c1c" if is_on else "#15803d",
        )

    def _on_vc_input_change(self, selection=None):
        vc = self.app.voice_converter
        if not vc:
            return
        idx = self._parse_device_index(selection or self._vc_input_combo.get())
        if idx is not None:
            vc.set_devices(input_device=idx)

    def _on_vc_output_change(self, selection=None):
        vc = self.app.voice_converter
        if not vc:
            return
        idx = self._parse_device_index(selection or self._vc_output_combo.get())
        if idx is not None:
            vc.set_devices(output_device=idx)

    def _on_vc_pitch_change(self, value=None):
        vc = self.app.voice_converter
        if not vc:
            return
        semitones = round(self._vc_pitch_var.get())
        self._vc_pitch_label.configure(text=f"{semitones:+d}")
        vc.set_pitch(semitones)

    def _on_vc_gain_change(self, value=None):
        vc = self.app.voice_converter
        if not vc:
            return
        gain = round(self._vc_gain_var.get(), 1)
        self._vc_gain_label.configure(text=f"{gain:.1f}")
        vc.set_gain(gain)

    # ------------------------------------------------------------------ #
    # Volume threshold                                                    #
    # ------------------------------------------------------------------ #

    def _on_vol_thresh_change(self, value=None):
        val = round(self._vol_thresh_var.get(), 4)
        self._vol_thresh_label.configure(text=f"{val:.4f}")
        self.app.config.volume_threshold = val

    # ------------------------------------------------------------------ #
    # Filters tab                                                         #
    # ------------------------------------------------------------------ #

    def _build_filters_tab(self, parent):
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill="x", padx=12, pady=(12, 4))

        ctk.CTkLabel(
            header_frame,
            text="Active Filters   (edit settings/filters.json to change)",
            font=LABEL_FONT,
        ).pack(side="left")

        ctk.CTkButton(
            header_frame, text="Reload", width=80, height=28,
            command=self._reload_filters_tab,
        ).pack(side="right")

        ctk.CTkLabel(
            parent,
            text="Filler / Stopping Words   (stripped inline from output)",
            font=BODY_FONT, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        self._filler_text = ctk.CTkTextbox(
            parent, fg_color=PANEL_FILLER,
            font=MONO_FONT, wrap="word", corner_radius=10,
        )
        self._filler_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._filler_text.configure(state="disabled")

        self._populate_filters_tab()

    def _populate_filters_tab(self):
        config = self.app.config

        filler = sorted(getattr(config, "filler_words", set()))
        filler_text = "  •  ".join(filler) if filler else "(none loaded)"
        self._filler_text.configure(state="normal")
        self._filler_text.delete("1.0", "end")
        self._filler_text.insert("end", filler_text)
        self._filler_text.configure(state="disabled")

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

    def _build_settings_tab(self, parent):
        settings_header = ctk.CTkFrame(parent, fg_color="transparent")
        settings_header.pack(fill="x", padx=12, pady=(12, 6))

        ctk.CTkLabel(
            settings_header, text="Current Configuration",
            font=HEADING_FONT,
        ).pack(side="left")

        self.edit_button = ctk.CTkButton(
            settings_header, text="Edit", width=84,
            command=self.toggle_settings_edit_mode,
        )
        self.edit_button.pack(side="right")

        self.settings_text = ctk.CTkTextbox(
            parent, fg_color=PANEL_SETTINGS,
            font=MONO_FONT, wrap="word", corner_radius=10,
        )
        self.settings_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.settings_text.configure(state="disabled")
        self.settings_edit_mode = False

    def toggle_settings_edit_mode(self):
        if not self.settings_text:
            return

        self.settings_edit_mode = not self.settings_edit_mode

        if self.settings_edit_mode:
            self.settings_text.configure(state="normal", fg_color=PANEL_SETTINGS_EDIT)
            self.edit_button.configure(text="Done", fg_color="#16a34a", hover_color="#15803d")
        else:
            self.settings_text.configure(state="disabled", fg_color=PANEL_SETTINGS)
            self.edit_button.configure(text="Edit", fg_color=None, hover_color=None)

    # ------------------------------------------------------------------ #
    # Live updates                                                        #
    # ------------------------------------------------------------------ #

    def update_debug_info(self):
        if not (self.debug_window and self.debug_window.winfo_exists()):
            return

        try:
            self.refresh_microphone_list()
            mic_info = self.get_microphone_info()

            self.mic_info_text.configure(state="normal")
            self.mic_info_text.delete("1.0", "end")
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
            self.mic_info_text.insert("end", current_info)
            self.mic_info_text.configure(state="disabled")

            if not self.settings_edit_mode:
                self.settings_text.configure(state="normal")
                self.settings_text.delete("1.0", "end")
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
                    f"Device                 {self.app.device}\n"
                    f"CUDA Available         {torch.cuda.is_available()}\n"
                )
                self.settings_text.insert("end", settings_info)
                self.settings_text.configure(state="disabled")

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
        if not (hasattr(self, 'settings_text') and self.settings_text):
            return
        was_disabled = str(self.settings_text.cget('state')) == 'disabled'
        if was_disabled:
            self.settings_text.configure(state='normal')
        self.settings_text.insert("end", note)
        if was_disabled:
            self.settings_text.configure(state='disabled')

    def cleanup(self):
        if self.debug_window:
            try:
                self.debug_window.destroy()
            except Exception:
                pass
            self.debug_window = None
