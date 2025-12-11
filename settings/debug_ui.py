import tkinter as tk
from tkinter import ttk, scrolledtext
import speech_recognition as sr
import torch
import json
import os
import pyaudio
from datetime import datetime


class DebugUI:
    """Debug UI manager for yiliVoice application"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        self.debug_window = None
        self.mic_info_text = None
        self.settings_text = None
        self.mic_selection_var = None
        self.mic_selection_combo = None
        self.settings_edit_mode = False
        self.edit_button = None

    def get_available_microphones(self):
        """Get list of available microphones for selection"""
        try:
            mic_list = []
            mic_names = sr.Microphone.list_microphone_names()
            
            # Filter out non-microphone devices
            input_keywords = ['microphone', 'mic', 'input', 'headset', 'webcam', 'broadcast']
            output_keywords = ['speaker', 'output', 'headphone', 'earphone']
            
            for i, name in enumerate(mic_names):
                name_lower = name.lower()
                
                # Skip obvious output devices
                if any(keyword in name_lower for keyword in output_keywords):
                    continue
                
                # Skip generic Windows sound mappers unless they're the only option
                if 'sound mapper' in name_lower and any(keyword in other_name.lower() for other_name in mic_names for keyword in input_keywords if other_name != name):
                    continue
                
                # Include devices that seem like microphones
                if any(keyword in name_lower for keyword in input_keywords) or 'sound mapper' in name_lower:
                    display_name = f"[{i}] {name}"
                    mic_list.append((display_name, i))
            
            # If no devices found after filtering, show all (safety fallback)
            if not mic_list:
                for i, name in enumerate(mic_names):
                    display_name = f"[{i}] {name}"
                    mic_list.append((display_name, i))
            
            return mic_list
        except Exception as e:
            print(f"Error getting available microphones: {e}")
            return [("Error getting microphones", None)]

    def on_microphone_selection_change(self, event=None):
        """Handle microphone selection change"""
        if not self.mic_selection_combo:
            return
            
        selection = self.mic_selection_combo.get()
        if not selection or selection.startswith("Error"):
            return
            
        try:
            # Extract device index from selection string like "[2] Microphone Name"
            device_index = int(selection.split(']')[0][1:])
            print(f"User selected microphone device index: {device_index}")
            
            # Update the app's selected device
            self.app.config.selected_microphone_index = device_index
            
            # If currently recording, restart recording with new device
            if self.app.recording_event.is_set():
                print("Switching microphone during recording...")
                self.app.toggle_recording()  # Stop current recording
                # Small delay to ensure cleanup
                self.app.root.after(500, self.app.toggle_recording)  # Restart with new device
            
            # Update debug display
            self.update_debug_info()
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing microphone selection: {e}")

    def refresh_microphone_list(self):
        """Refresh the microphone selection dropdown"""
        if not self.mic_selection_combo:
            return
            
        try:
            available_mics = self.get_available_microphones()
            mic_names = [mic[0] for mic in available_mics]
            
            self.mic_selection_combo['values'] = mic_names
            
            # Set current selection based on app config
            current_index = getattr(self.app.config, 'selected_microphone_index', None)
            if current_index is not None:
                for name in mic_names:
                    if name.startswith(f"[{current_index}]"):
                        self.mic_selection_var.set(name)
                        break
            else:
                # Set to first item if no selection made yet
                if mic_names:
                    self.mic_selection_var.set(mic_names[0])
                    
        except Exception as e:
            print(f"Error refreshing microphone list: {e}")

    def get_default_microphone_info(self):
        """Get information about the default microphone device"""
        try:
            # First try to get the system default device using PyAudio
            pa = pyaudio.PyAudio()
            try:
                default_input_info = pa.get_default_input_device_info()
                default_device_index = default_input_info['index']
                default_device_name = default_input_info['name']
                
                # Map PyAudio device index to speech_recognition device index
                mic_names = sr.Microphone.list_microphone_names()
                
                # Try to find the matching device name in speech_recognition list
                for sr_index, sr_name in enumerate(mic_names):
                    if sr_name.strip() == default_device_name.strip():
                        return {
                            'index': sr_index,
                            'name': default_device_name
                        }
                
                # If exact match not found, return the PyAudio info
                return {
                    'index': default_device_index,
                    'name': f"{default_device_name} (Default)"
                }
                
            finally:
                pa.terminate()
                
        except Exception as pa_error:
            print(f"PyAudio default device detection failed: {pa_error}")
            
            # Fallback to speech_recognition method
            try:
                temp_mic = sr.Microphone(sample_rate=16000)
                device_index = getattr(temp_mic, 'device_index', None)
                
                mic_names = sr.Microphone.list_microphone_names()
                
                if device_index is not None and isinstance(device_index, int):
                    if device_index < len(mic_names):
                        return {
                            'index': device_index,
                            'name': mic_names[device_index]
                        }
                
                # If we can't get specific device index, try to find the default
                # Usually the first device in the list is the system default
                if mic_names:
                    return {
                        'index': 0,
                        'name': f"{mic_names[0]} (System Default)"
                    }
                
                # Fallback if no devices found
                return {
                    'index': 'none',
                    'name': 'No Microphone Detected'
                }
            except Exception as e:
                print(f"Error getting default microphone info: {e}")
                return {
                    'index': 'error',
                    'name': f'Error: {str(e)}'
                }

    def get_microphone_info(self):
        """Get information about the current microphone device"""
        mic_info = {
            'current_device': None,
            'sample_rate': 16000,
            'energy_threshold': self.app.recorder.energy_threshold if self.app.recorder else None,
            'dynamic_energy_threshold': self.app.recorder.dynamic_energy_threshold if self.app.recorder else None
        }
        
        try:
            
            # Get current device info
            selected_index = self.app.config.selected_microphone_index
            
            if self.app.source and self.app.recording_event.is_set():
                # When actively recording, show the active microphone
                device_index = getattr(self.app.source, 'device_index', None)
                
                if device_index is not None:
                    # Use the specific device index from the active source
                    mic_names = sr.Microphone.list_microphone_names()
                    device_name = mic_names[device_index] if device_index < len(mic_names) else 'Unknown Device'
                    mic_info['current_device'] = {
                        'index': device_index,
                        'name': f"{device_name} (Recording)"
                    }
                else:
                    # Fallback if device_index not found
                    default_info = self.get_default_microphone_info()
                    mic_info['current_device'] = {
                        'index': default_info['index'],
                        'name': f"{default_info['name']} (Recording)"
                    }
            else:
                # When not recording, show the selected or default microphone
                if selected_index is not None:
                    # Show the user-selected microphone
                    try:
                        mic_names = sr.Microphone.list_microphone_names()
                        if selected_index < len(mic_names):
                            mic_info['current_device'] = {
                                'index': selected_index,
                                'name': f"{mic_names[selected_index]} (Selected)"
                            }
                        else:
                            mic_info['current_device'] = {
                                'index': selected_index,
                                'name': f"Device {selected_index} (Invalid)"
                            }
                    except Exception as e:
                        print(f"Error getting selected microphone: {e}")
                        mic_info['current_device'] = self.get_default_microphone_info()
                else:
                    # No selection made, show system default
                    mic_info['current_device'] = self.get_default_microphone_info()
                
        except Exception as e:
            print(f"Error getting microphone info: {e}")
            mic_info['error'] = str(e)
        
        return mic_info

    def toggle_debug_window(self):
        """Toggle the debug window visibility"""
        if self.debug_window and self.debug_window.winfo_exists():
            self.close_debug_window()
        else:
            self.create_debug_window()

    def close_debug_window(self):
        """Properly close the debug window"""
        if self.debug_window:
            try:
                self.debug_window.destroy()
            except Exception as e:
                print(f"Error closing debug window: {e}")
            finally:
                self.debug_window = None

    def create_debug_window(self):
        """Create the debug information window"""
        if self.debug_window and self.debug_window.winfo_exists():
            self.debug_window.lift()
            return
            
        self.debug_window = tk.Toplevel(self.app.root)
        self.debug_window.title("yiliVoice Debug Information")
        self.debug_window.geometry("600x500")
        self.debug_window.attributes('-topmost', True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.debug_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Microphone tab
        mic_frame = ttk.Frame(notebook)
        notebook.add(mic_frame, text="Microphone")
        
        # Current microphone info
        ttk.Label(mic_frame, text="Current Microphone:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        self.mic_info_text = scrolledtext.ScrolledText(mic_frame, height=10, width=70, bg='#d4edda')
        self.mic_info_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.mic_info_text.config(state='disabled')  # Read-only
        
        # Microphone selection
        selection_frame = ttk.Frame(mic_frame)
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(selection_frame, text="Select Microphone:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.mic_selection_var = tk.StringVar()
        self.mic_selection_combo = ttk.Combobox(selection_frame, textvariable=self.mic_selection_var, 
                                               state="readonly", width=60)
        self.mic_selection_combo.pack(fill=tk.X, pady=(5, 0))
        self.mic_selection_combo.bind('<<ComboboxSelected>>', self.on_microphone_selection_change)
        
        # Populate microphone dropdown
        self.refresh_microphone_list()
        
        # Settings tab
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # Settings header with edit button
        settings_header = ttk.Frame(settings_frame)
        settings_header.pack(fill=tk.X, pady=(10, 5))
        
        ttk.Label(settings_header, text="Current Configuration:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        self.edit_button = ttk.Button(settings_header, text="Edit", command=self.toggle_settings_edit_mode, width=8)
        self.edit_button.pack(side=tk.RIGHT)
        
        self.settings_text = scrolledtext.ScrolledText(settings_frame, height=20, width=70, bg='#f8f9fa')
        self.settings_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.settings_text.config(state='disabled')  # Read-only by default
        self.settings_edit_mode = False
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.debug_window)
        buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(buttons_frame, text="Refresh", command=self.update_debug_info).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="Save Settings", command=self.save_debug_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Close", command=self.close_debug_window).pack(side=tk.RIGHT)
        
        # Initial update
        self.update_debug_info()
        
        # Handle window close
        self.debug_window.protocol("WM_DELETE_WINDOW", self.close_debug_window)

    def toggle_settings_edit_mode(self):
        """Toggle between edit and read-only mode for settings"""
        if not self.settings_text:
            return
            
        self.settings_edit_mode = not self.settings_edit_mode
        
        if self.settings_edit_mode:
            self.settings_text.config(state='normal', bg='#fff3cd')  # Yellow tint for edit mode
            self.edit_button.config(text="Done")
        else:
            self.settings_text.config(state='disabled', bg='#f8f9fa')
            self.edit_button.config(text="Edit")

    def update_debug_info(self):
        """Update the debug information display"""
        if not (self.debug_window and self.debug_window.winfo_exists()):
            return
            
        try:
            # Refresh microphone list in case devices were added/removed
            self.refresh_microphone_list()
            # Update microphone info
            mic_info = self.get_microphone_info()
            
            # Current microphone (temporarily enable to update content)
            self.mic_info_text.config(state='normal')
            self.mic_info_text.delete(1.0, tk.END)
            is_recording = self.app.recording_event.is_set()
            device_status = "Active Recording Device" if is_recording else "Default Device (Ready)"
            current_info = f"""Status: {device_status}
Device: {mic_info['current_device']['name'] if mic_info['current_device'] else 'None'}
Index: {mic_info['current_device']['index'] if mic_info['current_device'] else 'None'}
Sample Rate: {mic_info['sample_rate']} Hz
Energy Threshold: {mic_info['energy_threshold']}
Dynamic Energy: {mic_info['dynamic_energy_threshold']}
Recording Status: {'Active' if is_recording else 'Inactive'}
"""
            self.mic_info_text.insert(tk.END, current_info)
            self.mic_info_text.config(state='disabled')  # Re-disable after update
            
            # Settings info (only update if not in edit mode)
            if not self.settings_edit_mode:
                self.settings_text.config(state='normal')
                self.settings_text.delete(1.0, tk.END)
                settings_info = f"""Model: {self.app.config.model}
Non-English: {self.app.config.non_english}
Energy Threshold: {self.app.config.energy_threshold}
Record Timeout: {self.app.config.record_timeout}s
Phrase Timeout: {self.app.config.phrase_timeout}s
Volume Threshold: {self.app.config.volume_threshold}
No Speech Threshold: {self.app.config.no_speech_threshold}
Trailing Silence: {self.app.config.trailing_silence}s
Threshold Adjustment: {self.app.config.threshold_adjustment}
Inactivity Timeout: {self.app.config.inactivity_timeout}s
Device: {self.app.device}
CUDA Available: {torch.cuda.is_available()}
"""
                self.settings_text.insert(tk.END, settings_info)
                self.settings_text.config(state='disabled')  # Re-disable after update
            
        except Exception as e:
            print(f"Error updating debug info: {e}")

    def save_debug_settings(self):
        """Save current settings to the settings directory"""
        try:
            # Save configuration using the config's save method
            success = self.app.config.save_to_file()
            
            if success:
                # Also save microphone info
                mic_info = self.get_microphone_info()
                settings_dir = "./settings"
                mic_file = os.path.join(settings_dir, "microphone_info.json")
                with open(mic_file, 'w') as f:
                    json.dump(mic_info, f, indent=2)
                
                # Show success message in debug window
                if hasattr(self, 'settings_text') and self.settings_text:
                    was_disabled = str(self.settings_text.cget('state')) == 'disabled'
                    if was_disabled:
                        self.settings_text.config(state='normal')
                    self.settings_text.insert(tk.END, f"\n\n✓ Settings saved to ./settings at {datetime.now().strftime('%H:%M:%S')}")
                    if was_disabled:
                        self.settings_text.config(state='disabled')
            else:
                if hasattr(self, 'settings_text') and self.settings_text:
                    was_disabled = str(self.settings_text.cget('state')) == 'disabled'
                    if was_disabled:
                        self.settings_text.config(state='normal')
                    self.settings_text.insert(tk.END, f"\n\n✗ Error saving settings")
                    if was_disabled:
                        self.settings_text.config(state='disabled')
                
        except Exception as e:
            print(f"Error saving settings: {e}")
            if hasattr(self, 'settings_text') and self.settings_text:
                was_disabled = str(self.settings_text.cget('state')) == 'disabled'
                if was_disabled:
                    self.settings_text.config(state='normal')
                self.settings_text.insert(tk.END, f"\n\n✗ Error saving settings: {e}")
                if was_disabled:
                    self.settings_text.config(state='disabled')

    def cleanup(self):
        """Clean up debug UI resources"""
        if self.debug_window:
            try:
                self.debug_window.destroy()
            except:
                pass
            self.debug_window = None
