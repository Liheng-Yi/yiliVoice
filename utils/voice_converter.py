import numpy as np
import threading

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class PitchShifter:
    """STFT phase vocoder for real-time pitch shifting.

    Uses overlap-add with 75% overlap (hop = fft_size / 4) for
    artifact-free reconstruction.  All hot-path operations are
    vectorized with NumPy — no Python for-loops in the audio path.
    """

    def __init__(self, pitch_factor=1.5, fft_size=2048, hop_size=512):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_bins = fft_size // 2 + 1

        self.window = np.hanning(fft_size).astype(np.float32)
        # For Hann window with 75% overlap (hop=N/4), the sum of w^2
        # across 4 overlapping frames is exactly 1.5 at every sample.
        overlap_factor = fft_size // hop_size
        w_sq_sum = sum(
            self.window[i * hop_size % fft_size] ** 2
            for i in range(overlap_factor)
        )
        self._ola_norm = 1.0 / w_sq_sum if w_sq_sum > 0 else 1.0

        self.prev_phase = np.zeros(self.n_bins)
        self.phase_acc = np.zeros(self.n_bins)

        self.expected_phase_advance = (
            2.0 * np.pi * np.arange(self.n_bins) * hop_size / fft_size
        )

        self._pitch_factor = pitch_factor
        self._update_pitch_mapping()

        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.output_accum = np.zeros(fft_size * 8, dtype=np.float32)
        self.output_ready = np.zeros(0, dtype=np.float32)
        self.accum_write = 0

    def _update_pitch_mapping(self):
        """Pre-compute frequency-bin remapping for the current pitch factor."""
        src = np.arange(self.n_bins)
        tgt = np.round(src * self._pitch_factor).astype(int)
        mask = (tgt >= 0) & (tgt < self.n_bins)
        self._valid_src = np.where(mask)[0]
        self._valid_tgt = tgt[mask]

    def set_pitch_factor(self, factor):
        self._pitch_factor = factor
        self._update_pitch_mapping()
        self.reset()

    def reset(self):
        self.prev_phase[:] = 0
        self.phase_acc[:] = 0
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.output_accum[:] = 0
        self.output_ready = np.zeros(0, dtype=np.float32)
        self.accum_write = 0

    def _process_frame(self, frame):
        """One analysis → modify → synthesis cycle."""
        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed)

        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        phase_diff = phase - self.prev_phase
        self.prev_phase = phase.copy()

        deviation = phase_diff - self.expected_phase_advance
        deviation -= 2.0 * np.pi * np.round(deviation / (2.0 * np.pi))
        true_freq = self.expected_phase_advance + deviation

        new_mag = np.zeros(self.n_bins, dtype=np.float64)
        new_freq = np.zeros(self.n_bins, dtype=np.float64)

        np.add.at(new_mag, self._valid_tgt, magnitude[self._valid_src])
        new_freq[self._valid_tgt] = true_freq[self._valid_src] * self._pitch_factor

        self.phase_acc += new_freq

        synth = new_mag * np.exp(1j * self.phase_acc)
        output = np.fft.irfft(synth).astype(np.float32)
        return output * self.window * self._ola_norm

    def process(self, block):
        """Feed an arbitrary-length audio block; returns same-length shifted output."""
        n = len(block)
        self.input_buffer = np.concatenate([
            self.input_buffer, block.astype(np.float32)
        ])

        while len(self.input_buffer) >= self.fft_size:
            frame = self.input_buffer[:self.fft_size].copy()
            out_frame = self._process_frame(frame)

            end = self.accum_write + self.fft_size
            if end > len(self.output_accum):
                self.output_accum = np.concatenate([
                    self.output_accum,
                    np.zeros(self.fft_size * 4, dtype=np.float32),
                ])

            self.output_accum[self.accum_write:end] += out_frame

            ready = self.output_accum[
                self.accum_write:self.accum_write + self.hop_size
            ].copy()
            self.output_ready = np.concatenate([self.output_ready, ready])
            self.output_accum[
                self.accum_write:self.accum_write + self.hop_size
            ] = 0
            self.accum_write += self.hop_size

            if self.accum_write > self.fft_size * 4:
                self.output_accum = self.output_accum[self.accum_write:].copy()
                self.accum_write = 0

            self.input_buffer = self.input_buffer[self.hop_size:]

        if len(self.output_ready) >= n:
            result = self.output_ready[:n].copy()
            self.output_ready = self.output_ready[n:]
            return result

        result = np.zeros(n, dtype=np.float32)
        avail = min(len(self.output_ready), n)
        if avail > 0:
            result[:avail] = self.output_ready[:avail]
            self.output_ready = self.output_ready[avail:]
        return result


class VoiceConverter:
    """Real-time voice converter for gaming.

    Audio path:  real mic  →  pitch shift  →  virtual audio cable  →  game

    The game should be configured to use the virtual cable output as
    its microphone.  On Windows, install VB-CABLE (free) or VoiceMeeter.
    """

    VIRTUAL_CABLE_KEYWORDS = ["cable", "virtual", "voicemeeter"]

    def __init__(
        self,
        input_device=None,
        output_device=None,
        pitch_semitones=7,
        sample_rate=48000,
        block_size=1024,
    ):
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not installed.  Run:  pip install sounddevice"
            )

        self.input_device = input_device
        self.output_device = output_device
        self.pitch_semitones = pitch_semitones
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.gain = 1.0

        self._pitch_factor = 2.0 ** (pitch_semitones / 12.0)
        self.shifter = PitchShifter(
            pitch_factor=self._pitch_factor,
            fft_size=2048,
            hop_size=512,
        )

        self.running = False
        self.stream = None
        self._lock = threading.Lock()
        self.route_to_speaker = False


    # ------------------------------------------------------------------ #
    # Parameter controls                                                   #
    # ------------------------------------------------------------------ #

    def set_pitch(self, semitones):
        with self._lock:
            self.pitch_semitones = semitones
            self._pitch_factor = 2.0 ** (semitones / 12.0)
            self.shifter.set_pitch_factor(self._pitch_factor)

    def set_gain(self, gain):
        with self._lock:
            self.gain = max(0.0, min(3.0, gain))

    def set_devices(self, input_device=None, output_device=None):
        was_running = self.running
        if was_running:
            self.stop()
        if input_device is not None:
            self.input_device = input_device
        if output_device is not None:
            self.output_device = output_device
        if was_running:
            self.start()

    # ------------------------------------------------------------------ #
    # Audio callback                                                       #
    # ------------------------------------------------------------------ #

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(f"[VoiceConverter] {status}")

        mono = indata[:, 0].copy()

        with self._lock:
            shifted = self.shifter.process(mono)
            shifted *= self.gain

        peak = np.max(np.abs(shifted))
        if peak > 0.95:
            shifted *= 0.95 / peak

        outdata[:, 0] = shifted

    # ------------------------------------------------------------------ #
    # Start / stop / toggle                                                #
    # ------------------------------------------------------------------ #

    def start(self):
        if self.running:
            return True

        target_output = None if getattr(self, "route_to_speaker", False) else self.output_device

        if target_output is None and not getattr(self, "route_to_speaker", False):
            print(
                "[VoiceConverter] Cannot start — no output device set.\n"
                "  Select a virtual audio cable (e.g. VB-CABLE) as the output\n"
                "  device in the Voice Changer tab so audio goes to your game,\n"
                "  not your speakers."
            )
            return False

        with self._lock:
            self.shifter.reset()

        try:
            self.stream = sd.Stream(
                device=(self.input_device, target_output),
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                latency="low",
            )
            self.stream.start()
            self.running = True
            dest_name = "SPEAKER (Default)" if target_output is None else str(target_output)
            print(
                f"[VoiceConverter] Started  pitch={self.pitch_semitones:+d} st  "
                f"in={self.input_device}  out={dest_name}"
            )
            return True
        except Exception as e:
            print(f"[VoiceConverter] Failed to start: {e}")
            self.running = False
            return False

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[VoiceConverter] Error stopping: {e}")
            self.stream = None
        print("[VoiceConverter] Stopped")

    def toggle(self):
        """Toggle on/off.  Returns True if now running."""
        if self.running:
            self.stop()
        else:
            self.start()
        return self.running

    def toggle_routing(self):
        """Toggle output between virtual cable and default speaker."""
        was_running = self.running
        if was_running:
            self.stop()
            
        with self._lock:
            self.route_to_speaker = not getattr(self, "route_to_speaker", False)
            
        if was_running:
            self.start()
            
        dest = "SPEAKER (Default)" if self.route_to_speaker else "VIRTUAL CABLE"
        print(f"[VoiceConverter] Output routed to {dest}")
        return self.route_to_speaker

    # ------------------------------------------------------------------ #
    # Device enumeration helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_input_devices():
        if not SOUNDDEVICE_AVAILABLE:
            return []
        result = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                result.append((i, dev["name"]))
        return result

    @staticmethod
    def get_output_devices():
        if not SOUNDDEVICE_AVAILABLE:
            return []
        result = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                result.append((i, dev["name"]))
        return result

    @classmethod
    def find_virtual_cables(cls):
        """Auto-detect virtual audio cable output devices."""
        cables = []
        for idx, name in cls.get_output_devices():
            if any(kw in name.lower() for kw in cls.VIRTUAL_CABLE_KEYWORDS):
                cables.append((idx, name))
        return cables
