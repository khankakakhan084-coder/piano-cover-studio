"""
piano_engine.py
Unified Acoustic Piano Cover Engine & Workflow:
1. Cloud GPU Pipeline: Direct connection to StrikeKhan/ai-piano-cover (ZeroGPU A100)
   - ByteDance Neural Piano Transcription
   - Continuous Legato Phrasing & Dissonance Filtering
   - Giga Grand SF2 Concert Acoustics
   - Multi-Token Auto-Failover
2. Local Offline Fallback Pipeline:
   - pYIN & Pitch-to-MIDI Transcription
   - Scale-Locked Left-Hand Accompaniment & Chord Progressions
   - Human Pianist Performance Simulation (Velocity + CC64 Sustain Pedal)
   - Pure-Python Sampled Grand Piano Synthesizer (SoundFont SF2)
   - Type-1 Standard Multi-Track MIDI Score Builder
"""
import os
import sys
import time
import json
import math
import shutil
import struct
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable

import numpy as np
from token_manager import token_manager
from audio_utils import (
    get_ffmpeg,
    get_audio_duration,
    trim_audio,
    convert_audio_format,
    detect_bpm_and_key,
    mix_piano_with_backing,
    save_numpy_to_wav,
    save_numpy_to_mp3,
)

logger = logging.getLogger("piano_engine")
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
SOUNDFONT_DIR = STORAGE_DIR / "soundfonts"
SOUNDFONT_DIR.mkdir(parents=True, exist_ok=True)

PIANO_SPACE_ID = "StrikeKhan/ai-piano-cover"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PURE-PYTHON TYPE-1 STANDARD MIDI BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _write_vlq(value: int) -> bytes:
    """Encode integer as Variable-Length Quantity for Standard MIDI."""
    value = max(0, int(value))
    buf = bytearray()
    buf.append(value & 0x7F)
    value >>= 7
    while value > 0:
        buf.append(0x80 | (value & 0x7F))
        value >>= 7
    buf.reverse()
    return bytes(buf)


def create_midi_file(
    right_hand_notes: List[Tuple[float, float, int, int]],  # (start_s, end_s, pitch, velocity)
    left_hand_notes: List[Tuple[float, float, int, int]],   # (start_s, end_s, pitch, velocity)
    bpm: float = 120.0,
    pedal_events: Optional[List[Tuple[float, bool]]] = None # (time_s, is_pedal_down)
) -> bytes:
    """
    Constructs a Standard Type-1 MIDI file:
    Track 0: Conductor (Tempo & Time Signature)
    Track 1: Right Hand (Lead Singing Melody)
    Track 2: Left Hand (Chords, Arpeggios, Bass & CC64 Sustain Pedal)
    """
    division = 480  # ticks per quarter note
    sec_per_beat = 60.0 / max(30.0, min(240.0, bpm))
    ticks_per_sec = division / sec_per_beat

    def time_to_ticks(t_s: float) -> int:
        return max(0, int(round(t_s * ticks_per_sec)))

    # Track 0: Conductor
    t0_events = bytearray()
    trk_name = b"Piano Cover Studio Master"
    t0_events += _write_vlq(0) + b"\xFF\x03" + _write_vlq(len(trk_name)) + trk_name
    us_per_beat = int(round(1_000_000 * sec_per_beat))
    tempo_bytes = us_per_beat.to_bytes(3, byteorder="big")
    t0_events += _write_vlq(0) + b"\xFF\x51\x03" + tempo_bytes
    t0_events += _write_vlq(0) + b"\xFF\x2F\x00"
    t0_chunk = b"MTrk" + struct.pack(">I", len(t0_events)) + t0_events

    def build_note_track(name: str, notes: List[Tuple[float, float, int, int]], channel: int = 0, is_left: bool = False) -> bytes:
        events = []
        for (st, et, pitch, vel) in notes:
            p = max(21, min(108, int(pitch)))
            v = max(1, min(127, int(vel)))
            events.append((time_to_ticks(st), 0, 0x90 | channel, p, v))
            events.append((time_to_ticks(max(st + 0.05, et)), 1, 0x80 | channel, p, 0))

        if is_left and pedal_events:
            for (pt, p_down) in pedal_events:
                val = 127 if p_down else 0
                events.append((time_to_ticks(pt), -1, 0xB0 | channel, 64, val))

        events.sort(key=lambda x: (x[0], x[1]))

        trk_bytes = bytearray()
        nb = name.encode("utf-8")
        trk_bytes += _write_vlq(0) + b"\xFF\x03" + _write_vlq(len(nb)) + nb
        trk_bytes += _write_vlq(0) + bytes([0xC0 | channel, 0x00])  # Acoustic Grand Piano (GM 0)

        last_tick = 0
        for ev in events:
            tick = ev[0]
            delta = max(0, tick - last_tick)
            last_tick = tick
            status = ev[2]
            d1 = ev[3]
            d2 = ev[4]
            trk_bytes += _write_vlq(delta) + bytes([status, d1, d2])

        trk_bytes += _write_vlq(0) + b"\xFF\x2F\x00"
        return b"MTrk" + struct.pack(">I", len(trk_bytes)) + trk_bytes

    t1_chunk = build_note_track("Right Hand - Melody", right_hand_notes, channel=0, is_left=False)
    t2_chunk = build_note_track("Left Hand - Accompaniment", left_hand_notes, channel=0, is_left=True)

    header = b"MThd" + struct.pack(">IHHH", 6, 1, 3, division)
    return header + t0_chunk + t1_chunk + t2_chunk


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SAMPLED GRAND PIANO SOUND ENGINE (SOUNDFONT SF2 SAMPLE PLAYBACK)
# ══════════════════════════════════════════════════════════════════════════════

def get_default_soundfont() -> Optional[Path]:
    """Finds or confirms cached acoustic Grand Piano SoundFont."""
    for local_sf in SOUNDFONT_DIR.glob("*.sf2"):
        if local_sf.stat().st_size > 1_000_000:
            return local_sf
    return None


class SampledGrandPianoEngine:
    """
    Pure-Python Sampled Grand Piano Synthesizer.
    Parses SoundFont 2 (SF2) RIFF chunks, extracts genuine recorded 16-bit acoustic
    piano PCM audio waves, and renders multi-velocity MIDI performances.
    Zero C-library / FluidSynth dependencies — runs universally on Windows.
    """

    def __init__(self, sf2_path: Optional[Path] = None):
        self.sf2_path = sf2_path or get_default_soundfont()
        self.samples: Dict[int, Tuple[np.ndarray, int]] = {}
        self._load_samples()

    def _load_samples(self):
        if not self.sf2_path or not self.sf2_path.exists():
            logger.warning("No SoundFont file available; physical waveguide fallback will be used.")
            return

        try:
            with open(self.sf2_path, "rb") as f:
                riff_head = f.read(12)
                if riff_head[:4] != b"RIFF" or riff_head[8:12] != b"sfbk":
                    logger.warning("Invalid SF2 header format.")
                    return

                smpl_data = None
                shdr_data = None

                while True:
                    ch_head = f.read(8)
                    if len(ch_head) < 8:
                        break
                    ch_id, ch_len = struct.unpack("<4sI", ch_head)

                    if ch_id == b"LIST":
                        list_type = f.read(4)
                        rem_len = ch_len - 4
                        if list_type == b"sdta":
                            sdta_end = f.tell() + rem_len
                            while f.tell() < sdta_end:
                                sub_head = f.read(8)
                                if len(sub_head) < 8:
                                    break
                                sub_id, sub_len = struct.unpack("<4sI", sub_head)
                                if sub_id == b"smpl":
                                    raw_pcm = f.read(sub_len)
                                    smpl_data = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                                else:
                                    f.seek(sub_len, 1)
                        elif list_type == b"pdta":
                            pdta_end = f.tell() + rem_len
                            while f.tell() < pdta_end:
                                sub_head = f.read(8)
                                if len(sub_head) < 8:
                                    break
                                sub_id, sub_len = struct.unpack("<4sI", sub_head)
                                if sub_id == b"shdr":
                                    shdr_data = f.read(sub_len)
                                else:
                                    f.seek(sub_len, 1)
                        else:
                            f.seek(rem_len, 1)
                    else:
                        f.seek(ch_len, 1)

            if smpl_data is None or shdr_data is None:
                return

            records = len(shdr_data) // 46
            for i in range(records):
                rec = shdr_data[i * 46 : (i + 1) * 46]
                name = rec[:20].decode("ascii", errors="ignore").rstrip("\x00")
                start, end, start_loop, end_loop, s_rate, orig_key, corr, link, stype = struct.unpack(
                    "<IIIIIBbhH", rec[20:46]
                )
                if start < end and orig_key < 128 and end <= len(smpl_data):
                    is_piano = any(k in name.lower() for k in ["piano", "pno", "grand", "key", "acp"]) or (i < 24)
                    if is_piano or orig_key not in self.samples:
                        wave = smpl_data[start:end]
                        if len(wave) > 100:
                            self.samples[orig_key] = (wave, s_rate)

            logger.info(f"Loaded {len(self.samples)} acoustic piano sample zones from SoundFont.")
        except Exception as e:
            logger.warning(f"Error loading SoundFont samples: {e}")

    def render(
        self,
        notes: List[Tuple[float, float, int, int]],  # (start_s, end_s, pitch, velocity)
        sample_rate: int = 44100,
        total_length_s: Optional[float] = None,
        pedal_events: Optional[List[Tuple[float, bool]]] = None
    ) -> np.ndarray:
        """
        Renders MIDI notes into a stereo audio stream using real acoustic piano samples.
        """
        if not notes:
            return np.zeros((2, int((total_length_s or 5.0) * sample_rate)), dtype=np.float32)

        if not self.samples:
            return self._synthesize_waveguide(notes, sample_rate, total_length_s)

        max_end = max(n[1] + 2.5 for n in notes)
        if total_length_s is not None:
            max_end = max(max_end, total_length_s)
        total_samples = int(math.ceil(max_end * sample_rate))

        audio_left = np.zeros(total_samples, dtype=np.float32)
        audio_right = np.zeros(total_samples, dtype=np.float32)

        avail_keys = sorted(self.samples.keys())

        for (start_t, end_t, pitch, vel) in notes:
            pitch = max(21, min(108, int(pitch)))
            vel_norm = max(0.1, min(1.0, float(vel) / 127.0))
            gain = (vel_norm ** 1.35) * 0.85

            best_key = min(avail_keys, key=lambda k: abs(k - pitch))
            base_wave, base_sr = self.samples[best_key]

            # Pitch shift ratio
            pitch_ratio = (2.0 ** ((pitch - best_key) / 12.0)) * (base_sr / sample_rate)

            note_dur = max(0.15, end_t - start_t)
            sustain_tail = 1.2
            total_note_dur = note_dur + sustain_tail
            out_note_len = int(round(total_note_dur * sample_rate))

            indices = np.arange(out_note_len) * pitch_ratio
            valid_mask = indices < (len(base_wave) - 1)
            if not np.any(valid_mask):
                continue

            valid_indices = indices[valid_mask]
            idx_int = valid_indices.astype(int)
            idx_frac = valid_indices - idx_int

            rendered_note = np.zeros(out_note_len, dtype=np.float32)
            rendered_note[valid_mask] = (
                (1.0 - idx_frac) * base_wave[idx_int] + idx_frac * base_wave[idx_int + 1]
            )

            # Natural acoustic envelope (smooth attack, sustained ringing, exponential damping)
            attack_len = min(int(0.005 * sample_rate), out_note_len)
            envelope = np.ones(out_note_len, dtype=np.float32)
            if attack_len > 0:
                envelope[:attack_len] = np.linspace(0.0, 1.0, attack_len)

            decay_start = int(note_dur * sample_rate)
            if decay_start < out_note_len:
                decay_len = out_note_len - decay_start
                envelope[decay_start:] = np.exp(-np.linspace(0.0, 3.2, decay_len))

            rendered_note *= envelope * gain

            # Keyboard stereo panning (low notes left, high notes right)
            pan = (pitch - 21) / 87.0  # 0.0 to 1.0
            pan_left = math.cos(pan * math.pi / 2.0)
            pan_right = math.sin(pan * math.pi / 2.0)

            start_idx = int(round(start_t * sample_rate))
            end_idx = min(start_idx + out_note_len, total_samples)
            slice_len = end_idx - start_idx

            if slice_len > 0:
                audio_left[start_idx:end_idx] += rendered_note[:slice_len] * pan_left
                audio_right[start_idx:end_idx] += rendered_note[:slice_len] * pan_right

        # Peak normalization with headroom
        stereo = np.vstack([audio_left, audio_right])
        peak = np.max(np.abs(stereo))
        if peak > 0.001:
            stereo = stereo / peak * 0.90

        return stereo

    def _synthesize_waveguide(
        self,
        notes: List[Tuple[float, float, int, int]],
        sample_rate: int = 44100,
        total_length_s: Optional[float] = None
    ) -> np.ndarray:
        """Karplus-Strong physical modeling fallback when SoundFont is not present."""
        max_end = max(n[1] + 2.0 for n in notes)
        if total_length_s:
            max_end = max(max_end, total_length_s)
        total_samples = int(math.ceil(max_end * sample_rate))

        out_left = np.zeros(total_samples, dtype=np.float32)
        out_right = np.zeros(total_samples, dtype=np.float32)

        for (st, et, pitch, vel) in notes:
            freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
            period = int(round(sample_rate / freq))
            if period < 2:
                continue

            dur = max(0.2, et - st + 1.2)
            n_samples = int(dur * sample_rate)

            # Excitation buffer
            buf = np.random.uniform(-1.0, 1.0, period).astype(np.float32)
            wave = np.zeros(n_samples, dtype=np.float32)
            decay = 0.994

            for i in range(n_samples):
                val = buf[i % period]
                wave[i] = val
                buf[i % period] = 0.5 * (val + buf[(i + 1) % period]) * decay

            gain = ((float(vel) / 127.0) ** 1.3) * 0.6
            pan = (pitch - 21) / 87.0
            pan_l = math.cos(pan * math.pi / 2.0)
            pan_r = math.sin(pan * math.pi / 2.0)

            s_idx = int(round(st * sample_rate))
            e_idx = min(s_idx + n_samples, total_samples)
            l = e_idx - s_idx
            if l > 0:
                out_left[s_idx:e_idx] += wave[:l] * gain * pan_l
                out_right[s_idx:e_idx] += wave[:l] * gain * pan_r

        stereo = np.vstack([out_left, out_right])
        peak = np.max(np.abs(stereo))
        if peak > 0.001:
            stereo = stereo / peak * 0.90
        return stereo


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: HUMAN PIANIST PERFORMANCE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class HumanPianoPerformance:
    """Simulates concert pianist rubato, expressive velocity, and CC64 sustain pedal."""

    def __init__(self, bpm: float = 120.0, seed: int = 42):
        self.bpm = bpm
        self.rng = np.random.RandomState(seed)

    def humanize(
        self,
        right_notes: List[Tuple[float, float, int, int]],
        left_notes: List[Tuple[float, float, int, int]],
        measures: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[float, float, int, int]], List[Tuple[float, float, int, int]], List[Tuple[float, bool]]]:
        h_right = []
        for (st, et, p, v) in right_notes:
            timing_jitter = self.rng.normal(0.0, 0.006)
            vel_jitter = int(self.rng.normal(0.0, 3.5))
            st_h = max(0.0, st + timing_jitter)
            et_h = max(st_h + 0.06, et + timing_jitter + 0.03)  # Legato phrasing overlap
            v_h = max(35, min(120, v + vel_jitter))
            h_right.append((st_h, et_h, p, v_h))

        h_left = []
        for (st, et, p, v) in left_notes:
            timing_jitter = self.rng.normal(0.005, 0.008)  # Left hand slight rolling delay
            vel_jitter = int(self.rng.normal(-4.0, 3.0))
            st_h = max(0.0, st + timing_jitter)
            et_h = max(st_h + 0.08, et + timing_jitter + 0.05)
            v_h = max(25, min(100, v + vel_jitter))
            h_left.append((st_h, et_h, p, v_h))

        # Sustain pedal timeline (down at measure start, brief lift at chord change)
        pedal_events = []
        for m in measures:
            m_start = m.get("start", 0.0)
            m_end = m.get("end", m_start + 2.0)
            pedal_events.append((m_start + 0.04, True))
            pedal_events.append((max(m_start + 0.1, m_end - 0.08), False))

        return h_right, h_left, pedal_events


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LOCAL PIANO COVER ARRANGEMENT WORKFLOW (FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

def run_local_piano_workflow(
    input_audio_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Executes the full local piano cover workflow:
    1. Audio analysis (BPM & Key)
    2. pYIN lead vocal melody transcription
    3. Harmonic chord progression and scale-locked left-hand accompaniment
    4. Pianist humanization (Legato + CC64 pedal)
    5. Standard Type-1 MIDI creation
    6. Sampled Grand Piano SoundFont rendering
    7. Master Solo Piano MP3 & Backing Mix MP3 exports
    """
    def _prog(p: int, m: str):
        if progress_callback:
            progress_callback(p, m)

    _prog(15, "Analyzing song tempo, key, and harmony...")
    theory = detect_bpm_and_key(input_audio_path)
    bpm = theory["bpm"]
    detected_key = theory["key"]

    _prog(25, "Extracting lead vocal melody with high-precision pitch tracking...")
    import librosa
    y, sr = librosa.load(str(input_audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Melody extraction via pYIN
    frame_length = 2048
    hop_length = 512
    fmin = librosa.note_to_hz("C3")
    fmax = librosa.note_to_hz("C6")

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Quantize pitch to discrete piano notes
    right_notes_raw = []
    current_pitch = None
    note_start = 0.0

    for t, pitch_hz, is_voiced in zip(times, f0, voiced_flag):
        if is_voiced and not np.isnan(pitch_hz):
            midi_pitch = int(round(librosa.hz_to_midi(pitch_hz)))
            if current_pitch is None:
                current_pitch = midi_pitch
                note_start = t
            elif abs(midi_pitch - current_pitch) > 0.6:
                if t - note_start > 0.08:
                    right_notes_raw.append((note_start, t, current_pitch, 88))
                current_pitch = midi_pitch
                note_start = t
        else:
            if current_pitch is not None:
                if t - note_start > 0.08:
                    right_notes_raw.append((note_start, t, current_pitch, 88))
                current_pitch = None

    _prog(45, "Generating flowing left-hand piano accompaniment & chords...")
    sec_per_beat = 60.0 / bpm
    sec_per_measure = sec_per_beat * 4.0
    num_measures = int(math.ceil(duration / sec_per_measure))

    # Base chord progressions based on key
    root_note = detected_key.split()[0]
    is_minor = "minor" in detected_key.lower()
    root_midi = librosa.note_to_midi(f"{root_note}3")

    # Common emotional ballad chords (I - V - vi - IV or i - VI - III - VII)
    if is_minor:
        chord_offsets = [0, 8, 3, 10]  # i - VI - III - VII
    else:
        chord_offsets = [0, 7, 9, 5]   # I - V - vi - IV

    left_notes_raw = []
    measures_meta = []

    for m_idx in range(num_measures):
        m_start = m_idx * sec_per_measure
        m_end = min(duration, m_start + sec_per_measure)
        chord_root = root_midi + chord_offsets[m_idx % len(chord_offsets)] - 12

        # 4-note flowing arpeggio pattern per measure
        for b in range(4):
            b_start = m_start + (b * sec_per_beat)
            b_end = b_start + (sec_per_beat * 1.8)
            if b_start >= duration:
                break
            if b == 0:
                p = chord_root  # Root bass
                v = 82
            elif b == 1:
                p = chord_root + 7  # Fifth
                v = 70
            elif b == 2:
                p = chord_root + (3 if is_minor else 4) + 12  # Octave third
                v = 65
            else:
                p = chord_root + 7 + 12  # Octave fifth
                v = 68
            left_notes_raw.append((b_start, b_end, p, v))

        measures_meta.append({"measure": m_idx + 1, "start": m_start, "end": m_end})

    # Humanize performance
    _prog(65, "Applying concert pianist humanization & legato phrasing...")
    humanizer = HumanPianoPerformance(bpm=bpm)
    h_right, h_left, pedal_events = humanizer.humanize(right_notes_raw, left_notes_raw, measures_meta)
    all_notes = sorted(h_right + h_left, key=lambda x: x[0])

    _prog(75, "Generating Standard Type-1 Multi-Track MIDI score...")
    midi_bytes = create_midi_file(h_right, h_left, bpm=bpm, pedal_events=pedal_events)
    midi_path = output_dir / "piano_cover.mid"
    midi_path.write_bytes(midi_bytes)

    _prog(85, "Rendering Master Acoustic Grand Piano audio via SoundFont SF2...")
    engine = SampledGrandPianoEngine()
    rendered_audio = engine.render(
        all_notes, sample_rate=44100, total_length_s=duration, pedal_events=pedal_events
    )

    # Save Solo Piano WAV & MP3
    _prog(92, "Mastering Hi-Fi Solo Piano audio (320kbps MP3)...")
    solo_wav = output_dir / "piano_cover.wav"
    save_numpy_to_wav(rendered_audio, 44100, solo_wav)

    solo_mp3 = output_dir / "piano_cover.mp3"
    save_numpy_to_mp3(rendered_audio, 44100, solo_mp3, bitrate=320)

    # Backing Mix
    piano_mix_mp3 = output_dir / "piano_mix.mp3"
    mix_piano_with_backing(solo_mp3, input_audio_path, piano_mix_mp3)

    # Chord sheet
    chord_sheet_path = output_dir / "chord_sheet.json"
    chord_info = {
        "bpm": bpm,
        "key": detected_key,
        "note_count": len(all_notes),
        "measures_count": len(measures_meta),
        "engine": "Piano Cover Studio v2.0 (High-Fidelity Sampled Grand Engine)",
    }
    chord_sheet_path.write_text(json.dumps(chord_info, indent=2), encoding="utf-8")

    _prog(100, "Local Piano Cover workflow completed!")

    return {
        "solo_piano_mp3": str(solo_mp3),
        "solo_piano_wav": str(solo_wav),
        "piano_mix_mp3": str(piano_mix_mp3),
        "piano_midi": str(midi_path),
        "chord_sheet": str(chord_sheet_path),
        "bpm": bpm,
        "key": detected_key,
        "note_count": len(all_notes),
        "engine_used": "Local High-Fidelity Grand Piano Studio Engine",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CLOUD GPU PIANO COVER WORKFLOW (ZEROGPU A100)
# ══════════════════════════════════════════════════════════════════════════════

class CloudPianoProcessor:
    """Manages full song Acoustic Piano Cover generation on NVIDIA ZeroGPU with automatic failover."""

    def __init__(self, space_id: str = PIANO_SPACE_ID):
        self.space_id = space_id
        self._client = None
        self._active_token = None
        self._client_created_at = 0.0

    def get_client(self, token: str):
        if self._client is not None and (time.time() - self._client_created_at > 90):
            self.reset_client()

        if self._client is None or self._active_token != token:
            from gradio_client import Client
            logger.info(f"Connecting to Mastered Piano Space: https://huggingface.co/spaces/{self.space_id}...")
            try:
                self._client = Client(self.space_id, token=token, httpx_kwargs={"timeout": 180})
            except TypeError:
                self._client = Client(self.space_id, token=token)
            self._active_token = token
            self._client_created_at = time.time()

        return self._client

    def reset_client(self):
        if self._client is not None:
            try:
                if hasattr(self._client, "session") and hasattr(self._client.session, "close"):
                    self._client.session.close()
            except Exception:
                pass
        self._client = None
        self._active_token = None
        self._client_created_at = 0.0

    def run_cloud_pipeline(
        self,
        input_audio_path: Path,
        output_dir: Path,
        stems_to_use: Optional[List[str]] = None,
        keep_drums: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        from gradio_client import handle_file

        if stems_to_use is None:
            stems_to_use = ["vocals", "other"]

        def _prog(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        _prog(12, "Connecting to Cloud NVIDIA ZeroGPU Studio...")

        tokens_to_try = token_manager.get_available_tokens()
        result_tuple = None
        last_error = None

        for idx, token in enumerate(tokens_to_try):
            try:
                logger.info(f"Dispatched to HF ZeroGPU (Token #{idx+1} ...{token[-6:]})")
                client = self.get_client(token)

                _prog(18, "Uploading song to GPU pipeline...")
                job = client.submit(
                    handle_file(str(input_audio_path)),
                    stems_to_use,
                    keep_drums,
                    api_name="/run_piano_pipeline"
                )

                _prog(22, "Initializing ZeroGPU container and AI models...")

                last_pct = 22
                while not job.done():
                    try:
                        st = job.status()
                        if st.progress_data:
                            unit = st.progress_data[-1]
                            desc = unit.desc or ""
                            val = float(unit.progress or 0.0)

                            mapped_pct = int(25 + (val * 65))
                            mapped_pct = max(last_pct, min(92, mapped_pct))
                            last_pct = mapped_pct

                            lbl = "Harmonizing Piano Performance..."
                            if "separat" in desc.lower():
                                lbl = "Separating Stems on GPU..."
                            elif "transcrib" in desc.lower():
                                lbl = f"Transcribing to Piano: {desc.replace('🎹', '').strip()}"
                            elif "render" in desc.lower():
                                lbl = "Rendering Acoustic Concert Grand SF2..."
                            elif "harmoniz" in desc.lower() or "smooth" in desc.lower():
                                lbl = "Applying Seamless Legato & Dissonance Filtering..."

                            _prog(mapped_pct, lbl)
                        else:
                            st_code = str(st.code)
                            if "STARTING" in st_code:
                                _prog(22, "Allocating ZeroGPU A100 & warming up Demucs AI...")
                            elif "PROCESSING" in st_code:
                                _prog(28, "Processing audio stems on NVIDIA GPU...")
                    except Exception:
                        pass
                    time.sleep(1.0)

                result_tuple = job.result(timeout=400)
                if result_tuple and (result_tuple[0] is None or not result_tuple[0]):
                    err_msg = str(result_tuple[2]) if len(result_tuple) > 2 else "ZeroGPU quota exceeded"
                    raise RuntimeError(f"Space returned: {err_msg}")

                token_manager.mark_used(token)
                break

            except Exception as ex:
                last_error = ex
                err_text = str(ex).lower()
                self.reset_client()
                if any(q in err_text for q in ["exceeded", "quota", "rate limit", "429"]):
                    token_manager.mark_exhausted(token, str(ex))
                time.sleep(1.0)

        if result_tuple is None:
            raise RuntimeError(f"Cloud GPU processing unavailable: {last_error}")

        _prog(94, "Finalizing Master Audio & MIDI Score...")

        final_wav = result_tuple[0] if len(result_tuple) > 0 else None
        final_midi = result_tuple[1] if len(result_tuple) > 1 else None
        stats_md = result_tuple[2] if len(result_tuple) > 2 else ""
        vocals_wav = result_tuple[3] if len(result_tuple) > 3 else None
        other_wav = result_tuple[4] if len(result_tuple) > 4 else None
        bass_wav = result_tuple[5] if len(result_tuple) > 5 else None
        drums_wav = result_tuple[6] if len(result_tuple) > 6 else None

        # Move files to output_dir
        out_piano_wav = output_dir / "piano_cover.wav"
        out_piano_mid = output_dir / "piano_cover.mid"

        if final_wav and Path(final_wav).exists():
            shutil.copy2(final_wav, out_piano_wav)
        if final_midi and Path(final_midi).exists():
            shutil.copy2(final_midi, out_piano_mid)

        # Convert WAV to 320k MP3
        out_piano_mp3 = output_dir / "piano_cover.mp3"
        if out_piano_wav.exists():
            convert_audio_format(out_piano_wav, output_format="mp3", output_path=out_piano_mp3, bitrate="320k")

        # Save backing mix
        out_mix_mp3 = output_dir / "piano_mix.mp3"
        if drums_wav and Path(drums_wav).exists() and out_piano_mp3.exists():
            mix_piano_with_backing(out_piano_mp3, Path(drums_wav), out_mix_mp3)
        elif out_piano_mp3.exists():
            shutil.copy2(out_piano_mp3, out_mix_mp3)

        # Move stems
        stems_dict = {}
        for s_name, s_file in [("vocals", vocals_wav), ("other", other_wav), ("bass", bass_wav), ("drums", drums_wav)]:
            if s_file and Path(s_file).exists():
                dst = output_dir / f"{s_name}.wav"
                shutil.copy2(s_file, dst)
                stems_dict[s_name] = str(dst)

        # Analysis
        theory = detect_bpm_and_key(input_audio_path)

        _prog(100, "Cloud Piano Cover complete!")

        return {
            "solo_piano_mp3": str(out_piano_mp3) if out_piano_mp3.exists() else None,
            "solo_piano_wav": str(out_piano_wav) if out_piano_wav.exists() else None,
            "piano_mix_mp3": str(out_mix_mp3) if out_mix_mp3.exists() else None,
            "piano_midi": str(out_piano_mid) if out_piano_mid.exists() else None,
            "stems": stems_dict,
            "stats": stats_md,
            "bpm": theory["bpm"],
            "key": theory["key"],
            "engine_used": "NVIDIA ZeroGPU (A100) — StrikeKhan/ai-piano-cover",
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: UNIFIED PIANO ENGINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedPianoEngine:
    """
    Main Entry Point: Orchestrates Cloud GPU and Local Fallback workflows seamlessly.
    """

    def __init__(self):
        self.cloud_processor = CloudPianoProcessor()

    def process(
        self,
        input_audio_path: str,
        output_dir: str,
        engine_mode: str = "auto",  # 'auto', 'cloud', 'local'
        trim_start: Optional[float] = None,
        trim_end: Optional[float] = None,
        keep_drums: bool = True,
        stems_to_use: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        in_p = Path(input_audio_path).resolve()
        out_p = Path(output_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        if not in_p.exists():
            raise FileNotFoundError(f"Audio file not found: {in_p}")

        def _prog(p: int, m: str):
            if progress_callback:
                progress_callback(p, m)

        # Handle Trimming if user requested a specific clip
        active_audio_path = in_p
        total_dur = get_audio_duration(in_p)
        needs_trim = False
        if trim_start and trim_start > 0.2:
            needs_trim = True
        if trim_end and total_dur > 0 and (total_dur - trim_end) > 1.0:
            needs_trim = True

        if needs_trim:
            _prog(5, f"Trimming audio ({trim_start or 0:.1f}s to {trim_end or 'end'}s)...")
            trimmed_mp3 = out_p / "trimmed_input.mp3"
            active_audio_path = trim_audio(in_p, trimmed_mp3, start_sec=trim_start or 0.0, end_sec=trim_end)

        # Execute based on preference
        if engine_mode == "local":
            _prog(10, "Starting Local Acoustic Piano Cover Engine...")
            return run_local_piano_workflow(active_audio_path, out_p, progress_callback=_prog)

        if engine_mode == "cloud":
            _prog(10, "Dispatching to Cloud NVIDIA ZeroGPU...")
            return self.cloud_processor.run_cloud_pipeline(
                active_audio_path, out_p, stems_to_use=stems_to_use, keep_drums=keep_drums, progress_callback=_prog
            )

        # Mode == 'auto': Try Cloud GPU first, fallback to Local on failure
        _prog(10, "Connecting to Cloud GPU (ZeroGPU A100)...")
        try:
            return self.cloud_processor.run_cloud_pipeline(
                active_audio_path, out_p, stems_to_use=stems_to_use, keep_drums=keep_drums, progress_callback=_prog
            )
        except Exception as cloud_err:
            logger.warning(f"Cloud GPU failed or quota full ({cloud_err}). Switching to Local Engine fallback...")
            _prog(20, "Cloud GPU offline. Seamlessly activating Local Acoustic Grand Engine...")
            res = run_local_piano_workflow(active_audio_path, out_p, progress_callback=_prog)
            res["notice"] = f"Processed on Local Engine (Cloud status: {cloud_err})"
            return res


unified_piano_engine = UnifiedPianoEngine()
