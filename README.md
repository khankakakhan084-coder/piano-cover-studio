# 🎹 Piano Cover AI Studio (Mastered Acoustic Grand Engine)

Autonomous Acoustic Piano Cover Engine & Studio Workflow with a Luxury Dark Studio Web Frontend.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ZeroGPU](https://img.shields.io/badge/NVIDIA-ZeroGPU%20A100-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/spaces/StrikeKhan/ai-piano-cover)
[![SoundFont](https://img.shields.io/badge/Audio-SF2%20Sampled%20Grand-f59e0b)](#soundfont-sf2-synthesis)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🏗️ Architecture & Full Workflow

```
┌────────────────────────────────────────────────────────┐
│               Interactive Studio Frontend              │
│  - Drag & Drop Audio File (MP3, WAV, FLAC, M4A, OGG)   │
│  - Built-in Range Trimmer (Full Song / Custom Range)   │
│  - Real-Time Live Pipeline Progress (SSE Stream)       │
│  - Multi-Track Studio Audio Player (Solo / Backing)    │
│  - Interactive Real-Time Canvas Piano Roll & 88 Keys   │
│  - 1-Click Downloads (MP3 320k, WAV, MIDI Score, ZIP)  │
└───────────────────────────▲────────────────────────────┘
                            │ HTTP REST / SSE Stream
┌───────────────────────────▼────────────────────────────┐
│              FastAPI Studio Server (Port 8055)         │
│  - /api/process (Job Dispatcher & Background Worker)   │
│  - /api/stream/{job_id} (Live Server-Sent Events)      │
│  - /api/audio/{job_id}/{track} (HTTP 206 Streaming)    │
│  - /api/download/{job_id}/{track} (Direct Attachments) │
│  - /api/download-all/{job_id} (All-in-One ZIP Archive) │
└───────────────────────────▲────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│  Primary: Cloud GPU Engine   │ │  Fallback: Local AI Engine   │
│  - ZeroGPU NVIDIA A100       │ │  - pYIN Pitch & Melody Lead  │
│  - 17-Token Auto-Failover    │ │  - Scale-Locked Harmonies    │
│  - ByteDance Neural Model    │ │  - Human Performance Legato  │
│  - StrikeKhan/ai-piano-cover │ │  - Pure-Python SF2 Synthesizer│
│  - Giga Grand SF2 Acoustics  │ │  - Type-1 Standard MIDI Score│
└──────────────────────────────┘ └──────────────────────────────┘
```

---

## ✨ Features

- **Dual-Engine Pipeline**:
  - **Cloud ZeroGPU (A100)**: Connects directly to `StrikeKhan/ai-piano-cover` on Hugging Face Spaces for deep neural transcription (Demucs + ByteDance High-Resolution Piano Transcription).
  - **Local Offline Fallback**: Pure-Python sampled grand piano sound engine reading authentic 16-bit PCM waves from SoundFont 2 (`.sf2`) with pYIN pitch-tracking, automatic chord progressions, and human pianist performance simulation (CC64 sustain pedal + dynamic velocity).
- **ZeroGPU Multi-Token Pool & Failover**:
  - Pool manager automatically monitors token quotas and rotates across available tokens when a ZeroGPU limit is reached.
- **Mastered Audio Exports**:
  - `piano_cover.mp3`: 320 kbps high-fidelity solo acoustic piano cover.
  - `piano_cover.wav`: 24-bit studio master audio.
  - `piano_mix.mp3`: Blended mix of solo piano with original backing drums/rhythm.
  - `piano_cover.mid`: Standard Type-1 multi-track MIDI score (Track 1: Right Hand Melody, Track 2: Left Hand Chords & CC64 Sustain Pedal).
  - All-in-one ZIP archive containing all audio masters, stems, and MIDI score.
- **Luxury Studio Frontend**:
  - Built-in audio trimmer with quick presets (Full Song, First 30s, First 60s, Custom Range).
  - 6-stage live neural pipeline indicator powered by Server-Sent Events (SSE).
  - HTML5 audio player supporting HTTP 206 Range requests for instantaneous seeking.
  - Real-time falling-notes visualizer and 88-key interactive virtual piano roll.

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/khankakakhan084-coder/piano-cover-studio.git
cd piano-cover-studio
pip install -r requirements.txt
```

### 2. Configure Hugging Face Tokens (Optional but Recommended)
To enable unlimited Cloud ZeroGPU execution, create `storage/hf_tokens.txt` and add your Hugging Face tokens (one per line):
```text
hf_YourToken1...
hf_YourToken2...
```
*(You can also set the `HF_TOKENS` environment variable).*

### 3. Launch Studio
Double-click `run_studio.bat` or run:
```bash
python main.py
```
Open your browser and navigate to:
```
http://localhost:8055
```

---

## 📂 Project Structure

```
piano_cover_studio/
├── main.py                     # FastAPI web server, SSE streaming & endpoints
├── piano_engine.py             # Dual engine orchestrator (ZeroGPU + Local SF2)
├── token_manager.py            # HF token pool manager with cooldown tracking
├── audio_utils.py              # FFmpeg audio trimming, lameenc MP3 & BPM/Key analysis
├── requirements.txt            # Python dependencies
├── run_studio.bat              # 1-click Windows launcher
├── storage/                    # SoundFonts, uploads, outputs, and token states
│   ├── soundfonts/             # SF2 soundfonts (e.g. TimGM6mb.sf2)
│   ├── hf_tokens.example.txt   # Configuration template
│   ├── uploads/                # Temporary user uploads
│   └── outputs/                # Processed jobs and download archives
└── static/                     # Web Frontend
    ├── index.html              # Studio dark luxury interface
    ├── style.css               # Modern glassmorphism CSS
    └── app.js                  # Frontend controller & canvas visualizer
```

---

## 📜 License
MIT License
