# 🎹 Piano Cover AI Studio (Mastered Acoustic Grand Engine)

Autonomous Acoustic Piano Cover Engine & Studio Workflow with a luxury Dark Studio frontend.

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

## ⚡ Quick Start

Double-click `run_studio.bat` or run:
```bash
python main.py
```
Open http://localhost:8055 in your browser.
