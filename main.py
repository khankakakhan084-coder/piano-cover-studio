"""
main.py
FastAPI Web Server for Piano Cover Studio.
Provides RESTful APIs, SSE Live Progress Streaming, Audio Streaming (Range requests),
Multi-Track Downloads, All-In-One ZIP Bundling, and Static Frontend Hosting.
"""
import os
import sys
import uuid
import json
import time
import shutil
import zipfile
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Header, Response
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from token_manager import token_manager
from audio_utils import get_audio_duration
from piano_engine import unified_piano_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("piano_studio_server")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
OUTPUTS_DIR = STORAGE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Piano Cover Studio Engine API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Job Store
JOBS: Dict[str, Dict[str, Any]] = {}


def _update_job_progress(job_id: str, progress: int, stage: str):
    if job_id in JOBS:
        JOBS[job_id]["progress"] = progress
        JOBS[job_id]["stage"] = stage
        JOBS[job_id]["updated_at"] = time.time()


def _run_job_background(
    job_id: str,
    input_audio_path: str,
    output_dir: str,
    engine_mode: str,
    trim_start: Optional[float],
    trim_end: Optional[float],
    keep_drums: bool,
    stems_to_use: Optional[List[str]],
):
    try:
        JOBS[job_id]["status"] = "processing"
        _update_job_progress(job_id, 5, "Initializing Piano Cover Engine...")

        res = unified_piano_engine.process(
            input_audio_path=input_audio_path,
            output_dir=output_dir,
            engine_mode=engine_mode,
            trim_start=trim_start,
            trim_end=trim_end,
            keep_drums=keep_drums,
            stems_to_use=stems_to_use,
            progress_callback=lambda p, m: _update_job_progress(job_id, p, m),
        )

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["stage"] = "Complete"
        JOBS[job_id]["result"] = res
        JOBS[job_id]["completed_at"] = time.time()
        logger.info(f"Job {job_id[:8]} completed successfully!")

    except Exception as ex:
        logger.error(f"Job {job_id[:8]} failed: {ex}", exc_info=True)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(ex)
        JOBS[job_id]["stage"] = f"Failed: {ex}"
        JOBS[job_id]["failed_at"] = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/process")
async def start_process(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    engine_mode: str = Form("auto"),
    trim_start: float = Form(0.0),
    trim_end: Optional[float] = Form(None),
    keep_drums: bool = Form(True),
    stems: str = Form("vocals,other"),
):
    job_id = str(uuid.uuid4())
    job_upload_dir = UPLOADS_DIR / job_id
    job_output_dir = OUTPUTS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_ext = Path(audio_file.filename or "audio.mp3").suffix or ".mp3"
    saved_input_path = job_upload_dir / f"input{file_ext}"

    with open(saved_input_path, "wb") as out_f:
        shutil.copyfileobj(audio_file.file, out_f)

    stems_list = [s.strip() for s in stems.split(",") if s.strip()]

    JOBS[job_id] = {
        "job_id": job_id,
        "filename": audio_file.filename,
        "status": "queued",
        "progress": 0,
        "stage": "Queued in studio pipeline...",
        "created_at": time.time(),
        "engine_mode": engine_mode,
        "input_path": str(saved_input_path),
        "output_dir": str(job_output_dir),
    }

    # Dispatch to background thread
    threading.Thread(
        target=_run_job_background,
        args=(
            job_id,
            str(saved_input_path),
            str(job_output_dir),
            engine_mode,
            trim_start if trim_start > 0 else None,
            trim_end if (trim_end and trim_end > 0) else None,
            keep_drums,
            stems_list,
        ),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    """Server-Sent Events (SSE) for live, real-time UI progress updates."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_pct = -1
        last_stage = ""
        while True:
            job = JOBS.get(job_id)
            if not job:
                break

            pct = job.get("progress", 0)
            stage = job.get("stage", "")
            status = job.get("status", "processing")

            if pct != last_pct or stage != last_stage or status in ("completed", "failed"):
                last_pct = pct
                last_stage = stage
                data = json.dumps({
                    "progress": pct,
                    "stage": stage,
                    "status": status,
                    "result": job.get("result"),
                    "error": job.get("error"),
                })
                yield f"data: {data}\n\n"

            if status in ("completed", "failed"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/audio/{job_id}/{track}")
async def stream_audio(job_id: str, track: str, range: Optional[str] = Header(None)):
    """
    Audio streaming endpoint with HTTP 206 Partial Content support for seamless web player seeking.
    """
    job = JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Audio not ready or job not found")

    res = job.get("result", {})
    file_path = None

    if track == "solo_piano":
        file_path = res.get("solo_piano_mp3") or res.get("solo_piano_wav")
    elif track == "piano_mix":
        file_path = res.get("piano_mix_mp3")
    elif track in ("vocals", "other", "drums", "bass"):
        stems = res.get("stems", {})
        file_path = stems.get(track)

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"Track '{track}' not found")

    path_obj = Path(file_path)
    file_size = path_obj.stat().st_size
    media_type = "audio/mpeg" if path_obj.suffix.lower() == ".mp3" else "audio/wav"

    if range:
        start, end = 0, None
        range_str = range.replace("bytes=", "").strip()
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iterfile():
            with open(path_obj, "rb") as f:
                f.seek(start)
                bytes_left = chunk_size
                while bytes_left > 0:
                    read_size = min(bytes_left, 64 * 1024)
                    chunk = f.read(read_size)
                    if not chunk:
                        break
                    bytes_left -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers, media_type=media_type)

    return FileResponse(path=str(path_obj), media_type=media_type)


@app.get("/api/download/{job_id}/{track}")
async def download_track(job_id: str, track: str):
    """File download endpoint with attachment header."""
    job = JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    res = job.get("result", {})
    file_path = None
    download_filename = f"piano_{track}.mp3"

    if track == "solo_piano_mp3":
        file_path = res.get("solo_piano_mp3")
        download_filename = "Mastered_Acoustic_Piano_Cover.mp3"
    elif track == "solo_piano_wav":
        file_path = res.get("solo_piano_wav")
        download_filename = "Mastered_Acoustic_Piano_Cover.wav"
    elif track == "piano_mix":
        file_path = res.get("piano_mix_mp3")
        download_filename = "Piano_and_Backing_Drums_Mix.mp3"
    elif track == "piano_midi":
        file_path = res.get("piano_midi")
        download_filename = "Piano_Score.mid"
    elif track == "chord_sheet":
        file_path = res.get("chord_sheet")
        download_filename = "Chord_Sheet_and_Harmony.json"

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Requested file not found")

    return FileResponse(
        path=str(file_path),
        filename=download_filename,
        media_type="application/octet-stream"
    )


@app.get("/api/download-all/{job_id}")
async def download_all_zip(job_id: str):
    """Packages all master tracks, MIDI score, and analysis into a single ZIP."""
    job = JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    out_dir = Path(job.get("output_dir", ""))
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    zip_path = out_dir / f"Piano_Cover_Studio_Bundle_{job_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in out_dir.glob("*"):
            if f != zip_path and f.is_file():
                zipf.write(f, arcname=f.name)

    return FileResponse(
        path=str(zip_path),
        filename=zip_path.name,
        media_type="application/zip"
    )


@app.get("/api/tokens")
async def get_tokens_info():
    """Returns token pool health and ZeroGPU status."""
    return token_manager.get_pool_status()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Piano Cover Studio Engine",
        "version": "2.0.0",
        "storage": str(STORAGE_DIR),
        "soundfont_present": get_default_soundfont() is not None,
    }


def get_default_soundfont():
    for f in (STORAGE_DIR / "soundfonts").glob("*.sf2"):
        if f.stat().st_size > 1_000_000:
            return str(f)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SERVE STATIC FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Piano Cover Studio API is live. Place index.html in static/"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8055))
    print("=" * 64)
    print("  🎹 PIANO COVER STUDIO ENGINE — V2.0")
    print(f"  🌐 Studio Web UI: http://localhost:{port}")
    print(f"  📚 Interactive API Docs: http://localhost:{port}/docs")
    print("=" * 64)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
