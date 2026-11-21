/**
 * Piano Cover AI Studio — Frontend Engine Controller
 * Manages audio uploads, trimming preview, SSE live progress,
 * multi-track audio playback, dynamic canvas piano roll, and asset downloads.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── State ───────────────────────────────────────────────────────────────────
  let selectedFile = null;
  let currentJobId = null;
  let activeTrack = 'solo_piano';
  let eventSource = null;
  let audioContext = null;
  let isPlaying = false;
  let currentResult = null;

  // ── DOM Elements ────────────────────────────────────────────────────────────
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('audioFileInput');
  const dropzoneEmpty = document.getElementById('dropzoneEmpty');
  const dropzoneLoaded = document.getElementById('dropzoneLoaded');
  const loadedFileName = document.getElementById('loadedFileName');
  const loadedFileSize = document.getElementById('loadedFileSize');
  const btnRemoveFile = document.getElementById('btnRemoveFile');

  const inputAudioElement = document.getElementById('inputAudioElement');
  const trimStart = document.getElementById('trimStart');
  const trimEnd = document.getElementById('trimEnd');
  const btnFullSong = document.getElementById('btnFullSong');
  const btnFirst30 = document.getElementById('btnFirst30');
  const btnFirst60 = document.getElementById('btnFirst60');
  const btnPlayPreview = document.getElementById('btnPlayPreview');

  const engineModeSelect = document.getElementById('engineModeSelect');
  const arrangementModeSelect = document.getElementById('arrangementModeSelect');
  const stemsToTranscribeSelect = document.getElementById('stemsToTranscribeSelect');
  const btnGenerate = document.getElementById('btnGenerate');

  const progressSection = document.getElementById('progressSection');
  const progressBarFill = document.getElementById('progressBarFill');
  const progressStageText = document.getElementById('progressStageText');
  const progressPctText = document.getElementById('progressPctText');
  const livePill = document.getElementById('livePill');

  const resultsSection = document.getElementById('resultsSection');
  const masterAudioPlayer = document.getElementById('masterAudioPlayer');
  const btnPlayMaster = document.getElementById('btnPlayMaster');
  const playIcon = document.getElementById('playIcon');
  const pauseIcon = document.getElementById('pauseIcon');
  const scrubberRange = document.getElementById('scrubberRange');
  const currentTimeLabel = document.getElementById('currentTimeLabel');
  const totalDurationLabel = document.getElementById('totalDurationLabel');
  const volumeRange = document.getElementById('volumeRange');
  const btnMute = document.getElementById('btnMute');

  const trackTabs = document.querySelectorAll('.track-tab');
  const tabVocals = document.getElementById('tabVocals');
  const tabOther = document.getElementById('tabOther');

  const insightKey = document.getElementById('insightKey');
  const insightBpm = document.getElementById('insightBpm');
  const insightNotes = document.getElementById('insightNotes');
  const insightEngine = document.getElementById('insightEngine');

  const btnDlMp3 = document.getElementById('btnDlMp3');
  const btnDlMidi = document.getElementById('btnDlMidi');
  const btnDlWav = document.getElementById('btnDlWav');
  const btnDlZip = document.getElementById('btnDlZip');

  const pianoRollCanvas = document.getElementById('pianoRollCanvas');
  const pianoKeysBar = document.getElementById('pianoKeysBar');
  const gpuStatusText = document.getElementById('gpuStatusText');

  // ── Init GPU Pool Status ────────────────────────────────────────────────────
  fetch('/api/tokens')
    .then(r => r.json())
    .then(data => {
      if (data && data.active_tokens) {
        gpuStatusText.textContent = `Cloud GPU Ready (${data.active_tokens}/${data.total_tokens} Pools)`;
      }
    })
    .catch(() => {
      gpuStatusText.textContent = 'Local Engine Ready';
    });

  // Build 88 Mini Visual Piano Keys
  buildPianoKeyboard();

  // ── File Upload & Dropzone Handling ─────────────────────────────────────────
  dropzone.addEventListener('click', (e) => {
    if (!e.target.closest('.audio-trimmer-box') && !e.target.closest('#btnRemoveFile')) {
      fileInput.click();
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.includes('audio') && !file.name.match(/\.(mp3|wav|flac|m4a|ogg|aac)$/i)) {
      alert('Please upload a valid audio file (MP3, WAV, FLAC, M4A, OGG).');
      return;
    }
    selectedFile = file;
    loadedFileName.textContent = file.name;
    loadedFileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

    dropzoneEmpty.style.display = 'none';
    dropzoneLoaded.style.display = 'block';
    btnGenerate.disabled = false;

    // Load into preview audio player
    const objectUrl = URL.createObjectURL(file);
    inputAudioElement.src = objectUrl;
    inputAudioElement.onloadedmetadata = () => {
      const dur = inputAudioElement.duration;
      trimEnd.value = dur.toFixed(1);
      trimEnd.max = dur.toFixed(1);
    };
  }

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    dropzoneEmpty.style.display = 'block';
    dropzoneLoaded.style.display = 'none';
    btnGenerate.disabled = true;
    inputAudioElement.pause();
    inputAudioElement.src = '';
  });

  // Trimmer Preset Chips
  btnFullSong.addEventListener('click', (e) => {
    e.stopPropagation();
    trimStart.value = '0.0';
    trimEnd.value = inputAudioElement.duration ? inputAudioElement.duration.toFixed(1) : '';
  });

  btnFirst30.addEventListener('click', (e) => {
    e.stopPropagation();
    trimStart.value = '0.0';
    trimEnd.value = '30.0';
  });

  btnFirst60.addEventListener('click', (e) => {
    e.stopPropagation();
    trimStart.value = '0.0';
    trimEnd.value = '60.0';
  });

  let previewPlaying = false;
  btnPlayPreview.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!inputAudioElement.src) return;

    if (previewPlaying) {
      inputAudioElement.pause();
      btnPlayPreview.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Preview Range`;
      previewPlaying = false;
    } else {
      const st = parseFloat(trimStart.value) || 0;
      const et = parseFloat(trimEnd.value) || inputAudioElement.duration;
      inputAudioElement.currentTime = st;
      inputAudioElement.play();
      btnPlayPreview.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause`;
      previewPlaying = true;

      const checkEndInterval = setInterval(() => {
        if (inputAudioElement.currentTime >= et || inputAudioElement.paused) {
          inputAudioElement.pause();
          btnPlayPreview.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Preview Range`;
          previewPlaying = false;
          clearInterval(checkEndInterval);
        }
      }, 200);
    }
  });

  // ── Generation Workflow Dispatch ────────────────────────────────────────────
  btnGenerate.addEventListener('click', async () => {
    if (!selectedFile) return;

    btnGenerate.disabled = true;
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    progressSection.scrollIntoView({ behavior: 'smooth' });

    updatePipelineStage('step-upload');
    updateProgressBar(5, 'Uploading audio file to Studio...');

    const formData = new FormData();
    formData.append('audio_file', selectedFile);
    formData.append('engine_mode', engineModeSelect.value);
    formData.append('trim_start', parseFloat(trimStart.value) || 0.0);

    const endVal = parseFloat(trimEnd.value);
    if (endVal && endVal > 0) {
      formData.append('trim_end', endVal);
    }

    formData.append('keep_drums', arrangementModeSelect.value === 'piano_mix');
    formData.append('stems', stemsToTranscribeSelect.value);

    try {
      const resp = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        throw new Error(`Upload failed with status ${resp.status}`);
      }

      const data = await resp.json();
      currentJobId = data.job_id;
      startProgressTracking(currentJobId);

    } catch (err) {
      alert(`Error starting generation: ${err.message}`);
      btnGenerate.disabled = false;
      progressSection.style.display = 'none';
    }
  });

  // ── Live SSE Progress Streaming ─────────────────────────────────────────────
  function startProgressTracking(jobId) {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource(`/api/stream/${jobId}`);

    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const pct = payload.progress || 0;
        const stage = payload.stage || 'Processing...';

        updateProgressBar(pct, stage);

        // Map stage to step cards
        const stageLower = stage.toLowerCase();
        if (stageLower.includes('upload') || stageLower.includes('ingest') || stageLower.includes('trim')) {
          updatePipelineStage('step-upload');
        } else if (stageLower.includes('separat') || stageLower.includes('stem')) {
          updatePipelineStage('step-separate');
        } else if (stageLower.includes('transcrib') || stageLower.includes('bytedance') || stageLower.includes('pyin')) {
          updatePipelineStage('step-transcribe');
        } else if (stageLower.includes('harmoniz') || stageLower.includes('legato') || stageLower.includes('dissonance')) {
          updatePipelineStage('step-harmonize');
        } else if (stageLower.includes('sf2') || stageLower.includes('render') || stageLower.includes('acoustic') || stageLower.includes('soundfont')) {
          updatePipelineStage('step-render');
        } else if (stageLower.includes('midi') || stageLower.includes('master') || stageLower.includes('final')) {
          updatePipelineStage('step-master');
        }

        if (payload.status === 'completed') {
          eventSource.close();
          onJobComplete(payload.result);
        } else if (payload.status === 'failed') {
          eventSource.close();
          onJobFailed(payload.error);
        }

      } catch (e) {
        console.error('Error parsing SSE data:', e);
      }
    };

    eventSource.onerror = () => {
      // Fallback to polling if SSE fails
      eventSource.close();
      pollJobStatus(jobId);
    };
  }

  function pollJobStatus(jobId) {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${jobId}`);
        const job = await res.json();
        updateProgressBar(job.progress, job.stage);

        if (job.status === 'completed') {
          clearInterval(pollInterval);
          onJobComplete(job.result);
        } else if (job.status === 'failed') {
          clearInterval(pollInterval);
          onJobFailed(job.error);
        }
      } catch (err) {
        console.error('Poll notice:', err);
      }
    }, 1500);
  }

  function updateProgressBar(pct, stage) {
    progressBarFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    progressPctText.textContent = `${pct}%`;
    progressStageText.textContent = stage;
  }

  function updatePipelineStage(stepId) {
    const allSteps = ['step-upload', 'step-separate', 'step-transcribe', 'step-harmonize', 'step-render', 'step-master'];
    const currentIdx = allSteps.indexOf(stepId);

    allSteps.forEach((id, idx) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.remove('active', 'completed');
      if (idx < currentIdx) {
        el.classList.add('completed');
      } else if (idx === currentIdx) {
        el.classList.add('active');
      }
    });
  }

  // ── Completion Handler ──────────────────────────────────────────────────────
  function onJobComplete(result) {
    currentResult = result;
    btnGenerate.disabled = false;
    livePill.textContent = 'Completed!';
    livePill.style.background = 'rgba(16, 185, 129, 0.2)';
    livePill.style.color = '#10b981';

    // Mark all steps completed
    document.querySelectorAll('.stage-step').forEach(s => s.classList.add('completed'));

    setTimeout(() => {
      progressSection.style.display = 'none';
      resultsSection.style.display = 'block';
      resultsSection.scrollIntoView({ behavior: 'smooth' });

      // Update Insights
      insightKey.textContent = result.key || 'C Major';
      insightBpm.textContent = `${result.bpm || 120} BPM`;
      insightNotes.textContent = result.note_count ? `${result.note_count} Notes` : 'Full Score';
      insightEngine.textContent = result.engine_used ? result.engine_used.split('—')[0].trim() : 'ZeroGPU A100';

      // Check if stems available
      if (result.stems && result.stems.vocals) {
        tabVocals.style.display = 'inline-block';
      }
      if (result.stems && (result.stems.other || result.stems.drums)) {
        tabOther.style.display = 'inline-block';
      }

      // Configure Download Links
      btnDlMp3.href = `/api/download/${currentJobId}/solo_piano_mp3`;
      btnDlMidi.href = `/api/download/${currentJobId}/piano_midi`;
      btnDlWav.href = `/api/download/${currentJobId}/solo_piano_wav`;
      btnDlZip.href = `/api/download-all/${currentJobId}`;

      // Load initial track
      switchTrack('solo_piano');

    }, 800);
  }

  function onJobFailed(error) {
    btnGenerate.disabled = false;
    alert(`Piano generation encountered an issue:\n${error}`);
    progressSection.style.display = 'none';
  }

  // ── Studio Audio Player & Track Switcher ────────────────────────────────────
  trackTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const track = tab.dataset.track;
      trackTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      switchTrack(track);
    });
  });

  function switchTrack(track) {
    activeTrack = track;
    const wasPlaying = !masterAudioPlayer.paused && masterAudioPlayer.currentTime > 0;
    const curTime = masterAudioPlayer.currentTime || 0;

    masterAudioPlayer.src = `/api/audio/${currentJobId}/${track}`;
    masterAudioPlayer.currentTime = curTime;

    if (wasPlaying) {
      masterAudioPlayer.play();
    }
  }

  btnPlayMaster.addEventListener('click', () => {
    if (masterAudioPlayer.paused) {
      masterAudioPlayer.play();
      isPlaying = true;
      playIcon.style.display = 'none';
      pauseIcon.style.display = 'block';
      startPianoRollAnimation();
    } else {
      masterAudioPlayer.pause();
      isPlaying = false;
      playIcon.style.display = 'block';
      pauseIcon.style.display = 'none';
    }
  });

  masterAudioPlayer.addEventListener('timeupdate', () => {
    const cur = masterAudioPlayer.currentTime || 0;
    const dur = masterAudioPlayer.duration || 1;
    scrubberRange.value = (cur / dur) * 100;
    currentTimeLabel.textContent = formatTime(cur);
  });

  masterAudioPlayer.addEventListener('loadedmetadata', () => {
    totalDurationLabel.textContent = formatTime(masterAudioPlayer.duration || 0);
  });

  masterAudioPlayer.addEventListener('ended', () => {
    isPlaying = false;
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
  });

  scrubberRange.addEventListener('input', () => {
    const dur = masterAudioPlayer.duration || 1;
    masterAudioPlayer.currentTime = (scrubberRange.value / 100) * dur;
  });

  volumeRange.addEventListener('input', () => {
    masterAudioPlayer.volume = parseFloat(volumeRange.value);
  });

  btnMute.addEventListener('click', () => {
    masterAudioPlayer.muted = !masterAudioPlayer.muted;
    btnMute.style.opacity = masterAudioPlayer.muted ? '0.4' : '1';
  });

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  // ── Interactive Piano Roll Canvas & 88-Key Keyboard ─────────────────────────
  function buildPianoKeyboard() {
    pianoKeysBar.innerHTML = '';
    // Generate 52 white keys and interleaved black keys (representing full standard range)
    for (let i = 0; i < 52; i++) {
      const wk = document.createElement('div');
      wk.className = 'key-white';
      wk.dataset.index = i;
      pianoKeysBar.appendChild(wk);

      // Add black keys pattern (skipping B-C and E-F)
      const noteInOct = i % 7;
      if (noteInOct !== 2 && noteInOct !== 6 && i < 51) {
        const bk = document.createElement('div');
        bk.className = 'key-black';
        bk.dataset.index = `b_${i}`;
        pianoKeysBar.appendChild(bk);
      }
    }
  }

  // Canvas Note Particles Animation
  const ctx = pianoRollCanvas.getContext('2d');
  let noteParticles = [];

  function generateSampleNotes() {
    noteParticles = [];
    const count = 120;
    for (let i = 0; i < count; i++) {
      noteParticles.push({
        x: Math.random() * pianoRollCanvas.width,
        y: Math.random() * pianoRollCanvas.height,
        w: 12 + Math.random() * 28,
        h: 6 + Math.random() * 8,
        speed: 1.5 + Math.random() * 2.5,
        color: i % 2 === 0 ? '#f59e0b' : '#3b82f6',
      });
    }
  }
  generateSampleNotes();

  function startPianoRollAnimation() {
    function renderRoll() {
      if (!isPlaying) return;

      ctx.fillStyle = '#08090d';
      ctx.fillRect(0, 0, pianoRollCanvas.width, pianoRollCanvas.height);

      // Grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      for (let y = 0; y < pianoRollCanvas.height; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(pianoRollCanvas.width, y);
        ctx.stroke();
      }

      // Draw and scroll falling notes
      noteParticles.forEach(n => {
        n.y += n.speed;
        if (n.y > pianoRollCanvas.height) {
          n.y = -n.h;
          n.x = Math.random() * pianoRollCanvas.width;
        }

        ctx.fillStyle = n.color;
        ctx.shadowColor = n.color;
        ctx.shadowBlur = 8;
        ctx.fillRect(n.x, n.y, n.w, n.h);
        ctx.shadowBlur = 0;
      });

      // Light up random piano keys on key hits
      const whiteKeys = document.querySelectorAll('.key-white');
      whiteKeys.forEach(k => k.classList.remove('active'));
      const activeIdx = Math.floor(Math.random() * whiteKeys.length);
      if (whiteKeys[activeIdx] && Math.random() > 0.4) {
        whiteKeys[activeIdx].classList.add('active');
      }

      requestAnimationFrame(renderRoll);
    }

    renderRoll();
  }

});
