/**
 * Demucs Web Demo App
 */
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.mjs';
import { DemucsProcessor, CONSTANTS } from '../src/index.js';

const { SAMPLE_RATE, TRAINING_SAMPLES, TRACKS, DEFAULT_MODEL_URL } = CONSTANTS;

// Use local model as fallback if cloud model fails
const LOCAL_MODEL_URL = '../models/htdemucs_embedded.onnx';
const SEGMENT_LENGTH = TRAINING_SAMPLES / SAMPLE_RATE;

// Global state
let processor = null;
let audioContext = null;
let audioBuffer = null;

// DOM elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const progressFill = document.getElementById('progressFill');
const status = document.getElementById('status');
const results = document.getElementById('results');
const trackList = document.getElementById('trackList');
const backendBadge = document.getElementById('backendBadge');
const audioFileName = document.getElementById('audioFileName');
const statusDetail = document.getElementById('statusDetail');
const statsRow = document.getElementById('statsRow');
const statElapsed = document.getElementById('statElapsed');
const statSegment = document.getElementById('statSegment');
const statSpeed = document.getElementById('statSpeed');
const statETA = document.getElementById('statETA');

let processStartTime = null;
let segmentTimes = [];

function log(phase, message) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('zh-TW', { hour12: false });
    const logLine = document.createElement('div');
    logLine.className = 'log-line';
    logLine.innerHTML = `<span class="log-time">[${timeStr}]</span><span class="log-phase">[${phase}]</span>${message}`;
    statusDetail.appendChild(logLine);
    statusDetail.scrollTop = statusDetail.scrollHeight;
    console.log(`[${phase}] ${message}`);
}

function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

async function init() {
    let backend = 'wasm';

    if ('gpu' in navigator) {
        try {
            const gpuAdapter = await navigator.gpu.requestAdapter();
            if (gpuAdapter) {
                backend = 'webgpu';
            }
        } catch (e) {
            console.log('WebGPU not available:', e);
        }
    }

    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

    if (backend === 'webgpu') {
        ort.env.webgpu = ort.env.webgpu || {};
        ort.env.webgpu.powerPreference = 'high-performance';
        backendBadge.textContent = 'WebGPU (GPU加速)';
        backendBadge.style.background = 'rgba(100, 255, 218, 0.2)';
        backendBadge.style.color = '#64ffda';
    } else {
        const threads = navigator.hardwareConcurrency || 4;
        backendBadge.textContent = `WebAssembly (CPU ${threads}線程)`;
        backendBadge.style.background = 'rgba(255, 200, 100, 0.2)';
        backendBadge.style.color = '#ffc864';
    }

    processor = new DemucsProcessor({
        ort,
        onProgress: ({ progress, currentSegment, totalSegments }) => {
            progressFill.style.width = (5 + progress * 90) + '%';

            // Update stats
            const elapsed = (Date.now() - processStartTime) / 1000;
            statElapsed.textContent = formatTime(elapsed);
            statSegment.textContent = `${currentSegment}/${totalSegments}`;

            // Calculate speed and ETA
            if (currentSegment > 0 && audioBuffer) {
                const processedDuration = (currentSegment / totalSegments) * audioBuffer.duration;
                const speed = processedDuration / elapsed;
                statSpeed.textContent = speed.toFixed(2) + 'x';

                const remainingSegments = totalSegments - currentSegment;
                const avgTimePerSegment = elapsed / currentSegment;
                const eta = remainingSegments * avgTimePerSegment;
                statETA.textContent = formatTime(eta);
            }
        },
        onLog: log,
        onDownloadProgress: (loaded, total) => {
            const percent = ((loaded / total) * 100).toFixed(1);
            const loadedMB = (loaded / 1024 / 1024).toFixed(1);
            const totalMB = (total / 1024 / 1024).toFixed(1);
            status.textContent = `下載模型中... ${loadedMB}MB / ${totalMB}MB (${percent}%)`;
            progressFill.style.width = (loaded / total * 100) + '%';
        }
    });

    status.textContent = '載入模型中...';

    try {
        // Try cloud model first, fallback to local
        try {
            status.textContent = '從 Hugging Face 下載模型中 (約 172MB)...';
            await processor.loadModel(DEFAULT_MODEL_URL);
        } catch {
            status.textContent = '載入本地模型...';
            await processor.loadModel(LOCAL_MODEL_URL);
        }
        status.textContent = '模型載入完成，請選擇音訊檔案';
    } catch (e) {
        status.textContent = '模型載入失敗: ' + e.message;
        console.error('Failed to load model:', e);
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE
    });
}

// Drag and drop handlers
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
        handleFile(file);
    }
});
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

async function handleFile(file) {
    audioFileName.textContent = file.name;
    status.textContent = '讀取音訊檔案...';

    try {
        const arrayBuffer = await file.arrayBuffer();
        audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const duration = audioBuffer.duration.toFixed(1);
        const channels = audioBuffer.numberOfChannels;
        status.textContent = `已載入: ${duration}秒, ${channels}聲道`;
        processBtn.disabled = false;
    } catch (e) {
        status.textContent = '無法讀取音訊檔案: ' + e.message;
        console.error('Failed to decode audio:', e);
    }
}

processBtn.addEventListener('click', async () => {
    if (!audioBuffer || !processor) return;

    processBtn.disabled = true;
    results.classList.remove('visible');
    processStartTime = Date.now();
    segmentTimes = [];
    statusDetail.innerHTML = '';
    statusDetail.classList.add('visible');
    statsRow.style.display = 'flex';

    try {
        log('初始化', '開始處理音訊...');
        status.textContent = '準備音訊資料...';
        progressFill.style.width = '2%';

        let leftChannel = audioBuffer.getChannelData(0);
        let rightChannel = audioBuffer.numberOfChannels > 1
            ? audioBuffer.getChannelData(1)
            : leftChannel;

        // Resample if needed
        if (audioBuffer.sampleRate !== SAMPLE_RATE) {
            log('重取樣', `${audioBuffer.sampleRate}Hz → ${SAMPLE_RATE}Hz`);
            const ratio = SAMPLE_RATE / audioBuffer.sampleRate;
            const newLength = Math.floor(leftChannel.length * ratio);
            const newLeft = new Float32Array(newLength);
            const newRight = new Float32Array(newLength);

            for (let i = 0; i < newLength; i++) {
                const srcIdx = i / ratio;
                const idx0 = Math.floor(srcIdx);
                const idx1 = Math.min(idx0 + 1, leftChannel.length - 1);
                const frac = srcIdx - idx0;
                newLeft[i] = leftChannel[idx0] * (1 - frac) + leftChannel[idx1] * frac;
                newRight[i] = rightChannel[idx0] * (1 - frac) + rightChannel[idx1] * frac;
            }

            leftChannel = newLeft;
            rightChannel = newRight;
        }

        status.textContent = '處理中...';
        const separatedTracks = await processor.separate(leftChannel, rightChannel);
        displayResults(separatedTracks);

        const totalTime = ((Date.now() - processStartTime) / 1000).toFixed(1);
        const totalDuration = audioBuffer.duration;
        const speedRatio = (totalDuration / parseFloat(totalTime)).toFixed(2);

        log('完成', `總耗時: ${totalTime}秒, 處理速度: ${speedRatio}x 即時`);
        status.textContent = `處理完成！(${totalTime}秒, ${speedRatio}x 即時速度)`;
        progressFill.style.width = '100%';

    } catch (e) {
        status.textContent = '處理失敗: ' + e.message;
        console.error('Processing failed:', e);
    }

    processBtn.disabled = false;
});

function displayResults(tracks) {
    trackList.innerHTML = '';

    for (const [name, track] of Object.entries(tracks)) {
        const trackDiv = document.createElement('div');
        trackDiv.className = 'track';

        const trackBuffer = audioContext.createBuffer(2, track.left.length, SAMPLE_RATE);
        trackBuffer.getChannelData(0).set(track.left);
        trackBuffer.getChannelData(1).set(track.right);

        const audioBlob = audioBufferToWav(trackBuffer);
        const audioUrl = URL.createObjectURL(audioBlob);

        trackDiv.innerHTML = `
            <span class="track-name">${name.charAt(0).toUpperCase() + name.slice(1)}</span>
            <div class="track-controls">
                <audio controls src="${audioUrl}" style="height: 32px;"></audio>
                <a href="${audioUrl}" download="${name}.wav" class="track-btn">下載</a>
            </div>
        `;

        trackList.appendChild(trackDiv);
    }

    results.classList.add('visible');
}

function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bitDepth = 16;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const samples = buffer.length;
    const dataSize = samples * blockAlign;
    const bufferSize = 44 + dataSize;

    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);

    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    const channels = [];
    for (let c = 0; c < numChannels; c++) {
        channels.push(buffer.getChannelData(c));
    }

    let offset = 44;
    for (let i = 0; i < samples; i++) {
        for (let c = 0; c < numChannels; c++) {
            const sample = Math.max(-1, Math.min(1, channels[c][i]));
            const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, intSample, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

init();
