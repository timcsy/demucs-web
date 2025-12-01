# Demucs Web

在瀏覽器中運行的音樂源分離 AI，使用 ONNX Runtime Web 實現 WebGPU/WASM 加速推論。

## 功能

- 純前端運行，無需後端伺服器
- 支援 WebGPU 加速（macOS Safari、Chrome 等）
- 支援 4 軌道分離：drums、bass、other、vocals
- 基於 Meta 的 HTDemucs 模型

## 安裝

```bash
npm install demucs-web
```

## 使用方式

```javascript
import * as ort from 'onnxruntime-web';
import { DemucsProcessor } from 'demucs-web';

// 初始化處理器
const processor = new DemucsProcessor({
  ort,
  onProgress: (progress) => console.log(`Progress: ${(progress * 100).toFixed(1)}%`),
  onLog: (phase, msg) => console.log(`[${phase}] ${msg}`)
});

// 載入模型（約 172MB）
await processor.loadModel('./htdemucs_embedded.onnx');

// 分離音軌（輸入為 44100Hz 立體聲）
const result = await processor.separate(leftChannel, rightChannel);

// 結果包含四個軌道
console.log(result.drums);   // { left: Float32Array, right: Float32Array }
console.log(result.bass);
console.log(result.other);
console.log(result.vocals);
```

## Demo

```bash
# 啟動開發伺服器
cd demo
python3 -m http.server 8080

# 需要 COOP/COEP headers 支援 SharedArrayBuffer
# 可使用以下 server.py：
python3 server.py
```

## 重要注意事項

### Cross-Origin Isolation

ONNX Runtime Web 需要 SharedArrayBuffer，必須設定以下 HTTP headers：

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### 模型檔案

ONNX 模型（約 172MB）託管於 Hugging Face Hub：

**下載連結**：https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx

```javascript
// 方式 1：使用預設 URL（從 Hugging Face 下載）
import { CONSTANTS } from 'demucs-web';
await processor.loadModel(CONSTANTS.DEFAULT_MODEL_URL);

// 方式 2：使用本地檔案
await processor.loadModel('./htdemucs_embedded.onnx');

// 方式 3：傳入 ArrayBuffer
const response = await fetch(modelUrl);
const buffer = await response.arrayBuffer();
await processor.loadModel(buffer);
```

**自行託管模型**：
```bash
# 安裝 huggingface_hub
uv tool install huggingface_hub
hf auth login

# 上傳到你的 HF 帳號
./scripts/upload-model.sh <your-username>
```

## API

### DemucsProcessor

```typescript
interface DemucsProcessorOptions {
  ort: typeof import('onnxruntime-web');
  modelPath?: string;
  onProgress?: (info: ProgressInfo) => void;
  onLog?: (phase: string, message: string) => void;
  onDownloadProgress?: (loaded: number, total: number) => void;
}

interface ProgressInfo {
  progress: number;        // 0-1 之間的進度值
  currentSegment: number;  // 目前處理的區段編號
  totalSegments: number;   // 總區段數量
}

class DemucsProcessor {
  constructor(options: DemucsProcessorOptions);
  loadModel(pathOrBuffer?: string | ArrayBuffer): Promise<void>;
  separate(left: Float32Array, right: Float32Array): Promise<SeparationResult>;
}

interface SeparationResult {
  drums: { left: Float32Array; right: Float32Array };
  bass: { left: Float32Array; right: Float32Array };
  other: { left: Float32Array; right: Float32Array };
  vocals: { left: Float32Array; right: Float32Array };
}
```

### 回呼函數詳細說明

#### onProgress - 處理進度回呼

在音訊分離過程中，每完成一個區段會觸發此回呼：

```javascript
const processor = new DemucsProcessor({
  ort,
  onProgress: ({ progress, currentSegment, totalSegments }) => {
    // progress: 0-1 之間的數值
    const percent = (progress * 100).toFixed(1);
    console.log(`進度: ${percent}%`);
    console.log(`區段: ${currentSegment}/${totalSegments}`);

    // 更新進度條
    progressBar.style.width = `${progress * 100}%`;

    // 計算處理速度與預估剩餘時間
    if (currentSegment > 0) {
      const elapsed = (Date.now() - startTime) / 1000;
      const processedDuration = (currentSegment / totalSegments) * audioDuration;
      const speed = processedDuration / elapsed;
      console.log(`處理速度: ${speed.toFixed(2)}x 即時`);

      const remainingSegments = totalSegments - currentSegment;
      const avgTimePerSegment = elapsed / currentSegment;
      const eta = remainingSegments * avgTimePerSegment;
      console.log(`預估剩餘時間: ${eta.toFixed(0)}秒`);
    }
  }
});
```

#### onLog - 處理階段日誌回呼

在各個處理階段會觸發此回呼，用於顯示詳細日誌：

```javascript
const processor = new DemucsProcessor({
  ort,
  onLog: (phase, message) => {
    const timeStr = new Date().toLocaleTimeString();
    console.log(`[${timeStr}][${phase}] ${message}`);

    // 常見的 phase 值：
    // - 'Init': 初始化
    // - 'Segment': 區段處理
    // - 'Inference': 模型推論
    // - 'PostProcess': 後處理
  }
});
```

#### onDownloadProgress - 模型下載進度回呼

在下載模型時觸發，用於顯示下載進度：

```javascript
const processor = new DemucsProcessor({
  ort,
  onDownloadProgress: (loaded, total) => {
    const percent = ((loaded / total) * 100).toFixed(1);
    const loadedMB = (loaded / 1024 / 1024).toFixed(1);
    const totalMB = (total / 1024 / 1024).toFixed(1);
    console.log(`下載模型: ${loadedMB}MB / ${totalMB}MB (${percent}%)`);

    // 更新下載進度條
    downloadBar.style.width = `${(loaded / total) * 100}%`;
  }
});
```

### 完整範例

```javascript
import * as ort from 'onnxruntime-web';
import { DemucsProcessor, CONSTANTS } from 'demucs-web';

// 配置 ONNX Runtime
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

// 檢測 WebGPU 支援
let backend = 'wasm';
if ('gpu' in navigator) {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      backend = 'webgpu';
      ort.env.webgpu = { powerPreference: 'high-performance' };
    }
  } catch (e) {
    console.log('WebGPU 不可用，使用 WASM');
  }
}

// 建立處理器
let startTime;
const processor = new DemucsProcessor({
  ort,
  onProgress: ({ progress, currentSegment, totalSegments }) => {
    console.log(`處理中: ${(progress * 100).toFixed(1)}% (${currentSegment}/${totalSegments})`);
  },
  onLog: (phase, msg) => {
    console.log(`[${phase}] ${msg}`);
  },
  onDownloadProgress: (loaded, total) => {
    console.log(`下載: ${(loaded / total * 100).toFixed(1)}%`);
  }
});

// 載入模型
await processor.loadModel(CONSTANTS.DEFAULT_MODEL_URL);

// 載入音訊（使用 Web Audio API）
const audioContext = new AudioContext({ sampleRate: CONSTANTS.SAMPLE_RATE });
const response = await fetch('your-audio-file.mp3');
const arrayBuffer = await response.arrayBuffer();
const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

// 取得聲道資料
const leftChannel = audioBuffer.getChannelData(0);
const rightChannel = audioBuffer.numberOfChannels > 1
  ? audioBuffer.getChannelData(1)
  : leftChannel;  // 單聲道時複製左聲道

// 開始分離
startTime = Date.now();
const result = await processor.separate(leftChannel, rightChannel);

// 處理結果
const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(`處理完成，耗時 ${totalTime} 秒`);

// result.drums, result.bass, result.other, result.vocals
// 每個軌道都有 { left: Float32Array, right: Float32Array }
```

### 常數

```javascript
import { CONSTANTS } from 'demucs-web';

CONSTANTS.SAMPLE_RATE       // 44100
CONSTANTS.FFT_SIZE          // 4096
CONSTANTS.HOP_SIZE          // 1024
CONSTANTS.TRAINING_SAMPLES  // 343980
CONSTANTS.TRACKS            // ['drums', 'bass', 'other', 'vocals']
```

### FFT 工具

```javascript
import { fft, ifft, stft, istft, reflectPad } from 'demucs-web';

// Cooley-Tukey radix-2 FFT
fft(realOut, imagOut, realIn, n);
ifft(realOut, imagOut, realIn, imagIn, n);

// Short-Time Fourier Transform
const spec = stft(signal, fftSize, hopSize);
const reconstructed = istft(spec.real, spec.imag, numFrames, numBins, fftSize, hopSize);
```

## 技術細節

詳細的移植經驗請參考 [EXPERIENCE_REPORT.md](./EXPERIENCE_REPORT.md)

## License

MIT
