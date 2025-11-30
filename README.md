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
  onProgress?: (progress: number) => void;
  onLog?: (phase: string, message: string) => void;
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
