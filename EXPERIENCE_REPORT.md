# Demucs Web 移植經驗報告

## 專案概述

將 Meta 的 Demucs（音樂源分離 AI）從 Python/PyTorch 移植到純瀏覽器環境，使用 ONNX Runtime Web 實現 WebGPU/WASM 加速推論。

### 目標
- 純前端運行，無需後端伺服器
- 利用使用者的 GPU 進行加速（WebGPU）
- 分離效果與原生 Python 版本相當

### 最終成果
- 成功在瀏覽器中運行 HTDemucs 模型
- 支援 4 軌道分離：drums、bass、other、vocals
- 處理速度：約 0.1-0.3x 即時（視硬體而定）

---

## 技術架構

```
┌─────────────────────────────────────────────────────────────┐
│                        瀏覽器環境                            │
├─────────────────────────────────────────────────────────────┤
│  音訊輸入 (Web Audio API)                                    │
│       ↓                                                      │
│  重採樣至 44100Hz                                            │
│       ↓                                                      │
│  分段處理 (343980 samples/segment, 25% overlap)              │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 前處理                                               │    │
│  │  - Reflect Padding                                   │    │
│  │  - STFT (FFT 4096, Hop 1024)                        │    │
│  │  - 生成 magSpec [4, 2048, 336]                      │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ONNX Runtime Web                                     │    │
│  │  - 輸入: waveform [1,2,343980] + magSpec [1,4,2048,336] │
│  │  - 輸出: freq [1,4,4,2048,336] + time [1,4,2,343980]    │
│  │  - Backend: WebGPU (優先) / WASM (fallback)         │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 後處理                                               │    │
│  │  - standaloneMask: 頻域輸出 → 複數頻譜              │    │
│  │  - standaloneIspec: iSTFT → 時域                    │    │
│  │  - 合併: final = time_output + freq_iSTFT           │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  Overlap-Add 合併分段                                        │
│       ↓                                                      │
│  輸出 4 軌道音訊                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 關鍵技術細節

### 1. 模型轉換 (PyTorch → ONNX)

**挑戰**：原始 Demucs 模型包含 STFT/iSTFT 操作，這些在 ONNX 中支援不佳。

**解決方案**：修改模型，將 STFT/iSTFT 移到模型外部處理。

```python
# 修改後的模型 forward 函數返回兩個輸出：
# - x: 頻域輸出 [B, S, C*2, Fr, T] 用於 masking
# - xt: 時域輸出 [B, S, C, length] 直接的時域預測
return x, xt  # 而非完整的 iSTFT 結果
```

**嵌入權重**（避免外部數據文件問題）：
```python
import onnx
model = onnx.load("htdemucs.onnx")
onnx.save(model, "htdemucs_embedded.onnx", save_as_external_data=False)
```

### 2. STFT 實現

**關鍵參數**：
```javascript
const FFT_SIZE = 4096;
const HOP_SIZE = 1024;  // FFT_SIZE / 4
const TRAINING_SAMPLES = 343980;  // 模型固定輸入長度
```

**Padding 邏輯**（必須與 PyTorch 完全一致）：
```javascript
// standalone_spec 的 padding
const le = Math.ceil(inputLength / HOP_SIZE);  // 336
const pad = Math.floor(HOP_SIZE / 2) * 3;      // 1536
const padRight = pad + le * HOP_SIZE - inputLength;  // 1620

// center padding (torch.stft center=True)
const centerPad = FFT_SIZE / 2;  // 2048
```

**FFT 實現**：使用 Cooley-Tukey radix-2 算法
```javascript
// 預計算 twiddle factors
function getFFTTwiddles(n) {
    const real = new Float32Array(n / 2);
    const imag = new Float32Array(n / 2);
    for (let k = 0; k < n / 2; k++) {
        const angle = -2 * Math.PI * k / n;
        real[k] = Math.cos(angle);
        imag[k] = Math.sin(angle);
    }
    return { real, imag };
}
```

### 3. 後處理流程（最關鍵的部分）

這是最容易出錯的地方。Demucs 的完整輸出需要合併時域和頻域分支：

```
最終輸出 = 時域輸出 (xt) + iSTFT(頻域輸出 (x))
```

#### standaloneMask
將模型的頻域輸出 `[1, 4, 4, 2048, 336]` 轉換為複數頻譜：
```javascript
// 輸入: [batch, tracks, channels, freq_bins, time_frames]
// channels: [left_real, left_imag, right_real, right_imag]
// 輸出: 每個 track 的 {leftReal, leftImag, rightReal, rightImag}
```

#### standaloneIspec (iSTFT)
**關鍵修復點**：PyTorch 的 `istft(center=True)` 會自動處理 `n_fft//2` 的偏移！

```javascript
function standaloneIspec(trackSpec, targetLength) {
    // ... padding 邏輯 ...

    // 關鍵：正確的偏移計算
    const centerPad = FFT_SIZE / 2;  // 2048 - PyTorch center=True 的偏移
    const pad = Math.floor(hopLength / 2) * 3;  // 1536 - standalone_ispec 的 padding
    const totalOffset = centerPad + pad;  // 3584

    // 從 iSTFT 輸出中截取正確的範圍
    const left = leftOut.subarray(totalOffset, totalOffset + targetLength);
    const right = rightOut.subarray(totalOffset, totalOffset + targetLength);
}
```

### 4. 數據佈局

**PyTorch vs JavaScript 的張量佈局**：

| 操作 | PyTorch 形狀 | JavaScript 存儲順序 |
|------|-------------|-------------------|
| 模型輸入 waveform | [1, 2, 343980] | `[left..., right...]` |
| 模型輸入 magSpec | [1, 4, 2048, 336] | `channel * bins * frames + bin * frames + frame` |
| 模型輸出 freq | [1, 4, 4, 2048, 336] | `track * 4 * bins * frames + channel * bins * frames + bin * frames + frame` |
| 模型輸出 time | [1, 4, 2, 343980] | `track * 2 * samples + channel * samples + sample` |
| STFT 輸出 | [frames, bins] | `frame * bins + bin` |

---

## 遇到的問題與解決方案

### 問題 1：ONNX 外部數據文件錯誤

**現象**：
```
Error: Model loading failed: external data file not found
```

**原因**：大型模型會將權重存儲在 `.onnx.data` 文件中。

**解決方案**：
```python
# 將所有權重嵌入到單一 .onnx 文件
onnx.save(model, "model_embedded.onnx", save_as_external_data=False)
```

### 問題 2：維度不匹配 (2049x337 vs 2048x336)

**現象**：推論時報錯，維度不符合預期。

**原因**：STFT 的 bin 數量是 `FFT_SIZE/2 + 1 = 2049`，但模型期望 2048。

**解決方案**：
- 去除最後一個 bin（Nyquist 頻率）
- 正確計算 frame offset

### 問題 3：輸出全是雜訊

**現象**：分離後的音訊都是雜訊。

**原因**：使用了錯誤的輸出張量（頻域輸出而非時域輸出）。

**解決方案**：
```javascript
// 找到時域輸出 (shape [1, 4, 2, samples])
for (const name of session.outputNames) {
    const tensor = inferResults[name];
    if (tensor.dims.length === 4 && tensor.dims[2] === 2) {
        timeData = tensor.data;
        break;
    }
}
```

### 問題 4：分離效果差（其他音軌混入）

**現象**：vocal 還可以，但 drums/bass/other 互相混入。

**原因**：只使用了時域輸出，沒有加上頻域分支的 iSTFT 結果。

**解決方案**：
```javascript
// 完整的後處理流程
const trackSpecs = standaloneMask(freqData);
for (let t = 0; t < 4; t++) {
    const freqOutput = standaloneIspec(trackSpecs[t], TRAINING_SAMPLES);
    // 合併時域和頻域
    combined.left[i] = timeLeft[i] + freqOutput.left[i];
    combined.right[i] = timeRight[i] + freqOutput.right[i];
}
```

### 問題 5：iSTFT 輸出偏移錯誤

**現象**：輸出的前幾百個樣本是 0，與 Python 參考值不符。

**原因**：JavaScript iSTFT 沒有處理 PyTorch `center=True` 的偏移。

**診斷方法**：
```javascript
// 找到第一個非零值的索引
let firstNonZero = -1;
for (let i = 0; i < output.length; i++) {
    if (output[i] !== 0) { firstNonZero = i; break; }
}
console.log('First non-zero at:', firstNonZero);  // 顯示 2049
```

**解決方案**：
```javascript
// 正確的偏移計算
const centerPad = FFT_SIZE / 2;  // 2048
const pad = Math.floor(hopLength / 2) * 3;  // 1536
const totalOffset = centerPad + pad;  // 3584
```

---

## 性能優化

### 1. FFT 優化
- 使用 Cooley-Tukey radix-2 算法替代 DFT
- 預計算 twiddle factors 並快取
- 速度提升：100-300 倍

### 2. 記憶體優化
- 重用 Float32Array 緩衝區
- 分段處理長音訊
- 及時釋放不需要的張量

### 3. WebGPU 考量
- ONNX Runtime Web 的 WebGPU backend 對 Conv1d 支援有限
- 目前 fallback 到 WASM backend
- 未來可考慮使用 WebGPU 加速 FFT

---

## 測試驗證方法

開發過程中使用以下方法驗證正確性：

### 1. STFT 驗證
比較 JavaScript 和 Python 的 STFT 輸出：
```javascript
// 生成相同的測試信號
const left = new Float32Array(TRAINING_SAMPLES);
for (let i = 0; i < TRAINING_SAMPLES; i++) {
    left[i] = Math.sin(2 * Math.PI * 440 * i / 44100);
}
// 比較輸出值與 Python torch.stft 結果
```

### 2. iSTFT 驗證
使用 Python 生成參考數據，在 JavaScript 中比對：
```javascript
// JavaScript 中比較
const diff = Math.abs(jsOutput[i] - pyExpected[i]);
console.log('Max diff:', maxDiff);  // 應該接近 0
```

### 3. 端到端驗證
比較 ONNX Runtime 和 PyTorch 的推論結果，確保輸出一致。

---

## 檔案結構

```
demucs-web/
├── src/
│   ├── index.js                   # 主要導出
│   ├── constants.js               # 常數定義
│   ├── fft.js                     # FFT/iFFT/STFT/iSTFT 實現
│   └── processor.js               # DemucsProcessor 主類別
├── demo/
│   ├── index.html                 # Demo 頁面
│   └── app.js                     # Demo 應用邏輯
├── models/
│   └── htdemucs_embedded.onnx     # ONNX 模型 (172MB)
├── package.json                   # npm 套件設定
├── server.py                      # 開發伺服器 (含 COOP/COEP headers)
├── README.md                      # 使用說明
└── EXPERIENCE_REPORT.md           # 本報告
```

---

## 未來改進方向

1. **WebGPU 加速**
   - 等待 ONNX Runtime Web 改善 Conv1d 支援
   - 或自行實現 WebGPU FFT kernel

2. **模型量化**
   - 嘗試 INT8/FP16 量化減少模型大小
   - 可能從 172MB 減少到 ~50MB

3. **串流處理**
   - 實現即時音訊串流分離
   - 需要更低延遲的分段策略

4. **Worker 執行緒**
   - 將推論移到 Web Worker
   - 避免阻塞 UI 執行緒

---

## 重要經驗教訓

1. **永遠先驗證中間結果**
   - 不要假設任何轉換是正確的
   - 建立測試頁面逐步驗證 STFT → 模型推論 → iSTFT

2. **注意 PyTorch 的隱式行為**
   - `center=True` 會自動處理偏移
   - `normalized=True` 會影響縮放因子
   - 張量的記憶體佈局（row-major vs column-major）

3. **數據佈局是最常見的錯誤來源**
   - PyTorch 張量是 row-major
   - JavaScript TypedArray 也是 row-major
   - 但多維索引的順序容易搞混

4. **建立完整的測試管道**
   - Python 參考實現
   - 中間數據導出
   - JavaScript 對比驗證

---

## 參考資源

- [Demucs 原始專案](https://github.com/facebookresearch/demucs)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [Cooley-Tukey FFT 算法](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
