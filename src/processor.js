/**
 * Demucs audio processor - Core separation logic
 */

import { CONSTANTS } from './constants.js';
import { stft, istft, reflectPad } from './fft.js';

const { SAMPLE_RATE, FFT_SIZE, HOP_SIZE, TRAINING_SAMPLES, MODEL_SPEC_BINS, MODEL_SPEC_FRAMES, SEGMENT_OVERLAP, TRACKS } = CONSTANTS;

/**
 * Convert model frequency output to complex spectrogram per track
 */
export function standaloneMask(freqOutput) {
  const numTracks = 4;
  const numChannels = 4;
  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const result = [];

  for (let t = 0; t < numTracks; t++) {
    const trackSpec = {
      leftReal: new Float32Array(numBins * numFrames),
      leftImag: new Float32Array(numBins * numFrames),
      rightReal: new Float32Array(numBins * numFrames),
      rightImag: new Float32Array(numBins * numFrames)
    };

    for (let f = 0; f < numFrames; f++) {
      for (let b = 0; b < numBins; b++) {
        const baseIdx = t * numChannels * numBins * numFrames;
        const outIdx = b * numFrames + f;
        trackSpec.leftReal[outIdx] = freqOutput[baseIdx + 0 * numBins * numFrames + b * numFrames + f];
        trackSpec.leftImag[outIdx] = freqOutput[baseIdx + 1 * numBins * numFrames + b * numFrames + f];
        trackSpec.rightReal[outIdx] = freqOutput[baseIdx + 2 * numBins * numFrames + b * numFrames + f];
        trackSpec.rightImag[outIdx] = freqOutput[baseIdx + 3 * numBins * numFrames + b * numFrames + f];
      }
    }
    result.push(trackSpec);
  }

  return result;
}

/**
 * Convert complex spectrogram back to time domain (iSTFT with proper offsets)
 */
export function standaloneIspec(trackSpec, targetLength) {
  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const hopLength = HOP_SIZE;
  const paddedBins = numBins + 1;
  const paddedFrames = numFrames + 4;

  const padChannel = (real, imag) => {
    const paddedReal = new Float32Array(paddedFrames * paddedBins);
    const paddedImag = new Float32Array(paddedFrames * paddedBins);

    for (let f = 0; f < numFrames; f++) {
      for (let b = 0; b < numBins; b++) {
        const srcIdx = b * numFrames + f;
        const dstFrame = f + 2;
        const dstIdx = dstFrame * paddedBins + b;
        paddedReal[dstIdx] = real[srcIdx];
        paddedImag[dstIdx] = imag[srcIdx];
      }
    }
    return { real: paddedReal, imag: paddedImag };
  };

  const leftPadded = padChannel(trackSpec.leftReal, trackSpec.leftImag);
  const rightPadded = padChannel(trackSpec.rightReal, trackSpec.rightImag);

  const centerPad = FFT_SIZE / 2;
  const pad = Math.floor(hopLength / 2) * 3;
  const istftLength = (paddedFrames - 1) * hopLength + FFT_SIZE;

  const leftOut = istft(leftPadded.real, leftPadded.imag, paddedFrames, paddedBins, FFT_SIZE, hopLength, istftLength);
  const rightOut = istft(rightPadded.real, rightPadded.imag, paddedFrames, paddedBins, FFT_SIZE, hopLength, istftLength);

  const totalOffset = centerPad + pad;
  const left = leftOut.subarray(totalOffset, totalOffset + targetLength);
  const right = rightOut.subarray(totalOffset, totalOffset + targetLength);

  return { left: new Float32Array(left), right: new Float32Array(right) };
}

/**
 * Prepare model input from stereo audio
 */
export function prepareModelInput(leftChannel, rightChannel) {
  const inputLength = TRAINING_SAMPLES;

  const paddedLeft = new Float32Array(inputLength);
  const paddedRight = new Float32Array(inputLength);
  const copyLen = Math.min(leftChannel.length, inputLength);
  paddedLeft.set(leftChannel.subarray(0, copyLen));
  paddedRight.set(rightChannel.subarray(0, copyLen));

  const le = Math.ceil(inputLength / HOP_SIZE);
  const pad = Math.floor(HOP_SIZE / 2) * 3;
  const padRight = pad + le * HOP_SIZE - inputLength;

  const stftInputLeft = reflectPad(paddedLeft, pad, padRight);
  const stftInputRight = reflectPad(paddedRight, pad, padRight);

  const centerPad = FFT_SIZE / 2;
  const centeredLeft = reflectPad(stftInputLeft, centerPad, centerPad);
  const centeredRight = reflectPad(stftInputRight, centerPad, centerPad);

  const stftLeft = stft(centeredLeft, FFT_SIZE, HOP_SIZE);
  const stftRight = stft(centeredRight, FFT_SIZE, HOP_SIZE);

  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const frameOffset = 2;

  const magSpec = new Float32Array(4 * numBins * numFrames);

  for (let f = 0; f < numFrames; f++) {
    const srcFrame = f + frameOffset;
    for (let b = 0; b < numBins; b++) {
      const srcIdx = srcFrame * stftLeft.numBins + b;
      magSpec[0 * numBins * numFrames + b * numFrames + f] = stftLeft.real[srcIdx];
      magSpec[1 * numBins * numFrames + b * numFrames + f] = stftLeft.imag[srcIdx];
      magSpec[2 * numBins * numFrames + b * numFrames + f] = stftRight.real[srcIdx];
      magSpec[3 * numBins * numFrames + b * numFrames + f] = stftRight.imag[srcIdx];
    }
  }

  const waveform = new Float32Array(2 * inputLength);
  waveform.set(paddedLeft, 0);
  waveform.set(paddedRight, inputLength);

  return { waveform, magSpec, numBins, numFrames, originalLength: leftChannel.length };
}

/**
 * Main Demucs processor class
 */
export class DemucsProcessor {
  constructor(options = {}) {
    this.ort = options.ort || null;
    this.session = null;
    this.modelPath = options.modelPath || './htdemucs_embedded.onnx';
    this.onProgress = options.onProgress || (() => {});
    this.onLog = options.onLog || (() => {});
    this.onDownloadProgress = options.onDownloadProgress || (() => {});
  }

  async loadModel(modelPathOrBuffer) {
    if (!this.ort) {
      throw new Error('ONNX Runtime not provided. Pass ort in constructor options.');
    }

    this.onLog('model', 'Loading model...');

    let modelBuffer;
    if (modelPathOrBuffer instanceof ArrayBuffer) {
      modelBuffer = modelPathOrBuffer;
    } else {
      const response = await fetch(modelPathOrBuffer || this.modelPath);

      // Check if we can track progress
      const contentLength = response.headers.get('Content-Length');
      if (contentLength && response.body) {
        const totalSize = parseInt(contentLength, 10);
        const reader = response.body.getReader();
        const chunks = [];
        let loadedSize = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          loadedSize += value.length;
          this.onDownloadProgress(loadedSize, totalSize);
        }

        // Combine chunks into single ArrayBuffer
        const combined = new Uint8Array(loadedSize);
        let offset = 0;
        for (const chunk of chunks) {
          combined.set(chunk, offset);
          offset += chunk.length;
        }
        modelBuffer = combined.buffer;
      } else {
        // Fallback: no progress tracking
        modelBuffer = await response.arrayBuffer();
      }
    }

    this.session = await this.ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'basic'
    });

    this.onLog('model', 'Model loaded successfully');
    return this.session;
  }

  async separate(leftChannel, rightChannel) {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    const totalSamples = leftChannel.length;
    const stride = Math.floor(TRAINING_SAMPLES * (1 - SEGMENT_OVERLAP));
    const numSegments = Math.ceil((totalSamples - TRAINING_SAMPLES) / stride) + 1;

    const outputs = TRACKS.map(() => ({
      left: new Float32Array(totalSamples),
      right: new Float32Array(totalSamples)
    }));
    const weights = new Float32Array(totalSamples);

    let segmentIdx = 0;

    for (let start = 0; start < totalSamples; start += stride) {
      const end = Math.min(start + TRAINING_SAMPLES, totalSamples);
      const segmentLength = end - start;

      const segLeft = new Float32Array(TRAINING_SAMPLES);
      const segRight = new Float32Array(TRAINING_SAMPLES);

      for (let i = 0; i < segmentLength; i++) {
        segLeft[i] = leftChannel[start + i];
        segRight[i] = rightChannel[start + i];
      }

      const input = prepareModelInput(segLeft, segRight);

      const waveformTensor = new this.ort.Tensor('float32', input.waveform, [1, 2, TRAINING_SAMPLES]);
      const magSpecTensor = new this.ort.Tensor('float32', input.magSpec, [1, 4, MODEL_SPEC_BINS, MODEL_SPEC_FRAMES]);

      const feeds = {};
      feeds[this.session.inputNames[0]] = waveformTensor;
      if (this.session.inputNames.length > 1) {
        feeds[this.session.inputNames[1]] = magSpecTensor;
      }

      const inferResults = await this.session.run(feeds);

      let timeData = null, timeShape = null;
      let freqData = null;

      for (const name of this.session.outputNames) {
        const tensor = inferResults[name];
        if (tensor.dims.length === 4 && tensor.dims[2] === 2) {
          timeData = tensor.data;
          timeShape = tensor.dims;
        } else if (tensor.dims.length === 5 && tensor.dims[2] === 4) {
          freqData = tensor.data;
        }
      }

      if (!timeData) {
        throw new Error('Could not find time-domain output tensor');
      }

      let combinedOutputs = null;
      if (freqData) {
        const trackSpecs = standaloneMask(freqData);
        combinedOutputs = [];

        for (let t = 0; t < 4; t++) {
          const freqOutput = standaloneIspec(trackSpecs[t], TRAINING_SAMPLES);
          const numChannels = timeShape[2];
          const samples = timeShape[3];
          const timeLeft = new Float32Array(samples);
          const timeRight = new Float32Array(samples);

          for (let i = 0; i < samples; i++) {
            timeLeft[i] = timeData[t * numChannels * samples + 0 * samples + i];
            timeRight[i] = timeData[t * numChannels * samples + 1 * samples + i];
          }

          const combined = {
            left: new Float32Array(samples),
            right: new Float32Array(samples)
          };
          for (let i = 0; i < samples; i++) {
            combined.left[i] = timeLeft[i] + (freqOutput.left[i] || 0);
            combined.right[i] = timeRight[i] + (freqOutput.right[i] || 0);
          }
          combinedOutputs.push(combined);
        }
      }

      const numTracks = timeShape[1];
      const numChannels = timeShape[2];
      const samples = timeShape[3];

      const overlapWindow = new Float32Array(segmentLength);
      for (let i = 0; i < segmentLength; i++) {
        const fadeIn = Math.min(i / (stride * 0.5), 1);
        const fadeOut = Math.min((segmentLength - i) / (stride * 0.5), 1);
        overlapWindow[i] = Math.min(fadeIn, fadeOut);
      }

      for (let t = 0; t < numTracks; t++) {
        for (let i = 0; i < segmentLength && start + i < totalSamples; i++) {
          let leftVal, rightVal;
          if (combinedOutputs) {
            leftVal = combinedOutputs[t].left[i];
            rightVal = combinedOutputs[t].right[i];
          } else {
            const leftIdx = t * numChannels * samples + 0 * samples + i;
            const rightIdx = t * numChannels * samples + 1 * samples + i;
            leftVal = timeData[leftIdx];
            rightVal = timeData[rightIdx];
          }
          outputs[t].left[start + i] += leftVal * overlapWindow[i];
          outputs[t].right[start + i] += rightVal * overlapWindow[i];
        }
      }

      for (let i = 0; i < segmentLength && start + i < totalSamples; i++) {
        weights[start + i] += overlapWindow[i];
      }

      segmentIdx++;
      this.onProgress({
        progress: segmentIdx / numSegments,
        currentSegment: segmentIdx,
        totalSegments: numSegments
      });
    }

    for (let t = 0; t < TRACKS.length; t++) {
      for (let i = 0; i < totalSamples; i++) {
        if (weights[i] > 0) {
          outputs[t].left[i] /= weights[i];
          outputs[t].right[i] /= weights[i];
        }
      }
    }

    return {
      drums: outputs[0],
      bass: outputs[1],
      other: outputs[2],
      vocals: outputs[3]
    };
  }
}
