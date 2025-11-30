/**
 * Constants for Demucs model
 */
export const CONSTANTS = {
  SAMPLE_RATE: 44100,
  FFT_SIZE: 4096,
  HOP_SIZE: 1024,
  TRAINING_SAMPLES: 343980,
  MODEL_SPEC_BINS: 2048,
  MODEL_SPEC_FRAMES: 336,
  SEGMENT_OVERLAP: 0.25,
  TRACKS: ['drums', 'bass', 'other', 'vocals'],

  // Default model URL (Hugging Face Hub)
  DEFAULT_MODEL_URL: 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx'
};
