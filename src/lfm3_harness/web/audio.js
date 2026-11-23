// Vanilla ES module for browser audio capture (16 kHz mono WAV), upload, and Tier-1 TTS playback.

export function canRecordInBrowser() {
  return Boolean(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function encodePcm16Wav(float32Samples, sampleRate) {
  const numSamples = float32Samples.length;
  const buf = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);
  let off = 44;
  for (let i = 0; i < numSamples; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, float32Samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

export async function blobTo16kMonoWav(blob) {
  const arrBuf = await blob.arrayBuffer();
  const decodeCtx = new AudioContext();
  const audioBuf = await decodeCtx.decodeAudioData(arrBuf);
  await decodeCtx.close();

  const targetRate = 16000;
  const outLen = Math.max(1, Math.ceil(audioBuf.duration * targetRate));
  const offCtx = new OfflineAudioContext(1, outLen, targetRate);
  const src = offCtx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(offCtx.destination);
  src.start(0);
  const rendered = await offCtx.startRendering();
  return encodePcm16Wav(rendered.getChannelData(0), targetRate);
}

export class MicRecorder {
  constructor() {
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
  }

  async start() {
    if (!canRecordInBrowser()) {
      throw new Error("Browser microphone requires HTTPS or localhost Secure Context; use +Audio file upload on plain LAN HTTP.");
    }
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.recorder = new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) this.chunks.push(ev.data);
    };
    this.recorder.start();
  }

  stop() {
    return new Promise((resolve, reject) => {
      if (!this.recorder) {
        reject(new Error("Recorder not started"));
        return;
      }
      this.recorder.onstop = async () => {
        try {
          const blob = new Blob(this.chunks, { type: this.recorder.mimeType || "audio/webm" });
          const wav = await blobTo16kMonoWav(blob);
          resolve(wav);
        } catch (err) {
          reject(err);
        } finally {
          if (this.stream) {
            this.stream.getTracks().forEach((t) => t.stop());
          }
          this.stream = null;
          this.recorder = null;
        }
      };
      this.recorder.stop();
    });
  }
}

export async function uploadAudioBlob(blob, filename) {
  const url = `/api/audio/upload?filename=${encodeURIComponent(filename || "clip.wav")}`;
  const resp = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": blob.type || "application/octet-stream" },
    body: blob,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Audio upload failed (${resp.status})`);
  }
  return await resp.json();
}

let activeAudioEl = null;

export async function speakText(text, opts = {}) {
  if (activeAudioEl) {
    activeAudioEl.pause();
    activeAudioEl = null;
  }
  const bodyObj = { text };
  if (opts.engine) bodyObj.engine = opts.engine;
  if (opts.language) bodyObj.language = opts.language;
  if (opts.gender) bodyObj.gender = opts.gender;
  const resp = await fetch("/api/tts", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `TTS failed (${resp.status})`);
  }
  const blob = await resp.blob();
  const objUrl = URL.createObjectURL(blob);
  const el = new Audio(objUrl);
  activeAudioEl = el;
  el.onended = () => URL.revokeObjectURL(objUrl);
  await el.play();
  return {
    engine: resp.headers.get("X-TTS-Engine") || "unknown",
    latencyMs: resp.headers.get("X-TTS-Latency-Ms") || "?",
    frames: resp.headers.get("X-TTS-Frames") || "",
    durationS: resp.headers.get("X-TTS-Duration-S") || "",
    rtf: resp.headers.get("X-TTS-RTF") || "",
    eos: resp.headers.get("X-TTS-EOS") || "",
  };
}
