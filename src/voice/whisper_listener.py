"""Push-to-talk voice dictation via faster-whisper on CUDA (INT8)."""
import logging

import numpy as np

from config.settings import (
    SAMPLE_RATE,
    WHISPER_COMPUTE,
    WHISPER_DEVICE,
    WHISPER_LOCAL_DIR,
    WHISPER_MODEL,
)

log = logging.getLogger(__name__)


# Returned cookie objects MUST stay referenced — GC removes the dir otherwise.
_DLL_COOKIES = []


def _register_nvidia_dlls():
    """Make pip-installed nvidia-*/bin DLLs (cuBLAS/cuDNN) visible to ctranslate2."""
    import glob
    import os
    import sys

    for pattern in (
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin"),
        os.path.join(sys.prefix, "lib", "site-packages", "nvidia", "*", "bin"),
    ):
        for path in glob.glob(pattern):
            _DLL_COOKIES.append(os.add_dll_directory(path))
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


class VoiceDictator:
    def __init__(self, model_size: str = WHISPER_MODEL):
        _register_nvidia_dlls()
        from faster_whisper import WhisperModel

        # Prefer the vendored copy — loads fully offline, no HF download
        target = str(WHISPER_LOCAL_DIR) if WHISPER_LOCAL_DIR.exists() else model_size
        try:
            self.model = WhisperModel(
                target, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
            )
        except Exception as e:  # noqa: BLE001 — no GPU? fall back to CPU
            log.warning("CUDA whisper init failed (%s) — falling back to CPU int8.", e)
            self.model = WhisperModel(target, device="cpu", compute_type="int8")

    def record_and_transcribe(self, duration: float = 5.0, samplerate: int = SAMPLE_RATE) -> str:
        import sounddevice as sd

        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return self.transcribe(audio.flatten())

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, beam_size=1)
        return " ".join(segment.text for segment in segments).strip()

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Decode a WAV/audio buffer (e.g. st.chat_input mic upload) and transcribe."""
        return self.transcribe(_decode_audio_bytes(audio_bytes))


def _decode_audio_bytes(data: bytes) -> np.ndarray:
    """Decode browser-recorded audio to 16 kHz mono float32.

    Uses stdlib `wave` for PCM WAV (what st.chat_input produces) and falls
    back to PyAV for other containers — bypasses faster_whisper.audio.decode_audio,
    which is incompatible with PyAV 19 (removed `metadata_errors` kwarg).
    """
    import io
    import wave

    try:
        with wave.open(io.BytesIO(data)) as w:
            sr, channels, sampwidth = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sampwidth == 2:
            audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            audio = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported sample width {sampwidth}")
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
    except (wave.Error, EOFError, ValueError):
        import av

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        container = av.open(io.BytesIO(data), mode="r")
        chunks = [
            rf.to_ndarray().reshape(-1)
            for frame in container.decode(audio=0)
            for rf in resampler.resample(frame)
        ]
        container.close()
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    if sr != SAMPLE_RATE and len(audio):
        n = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype(np.float32)
