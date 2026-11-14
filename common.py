"""Shared helpers: paths, audio I/O (always 16 kHz mono), metrics."""
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000  # every model and file in this project uses 16 kHz mono


def data_path(*parts) -> Path:
    """Path inside the shared data folder (env var HUSH_DATA, default ./HUSH-data)."""
    root = Path(os.environ.get("HUSH_DATA", Path(__file__).resolve().parent / "HUSH-data"))
    return root.joinpath(*parts)


def load_wav(path) -> np.ndarray:
    """Load any WAV as float32 mono at 16 kHz."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def save_wav(path, x: np.ndarray) -> None:
    """Save float audio as 16 kHz mono 16-bit WAV (creates folders, clips to [-1, 1])."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), SR, subtype="PCM_16")


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (higher = better)."""
    n = min(len(estimate), len(reference))
    est, ref = estimate[:n] - estimate[:n].mean(), reference[:n] - reference[:n].mean()
    scale = np.dot(est, ref) / (np.dot(ref, ref) + 1e-9)
    target = scale * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + 1e-9) / (np.dot(noise, noise) + 1e-9)))


def leakage_db(target_after: np.ndarray, target_before: np.ndarray) -> float:
    """How loud the target speaker still is after removal, relative to before (dB). Lower = better.

    `target_after` = the target's part of the output: apply the same mask / subtraction
    to the clean target signal alone (possible because test mixes store target.wav).
    """
    return rms_db(target_after) - rms_db(target_before)
