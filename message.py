#!/usr/bin/env python3
"""
Record a short message from the mic, remove Mikolaj's voice from it (via
Prema/remove_mikolaj_16k.py - SepFormer + ECAPA), and play back the result.

  python message.py [--seconds 12] [--in DEVICE] [--out DEVICE]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

import common

HERE = Path(__file__).resolve().parent
MESSAGES_DIR = HERE / "HUSH-data" / "messages"


def dev(x):
    """sounddevice treats string device IDs as name substrings, not indices -
    convert numeric strings to int so '0' means device 0, not 'name contains 0'."""
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        return x


def record(seconds, device=None):
    device = dev(device)
    info = sd.query_devices(device, "input")
    native_sr = int(info["default_samplerate"])
    print(f"Recording {seconds:.0f}s - talk naturally. Starting now!")
    rec = sd.rec(int(seconds * native_sr), samplerate=native_sr, channels=1,
                device=device, dtype="float32")
    for s in range(int(seconds), 0, -1):
        print(f"  {s:2d}s left", end="\r", flush=True)
        time.sleep(1)
    sd.wait()
    print()
    wav = rec[:, 0]
    if native_sr != common.SR:
        g = np.gcd(native_sr, common.SR)
        wav = resample_poly(wav, common.SR // g, native_sr // g).astype(np.float32)
    level = common.rms_db(wav)
    if level < -50:
        print(f"Warning: that was very quiet ({level:.0f} dBFS) - check the mic and try again.")
    return wav


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--in", dest="inp", default=None, help="input device number or name")
    p.add_argument("--out", dest="outp", default=None, help="output device number or name")
    a = p.parse_args()

    wav = record(a.seconds, device=a.inp)
    MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mix_path = MESSAGES_DIR / f"message_{stamp}.wav"
    common.save_wav(mix_path, wav)
    print(f"Saved {mix_path}")

    print("\nRemoving Mikolaj's voice (this takes a while on CPU)...")
    removed_path = MESSAGES_DIR / f"message_{stamp}_mikolaj_removed.wav"
    subprocess.run([
        sys.executable, str(HERE / "Prema" / "remove_mikolaj_16k.py"),
        "--mix", str(mix_path), "--out-dir", str(MESSAGES_DIR),
    ], check=True, cwd=str(HERE / "Prema"))
    expected = MESSAGES_DIR / f"message_{stamp}_mikolaj_removed.wav"

    print(f"\nPlaying back: {expected.name}")
    out_wav = common.load_wav(expected)
    sd.play(out_wav, common.SR, device=dev(a.outp))
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    main()
