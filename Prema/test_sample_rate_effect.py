
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from speechbrain.inference.classifiers import EncoderClassifier

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "audiofiles" / "processed"
SEPARATED_DIR = AUDIO_DIR / "separated"
MODEL_DIR = BASE_DIR / "models" / "ecapa"
PROFILE_DIR = BASE_DIR / "speaker_profiles"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_RATE = 16000
WINDOW_SECONDS = 3
STEP_SECONDS = 1.5
MIN_RMS = 0.003

print("Loading ECAPA...")
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR),
    run_opts={"device": DEVICE}
)

profile = torch.load(
    PROFILE_DIR / "mikolaj.pt",
    map_location=DEVICE,
    weights_only=False
)

if isinstance(profile, dict):
    for key in ("embedding", "profile", "speaker_embedding"):
        if key in profile:
            profile = profile[key]
            break

profile = torch.as_tensor(
    profile,
    dtype=torch.float32,
    device=DEVICE
).reshape(-1)

profile = F.normalize(profile, p=2, dim=0)


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != TARGET_RATE:
        audio = torchaudio.functional.resample(
            torch.tensor(audio),
            sr,
            TARGET_RATE
        ).numpy()

    return audio


def score_audio(audio):
    window = WINDOW_SECONDS * TARGET_RATE
    step = int(STEP_SECONDS * TARGET_RATE)

    scores = []

    for start in range(0, len(audio) - window + 1, step):
        segment = audio[start:start + window]

        if np.sqrt(np.mean(segment ** 2)) < MIN_RMS:
            continue

        waveform = torch.tensor(
            segment,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            embedding = model.encode_batch(waveform).reshape(-1)

        embedding = F.normalize(embedding, p=2, dim=0)

        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            profile.unsqueeze(0)
        ).item()

        scores.append(similarity)

    if not scores:
        return None

    return np.mean(scores), np.min(scores), np.max(scores), len(scores)


def report(name, audio):
    result = score_audio(audio)

    print(f"\n{name}")

    if result is None:
        print("No usable windows.")
        return

    mean, minimum, maximum, count = result

    print(f"Windows: {count}")
    print(f"Mean:    {mean:.4f}")
    print(f"Minimum: {minimum:.4f}")
    print(f"Maximum: {maximum:.4f}")


# Original recording
original_path = AUDIO_DIR / "mikolaj_4_processed.wav"
original = load_audio(original_path)

# Simulate SepFormer's 8 kHz bandwidth
downsampled = torchaudio.functional.resample(
    torch.tensor(original),
    orig_freq=16000,
    new_freq=8000
)

upsampled = torchaudio.functional.resample(
    downsampled,
    orig_freq=8000,
    new_freq=16000
).numpy()

# Actual separated Mikolaj
separated = load_audio(
    SEPARATED_DIR / "speaker_2.wav"
)

print("\n" + "=" * 55)
print("SAMPLE RATE AND SEPARATION COMPARISON")
print("=" * 55)

report("Original Mikolaj 4 (16 kHz)", original)
report("Original Mikolaj 4 (8 kHz -> 16 kHz)", upsampled)
report("SepFormer Mikolaj (8 kHz -> 16 kHz)", separated)

print("\nComparison complete.")
