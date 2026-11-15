
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from speechbrain.inference.classifiers import EncoderClassifier


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "audiofiles" / "processed"
SEPARATED_DIR = AUDIO_DIR / "separated"

PROFILE_DIR = BASE_DIR / "speaker_profiles"
MODEL_DIR = BASE_DIR / "models" / "ecapa"

TARGET_RATE = 16000
WINDOW_SECONDS = 3
STEP_SECONDS = 1.5
MIN_RMS = 0.003

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Original Mikolaj validation recordings
TEST_FILES = {
    "Original Mikolaj 4": AUDIO_DIR / "mikolaj_4_processed.wav",
    "Original Mikolaj 5": AUDIO_DIR / "mikolaj_5_processed.wav",
    "Original Mikolaj 6": AUDIO_DIR / "mikolaj_6_processed.wav",

    # Separated recordings
    "Separated Mercedes": SEPARATED_DIR / "speaker_1.wav",
    "Separated Mikolaj": SEPARATED_DIR / "speaker_2.wav",
    "Separated Prema": SEPARATED_DIR / "speaker_3.wav",
}

PROFILE_FILES = {
    "Mikolaj": PROFILE_DIR / "mikolaj.pt",
    "Mercedes": PROFILE_DIR / "mercedes.pt",
}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print(f"Device: {DEVICE}")
print("Loading ECAPA model...")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR),
    run_opts={"device": DEVICE}
)

print("ECAPA loaded.")


# --------------------------------------------------
# LOAD PROFILES
# --------------------------------------------------

profiles = {}

for name, path in PROFILE_FILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Missing profile: {path}")

    profile = torch.load(
        path,
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

    profiles[name] = F.normalize(profile, p=2, dim=0)

print("Speaker profiles loaded.")


# --------------------------------------------------
# LOAD AND PREPROCESS AUDIO
# --------------------------------------------------

def load_audio(path):

    if not path.exists():
        raise FileNotFoundError(f"Missing audio: {path}")

    audio, sample_rate = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != TARGET_RATE:
        audio = torchaudio.functional.resample(
            torch.tensor(audio),
            orig_freq=sample_rate,
            new_freq=TARGET_RATE
        ).numpy()

    return audio


# --------------------------------------------------
# EXTRACT EMBEDDING
# --------------------------------------------------

def extract_embedding(audio):

    waveform = torch.tensor(
        audio,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_batch(waveform)

    embedding = embedding.reshape(-1)

    return F.normalize(embedding, p=2, dim=0)


# --------------------------------------------------
# PROCESS AUDIO IN OVERLAPPING WINDOWS
# --------------------------------------------------

def analyse_file(name, path):

    audio = load_audio(path)

    window_size = WINDOW_SECONDS * TARGET_RATE
    step_size = int(STEP_SECONDS * TARGET_RATE)

    scores = {
        "Mikolaj": [],
        "Mercedes": []
    }

    usable_windows = 0

    for start in range(0, len(audio) - window_size + 1, step_size):

        segment = audio[start:start + window_size]

        rms = np.sqrt(np.mean(segment ** 2))

        if rms < MIN_RMS:
            continue

        embedding = extract_embedding(segment)
        usable_windows += 1

        for speaker, profile in profiles.items():

            similarity = F.cosine_similarity(
                embedding.unsqueeze(0),
                profile.unsqueeze(0)
            ).item()

            scores[speaker].append(similarity)

    if usable_windows == 0:
        print(f"\n{name}: No usable audio windows.")
        return

    print(f"\n{name}")
    print("-" * 55)
    print(f"Usable windows: {usable_windows}")

    for speaker in scores:

        values = scores[speaker]

        print(
            f"{speaker:<12} "
            f"Mean: {np.mean(values):.4f} | "
            f"Min: {np.min(values):.4f} | "
            f"Max: {np.max(values):.4f}"
        )


# --------------------------------------------------
# RUN COMPARISON
# --------------------------------------------------

print("\n" + "=" * 65)
print("ECAPA BASELINE VS SEPARATED AUDIO")
print("=" * 65)

for name, path in TEST_FILES.items():
    analyse_file(name, path)

print("\n" + "=" * 65)
print("COMPARISON COMPLETE")
print("=" * 65)
