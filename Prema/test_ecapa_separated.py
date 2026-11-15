
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from speechbrain.inference.classifiers import EncoderClassifier


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models" / "ecapa"
PROFILE_DIR = BASE_DIR / "speaker_profiles"

SEPARATED_DIR = (
    BASE_DIR / "audiofiles" / "processed" / "separated"
)

SPEAKER_FILES = {
    "speaker_1": SEPARATED_DIR / "speaker_1.wav",
    "speaker_2": SEPARATED_DIR / "speaker_2.wav",
    "speaker_3": SEPARATED_DIR / "speaker_3.wav",
}

PROFILES = {
    "Mikolaj": PROFILE_DIR / "mikolaj.pt",
    "Mercedes": PROFILE_DIR / "mercedes.pt",
}

SAMPLE_RATE = 16000

SIMILARITY_THRESHOLD = 0.30
MIN_SCORE_MARGIN = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# LOAD ECAPA MODEL
# --------------------------------------------------

print(f"Using device: {DEVICE}")
print("Loading ECAPA model...")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR),
    run_opts={"device": DEVICE}
)

print("ECAPA model loaded.")


# --------------------------------------------------
# LOAD ENROLLED SPEAKER PROFILES
# --------------------------------------------------

profiles = {}

for name, path in PROFILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Missing profile: {path}")

    profile = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    # Support profiles saved directly as tensors
    # or inside a dictionary.
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

    profiles[name] = profile

    print(f"Loaded {name} profile: {profile.numel()} dimensions")


# --------------------------------------------------
# EXTRACT ECAPA EMBEDDING
# --------------------------------------------------

def extract_embedding(audio_path):

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        import torchaudio

        audio = torchaudio.functional.resample(
            torch.tensor(audio),
            orig_freq=sample_rate,
            new_freq=SAMPLE_RATE
        ).numpy()

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
# IDENTIFY EACH SEPARATED SPEAKER
# --------------------------------------------------

print("\n" + "=" * 60)
print("ECAPA SEPARATED SPEAKER IDENTIFICATION")
print("=" * 60)

results = {}

for speaker, path in SPEAKER_FILES.items():

    print(f"\nTesting {speaker}...")

    if not path.exists():
        raise FileNotFoundError(f"Missing audio: {path}")

    embedding = extract_embedding(path)

    scores = {}

    for name, profile in profiles.items():

        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            profile.unsqueeze(0)
        ).item()

        scores[name] = similarity

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_name, best_score = ranked[0]
    second_score = ranked[1][1]

    margin = best_score - second_score

    if (
        best_score >= SIMILARITY_THRESHOLD
        and margin >= MIN_SCORE_MARGIN
    ):
        prediction = best_name
    else:
        prediction = "Unknown"

    results[speaker] = prediction

    print(f"  Mikolaj similarity:  {scores['Mikolaj']:.4f}")
    print(f"  Mercedes similarity: {scores['Mercedes']:.4f}")
    print(f"  Best match:          {best_name}")
    print(f"  Score margin:        {margin:.4f}")
    print(f"  Final prediction:    {prediction}")


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL IDENTIFICATION RESULTS")
print("=" * 60)

ground_truth = {
    "speaker_1": "Mercedes",
    "speaker_2": "Mikolaj",
    "speaker_3": "Prema",
}

for speaker, prediction in results.items():

    actual = ground_truth[speaker]

    print(
        f"{speaker}: "
        f"Predicted={prediction:<10} "
        f"Actual={actual}"
    )

    if prediction == actual:
        print("  Correct prediction")
    else:
        print("  Incorrect prediction")

print("\nTesting complete.")
