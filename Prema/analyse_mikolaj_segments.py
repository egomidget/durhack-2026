
from pathlib import Path
import csv

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

AUDIO_FILE = (
    BASE_DIR
    / "audiofiles"
    / "processed"
    / "separated"
    / "speaker_2.wav"
)

PROFILE_FILE = BASE_DIR / "speaker_profiles" / "mikolaj.pt"
MODEL_DIR = BASE_DIR / "models" / "ecapa"

OUTPUT_FILE = BASE_DIR / "audiofiles" / "processed" / "mikolaj_segment_scores.csv"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 3
STEP_SECONDS = 1.5

MIN_RMS = 0.003
SIMILARITY_THRESHOLD = 0.30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# LOAD ECAPA
# --------------------------------------------------

print(f"Using device: {DEVICE}")
print("Loading ECAPA model...")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR),
    run_opts={"device": DEVICE}
)

# --------------------------------------------------
# LOAD MIKOLAJ PROFILE
# --------------------------------------------------

profile = torch.load(
    PROFILE_FILE,
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

# --------------------------------------------------
# LOAD SEPARATED AUDIO
# --------------------------------------------------

audio, sample_rate = sf.read(
    AUDIO_FILE,
    dtype="float32"
)

if audio.ndim == 2:
    audio = audio.mean(axis=1)

if sample_rate != SAMPLE_RATE:
    audio = torchaudio.functional.resample(
        torch.tensor(audio),
        orig_freq=sample_rate,
        new_freq=SAMPLE_RATE
    ).numpy()

# --------------------------------------------------
# ANALYSE EACH WINDOW
# --------------------------------------------------

window_size = WINDOW_SECONDS * SAMPLE_RATE
step_size = int(STEP_SECONDS * SAMPLE_RATE)

results = []

print("\nAnalysing separated Mikolaj track...")

for start in range(0, len(audio) - window_size + 1, step_size):

    segment = audio[start:start + window_size]

    rms = float(np.sqrt(np.mean(segment ** 2)))

    if rms < MIN_RMS:
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

    start_time = start / SAMPLE_RATE
    end_time = (start + window_size) / SAMPLE_RATE

    results.append({
        "start": start_time,
        "end": end_time,
        "rms": rms,
        "similarity": similarity,
        "above_threshold": similarity >= SIMILARITY_THRESHOLD
    })

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if not results:
    raise RuntimeError("No usable audio windows found.")

results.sort(
    key=lambda item: item["similarity"],
    reverse=True
)

scores = [item["similarity"] for item in results]

above_threshold = [
    item for item in results
    if item["above_threshold"]
]

print("\n" + "=" * 65)
print("MIKOLAJ SEGMENT ANALYSIS")
print("=" * 65)

print(f"Total usable windows: {len(results)}")
print(f"Mean similarity: {np.mean(scores):.4f}")
print(f"Highest similarity: {max(scores):.4f}")
print(f"Lowest similarity: {min(scores):.4f}")
print(f"Windows above 0.30: {len(above_threshold)}")
print(
    f"Percentage above threshold: "
    f"{len(above_threshold) / len(results) * 100:.1f}%"
)

print("\nTOP 15 MATCHING WINDOWS")
print("-" * 65)

for item in results[:15]:

    print(
        f"{item['start']:6.2f}s - "
        f"{item['end']:6.2f}s | "
        f"Similarity: {item['similarity']:.4f} | "
        f"RMS: {item['rms']:.4f}"
    )

# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "start",
            "end",
            "rms",
            "similarity",
            "above_threshold"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nFull results saved to: {OUTPUT_FILE}")
print("\nAnalysis complete.")