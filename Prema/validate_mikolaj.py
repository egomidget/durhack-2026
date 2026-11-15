
import torch
import torch.nn.functional as F
import soundfile as sf

from pathlib import Path
from speechbrain.inference.classifiers import EncoderClassifier


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "audiofiles" / "processed"
PROFILE_DIR = BASE_DIR / "speaker_profiles"
MODEL_DIR = BASE_DIR / "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
SEGMENT_DURATION = 3.0
STEP_DURATION = 1.5
MIN_RMS = 0.003

SIMILARITY_THRESHOLD = 0.30
MIN_SCORE_MARGIN = 0.05

VALIDATION_RECORDINGS = [
    "mikolaj_4_processed.wav",
    "mikolaj_5_processed.wav",
    "mikolaj_6_processed.wav"
]


# ==========================================
# LOAD ECAPA-TDNN
# ==========================================

print("Loading ECAPA-TDNN model...")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR / "ecapa"),
    run_opts={"device": DEVICE}
)

print("ECAPA-TDNN loaded successfully!")


# ==========================================
# LOAD SPEAKER PROFILES
# ==========================================

def load_speaker_profiles():

    profiles = {}

    for profile_path in PROFILE_DIR.glob("*.pt"):

        profile = torch.load(
            profile_path,
            map_location="cpu",
            weights_only=True
        )

        speaker_name = profile["speaker_name"]

        embedding = F.normalize(
            profile["embedding"].float(),
            p=2,
            dim=0
        )

        profiles[speaker_name] = embedding

        print(f"Loaded profile: {speaker_name}")

    if not profiles:
        raise RuntimeError("No speaker profiles found.")

    return profiles


# ==========================================
# LOAD AUDIO
# ==========================================

def load_audio(audio_path):

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    audio_data, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True
    )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{audio_path.name} must be 16000 Hz."
        )

    if audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    else:
        audio_data = audio_data[:, 0]

    if len(audio_data) == 0:
        raise ValueError("Audio recording is empty.")

    return torch.from_numpy(audio_data.copy())


# ==========================================
# EXTRACT EMBEDDING
# ==========================================

def extract_embedding(segment):

    waveform = segment.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        embedding = model.encode_batch(waveform)

    embedding = embedding.squeeze()

    embedding = F.normalize(
        embedding,
        p=2,
        dim=0
    )

    return embedding.cpu()


# ==========================================
# IDENTIFY SEGMENT
# ==========================================

def identify_segment(segment, profiles):

    embedding = extract_embedding(segment)

    scores = {}

    for speaker_name, profile_embedding in profiles.items():

        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            profile_embedding.unsqueeze(0)
        )

        scores[speaker_name] = similarity.item()

    ranked_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_match, best_score = ranked_scores[0]

    second_best_score = (
        ranked_scores[1][1]
        if len(ranked_scores) > 1
        else 0.0
    )

    margin = best_score - second_best_score

    if (
        best_score < SIMILARITY_THRESHOLD
        or margin < MIN_SCORE_MARGIN
    ):
        prediction = "Unknown"
    else:
        prediction = best_match

    return prediction, scores, margin


# ==========================================
# VALIDATE ONE RECORDING
# ==========================================

def validate_recording(audio_path, profiles):

    print("\n" + "=" * 50)
    print(f"VALIDATING: {audio_path.name}")
    print("=" * 50)

    waveform = load_audio(audio_path)

    duration = len(waveform) / SAMPLE_RATE

    segment_samples = int(
        SEGMENT_DURATION * SAMPLE_RATE
    )

    step_samples = int(
        STEP_DURATION * SAMPLE_RATE
    )

    print(f"Duration: {duration:.2f} seconds")

    results = []

    start = 0
    segment_number = 1

    while start + segment_samples <= len(waveform):

        end = start + segment_samples

        segment = waveform[start:end]

        start_time = start / SAMPLE_RATE
        end_time = end / SAMPLE_RATE

        rms = torch.sqrt(
            torch.mean(segment ** 2)
        ).item()

        if rms < MIN_RMS:

            print(
                f"\nSegment {segment_number}: "
                f"{start_time:.1f}-{end_time:.1f}s "
                f"- Too quiet, skipped"
            )

        else:

            prediction, scores, margin = identify_segment(
                segment,
                profiles
            )

            print(
                f"\nSegment {segment_number}: "
                f"{start_time:.1f}-{end_time:.1f}s"
            )

            for speaker_name, score in scores.items():
                print(f"{speaker_name}: {score:.4f}")

            print(f"Margin: {margin:.4f}")
            print(f"Prediction: {prediction}")

            results.append(prediction)

        start += step_samples
        segment_number += 1

    # ======================================
    # RECORDING SUMMARY
    # ======================================

    print("\n" + "-" * 40)
    print("RECORDING SUMMARY")
    print("-" * 40)

    counts = {}

    for prediction in results:
        counts[prediction] = counts.get(prediction, 0) + 1

    total = len(results)

    if total == 0:
        print("No usable segments.")
        return results

    for speaker, count in counts.items():

        percentage = count / total * 100

        print(
            f"{speaker}: {count} segments "
            f"({percentage:.1f}%)"
        )

    print(f"Total usable segments: {total}")

    return results


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\nLoading speaker profiles...")

    profiles = load_speaker_profiles()

    print("\nStarting independent validation...")

    all_results = {}

    for filename in VALIDATION_RECORDINGS:

        audio_path = AUDIO_DIR / filename

        results = validate_recording(
            audio_path,
            profiles
        )

        all_results[filename] = results

    # ======================================
    # OVERALL VALIDATION SUMMARY
    # ======================================

    print("\n\n" + "#" * 50)
    print("OVERALL VALIDATION RESULTS")
    print("#" * 50)

    total_segments = 0
    total_mikolaj = 0
    total_mercedes = 0
    total_unknown = 0

    for filename, results in all_results.items():

        mikolaj = results.count("Mikolaj")
        mercedes = results.count("Mercedes")
        unknown = results.count("Unknown")

        total = len(results)

        total_segments += total
        total_mikolaj += mikolaj
        total_mercedes += mercedes
        total_unknown += unknown

        print(f"\n{filename}")
        print(f"Mikolaj: {mikolaj}")
        print(f"Mercedes: {mercedes}")
        print(f"Unknown: {unknown}")

    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    print("=" * 40)

    print(f"Total segments: {total_segments}")
    print(f"Mikolaj: {total_mikolaj}")
    print(f"Mercedes: {total_mercedes}")
    print(f"Unknown: {total_unknown}")

    if total_segments > 0:

        recognition_rate = (
            total_mikolaj / total_segments
        ) * 100

        print(
            f"\nMikolaj identification rate: "
            f"{recognition_rate:.1f}%"
        )
