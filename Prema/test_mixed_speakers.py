
import torch
import torch.nn.functional as F
import soundfile as sf

from pathlib import Path
from speechbrain.inference.classifiers import EncoderClassifier


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_PATH = (
    BASE_DIR / "audiofiles" / "processed"
    / "mixed_talking_processed.wav"
)

PROFILE_DIR = BASE_DIR / "speaker_profiles"
MODEL_DIR = BASE_DIR / "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000

SEGMENT_DURATION = 3.0
STEP_DURATION = 1.5

MIN_RMS = 0.003

SIMILARITY_THRESHOLD = 0.30
MIN_SCORE_MARGIN = 0.05


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
# LOAD MIXED RECORDING
# ==========================================

def load_audio():

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Recording not found: {AUDIO_PATH}"
        )

    audio_data, sample_rate = sf.read(
        str(AUDIO_PATH),
        dtype="float32",
        always_2d=True
    )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected 16000 Hz, got {sample_rate} Hz."
        )

    if audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    else:
        audio_data = audio_data[:, 0]

    if len(audio_data) == 0:
        raise ValueError("Recording is empty.")

    waveform = torch.from_numpy(audio_data.copy())

    duration = len(waveform) / SAMPLE_RATE

    print("\nMIXED RECORDING DETAILS")
    print("-" * 40)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: Mono")

    return waveform


# ==========================================
# EXTRACT SPEAKER EMBEDDING
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
# IDENTIFY SPEAKER
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

    if len(ranked_scores) > 1:
        second_best_score = ranked_scores[1][1]
    else:
        second_best_score = 0.0

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
# ANALYSE MIXED RECORDING
# ==========================================

def analyse_mixed_recording():

    print("\n" + "=" * 50)
    print("MIXED SPEAKER IDENTIFICATION EXPERIMENT")
    print("=" * 50)

    profiles = load_speaker_profiles()

    waveform = load_audio()

    segment_samples = int(
        SEGMENT_DURATION * SAMPLE_RATE
    )

    step_samples = int(
        STEP_DURATION * SAMPLE_RATE
    )

    total_duration = len(waveform) / SAMPLE_RATE

    print(f"\nSegment duration: {SEGMENT_DURATION}s")
    print(f"Step duration: {STEP_DURATION}s")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Minimum score margin: {MIN_SCORE_MARGIN}")

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

        print(
            f"\nSegment {segment_number}: "
            f"{start_time:.1f}-{end_time:.1f}s"
        )

        if rms < MIN_RMS:

            print("Too quiet - skipped")

        else:

            prediction, scores, margin = identify_segment(
                segment,
                profiles
            )

            for speaker_name, score in scores.items():

                print(f"{speaker_name}: {score:.4f}")

            print(f"Score margin: {margin:.4f}")
            print(f"Prediction: {prediction}")

            results.append({
                "segment": segment_number,
                "start": start_time,
                "end": end_time,
                "prediction": prediction,
                "scores": scores,
                "margin": margin
            })

        start += step_samples
        segment_number += 1

    # ======================================
    # OVERALL SUMMARY
    # ======================================

    print("\n" + "=" * 50)
    print("MIXED RECORDING SUMMARY")
    print("=" * 50)

    counts = {}

    for result in results:

        prediction = result["prediction"]

        counts[prediction] = (
            counts.get(prediction, 0) + 1
        )

    total_segments = len(results)

    if total_segments == 0:
        print("No usable segments found.")
        return results

    for speaker, count in counts.items():

        percentage = count / total_segments * 100

        print(
            f"{speaker}: {count} segments "
            f"({percentage:.1f}%)"
        )

    print(f"\nTotal usable segments: {total_segments}")

    # Average similarity scores
    print("\nAVERAGE SIMILARITY SCORES")
    print("-" * 40)

    for speaker_name in profiles:

        speaker_scores = [
            result["scores"][speaker_name]
            for result in results
        ]

        average_score = (
            sum(speaker_scores) / len(speaker_scores)
        )

        print(f"{speaker_name}: {average_score:.4f}")

    # Average Mikolaj score over the timeline
    if "Mikolaj" in profiles:

        mikolaj_scores = [
            result["scores"]["Mikolaj"]
            for result in results
        ]

        print(
            f"\nHighest Mikolaj similarity: "
            f"{max(mikolaj_scores):.4f}"
        )

        print(
            f"Lowest Mikolaj similarity: "
            f"{min(mikolaj_scores):.4f}"
        )

    return results


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    analyse_mixed_recording()
