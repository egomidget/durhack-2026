
import torch
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path
from speechbrain.inference.classifiers import EncoderClassifier

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


print("Loading ECAPA-TDNN model...")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR / "ecapa"),
    run_opts={"device": DEVICE}
)

print("ECAPA-TDNN loaded successfully!")


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


def load_audio(audio_path):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio_data, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True
    )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            "Audio must be preprocessed to 16000 Hz first."
        )

    if audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    else:
        audio_data = audio_data[:, 0]

    if len(audio_data) == 0:
        raise ValueError("Audio recording is empty.")

    return torch.from_numpy(audio_data.copy())


def extract_segment_embedding(segment):
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



def identify_segment(segment, profiles):

    embedding = extract_segment_embedding(segment)

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

    score_margin = best_score - second_best_score

    if (
        best_score < SIMILARITY_THRESHOLD
        or score_margin < MIN_SCORE_MARGIN
    ):
        prediction = "Unknown"
    else:
        prediction = best_match

    return prediction, scores, score_margin


def identify_recording(audio_path):

    print("\n" + "=" * 40)
    print("SEGMENT-BASED SPEAKER IDENTIFICATION")
    print("=" * 40)

    profiles = load_speaker_profiles()

    waveform = load_audio(audio_path)

    total_duration = len(waveform) / SAMPLE_RATE

    segment_samples = int(
        SEGMENT_DURATION * SAMPLE_RATE
    )

    step_samples = int(
        STEP_DURATION * SAMPLE_RATE
    )

    print(f"\nRecording: {Path(audio_path).name}")
    print(f"Duration: {total_duration:.2f} seconds")
    print(f"Segment length: {SEGMENT_DURATION} seconds")
    print(f"Step length: {STEP_DURATION} seconds")

    print(f"\nSimilarity threshold: {SIMILARITY_THRESHOLD}")
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

            print(
                f"Segment too quiet (RMS: {rms:.4f}). Skipping."
            )

        else:

            prediction, scores, margin = identify_segment(
                segment,
                profiles
            )

            for speaker_name, score in scores.items():
                print(f"{speaker_name}: {score:.4f}")

            print(f"Score margin: {margin:.4f}")
            print(f"Predicted speaker: {prediction}")

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

    print("\n" + "=" * 40)
    print("IDENTIFICATION SUMMARY")
    print("=" * 40)

    speaker_counts = {}

    for result in results:

        prediction = result["prediction"]

        speaker_counts[prediction] = (
            speaker_counts.get(prediction, 0) + 1
        )

    total_segments = len(results)

    if total_segments == 0:
        print("No usable segments found.")
        return results

    for speaker, count in speaker_counts.items():

        percentage = (
            count / total_segments
        ) * 100

        print(
            f"{speaker}: {count} segments "
            f"({percentage:.1f}%)"
        )

    print(f"\nTotal usable segments: {total_segments}")

    return results


if __name__ == "__main__":

    test_recordings = [
        "merc_yap_processed.wav",
        "prema_yap_processed.wav"
    ]

    for recording in test_recordings:

        print("\n\n" + "#" * 50)
        print(f"TESTING: {recording}")
        print("#" * 50)

        test_audio = AUDIO_DIR / recording

        identify_recording(test_audio)
