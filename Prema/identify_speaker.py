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


print("Loading ECAPA-TDNN model...")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR / "ecapa"),
    run_opts={"device": DEVICE}
)

print("ECAPA-TDNN loaded successfully!")


# ==========================================
# LOAD SAVED SPEAKER PROFILES
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
        embedding = profile["embedding"]

        profiles[speaker_name] = F.normalize(
            embedding.float(),
            p=2,
            dim=0
        )

        print(f"Loaded profile: {speaker_name}")

    if not profiles:
        raise RuntimeError("No speaker profiles found.")

    return profiles


def extract_embedding(audio_path):

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

    if audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1, keepdims=True)

    if sample_rate != 16000:
        raise ValueError(
            "Audio must be preprocessed to 16000 Hz first."
        )

    if len(audio_data) == 0:
        raise ValueError("Audio recording is empty.")

    print(f"\nAnalysing: {audio_path.name}")
    print(f"Duration: {len(audio_data) / sample_rate:.2f}s")

    waveform = torch.from_numpy(audio_data.T).to(DEVICE)

    with torch.inference_mode():

        embedding = model.encode_batch(waveform)

    embedding = embedding.squeeze()

    embedding = F.normalize(
        embedding,
        p=2,
        dim=0
    )

    return embedding.cpu()


def identify_speaker(audio_path):

    print("\n========== SPEAKER IDENTIFICATION ==========")

    profiles = load_speaker_profiles()

    test_embedding = extract_embedding(audio_path)

    print("\nSimilarity scores")
    print("-----------------")

    scores = {}

    for speaker_name, profile_embedding in profiles.items():

        similarity = F.cosine_similarity(
            test_embedding.unsqueeze(0),
            profile_embedding.unsqueeze(0)
        )

        scores[speaker_name] = similarity.item()

        print(f"{speaker_name}: {similarity.item():.4f}")

    best_match = max(scores, key=scores.get)

    print("\n========== RESULT ==========")
    print(f"Closest matching speaker: {best_match}")
    print(f"Similarity: {scores[best_match]:.4f}")

    print(
        "\nNote: This is the closest match, "
        "not yet a verified identity."
    )

    return best_match, scores

if __name__ == "__main__":

    test_recordings = [
        "merc_yap_processed.wav",
        "prema_yap_processed.wav"
    ]

    for recording in test_recordings:

        print("\n\n")
        print("=" * 50)
        print(f"TESTING: {recording}")
        print("=" * 50)

        test_audio = AUDIO_DIR / recording

        identify_speaker(test_audio)
