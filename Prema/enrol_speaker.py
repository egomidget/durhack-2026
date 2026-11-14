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
print(f"Device: {DEVICE}")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR / "ecapa"),
    run_opts={"device": DEVICE}
)

print("ECAPA-TDNN loaded successfully!")


def enrol_speaker(speaker_name, audio_path):

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print(f"\nEnrolling {speaker_name}...")

    audio_data, sample_rate = sf.read(
        str(audio_path),
        dtype="float32"
    )

    if sample_rate != 16000:
        raise ValueError("Audio must be 16 kHz.")

    if audio_data.ndim != 1 or len(audio_data) == 0:
        raise ValueError("Expected non-empty mono audio.")

    waveform = torch.from_numpy(audio_data).unsqueeze(0)

    waveform = waveform.to(DEVICE)

    print(f"Duration: {len(audio_data) / sample_rate:.2f}s")

    with torch.inference_mode():

        embedding = model.encode_batch(waveform)

    embedding = embedding.squeeze()

    embedding = F.normalize(embedding, p=2, dim=0)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    profile_path = PROFILE_DIR / f"{speaker_name.lower()}.pt"

    torch.save(
        {
            "speaker_name": speaker_name,
            "embedding": embedding.cpu(),
            "sample_rate": sample_rate,
            "model": "speechbrain/spkrec-ecapa-voxceleb"
        },
        profile_path
    )

    print(f"Embedding dimensions: {embedding.shape[0]}")
    print(f"Profile saved: {profile_path}")

    return embedding


if __name__ == "__main__":

    print("\n========== SPEAKER ENROLMENT ==========")

    mikolaj_embedding = enrol_speaker(
        "Mikolaj",
        AUDIO_DIR / "mikolaj_processed.wav"
    )

    mercedes_embedding = enrol_speaker(
        "Mercedes",
        AUDIO_DIR / "mercedes_processed.wav"
    )

    similarity = F.cosine_similarity(
        mikolaj_embedding.unsqueeze(0),
        mercedes_embedding.unsqueeze(0)
    )

    print("\n========== RESULTS ==========")

    print("Mikolaj embedding:", mikolaj_embedding.shape)
    print("Mercedes embedding:", mercedes_embedding.shape)

    print(
        f"Cosine similarity: {similarity.item():.4f}"
    )

    print("\nBoth speakers enrolled successfully!")
