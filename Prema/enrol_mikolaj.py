
import torch
import torch.nn.functional as F
import soundfile as sf

from pathlib import Path
from shutil import copy2
from datetime import datetime

from speechbrain.inference.classifiers import EncoderClassifier


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "audiofiles" / "processed"
PROFILE_DIR = BASE_DIR / "speaker_profiles"
MODEL_DIR = BASE_DIR / "models"

PROFILE_PATH = PROFILE_DIR / "mikolaj.pt"
BACKUP_DIR = PROFILE_DIR / "backups"

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RECORDINGS = [
    "mikolaj_1_processed.wav",
    "mikolaj_2_processed.wav",
    "mikolaj_3_processed.wav"
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
# LOAD AND VALIDATE RECORDING
# ==========================================

def load_recording(audio_path):

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Recording not found: {audio_path}"
        )

    audio_data, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True
    )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{audio_path.name} has sample rate "
            f"{sample_rate} Hz. Expected {SAMPLE_RATE} Hz. "
            "Preprocess this recording first."
        )

    # Convert stereo to mono
    if audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    else:
        audio_data = audio_data[:, 0]

    if len(audio_data) == 0:
        raise ValueError(
            f"{audio_path.name} is empty."
        )

    waveform = torch.from_numpy(audio_data.copy())

    duration = len(waveform) / SAMPLE_RATE

    rms = torch.sqrt(
        torch.mean(waveform ** 2)
    ).item()

    peak = waveform.abs().max().item()

    if duration < 3:
        raise ValueError(
            f"{audio_path.name} is too short "
            f"({duration:.2f}s)."
        )

    if rms < 0.003:
        raise ValueError(
            f"{audio_path.name} is too quiet "
            f"(RMS: {rms:.4f})."
        )

    print(f"\nRecording: {audio_path.name}")
    print(f"Duration: {duration:.2f}s")
    print(f"RMS: {rms:.4f}")
    print(f"Peak amplitude: {peak:.4f}")

    if peak >= 0.999:
        print("WARNING: Recording may contain clipping.")

    return waveform


# ==========================================
# EXTRACT ECAPA EMBEDDING
# ==========================================

def extract_embedding(waveform):

    waveform = waveform.unsqueeze(0).to(DEVICE)

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
# ENROL MIKOLAJ
# ==========================================

def enrol_mikolaj():

    print("\n" + "=" * 50)
    print("MIKOLAJ MULTI-RECORDING ENROLMENT")
    print("=" * 50)

    embeddings = []

    # Process all three enrolment recordings
    for filename in RECORDINGS:

        audio_path = PROCESSED_DIR / filename

        print("\n" + "-" * 40)

        waveform = load_recording(audio_path)

        embedding = extract_embedding(waveform)

        embeddings.append(embedding)

        print(
            f"Embedding extracted: "
            f"{embedding.shape[0]} dimensions"
        )

    # ======================================
    # AVERAGE EMBEDDINGS
    # ======================================

    print("\nCombining speaker embeddings...")

    stacked_embeddings = torch.stack(embeddings)

    average_embedding = torch.mean(
        stacked_embeddings,
        dim=0
    )

    # Normalise the final average
    final_embedding = F.normalize(
        average_embedding,
        p=2,
        dim=0
    )

    print(
        f"Final embedding dimensions: "
        f"{final_embedding.shape[0]}"
    )

    print(
        f"Final embedding norm: "
        f"{torch.norm(final_embedding).item():.4f}"
    )

    # ======================================
    # BACK UP EXISTING PROFILE
    # ======================================

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    if PROFILE_PATH.exists():

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        backup_path = (
            BACKUP_DIR /
            f"mikolaj_backup_{timestamp}.pt"
        )

        copy2(PROFILE_PATH, backup_path)

        print(f"\nPrevious profile backed up to:")
        print(backup_path)

    # ======================================
    # SAVE NEW PROFILE
    # ======================================

    profile = {
        "speaker_name": "Mikolaj",
        "embedding": final_embedding,
        "sample_rate": SAMPLE_RATE,
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "enrolment_recordings": RECORDINGS,
        "num_recordings": len(RECORDINGS)
    }

    torch.save(profile, PROFILE_PATH)

    print("\n" + "=" * 50)
    print("ENROLMENT SUCCESSFUL")
    print("=" * 50)

    print(f"Speaker: Mikolaj")
    print(f"Recordings used: {len(embeddings)}")
    print(f"Embedding size: {final_embedding.shape[0]}")
    print(f"Profile saved to: {PROFILE_PATH}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    enrol_mikolaj()
