from pathlib import Path
import torch
import torchaudio
import soundfile as sf
from speechbrain.inference.separation import SepformerSeparation

# Project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "audiofiles" / "processed" / "mixed_talking_processed.wav"
OUTPUT_DIR = BASE_DIR / "audiofiles" / "processed" / "separated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading SepFormer model...")

# Load pretrained three-speaker separation model
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-libri3mix",
    savedir=str(BASE_DIR / "models" / "sepformer-libri3mix"),
    run_opts={"device": DEVICE}
)

print("Loading mixed recording...")

audio, sample_rate = sf.read(INPUT_FILE, dtype="float32")

# Ensure mono
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Resample to 8 kHz
if sample_rate != 8000:
    audio = torchaudio.functional.resample(
        torch.tensor(audio),
        orig_freq=sample_rate,
        new_freq=8000
    ).numpy()

# Shape: [batch, time]
mixture = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

print("Separating speakers...")

with torch.no_grad():
    separated = separator.separate_batch(mixture)

# Expected shape: [batch, time, speakers]
separated = separated.squeeze(0).cpu().numpy()

print(f"Separation complete. Output shape: {separated.shape}")

# Save each separated speaker
for i in range(separated.shape[-1]):
    output_file = OUTPUT_DIR / f"speaker_{i + 1}.wav"

    sf.write(
        output_file,
        separated[:, i],
        8000
    )

    print(f"Saved: {output_file}")

print("\nAll speaker tracks saved.")
