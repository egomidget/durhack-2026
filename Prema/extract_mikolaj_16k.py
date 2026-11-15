"""
Improved version of extract_mikolaj.py.

extract_mikolaj.py used speechbrain/sepformer-libri3mix, which:
  - runs at 8 kHz, so output is permanently missing everything above 4 kHz
    (upsampling back to 16 kHz afterwards can't recover that detail - this is
    why the separated audio sounded degraded)
  - is a 3-speaker model trained on clean synthetic mixes, but our real
    recordings are 2 people talking in an actual room (reverb, background noise)

This swaps to speechbrain/sepformer-whamr16k: native 16 kHz, 2-speaker, and
trained on WHAMR (noisy + reverberant mixes) - a much closer match to a real
room recording. No resampling needed; everything stays at the project's
standard 16 kHz throughout.
"""
from pathlib import Path
import torch
import soundfile as sf
from speechbrain.inference.separation import SepformerSeparation

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "audiofiles" / "processed" / "mixed_talking_processed.wav"
OUTPUT_DIR = BASE_DIR / "audiofiles" / "processed" / "separated_16k"
SR = 16000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print("Loading SepFormer (whamr16k, 16 kHz, 2-speaker) model...")

separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-whamr16k",
    savedir=str(BASE_DIR / "models" / "sepformer-whamr16k"),
    run_opts={"device": DEVICE}
)

print("Loading mixed recording...")
audio, sample_rate = sf.read(INPUT_FILE, dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)
if sample_rate != SR:
    raise ValueError(f"Expected {SR} Hz input (run preprocess_mixed.py), got {sample_rate} Hz")

mixture = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

print("Separating speakers...")
with torch.no_grad():
    separated = separator.separate_batch(mixture)
separated = separated.squeeze(0).cpu().numpy()
print(f"Separation complete. Output shape: {separated.shape}")

for i in range(separated.shape[-1]):
    output_file = OUTPUT_DIR / f"speaker_{i + 1}.wav"
    sf.write(output_file, separated[:, i], SR)
    print(f"Saved: {output_file}")

print("\nAll speaker tracks saved.")
