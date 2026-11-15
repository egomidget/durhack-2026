
import torch
import torchaudio
import soundfile as sf

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "audiofiles" / "mixed_talking.wav"
OUTPUT_PATH = (
    BASE_DIR / "audiofiles" / "processed"
    / "mixed_talking_processed.wav"
)

TARGET_SAMPLE_RATE = 16000


def main():

    audio, original_rate = sf.read(
        str(INPUT_PATH),
        dtype="float32",
        always_2d=True
    )

    print(f"Original sample rate: {original_rate}")
    print(f"Channels: {audio.shape[1]}")

    waveform = torch.from_numpy(audio.T)

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if original_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            original_rate,
            TARGET_SAMPLE_RATE
        )

    waveform = waveform.squeeze(0)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        str(OUTPUT_PATH),
        waveform.numpy(),
        TARGET_SAMPLE_RATE,
        subtype="PCM_16"
    )

    print("\nPreprocessing complete!")
    print(f"New sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Duration: {len(waveform) / TARGET_SAMPLE_RATE:.2f}s")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
