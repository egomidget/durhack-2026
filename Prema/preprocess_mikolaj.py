
import torch
import torchaudio
import soundfile as sf

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ENROLMENT_DIR = BASE_DIR / "audiofiles" / "mikolaj_enrolment"
VALIDATION_DIR = BASE_DIR / "audiofiles" / "mikolaj_validation"
OUTPUT_DIR = BASE_DIR / "audiofiles" / "processed"

TARGET_SAMPLE_RATE = 16000

def preprocess_recording(input_path, output_path):

    audio_data, original_rate = sf.read(
        str(input_path),
        dtype="float32",
        always_2d=True
    )

    waveform = torch.from_numpy(audio_data.T)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    print(f"\nProcessing: {input_path.name}")
    print(f"Original sample rate: {original_rate} Hz")

    # Resample to 16 kHz
    if original_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=original_rate,
            new_freq=TARGET_SAMPLE_RATE
        )

    waveform = waveform.squeeze(0)

    # Save processed recording
    sf.write(
        str(output_path),
        waveform.numpy(),
        TARGET_SAMPLE_RATE,
        subtype="PCM_16"
    )

    duration = len(waveform) / TARGET_SAMPLE_RATE

    print(f"New sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Saved to: {output_path}")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recordings = [
        (ENROLMENT_DIR, ["1.wav", "2.wav", "3.wav"]),
        (VALIDATION_DIR, ["4.wav", "5.wav", "6.wav"])
    ]

    for folder, filenames in recordings:

        for filename in filenames:

            input_path = folder / filename

            output_name = (
                f"mikolaj_{filename[0]}_processed.wav"
            )

            output_path = OUTPUT_DIR / output_name

            if not input_path.exists():
                raise FileNotFoundError(input_path)

            preprocess_recording(
                input_path,
                output_path
            )

    print("\nAll six recordings processed successfully!")


if __name__ == "__main__":
    main()
