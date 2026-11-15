
import torch
import torchaudio
import soundfile as sf

from pathlib import Path



TARGET_SAMPLE_RATE = 16000


def preprocess_audio(input_path, output_path=None):

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")


    audio_data, sample_rate = sf.read(
        str(input_path),
        dtype="float32",
        always_2d=True
    )

    waveform = torch.from_numpy(audio_data.T)

    if sample_rate <= 0 or waveform.shape[1] == 0:
        raise ValueError("Invalid or empty audio file.")


    print("\nOriginal audio information")
    print("--------------------------")
    print(f"File: {input_path.name}")
    print(f"Sample rate: {sample_rate}")
    print(f"Channels: {waveform.shape[0]}")
    print(f"Duration: {waveform.shape[1] / sample_rate:.2f}s")

    duration = waveform.shape[1] / sample_rate

    if duration < 3:
        raise ValueError("Recording is too short.")


    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)


    peak = waveform.abs().max().item()

    if peak >= 0.999:
        print("WARNING: Audio may contain clipping.")


    if sample_rate != TARGET_SAMPLE_RATE:

        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE
        )

    waveform = waveform.to(torch.float32)


    rms = torch.sqrt(torch.mean(waveform ** 2)).item()

    if rms < 0.005:
        print("WARNING: Audio appears to be very quiet.")


    if output_path:

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            str(output_path),
            waveform.squeeze(0).numpy(),
            TARGET_SAMPLE_RATE,
            subtype="PCM_16"
        )

        print(f"Saved processed audio: {output_path}")


    print("\nProcessed audio information")
    print("---------------------------")
    print(f"Sample rate: {TARGET_SAMPLE_RATE}")
    print("Channels: 1")
    print(f"Duration: {waveform.shape[1] / TARGET_SAMPLE_RATE:.2f}s")
    print(f"RMS: {rms:.4f}")

    return waveform


if __name__ == "__main__":

    print("\n========== MIKOLAJ ==========")

    mikolaj_audio = preprocess_audio(
        "Prema/audiofiles/mikolaj.wav",
        "Prema/audiofiles/processed/mikolaj_processed.wav"
    )

    print("\n========== MERCEDES ==========")

    mercedes_audio = preprocess_audio(
        "Prema/audiofiles/mercedes.wav",
        "Prema/audiofiles/processed/mercedes_processed.wav"
    )

    print("\n================================")
    print("Both recordings processed successfully!")
    print("================================")

    print("\n========== TEST RECORDING ==========")

    test_audio = preprocess_audio(
        "Prema/audiofiles/test.wav",
        "Prema/audiofiles/processed/test_processed.wav"
    )

    print("\n========== MERCEDES TEST ==========")

    preprocess_audio(
        "Prema/audiofiles/Merc yap.wav",
        "Prema/audiofiles/processed/merc_yap_processed.wav"
    )

    print("\n========== PREMA TEST ==========")

    preprocess_audio(
        "Prema/audiofiles/Prema yap.wav",
        "Prema/audiofiles/processed/prema_yap_processed.wav"
    )