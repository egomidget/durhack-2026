"""
Combine extract_mikolaj.py (SepFormer blind 3-speaker separation) with
identify_speaker.py (ECAPA profile matching) to actually remove Mikolaj:

  1. Take the 3 channels extract_mikolaj.py already separated out of the mix.
  2. Resample each to 16 kHz (ECAPA/the enrolled profile expect 16 kHz; SepFormer
     works at 8 kHz) and embed each with the enrolled ECAPA model.
  3. Whichever channel is most similar to the "Mikolaj" profile is assumed to be him.
  4. "Removed" output = the other two channels mixed together.

Run extract_mikolaj.py first so audiofiles/processed/separated/speaker_{1,2,3}.wav exist.
"""
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path

import identify_speaker as idsp

BASE_DIR = Path(__file__).resolve().parent
SEPARATED_DIR = BASE_DIR / "audiofiles" / "processed" / "separated"
OUTPUT_DIR = BASE_DIR / "audiofiles" / "processed" / "removed"
SEPFORMER_SR = 8000
TARGET_SR = 16000


def _resample(wav, sr_in, sr_out):
    t = torch.from_numpy(wav).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr_in, sr_out)
    return t.squeeze(0).numpy()


def main():
    channels = sorted(SEPARATED_DIR.glob("speaker_*.wav"))
    if len(channels) < 2:
        raise RuntimeError(f"Expected separated speaker_*.wav in {SEPARATED_DIR} - run extract_mikolaj.py first.")

    profiles = idsp.load_speaker_profiles()
    if "Mikolaj" not in profiles:
        raise RuntimeError("No 'Mikolaj' profile found - run enrol_mikolaj.py first.")
    target = profiles["Mikolaj"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores, waves_16k = {}, {}
    for path in channels:
        wav, sr = sf.read(str(path), dtype="float32")
        wav16 = _resample(wav, sr, TARGET_SR)
        waves_16k[path.name] = wav16

        tmp = OUTPUT_DIR / f"_tmp_{path.stem}_16k.wav"
        sf.write(str(tmp), wav16, TARGET_SR, subtype="PCM_16")
        emb = idsp.extract_embedding(tmp)
        tmp.unlink()

        scores[path.name] = F.cosine_similarity(emb.unsqueeze(0), target.unsqueeze(0)).item()
        print(f"{path.name}: similarity to Mikolaj = {scores[path.name]:.4f}")

    mikolaj_channel = max(scores, key=scores.get)
    print(f"\nAssuming {mikolaj_channel} is Mikolaj (similarity {scores[mikolaj_channel]:.4f}).")

    others = [wav for name, wav in waves_16k.items() if name != mikolaj_channel]
    n = min(len(w) for w in others)
    removed = sum(w[:n] for w in others)

    out_path = OUTPUT_DIR / "mixed_talking_mikolaj_removed.wav"
    sf.write(str(out_path), removed, TARGET_SR, subtype="PCM_16")
    print(f"Saved: {out_path}")

    mikolaj_path = OUTPUT_DIR / "mixed_talking_mikolaj_only.wav"
    sf.write(str(mikolaj_path), waves_16k[mikolaj_channel], TARGET_SR, subtype="PCM_16")
    print(f"Saved: {mikolaj_path}")


if __name__ == "__main__":
    main()
