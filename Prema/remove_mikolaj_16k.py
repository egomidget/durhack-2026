"""
Full pipeline, chunked: SepFormer-whamr16k separation + ECAPA identification,
combined per-chunk so Mikolaj's voice stays on a consistent output stream across
the whole clip.

Why chunked: SepFormer at 16 kHz on the ~2 minute mixed_talking clip is too slow/
heavy to run in one shot on a laptop CPU (16 kHz doubles the sequence length vs
the old 8 kHz libri3mix run). We split into short chunks instead.

Why re-identify per chunk: SepFormer doesn't guarantee which output channel
(1 or 2) is which speaker between separate runs - so after separating each
chunk, we ECAPA-embed both of that chunk's channels against the enrolled
Mikolaj profile and keep whichever one matches, appending to a running
"Mikolaj" stream and an "everyone else" stream.

Run preprocess_mixed.py and enrol_mikolaj.py first.

  python remove_mikolaj_16k.py [--mix path/to/clip.wav] [--out-dir path/to/dir]
Defaults to the original mixed_talking_processed.wav / removed_16k/ if omitted.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from speechbrain.inference.separation import SepformerSeparation

import identify_speaker as idsp

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MIX_PATH = BASE_DIR / "audiofiles" / "processed" / "mixed_talking_processed.wav"
DEFAULT_OUTPUT_DIR = BASE_DIR / "audiofiles" / "processed" / "removed_16k"
SR = 16000
CHUNK_S = 10.0
# From real data: genuine Mikolaj chunks scored 0.29-0.56 against his profile,
# the other (non-Mikolaj) channel scored 0.05-0.19. Below this, assume he
# isn't actually in this chunk at all - don't force a removal that isn't real.
MIKOLAJ_THRESHOLD = 0.25


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mix", default=str(DEFAULT_MIX_PATH))
    p.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    a = p.parse_args()

    mix_path = Path(a.mix)
    OUTPUT_DIR = Path(a.out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = idsp.load_speaker_profiles()
    if "Mikolaj" not in profiles:
        raise RuntimeError("No 'Mikolaj' profile found - run enrol_mikolaj.py first.")
    target = profiles["Mikolaj"]

    print("Loading SepFormer (whamr16k, 16 kHz, 2-speaker) model...")
    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr16k",
        savedir=str(BASE_DIR / "models" / "sepformer-whamr16k"),
        run_opts={"device": "cpu"})

    mix, sr = sf.read(str(mix_path), dtype="float32")
    if mix.ndim == 2:
        mix = mix.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        mix = resample_poly(mix, SR // g, sr // g).astype(np.float32)

    chunk_n = int(CHUNK_S * SR)
    n_chunks = (len(mix) + chunk_n - 1) // chunk_n
    mikolaj_stream, other_stream = [], []

    for i in range(n_chunks):
        seg = mix[i * chunk_n:(i + 1) * chunk_n]
        print(f"\nChunk {i + 1}/{n_chunks} ({len(seg) / SR:.1f}s)")

        with torch.no_grad():
            out = separator.separate_batch(torch.tensor(seg, dtype=torch.float32).unsqueeze(0))
        out = out.squeeze(0).cpu().numpy()  # (T, 2)

        scores = []
        for ch in range(out.shape[1]):
            tmp = OUTPUT_DIR / "_tmp_chunk.wav"
            sf.write(str(tmp), out[:, ch], SR, subtype="PCM_16")
            emb = idsp.extract_embedding(tmp)
            scores.append(F.cosine_similarity(emb.unsqueeze(0), target.unsqueeze(0)).item())
        tmp.unlink(missing_ok=True)

        mikolaj_idx = int(np.argmax(scores))
        if scores[mikolaj_idx] < MIKOLAJ_THRESHOLD:
            print(f"  channel similarities to Mikolaj: {scores} -> below threshold, "
                 "Mikolaj not detected in this chunk - leaving audio untouched")
            mikolaj_stream.append(np.zeros_like(seg))
            other_stream.append(seg)
            continue

        print(f"  channel similarities to Mikolaj: {scores} -> channel {mikolaj_idx + 1} is him")
        mikolaj_stream.append(out[:, mikolaj_idx])
        other_stream.append(out[:, 1 - mikolaj_idx])

    mikolaj_full = np.concatenate(mikolaj_stream)
    other_full = np.concatenate(other_stream)

    stem = mix_path.stem
    removed_path = OUTPUT_DIR / f"{stem}_mikolaj_removed.wav"
    only_path = OUTPUT_DIR / f"{stem}_mikolaj_only.wav"
    sf.write(str(removed_path), other_full, SR, subtype="PCM_16")
    sf.write(str(only_path), mikolaj_full, SR, subtype="PCM_16")
    print(f"\nSaved: {removed_path}")
    print(f"Saved: {only_path}")


if __name__ == "__main__":
    main()
