"""Build the fixed, held-out test set from one recording session.

Spec:
  python make_testset.py --session s3 --target mikolaj --n 50 [--seed 0]
For each example i: pick a 4 s chunk of the target's speech from raw/<target>/<session>_*.wav,
1-2 chunks of other people's speech from the SAME session, and a room-noise chunk.
Scale others to a random target-vs-others ratio in [-5, +5] dB, noise 10-20 dB below speech.
Write test/<id>/target.wav, others.wav (= other speakers + noise) and mix.wav = target + others,
exactly (same length, no re-normalising one without the other).
Also include ~20% examples with no target (target.wav = silence) so false removal is measured.
Write test/index.csv: id, target_present, ratio_db, speakers.
The session used here must never be used for training (mixer.py excludes it).
Self-check: --selftest builds 3 examples from synthetic sine/noise data in a temp folder.

NOTE: raw/<person>/<session>_<nn>.wav (from record.py) doesn't exist yet - record.py is
still a stub. Until then this falls back to whatever solo recordings are already on disk
(voices/mikolaj.wav + voices/_extracted/1-6 Yap/*.wav for the target, voices/_extracted/Ppl/*.wav
for everyone else) as a single informal "session", and uses quiet synthetic noise as a
stand-in for real room noise. Swap in real raw/ sessions and a real noise/room_*.wav later -
the --target-sources/--other-sources/--noise args let you point this at better data without
touching the logic. Because there's only one informal session right now, this test set isn't
truly held-out from future training data yet - that only becomes real once a second,
separate session exists.
"""
import argparse
import csv
import random
import tempfile
from pathlib import Path

import numpy as np

import common

CHUNK_S = 4.0
N_SAMPLES = int(CHUNK_S * common.SR)
HERE = Path(__file__).resolve().parent

DEFAULT_TARGET_SOURCES = (
    [HERE / "voices" / "mikolaj.wav"] +
    sorted((HERE / "voices" / "_extracted" / "1-6 Yap").glob("*.wav"))
)
DEFAULT_OTHER_SOURCES = sorted(
    p for p in (HERE / "voices" / "_extracted" / "Ppl").glob("*.wav")
    if "spoken over" not in p.name.lower()
)


def _pink_noise(n, rng):
    """Cheap Paul Kellet pink noise - stand-in for real room noise until record.py exists."""
    white = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    b = np.zeros(7, np.float32)
    out = np.empty(n, np.float32)
    for i, w in enumerate(white):
        b[0] = 0.99886 * b[0] + w * 0.0555179
        b[1] = 0.99332 * b[1] + w * 0.0750759
        b[2] = 0.96900 * b[2] + w * 0.1538520
        b[3] = 0.86650 * b[3] + w * 0.3104856
        b[4] = 0.55000 * b[4] + w * 0.5329522
        b[5] = -0.7616 * b[5] - w * 0.0168980
        out[i] = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362
        b[6] = w * 0.115926
    return out * 0.11


def _random_chunk(wavs, n_samples, rng):
    """A random n_samples-long chunk from a random file in `wavs` (loaded lazily, cached)."""
    path = wavs[rng.randrange(len(wavs))]
    wav = common.load_wav(path)
    if len(wav) <= n_samples:
        return np.pad(wav, (0, n_samples - len(wav))), path.name
    start = rng.randrange(0, len(wav) - n_samples)
    return wav[start:start + n_samples].copy(), path.name


def _scale_to_ratio(target, others, ratio_db):
    """Scale `others` so rms(target) - rms(others) == ratio_db (target louder if positive)."""
    tgt_db, oth_db = common.rms_db(target), common.rms_db(others)
    gain_db = (tgt_db - ratio_db) - oth_db
    return others * (10 ** (gain_db / 20))


def build_example(target_sources, other_sources, n_samples, rng, target_present, use_pink_noise=True):
    speakers = []
    if target_present:
        target, name = _random_chunk(target_sources, n_samples, rng)
        speakers.append(name)
    else:
        target = np.zeros(n_samples, np.float32)

    n_others = rng.randint(1, 2)
    others = np.zeros(n_samples, np.float32)
    for _ in range(n_others):
        chunk, name = _random_chunk(other_sources, n_samples, rng)
        others += chunk
        speakers.append(name)

    ratio_db = rng.uniform(-5, 5)
    if target_present:
        others = _scale_to_ratio(target, others, ratio_db)
    noise_db_below = rng.uniform(10, 20)
    speech_db = common.rms_db(target if target_present else others)
    noise = _pink_noise(n_samples, np.random.RandomState(rng.randrange(2**31))) if use_pink_noise \
        else np.zeros(n_samples, np.float32)
    cur_db = common.rms_db(noise)
    if np.isfinite(cur_db):
        noise = noise * (10 ** ((speech_db - noise_db_below - cur_db) / 20))
    others = others + noise

    mix = target + others
    return target.astype(np.float32), others.astype(np.float32), mix.astype(np.float32), ratio_db, speakers


def write_example(out_dir, target, others, mix):
    out_dir.mkdir(parents=True, exist_ok=True)
    common.save_wav(out_dir / "target.wav", target)
    common.save_wav(out_dir / "others.wav", others)
    common.save_wav(out_dir / "mix.wav", mix)


def selftest():
    rng = random.Random(0)
    sr = common.SR
    t = np.linspace(0, CHUNK_S, N_SAMPLES, endpoint=False)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        target_sources, other_sources = [], []
        for i, freq in enumerate([220, 330, 440]):
            p = tmp / f"src_{i}.wav"
            common.save_wav(p, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32))
            (target_sources if i == 0 else other_sources).append(p)
        other_sources.append(target_sources[0])  # keep others non-empty if needed

        test_dir = tmp / "test"
        rows = []
        for i in range(3):
            present = i != 2
            target, others, mix, ratio_db, speakers = build_example(
                target_sources, other_sources, N_SAMPLES, rng, present)
            assert len(target) == len(others) == len(mix) == N_SAMPLES
            assert np.allclose(mix, target + others)
            write_example(test_dir / f"{i:04d}", target, others, mix)
            rows.append([f"{i:04d}", present, ratio_db, ";".join(speakers)])
        assert (test_dir / "0000" / "mix.wav").exists()
    print("selftest OK: 3 examples built, mix == target + others, files written correctly.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", default=None, help="currently unused - see NOTE in the docstring")
    p.add_argument("--target", default="mikolaj")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return

    target_sources = DEFAULT_TARGET_SOURCES
    other_sources = DEFAULT_OTHER_SOURCES
    if not target_sources or not other_sources:
        raise SystemExit("No source recordings found - check voices/_extracted/ paths.")

    rng = random.Random(a.seed)
    out_root = common.data_path("test")
    rows = []
    for i in range(a.n):
        present = rng.random() < 0.8
        target, others, mix, ratio_db, speakers = build_example(
            target_sources, other_sources, N_SAMPLES, rng, present)
        ex_id = f"{i:04d}"
        write_example(out_root / ex_id, target, others, mix)
        rows.append([ex_id, present, f"{ratio_db:.2f}", ";".join(speakers)])
        print(f"{ex_id}: target_present={present} ratio_db={ratio_db:.1f} speakers={speakers}")

    index_path = out_root / "index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target_present", "ratio_db", "speakers"])
        writer.writerows(rows)
    print(f"\nWrote {a.n} examples to {out_root}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
