"""Score a remover on the held-out test set and append rows to results/metrics.csv.

Spec:
  python score.py --ckpt <checkpoint dir or .pt> --name baseline [--alpha 1.0] [--target mikolaj]
  python score.py --selftest      # checks si_sdr / leakage on synthetic signals, no model needed
For every test/<id>/: run remove.remove() on mix.wav with enroll/<target>.wav, then compute
  si_sdr_db          = common.si_sdr(removed, others)            higher = better
  leakage_db         = common.leakage_db(target_after, target)    lower = better
                       (target_after = same subtraction applied to target.wav; skip if no target)
  voiceprint_before  = Resemblyzer cosine(mix, enroll)
  voiceprint_after   = Resemblyzer cosine(removed, enroll)      should drop
Append rows: clip,model,si_sdr_db,leakage_db,voiceprint_before,voiceprint_after
Print a summary (mean of each column) for this --name. Also score the "do nothing" baseline
once (removed = mix) under --name passthrough so the table has a floor.

--name passthrough (no --ckpt needed) scores "do nothing" (removed = mix) as the floor.
Requires make_testset.py to have been run first (test/index.csv + test/<id>/*.wav).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import common
import remove as remove_mod

HERE = Path(__file__).resolve().parent
METRICS_COLUMNS = ["clip", "model", "si_sdr_db", "leakage_db", "voiceprint_before", "voiceprint_after"]

_encoder = None


def _resemblyzer_embed(wav):
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder("cpu", verbose=False)
    return _encoder.embed_utterance(np.asarray(wav, dtype=np.float32))


def _find_enroll(target):
    candidates = [common.data_path("enroll", f"{target}.wav"), HERE / "voices" / f"{target}.wav"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No enrollment WAV for '{target}' - checked {candidates}")


def _passthrough_remove(mix, enroll, alpha):
    return mix.copy(), np.zeros_like(mix)


# SepFormer-whamr16k separation + ECAPA identification, scored against the same test
# clips. Unlike remove.py, it doesn't take an enroll array - the speaker profile is
# already baked in by Prema/enrol_mikolaj.py - but keeps the same remover(mix, enroll,
# alpha) shape so score_clip() doesn't need to know which engine it's calling.
# Test clips are only 4s (shorter than a chunk), so no chunking needed here.
MIKOLAJ_THRESHOLD = 0.25


def build_sepformer_remover(target_name="Mikolaj"):
    import sys
    import tempfile
    prema_dir = HERE / "Prema"
    if str(prema_dir) not in sys.path:
        sys.path.insert(0, str(prema_dir))
    import identify_speaker as idsp
    from speechbrain.inference.separation import SepformerSeparation

    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr16k",
        savedir=str(prema_dir / "models" / "sepformer-whamr16k"),
        run_opts={"device": "cpu"})
    profiles = idsp.load_speaker_profiles()
    if target_name not in profiles:
        raise RuntimeError(f"No '{target_name}' profile - run Prema/enrol_mikolaj.py first.")
    target_emb = profiles[target_name]

    def remover(mix, enroll, alpha):
        import soundfile as sf
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            out = separator.separate_batch(torch.tensor(mix, dtype=torch.float32).unsqueeze(0))
        out = out.squeeze(0).cpu().numpy()  # (T, 2)

        scores = []
        for ch in range(out.shape[1]):
            tmp = Path(tempfile.gettempdir()) / f"_score_sepformer_tmp_{ch}.wav"
            sf.write(str(tmp), out[:, ch], common.SR, subtype="PCM_16")
            emb = idsp.extract_embedding(tmp)
            scores.append(float(F.cosine_similarity(emb.unsqueeze(0), target_emb.unsqueeze(0)).item()))
            tmp.unlink(missing_ok=True)

        idx = int(np.argmax(scores))
        if scores[idx] < MIKOLAJ_THRESHOLD:
            return mix.copy(), np.zeros_like(mix)

        n = min(len(mix), out.shape[0])
        removed = np.pad(out[:n, 1 - idx], (0, len(mix) - n)).astype(np.float32)
        target_est = np.pad(out[:n, idx], (0, len(mix) - n)).astype(np.float32)
        return removed, target_est

    return remover


def load_test_set():
    index_path = common.data_path("test", "index.csv")
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found - run make_testset.py first")
    with open(index_path, newline="") as f:
        return list(csv.DictReader(f))


def score_clip(clip_id, row, remover, enroll, alpha):
    clip_dir = common.data_path("test", clip_id)
    mix = common.load_wav(clip_dir / "mix.wav")
    others = common.load_wav(clip_dir / "others.wav")
    target_present = row["target_present"] in ("True", "true", "1")

    removed, _ = remover(mix, enroll, alpha)
    si_sdr_db = common.si_sdr(removed, others)

    leakage_db = None
    if target_present:
        target = common.load_wav(clip_dir / "target.wav")
        target_after, _ = remover(target, enroll, alpha)
        leakage_db = common.leakage_db(target_after, target)

    voiceprint_before = float(np.dot(_resemblyzer_embed(mix), _resemblyzer_embed(enroll)))
    voiceprint_after = float(np.dot(_resemblyzer_embed(removed), _resemblyzer_embed(enroll)))
    return si_sdr_db, leakage_db, voiceprint_before, voiceprint_after


def append_rows(rows):
    out_path = common.data_path("results", "metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(METRICS_COLUMNS)
        writer.writerows(rows)
    return out_path


def print_summary(name, rows):
    # target_present clips and target_absent clips aren't comparable on si_sdr_db: a
    # target_absent clip trivially gets a huge score from any remover that doesn't add
    # spurious energy (others.wav == mix.wav there), so mixing them inflates the mean
    # and makes it useless for comparing models. leakage_db is None exactly when the
    # target was absent (see score_clip), so we reuse that to split the two groups.
    si_present = [r[2] for r in rows if r[3] is not None]
    si_absent = [r[2] for r in rows if r[3] is None]
    leak = [r[3] for r in rows if r[3] is not None]
    vb = [r[4] for r in rows]
    va = [r[5] for r in rows]
    print(f"\n=== {name} ===  ({len(rows)} clips)")
    print(f"mean si_sdr_db (target present, n={len(si_present)}): {np.mean(si_present):.2f}"
         if si_present else "mean si_sdr_db (target present): n/a")
    print(f"mean si_sdr_db (target absent,  n={len(si_absent)}):  {np.mean(si_absent):.2f}"
         if si_absent else "mean si_sdr_db (target absent): n/a")
    print(f"mean leakage_db:        {np.mean(leak):.2f}  (n={len(leak)} clips with target present)")
    print(f"mean voiceprint_before: {np.mean(vb):.3f}")
    print(f"mean voiceprint_after:  {np.mean(va):.3f}  (drop = {np.mean(vb) - np.mean(va):.3f})")


def selftest():
    sr = common.SR
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sine = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    noise = (0.3 * np.random.RandomState(0).randn(len(t))).astype(np.float32)

    identical = common.si_sdr(sine, sine)
    assert identical > 100, f"si_sdr of identical signals should be very high, got {identical}"

    # Note: truly uncorrelated signals give strongly negative SI-SDR (the projection
    # scale is near zero, so almost none of the noise's energy counts as "target") -
    # not ~0 dB. The real sanity check is that it's far below the identical-signal case.
    unrelated = common.si_sdr(noise, sine)
    assert unrelated < identical - 50, f"si_sdr of unrelated signals should be far below identical, got {unrelated}"

    same_level = common.leakage_db(sine, sine)
    assert abs(same_level) < 0.1, f"leakage_db of identical levels should be ~0, got {same_level}"

    quieter = common.leakage_db(sine * 0.1, sine)
    assert quieter < -15, f"leakage_db after a 10x quieter signal should be strongly negative, got {quieter}"

    print("selftest OK: si_sdr and leakage_db behave as expected on synthetic signals.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", help="checkpoint dir (wesep engine only; not needed for passthrough)")
    p.add_argument("--name", help="model name for the results table")
    p.add_argument("--engine", choices=["wesep", "sepformer"], default="wesep",
                   help="wesep = remove.py's WeSpeaker/BSRNN; sepformer = Prema's SepFormer+ECAPA")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--target", default="mikolaj")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return

    if not a.name:
        raise SystemExit("--name is required (e.g. --name baseline, or --name passthrough)")

    rows_for_table = load_test_set()
    enroll = common.load_wav(_find_enroll(a.target))

    if a.name == "passthrough":
        remover = lambda mix, enroll, alpha: _passthrough_remove(mix, enroll, alpha)
    elif a.engine == "sepformer":
        remover = build_sepformer_remover(a.target.capitalize())
    else:
        if not a.ckpt:
            raise SystemExit("--ckpt is required unless --name passthrough")
        model = remove_mod.load_model(common.data_path(a.ckpt))
        remover = lambda mix, enroll, alpha: remove_mod.remove(mix, enroll, model, alpha)[:2]

    csv_rows = []
    for row in rows_for_table:
        clip_id = row["id"]
        si_sdr_db, leakage_db, vb, va = score_clip(clip_id, row, remover, enroll, a.alpha)
        print(f"{clip_id}: si_sdr={si_sdr_db:.2f}dB leakage={leakage_db if leakage_db is None else f'{leakage_db:.2f}dB'} "
             f"voiceprint {vb:.3f} -> {va:.3f}")
        csv_rows.append([clip_id, a.name, si_sdr_db, leakage_db, vb, va])

    append_rows(csv_rows)
    print_summary(a.name, csv_rows)


if __name__ == "__main__":
    main()
