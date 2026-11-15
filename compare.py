#!/usr/bin/env python3
"""
Score every removal pipeline on the held-out test set and print one side-by-side
comparison table. Doesn't write to results/metrics.csv (score.py does that, and is
the permanent record) - this is a quick, repeatable presentation view, safe to run
from the launcher as many times as you like.

  python compare.py [--alpha 1.1] [--target mikolaj]
Requires make_testset.py to have been run first.
"""
import argparse

import numpy as np

import common
import score


def run_one(name, rows, remover, enroll, alpha):
    results = []
    for i, row in enumerate(rows, 1):
        print(f"  [{name}] clip {row['id']} ({i}/{len(rows)})...")
        si_sdr_db, leakage_db, vb, va = score.score_clip(row["id"], row, remover, enroll, alpha)
        results.append((si_sdr_db, leakage_db, vb, va))
    return results


def summarize(name, results):
    si_present = [r[0] for r in results if r[1] is not None]
    si_absent = [r[0] for r in results if r[1] is None]
    leak = [r[1] for r in results if r[1] is not None]
    vb = [r[2] for r in results]
    va = [r[3] for r in results]
    return {
        "name": name,
        "si_present": np.mean(si_present) if si_present else float("nan"),
        "si_absent": np.mean(si_absent) if si_absent else float("nan"),
        "leakage": np.mean(leak) if leak else float("nan"),
        "vp_before": np.mean(vb),
        "vp_after": np.mean(va),
    }


def print_table(summaries):
    cols = [("Model", "name", 18, "s"), ("SI-SDR present", "si_present", 14, ".2f"),
           ("SI-SDR absent", "si_absent", 13, ".2f"), ("Leakage dB", "leakage", 11, ".2f"),
           ("VP before", "vp_before", 10, ".3f"), ("VP after", "vp_after", 9, ".3f")]
    header = "  ".join(f"{label:<{w}}" if t == "s" else f"{label:>{w}}" for label, _, w, t in cols)
    print(header)
    print("-" * len(header))
    for s in summaries:
        line = "  ".join(
            f"{s[key]:<{w}}" if t == "s" else f"{s[key]:>{w}{t}}" for _, key, w, t in cols)
        print(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=1.1)
    p.add_argument("--target", default="mikolaj")
    a = p.parse_args()

    rows = score.load_test_set()
    enroll = common.load_wav(score._find_enroll(a.target))

    configs = [
        ("passthrough", lambda: (lambda mix, enroll, alpha: score._passthrough_remove(mix, enroll, alpha))),
        ("wesep (remove.py)", lambda: (lambda model=score.remove_mod.load_model(
            common.data_path("checkpoints/pretrained/spk_emb_100")):
            (lambda mix, enroll, alpha: score.remove_mod.remove(mix, enroll, model, alpha)[:2]))()),
        ("sepformer (Prema)", lambda: score.build_sepformer_remover(a.target.capitalize())),
    ]

    summaries = []
    for name, build in configs:
        print(f"\nLoading {name} (model load can take a minute)...")
        remover = build()
        print(f"Scoring {name} on {len(rows)} clips...")
        results = run_one(name, rows, remover, enroll, a.alpha)
        summaries.append(summarize(name, results))

    print("\n" + "=" * 70)
    print("COMPARISON (held-out test set, n={})".format(len(rows)))
    print("=" * 70)
    print_table(summaries)
    print("\nSI-SDR present = quality of 'everyone else' when target was actually there (higher better)")
    print("SI-SDR absent  = collateral damage when target wasn't there at all (higher better, closer to the floor)")
    print("Leakage dB     = how much of the target's voice remains after removal (lower better)")
    print("VP before/after = Resemblyzer similarity to the target's voiceprint (should drop)")


if __name__ == "__main__":
    main()
