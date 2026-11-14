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
"""


def main():
    raise NotImplementedError("TODO (M2): implement per the docstring")


if __name__ == "__main__":
    main()
