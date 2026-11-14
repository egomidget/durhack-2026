"""Make the slide figures from results/.

Spec:
  python plots.py [--clip overlaps/clip1.wav --removed results/clip1_removed.wav]
Writes to results/:
  - results_table.png    mean si_sdr_db / leakage_db / voiceprint per model (passthrough, baseline, ours...)
  - voiceprint_drop.png  voiceprint score before vs after, per model (the "he's gone" chart)
  - spectrograms.png     before / after spectrograms of one clip, same colour scale, 16 kHz
  - training_curve.png   if checkpoints/ours/log.csv exists (step, loss, val_si_sdr)
matplotlib only; large fonts (these go on slides).
"""


def main():
    raise NotImplementedError("TODO (M4): implement per the docstring")


if __name__ == "__main__":
    main()
