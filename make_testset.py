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
"""


def main():
    raise NotImplementedError("TODO (M2): implement per the docstring")


if __name__ == "__main__":
    main()
