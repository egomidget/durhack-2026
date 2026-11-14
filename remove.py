"""Remove one speaker from a recording using a target-speaker extraction model, run backwards.

Spec:
  python remove.py --mix overlaps/clip1.wav --enroll enroll/mikolaj.wav \
                   --ckpt checkpoints/pretrained/spk_emb_100 --alpha 1.1 \
                   --out results/clip1_removed.wav [--save-target results/clip1_target.wav] [--save-mask mask.npy]
Paths are relative to HUSH_DATA unless absolute.
How: start from wesep-real-tse's evaluate.py (it loads config.yaml + avg_model.pt, computes the
speaker embedding from the enroll WAV and runs the BSRNN separator). The model outputs the
estimated TARGET STFT S_hat. Removal: Y = X - alpha * S_hat on complex STFTs
(512-pt FFT, hop 128, Hann - match the repo's settings), then iSTFT. Equivalent to a (1 - M) mask.
Also expose a function for score.py / the notebook:
  remove(mix: np.ndarray, enroll: np.ndarray, model, alpha=1.0) -> (removed, target_estimate, mask)
Optional: register a forward hook on the mask-estimation module to return the mask for plots.
Long files: process in 8 s chunks with overlap-add.
Runs on Colab GPU; should also run (slowly) on CPU for short clips.
"""


def main():
    raise NotImplementedError("TODO (M2): implement per the docstring")


if __name__ == "__main__":
    main()
