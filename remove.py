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
import argparse
import sys
from pathlib import Path

import numpy as np

import common

WESEP_REPO = Path(__file__).resolve().parent / "wesep-real-tse"


def _stub_diarization_deps():
    """wespeaker's __init__ eagerly imports its speaker-diarization CLI and an optional
    s3prl frontend, neither of which we use (we only use speaker embeddings via the
    plain Fbank/ECAPA path). hdbscan's Cython extension doesn't even build on newer
    Pythons - so fake these modules out rather than install packages that are never
    actually called on our code path."""
    import types
    for name in ("umap", "hdbscan", "s3prl", "s3prl.nn", "whisper", "transformers", "peft",
                "distutils", "distutils.util"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["distutils"].util = sys.modules["distutils.util"]
    sys.modules["distutils.util"].strtobool = lambda v: str(v).strip().lower() in (
        "y", "yes", "t", "true", "on", "1")
    sys.modules["umap"].UMAP = object
    sys.modules["hdbscan"].HDBSCAN = object
    sys.modules["s3prl"].nn = sys.modules["s3prl.nn"]
    sys.modules["s3prl.nn"].Featurizer = object
    sys.modules["s3prl.nn"].S3PRLUpstream = object
    for attr in ("Wav2Vec2BertModel", "Wav2Vec2BertConfig", "BitsAndBytesConfig",
                "AutoFeatureExtractor", "AutoModel"):
        setattr(sys.modules["transformers"], attr, object)
    sys.modules["peft"].LoraConfig = object
    sys.modules["peft"].get_peft_model = lambda *a, **k: None


def load_model(ckpt_dir, device="cpu"):
    """Load a wesep-real-tse Extractor from a checkpoint folder (avg_model.pt + config.yaml).

    output_norm is disabled: the repo's default peak-normalises the extracted speech to
    0.9, which would break the amplitude relationship that `remove()` relies on (it
    subtracts the estimate directly from the mix, so the estimate must stay on the mix's
    original scale).
    """
    if str(WESEP_REPO) not in sys.path and WESEP_REPO.exists():
        sys.path.insert(0, str(WESEP_REPO))
    _stub_diarization_deps()
    import wesep
    model = wesep.load_model_local(str(ckpt_dir))
    model.set_resample_rate(common.SR)
    model.set_vad(False)
    model.set_device(device)
    model.set_output_norm(False)
    return model


def _extract(model, mix_chunk: np.ndarray, enroll) -> np.ndarray:
    """Run the model on one chunk, aligned back to the chunk's own length."""
    import torch
    mix_t = torch.from_numpy(np.asarray(mix_chunk, dtype=np.float32)).unsqueeze(0)
    out = model.extract_speech_from_pcm(mix_t, common.SR, enroll, common.SR)
    out = out.squeeze(0).numpy().astype(np.float32)
    n = len(mix_chunk)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    else:
        out = out[:n]
    return out


def remove(mix: np.ndarray, enroll: np.ndarray, model, alpha: float = 1.0,
          chunk_s: float = 8.0):
    """Estimate the target's voice in `mix` and subtract it. mask is always None for now
    (the mask hook is optional per the spec and not implemented yet)."""
    import torch
    enroll_t = torch.from_numpy(np.asarray(enroll, dtype=np.float32)).unsqueeze(0)
    n = len(mix)
    chunk_n = int(chunk_s * common.SR)

    if n <= chunk_n:
        target = _extract(model, mix, enroll_t)
    else:
        hop_n = chunk_n // 2
        win = np.hanning(chunk_n).astype(np.float32)
        acc = np.zeros(n, dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)
        start = 0
        while True:
            end = min(start + chunk_n, n)
            seg_win = win[:end - start]
            acc[start:end] += _extract(model, mix[start:end], enroll_t) * seg_win
            wsum[start:end] += seg_win
            if end == n:
                break
            start += hop_n
        target = acc / np.where(wsum == 0, 1.0, wsum)

    removed = (mix - alpha * target).astype(np.float32)
    return removed, target.astype(np.float32), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mix", required=True)
    p.add_argument("--enroll", required=True)
    p.add_argument("--ckpt", required=True, help="checkpoint folder (avg_model.pt + config.yaml)")
    p.add_argument("--alpha", type=float, default=1.0, help="removal strength")
    p.add_argument("--out", required=True)
    p.add_argument("--save-target", help="also save the estimated target speech")
    p.add_argument("--save-mask", help="not implemented yet - mask hook is optional per the spec")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    mix = common.load_wav(common.data_path(a.mix))
    enroll = common.load_wav(common.data_path(a.enroll))
    model = load_model(common.data_path(a.ckpt), device=a.device)

    removed, target, mask = remove(mix, enroll, model, alpha=a.alpha)

    out_path = common.data_path(a.out)
    common.save_wav(out_path, removed)
    print(f"Saved {out_path}  (mix {common.rms_db(mix):.1f} dB -> removed {common.rms_db(removed):.1f} dB)")

    if a.save_target:
        common.save_wav(common.data_path(a.save_target), target)
    if a.save_mask:
        print("--save-mask: not implemented yet (optional mask hook, see docstring)")


if __name__ == "__main__":
    main()
