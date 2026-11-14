"""Training examples built on the fly (imported by notebooks/finetune.ipynb, never run).

Spec:
  from mixer import MixDataset
  ds = MixDataset(target="mikolaj", exclude_sessions=["s3"], seconds=3.0, extra_speakers_dir=None)
  mix, target, others, enroll = ds[i]     # float32 torch tensors, 16 kHz
Each item:
  - 3 s of target speech + 1-2 other speakers (teammates, plus LibriSpeech dev-clean
    clips from extra_speakers_dir if given) at a random ratio in [-5, +5] dB,
    + room noise 10-20 dB below speech. others = everything except the target.
  - 30% of items: no target (target = zeros, others = mix)  -> remover must leave audio alone.
  - 15% of items: target only (others = zeros)               -> remover should output silence.
  - enroll = a random 6-10 s chunk of the target's speech from a DIFFERENT file than the mix.
  - Random gain on the whole mix (-6..+6 dB). mix == target + others exactly.
  - __len__ = a large number (e.g. 100_000); items are random, seeded per index for repeatability.
Load all source WAVs into memory once (it's a few minutes of audio).
"""


class MixDataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO (M2): implement per the module docstring")
