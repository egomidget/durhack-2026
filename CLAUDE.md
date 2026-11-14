# CLAUDE.md — HUSH hackathon project

Read this first. The full plan is in `docs/PLAN.md`; the live team version is
https://claude.ai/artifact/ACjxeT91Wj2nD325gbnZZm

## What we're building

HUSH lets a listener mute one specific person's voice ("hush, mute Mikołaj") while
everyone else stays audible. Pitch: a cot-side device that tunes out parents' voices
for a sleeping baby. Demo: judges wear wired headphones; a mic is plugged into the
laptop; Mikołaj (teammate) is the target.

Three tracks, safest first:
1. `hush.py` (DONE): live app. Resemblyzer voiceprints detect who's talking; ducks the
   headphone volume while a muted person speaks. Vosk voice commands + GUI buttons.
2. Flipped pretrained model: WeSep BSRNN target-speaker extractor
   (REAL-TSE checkpoint, https://github.com/REAL-TSE/wesep-real-tse), run backwards:
   output = mix STFT − α · estimated target STFT. Offline, on recorded clips.
3. Fine-tuned remover: same model fine-tuned on our voices with a removal loss
   (−SI-SDR on "everyone else" + β · leftover-target energy). Start from the CAUSAL
   checkpoint `spk_emb_causal_100` so it can later run live.

## Conventions (follow these)

- **No database, no server, no API.** Every tool is a CLI script: `argparse` flags in,
  files out, `if __name__ == "__main__": main()`. Shared helpers go in `common.py`.
- **Audio is always WAV, 16 kHz, mono, 16-bit.** Use `common.load_wav` / `common.save_wav`
  (they resample/convert). Never pass other sample rates to the models.
- **Data lives outside the repo** in `HUSH-data/` (a shared Google Drive folder). Path comes
  from env var `HUSH_DATA` (default `./HUSH-data`); use `common.data_path(...)`.
  Never commit audio, checkpoints or results (see `.gitignore`).
- Layout of `HUSH-data/`:
  ```
  raw/<person>/<session>_<nn>.wav     solo recordings
  enroll/<person>.wav                 20 s enrollment clips
  noise/room_<nn>.wav                 empty-room noise
  overlaps/<clip>.wav                 real overlapping-speech recordings (demo)
  test/<id>/mix.wav target.wav others.wav   (mix = target + others, exactly)
  checkpoints/pretrained/spk_emb_causal_100/  avg_model.pt config.yaml
  checkpoints/pretrained/spk_emb_100/
  checkpoints/ours/step_<n>.pt
  results/metrics.csv  results/*.png  results/*.wav
  ```
- Results CSV columns: `clip,model,si_sdr_db,leakage_db,voiceprint_before,voiceprint_after`.
- Target speaker in code/CLI = `mikolaj` (ASCII). Command alias in `hush.py` = `michael`.
- **Never train on the held-out session** (the session passed to `make_testset.py`).
- Keep scripts small and readable — teammates are learning this stack.

## Two environments

- **Laptop** (Python 3.10 venv, `requirements.txt`): `hush.py`, `record.py`,
  `make_testset.py`, `plots.py`.
- **Google Colab GPU** (`requirements-colab.txt` + the wesep-real-tse repo,
  torch 2.7.1): `remove.py`, `score.py`, `notebooks/finetune.ipynb`.
  Laptops won't train BSRNN; don't try.

## Build order (milestones)

1. **M1 Fallback secured** — `hush.py` works on demo hardware (humans test this);
   `record.py` done so data can be recorded; checkpoint downloaded.
2. **M2 Baseline** — `make_testset.py`, `remove.py`, `score.py` → baseline rows in
   `results/metrics.csv`; `mixer.py` yields training examples.
3. **M3 Model trained** — `notebooks/finetune.ipynb` trains from the causal checkpoint,
   saves every 500 steps to Drive.
4. **M4 Results** — `plots.py`: results table, voiceprint-drop chart, before/after
   spectrograms. Go/no-go on `live_remove.py` (stretch: chunked causal model inside
   `hush.py` instead of ducking).
5. **M5 Demo frozen** — no code changes after this.

Each stub file has a docstring that is its spec. Implement in the order above.

## Things a human must do (don't try to automate)

- Download the checkpoints from Google Drive (link in the wesep-real-tse README).
- Download `vosk-model-small-en-us-0.15` next to `hush.py`.
- Record audio; run anything needing a mic or GPU.

## Testing without real data

Write quick self-checks that use synthetic audio (sine waves / noise) so scripts can be
verified on a laptop before real recordings exist, e.g. `python score.py --selftest`.
`common.si_sdr` must return ~+inf (very large) for identical signals and ~0 dB-ish for
unrelated ones.
