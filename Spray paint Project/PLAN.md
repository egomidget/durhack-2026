# HUSH — Project & Milestone Plan

Repo copy of the team doc (live version: https://claude.ai/artifact/ACjxeT91Wj2nD325gbnZZm).

## Overview

Secure a working live demo first, then spend the rest of the time on the ML result:
a model that removes one chosen voice from overlapping speech.

1. **Live app (`hush.py`, done)** — voiceprint detection + volume ducking, voice commands. Works when people take turns.
2. **Flipped pretrained model** — WeSep BSRNN (REAL-TSE checkpoint) run backwards: mix minus target. Offline, on recorded clips.
3. **Our fine-tuned remover** — same model trained further on our voices/room/mic with a loss that punishes leaving Mikołaj in. Causal checkpoint, so it can later run live.

Headline for judges: "We fine-tuned a published speech-extraction model to remove a voice instead of isolating it, and cut Mikołaj's leftover voice by X dB."

## Roles

| Workstream | Owns | Hands off to |
|---|---|---|
| A · Data | Recordings, data mixer, held-out test set | B (test clips), C (training data) |
| B · Flipped model | Colab setup, checkpoint, `remove.py`, `score.py` | C (baseline numbers), D (before/after clips) |
| C · Fine-tuning | Mask hook, removal loss, training run, evaluation | D (results table) |
| D · Live demo + pitch | `hush.py` on demo hardware, live wrapper (stretch), slides, rehearsal | Everyone |

## Milestones (in order)

1. **M1 Fallback secured** — `hush.py` mutes Mikołaj by voice command on demo hardware. All recordings done. Checkpoint loads in Colab.
2. **M2 Baseline** — flipped model removes Mikołaj from a real overlap clip. Baseline rows in `results/metrics.csv`. Mixer produces training examples.
3. **M3 Model trained** — fine-tune finished (or best checkpoint saved) + training curve. Start it before people sleep.
4. **M4 Results** — fine-tuned vs baseline table. Go/no-go on the live causal wrapper.
5. **M5 Demo frozen** — no code changes after this. Slides done, before/after clips rendered.
6. **Demo.**

## Tasks

### A · Data
- [ ] Solo recordings, 5–10 min per teammate (reading, chatting, loud and quiet) — Mikołaj first
- [ ] Enrollment clips, 20 s each, recorded separately
- [ ] 2 min of empty-room noise
- [ ] 4–5 real overlap clips, 20 s each (Mikołaj + Prema talking over each other)
- [ ] Hold back one full session as the test set
- [ ] `make_testset.py`, `mixer.py` (see docstrings)
- [ ] Small LibriSpeech dev-clean subset so the model doesn't memorise the team

### B · Flipped model
- [ ] Colab: Python 3.10, torch 2.7.1, WeSpeaker, wesep-real-tse
- [ ] Download checkpoints now (`spk_emb_100`, `spk_emb_causal_100`)
- [ ] Sanity check: repo `evaluate.py` extracts Mikołaj from one of our mixes
- [ ] `remove.py` with removal strength α
- [ ] Mask hook for plots
- [ ] `score.py`: SI-SDR, leakage dB, voiceprint before/after
- [ ] Baseline numbers + before/after WAVs of the overlap clips

### C · Fine-tuning (see `docs/FINETUNE.md`)
- [ ] Find and hook the mask module
- [ ] Removal loss (β ≈ 3)
- [ ] Training loop from `spk_emb_causal_100`, encoder frozen, Adam 1e-4, batch 8, AMP
- [ ] Time first 50 steps; size the run to finish overnight
- [ ] Checkpoint every 500 steps to Drive; resume support
- [ ] Evaluate vs baseline; optional β ∈ {1, 3, 10}

### D · Live demo + pitch
- [ ] `hush.py` on demo laptop; enroll everyone incl. presenter
- [ ] Wired headphones + splitter; tune threshold in the room
- [ ] Voice commands tested; keys 1–9 as backup
- [ ] Stretch: `live_remove.py` (causal model on 0.5 s chunks) instead of ducking
- [ ] Slides: problem, how it works, the flip, results, voiceprint-drop chart, spectrograms
- [ ] Rehearse twice with full hardware

## Demo script

1. Pitch (30 s): sleeping baby, loud household, a cot mobile that tunes out one voice.
2. Live (60 s): judges on headphones; "hush, mute Mikołaj" → gone; "mute Prema"; "unmute Mikołaj".
3. ML showpiece (60 s): real overlap clip, then our output — Mikołaj gone, Prema still there; spectrograms + voiceprint drop.
4. Results (30 s): flipped baseline vs fine-tuned on the held-out set.
5. Roadmap (15 s): live overlap removal; cot-mobile version that masks with shaped noise.

Fallbacks: keys/buttons if voice commands misfire · raise threshold or play recorded video if the room is loud · present fine-tune as WIP with its training curve if it didn't beat baseline · pre-rendered clips on a second device.

## Risks

| Risk | Mitigation |
|---|---|
| torch / WeSpeaker version conflicts | Colab with pinned versions only |
| Drive download limit on checkpoint | Download now; keep two copies |
| Colab disconnects overnight | Checkpoint every 500 steps; resume |
| Pretrained model sounds poor on our room | That's the fine-tuning story; show it as baseline |
| Too slow on CPU for live | Stretch only; `hush.py` stays the live demo |
| Similar voices confused | Enroll everyone; tune threshold in the room |
| Checkpoint has no stated license | Fine for a hackathon |

## Resources

- https://github.com/REAL-TSE/wesep-real-tse — checkpoints, `evaluate.py`
- https://huggingface.co/ferdinandl007/REAL-TSE-BSRNN-CoreML — I/O format, STFT settings
- https://github.com/wenet-e2e/WeSep
- https://www.isca-archive.org/interspeech_2022/ronssin22_interspeech.pdf — real-time causal extraction on CPU
- https://google.github.io/speaker-id/publications/VoiceFilter/ · https://arxiv.org/pdf/1810.04826
- https://research.google/pubs/voicefilter-lite-streaming-targeted-voice-separation-for-on-device-speech-recognition/
- https://huggingface.co/nguyenvulebinh/voice-filter — backup model
- https://alphacephei.com/vosk/models — `vosk-model-small-en-us-0.15`
