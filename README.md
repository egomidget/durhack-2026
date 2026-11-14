# HUSH — mute one voice

Say **"hush, mute Mikołaj"** and stop hearing one person while everyone else stays audible.
Hackathon project: live voice-matching demo + a speech-extraction model fine-tuned to *remove* a voice.

- Plan, milestones, roles: [`docs/PLAN.md`](docs/PLAN.md) (live team doc: https://claude.ai/artifact/ACjxeT91Wj2nD325gbnZZm)
- Fine-tuning spec: [`docs/FINETUNE.md`](docs/FINETUNE.md)
- Instructions for Claude Code: [`CLAUDE.md`](CLAUDE.md)

## Setup (laptop)

```bash
python3.10 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HUSH_DATA=~/path/to/Google\ Drive/HUSH-data          # shared data folder
```
- `webrtcvad` won't build? `pip install webrtcvad-wheels` then `pip install resemblyzer`.
- Voice commands: download `vosk-model-small-en-us-0.15` from https://alphacephei.com/vosk/models and unzip it next to `hush.py`.
- Linux: `sudo apt install python3-tk libportaudio2`. Mac: allow your terminal microphone access.
- Mac + Homebrew Python: if `import tkinter` fails with `_tkinter` errors, run
  `brew install python-tk@3.13` (match the version to your `python3`), then recreate `.venv`.

## Control panel (`launcher.py`, works now)

```bash
python launcher.py
```
One window with a button per script (devices, enroll, run, record, make test set, score,
plots) instead of typing each command. Runs the script as a subprocess and streams its
output into the panel; "Stop running script" kills a long-running one (e.g. `run`).
Buttons for scripts that are still stubs (`record.py`, `make_testset.py`, `score.py`,
`plots.py`) work too — they'll just show the stub's `NotImplementedError` until those are
implemented. Built with [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
(in `requirements.txt`).

## Live demo (`hush.py`, works now)

```bash
python hush.py devices                               # find mic / headphone numbers
python hush.py enroll Mikolaj --alias michael --in 2
python hush.py enroll Prema --in 2
python hush.py enroll Presenter --alias me --in 2
python hush.py run --in 2 --out 4
```
Say "hush mute michael" / "hush unmute michael" / "hush mute everyone", or press 1–9 (0 = unmute all).
Use wired headphones; enroll everyone in the demo room on the demo mic; tune the threshold slider live.

## Scripts

| File | Status | Run |
|---|---|---|
| `hush.py` | done | `python hush.py run --in 2 --out 4` |
| `launcher.py` | done | `python launcher.py` |
| `common.py` | done | shared helpers (16 kHz WAV I/O, SI-SDR) |
| `record.py` | stub (M1) | `python record.py --person mikolaj --session s1 --minutes 5` |
| `make_testset.py` | stub (M2) | `python make_testset.py --session s3 --n 50` |
| `mixer.py` | stub (M2) | imported by the notebook |
| `remove.py` | stub (M2) | `python remove.py --mix ... --enroll ... --ckpt ... --out ...` |
| `score.py` | stub (M2) | `python score.py --ckpt ... --name baseline` |
| `plots.py` | stub (M4) | `python plots.py` |
| `notebooks/finetune.ipynb` | to create (M3) | Colab, top to bottom |

Each stub's docstring is its spec.
