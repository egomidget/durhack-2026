#!/usr/bin/env python3
"""
HUSH - selectively mute people's voices in your headphones.

  python hush.py devices                         # list mics / outputs
  python hush.py enroll Mikolaj --alias michael  # record a voiceprint
  python hush.py run --in 2 --out 4              # start the demo window

Voice commands (spoken into the mic):
  "hush mute michael"   "hush unmute michael"
  "hush mute everyone"  "hush unmute everyone"
Backup: click the buttons, or press 1-9 to toggle person 1-9, 0 = unmute all.

Pipeline: mic -> 1 s rolling window -> speaker embedding (Resemblyzer)
-> cosine similarity vs enrolled voiceprints -> if the best match is muted,
duck the (slightly delayed) audio going to the headphones.
"""
import argparse
import json
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
VOICES = HERE / "voices"
AN_SR = 16000  # sample rate the ML models want


# ---------------------------------------------------------------- audio utils
def to_16k(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if int(sr) == AN_SR:
        return x
    g = np.gcd(int(sr), AN_SR)
    return resample_poly(x, AN_SR // g, int(sr) // g).astype(np.float32)


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


# ------------------------------------------------------- speaker embeddings
_encoder = None
_enc_lock = threading.Lock()


def get_encoder():
    global _encoder
    with _enc_lock:
        if _encoder is None:
            from resemblyzer import VoiceEncoder
            _encoder = VoiceEncoder("cpu", verbose=False)
    return _encoder


def embed_live(wav16):
    """Embedding of a short live window (volume-normalised, no trimming)."""
    wav16 = wav16 * (10 ** (-30 / 20) / (np.sqrt(np.mean(wav16 ** 2)) + 1e-9))
    enc = get_encoder()
    with _enc_lock:
        return enc.embed_utterance(wav16.astype(np.float32))


def embed_enroll(wav16):
    """Embedding of an enrollment recording (silences trimmed)."""
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(wav16, source_sr=AN_SR)
    if len(wav) < AN_SR * 3:
        raise RuntimeError("Not enough speech captured - talk for longer or get closer to the mic.")
    enc = get_encoder()
    with _enc_lock:
        return enc.embed_utterance(wav)


# ------------------------------------------------------------- voice store
def _meta():
    p = VOICES / "people.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_people():
    people = {}
    for name, info in _meta().items():
        f = VOICES / f"{name}.npy"
        if f.exists():
            people[name] = {"alias": info.get("alias", name.lower()), "emb": np.load(f)}
    return people


def save_person(name, alias, emb, wav16=None):
    VOICES.mkdir(exist_ok=True)
    np.save(VOICES / f"{name}.npy", emb)
    if wav16 is not None:
        wavfile.write(VOICES / f"{name}.wav", AN_SR,
                      (np.clip(wav16, -1, 1) * 32767).astype(np.int16))
    meta = _meta()
    meta[name] = {"alias": alias}
    (VOICES / "people.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def missing_vocab(model_path, words):
    """Words the Vosk model doesn't know (they can't be recognised)."""
    try:
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        m = Model(str(model_path))
        return [w for w in words if m.find_word(w) < 0]
    except Exception:
        return []


# -------------------------------------------------------------- the engine
class Hush:
    def __init__(self, args):
        self.args = args
        self.sr = args.rate
        self.lock = threading.Lock()
        self.hist_lock = threading.Lock()
        self.people = load_people()
        self.muted = set()
        self.scores = {}
        self.current = None          # (name, score) of best match above threshold
        self.level = -100.0
        self.threshold = args.threshold
        self.target_gain = 1.0
        self.gain = 1.0
        self.hold_until = 0.0
        self.last_cmd = ""
        self.names_version = 0
        self.recording = None
        self.running = True
        self.hist = np.zeros(int(self.sr * max(2.0, args.window + 0.5)), np.float32)
        self.delay_buf = np.zeros(int(self.sr * args.delay), np.float32)
        self.asr_q = queue.Queue(maxsize=500)
        self._pink_state = np.zeros(7, np.float32)  # Paul Kellet pink noise filter

    # ---- soft masking noise instead of dead silence (docs/PLAN.md: "masks with
    # shaped noise") - cheap pink noise via a one-pole-per-octave filter bank ----
    def _pink_noise(self, n):
        white = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
        b = self._pink_state
        out = np.empty(n, np.float32)
        for i in range(n):
            w = white[i]
            b[0] = 0.99886 * b[0] + w * 0.0555179
            b[1] = 0.99332 * b[1] + w * 0.0750759
            b[2] = 0.96900 * b[2] + w * 0.1538520
            b[3] = 0.86650 * b[3] + w * 0.3104856
            b[4] = 0.55000 * b[4] + w * 0.5329522
            b[5] = -0.7616 * b[5] - w * 0.0168980
            out[i] = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362
            b[6] = w * 0.115926
        return out * 0.11  # roughly normalise to [-1, 1]-ish

    # ---- realtime audio callback (keep it cheap) ----
    def callback(self, indata, outdata, frames, time_info, status):
        x = indata[:, 0].astype(np.float32, copy=True)
        n = len(x)
        with self.hist_lock:
            self.hist[:-n] = self.hist[n:]
            self.hist[-n:] = x
        rec = self.recording
        if rec is not None:
            rec.append(x)
        if not self.args.no_voice:
            try:
                self.asr_q.put_nowait(x)
            except queue.Full:
                pass
        # delay line: gives the detector a head start on what the judges hear
        if len(self.delay_buf):
            buf = np.concatenate((self.delay_buf, x))
            out, self.delay_buf = buf[:n], buf[n:]
        else:
            out = x
        # smooth gain ramp (fast duck, slower release) to avoid clicks
        g0, tgt = self.gain, self.target_gain
        if tgt < g0:
            g1 = max(tgt, g0 - n / (self.args.attack * self.sr))
        else:
            g1 = min(tgt, g0 + n / (self.args.release * self.sr))
        self.gain = g1
        ramp = np.linspace(g0, g1, n, dtype=np.float32)
        mixed = out * ramp
        if self.args.mask_level > 0:
            # crossfade the ducked voice into soft noise instead of dead silence -
            # (1 - ramp) is how "muted" we currently are, 0 when fully audible
            mixed = mixed + self._pink_noise(n) * (1.0 - ramp) * self.args.mask_level
        outdata[:] = (mixed * self.args.volume)[:, None]

    # ---- who is talking? ----
    def analyze_loop(self):
        a = self.args
        win_n = int(self.sr * a.window)
        gate_n = int(self.sr * 0.3)
        while self.running:
            t0 = time.time()
            with self.hist_lock:
                win = self.hist[-win_n:].copy()
            self.level = rms_db(win[-gate_n:])
            with self.lock:
                people = dict(self.people)
                muted = set(self.muted)
            thr = self.threshold
            scores, cur = {}, None
            if self.level > a.gate and people:
                e = embed_live(to_16k(win, self.sr))
                scores = {n: float(np.dot(e, p["emb"])) for n, p in people.items()}
                best = max(scores, key=scores.get)
                if scores[best] >= thr:
                    cur = (best, scores[best])
                    if best in muted:
                        self.hold_until = time.time() + a.hold
            self.scores, self.current = scores, cur
            self.target_gain = a.floor if time.time() < self.hold_until else 1.0
            time.sleep(max(0.0, a.hop - (time.time() - t0)))

    def set_muted(self, name, on):
        with self.lock:
            for n in (list(self.people) if name == "*" else [name]):
                if on:
                    self.muted.add(n)
                else:
                    self.muted.discard(n)
        if not on:
            self.hold_until = 0.0

    # ---- voice commands (Vosk, offline, restricted grammar) ----
    def asr_loop(self):
        a = self.args
        try:
            from vosk import Model, KaldiRecognizer, SetLogLevel
        except ImportError:
            self.last_cmd = "Voice commands off (pip install vosk) - use buttons / keys 1-9"
            return
        if not Path(a.vosk_model).exists():
            self.last_cmd = f"Voice commands off: Vosk model not found at {a.vosk_model}"
            return
        SetLogLevel(-1)
        model = Model(str(a.vosk_model))
        rec, ver = None, -1
        chunk, have, chunk_n = [], 0, int(self.sr * 0.1)
        while self.running:
            try:
                x = self.asr_q.get(timeout=0.5)
            except queue.Empty:
                continue
            chunk.append(x)
            have += len(x)
            if have < chunk_n:
                continue
            x = np.concatenate(chunk)
            chunk, have = [], 0
            if rec is None or ver != self.names_version:
                ver = self.names_version
                rec = self._recognizer(model, KaldiRecognizer)
            pcm = (np.clip(to_16k(x, self.sr), -1, 1) * 32767).astype(np.int16).tobytes()
            if rec.AcceptWaveform(pcm):
                self._handle(json.loads(rec.Result()))

    def _recognizer(self, model, KaldiRecognizer):
        w = self.args.wake
        with self.lock:
            aliases = sorted({p["alias"] for p in self.people.values()})
        unknown = [x for x in [w, "mute", "unmute", "everyone"] + aliases if model.find_word(x) < 0]
        if unknown:
            self.last_cmd = f"Speech model doesn't know: {', '.join(unknown)} - pick other alias words"
        phrases = [f"{w} {v} {t}" for v in ("mute", "unmute") for t in aliases + ["everyone"]]
        rec = KaldiRecognizer(model, AN_SR, json.dumps(phrases + ["[unk]"]))
        rec.SetWords(True)
        return rec

    def _handle(self, res):
        text = res.get("text", "").strip()
        if not text or "[unk]" in text:
            return
        m = re.fullmatch(rf"{re.escape(self.args.wake)} (mute|unmute) (\S+)", text)
        if not m:
            return
        conf = min((wd.get("conf", 0.0) for wd in res.get("result", [])), default=0.0)
        if conf < self.args.min_conf:
            self.last_cmd = f'ignored (unsure {conf:.2f}): "{text}"'
            return
        verb, who = m.groups()
        if who == "everyone":
            name = "*"
        else:
            with self.lock:
                name = next((n for n, p in self.people.items() if p["alias"] == who), None)
            if name is None:
                return
        self.set_muted(name, verb == "mute")
        self.last_cmd = f'heard: "{text}"  ({conf:.2f})'

    # ---- enroll a new voice while running ----
    def enroll_live(self, name, alias, seconds, done):
        self.recording = []

        def worker():
            time.sleep(seconds)
            chunks, self.recording = self.recording, None
            try:
                wav16 = to_16k(np.concatenate(chunks), self.sr)
                emb = embed_enroll(wav16)
                save_person(name, alias, emb, wav16)
                with self.lock:
                    self.people[name] = {"alias": alias, "emb": emb}
                    self.names_version += 1
                done(None)
            except Exception as ex:  # noqa: BLE001
                done(str(ex))

        threading.Thread(target=worker, daemon=True).start()


# ------------------------------------------------------------------ the GUI
def run_gui(h):
    import tkinter as tk
    from tkinter import simpledialog

    BG, PANEL, FG, DIM = "#101317", "#1d2229", "#f2f2f2", "#7d8590"
    RED, GREEN, AMBER = "#ff4d4f", "#3ddc84", "#f5a623"
    BAR_W, BAR_H = 380, 28
    LO = 0.5  # similarity shown at the left edge of the bars

    root = tk.Tk()
    root.title("HUSH")
    root.configure(bg=BG)
    root.geometry("820x600")

    tk.Label(root, text="HUSH", font=("Helvetica", 46, "bold"), fg=FG, bg=BG).pack(pady=(14, 0))
    status = tk.Label(root, font=("Helvetica", 24, "bold"), bg=BG)
    status.pack()
    hearing = tk.Label(root, font=("Helvetica", 18), fg=DIM, bg=BG)
    hearing.pack()
    cmd = tk.Label(root, font=("Helvetica", 13), fg=DIM, bg=BG)
    cmd.pack(pady=(2, 10))
    rows_frame = tk.Frame(root, bg=BG)
    rows_frame.pack(fill="x", padx=28)
    rows = {}
    ui = {"enroll_until": 0.0, "rebuild": False}

    def toggle(name):
        h.set_muted(name, name not in h.muted)

    def build_rows():
        for w in rows_frame.winfo_children():
            w.destroy()
        rows.clear()
        if not h.people:
            tk.Label(rows_frame, text="No voices yet - click '+ Enroll voice'",
                     font=("Helvetica", 16), fg=DIM, bg=BG).pack(pady=20)
        for i, name in enumerate(sorted(h.people), 1):
            f = tk.Frame(rows_frame, bg=BG)
            f.pack(fill="x", pady=6)
            lbl = tk.Label(f, text=f"{i}  {name}", font=("Helvetica", 18), fg=FG, bg=BG, width=13, anchor="w")
            lbl.pack(side="left")
            c = tk.Canvas(f, width=BAR_W, height=BAR_H, bg=PANEL, highlightthickness=0)
            c.pack(side="left", padx=10)
            bar = c.create_rectangle(0, 0, 0, BAR_H, fill=DIM, width=0)
            tl = c.create_line(0, 0, 0, BAR_H, fill=FG, width=2)
            b = tk.Button(f, width=8, font=("Helvetica", 13), command=lambda n=name: toggle(n))
            b.pack(side="left")
            sub = tk.Label(f, text=f'say "{h.people[name]["alias"]}"', font=("Helvetica", 11), fg=DIM, bg=BG)
            sub.pack(side="left", padx=8)
            rows[name] = (lbl, c, bar, tl, b)

    def enroll():
        name = simpledialog.askstring("Enroll voice", "Name to display:", parent=root)
        if not name or not name.strip():
            return
        name = name.strip()
        alias = name.lower()
        secs = h.args.enroll_seconds
        ui["enroll_until"] = time.time() + secs

        def done(err):
            h.last_cmd = f"Enroll failed: {err}" if err else f"Enrolled {name}"
            ui["rebuild"] = True

        h.enroll_live(name, alias, secs, done)

    ctl = tk.Frame(root, bg=BG)
    ctl.pack(side="bottom", fill="x", padx=28, pady=14)
    thr = tk.Scale(ctl, from_=0.55, to=0.95, resolution=0.01, orient="horizontal",
                   label="match threshold", length=240, bg=BG, fg=FG, troughcolor=PANEL,
                   highlightthickness=0, command=lambda v: setattr(h, "threshold", float(v)))
    thr.set(h.threshold)
    thr.pack(side="left")
    mask = tk.Scale(ctl, from_=0.0, to=0.6, resolution=0.01, orient="horizontal",
                    label="interference level", length=200, bg=BG, fg=FG, troughcolor=PANEL,
                    highlightthickness=0, command=lambda v: setattr(h.args, "mask_level", float(v)))
    mask.set(h.args.mask_level)
    mask.pack(side="left", padx=(12, 0))
    tk.Button(ctl, text="Mute all", command=lambda: h.set_muted("*", True)).pack(side="left", padx=8)
    tk.Button(ctl, text="Unmute all", command=lambda: h.set_muted("*", False)).pack(side="left")
    tk.Button(ctl, text="+ Enroll voice", command=enroll).pack(side="right")
    level = tk.Label(ctl, font=("Helvetica", 11), fg=DIM, bg=BG)
    level.pack(side="right", padx=12)

    def on_key(e):
        if e.char == "0":
            h.set_muted("*", False)
        elif e.char.isdigit():
            names = sorted(h.people)
            i = int(e.char) - 1
            if i < len(names):
                toggle(names[i])
    root.bind("<Key>", on_key)

    def tick():
        if ui["rebuild"]:
            ui["rebuild"] = False
            build_rows()
        left = ui["enroll_until"] - time.time()
        if left > 0:
            status.config(text=f"● RECORDING VOICE… {left:.0f}s  (keep talking)", fg=AMBER)
        elif h.gain < 0.5:
            status.config(text="● MUTING", fg=RED)
        else:
            status.config(text="● LISTENING", fg=GREEN)
        cur = h.current
        if cur:
            hearing.config(text=f"hearing: {cur[0]}  ({cur[1]:.2f})")
        elif h.level > h.args.gate:
            hearing.config(text="hearing: someone not enrolled")
        else:
            hearing.config(text="hearing: —")
        cmd.config(text=h.last_cmd)
        level.config(text=f"mic {h.level:5.1f} dB  (gate {h.args.gate:.0f})")
        tx = (h.threshold - LO) / (1 - LO) * BAR_W
        for name, (lbl, c, bar, tl, b) in rows.items():
            s = h.scores.get(name, 0.0)
            muted = name in h.muted
            frac = min(max((s - LO) / (1 - LO), 0.0), 1.0)
            hit = cur is not None and cur[0] == name
            col = (RED if muted else GREEN) if hit else DIM
            c.coords(bar, 0, 0, frac * BAR_W, BAR_H)
            c.itemconfig(bar, fill=col)
            c.coords(tl, tx, 0, tx, BAR_H)
            b.config(text="MUTED" if muted else "live")
            lbl.config(fg=RED if muted else FG)
        root.after(80, tick)

    build_rows()
    tick()
    root.mainloop()


# --------------------------------------------------------------- commands
def dev(x):
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        return x  # sounddevice accepts a name substring


def cmd_devices(a):
    import sounddevice as sd
    print(sd.query_devices())
    print("\nUse the number on the left with --in / --out (> = default input, < = default output).")


def cmd_enroll(a):
    import sounddevice as sd
    if not a.name:
        raise SystemExit("usage: python hush.py enroll NAME [--alias word]")
    alias = (a.alias or a.name).strip().lower()
    print("Loading voice model…")
    get_encoder()
    print(f"\nRecording {a.name} for {a.seconds:.0f}s - talk naturally "
          "(describe your morning, read something aloud). Starting now!")
    rec = sd.rec(int(a.seconds * a.rate), samplerate=a.rate, channels=1, device=dev(a.inp), dtype="float32")
    for s in range(int(a.seconds), 0, -1):
        print(f"  {s:2d}s left", end="\r", flush=True)
        time.sleep(1)
    sd.wait()
    wav16 = to_16k(rec[:, 0], a.rate)
    if rms_db(wav16) < -50:
        print("\nWarning: that was very quiet - check the mic (--in) and try again.")
    emb = embed_enroll(wav16)
    save_person(a.name, alias, emb, wav16)
    print(f"\nSaved voiceprint for {a.name}. Command: \"{a.wake} mute {alias}\"")
    bad = missing_vocab(a.vosk_model, [a.wake, alias])
    if bad:
        print(f"Heads up: the speech model doesn't know {bad} - re-enroll with a different --alias.")


def cmd_run(a):
    import sounddevice as sd
    h = Hush(a)
    print("Loading voice model…")
    get_encoder()
    print(f"Enrolled: {', '.join(h.people) or '(none yet)'}")
    stream = sd.Stream(samplerate=a.rate, blocksize=int(a.rate * a.block_ms / 1000),
                       device=(dev(a.inp), dev(a.out)), channels=(1, a.out_channels),
                       dtype="float32", latency="low", callback=h.callback)
    threading.Thread(target=h.analyze_loop, daemon=True).start()
    if not a.no_voice:
        threading.Thread(target=h.asr_loop, daemon=True).start()
    with stream:
        try:
            run_gui(h)
        finally:
            h.running = False


def main():
    p = argparse.ArgumentParser(description="HUSH - selective voice muting")
    p.add_argument("cmd", choices=["run", "enroll", "devices"])
    p.add_argument("name", nargs="?", help="person's name (enroll)")
    p.add_argument("--alias", help="word you say in commands for this person")
    p.add_argument("--seconds", type=float, default=20, help="enroll recording length")
    p.add_argument("--enroll-seconds", type=float, default=15, help="enroll length from the GUI")
    p.add_argument("--in", dest="inp", help="input device number or name")
    p.add_argument("--out", help="output device number or name")
    p.add_argument("--rate", type=int, default=48000)
    p.add_argument("--out-channels", type=int, default=2)
    p.add_argument("--block-ms", type=float, default=10)
    p.add_argument("--delay", type=float, default=0.45,
                   help="seconds of delay before headphones (gives detector a head start - "
                        "a bit more than before trades a small lag for fewer false triggers)")
    p.add_argument("--window", type=float, default=1.0, help="analysis window (s)")
    p.add_argument("--hop", type=float, default=0.15, help="how often to analyse (s)")
    p.add_argument("--hold", type=float, default=0.8, help="stay muted this long after last detection")
    p.add_argument("--threshold", type=float, default=0.75, help="voice match threshold")
    p.add_argument("--gate", type=float, default=-45, help="mic level (dBFS) counted as speech")
    p.add_argument("--floor", type=float, default=0.0, help="gain when muted (0 = silent)")
    p.add_argument("--mask-level", type=float, default=0.3, dest="mask_level",
                   help="soft pink-noise 'interference' level to crossfade in while muted, "
                        "instead of dead silence (0 = silent like before, try 0.2-0.4)")
    p.add_argument("--attack", type=float, default=0.04)
    p.add_argument("--release", type=float, default=0.3)
    p.add_argument("--volume", type=float, default=1.0, help="headphone gain")
    p.add_argument("--wake", default="hush", help="wake word for voice commands")
    p.add_argument("--vosk-model", default=str(HERE / "vosk-model-small-en-us-0.15"))
    p.add_argument("--min-conf", type=float, default=0.6, help="min speech confidence for commands")
    p.add_argument("--no-voice", action="store_true", help="disable voice commands")
    a = p.parse_args()
    {"run": cmd_run, "enroll": cmd_enroll, "devices": cmd_devices}[a.cmd](a)


if __name__ == "__main__":
    main()
