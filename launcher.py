#!/usr/bin/env python3
"""
HUSH Control Panel - one window to run every script in this repo.

  python launcher.py

Buttons on the left run the pipeline scripts as subprocesses (same as typing
the command yourself); their stdout/stderr streams into the panel on the
right. Stub scripts (record.py, make_testset.py, remove.py, score.py,
plots.py) will just print NotImplementedError until they're implemented -
that's expected, not a launcher bug.
"""
import queue
import subprocess
import sys
import threading
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog

HERE = Path(__file__).resolve().parent
PY = sys.executable


class Launcher(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("HUSH Control Panel")
        self.geometry("980x620")
        self.minsize(820, 520)

        self.proc = None
        self.out_q = queue.Queue()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=230, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(sidebar, text="HUSH", font=ctk.CTkFont(size=28, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        ctk.CTkLabel(sidebar, text="mute one voice", text_color="gray60").grid(
            row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        actions = [
            ("List devices", self.run_devices),
            ("Enroll a voice", self.open_enroll),
            ("Run live demo", self.run_live),
            ("Remove a voice", self.open_remove),
            ("Record audio", self.run_record),
            ("Make test set", self.run_make_testset),
            ("Score a model", self.run_score),
            ("Make plots", self.run_plots),
        ]
        for i, (label, cb) in enumerate(actions, start=2):
            ctk.CTkButton(sidebar, text=label, command=cb, anchor="w").grid(
                row=i, column=0, padx=16, pady=6, sticky="we")

        self.stop_btn = ctk.CTkButton(sidebar, text="Stop running script", fg_color="#b03030",
                                       hover_color="#8a2424", command=self.stop_running, state="disabled")
        self.stop_btn.grid(row=11, column=0, padx=16, pady=(6, 20), sticky="swe")

        main = ctk.CTkFrame(self, corner_radius=0)
        main.grid(row=0, column=1, sticky="nswe", padx=(0, 0))
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.status = ctk.CTkLabel(main, text="Idle", font=ctk.CTkFont(size=16, weight="bold"))
        self.status.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.output = ctk.CTkTextbox(main, font=ctk.CTkFont(family="Menlo", size=13))
        self.output.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nswe")
        self.output.configure(state="disabled")

        self.after(80, self._drain_queue)

    # ---------------------------------------------------------------- running
    def _append(self, text):
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _drain_queue(self):
        try:
            while True:
                line = self.out_q.get_nowait()
                if line is None:
                    self.status.configure(text="Idle")
                    self.stop_btn.configure(state="disabled")
                    self.proc = None
                else:
                    self._append(line)
        except queue.Empty:
            pass
        self.after(80, self._drain_queue)

    def run_script(self, script, args=None, label=None):
        if self.proc is not None:
            self._append("\n[a script is already running - stop it first]\n")
            return
        args = args or []
        cmd = [PY, str(HERE / script), *args]
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self._append(f"$ {' '.join(cmd)}\n\n")
        self.status.configure(text=label or f"Running {script}…")
        self.stop_btn.configure(state="normal")

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(HERE))
                for line in self.proc.stdout:
                    self.out_q.put(line)
                self.proc.wait()
                self.out_q.put(f"\n[exited with code {self.proc.returncode}]\n")
            except Exception as ex:  # noqa: BLE001
                self.out_q.put(f"\n[launcher error: {ex}]\n")
            finally:
                self.out_q.put(None)

        threading.Thread(target=worker, daemon=True).start()

    def stop_running(self):
        if self.proc is not None:
            self.proc.terminate()
            self._append("\n[stopped]\n")

    # ------------------------------------------------------------- shortcuts
    def run_devices(self):
        self.run_script("hush.py", ["devices"], "Listing devices…")

    def run_live(self):
        in_dialog = ctk.CTkInputDialog(
            text="Input device number (see 'List devices'):", title="Run live demo")
        inp = in_dialog.get_input()
        if not inp or not inp.strip():
            return
        out_dialog = ctk.CTkInputDialog(
            text="Output device number (needs 2 channels, e.g. headphones):", title="Run live demo")
        out = out_dialog.get_input()
        if not out or not out.strip():
            return
        self.run_script("hush.py", ["run", "--in", inp.strip(), "--out", out.strip()],
                         "Running live demo…")

    def run_record(self):
        self.run_script("record.py", [], "Recording…")

    def run_make_testset(self):
        self.run_script("make_testset.py", ["--selftest"], "Building test set (selftest)…")

    def run_score(self):
        self.run_script("score.py", ["--selftest"], "Scoring (selftest)…")

    def run_plots(self):
        self.run_script("plots.py", [], "Making plots…")

    def open_remove(self):
        mix = filedialog.askopenfilename(
            title="Pick the mix (overlapping speech) WAV", filetypes=[("WAV", "*.wav")])
        if not mix:
            return
        enroll = filedialog.askopenfilename(
            title="Pick an enrollment WAV of the voice to remove", filetypes=[("WAV", "*.wav")])
        if not enroll:
            return
        alpha_dialog = ctk.CTkInputDialog(
            text="Removal strength --alpha (1.0 = default, try 1.1-1.3 for stronger):",
            title="Remove a voice")
        alpha = alpha_dialog.get_input()
        alpha = (alpha or "1.0").strip() or "1.0"
        out_name = Path(mix).stem.replace(" ", "_") + "_removed.wav"
        target_name = Path(mix).stem.replace(" ", "_") + "_target.wav"
        self.run_script("remove.py", [
            "--mix", mix, "--enroll", enroll,
            "--ckpt", "checkpoints/pretrained/spk_emb_100",
            "--alpha", alpha,
            "--out", f"results/{out_name}",
            "--save-target", f"results/{target_name}",
        ], "Removing voice…")

    def open_enroll(self):
        dialog = ctk.CTkInputDialog(text="Name to enroll (e.g. Mikolaj):", title="Enroll a voice")
        name = dialog.get_input()
        if not name or not name.strip():
            return
        name = name.strip()
        self.run_script("hush.py", ["enroll", name], f"Enrolling {name}…")


if __name__ == "__main__":
    Launcher().mainloop()
