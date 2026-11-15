#!/usr/bin/env python3
"""
HUSH Control Panel.

  python launcher.py

"Go Live" runs hush.py's live mute-by-voice demo (ducks the whole channel
when Mikolaj is detected). "Record & Remove Mikolaj" records a short message
and plays back a version with just his voice removed, via message.py (SepFormer
separation + ECAPA identification). Both run as subprocesses; their output
streams into the panel on the right.
"""
import queue
import subprocess
import sys
import threading
from pathlib import Path

import customtkinter as ctk

HERE = Path(__file__).resolve().parent
PY = sys.executable


class Launcher(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("HUSH Control Panel")
        self.geometry("900x560")
        self.minsize(760, 460)

        self.proc = None
        self.out_q = queue.Queue()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=230, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(sidebar, text="HUSH", font=ctk.CTkFont(size=28, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        ctk.CTkLabel(sidebar, text="mute one voice", text_color="gray60").grid(
            row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        ctk.CTkButton(sidebar, text="Go Live", height=48, font=ctk.CTkFont(size=16, weight="bold"),
                      command=self.run_live).grid(row=2, column=0, padx=16, pady=6, sticky="we")
        ctk.CTkButton(sidebar, text="Record & Remove Mikolaj", height=48,
                      font=ctk.CTkFont(size=14), command=self.run_message).grid(
            row=3, column=0, padx=16, pady=6, sticky="we")
        ctk.CTkButton(sidebar, text="Score & Compare", height=40,
                      font=ctk.CTkFont(size=13), fg_color="gray30", hover_color="gray25",
                      command=self.run_compare).grid(row=4, column=0, padx=16, pady=6, sticky="we")

        self.stop_btn = ctk.CTkButton(sidebar, text="Stop", fg_color="#b03030",
                                       hover_color="#8a2424", command=self.stop_running, state="disabled")
        self.stop_btn.grid(row=6, column=0, padx=16, pady=(6, 20), sticky="swe")

        main = ctk.CTkFrame(self, corner_radius=0)
        main.grid(row=0, column=1, sticky="nswe")
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
        # -u: Python block-buffers stdout when it's piped (not a TTY), so without this
        # output can sit invisible for minutes on a long script - looks frozen even
        # though it's working, and someone hits Stop thinking it's hung.
        cmd = [PY, "-u", str(HERE / script), *args]
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

    # ------------------------------------------------------------- actions
    def run_live(self):
        self.run_script("hush.py", ["run"], "Going live…")

    def run_message(self):
        self.run_script("message.py", ["--seconds", "12"], "Recording message…")

    def run_compare(self):
        self.run_script("compare.py", [], "Scoring both pipelines on the test set…")


if __name__ == "__main__":
    Launcher().mainloop()
