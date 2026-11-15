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
from tkinter import filedialog

import sounddevice as sd

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
        sidebar.grid_rowconfigure(11, weight=1)

        ctk.CTkLabel(sidebar, text="HUSH", font=ctk.CTkFont(size=28, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        ctk.CTkLabel(sidebar, text="mute one voice", text_color="gray60").grid(
            row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        ctk.CTkLabel(sidebar, text="Input device", text_color="gray60", font=ctk.CTkFont(size=11)).grid(
            row=2, column=0, padx=16, pady=(0, 0), sticky="w")
        self.input_var = ctk.StringVar(value="")
        self.input_menu = ctk.CTkOptionMenu(sidebar, variable=self.input_var, values=["(loading...)"])
        self.input_menu.grid(row=3, column=0, padx=16, pady=(0, 6), sticky="we")

        ctk.CTkLabel(sidebar, text="Output device", text_color="gray60", font=ctk.CTkFont(size=11)).grid(
            row=4, column=0, padx=16, pady=(0, 0), sticky="w")
        self.output_var = ctk.StringVar(value="")
        self.output_menu = ctk.CTkOptionMenu(sidebar, variable=self.output_var, values=["(loading...)"])
        self.output_menu.grid(row=5, column=0, padx=16, pady=(0, 6), sticky="we")

        ctk.CTkButton(sidebar, text="Refresh devices", height=26, font=ctk.CTkFont(size=11),
                      fg_color="gray30", hover_color="gray25",
                      command=self.refresh_devices).grid(row=6, column=0, padx=16, pady=(0, 14), sticky="we")

        ctk.CTkButton(sidebar, text="Go Live", height=48, font=ctk.CTkFont(size=16, weight="bold"),
                      command=self.run_live).grid(row=7, column=0, padx=16, pady=6, sticky="we")
        ctk.CTkButton(sidebar, text="Record & Remove Mikolaj", height=48,
                      font=ctk.CTkFont(size=14), command=self.run_message).grid(
            row=8, column=0, padx=16, pady=6, sticky="we")
        ctk.CTkButton(sidebar, text="Score & Compare", height=40,
                      font=ctk.CTkFont(size=13), fg_color="gray30", hover_color="gray25",
                      command=self.run_compare).grid(row=9, column=0, padx=16, pady=6, sticky="we")
        ctk.CTkButton(sidebar, text="Spray Paint Mikolaj", height=44,
                      font=ctk.CTkFont(size=14), fg_color="#7a2ff0", hover_color="#6020c8",
                      command=self.run_spray_paint).grid(row=10, column=0, padx=16, pady=6, sticky="we")

        self.stop_btn = ctk.CTkButton(sidebar, text="Stop", fg_color="#b03030",
                                       hover_color="#8a2424", command=self.stop_running, state="disabled")
        self.stop_btn.grid(row=12, column=0, padx=16, pady=(6, 20), sticky="swe")

        self.refresh_devices()

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

    def run_script(self, script, args=None, label=None, python=None, cwd=None):
        if self.proc is not None:
            self._append("\n[a script is already running - stop it first]\n")
            return
        args = args or []
        # -u: Python block-buffers stdout when it's piped (not a TTY), so without this
        # output can sit invisible for minutes on a long script - looks frozen even
        # though it's working, and someone hits Stop thinking it's hung.
        cmd = [python or PY, "-u", str(script), *args]
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
                    text=True, bufsize=1, cwd=str(cwd or HERE))
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

    # ------------------------------------------------------------- devices
    def refresh_devices(self):
        try:
            devices = sd.query_devices()
            default_in, default_out = sd.default.device
        except Exception as ex:  # noqa: BLE001
            self._append(f"\n[could not query audio devices: {ex}]\n")
            return

        in_opts, out_opts = [], []
        default_in_label = default_out_label = None
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                label = f"{i}: {d['name']}"
                in_opts.append(label)
                if i == default_in:
                    default_in_label = label
            if d["max_output_channels"] > 0:
                label = f"{i}: {d['name']}"
                out_opts.append(label)
                if i == default_out:
                    default_out_label = label

        self.input_menu.configure(values=in_opts or ["(none found)"])
        self.output_menu.configure(values=out_opts or ["(none found)"])
        # keep whatever the user already picked if it's still available - only fall
        # back to the system default if that device disappeared (e.g. unplugged)
        if self.input_var.get() not in in_opts:
            self.input_var.set(default_in_label or (in_opts[0] if in_opts else "(none found)"))
        if self.output_var.get() not in out_opts:
            self.output_var.set(default_out_label or (out_opts[0] if out_opts else "(none found)"))

    @staticmethod
    def _device_index(label):
        return label.split(":", 1)[0].strip()

    # ------------------------------------------------------------- actions
    def run_live(self):
        self.run_script(HERE / "hush.py", [
            "run", "--in", self._device_index(self.input_var.get()),
            "--out", self._device_index(self.output_var.get()),
        ], "Going live…")

    def run_message(self):
        self.run_script(HERE / "message.py", [
            "--seconds", "12", "--in", self._device_index(self.input_var.get()),
            "--out", self._device_index(self.output_var.get()),
        ], "Recording message…")

    def run_compare(self):
        self.run_script(HERE / "compare.py", [], "Scoring both pipelines on the test set…")

    def run_spray_paint(self):
        path = filedialog.askopenfilename(
            title="Pick a background image to spray-paint over",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        spray_dir = HERE / "spray_paint"
        spray_py = spray_dir / ".venv" / "bin" / "python"
        if not spray_py.exists():
            self._append(f"\n[spray_paint venv not found at {spray_py} - run spray_paint/setup.sh first]\n")
            return
        self.run_script(spray_dir / "spray_canvas.py", ["--background", path],
                        "Spray-painting…", python=str(spray_py), cwd=str(spray_dir))


if __name__ == "__main__":
    Launcher().mainloop()
