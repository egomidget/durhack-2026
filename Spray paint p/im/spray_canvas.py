"""Finger-Spray Composition Canvas: touchless projected painting and music."""
from __future__ import annotations

import argparse
import math
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MODEL_PATH = ROOT / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
W, H = 1920, 1080
PIP_W, PIP_H, MARGIN = 320, 180, 20
BG = (10, 10, 15)
PALETTES = [("NEON MAGENTA", (210, 50, 255)), ("ELECTRIC CYAN", (255, 230, 0)),
            ("ACID LIME", (50, 255, 130)), ("SOLAR ORANGE", (0, 140, 255)),
            ("ULTRAVIOLET", (255, 70, 130))]


def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return
    print("Downloading MediaPipe hand model (~12 MB)...")
    temp = MODEL_PATH.with_suffix(".download")
    try:
        urllib.request.urlretrieve(MODEL_URL, temp)
        if temp.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small")
        temp.replace(MODEL_PATH)
    finally:
        temp.unlink(missing_ok=True)


class ThreadedCamera:
    """Latest-frame DirectShow capture; old frames are discarded to limit lag."""
    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open camera {index}. Try --camera 1 or check Windows camera permissions.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, raw = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = cv2.flip(raw, 1)
            else:
                time.sleep(.01)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def telemetry(lm, world):
    palm = math.hypot(lm[9].x-lm[0].x, lm[9].y-lm[0].y)
    pitch = math.degrees(math.atan2(world[9].z-world[0].z, abs(world[9].y-world[0].y)+1e-6))
    roll = math.degrees(math.atan2(lm[17].y-lm[5].y, lm[17].x-lm[5].x))
    if palm < .08: msg = "TOO FAR - STEP CLOSER"
    elif palm > .22: msg = "TOO CLOSE - STEP BACK"
    elif abs(pitch) > 20: msg = f"HAND ANGLE: {pitch:+.0f} deg"
    elif abs(roll) > 12: msg = f"HAND ROLL: {roll:+.0f} deg"
    elif lm[8].x < .10: msg = "EDGE: MOVE RIGHT"
    elif lm[8].x > .90: msg = "EDGE: MOVE LEFT"
    elif lm[8].y < .10: msg = "EDGE: MOVE DOWN"
    elif lm[8].y > .90: msg = "EDGE: MOVE UP"
    else: msg = "OPTIMAL RANGE & ANGLE"
    return msg, palm


class Audio:
    """Loads supplied loops when present; otherwise runs silently with HUD notice."""
    FILES = ("hiss.wav", "bass.wav", "pad.wav", "arp.wav")
    def __init__(self):
        self.muted = False
        self.enabled = False
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            paths = [ASSETS / n for n in self.FILES]
            if not all(p.exists() for p in paths):
                return
            self.channels = []
            for i, p in enumerate(paths):
                ch = pygame.mixer.Channel(i)
                ch.set_volume(0)
                ch.play(pygame.mixer.Sound(str(p)), loops=-1)
                self.channels.append(ch)
            self.enabled = True
        except Exception as exc:
            print(f"Audio disabled: {exc}")

    def update(self, spraying, y, speed):
        if not self.enabled: return
        elev = 1.0 - float(np.clip(y, 0, 1))
        vals = [0.8 if spraying else 0.0,
                float(np.clip((1.1-elev)*.85, .05, .9)) if spraying else .12,
                float(np.clip(1-abs(elev-.5)*2, .15, .85)),
                float(np.clip((elev-.35)*1.4+min(speed/45, .35), 0, .9)) if spraying else .08]
        for ch, v in zip(self.channels, vals): ch.set_volume(0 if self.muted else v)


def run(args):
    ensure_model()
    opts = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=.60, min_tracking_confidence=.60)
    detector = vision.HandLandmarker.create_from_options(opts)
    cam = ThreadedCamera(args.camera, args.camera_width, args.camera_height)
    pygame.init()
    audio = Audio()
    paint = np.full((H, W, 3), BG, np.uint8)
    saturation = np.zeros((H//8, W//8), np.float32)
    drips, ema, spray_state = [], {}, {}
    palette, started = 0, time.perf_counter()
    window = "Finger-Spray Composition Canvas"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(.005)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((time.perf_counter()-started)*1000))
            out = paint.copy()
            pip = cv2.resize(frame, (PIP_W, PIP_H), interpolation=cv2.INTER_AREA)
            msg, hud = "RAISE HAND IN FRONT OF CAMERA", (140, 140, 150)
            spraying_any, lead_y, lead_speed = False, .5, 0.
            seen = set()
            if result.hand_landmarks:
                for i, (lm, world) in enumerate(zip(result.hand_landmarks, result.hand_world_landmarks)):
                    seen.add(i)
                    _, color = PALETTES[(palette+i) % len(PALETTES)]
                    palm_ref = math.hypot(lm[5].x-lm[0].x, lm[5].y-lm[0].y)+1e-6
                    ratio = math.hypot(lm[8].x-lm[4].x, lm[8].y-lm[4].y)/palm_ref
                    spraying = ratio < .28 if ratio < .28 else (False if ratio > .34 else spray_state.get(i, False))
                    spray_state[i] = spraying
                    raw = (lm[8].x*W, lm[8].y*H)
                    prev = ema.get(i, raw)
                    cx, cy = .35*raw[0]+.65*prev[0], .35*raw[1]+.65*prev[1]
                    ema[i] = (cx, cy)
                    speed = math.hypot(cx-prev[0], cy-prev[1])
                    if i == 0:
                        msg, palm = telemetry(lm, world)
                        hud = (60,235,110) if .08 <= palm <= .22 and msg == "OPTIMAL RANGE & ANGLE" else (0,165,255)
                        lead_y, lead_speed = lm[8].y, speed
                    if spraying:
                        spraying_any = True
                        angles = np.random.uniform(0, 2*np.pi, 65)
                        radii = np.random.exponential(16, 65)
                        xs = np.clip((cx+radii*np.cos(angles)).astype(np.int32), 0, W-1)
                        ys = np.clip((cy+radii*np.sin(angles)).astype(np.int32), 0, H-1)
                        paint[ys, xs] = color
                        gx, gy = int(np.clip(cx//8, 0, saturation.shape[1]-1)), int(np.clip(cy//8, 0, saturation.shape[0]-1))
                        saturation[gy, gx] += 1
                        if saturation[gy,gx] > 18 and len(drips) < 250:
                            saturation[gy,gx] = 0
                            drips.append([cx+np.random.uniform(-6,6), cy, np.random.uniform(1.8,3.8), color])
                    cv2.circle(out, (int(cx),int(cy)), 14 if spraying else 18, color if spraying else (220,220,230), 2, cv2.LINE_AA)
                    for p in lm: cv2.circle(pip, (int(p.x*PIP_W),int(p.y*PIP_H)), 2, (0,255,255), -1)
                    cv2.line(pip, (int(lm[4].x*PIP_W),int(lm[4].y*PIP_H)), (int(lm[8].x*PIP_W),int(lm[8].y*PIP_H)), color if spraying else (255,255,255), 2)
            for i in set(spray_state)-seen: spray_state[i] = False
            alive = []
            for d in drips:
                d[1] += d[2]; d[2] *= .985
                if d[1] < H-2 and d[2] > .15:
                    cv2.circle(paint, (int(d[0]),int(d[1])), 2, d[3], -1); alive.append(d)
            drips = alive
            saturation *= .94
            audio.update(spraying_any, lead_y, lead_speed)
            border = (255,230,0) if spraying_any else hud
            cv2.rectangle(pip,(0,0),(PIP_W,24),(15,23,42),-1)
            cv2.putText(pip,msg,(6,16),cv2.FONT_HERSHEY_SIMPLEX,.40,border,1,cv2.LINE_AA)
            label = PALETTES[palette][0] + (" | AUDIO OFF" if not audio.enabled else (" | MUTED" if audio.muted else ""))
            cv2.putText(pip,label,(6,PIP_H-8),cv2.FONT_HERSHEY_SIMPLEX,.38,(220,220,220),1,cv2.LINE_AA)
            cv2.rectangle(pip,(0,0),(PIP_W-1,PIP_H-1),border,2)
            out[H-PIP_H-MARGIN:H-MARGIN,W-PIP_W-MARGIN:W-MARGIN] = pip
            cv2.imshow(window,out)
            key = cv2.waitKey(1)&0xff
            if key in (ord('q'),27): break
            if key == ord('c'):
                paint[:] = BG; saturation[:] = 0; drips.clear()
            elif key == ord('m'): audio.muted = not audio.muted
            elif ord('1') <= key <= ord('5'): palette = key-ord('1')
    finally:
        cam.release(); detector.close(); pygame.quit(); cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="DirectShow camera index (default: 0)")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    args = parser.parse_args()
    try: run(args)
    except KeyboardInterrupt: pass
    except Exception as exc:
        raise SystemExit(f"Finger-Spray could not start: {exc}\nSee README.md for setup and troubleshooting.")

if __name__ == "__main__": main()
