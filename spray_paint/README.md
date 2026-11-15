# Finger-Spray Composition Canvas

A Windows touchless painting and music installation. A mirrored USB webcam feed is tracked with MediaPipe; pinching thumb and index finger sprays persistent neon paint onto a 1080p canvas. Two hands can paint in different palette colors. A camera inset shows tracking and framing prompts, and four audio loops respond to pinch, height, and movement.

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.10, 3.11, or 3.12 (64-bit)
- Webcam; projector or second display is optional for trying it out
- **First launch fails to fetch the model:** connect to the internet and run `run.ps1`, or download the official model from the URL in `spray_canvas.py` to `hand_landmarker.task` in this folder.
- Speakers/headphones for audio

## Install

Open PowerShell in this project folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

The script creates `.venv`, installs dependencies, and generates four original WAV loops under `assets/`. The hand model downloads on first launch into `hand_landmarker.task`.

## Run

```powershell
.\run.ps1
```

To choose a different camera:

```powershell
.\run.ps1 --camera 1
```

The application opens a fullscreen 1920x1080 window. For a projector, use **Win+P > Extend**, then move the canvas window to the projector display. The app requests fullscreen at startup.

## Controls

| Key | Action |
| --- | --- |
| Pinch thumb and index | Spray; release to stop |
| `1`-`5` | Choose a neon palette |
| `C` | Clear wall canvas and reset drips |
| `M` | Mute/unmute the music |
| `Q` or `Esc` | Quit |

The interaction corridor uses apparent palm size as a distance proxy. Pitch and roll prompts are hand-pose estimates, not calibrated measurements of the camera or tripod; treat them as rough setup hints. The installation works without audio hardware, and its HUD reports when audio is unavailable.

## Setup tips

- In Windows Settings > Privacy & security > Camera, enable camera access for desktop apps.
- If the default camera does not open, try `--camera 1` (then `2`, if needed).
- Place the webcam near chest/shoulder height, roughly 1.2-1.4 m from the floor. Avoid putting the participant between the projector and wall.
- Keep the camera view well lit and frame the hands with some room around the image edges.
- Start around 1.1-2.2 m from the webcam; the camera inset will show distance/framing prompts.

## Troubleshooting

- **First launch fails to fetch the model:** connect to the internet and run `run.ps1`, or download the official model from the URL in `spray_canvas.py` to `hand_landmarker.task` in this folder.
- **Camera error:** close other camera apps, check Windows camera privacy settings, and try `--camera 1`.
- **PowerShell blocks scripts:** run `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell window, then run `setup.ps1`.
- **No music:** verify the four WAV files exist in `assets`. Regenerate them with `.venv\Scripts\python.exe generate_audio.py`.
- **Poor frame rate:** try a smaller camera mode, for example `run.ps1 --camera-width 640 --camera-height 480`.

## Project files

- `spray_canvas.py` - tracking, painting, audio mixing, telemetry, controls
- `generate_audio.py` - creates the original loopable music and spray beds
- `assets/` - generated WAV audio loops
- `setup.ps1` / `run.ps1` - Windows setup and launch helpers
- `requirements.txt` - Python package dependencies
