"""Generate the four original, loopable audio beds used by the canvas."""
from pathlib import Path
import wave
import numpy as np

OUT = Path(__file__).resolve().parent / "assets"
RATE = 44100
DURATION = 8.0

def save(name, samples):
    samples = np.clip(samples, -1, 1)
    stereo = np.column_stack((samples, samples))
    pcm = (stereo * 32767).astype("<i2")
    with wave.open(str(OUT / name), "wb") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(RATE); f.writeframes(pcm.tobytes())

def main():
    OUT.mkdir(exist_ok=True)
    n = int(RATE * DURATION); t = np.arange(n) / RATE
    rng = np.random.default_rng(17)
    noise = rng.normal(0, 1, n)
    # Softly filtered noise with a sine-shaped loop envelope to avoid a hard seam.
    hiss = np.convolve(noise, np.ones(64)/64, mode="same") * 2.7
    hiss += rng.normal(0, .12, n)
    save("hiss.wav", hiss * (.68 + .12*np.sin(2*np.pi*t/DURATION)))

    beat = 60/110
    bass = np.zeros(n)
    for step in range(16):
        start = int(step * beat/2 * RATE)
        end = min(n, start + int(.38*RATE))
        if start >= n: break
        u = np.arange(end-start)/RATE
        freq = [65.41, 77.78, 98.0, 77.78][(step//4) % 4]
        env = np.exp(-u*5)
        bass[start:end] += .36*env*np.sin(2*np.pi*(freq*u - 7*u*u))
        if step % 4 in (0, 2):
            kick_end = min(n, start+int(.18*RATE)); ku=np.arange(kick_end-start)/RATE
            bass[start:kick_end] += .30*np.exp(-ku*18)*np.sin(2*np.pi*(90*ku-90*ku*ku))
    save("bass.wav", bass)

    pad = np.zeros(n)
    chords = [(130.81,155.56,196.0),(116.54,138.59,174.61),(103.83,130.81,155.56),(116.54,146.83,174.61)]
    for bar, chord in enumerate(chords):
        a=int(bar*2*RATE); b=min(n,a+2*RATE); u=np.arange(b-a)/RATE
        env=np.minimum(u/.25,1)*np.minimum((2-u)/.35,1)
        for j,f in enumerate(chord): pad[a:b] += (.095/(j+1))*env*np.sin(2*np.pi*f*u)
    save("pad.wav", pad)

    arp=np.zeros(n); scale=[523.25,622.25,698.46,783.99,932.33,1046.5,932.33,783.99]
    for step in range(64):
        a=int(step*beat/4*RATE); b=min(n,a+int(.22*RATE)); u=np.arange(b-a)/RATE
        f=scale[(step + (step//16))%len(scale)]
        env=np.exp(-u*12)
        arp[a:b] += .19*env*(np.sin(2*np.pi*f*u)+.25*np.sin(4*np.pi*f*u))
    save("arp.wav", arp)
    print(f"Created four 8-second stereo WAV loops in {OUT}")

if __name__ == "__main__": main()
