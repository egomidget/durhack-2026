"""Record solo / enrollment / noise / overlap audio from the mic.

Spec:
  python record.py --person mikolaj --session s1 --minutes 5   -> raw/mikolaj/s1_01.wav (next free number)
  python record.py --enroll mikolaj --seconds 20               -> enroll/mikolaj.wav
  python record.py --noise --minutes 2                         -> noise/room_01.wav
  python record.py --overlap clip1 --seconds 20                -> overlaps/clip1.wav
  python record.py --list-devices
  Options: --in <device number or name>.
Behaviour: countdown before starting, live level meter in the terminal, warn if the
recording is very quiet (rms < -50 dBFS), save with common.save_wav (16 kHz mono).
Record at the device's native rate (e.g. 48 kHz) and let save_wav's caller resample.
"""


def main():
    raise NotImplementedError("TODO (M1): implement per the docstring")


if __name__ == "__main__":
    main()
