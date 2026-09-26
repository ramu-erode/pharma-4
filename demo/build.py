"""Stitch recorded clips and title stills into one MP4."""

import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
HERE = Path(__file__).parent
CLIPS = HERE / "clips"
PARTS = HERE / "parts"
OUT = Path(sys.argv[1])
TRIM = 1.2  # blank page load at the start of every browser clip
FADE = 0.4
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30"]


def run(args: list[str]) -> None:
    subprocess.run([FF, "-loglevel", "error", "-y", *args], check=True)


def duration(path: Path) -> float:
    err = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def fades(d: float) -> str:
    return f"fade=t=in:st=0:d={FADE},fade=t=out:st={d - FADE:.2f}:d={FADE}"


PARTS.mkdir(exist_ok=True)
for f in PARTS.glob("*"):
    f.unlink()

items = sorted([*CLIPS.glob("*.webm"), *CLIPS.glob("*.png")])
parts = []
for i, src in enumerate(items):
    dst = PARTS / f"{i:02d}.mp4"
    if src.suffix == ".png":
        hold = float(re.search(r"-([\d.]+)s\.png$", src.name).group(1))
        run(["-loop", "1", "-t", f"{hold}", "-i", str(src), "-vf", fades(hold), *ENC, str(dst)])
    else:
        d = duration(src) - TRIM
        run(["-ss", f"{TRIM}", "-i", str(src), "-t", f"{d:.2f}", "-vf", fades(d), *ENC, "-an", str(dst)])
    parts.append(dst)
    print(f"{src.name} -> {dst.name} ({duration(dst):.1f}s)")

listing = PARTS / "list.txt"
listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
OUT.parent.mkdir(parents=True, exist_ok=True)
run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(OUT)])
print(f"{OUT} {duration(OUT):.1f}s {OUT.stat().st_size / 1e6:.1f} MB")
