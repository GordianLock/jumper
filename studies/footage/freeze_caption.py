"""Hold one frame of a clip for a few seconds with a caption on it, then carry on.

    python studies/footage/freeze_caption.py <in.mp4> <out.mp4> --at 6.76 --hold 3 \\
        --caption studies/footage/captions/stock_shell_warning.json

The caption file is {"en": str, "zh": str?, "colour": [r, g, b]?}: the English, then
the Chinese beneath it, wrapped to the upper-left quarter of the frame, with a soft
dark panel behind it so it reads on any background. Sizes scale with the frame's width, so a
preview and the full render look the same.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

INTER = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
CJK_SC = 2

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("out")
p.add_argument("--at", type=float, required=True, help="s into the clip to hold")
p.add_argument("--hold", type=float, default=3.0)
p.add_argument("--caption", required=True)
a = p.parse_args()
cap = json.load(open(a.caption))


def probe(key):
    return subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v",
                                    "-show_entries", f"stream={key}", "-of", "csv=p=0",
                                    a.src], text=True).strip()


w, h = (int(x) for x in probe("width,height").split(","))
fps = probe("r_frame_rate")
k = w / 1920
tmp = Path(tempfile.mkdtemp())
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{a.at}", "-i", a.src,
                "-frames:v", "1", str(tmp / "frame.png")], check=True)

img = Image.open(tmp / "frame.png").convert("RGBA")
layer = Image.new("RGBA", img.size)
shadow = Image.new("RGBA", img.size)
d, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
colour = tuple(cap.get("colour", [235, 40, 40]))
f_en = ImageFont.truetype(INTER, round(40 * k))
f_zh = ImageFont.truetype(CJK, round(34 * k), index=CJK_SC)
margin = round(64 * k)
pad = round(24 * k)
width = w // 2 - 2 * margin                  # inside the upper-left quarter


def wrap(text, font, by_char):
    units = list(text) if by_char else text.split()
    lines, cur = [], ""
    for u in units:
        t = cur + u if by_char or not cur else cur + " " + u
        if d.textlength(t, font=font) > width and cur:
            lines.append(cur)
            cur = u.lstrip() if by_char else u
        else:
            cur = t
    return lines + [cur]


y = margin + pad
for text, font, by_char in ((cap["en"], f_en, False), (cap.get("zh"), f_zh, True)):
    if not text:
        continue
    for line in wrap(text, font, by_char):
        off = max(1, round(3 * k))
        ds.text((margin + off, y + off), line, font=font, fill=(0, 0, 0, 200))
        d.text((margin, y), line, font=font, fill=colour + (255,))
        y += round(font.size * 1.35)
    y += round(18 * k)
panel = Image.new("RGBA", img.size)
ImageDraw.Draw(panel).rounded_rectangle(
    [margin - pad, margin - pad, margin + width + pad, y - round(18 * k) + pad],
    radius=round(14 * k), fill=(0, 0, 0, 165))
out = Image.alpha_composite(img, panel)
out = Image.alpha_composite(out, shadow.filter(ImageFilter.GaussianBlur(max(1, 3 * k))))
Image.alpha_composite(out, layer).convert("RGB").save(tmp / "held.png")

enc = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", fps]
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.src, "-loop", "1", "-t", f"{a.hold}",
                "-i", str(tmp / "held.png"), "-filter_complex",
                f"[0]trim=0:{a.at},setpts=PTS-STARTPTS[a];"
                f"[1]fps={fps},format=yuv420p,setsar=1[b];"
                f"[0]trim=start={a.at},setpts=PTS-STARTPTS[c];"
                "[a][b][c]concat=n=3:v=1:a=0", *enc, a.out], check=True)
print("wrote", a.out)
