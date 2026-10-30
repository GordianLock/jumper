"""Make a title slide for footage: black, white text, a header and a stats table.

    python studies/footage/make_slide.py <slide.json> <out.png>

slide.json: {"title": str, "rows": [[label, value, style?], ...], "footnote": str,
"schema": str (optional, the colour pattern, drawn at the bottom)}.
A row whose style is "note" is drawn smaller, grey and italic (a comparison line).
1920x1080, Inter.
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FONT = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
spec = json.load(open(sys.argv[1]))
img = Image.new("RGB", (W, H), (0, 0, 0))
dr = ImageDraw.Draw(img)
f_title = ImageFont.truetype(FONT.format("SemiBold"), 76)
f_label = ImageFont.truetype(FONT.format("Regular"), 40)
f_value = ImageFont.truetype(FONT.format("Medium"), 40)
f_note = ImageFont.truetype(FONT.format("Italic"), 32)
f_foot = ImageFont.truetype(FONT.format("Regular"), 26)
WHITE, GREY, LINE = (255, 255, 255), (165, 165, 165), (70, 70, 70)

rows = spec["rows"]
row_h = [66 if (r[2:] or [""])[0] != "note" else 50 for r in rows]
table_h = sum(row_h)
top = (H - (110 + 50 + table_h + 90)) // 2 - (40 if spec.get("schema") else 0)
tw = dr.textlength(spec["title"], font=f_title)
dr.text(((W - tw) / 2, top), spec["title"], font=f_title, fill=WHITE)
y = top + 110 + 50
x_label, x_value, x_end = 330, 1240, 1590
dr.line([(x_label, y), (x_end, y)], fill=LINE, width=2)
for r, rh in zip(rows, row_h):
    note = (r[2:] or [""])[0] == "note"
    fl, fv, col = (f_note, f_note, GREY) if note else (f_label, f_value, WHITE)
    ty = y + (2 if note else 8)
    dr.text((x_label + (30 if note else 0), ty), r[0], font=fl, fill=col)
    dr.text((x_value, ty), r[1], font=fv, fill=col)
    y += rh
    if not note:
        nxt = rows.index(r) + 1
        if nxt >= len(rows) or (rows[nxt][2:] or [""])[0] != "note":
            dr.line([(x_label, y), (x_end, y)], fill=LINE, width=1)
fw = dr.textlength(spec["footnote"], font=f_foot)
dr.text(((W - fw) / 2, y + 40), spec["footnote"], font=f_foot, fill=GREY)
if spec.get("schema"):
    f_schema = ImageFont.truetype(FONT.format("Regular"), 34)
    sw = dr.textlength(spec["schema"], font=f_schema)
    dr.text(((W - sw) / 2, H - 120), spec["schema"], font=f_schema, fill=WHITE)
img.save(sys.argv[2])
print("wrote", sys.argv[2])
