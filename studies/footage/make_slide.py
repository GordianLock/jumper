"""Make a title slide for footage: black, white text, a header and a stats table, in
English and, optionally, Simplified Chinese under each English line.

    python studies/footage/make_slide.py <slide.json> <out.png>

slide.json:

    {"title": str, "title_zh": str?,
     "rows": [{"en": str, "zh": str?, "value": str, "note": bool?}, ...]
             (or [label, value, "note"?] lists, English only),
     "footnote": str, "footnote_zh": str?,
     "schema": str?, "schema_zh": str?}

A note row is drawn smaller, grey and italic (a comparison line). 1920x1080; Inter
for Latin, Noto Sans CJK SC for Chinese.
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
INTER = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"
CJK_SC = 2                       # the Simplified Chinese face inside the collection
WHITE, GREY, DIM, LINE = (255, 255, 255), (165, 165, 165), (190, 190, 190), (70, 70, 70)

spec = json.load(open(sys.argv[1]))
rows = [r if isinstance(r, dict) else {"en": r[0], "value": r[1], "note": (r[2:] or [""])[0] == "note"}
        for r in spec["rows"]]
zh = any(r.get("zh") for r in rows) or bool(spec.get("title_zh"))
k = 0.76 if zh else 1.0          # everything a little smaller when there are two languages


def inter(style, size):
    return ImageFont.truetype(INTER.format(style), round(size * k))


def cjk(style, size):
    return ImageFont.truetype(CJK.format(style), round(size * k), index=CJK_SC)


img = Image.new("RGB", (W, H), (0, 0, 0))
dr = ImageDraw.Draw(img)
f_title, f_title_zh = inter("SemiBold", 76), cjk("Medium", 52)
f_label, f_value, f_label_zh = inter("Regular", 40), inter("Medium", 40), cjk("Regular", 28)
f_note, f_note_zh = inter("Italic", 32), cjk("Light", 24)
f_foot, f_foot_zh = inter("Regular", 26), cjk("Light", 22)
f_schema, f_schema_zh = inter("Regular", 34), cjk("Regular", 28)


def centred(y, text, font, fill):
    dr.text(((W - dr.textlength(text, font=font)) / 2, y), text, font=font, fill=fill)


def h(size):
    return round(size * k)


# Heights, top to bottom, so the block can be centred.
title_h = h(96) + (h(70) if spec.get("title_zh") else 0)
row_hs = []
for r in rows:
    base = h(50) if r.get("note") else h(66)
    row_hs.append(base + (h(30 if r.get("note") else 38) if r.get("zh") else 0))
foot_h = h(40) + (h(32) if spec.get("footnote_zh") else 0)
schema_h = (h(50) + (h(44) if spec.get("schema_zh") else 0)) if spec.get("schema") else 0
block = title_h + h(40) + sum(row_hs) + h(30) + foot_h + (h(40) + schema_h if schema_h else 0)
y = (H - block) // 2

centred(y, spec["title"], f_title, WHITE)
y += h(96)
if spec.get("title_zh"):
    centred(y, spec["title_zh"], f_title_zh, WHITE)
    y += h(70)
y += h(40)

x_label, x_value, x_end = 330, 1240, 1590
dr.line([(x_label, y), (x_end, y)], fill=LINE, width=2)
for i, (r, rh) in enumerate(zip(rows, row_hs)):
    note = r.get("note")
    indent = 30 if note else 0
    top = y + (h(2) if note else h(8))
    dr.text((x_label + indent, top), r["en"], font=f_note if note else f_label,
            fill=GREY if note else WHITE)
    if r.get("zh"):
        dr.text((x_label + indent, top + (h(40) if note else h(50))), r["zh"],
                font=f_note_zh if note else f_label_zh, fill=GREY if note else DIM)
    # The value sits level with the English line.
    dr.text((x_value, top), r["value"], font=f_note if note else f_value,
            fill=GREY if note else WHITE)
    y += rh
    nxt_note = i + 1 < len(rows) and rows[i + 1].get("note")
    if not note and not nxt_note:
        dr.line([(x_label, y), (x_end, y)], fill=LINE, width=1)
    if note:
        dr.line([(x_label, y), (x_end, y)], fill=LINE, width=1)

y += h(30)
centred(y, spec["footnote"], f_foot, GREY)
y += h(40)
if spec.get("footnote_zh"):
    centred(y, spec["footnote_zh"], f_foot_zh, GREY)
    y += h(32)
if spec.get("schema"):
    y += h(40)
    centred(y, spec["schema"], f_schema, WHITE)
    y += h(50)
    if spec.get("schema_zh"):
        centred(y, spec["schema_zh"], f_schema_zh, DIM)

img.save(sys.argv[2])
print("wrote", sys.argv[2])
