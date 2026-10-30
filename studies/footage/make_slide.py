"""Make a title slide for footage: black, white text, a header and a stats table, in
English and, optionally, Simplified Chinese under each English line.

    python studies/footage/make_slide.py <slide.json> <out.png>

slide.json:

    {"title": str, "title_zh": str?,
     "rows": [{"en": str, "zh": str?, "value": str, "note": bool?}, ...]
             (or [label, value, "note"?] lists, English only),
     "footnote": str, "footnote_zh": str?,
     "schema": str?, "schema_zh": str?}

or, for a statement slide,

    {"paragraphs": [{"en": str, "zh": str?, "lead": bool?}, ...]}

each paragraph wrapped and centred, its Chinese beneath it; a lead paragraph (an
opening or closing line) is drawn in a heavier weight, at the same size.

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


def statement(spec):
    """A slide of paragraphs, English then Chinese, centred."""
    img = Image.new("RGB", (W, H), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    width = 1320
    fonts = {True: (ImageFont.truetype(INTER.format("SemiBold"), 32),
                    ImageFont.truetype(CJK.format("Medium"), 25, index=CJK_SC)),
             False: (ImageFont.truetype(INTER.format("Regular"), 32),
                     ImageFont.truetype(CJK.format("Regular"), 25, index=CJK_SC))}

    def units(text, by_char):
        """English by word; Chinese by clause (after ，、。) where a clause fits, else
        by character, so a line does not break inside a phrase if it can help it."""
        if not by_char:
            return text.split()
        out, cur = [], ""
        for ch in text:
            cur += ch
            if ch in "，、。；：":
                out.append(cur)
                cur = ""
        return out + ([cur] if cur else [])

    def greedy(text, font, by_char, w_max):
        words, lines, cur = [], [], ""
        for u in units(text, by_char):
            words += [u] if dr.textlength(u, font=font) <= w_max else list(u)
        for w in words:
            t = cur + w if by_char or not cur else cur + " " + w
            if dr.textlength(t, font=font) > w_max and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        return lines + [cur]

    def wrap(text, font, by_char):
        """As few lines as fit `width`, then made as even as that many lines allow,
        so a paragraph never ends on a word or two alone."""
        n = len(greedy(text, font, by_char, width))
        if by_char:                      # clause breaks already keep it even enough
            return greedy(text, font, by_char, width)
        w_max = width
        while w_max > 200 and len(greedy(text, font, by_char, w_max - 10)) == n:
            w_max -= 10
        return greedy(text, font, by_char, w_max)

    blocks = []
    for para in spec["paragraphs"]:
        f_en, f_zh = fonts[bool(para.get("lead"))]
        lines = [(t, f_en, WHITE, round(f_en.size * 1.32)) for t in wrap(para["en"], f_en, False)]
        if para.get("zh"):
            lines.append(("", f_zh, DIM, round(f_zh.size * 0.3)))
            lines += [(t, f_zh, DIM, round(f_zh.size * 1.5)) for t in wrap(para["zh"], f_zh, True)]
        blocks.append(lines)
    gap = 40
    total = sum(sum(h for *_, h in b) for b in blocks) + gap * (len(blocks) - 1)
    y = (H - total) // 2
    for b in blocks:
        for text, font, fill, h in b:
            dr.text(((W - dr.textlength(text, font=font)) / 2, y), text, font=font, fill=fill)
            y += h
        y += gap
    img.save(sys.argv[2])
    print("wrote", sys.argv[2])
    sys.exit(0)


if "paragraphs" in spec:
    statement(spec)
rows = [r if isinstance(r, dict) else {"en": r[0], "value": r[1], "note": (r[2:] or [""])[0] == "note"}
        for r in spec["rows"]]
zh = any(r.get("zh") for r in rows) or bool(spec.get("title_zh"))
k = 1.0                          # the English stays full size; Chinese goes alongside


def inter(style, size):
    return ImageFont.truetype(INTER.format(style), round(size * k))


def cjk(style, size):
    return ImageFont.truetype(CJK.format(style), round(size * k), index=CJK_SC)


img = Image.new("RGB", (W, H), (0, 0, 0))
dr = ImageDraw.Draw(img)
f_title, f_title_zh = inter("SemiBold", 76), cjk("Medium", 52)
f_label, f_value, f_label_zh = inter("Regular", 40), inter("Medium", 40), cjk("Regular", 34)
f_note, f_note_zh = inter("Italic", 32), cjk("Light", 25)
f_foot, f_foot_zh = inter("Regular", 26), cjk("Light", 22)
f_schema, f_schema_zh = inter("Regular", 34), cjk("Regular", 28)


def centred(y, text, font, fill):
    dr.text(((W - dr.textlength(text, font=font)) / 2, y), text, font=font, fill=fill)


def h(size):
    return round(size * k)


# Heights, top to bottom, so the block can be centred.
title_h = h(96) + (h(62) if spec.get("title_zh") else 0)
row_hs = [h(50) if r.get("note") else h(66) for r in rows]
foot_h = h(40) + (h(32) if spec.get("footnote_zh") else 0)
schema_h = (h(50) + (h(44) if spec.get("schema_zh") else 0)) if spec.get("schema") else 0
gap = h(30) if zh else h(40)
block = title_h + gap + sum(row_hs) + h(30) + foot_h + (gap + schema_h if schema_h else 0)
y = (H - block) // 2

centred(y, spec["title"], f_title, WHITE)
y += h(96)
if spec.get("title_zh"):
    centred(y, spec["title_zh"], f_title_zh, WHITE)
    y += h(62)
y += gap

# With Chinese, a third column: English label, value, Chinese label.
x_label, x_value, x_zh, x_end = (100, 910, 1290, 1820) if zh else (330, 1240, None, 1590)
dr.line([(x_label, y), (x_end, y)], fill=LINE, width=2)
for i, (r, rh) in enumerate(zip(rows, row_hs)):
    note = r.get("note")
    indent = 30 if note else 0
    top = y + (h(2) if note else h(8))
    dr.text((x_label + indent, top), r["en"], font=f_note if note else f_label,
            fill=GREY if note else WHITE)
    if r.get("zh"):
        # The Chinese note starts at the column edge: its line is the longest there.
        dr.text((x_zh, top + (h(4) if note else h(2))), r["zh"],
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
    y += gap
    centred(y, spec["schema"], f_schema, WHITE)
    y += h(50)
    if spec.get("schema_zh"):
        centred(y, spec["schema_zh"], f_schema_zh, DIM)

img.save(sys.argv[2])
print("wrote", sys.argv[2])
