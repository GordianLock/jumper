import json, sys
import numpy as np
R = json.load(open(sys.argv[1])); W = 2.543 * 9.81
LEGS = ["LF", "RF", "LM", "RM", "LR", "RR"]

def metrics(r):
    L = r["legs"]; roll, pitch = r["roll"], r["pitch"]
    tot = lambda k: sum(L[x].get(k, 0) for x in LEGS)
    # righting direction: roll<180 means the right side is lower; tip on over it
    mr = tot("roll-") if roll < 180 else tot("roll+") if roll > 180 else max(tot("roll+"), tot("roll-"))
    # head up: flip over the tail (pitch-); head down: over the head (pitch+)
    mp = tot("pitch-") if pitch > 0 else tot("pitch+") if pitch < 0 else max(tot("pitch+"), tot("pitch-"))
    return dict(roll=roll, pitch=pitch, h=r["h"], head_up=r["head_up"], reach=sum(L[x]["reach"] for x in LEGS),
                M_roll=mr, M_pitch=mp, lever_roll=100 * mr / W, lever_pitch=100 * mp / W,
                rear_stroke=100 * min(L["LR"]["stroke"], L["RR"]["stroke"]),
                rear_lift=L["LR"].get("lift", 0) + L["RR"].get("lift", 0),
                low_side=[x for x in LEGS if L[x]["reach"]])

F = [metrics(r) for r in R if r["feasible"]]
# a carapace holds the body at its lowest feasible height plus its thickness:
# look at the lowest feasible height and up to 3 cm above it
lowest = {}
for f in F: lowest[(f["roll"], f["pitch"])] = min(lowest.get((f["roll"], f["pitch"]), 9), f["h"])
near = [f for f in F if f["h"] <= lowest[(f["roll"], f["pitch"])] + 0.03]

def table(key, title):
    print(f"\n== {title} (N*m; cm = lever at the robot's weight) ==")
    print("roll\\pitch " + " ".join(f"{p:>6}" for p in range(-40, 41, 10)))
    for roll in range(120, 241, 10):
        row = []
        for p in range(-40, 41, 10):
            c = [f[key] for f in near if f["roll"] == roll and f["pitch"] == p]
            row.append(f"{max(c):6.1f}" if c else "     -")
        print(f"{roll:>10} " + " ".join(row))

table("M_roll", "best sideways righting moment")
table("M_pitch", "best end-over-end righting moment")
table("rear_stroke", "rear-leg spring stroke, cm (both rear legs)")
print("\nlowest feasible COM height (m):")
for roll in range(120, 241, 20):
    print(roll, [lowest.get((roll, p)) for p in range(-40, 41, 10)])
for key in ["M_roll", "M_pitch"]:
    print(f"\ntop 8 by {key}:")
    for f in sorted(near, key=lambda f: -f[key])[:8]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()})
flat = [f for f in F if f["roll"] == 180 and f["pitch"] == 0]
print("\nflat on back, by height:")
for f in sorted(flat, key=lambda f: f["h"]):
    print(f["h"], "roll %.2f pitch %.2f reach %d rear stroke %.1f" % (f["M_roll"], f["M_pitch"], f["reach"], f["rear_stroke"]))
