"""Compile every roll result in logs/ into results.csv and RESULTS.md.

Run from studies/carapace/:  python compile_results.py
Reads only the logs (no simulation), so the two outputs always match them.

Sources:
- searches: the righting*.log files listed in SEARCH_LOGS, three seeds per shell and roll,
  in seed order. Each "best plan rights N/32" line is that search's 32-start retry.
- loads: the gentle*.log files listed in LOAD_LOGS. A plan measured twice keeps its first
  line (the re-measurements reproduced the same numbers).
"""
import csv
import re
from collections import defaultdict

SEARCH_LOGS = ["righting3.log", "righting_axes2.log", "righting_tri.log", "righting_rear.log",
               "righting_rear2.log", "righting_tri20s_rear.log", "righting_tri30.log",
               "righting_tri30_rc.log"]
LOAD_LOGS = ["gentle.log", "gentle_rear.log", "gentle_tri20s_rear.log", "gentle_horn30.log",
             "gentle_tri30.log", "gentle_tri30_rc.log"]

# Stock jumps, measured the same way (median over attempts of each attempt's worst
# floor-contact torque on any joint). See HANDOFF.md, "Baselines".
HIGH_JUMP = dict(torque=5.9, p90=7.0, limbs=76, shell=0, feet=175)   # jumper.jump, 400 Hz, 32 attempts
REF_FREE_JUMP = dict(torque=4.8, worst=5.6)                          # jumper.ref_free_jump, 8 attempts

# Shells with two belly-up rest poses (55 deg sitting, 16 deg lying). Their plans were
# searched from the sitting pose; in each seed's 32 retry starts this many sit.
TWO_POSE = {"tri30s_rc", "tri30m_rc"}
SITTING = {1: 20, 2: 28, 3: 24}

SHELLS = {
    "stock": "no horns",
    "horn20": "20 deg centre horn",
    "horn30": "30 deg centre horn",
    "tri20s": "20 deg centre + low (2.5 cm) front corners",
    "tri20m": "20 deg centre + tall (3.5 cm) front corners",
    "rt20": "20 deg centre + tail",
    "rc20": "20 deg centre + rear corners",
    "rct20": "20 deg centre + rear corners + tail",
    "tri20s_rt": "20 deg centre + low front corners + tail",
    "tri20s_rc": "20 deg centre + low front corners + rear corners",
    "tri20s_rct": "20 deg centre + low front corners + rear corners + tail",
    "tri30s": "30 deg centre + low (2.5 cm) front corners",
    "tri30m": "30 deg centre + tall (3.5 cm) front corners",
    "tri30s_rc": "30 deg centre + low front corners + rear corners",
    "tri30m_rc": "30 deg centre + tall front corners + rear corners",
}
ROLLS = ("flip", "side", "diag")


def split(tag):
    for r in ("side", "diag"):
        if tag.endswith("_" + r):
            return tag[: -len(r) - 1], r
    return tag, "flip"


def read_searches():
    rate, rest, axis = {}, {}, {}
    for log in SEARCH_LOGS:
        seen = defaultdict(int)
        for line in open("logs/" + log):
            m = re.match(r"(\S+) rests at body-up (\S+), head pitch (\S+) deg, base height (\S+) m", line)
            if m:
                rest.setdefault(m.group(1), (float(m.group(3)), float(m.group(4))))
            m = re.match(r"\s+(\S+): flip axis at (\S+) deg.*?(?:alignment with target (\S+))?$", line)
            if m:
                axis[(m.group(1), seen[m.group(1)] + 1)] = (float(m.group(2)), m.group(3))
            m = re.match(r"\s+(\S+): best plan rights (\d+)/32", line)
            if m:
                seen[m.group(1)] += 1
                rate[(m.group(1), seen[m.group(1)])] = int(m.group(2))
    return rate, rest, axis


def read_loads():
    loads = {}
    pat = re.compile(r"(\S+)\s+(\S+)\s+plan_(\S+)\s+rights (\d+)/32 \| joint torque median (\S+) "
                     r"p90 (\S+) N\*m \(worst at (\S+)\) \| median peaks N: limbs (\S+) shell (\S+) "
                     r"horn (\S+) feet (\S+)")
    for log in LOAD_LOGS:
        for line in open("logs/" + log):
            m = pat.match(line)
            if m and m.group(3) not in loads:
                loads[m.group(3)] = dict(rights=int(m.group(4)), torque=float(m.group(5)),
                                         p90=float(m.group(6)), worst_joint=m.group(7).replace("_joint", ""),
                                         limbs=float(m.group(8)), shell=float(m.group(9)),
                                         horn=float(m.group(10)), feet=float(m.group(11)))
    return loads


def works(shell, seed, n, roll, alignment):
    """A search found a working plan: 29/32 or better, or, on a two-pose shell, every
    sitting start. A side or diagonal plan must also turn about its target axis: alignment
    0.9 or better (within about 25 deg). One plan fails only that test, horn20_diag_s2, which
    rights 32/32 but rolls 28 deg off the diagonal."""
    if roll != "flip" and (not alignment or float(alignment) < 0.9):
        return False
    return n >= 29 or (shell in TWO_POSE and n >= SITTING[seed])


def beats_high_jump(ld):
    return (ld["torque"] < HIGH_JUMP["torque"] and ld["p90"] < HIGH_JUMP["p90"]
            and ld["limbs"] < HIGH_JUMP["limbs"] and ld["feet"] < HIGH_JUMP["feet"])


def main():
    rate, rest, axis = read_searches()
    loads = read_loads()
    rows = []
    for (tag, seed), n in sorted(rate.items()):
        shell, roll = split(tag)
        plan = f"{tag}_s{seed}"
        ld = loads.get(plan)
        ax = axis.get((tag, seed), (None, None))
        row = dict(shell=shell, roll=roll, seed=seed, plan=f"plan_{plan}", search_rights_of_32=n,
                   works=works(shell, seed, n, roll, ax[1]), axis_deg=ax[0], axis_alignment=ax[1] or "")
        if ld:
            row.update(load_rights_of_32=ld["rights"], torque_median_Nm=ld["torque"], torque_p90_Nm=ld["p90"],
                       worst_joint=ld["worst_joint"], limbs_N=ld["limbs"], shell_N=ld["shell"],
                       horn_N=ld["horn"], feet_N=ld["feet"], beats_high_jump=beats_high_jump(ld),
                       at_or_below_ref_free_jump_torque=ld["torque"] <= REF_FREE_JUMP["torque"])
        rows.append(row)
    cols = ["shell", "roll", "seed", "plan", "search_rights_of_32", "works", "axis_deg", "axis_alignment",
            "load_rights_of_32", "torque_median_Nm", "torque_p90_Nm", "worst_joint", "limbs_N", "shell_N",
            "horn_N", "feet_N", "beats_high_jump", "at_or_below_ref_free_jump_torque"]
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[(r["shell"], r["roll"])].append(r)

    out = [open("RESULTS_HEADER.md").read().rstrip(), ""]
    out += ["## Which rolls each shell can do", "",
            "Searches that found a working plan, out of 3, with the gentlest working plan's median "
            "joint torque in N·m. A dash means not searched. `*` marks the two-pose shells, where "
            "\"working\" means it works from every start in the 55° sitting pose.", "",
            "| Shell | Description | Rests head-up / height | Flip | Side | Diagonal |",
            "|---|---|---|---|---|---|"]
    for shell, desc in SHELLS.items():
        cells = []
        for roll in ROLLS:
            rs = by.get((shell, roll))
            if not rs:
                cells.append("—"); continue
            ok = [r for r in rs if r["works"]]
            tq = [r["torque_median_Nm"] for r in ok if "torque_median_Nm" in r]
            star = "*" if shell in TWO_POSE else ""
            cells.append(f"{len(ok)}/3{star}" + (f" ({min(tq):.1f})" if tq else ""))
        pitch, z = rest.get(shell, (float("nan"), float("nan")))
        pose = f"{pitch:.0f}° / {z * 100:.1f} cm" + (" (or 16°)" if shell in TWO_POSE else "")
        out.append(f"| `{shell}` | {desc} | {pose} | " + " | ".join(cells) + " |")

    def table(rs):
        lines = ["| Roll | Shell | Plan | Starts | Joint torque (p90) | Limbs | Feet | Shell | Horn |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            lines.append(f"| {r['roll']} | `{r['shell']}` | `{r['plan']}` | {r['load_rights_of_32']}/32 | "
                         f"{r['torque_median_Nm']:.1f} ({r['torque_p90_Nm']:.1f}) | {r['limbs_N']:.0f} | "
                         f"{r['feet_N']:.0f} | {r['shell_N']:.0f} | {r['horn_N']:.0f} |")
        return lines

    measured = sorted((r for r in rows if "torque_median_Nm" in r and r["works"]),
                      key=lambda r: r["torque_median_Nm"])
    hj = HIGH_JUMP
    out += ["", "## Against the stock jumps", "",
            "| Stock motion | Attempts | Joint torque (p90) | Limbs | Feet | Shell |",
            "|---|---|---|---|---|---|",
            f"| High jump `jumper.jump` (the shipped jump) | 32 | {hj['torque']} ({hj['p90']}) | "
            f"{hj['limbs']} | {hj['feet']} | {hj['shell']} |",
            f"| Reference-free jump `jumper.ref_free_jump` | 8 | {REF_FREE_JUMP['torque']} "
            f"(worst {REF_FREE_JUMP['worst']}) | — | — | — |", "",
            "### Working plans that beat the high jump on all four measures", "",
            "Joint torque, p90, limbs and feet all below the high jump's.", ""]
    out += table([r for r in measured if r["beats_high_jump"]])
    for roll in ROLLS:
        best = [r for r in measured if r["beats_high_jump"] and r["roll"] == roll]
        out.append("")
        out.append(f"- {roll}: {len(best)} plan(s)" + (f"; gentlest `{best[0]['plan']}` at "
                   f"{best[0]['torque_median_Nm']:.1f} N·m" if best else ""))
    out += ["", "### Working plans at or below the reference-free jump's joint torque (4.8 N·m)", ""]
    out += table([r for r in measured if r["at_or_below_ref_free_jump_torque"]])
    out += ["", "## Every load-measured working plan, gentlest first", ""]
    out += table(measured)
    out += ["", "Every search, including the ones that failed and the axis checks, is in `results.csv`."]
    open("RESULTS.md", "w").write("\n".join(out) + "\n")
    print(f"{len(rows)} searches, {len(measured)} measured working plans, "
          f"{sum(r['beats_high_jump'] for r in measured)} beat the high jump")


if __name__ == "__main__":
    main()
