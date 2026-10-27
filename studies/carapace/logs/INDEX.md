# What each log holds (first session)

All runs: native MuJoCo, 2 CPU cores, 2026-10-26.

| File | Contents |
|---|---|
| `gentle.log` | Load replays over 32 perturbed starts (seed 7): horn20, tri20s, tri20m working plans |
| `gentle_rear.log` | The same for rc20 and rct20 working plans |
| `righting3.log` | Head-first flip searches, seeds 1-3: stock, horn20, horn30 (with hit costs) |
| `righting_axes2.log` | Side and diagonal searches, seeds 1-3: horn20, horn30 |
| `righting_tri.log` | Flip, side and diagonal searches: tri20s, tri20m (corners clear of the legs) |
| `righting_rear.log` | Flip, side and diagonal searches: rt20, rc20 |
| `righting_rear2.log` | Flip, side and diagonal searches: rct20 |
| `righting_*.json` | The same search summaries as JSON (rest pose, success, flip axis, median peaks) |
| `stroke_by_overhang.json`, `keepout2.log` | Leg push past the floor by carapace overhang and pose |

Second session (native MuJoCo, 2 CPU cores, 2026-10-26 to 27):

| File | Contents |
|---|---|
| `righting_tri20s_rear.log` | Flip, side and diagonal searches: tri20s_rc, tri20s_rt, tri20s_rct |
| `gentle_tri20s_rear.log` | Loads of their working plans, plus three first-session plans re-measured as a check |
| `gentle_horn30.log` | Loads of horn30's working plans from the first session's searches |
| `righting_tri30.log`, `gentle_tri30.log` | Searches and loads: tri30s, tri30m |
| `righting_tri30_rc.log`, `gentle_tri30_rc.log` | Searches and loads: tri30s_rc, tri30m_rc (two rest poses; see HANDOFF.md) |
| `gentle_stock_jump.log` | The stock high jump (`jumper.jump`) measured by `jump_loads.py` at 1000, 200 and 400 Hz physics |

`compile_results.py` reads every search and load log above, except the superseded ones,
into `RESULTS.md` and `results.csv`.

Superseded, kept for the record:

| File | Why it was superseded |
|---|---|
| `righting.log` | First flip search, no hit costs |
| `righting2.log` | Hit costs without the two-stage phase-in; every search settled on doing nothing |
| `righting_axes.log` | Axis searches before progress credit; every search settled on doing nothing |
| `grid_rear.log` | Render progress only |

In the search logs, each block starts with a "rests at" line. Its "best plan rights N/32" line
is the 32-start retry, seeded per search. Seeds run 1, 2, 3 in order within each tag.
