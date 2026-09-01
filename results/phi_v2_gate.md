# `PHI_V2` — declared 2026-09-01, before the runs

`docs/REDUCTION.md` task 3. One flag, `--phi-v2`, 5 seeds against 5 seeds.

## Why now, and why this is not one of the six killed interventions

📏 Re-measured 2026-09-01 on the **fixed** trainer, `scripts/measure_potential.py`,
eval split, stage 4, F4, CUDA. Every number in the inherited Φ audit replicates:

| | shipped | v2 |
|---|---|---|
| closing 250 → 60 m, per 8 m step | **0.0133** (0.25× the 0.0544 bar) | **0.0774** (1.42×) |
| role-less drone, 8 m toward the axis, at 200 / 500 / 800 m | **+0.0000 / +0.0000 / +0.0000** | +0.0078 / +0.0010 / +0.0003 |
| whole trip home | **+0.000** | +0.339 / +0.446 / +0.465 |
| mean Φ(B0) − mean Φ(learned) | +1.771 | **+2.988** |
| corr with discounted future capability | +0.346 | +0.344 |

🔒 **The shipped potential is still *exactly zero* for four drones out of five**,
at every distance. That survived the trainer fix, so it is a property of the
reward and not of the learner.

⛔ **This is not a re-proposal of a killed intervention.** The six nulls
(`w_hold`, `w_relay`, `d_ref` 1500→400, `potential_scale` 10→30, recurrence,
agent-specific critic) were knobs that **scaled** a potential whose directional
gradient in the operating regime was 0.013–0.03/step.
`docs/inherited/DECISIONS.md`'s own retro-explanation: *"The reason they failed is
arithmetic, not mechanism."* `PHI_V2` changes that arithmetic — 5.8× the closing
gradient, and a non-zero gradient where there was an exact zero.

⚠️ **PBRS cannot change what is optimal** (Ng, Harada & Russell 1999; Devlin &
Kudenko 2011). So this can only change how far the policy gets in a fixed budget.
A large `mission_capable` gain would be a *learning-speed* result, not a
better objective, and must be reported that way.

## Condition

Everything fixed except the flag: `deepsets` / `deep` cadence, F4, curriculum on,
12 M env-steps, `value_clip` 0.2, lr 3e-4, N = 5, 5 seeds. Control is the
existing `rq2-deepsets-deep` cell; the treatment adds `--phi-v2` and nothing else.

## Baseline, on the eval split, 5 seeds — the numbers the gate is read against

| | deepsets (control) | B0 |
|---|---|---|
| `observer_range_m` | **218.9** [188.5 – 230.2] | **88.7** [86.7 – 97.6] |
| `mission_capable` | 39.8 % [36.5 – 45.8] | 55.7 % [52.3 – 61.2] |
| `observed` | 59.6 % [57.5 – 64.2] | 93.6 % |
| `off_axis_m` | 266.0 [219.0 – 288.3] | 107.5 |

## Decision rule

🔒 **Primary readout is `observer_range_m`, not `mission_capable`.** The measured
deficit is that the observer does not close — 218.9 m against B0's 88.7 m — and
`Φ_standoff` centres its logistic at 127 m for exactly that. Judging on the
headline metric would let a shaping change pass or fail on noise in a quantity it
only affects indirectly.

| | rule |
|---|---|
| **promote** | median `observer_range_m` falls by **≥ 20 m** (to ≤ 198.9) **and** median `mission_capable` does not regress by more than **2 pp** (≥ 37.8 %) **and** the worst seed does not regress by more than 2 pp (≥ 34.5 %) |
| **kill** | `observer_range_m` moves less than 20 m. The audit's arithmetic was right and still did not move the behaviour, which sends the search elsewhere — and that is a stronger negative result than the six before it, because this is the first intervention whose gradient is *measured* to be adequate |
| **report** | `mission_capable`, `observed` and `off_axis_m` either way, at 5 seeds, median and worst |

⛔ Not judged on `hop_mean | observed` (measures geometry) and not on
`chain_occluded` (confounds with hop count, corr = 0.963).

⚠️ **A secondary prediction worth recording before the run**, because it is what
`Φ_cover` was built to do and it is falsifiable: `off_axis_m` should fall from
266 m toward B0's 107 m. If `observer_range_m` moves and `off_axis_m` does not,
the gain came from `Φ_standoff` alone and `w_cover` is carrying nothing.

## Results

*(appended below; the declaration is not edited)*

## 📏 Result — measured 2026-09-01, CUDA, eval split, 5 seeds. ⛔ **KILL.**

| metric | control (`shipped`) | `PHI_V2` | Δ | rule |
|---|---|---|---|---|
| **`observer_range_m`** | **218.9** [188.5–230.2] | **207.0** [179.0–240.2] | **−11.8 m** | needed **≥ −20 m** ⛔ |
| `mission_capable` | 39.84 % [36.50–45.82] | 40.76 % [37.46–44.33] | +0.92 pp | no regression ✅ |
| `off_axis_m` | 266.0 [219.0–288.3] | 252.5 [246.9–264.9] | −13.5 m | prediction: toward 107 ⛔ |
| `observer_tenure` | 39.2 | 42.9 | +3.7 | — |
| `role_entropy` | 0.51 | 0.54 | +0.03 | — |

Per-seed `observer_range_m` — control 188.5 · 202.4 · 218.9 · 219.5 · 230.2;
`PHI_V2` 179.0 · 202.7 · 207.0 · 232.3 · 240.2. **Heavily overlapping**, and two
treatment seeds are worse than the control's worst.

**The kill rule is met outright.** The observer moved **11.8 m of a 130 m gap**
against B0's 88.7 m.

### ☠️ This is the seventh Φ null, and it is the one that closes the axis

The six before it were retro-explained in `docs/inherited/DECISIONS.md` as
arithmetic rather than mechanism: *"Each scaled a potential whose directional
gradient in the operating regime was 0.013–0.03 per step. The reason they failed
is arithmetic, not mechanism."*

📏 **`PHI_V2` fixed the arithmetic and the behaviour still did not move.** The
closing gradient went **0.0133 → 0.0774 per 8 m step (5.8×, from 0.25× the energy
bar to 1.42×)**, and the recall gradient for a role-less drone went from **exactly
0.0000** to +0.0078/+0.0010/+0.0003 with a whole-trip value of +0.34 to +0.47.
Every one of those is measured, on this trainer, on the eval split.

🔒 **So the retro-explanation is falsified.** The potential's gradient magnitude
is **not** the mechanism behind the observer stand-off. That is a stronger and
more useful negative result than the six that preceded it, because it was
declared in advance against a mechanism that had been *measured to be adequate*
rather than assumed.

⛔ **Do not propose another Φ intervention without a new mechanism.** "Make the
potential steeper / redistribute it / centre it elsewhere" is now closed on
evidence, at 5 seeds, with the gradient measured before and after.

⚠️ `mission_capable` +0.92 pp and `observer_tenure` +3.7 are *not* findings —
both sit inside the seed ranges, and PBRS cannot change the optimum in any case.
Ships at the shipped potential; `PHI_V2` stays available and off.
