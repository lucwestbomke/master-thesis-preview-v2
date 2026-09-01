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
