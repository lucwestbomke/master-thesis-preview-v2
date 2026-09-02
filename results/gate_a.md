# Gate A — velocity setpoints. Measured 2026-09-02, CUDA, eval split, 5 seeds.

Rule declared in [`../PLAN.md`](../PLAN.md) before the run:

> **keep** — steps at the speed cap fall below **20 %** *and* steps at the map
> boundary below **5 %**, regardless of `mission_capable`.
> **kill** — both pathologies persist.

## ⛔ The keep-rule is not met

📏 `scripts/measure_potential.py`, eval split, stage 4, F4, deterministic policy:

| | > 24 m/s | at map boundary |
|---|---|---|
| rule | **< 20 %** | **< 5 %** |
| B0 | 3.6 % | 2.6 % |
| **acceleration**, seed 0 / seed 3 | 10.8 % / 26.1 % | 19.7 % / 14.5 % |
| **velocity**, seed 0 / seed 3 | **0.1 % / 0.4 %** ✅ | **41.6 % / 72.6 %** ⛔ |

🔍 **The speed-cap pathology is solved outright** — 26.1 % → 0.4 %, and the
inherited figure was 57 %. A held action now converges to that velocity and stops
there, so there is nothing to saturate. That is the interface change doing
exactly what it was predicted to do.

☠️ **The boundary pathology got dramatically worse** — 14.5 % → **72.6 %**. The
swarm flies to the map edge and stays there.

## And it costs 18 pp, with disjoint seed ranges

| | per seed | median |
|---|---|---|
| acceleration (control) | 36.5 · 38.9 · 39.8 · 41.3 · 45.8 | **39.8 %** |
| **velocity** | 10.8 · 18.6 · 21.5 · 24.8 · 32.1 | **21.5 %** |

**Velocity's best seed (32.1) is below acceleration's worst (36.5).** The ranges
are completely disjoint, which is the same standard by which RQ2's
permutation-invariance result was confirmed. `observed` falls 59.6 → 40.2 %,
`observer_range_m` rises 218.9 → 381.8 m, `off_axis_m` rises 266 → 461 m.

⚠️ **The declared rules do not partition the outcome space**, and this is the
second time in this project that has happened (`trainer_validation.md` records the
first). `keep` needs both pathologies resolved; `kill` needs both to persist. Here
**one resolved completely and the other roughly quadrupled**. Recorded as what it
is rather than resolved by picking whichever half is convenient. What is not in
doubt is that the keep-rule fails and the change does not ship on this evidence.

## 🔍 The mechanism, and the random baseline is the evidence for it

📏 `random`, same harness, same split, before and after the action-space change:

| | `mission_capable` | `observed` |
|---|---|---|
| random, acceleration | 10.6 % | 21.9 % |
| random, **velocity** | **5.2 %** | **5.2 %** |

**A uniform random policy is half as good and sees the target a quarter as
often.** Nothing about control changed for a random policy — only what its noise
*does*.

Under **acceleration**, Gaussian action noise integrates **twice**: velocity
random-walks at ~2.4 m/s per tick, reaches the 25 m/s cap in ~100 ticks, and the
drone then moves *ballistically* for hundreds of metres. Under **velocity
setpoints** the same noise integrates **once**: each tick commands ~15 m/s in an
independent direction, giving ~6 m of displacement per tick and ~150 m of
*diffusive* travel over a 600-step episode.

🔒 **So the velocity interface is easier to control and harder to explore**, and
those two effects were changed together. The 18 pp is not cleanly attributable to
the control interface.

## What happens next was pre-declared

`PLAN.md`'s own kill branch names the follow-up: *"the next suspect is
exploration — `entropy_loss_scale` has been 0.0 throughout."* That is written
into the plan **before** this run, so pursuing it is the declared path and not a
post-hoc rescue of a failed change.

⚠️ **But note the trap.** The principled fix for a velocity interface is
*temporally correlated* exploration (the action *is* the velocity, so correlated
noise is sustained motion). That breaks the i.i.d. assumption the PPO ratio rests
on: the log-probability stored would no longer be the density the action came
from. **That is exactly the shape of skrl's `clip_actions` bug** — a mismatch
between the sampled action and the density used in the ratio, which inverted
learning and cost 30 % → 4.6 %. Do not implement correlated exploration without
a probe that would catch it.

⛔ Until that resolves, **acceleration remains the shipped action space** and
contribution C3 is an open question rather than a delivered result.
