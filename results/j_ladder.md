# The adversary ladder J0–J3 — built 2026-09-02

`PLAN.md` §3. `EnvConfig.jammer` selects the rung; `src/env/test_jammer.py` pins
eleven properties. 🔒 The rung is **orthogonal** to the fidelity ladder and to the
curriculum: `channel_jammer` (F3/F4) decides whether the emitter is in the SINR
denominator, the curriculum's `jammer_on` decides whether it is on this episode,
and the rung decides only the **pattern**. Collapsing them would confound RQ1's
jammer rung with the curriculum ramp.

## What was built

**The beam.** 3GPP TR 38.901 §7.3 element pattern, azimuth only:
`A(θ) = −min[12·(θ/25°)², 30 dB]`, added to a 12 dBi peak — the 25°/12 dBi the
inherited probe used. Continuous in angle so a learned rung would have a usable
gradient. ⛔ Beamwidth is not an action and power is fixed; both smuggle the
transmit-power axis back in, and that axis is three framings and three nulls.

**Aiming, with one step of latency.** 🔒 Required rather than tolerated: the
emitter changes capacity, capacity changes the routed chain, and the chain is
what J3 targets — aiming at *this* step's chain would be circular. It is also the
honest model, since `PLAN.md` gives the jammer an ESM receiver that observes and
then reacts.

**J3 targets the receiver** of the lowest-capacity edge on the chain. 🔍 Targeting
the *receiver* is what makes it physically coherent — the jammer raises a noise
floor at whatever antenna is listening, so there is no such thing as jamming an
outgoing signal. The observer, a pure source, is therefore never selected, which
is the geometry behind the inherited "pointing at the observer scores 59.9 %
against isotropic's 58.6 % — worse than not aiming at all".

⛔ **J4 is deliberately absent from the enum.** `PLAN.md` marks it the stretch and
not load-bearing, and a rung that silently emits an arbitrary bearing is the
half-specified condition `BLOCK_F.md` decision 5 exists to prevent. The seam it
will use is `_jammer_boresight`.

## 📏 First measurement — B0, CPU, 32 eval episodes, F4, stage 4

⚠️ A smoke measurement, not a result: one seed, 32 episodes, CPU. Reported
because the **ordering** is what matters and it is not the expected one.

| rung | `mission_capable` | mean e2e | `observed` |
|---|---|---|---|
| **J0** none | 60.2 % | 24.0 Mbps | 94.2 % |
| **J1** isotropic | 54.3 % | 21.1 Mbps | 94.2 % |
| **J2** beam, fixed on the MCV | **41.8 %** | 15.8 Mbps | 94.2 % |
| **J3** beam, greedy retargeting | **44.5 %** | 16.7 Mbps | 94.2 % |

✅ `observed` is **identical at every rung**, which is the property
`test_jammer.py` pins: the sensor is geometry and the emitter must not touch it.
Without that the exploitability gap would be uninterpretable.

## ☠️ J3 is *weaker* than J2, and that has to be resolved before Gate B

The ladder is **non-monotone**: greedy adaptation is 2.7 pp *less* damaging than a
beam parked on the MCV. That inverts the inherited probe's ordering (MCV 50.5 %,
best-target-each-step 47.7 %).

🔍 **The cause is that `PLAN.md`'s J3 specification and the probe's are different
adversaries**, and the difference was not visible until both existed:

* `PLAN.md` §3 specifies J3 as *"retarget the chain's weakest receiver each
  step"* — a **greedy heuristic**, which is what is built.
* The probe that measured **47.7 %** is described as *"best target each step"* —
  an **exhaustive best response**, which evaluates every candidate target and
  picks the one that most reduces end-to-end capacity.

Two mechanisms plausibly explain why the heuristic underperforms, and they are
testable rather than decorative:

1. **The weakest link is not the most damageable one.** A link already at the
   bottleneck may be nearly saturated in SINR terms, while 12 dBi aimed at the
   MCV — which every chain must reach, and which enjoys 31 dB of ground-clutter
   shelter that the beam exactly offsets — breaks *all* paths at once.
2. **Greedy retargeting chases its own effect.** Jam node X → X's link degrades →
   the router reroutes → Y is now weakest → jam Y → X recovers. The adversary
   oscillates and commits to nothing, while a fixed beam sustains pressure.

⚠️ **Consequence for Gate B.** Its readout is the *exploitability gap* between J1
and **the strongest adversary reached**. An adaptive rung that is weaker than the
non-adaptive control cannot support the claim "learned policies degrade less when
the adversary adapts" — it would measure degradation under a weak adversary and
attribute it to adaptivity.

🔒 **So Gate B does not run until this is resolved**, and the resolution is a
decision about what "the adversary" *is*, which is the most load-bearing choice
in the thesis. The options, in order of preference:

| | option | cost |
|---|---|---|
| **A** | add **J3-BR**, the exhaustive best response the probe actually measured: for each of the `n_radio` candidate targets recompute capacity and routing, take the argmin of end-to-end capacity | ~6× the channel+routing cost per step, which is small next to occlusion. This is the adversary Gate B needs |
| **B** | keep greedy J3 and report the non-monotonicity as a secondary finding | free, but leaves Gate B without a strong adaptive rung |
| **C** | fix the greedy criterion | ⛔ inventing a heuristic to make a number move, with no measurement behind it |

📏 The non-monotonicity is worth reporting **either way**, and it rhymes with a
result this project already has: the fidelity ladder is also non-monotone in
difficulty (F1 is harder than F4). "Adding capability to the adversary does not
monotonically increase the pressure it applies" is a real and counter-intuitive
finding about adversarial evaluation.

---

## 📏 J3B built and measured — and it does **not** rescue adaptivity

`J3B` is the exhaustive best response: for each of the `n_radio` candidate
boresights, recompute what end-to-end capacity *would* be, and take the argmin.
`test_jammer.py` asserts the selection really is the argmin, and that it is never
worse than holding the MCV *on the geometry it was given*.

| rung | `mission_capable` | mean e2e | targets chosen (drones … MCV) |
|---|---|---|---|
| **J0** none | 60.2 % | 24.0 | — |
| **J1** isotropic | 54.3 % | 21.1 | — |
| **J2** beam, fixed on the MCV | **41.8 %** | 15.8 | fixed |
| **J3** beam, greedy | 44.5 % | 16.7 | 2·3·4·6·4 · **13** |
| **J3B** beam, best response | **42.2 %** | 15.7 | 1·2·5·7·5 · **12** |

⚠️ B0, CPU, **32 eval episodes, one seed**. A smoke measurement. The *ordering*
is what is reported, not the levels.

## ☠️ Adaptivity does not help this adversary. At all.

The strongest rung is **J2 — the one that does not adapt.** A per-step best
response ties it (42.2 % against 41.8 %) and the greedy heuristic is 2.7 pp
worse. That is not a bug in J3B: it provably picks the argmin of end-to-end
capacity every step, and it selects the MCV only 12 times in 32 — so it is
genuinely exploiting the freedom and gaining nothing for it.

🔍 **The mechanism is myopia, not latency.** At 1 km an aiming lag of one 0.4 s
tick is ~0.6° of bearing against a 25° beamwidth — negligible. What is not
negligible is that **a one-step best response is not an optimal adversary against
an opponent that adapts.** B0 hill-climbs on observable clearance and capacity,
so a jammer that re-optimises every step hands it a *moving* problem it can track,
while a beam parked on the MCV forces a **persistent** detour. Committing beats
re-optimising. This is the classic gap between greedy and optimal play in a
repeated game, and here it is worth ~0 to 3 pp.

### ⛔ What this does to the thesis, stated plainly

`PLAN.md`'s reframe is: *"how far does each policy degrade when the adversary
adapts to it?"* — and the answer so far is **that adaptation, as specified, does
not increase adversarial pressure**. Gate B's exploitability gap would then be
measuring a beam, not an adaptation, and the J2 control row exists precisely to
catch that.

⚠️ **This does not sink the thesis, and it changes what J4 is for.** `PLAN.md`
lists J4 (a *learned* jammer) as "the stretch" and explicitly **not
load-bearing**, on the reasoning that J3 already extracts the full 11 pp. 📏 That
reasoning rested on the inherited probe's 47.7 %, which is now shown to belong to
an adversary no stronger than a fixed beam. **A learned, non-myopic adversary is
therefore no longer optional** — it is the only rung left that could make
"adaptive" mean something, because it is the only one that can *commit* to a
strategy over time rather than re-optimise each step.

🔒 **Before anything is concluded, this needs 5 seeds on CUDA.** The J2/J3B gap is
0.4 pp on 32 episodes and one seed; the J2/J3 gap is 2.7 pp on the same. Neither
is a finding yet. The commands are in the session notes; nothing downstream runs
until they are measured properly.

📏 Worth noting either way: this is the **second** non-monotone ladder in this
project. The fidelity ladder is non-monotone in difficulty (F1 harder than F4)
and the adversary ladder is non-monotone in pressure. *"Adding capability to a
component does not monotonically increase its effect"* is becoming a theme rather
than an oddity, and it is a genuinely useful caution about how ablation ladders
are read.
