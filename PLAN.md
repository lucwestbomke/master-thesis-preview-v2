# Contested Relay — the plan

**Drafted 2026-08-27.** Presentation version:
<https://claude.ai/code/artifact/996e832f-ab50-47ec-9da9-b4d8941e73e6>

`docs/INHERITED.md` records *what is already known*. This file records *what
happens next*, in order, with the decision rules declared **before** the runs that
resolve them — because two claims in the predecessor project were overturned by
reading a single run, and a rule invented after the fact is not a rule.

---

## 1. The reframe

Sixteen months produced an excellent simulator and a learned policy that loses to
a 200-line scripted baseline by **16.1 pp**. Six pre-declared interventions ran at
five seeds each; all six are nulls.

🔍 **The framing is what is wrong.** The environment is static, the jammer is a
fixed barrage emitter that rides the target and does nothing else, the objective
is fully specified, and the geometry is something a competent engineer can reason
about. **That is what heuristics are for.** B0 winning is the expected outcome,
and no amount of reward shaping changes it.

Learning earns its keep when the environment *adapts to you*. B0 hill-climbs
deterministically on clearance and capacity features, so an adversary that learns
will find that response function and exploit it. A policy trained against a
learning adversary need not be exploitable the same way.

⛔ **So stop asking whether learned control beats the heuristic on the static
task.** Ask how far each policy degrades when the adversary best-responds to it.
That is a relative comparison, which is what `src/baselines/evaluate.py` was
already built for.

---

## 2. Contributions

Three claims, each able to fail on its own.

**C1 — the adversary learns.** A directional jammer that chooses where to point.
📏 The window is measured at **11 pp**: B0 falls 58.6 % → 47.7 % under per-step
retargeting. See §4.

**C2 — the policy runs on the drone.** ONNX → TensorRT on a Jetson Orin Nano 8 GB.
Latency, **p99 jitter** and power at `N` = 3/5/8 in FP32, FP16 and INT8, against
the 400 ms control period. The research question inside it: **does quantisation
degrade coordination more than it degrades control?** Push the INT8 policy through
the same harness and compare `role_entropy` and `observer_range_m` against
`mission_capable`.

**C3 — the action space is velocity, not acceleration.** `docs/REDUCTION.md`
task 1, and Gate A below.

---

## 3. The jammer, J0–J4

The jammer enters SINR **at the receiver** (`core.py`: `denom_mw` is indexed by
the listening node and broadcast across transmitters). That is correct physics —
jamming raises a noise floor at an antenna, and there is no such thing as jamming
an outgoing signal.

🔍 It inverts the obvious strategy. The feed flows observer → relays → MCV and the
observer *never listens*, so the drone closest to the jammer, most exposed and
most illuminated, is the one target that cannot be hurt. 📏 Measured: pointing at
the observer scores **59.9 %** against isotropic's 58.6 % — **worse than not
aiming at all**.

📏 And the MCV is not a degenerate target. Isotropic jammer power received:
drones **−58 to −61 dBm**, MCV **−91.9 dBm** — 31 dB of shelter, because the
drones are airborne with line of sight while the MCV sits in ground clutter a
kilometre away and takes the NLoS branch.

| rung | jammer | what it isolates |
|---|---|---|
| **J0** | none | exists |
| **J1** | isotropic, fixed power | exists — the inherited jammer. B0 scores 58.6 % |
| **J2** | directional, **fixed** target | separates *directionality* from *adaptivity*. ⛔ Not optional: without it, "the adaptive jammer beat B0" might only mean "a beam beat B0" |
| **J3** | directional, **greedy-adaptive** — retarget the chain's weakest receiver each step | adaptive without learning. Scripted, reproducible, no training loop. 📏 47.7 % |
| **J4** | directional, **learned** by alternating best response with an opponent pool | the stretch. ⛔ **Not built** — deliberately absent from `EnvConfig.jammer`, because an enum value that silently emits an arbitrary bearing is the half-specified condition `BLOCK_F.md` decision 5 exists to prevent |

📏 **Built and measured 2026-09-02** — `src/env/core.py`, `src/env/test_jammer.py`,
[`results/j_ladder.md`](results/j_ladder.md). ⚠️ **J3 as specified here is
weaker than J2**, which was not anticipated: see that file before running Gate B.

🔒 **J3 is sufficient to test the claim.** If the question is whether learned
policies degrade less than B0 under an adaptive adversary, a greedy scripted
adversary answers it — and it already extracts the full 11 pp. J4 makes the claim
stronger and supplies the self-play line, but **it is not load-bearing**.

### The beam

No array processing. The 3GPP element pattern, TR 38.901 §7.3:

```
A(θ) = −min[ 12·(θ/θ_3dB)² , A_max ]   dB,   θ_3dB ≈ 25°,  A_max = 30 dB
```

In `_jammer_mw` it is one `atan2` for the bearing to each radio node, one angle
wrap, one polynomial — about eight lines. Path loss and line-of-sight machinery
are untouched.

- **Continuous in angle**, so the jammer's policy has a usable gradient. A hard
  cone makes the action effectively discrete.
- **Azimuth only.** Drones at 40–80 m, jammer at ground level a kilometre out:
  elevation angles are ~5° and elevation discrimination buys nothing.
- ⛔ **Beamwidth is not an action.** It smuggles the power axis back in — a wide
  beam *is* a barrage jammer and a narrow one is a spotlight.
- ⛔ **Power stays fixed.** A barrage jammer with no cost always plays maximum.
- **Position stays fixed** to the target for v1. A second road-mobile emitter is
  the v2 escalation if the swarm beats the first one too easily.

**What the jammer sees:** bearings and received powers of the drones'
transmissions, not their positions. Realistic ESM, nearly free because the
received-power matrix is already computed. Keep a ground-truth-fed variant as the
ablation that bounds what the restriction costs — the same idiom as `b0` against
`b0-oracle`.

⚠️ **The simplification to state, not to fix.** The model treats the relay as a
one-way rate-limited sensor feed, so jamming affects receivers only. A real link
layer carries acknowledgements and routing control, so a jammed observer would
eventually lose its link too. Requiring `min(SINR_ij, SINR_ji)` to clear would
capture that in one line — but measure before adopting it, because it may hand
the jammer the degenerate "kill the observer" attack that geometry currently
denies. For a video downlink, one-way is the defensible abstraction.

---

## 4. Gates, declared before the runs

Every gate is judged on the **worst seed**, at >= 5 seeds.

### Gate A — velocity setpoints

The question is not whether `mission_capable` rises. It is whether the measured
pathologies are artefacts of the action space. Both answers are useful.

| | rule |
|---|---|
| **keep** | steps at the speed cap fall below **20 %** *and* steps at the map boundary below **5 %** — regardless of `mission_capable`. Velocity setpoints are the more faithful interface anyway; the pathologies are the evidence they were needed |
| **kill** | both pathologies persist. The saturation is then a property of the policy, not the parameterisation, and the next suspect is exploration — `entropy_loss_scale` has been 0.0 throughout |
| **report** | `mission_capable` either way. A large rise says the failure was control all along; no movement with the pathologies resolved says it was coordination |

### Gate B — adversarial robustness

Primary readout is the **exploitability gap**: how far each policy's
`mission_capable` falls between **J1** and the strongest adversary reached — J3 at
minimum, J4 if it trains. 📏 The window is 11 pp.

| | rule |
|---|---|
| **confirm** | B0's exploitability gap exceeds the adversarially-trained policy's, at 5 seeds, on the worst seed |
| **refute** | B0 degrades no more than the learned policies. The scripted baseline is then robust as well as strong — a legitimate reportable result about the task rather than about the method |
| **control** | ⛔ **J2 is required.** Without a fixed-target directional rung, a result at J3 or J4 cannot distinguish "the adversary adapted" from "the adversary had a beam" |
| **report** | the full cross-product, not just the diagonal. A policy robust only to the adversary it trained against has overfitted to one opponent, and the off-diagonal is the only place that shows |

⛔ **Not** `hop_mean | observed` (it measures geometry) and **not**
`chain_occluded` (it confounds with hop count, `corr = 0.963`).

### Gate C — quantisation and coordination

| | rule |
|---|---|
| **finding** | `role_entropy` and `observer_range_m` degrade proportionally more than `mission_capable`. Coordination is then more quantisation-sensitive than control — a new, deployment-relevant claim |
| **null** | degradation is proportional. Report latency and power as an engineering result and move on; it still supplies the sim-to-real line, which is most of its value |
| **report** | which architectures export at all. "The GNN buys +0.4 pp and is substantially harder to deploy" is a good sentence in a paper written for people who fly things |

---

## 5. Phases

| | when | what |
|---|---|---|
| **1** | Sep 2026 | `docs/REDUCTION.md` tasks 1–4. Gate A. In parallel — it depends on nothing else — export to ONNX and get one architecture running under TensorRT on the Orin Nano. ⚠️ Doing the export *now* is deliberate: PyTorch Geometric exports badly, and finding that out early is worth more than finding it out late |
| **2** | Oct–Nov 2026 | The beam pattern, then **J2 and J3** — scripted, days rather than weeks. Only then attempt **J4**. `docs/REDUCTION.md` task 5 (own PPO) lands here |
| **3** | Dec 2026 – Jan 2027 | The full policy × adversary cross-product at 5 seeds → Gate B. FP32/FP16/INT8 through the same harness → Gate C. Optional: a coarse bearing-to-strongest-emitter observation |
| **4** | Feb–Mar 2027 | 🔒 **Freeze**, recorded with a date and scope. Close the TR 36.777 NLoS intercept — one human reading of one table, and everything rests on it. `docs/REDUCTION.md` task 7. First paper draft |
| **5** | Apr–Aug 2027 | Submission and thesis. **IEEE RA-L** (rolling, no deadline pressure) with **IROS 2027** as the alternative — verify its deadline early. A NeurIPS or ICML workshop submission sooner, as insurance and as a first publication |

---

## 6. Risks

| risk | mitigation |
|---|---|
| **Alternating best response cycles.** The normal failure mode of J4 | the ladder stops at **J3**, which is adaptive without being learned and already extracts the 11 pp. Gate B runs against it unchanged; J4 becomes future work rather than a hole in the results |
| **PyTorch Geometric will not export cleanly.** Scatter ops, dynamic shapes | attempt it in Phase 1, not Phase 3. If the GNN cannot be deployed that is a *reported result*, and a useful one — its measured advantage over DeepSets is a null |
| **TR 36.777 NLoS intercept is wrong.** Every number re-derives | verify in Phase 1 if possible, Phase 4 at the latest. Cost if right: an afternoon. Cost if wrong and found in month ten: the paper |
| **Seed spread swamps every effect.** 📏 60–78 % over five runs, bimodal, undiagnosed | judge on the worst seed meanwhile; diagnose once the trainer is ours and instrumentable |
| **Scope.** This is more ambitious than what exists | the cuts in `docs/REDUCTION.md` are not optional — they are what pays for the adversary |

---

## 7. What is deliberately not being built

- **A bigger map.** Longer episodes, harder credit assignment, a full geometry
  rebuild — spending free compute to make the failing thing harder. 📏 Compute is
  not the constraint; a 10 M-step run costs 2.2 minutes.
- **A better channel model.** It is already past the point where it limits the
  paper. Adversarial pressure on the channel we have is worth more than a more
  faithful channel nobody attacks. *One exception worth considering:* a knife-edge
  diffraction term on the clearance margin already computed (ITU-R P.526, ~20
  lines). That removes a **discontinuity** — the 16–30 dB cliff at clearance = 0 —
  rather than adding fidelity, and it smooths the landscape where the swarm
  operates.
- **Flying below 40 m.** TR 36.777 stops at 22.5 m and drones start sitting inside
  building boxes.
- **A second city.** ⚠️ But worth 30 minutes before accepting: the predecessor cut
  it because LoD2 is a Hessen-only service — yet Hessen contains Wiesbaden,
  Darmstadt, Kassel and Offenbach. That reasoning kills a city in Bavaria; it
  should not kill a second box 30 km down the road. The current eval split is
  held-out routes through *the same buildings*, which a reviewer will correctly
  call in-distribution. Held-out map tiles inside the existing box are free
  regardless.
