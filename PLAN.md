# Contested Relay — the plan

**Drafted 2026-08-27. Rewritten 2026-09-02** around a single claim, because the
deliverable changed from an 80-page thesis to a paper and a paper carries one
claim, not three independent contributions.

`docs/INHERITED.md` records *what is already known*. This file records *what
happens next*, in order, with the decision rules declared **before** the runs that
resolve them — because two claims in the predecessor project were overturned by
reading a single run, and a rule invented after the fact is not a rule.

> ⚠️ **Gates B and C below are reproduced verbatim from the 2026-08-27 draft.**
> They were declared before their runs and are not edited here. Gate A has since
> resolved; its verdict is recorded in §5 and in
> [`results/gate_a.md`](results/gate_a.md).

---

## 1. The claim

> 🔍 **Exploitability, not capability, is the right axis on which to compare
> learned and scripted multi-agent policies.**

A swarm of `N = 5` UAVs observes a moving ground target and relays the feed to a
command vehicle over a multi-hop chain at >= 15 Mbps, while a jammer degrades
links. 📏 **The scripted baseline B0 wins the static task by 15.0 pp** (55.7 %
against the GNN's 40.7 %, eval split, 5 seeds), and — see §3 — that is now a
**settled premise of this work rather than an open question.**

So the protagonist of this paper is *the strongest available policy*, and it
happens to be scripted. The question asked of it is not how capable it is but
**how much of that capability an adversary can take away**, and whether a policy
that learned its behaviour is exploitable in the same way and to the same degree.

⛔ **Stop asking whether learned control beats the heuristic on the static task.**
It does not, the reason is measured, and §3 closes the axis.

---

## 2. The four objectives

One arc: **an adversary that adapts → a policy co-trained against it → running on
the hardware that has to fly it.**

### RQ1 — Does the heuristic's advantage survive an adversary that adapts to it?

The **exploitability gap**: how far each policy's `mission_capable` falls between
J1 and the strongest adversary reached. B0 hill-climbs deterministically on
observable clearance and capacity features, so it has a response function an
opponent can find and exploit. A learned policy need not be exploitable the same
way. Resolved by **Gate B** (§5), which is unchanged from its declaration.

### RQ2 — Is adversary capability monotone in adversarial pressure?

📏 **The answer already measured is *no*, and it is counter-intuitive.**
J2 (a beam parked on the MCV) is **stronger** than J3B (an exhaustive per-step
best response) and than J3 (greedy retargeting) — 41.8 % against 42.2 % and
44.5 %. Committing beats re-optimising, because a jammer that re-optimises every
step hands a hill-climbing opponent a *moving* problem it can track while a
parked beam forces a **persistent** detour.

🔍 This is the **second** non-monotone ladder in the project: the fidelity ladder
is non-monotone in *difficulty* (📏 F1 is harder than F4, 27.9 % against 56.0 %
under B0). *"Adding capability to a component does not monotonically increase its
effect"* generalises into a methodological caution about how ablation ladders are
read, and it is the finding most likely to outlive the rest of this paper.

🔒 **RQ2 is the insurance.** It stands whether or not J4 trains, and it gets
*stronger* if J4 fails — "we trained a learned adversary and it still did not beat
a parked beam" is a louder version of the same claim than the scripted rungs can
make alone.

⛔ **Blocked on one measurement.** The J2/J3/J3B ordering is currently a smoke
test — one seed, 32 episodes, CPU. [`results/j_ladder.md`](results/j_ladder.md)
declares that nothing downstream runs until it is re-measured at **5 seeds on
CUDA**. The J2/J3B gap is 0.4 pp and the J2/J3 gap is 2.7 pp; neither is a
finding yet. **This is the first run of the sequence in §7.**

### RQ3 — Does adversarial co-training produce robustness, or overfitting to the training opponent?

J4: a **learned** jammer, trained by alternating best response against an
**opponent pool**. Then the full policy × adversary cross-product.

🔒 **The off-diagonal is the result, not the diagonal.** A policy robust only to
the adversary it trained against has overfitted to one opponent, and the
off-diagonal is the only place that shows. Gate B already requires this and it is
not negotiable down to a diagonal for budget reasons — 📏 a 10 M-step run costs
2.2 minutes, so the cross-product is affordable by a wide margin.

📏 **J4 is now load-bearing, which it was not on 2026-08-27.** The original plan
called it "the stretch" and "explicitly not load-bearing", reasoning that J3
already extracted the full 11 pp of the inherited probe's window. That reasoning
rested on the probe's 47.7 %, which J3B has since shown belongs to an adversary
no stronger than a fixed beam. A learned, **non-myopic** adversary is the only
rung left that can commit to a strategy over time rather than re-optimise each
step, and therefore the only one that can make "adaptive" mean anything.

### RQ4 — Does the robustness survive the airframe?

ONNX → TensorRT on a Jetson Orin Nano 8 GB. Latency, **p99 jitter** and power at
`N` = 3/5/8 in FP32, FP16 and INT8, against the 400 ms control period. The
research question inside it: **does quantisation degrade coordination more than it
degrades control?** Push the INT8 policy through the same harness and compare
`role_entropy` and `observer_range_m` against `mission_capable`. Resolved by
**Gate C** (§5), unchanged from its declaration.

🔧 **Pure Python.** TensorRT's Python bindings, `trtexec` for the benchmark,
`tegrastats` for power. No C++ or Rust runtime is built for this paper.

---

## 3. The premise: B0 wins the static task, and the axis is closed

This section exists so that nobody — including a future reader of this repo —
re-opens it. **Four independent lines of evidence**, and the fourth is
structural rather than empirical.

### 📏 1. The gap is `observed`, and nothing else

Conditioned on holding a sightline, the learned policy converts it into mission
capability **exactly as well as B0 does**:

| conditioned on holding a sightline | random | MLP | DeepSets | GNN | B0 |
|---|---|---|---|---|---|
| `capable / observed` | 0.489 | 0.580 | 0.583 | **0.620** | **0.617** |

The entire gap is `observed` — 📏 59.6 % against 93.6 %, observer range 218.9 m
against 88.7 m, `role_entropy` 0.51 against 0.10. **One behaviour is missing: a
drone commits to closing and the others commit to extending behind it.**
`hop_mean | observed` is a *shadow* of observer position, not a separate relay
failure — six interventions moved it over the range 1.86–1.93 against random's
1.83.

### 📏 2. The objective is not the problem — B0 wins the reward too

Re-read 2026-09-02 from [`results/rq2_stageB.jsonl`](results/rq2_stageB.jsonl);
no new runs. `RolloutMetrics.summary()` has always reported `episode_return` and
every policy goes through the same harness. Eval split, N = 5, 5 seeds, CUDA:

| policy | `mission_capable` | `episode_return` |
|---|---|---|
| random | 10.6 % | −170.0 |
| mlp | 30.7 % | −3.0 |
| deepsets | 39.8 % | 71.8 |
| gnn | 40.7 % | 85.8 |
| **B0** | **55.7 %** | **222.9** |

📏 **B0 beats the best learned policy by 2.6× on the reward that policy is trained
on.** And across 20 rows spanning 5 policies, 3 swarm sizes, both action spaces
and both shaping variants, `episode_return` and `mission_capable` rank-correlate
at **ρ = 0.987**.

🔍 **The margin is understated.** PBRS pays `(γ−1)·Φ` per step for *holding* a good
state, so shaping is a drag proportional to `Φ` and therefore largest for the best
policy: 📏 B0's mean shaping is **−0.018/step** while the learned policy receives
**7× more** shaping amplitude. B0 wins by 137 return points while collecting
*less* shaping than its opponent. Under the mission term alone the gap is wider.

⛔ **So the "the learner is winning at its true objective and losing at a metric it
was never given" hypothesis is dead.** It loses at both, consistently. The reward
points the right way.

⚠️ **What ρ = 0.987 does *not* show.** Rank-alignment says the reward *points*
correctly, not that its *shape* is easy to optimise. The remaining failure is
optimisation, not specification.

### 📏 3. Eight interventions, eight nulls, and the last one had a measured mechanism

`w_hold`, `w_relay`, `d_ref` 1500→400, `potential_scale` 10→30, recurrence, the
agent-specific critic, `PHI_V2` — and separately the velocity action space.
`docs/inherited/DECISIONS.md` retro-explained the first six as *arithmetic, not
mechanism*: each scaled a potential whose directional gradient in the operating
regime was 0.013–0.03 per step.

📏 [`results/phi_v2_gate.md`](results/phi_v2_gate.md) **falsified that
retro-explanation.** `PHI_V2` raised the closing gradient **5.8×** (0.0133 →
0.0774 per 8 m step, from 0.25× the energy bar to 1.42×) and turned an *exact
zero* into a real gradient for the four drones out of five that had none — and the
observer moved **11.8 m of a 130 m gap**. The arithmetic was fixed and the
behaviour did not move.

⛔ **The Φ axis is closed.** Do not propose a ninth shaping intervention. Do not
propose a second action space. Both were pre-declared, both were measured, both
failed, and continuing is the "always one more thing to try" failure this project
has been warned about since `ROADMAP.md` was written.

### 📏 4. And the reason is structural: the advantage cannot tell drones apart

Measured 2026-09-02, declared before the run —
[`results/credit_assignment.md`](results/credit_assignment.md),
`scripts/measure_credit.py`.

One value per global state is broadcast across `N` rows, so
`A[t,b,i] = G[t,b,i] − V[t,b]` and therefore `Var_i(A) = Var_i(G)` **exactly**.
That between-drone variance is the entire budget of drone-differentiating credit:
whatever the policy gradient knows about *which drone should do what*, it knows
through that and nothing else.

📏 Decomposing the per-drone return by the law of total variance, eval split,
stage 4, F4/J1, 64 envs × 600 steps, 3 seeds:

| policy | **differentiable share** |
|---|---|
| random | **0.04 %** |
| B0 | **0.11 %** |
| GNN | **0.16 %** |

🔒 The declared rule was `< 5 %` confirms. Measured **0.04–0.16 %**, two orders of
magnitude inside it, and unchanged at the `γλ = 0.947` horizon GAE actually sees.
**99.84–99.96 % of the return variance is identical across drones.**

⛔ **And the structural half is exact rather than measured.** `reward_terms()`
broadcasts `mission`, `idle`, `battery_variance` and `shaping` through `team(x)`,
so they cancel out of `Var_i` *exactly*; `w_relay` ships at 0.0. **Only `energy`
and `effort` — two motion costs — can differ between drones**, and the
measurement returns four exact zeros matching that term for term. Pinned by
`tests/test_measure_credit.py`.

☠️ **This converts eight nulls into one mechanism with a prediction:** an
intervention that modifies only team reward terms cannot change role
differentiation, because team terms contribute exactly zero to the only variance
that can distinguish drones. `w_hold`, `d_ref`, `potential_scale` and `PHI_V2`
are all team terms. ⚠️ It also reframes `w_relay`'s null — its 71× rise took the
differentiating share from ~0.04 % to ~2.9 %, which is *"still negligible"*, not
*"per-drone credit does not help"*.

🔍 **So the reward axis is closed structurally, not by exhaustion** — a better
reason than "we tried eight things", and it redirects the search to the critic
and the advantage, neither of which has been touched. ⛔ That is not a licence for
a ninth intervention: anything built on it needs its own gate, declared before
its own run, at 5 seeds, judged on the worst.

### 🔧 The one remaining probe, and it is timeboxed to one week

**Behaviour-clone B0, then PPO from that initialisation.** `B0Policy.act(flat)`
returns actions in the same `[-1, 1]` space the actor consumes, reads only
`obs["flat"]`, and is deterministic — so this is supervised regression against a
teacher samplable at 📏 3.17 M env-steps/s. A day of work.

🔒 **Its purpose is not to re-open §3.** It exists to make the learned protagonist
as strong as possible so that RQ1's comparison is interesting. If it works, the
exploitability numbers improve. If it does not, nothing downstream changes.

| outcome | reading |
|---|---|
| BC policy holds ~90 % `observed` and PPO improves from there | **exploration.** The coordination trap is real and BC-init is itself the fix |
| PPO decays back toward the 218 m stand-off | **credit assignment.** PPO is walking *down* a return gradient, 223 → 86, under a reward measured to point the right way. The named suspect is that the swarm is **one parameter-shared agent with a single centralised critic** (`src/training/ppo.py`), which cannot represent an advantage that differs by role — and role differentiation is exactly the missing behaviour |

⛔ **One week. Then stop regardless of outcome.**

---

## 4. The jammer, J0–J4

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

| rung | jammer | what it isolates | state |
|---|---|---|---|
| **J0** | none | exists | ✅ built |
| **J1** | isotropic, fixed power | the inherited jammer. B0 scores 58.6 % | ✅ built |
| **J2** | directional, **fixed** target | separates *directionality* from *adaptivity*. ⛔ Not optional: without it, "the adaptive jammer beat B0" might only mean "a beam beat B0" | ✅ built |
| **J3** | directional, **greedy-adaptive** — retarget the chain's weakest receiver each step | adaptive without learning | ✅ built |
| **J3B** | directional, **exhaustive best response** — argmin of end-to-end capacity over every candidate boresight | one-step-optimal without learning | ✅ built |
| **J4** | directional, **learned** by alternating best response with an **opponent pool** | RQ3. The only rung that can *commit* over time | ⛔ **not built** |

📏 First measurement, [`results/j_ladder.md`](results/j_ladder.md) — ⚠️ B0, CPU,
32 episodes, **one seed**. A smoke test; the *ordering* is what is reported:

| rung | `mission_capable` | mean e2e | `observed` |
|---|---|---|---|
| **J0** none | 60.2 % | 24.0 Mbps | 94.2 % |
| **J1** isotropic | 54.3 % | 21.1 Mbps | 94.2 % |
| **J2** beam on the MCV | **41.8 %** | 15.8 Mbps | 94.2 % |
| **J3** beam, greedy | 44.5 % | 16.7 Mbps | 94.2 % |
| **J3B** beam, best response | 42.2 % | 15.7 Mbps | 94.2 % |

✅ `observed` is **identical at every rung** — the property `test_jammer.py` pins.
The sensor is geometry and the emitter must not touch it; without that the
exploitability gap would be uninterpretable.

### The beam

No array processing. The 3GPP element pattern, TR 38.901 §7.3:

```
A(θ) = −min[ 12·(θ/θ_3dB)² , A_max ]   dB,   θ_3dB ≈ 25°,  A_max = 30 dB
```

- **Continuous in angle**, so J4's policy has a usable gradient. A hard cone makes
  the action effectively discrete.
- **Azimuth only.** Drones at 40–80 m, jammer at ground level a kilometre out:
  elevation angles are ~5° and elevation discrimination buys nothing.
- ⛔ **Beamwidth is not an action.** It smuggles the power axis back in — a wide
  beam *is* a barrage jammer and a narrow one is a spotlight.
- ⛔ **Power stays fixed.** A barrage jammer with no cost always plays maximum.
- **Position stays fixed** to the target. A second road-mobile emitter is the v2
  escalation if the swarm beats the first one too easily.

**Aiming carries one step of latency.** 🔒 Required rather than tolerated: the
emitter changes capacity, capacity changes the routed chain, and the chain is what
J3 targets — aiming at *this* step's chain would be circular.

**What the jammer sees:** bearings and received powers of the drones'
transmissions, not their positions. Realistic ESM, nearly free because the
received-power matrix is already computed. Keep a ground-truth-fed variant as the
ablation that bounds what the restriction costs — the same idiom as `b0` against
`b0-oracle`.

⚠️ **The simplification to state, not to fix.** The model treats the relay as a
one-way rate-limited sensor feed, so jamming affects receivers only. A real link
layer carries acknowledgements, so a jammed observer would eventually lose its
link too. Requiring `min(SINR_ij, SINR_ji)` to clear would capture that in one
line — but measure before adopting it, because it may hand the jammer the
degenerate "kill the observer" attack that geometry currently denies. For a video
downlink, one-way is the defensible abstraction.

---

## 5. Gates

Every gate is judged on the **worst seed**, at >= 5 seeds.

### Gate A — velocity setpoints. ⛔ **RESOLVED 2026-09-02: not met. Does not ship.**

Declared 2026-08-27; full record in [`results/gate_a.md`](results/gate_a.md).
The rule required steps at the speed cap below 20 % **and** steps at the map
boundary below 5 %.

📏 Velocity setpoints eliminated the speed-cap pathology outright (26.1 % → 0.4 %,
against an inherited 57 %) and **quadrupled** boundary occupancy (14.5 % → 72.6 %),
at a cost of **18.3 pp** with disjoint seed ranges. Raising exploration to escape
the boundary re-created the saturation the interface had removed. `EnvConfig.
action_space` defaults to `"acceleration"`; `"velocity"` is retained behind the
flag and exercised by tests on both branches.

✅ **The action-space question became a negative result with a mechanism**, and it earns one paragraph in
the paper: an interface whose exploration noise does **not accumulate** cannot
reliably escape the absorbing region the env creates by zeroing the velocity
component that hits a limit.

☠️ It also produced a **methodological finding that governs every learned number
here**: at `sigma ≈ 1.08` the same policy measured 1.3 % of steps above 24 m/s
when sampled and **69.9 %** when evaluated at the Gaussian's mean — a 54× gap. A
`mission_capable` number is only trustworthy while sigma is small enough that the
mean represents the behaviour that was optimised. ⛔ Do not raise
`initial_log_std` above ~0 without reporting the sampled-vs-mean gap alongside.

### Gate B — adversarial robustness

🔒 **Reproduced verbatim as declared 2026-08-27. Not edited.**

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

> ⚠️ **Amendment, 2026-09-02 — recorded, not substituted.** The declaration's
> "📏 the window is 11 pp" cited the inherited probe's 47.7 %, which J3B has since
> shown belongs to an adversary no stronger than a fixed beam. The **rules above
> are unchanged**; what changed is which rung supplies "the strongest adversary
> reached", and the honest current answer is **J2**. Gate B does not run until
> RQ2's 5-seed CUDA re-measurement (§7, run 1) settles the ordering.

### Gate C — quantisation and coordination

🔒 **Reproduced verbatim as declared 2026-08-27. Not edited.**

| | rule |
|---|---|
| **finding** | `role_entropy` and `observer_range_m` degrade proportionally more than `mission_capable`. Coordination is then more quantisation-sensitive than control — a new, deployment-relevant claim |
| **null** | degradation is proportional. Report latency and power as an engineering result and move on; it still supplies the sim-to-real line, which is most of its value |
| **report** | which architectures export at all. "The GNN buys +0.4 pp and is substantially harder to deploy" is a good sentence in a paper written for people who fly things |

---

## 6. What was cut, and it is not coming back

A paper carries one claim. These were live in the 2026-08-27 draft and are now
closed, demoted or deleted. ⛔ **Each line is a decision, not a backlog item.**

| cut | to what | why |
|---|---|---|
| **Further `Φ` work** | ⛔ deleted | Eight nulls. §3.3. The last one had a *measured-adequate* gradient and still moved nothing |
| **The action-space axis** | one paragraph | Gate A resolved. §5 |
| **RQ2 architecture ladder** | one table | Finished and null: MLP → DeepSets **+9.1 pp** disjoint, DeepSets → GNN **+0.9 pp** overlapping. The live part is the N = 8 zero-shot column, which stays |
| **Fidelity ladder, 5 rungs → 2** | `docs/REDUCTION.md` task 2 | Keep only 📏 F1-harder-than-F4, which now *serves* RQ2's non-monotonicity theme rather than standing alone |
| **"Three contributions, each able to fail on its own"** | one claim, four RQs | The framing that fits 80 pages does not fit 8 |
| **A C++/Rust inference runtime** | ⛔ not built | RQ4 is Python: TensorRT bindings, `trtexec`, `tegrastats` |
| **A bigger map, a second city, a better channel** | ⛔ not built | §9, unchanged |

⚠️ **The one thing that is *not* cut and looks like it should be:** the full
policy × adversary **cross-product**. It is the whole of RQ3 and 📏 it costs
minutes. Do not reduce it to a diagonal.

---

## 7. Sequence

**Today: 2026-09-02.** ~6 months of runway, then the official 5-month window from
March 2027. The official window is for **writing**, which is how it ends up good.

| when | what | why then |
|---|---|---|
| **run 1, this week** | 📏 **J-ladder at 5 seeds on CUDA.** J0/J1/J2/J3/J3B, B0 | Everything downstream reads the ordering, and it is currently one seed of 32 CPU episodes. [`results/j_ladder.md`](results/j_ladder.md) already declares that nothing runs until this does |
| **Sep–Oct 2026** | **RQ4 end-to-end.** ONNX export, TensorRT, latency / p99 jitter / power on the Orin Nano | ⚠️ Deliberately first: PyTorch Geometric exports badly, and finding that out in month one is worth more than in month ten. It also depends on nothing else |
| *in parallel, 1 week* | The BC-init probe (§3) | Cheap, timeboxed, upside only. ⛔ One week |
| *in parallel* | `docs/REDUCTION.md` tasks 2–4, then 6 | Housekeeping that pays for the adversary |
| **Nov 2026 – Jan 2027** | **J4**: opponent pool, alternating best response | The hard, risky part. The buffer lives here |
| **Feb 2027** | **The full policy × adversary cross-product, 5 seeds** → Gate B | 📏 A 10 M-step run costs 2.2 min. This month is compute-free by measurement |
| **Mar – Jul 2027** | Gate C quantisation sweep. 🔒 **Freeze**, recorded with a date and scope. `docs/REDUCTION.md` task 7. Writing and submission | The official window |

🔒 **Close the TR 36.777 NLoS intercept before the freeze.** One human reading of
one table, and everything downstream rests on it. Cost if right: an afternoon.
Cost if wrong and found late: the paper.

**Venue.** **IEEE RA-L** (rolling, no deadline pressure) remains the target, with
a NeurIPS or ICML workshop submission sooner as insurance and as a first
publication. ⚠️ Verify any deadline-driven alternative's date early rather than
discovering it.

---

## 8. Risks, and the fallback for each

| risk | mitigation |
|---|---|
| **J4 cycles.** Alternating best response's normal failure mode, and RQ3 is now load-bearing where it once was not | 🔒 **RQ1, RQ2 and RQ4 stand without it.** The paper's fallback spine is the exploitability *methodology* plus the non-monotonicity finding. And RQ2 gets **stronger** if J4 fails: "a learned adversary still did not beat a parked beam" is the loudest version of that claim |
| **The 5-seed J-ladder re-measurement collapses the J2/J3/J3B differences into noise** | Then RQ2's claim weakens to "adaptivity buys nothing", which is still the finding — the ordering mattered, not the levels. RQ1 and Gate B are unaffected: they need *a* strongest adversary, and J2 is it either way |
| **PyTorch Geometric will not export cleanly.** Scatter ops, dynamic shapes | Attempt it in month one. If the GNN cannot be deployed that is a *reported result* under Gate C, and a useful one — its measured advantage over DeepSets is a null anyway |
| **TR 36.777 NLoS intercept is wrong.** Every number re-derives | Verify before the freeze; earlier if possible |
| **Seed spread swamps every effect.** 📏 Historically 60–78 % over five runs, bimodal | Judge on the worst seed. 📏 The trainer fix already collapsed the GNN seed IQR 3.9 → 0.1, so this is substantially better than it was |
| **⚠️ Institutional.** Most of this work lands *before* the official March 2027 start | **Not a technical risk and not resolvable in this repo.** Confirm in writing with the supervisor that pre-start work is admissible, before building six months on the assumption |

---

## 9. What is deliberately not being built

Unchanged from 2026-08-27.

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
  Darmstadt, Kassel and Offenbach. The current eval split is held-out routes
  through *the same buildings*, which a reviewer will correctly call
  in-distribution. Held-out map tiles inside the existing box are free regardless,
  and 📏 the trainer fix opened a new 3.8 pp train→eval gap that makes this
  criticism more pressing, not less.
