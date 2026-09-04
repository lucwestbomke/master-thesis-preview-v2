# Contested Relay — the plan

**Rewritten 2026-09-04**, around the claim the frontier run produced. The two
framings before this one are recorded in §6, because a claim that was refuted is
part of the evidence for the one that replaced it.

`docs/INHERITED.md` records *what is already known*. `results/` records *what has
been measured, with the rule declared before each run*. **This file records what
happens next.**

---

## 1. The claim

> 🔍 **Exploitability decomposes into damage and a control loop:**
> `gap = f(threshold proximity) + g(loop AMPLITUDE)`.
>
> An adversary exploits a policy by **leading it out of position**. The cost scales
> with how far the policy is willing to be led — **not** with what its response is
> computed from. Capability saturates in that amplitude long before the cost does.
>
> ⛔ The loop is not the only term, and for a policy sitting on the capability
> threshold it is not the dominant one.

A swarm of `N = 5` UAVs observes a moving ground target and relays the feed to a
command vehicle over a multi-hop chain at >= 15 Mbps, while a directional jammer
degrades links. **Exploitability** is how much `mission_capable` a policy loses
between the isotropic emitter (J1) and the strongest adversary built (J3B).

### 📏 The evidence, in one table

`b0-geodesic` and `B0` are the **same scripted family** — same code path, same
stations, same chain length — differing by ranked roles, a belief filter, and
`_update_repair`, which is *"a 1-D hill climb on observable clearance"*.

| | capability (J1) | hops | **exploitability** | range |
|---|---|---|---|---|
| `b0-geodesic` | 45.6 % | 2.00 | **6.39 pp** | [5.68 – 6.54] |
| `B0` | 57.3 % | 2.13 | **13.24 pp** | [11.42 – 13.58] |

✅ **Disjoint.** 📏 [`b0_ablation.md`](results/b0_ablation.md) prices link repair
at **+6.90 pp** of capability; the exploitability difference is **6.85 pp**.
**The loop buys ~6.9 pp and costs ~6.9 pp, one for one.**

🔍 **The mechanism is a named subroutine.** The beam degrades a link → repair
hill-climbs to recover it → J3B re-optimises against the new geometry → repair
chases. `b0-geodesic` returns from `_update_repair` immediately and has half the
gap; `random` adapts to nothing and has the smallest gap of all (2.05 pp).

⛔ **This is not a scripted-versus-learned claim.** 📏 Both extremes of the range
are *scripted* and every learned policy sits between them. The variable is whether
the policy closes a loop on the jammed quantity, and whether that loop was trained
against an adversary. Full record: [`results/frontier.md`](results/frontier.md).

⚠️ **The loop is worth its cost.** At J3B, B0 still scores **43.8 %** against
geodesic's **39.7 %**. Adaptivity is a good trade; the point is that the cost is
real, measurable, and *separable* from what it buys.

📏 **Two factors, varied separately** ([`results/repair_gates.md`](results/repair_gates.md)):

| what was varied | result |
|---|---|
| the loop's **target** — clearance (jammer-proof) vs capacity | ⛔ **no effect**: 13.37 vs 13.24, overlapping. The router picks which edges the loop scores, and the jammer sets the router — so the adversary drives the loop *whatever* it looks at |
| the loop's **amplitude** — 0 / 50 / 100 / 200 m | ✅ **dose–response**, 7.94 → 13.24 pp, disjoint endpoints |

⭐ 📏 And it is **mis-set**: at J3B, `repair_amplitude_m = 100` scores **47.3 %**
against the shipped 200's **43.8 %** — **5/5 paired seeds**, for 0.46 pp of J1
capability. The last doubling buys +0.4 pp of capability and costs +3.7 pp of
exploitability. **B0's adaptation gain is tuned for an unjammed world.**

🔒 **And the pair is a conservative test of it.** B0 has *more* capacity headroom
than geodesic (+6.6 Mbps over the bar against +3.3), so its damage term `f` is
**smaller** — and its gap is still twice geodesic's. **6.85 pp is a lower bound on
the loop's contribution.**

📏 **The second term is why nine policies need two.** Every learned policy in this
project sits within **1.1 Mbps of the 15 Mbps threshold**, where each dB the
jammer removes crosses the bar without any behavioural response — damage, not
exploitation. That is why masking the jammed observations was a **null**
([`results/obs_mask_gate.md`](results/obs_mask_gate.md)): their loop is worth
1.51 pp of capability, and 11 pp of gap cannot come out of it.

⛔ **So the learned policies in this project cannot test the loop claim.** Doing
that needs a policy with real capacity headroom — i.e. a genuinely capable one —
which is the 15 pp gap §3 closed. **Do not re-open §3 to rescue §1.**

---

## 2. The objectives

### RQ1 — Is exploitability a cost of adaptivity? — 🔶 **supported, n = 1 pair**

📏 The geodesic/B0 pair above, disjoint, with the mechanism named and priced.
⚠️ It is **one controlled pair**. §7 runs 1 and 2 turn it into a dose–response
curve and a constructive test, both with **zero training**.

### RQ2 — Where does an adversary's power come from? — ✅ **answered**

📏 5 seeds × 128 episodes: **directionality −10.6 pp, adaptivity −2.9 pp** on B0.
An adversary's power is overwhelmingly about *where the energy goes*.
[`results/j_ladder.md`](results/j_ladder.md).

### RQ3 — Does co-training reduce the exploitability of a learned loop? — 🔶 **effect measured, mechanism untested**

📏 Co-training moves the learned policies **down** the exploitability axis:
11.12 → 7.51 and 10.45 → 7.29, at unchanged capability and unchanged chain length,
with **no cost on the clean rung** (+0.3 pp at J1). The off-diagonal shows
robustness rather than opponent-overfit — `advtrain-J2` beats `advtrain-J3B` *on
J3B*. [`results/gate_b.md`](results/gate_b.md).

⚠️ **Why it works is untested.** The leading candidate is **path redundancy**:
`routing.py` picks the widest *single* path and the jammer has *one* beam, so a
second threshold-clearing, edge-disjoint path would make the beam's kill
recoverable. ⛔ **No metric for this exists** — §7 run 3.
⛔ **J4**, a learned jammer, is still not built.

### RQ4 — Does it survive the airframe? — ⛔ **not started**

ONNX → TensorRT on a Jetson Orin Nano: latency, p99 jitter, power against the
400 ms control period, and *does quantisation degrade coordination more than
control?* Gate C. 🔧 Pure Python. Hardware is in hand; **the export risk is
unretired.**

---

## 3. The premise: B0 wins the static task, and that axis is closed

⛔ **Do not re-open this.** Five independent lines, each measured:

| | finding |
|---|---|
| 1 | The gap is **`observed` and nothing else** — conditioned on a sightline the GNN converts it as well as B0, 0.620 vs 0.617 |
| 2 | **B0 wins the reward too** — 222.9 vs 85.8 `episode_return`, and return rank-correlates with `mission_capable` at **ρ = 0.987** over 20 rows |
| 3 | **Eight pre-declared interventions, eight nulls**, the last with a *measured-adequate* gradient |
| 4 | **Structural**: `Var_i(A) = Var_i(G)` exactly, and that between-drone variance is **0.04–0.16 %** of the total. Every team reward term cancels *exactly*, so no shaping knob can move role differentiation — [`credit_assignment.md`](results/credit_assignment.md) |
| 5 | **Not memory either**: perfect target state is worth **−0.4 pp**, a hard upper bound — [`memory_horizon.md`](results/memory_horizon.md) |

📏 And the budget is priced: B0's whole design advantage is **~10.3 pp** (link
repair 6.9, ranked roles 3.4, belief ~0) against a **15.0 pp** gap. Acquiring
every component would not close it — [`b0_ablation.md`](results/b0_ablation.md).

---

## 4. The adversary ladder

| rung | emitter | isolates | state |
|---|---|---|---|
| **J0** | none | exists | ✅ |
| **J1** | isotropic, fixed power | the inherited emitter | ✅ |
| **J2** | directional, **fixed** on the MCV | separates *directionality* from *adaptivity* | ✅ |
| **J3** | directional, greedy retarget | adaptive without learning | ✅ |
| **J3B** | directional, **exhaustive best response** | one-step-optimal | ✅ |
| **J4** | directional, **learned**, opponent pool | RQ3's stretch | ⛔ **not built** |

🔒 The beam is 3GPP TR 38.901's element pattern, `A(θ) = −min[12(θ/θ_3dB)², 30]`
with **θ_3dB = 25° the FULL half-power beamwidth** — the −3 dB point is at 12.5°.
⛔ Beamwidth is not an action and power is fixed; both smuggle the transmit-power
axis back in. Aiming carries one step of latency, which is required rather than
tolerated: aiming at *this* step's chain would be circular.

---

## 5. Gates

🔒 Every gate is judged on the **worst seed**, at >= 5 seeds, with the rule
declared before the run and never edited afterwards.

| gate | question | verdict |
|---|---|---|
| **A** | velocity setpoints as the action space | ⛔ **not met** — 18.3 pp cost, disjoint. [`gate_a.md`](results/gate_a.md) |
| **B** | is the heuristic more exploitable? | ✅ **confirmed** — and survived its own `capable_no_division` control. [`gate_b.md`](results/gate_b.md) |
| **C** | does quantisation hurt coordination more than control? | ⛔ **not run** (RQ4) |
| Φ v2 | does a steeper potential move the observer? | ⛔ **killed** — 11.8 m of a 130 m gap |
| k = 2 | does one step of history buy link repair? | ⚠️ **inconclusive** — +1.94 pp, worst seed −1.25 |

⚠️ **Gate B's verdict stands as declared, and its interpretation has moved.** It
was read as *"scripted policies are more exploitable"*; §1 shows the variable is
the loop, not the script. The number is unchanged; the claim it supports is not.
Gate B's declaration, its two amendments and its verdict live in
[`results/gate_b.md`](results/gate_b.md), reproduced verbatim there.

### Gate C — quantisation and coordination. ⛔ **Not yet run.**

🔒 **Declared 2026-08-27 and reproduced verbatim. Not edited.**

| | rule |
|---|---|
| **finding** | `role_entropy` and `observer_range_m` degrade proportionally more than `mission_capable`. Coordination is then more quantisation-sensitive than control — a new, deployment-relevant claim |
| **null** | degradation is proportional. Report latency and power as an engineering result and move on; it still supplies the sim-to-real line, which is most of its value |
| **report** | which architectures export at all. "The GNN buys +0.4 pp and is substantially harder to deploy" is a good sentence in a paper written for people who fly things |

---

## 6. Framings that were refuted, and why they are kept

⛔ **A refuted framing is evidence, not embarrassment.** Each was killed by a run
designed to test it, and the sequence is why the current claim should be trusted.

| framing | killed by | when |
|---|---|---|
| *Learned control beats the scripted baseline* | 8 nulls, then §3's five lines | 2026-09-02 |
| *The adversary ladder is non-monotone; adaptivity does not help* | the 5-seed CUDA re-run **reversed** a one-seed CPU result | 2026-09-03 |
| *Exploitability is a cost of **capability*** | the frontier run: `b0-geodesic` is **more capable than every learned policy and less exploitable than all of them** | 2026-09-04 |

🔒 **The third was refuted *before it was declared*,** because it was fitted over
eight policies rather than written down after four. ⛔ Hold the current claim to
the same standard: §7 runs 1 and 2 exist to break it.

---

## 7. The roadmap

🔒 **Ordered by what the thesis is for.** ⚠️ An earlier version of this section
ordered by evidence-per-hour and put the scripted controls first — which leads
somewhere this project does not want to go, because if a B0 variant is the
frontier-breaking policy then the deliverable is *an improved heuristic*. Runs 2–4
are controls and need no training; **run 1 is the policy** and it is first.

### ~~Run 1~~ — ⚠️ **DONE 2026-09-04. NULL.** [`obs_mask_gate.md`](results/obs_mask_gate.md)

📏 `obs["flat"]` carries **nine** features the emitter can move: `noise_dbm`,
`e2e_capacity`, and each neighbour's edge capacity. Everything else is geometry,
kinematics or the sensor — and 🔒 `clr_hvt`, `clr_mcv` and the per-edge *clearance*
come from building occlusion, which the jammer **cannot touch**.

`--mask-jammed-obs` zeroes exactly those nine. **The result is a policy that can
still adapt — on geometry — but has no loop on the quantity it is attacked
through.** That is the learned counterpart of `repair_score="clearance"`, and it
is the policy this project is actually trying to build: adaptive, capable, and
not exploitable.

| | prediction |
|---|---|
| **the claim holds constructively** | capability within ~2 pp of the unmasked control, exploitability falling toward `b0-geodesic`'s 6.39 pp. ⭐ **That is the result the thesis wants** — a learned policy off the tradeoff |
| **the claim holds, expensively** | capability drops with exploitability. The loop was load-bearing for capability too, and the trade is real rather than avoidable |
| **null** | neither moves. 📏 Plausible: the learned policies already sit at their **sensor** ceiling (`no-div \| observed` = 0.887), so their loop on capacity may be doing very little to begin with |

🔒 Declare the rule in `results/obs_mask_gate.md` **before** running. Control is
the same architecture, cadence and seeds with the flag off.

### ~~Run 2~~ — ⛔ **DONE 2026-09-04. The target does not matter.** [`repair_gates.md`](results/repair_gates.md)

`B0Config.repair_score` already takes **`"clearance"`** — the same idea on the
scripted side, and a one-word config change. ⚠️ **It is a control, not the
deliverable**: if it works, the frontier-breaking policy would be a B0 variant,
and improving the heuristic is not what this thesis is for. Run it because it
prices the mechanism cheaply and because it makes run 1 interpretable either way.

### ~~Run 3~~ — ✅ **DONE 2026-09-04. Dose–response confirmed, and the shipped amplitude is 2x too large.** [`repair_gates.md`](results/repair_gates.md)

`repair_amplitude_m`: **0** → 50 → 100 → **200**. If capability and exploitability
both rise monotonically with the loop's amplitude, RQ1 stops being one controlled
pair and becomes a **curve**.

### Run 4 — the redundancy metric. Half a day, then zero training.

*"Does a second threshold-clearing path exist that is edge-disjoint from the
chosen one?"* — computable from the capacity matrix `routing.py` already builds.
⛔ Nothing measures this today, and it is RQ3's candidate mechanism.

### Then, in order

| | what | why then |
|---|---|---|
| **RQ4 / Gate C** | ONNX export **locally first**, then TensorRT on the Orin | The only deliverable that cannot fail. PyTorch Geometric may not export; that risk is unretired and cheap to close |
| **J4** | learned jammer, opponent pool | Strengthens RQ1 and RQ3. ⚠️ Gate B stands without it — the fallback declared in §8 |
| **Write** | exposé, then the paper | 🔒 **After runs 1–3.** The claim has moved three times in ten days; let the data settle it before it is promised to anyone |

⛔ **Not on the roadmap:** any further reward shaping (§3.4 closed it
structurally), any further action-space work (Gate A), recurrence (bounded at
0.4 pp), frame stacking beyond k = 2 (a one-step state needs no longer history).

---

## 8. Risks

| risk | mitigation |
|---|---|
| **Run 1 shows the loop's target does not matter** | §1's mechanism is then wrong and RQ1 reverts to a bare correlation. ✅ RQ2 and Gate B's number are unaffected — this is why they are separate objectives |
| **J4 cycles.** Alternating best response's normal failure | 🔒 RQ1, RQ2 and RQ4 stand without it; the strongest adversary reached is then scripted and the paper says so |
| **PyTorch Geometric will not export** | ⚠️ Attempt it *locally*, before touching the Jetson. A GNN that cannot deploy is a **reported result** — its measured advantage over DeepSets is a null anyway |
| **TR 36.777 NLoS intercept is wrong.** Every number re-derives | 🔒 One human reading of one table. Close it before the freeze |
| **n = 1 environment** | ⚠️ Not resolvable within this project. State it as the limitation it is; the mechanism is at least *named and priced* rather than statistical |
| **The claim moves a fourth time** | 🔒 Runs 1–2 are pre-declared attempts to break the current one. If it survives them it has been tested, not just fitted |

---

## 9. Deliberately not built

📏 Each measured, not assumed — `docs/inherited/DECISIONS.md`.

- **A bigger map, a second city, a better channel.** Compute is not the
  constraint; a 10 M-step run costs 2.2 minutes. ⚠️ Held-out map *tiles* inside
  the existing box remain free and would answer the in-distribution objection.
- **Flying below 40 m.** TR 36.777 stops at 22.5 m.
- **Transmit power or beamwidth as actions.** Three framings, three nulls, and a
  degenerate optimum for the jammer.
- **Training at more than one `N`.** It turns the zero-shot columns into
  in-distribution tests.
