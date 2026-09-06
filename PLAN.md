# Contested Relay — the plan

**Rewritten 2026-09-04**, around the claim the frontier run produced. The two
framings before this one are recorded in §6, because a claim that was refuted is
part of the evidence for the one that replaced it.

⚠️ **Amended the same day.** §3 said *"closed, do not re-open"* and now says
*"re-opened, on named grounds"* — a **measured confound in the optimiser** that
every number in `results/` shares. 🔒 Four of its five lines are untouched and
§3 says which. The amendment is recorded here rather than made silently, for the
same reason §6 keeps the refuted framings.

⭐ **Restructured 2026-09-06.** Four objectives became **three research
questions**: the old RQ2 (*where does an adversary's power come from*) is folded
into RQ1 as a supporting result, and the capability programme stopped being an
objective and became §3, **the instrument**. ⛔ **No scientific content changed** —
no number, no rule, no verdict. The reframing is recorded in §6 rather than
applied silently, for the same reason the refuted framings are kept there.

`docs/INHERITED.md` records *what is already known*. `results/` records *what has
been measured, with the rule declared before each run*. **This file records what
happens next.**

---

## 1. The claim

### Working title

> **The cost of adaptation: measuring and removing exploitability in multi-agent
> relay under directed jamming.**

### Goal

> Show that a controller's own repositioning loop is what makes it exploitable,
> measure how that cost scales, and demonstrate that adversarial co-training
> removes it.

### The claim itself

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
that needs a policy with real capacity headroom — i.e. a genuinely capable one.

⚠️ **This used to end "which is the 15 pp gap §3 closed. Do not re-open §3 to
rescue §1."** §3 is now re-opened — not to rescue this claim, but on a **named,
measured confound in the optimiser** that every result in `results/` shares. The
distinction is the whole point and §3 states it. ⭐ Note which way the dependency
runs: **a capable learned policy is a *prerequisite* for RQ1**, because a policy
1.1 Mbps from the threshold measures damage rather than exploitation. §1 is
unaffected by the outcome either way — its evidence is a *scripted* controlled
pair.

---

## 2. The research questions

⭐ **Restructured 2026-09-06.** There were four objectives; there are now three
questions. ⛔ Nothing measured changed — what changed is which numbers are
**claims** and which are **support**.

| old | new home |
|---|---|
| RQ1 — is exploitability a cost of adaptivity | **RQ1**, unchanged in substance |
| RQ2 — where does an adversary's power come from | **folded into RQ1** as a supporting result. It justifies the adversary ladder rather than standing alone; every number and link is kept |
| RQ3 — does co-training reduce exploitability | **RQ2** |
| RQ4 — does it survive the airframe | **RQ3** |
| §3 — the capability question | ⛔ **no longer an objective.** It is §3, **the instrument** |

⚠️ **`results/` predates this renumbering and is never edited.** A file there that
says *"RQ3"* for co-training means what this section now calls **RQ2**, and
[`rq2_ladder.md`](results/rq2_ladder.md) keeps the predecessor's numbering
entirely — its "RQ2" is the **architecture ladder**. Read the question, not the
number. Terms are defined once, in [`docs/GLOSSARY.md`](docs/GLOSSARY.md).

### RQ1 — Does adaptation create exploitability, and how does it scale? — 🔶 **supported**

> Do controllers that reposition in response to the attacked quantity lose more
> capability to an *aiming* adversary than controllers that do not — and does the
> loss scale with how far they are willing to reposition?

📏 **The evidence, all of it set out in §1:**

| | |
|---|---|
| the **controlled pair** | `b0-geodesic` **6.39 pp** [5.68 – 6.54] against `B0` **13.24 pp** [11.42 – 13.58] — **disjoint**, same code path, same stations. [`frontier.md`](results/frontier.md) |
| the **dose–response** | `repair_amplitude_m` 0 / 50 / 100 / 200 m → **7.94 → 13.24 pp**, disjoint endpoints. [`repair_gates.md`](results/repair_gates.md) |
| the null on the loop's **target** | clearance **13.37** against capacity **13.24**, overlapping — the adversary drives the loop *whatever* it looks at. Same file |
| the **mechanism**, named | degrade → repair → re-optimise → **chase**. `b0-geodesic` returns from `_update_repair` immediately and has half the gap; `random` adapts to nothing and has 2.05 pp |

✅ **Gate B is RQ1's confirmation against a co-trained comparison**, and it
survived its own `capable_no_division` control: B0's gap moves **0.39 pp**
(13.24 → 12.85) while the learned policies' collapse to **0.44 – 5.46 pp**.
⚠️ Half the confound survives — *"a longer chain has more links for the beam to
find"* is **not** closed, and no policy in this project supplies the hop-matched
comparison that would close it. [`gate_b.md`](results/gate_b.md).

#### Supporting result — where an adversary's power comes from. ✅ **answered**

⭐ **This was RQ2 until 2026-09-06.** It is kept in full and demoted to support:
it is what justifies the adversary ladder in §4, not a claim of its own.

📏 5 seeds × 128 episodes, CUDA: **directionality −10.6 pp, adaptivity −2.9 pp**
on B0. An adversary's power is overwhelmingly about *where the energy goes*, not
about re-deciding where it goes. ⚠️ This **replaced** a non-monotonicity claim
that came from one CPU seed and did not replicate: the ladder is monotone,
J3B > J3 > J2, on **5/5** paired seeds.
[`results/j_ladder.md`](results/j_ladder.md).

#### ⚠️ Open — and it needs no training

📏 The dose–response is measured on **one map**. Reproducing it in a second,
**minimal abstract relay environment** — scripted controllers only, no learned
policy — is what turns `n = 1 environment` from a limitation into an
**external-validity argument**. §7 item 4, §8.

### RQ2 — Can adversarial co-training remove the cost without losing capability? — 🔶 **effect measured, mechanism untested**

> If the loop is what makes a controller exploitable, can the loop be kept but
> trained so that it cannot be led?

📏 Co-training moves the learned policies **down** the exploitability axis:
11.12 → 7.51 and 10.45 → 7.29, at unchanged capability and unchanged chain length,
with **no cost on the clean rung** (+0.3 pp at J1). The **off-diagonal** shows
robustness rather than opponent-overfit — `advtrain-J2` beats `advtrain-J3B` *on
J3B*. [`results/gate_b.md`](results/gate_b.md).

⚠️ **Why it works is untested.** The leading candidate is **path redundancy**:
`routing.py` picks the widest *single* path and the jammer has *one* beam, so a
second threshold-clearing, edge-disjoint path would make the beam's kill
recoverable. ⛔ **No metric for this exists** — §7 item 2, computable from the
capacity matrix `routing.py` already builds.
⛔ **J4**, a learned jammer, is still not built, so the result holds against
**scripted** adversaries only. §8 carries the fallback.

### RQ3 — Does it survive deployment? — ⛔ **not started**

ONNX → TensorRT on a Jetson Orin Nano: latency, p99 jitter and power against the
400 ms control period. Then: *does quantisation degrade **coordination** more than
it degrades **control**?*

* **coordination** = role structure and observer geometry — `role_entropy`,
  `observer_range_m`;
* **control** = basic competence — `mission_capable`, on-station behaviour.

🔒 **Gate C's declaration (§5) already covers this and is reproduced verbatim,
not rewritten.** 🔧 Pure Python. Hardware is in hand; **the export risk is
unretired**, which is why §7 puts the local export first.

---

## 3. The instrument — a capable learned policy

⭐ **Retitled 2026-09-06, and it is no longer a research question.** The
capability programme is an **instrument**. The justification is already in this
document: every learned policy here sits within **1.1 Mbps of the 15 Mbps
threshold**, where the adversary's *damage* term dominates and **no behavioural
response is being measured at all**. A capable learned policy is therefore a
**prerequisite for testing RQ1 on learned controllers**, and nothing more.

> 🔒 **The thesis does not depend on this section.** RQ1's evidence is a
> *scripted* controlled pair. If Gates D–F all null, RQ1, RQ2 and RQ3 stand
> unchanged and the paper reports that the scripted baseline remains the
> strongest policy.

> 🔒 **Gates D–F are budgeted at three weeks.** At the end of that budget the
> programme stops regardless of outcome and writing begins. If no capable learned
> policy exists by then, [`scripts/bc_init.py`](scripts/bc_init.py) is used to
> manufacture one for RQ1's learned-loop arm. ⚠️ A teacher-initialised policy
> remains a **probe, not an arm** — it is not a like-for-like comparison for the
> architecture ladder or for Gate B, and any result using it must say so.

⛔ **The retitling does not soften the re-opening.** ⚠️ **This section said
"closed. Do not re-open." until 2026-09-04.** It was re-opened on a named
confound in the optimiser, and ⭐ **on 2026-09-06 that confound was measured and
refuted** — see below. Four of the five lines were untouched throughout; the
fifth is now restored.

🔍 **The re-opening was still worth it.** It converted *"nobody checked"* into
*"checked, and it is not the explanation"*, which is what §3 needed to be able to
say. And it leaves the search pointed exactly where
[`credit_assignment.md`](results/credit_assignment.md) pointed it: **the return
and the advantage**, not the optimiser.

### 📏 What changed: the optimisation budget was frozen and never examined

[`docs/inherited/BLOCK_G.md`](docs/inherited/BLOCK_G.md) built three cadences
holding *"gradient density constant at 488 optimizer steps per M env-steps"*, and
recorded — without following it up — that this pins **the minibatch at 40,960
rows in all three**. It also states *"⛔ Not swept, deliberately: the learning
rate."* So at the `deep` cadence a 12 M-step run is

```
12e6 / (4096 * 64)               =     46 PPO updates
46 * 4 epochs * 32 mini-batches  =  5,888 Adam steps, total
```

on a **137 k-parameter** actor. 📏 `runs/val-gnn-deep-s*/log.jsonl` confirms the
consequence: `approx_kl` sits at **0.002 – 0.004** for whole runs against PPO's
usual 0.01 – 0.02 — about **0.14 nats of total policy movement, end to end**. And
`grad_kept`, instrumented for the joint-clip question `BLOCK_G` lists as open, was
**NaN in every log in `runs/`**.

⚠️ **A claim made here on 2026-09-04 and CORRECTED 2026-09-06.** This paragraph
read *"three quarters of what remains is discarded by the norm clip"*, from a
120 k-step **MPS smoke run at 128 envs**. 📏 On a real CUDA run at 4096 envs,
`grad_kept` is **0.54 – 0.91** and `grad_norm_actor` is **0.032 – 0.060** against
a 0.5 clip — the actor's gradient never reaches the clip at all, and the
throttling costs ~1.1–1.8x, not ~4x. A toy-config number was quoted as though it
described the condition under study. 🔒 The `approx_kl` figure is independent and
reproduces at **0.0021 – 0.0028**.

☠️ **Every number in `results/` was measured under that budget** — which is why
Gate D was run first, and ⭐ **Gate D answered it.**

### ✅ 2026-09-06: the confound was tested, and it is not the explanation

📏 [`capability_gates.md`](results/capability_gates.md), 2 × 4 factorial, 3 seeds,
train split:

| | result |
|---|---|
| **λ** ∈ {0.95, 0.98, 0.99, 0.995} | ⛔ **null to harmful.** The shipped 0.95 wins on the worst seed (44.28 %); 0.995 falls to 36.13 % |
| **10x the optimisation budget** | ☠️ **−32 pp.** 13.10 % against the control's 45.18 %, with a final policy indistinguishable from random on `hop_mean`, `observer_tenure` and `episode_return` |

🔒 **So "the policy cannot move" is not the binding constraint.** It was given 10x
the gradient steps and 17x the learning rate, it moved (`approx_kl` 0.0027 →
0.0107, `lr_actor` plateauing at 5.13e-3), and it got **worse**. ⛔ My own
prediction was the opposite, and it is recorded as refuted rather than quietly
dropped.

### 🔒 What that does, and does NOT do, to the five lines

| | line | status |
|---|---|---|
| 1 | the gap is **`observed` and nothing else** — conditioned on a sightline the GNN converts it as well as B0, 0.620 vs 0.617 | ✅ **stands.** It is a *description* of the gap, not a closure of it — and it is the target |
| 2 | **B0 wins the reward too**, 222.9 vs 85.8, and return rank-correlates with `mission_capable` at **ρ = 0.987** | ✅ **stands.** The objective is not misspecified, whatever the optimiser did |
| 3 | **eight pre-declared interventions, eight nulls** | ✅ **RESTORED 2026-09-06.** Raised as confounded on 2026-09-04 — all eight measured at ~5,900 Adam steps. ⭐ Gate D tested that confound directly and it went the *other* way: 10x the budget costs 32 pp. The eight nulls stand, and now stand **tested** rather than merely un-examined |
| 4 | **structural**: `Var_i(A) = Var_i(G)` exactly, and that between-drone variance is **0.04–0.16 %** | ✅ **stands, exactly.** Team terms cancel *by construction*; no optimisation changes that. ⭐ And it **names its own successor**: *"What is left is the critic and the advantage, none of which has been touched"* |
| 5 | **not memory either**: perfect target state is worth **−0.4 pp** | ⚠️ **stands as a bound on TARGET memory, for B0.** [`memory_horizon.md`](results/memory_horizon.md) itself leaves **role-commitment** memory open |

### ⭐ 2026-09-06: line 4's successor axis was tested too, and it is closed

📏 Gate E supplied the per-drone credit `credit_assignment.md` measured as absent.
`D_i = G(z) − G(z_{−i})` over **seven weights spanning 12x**: the whole capability
axis sits inside a single cell's seed noise (2.68 pp against 4.23 pp), and
`role_entropy` (0.492 → 0.603) and `observer_range_m` (192.8 → 211.7 m) get
**worse**. 🔒 And it is not a signal-delivery failure: the differentiable share
reaches **35.5 %** against a **0.04–0.16 %** control class, above the band
`measure_credit.py` pre-declared as refuting its own mechanism.

☠️ **So *"the advantage cannot tell one drone from another"* is true, and is not
the reason the swarm fails to differentiate.** Supplying the signal makes role
differentiation *worse*. Line 4 stands as a measurement and falls as an
explanation.

🔍 **Line 4 was still the argument for running Gate E.** It says
no *shaping* knob can move role credit, and points at the return. That is exactly
what §7's Gate E changes, and it is why the instrument is a difference reward
rather than a ninth weight.

### 🔒 And §1 requires this

⚠️ §1 already says, in its own words: *"the learned policies in this project
cannot test the loop claim. Doing that needs a policy with real capacity headroom
— i.e. a genuinely capable one."* Every learned policy here sits within
**1.1 Mbps of the 15 Mbps threshold**, where the jammer's damage term dominates
and no behavioural response is being measured at all.

⭐ **So a capable learned policy is a prerequisite for RQ1, not a distraction from
it.** The old instruction — *"do not re-open §3 to rescue §1"* — was right about
the failure mode it feared (fitting §3 to save a claim) and wrong about the
remedy. §3 is re-opened on a **named, measured confound in the optimiser**, with
gates declared before the runs, and §1 is untouched by the outcome either way.

### 📏 What has NOT changed

⛔ **B0 is still the strongest policy in this project**, 57.3 % [54.8 – 60.6]
against the best learned 40.7 %. ⛔ **B0's design advantage is still ~10.3 pp**
against a 15.0 pp gap ([`b0_ablation.md`](results/b0_ablation.md)), so acquiring
every B0 component would still not close it — which is why Gates D–F attack the
optimiser and the advantage rather than trying to clone the heuristic.

🔒 **The bar, in this project's own standard** (Gate A and the J-ladder both
judge on disjoint seed ranges): *clears B0* is a median above **57.3 %**; *beats B0* is a
**worst seed above 60.6 %**. Full declaration:
[`results/capability_gates.md`](results/capability_gates.md).

## 4. The adversary ladder

| rung | emitter | isolates | state |
|---|---|---|---|
| **J0** | none | exists | ✅ |
| **J1** | isotropic, fixed power | the inherited emitter | ✅ |
| **J2** | directional, **fixed** on the MCV | separates *directionality* from *adaptivity* | ✅ |
| **J3** | directional, greedy retarget | adaptive without learning | ✅ |
| **J3B** | directional, **exhaustive best response** | one-step-optimal | ✅ |
| **J4** | directional, **learned**, opponent pool | RQ2's stretch | ⛔ **not built** |

🔒 The beam is 3GPP TR 38.901's element pattern, `A(θ) = −min[12(θ/θ_3dB)², 30]`
with **θ_3dB = 25° the FULL half-power beamwidth** — the −3 dB point is at 12.5°.
⛔ Beamwidth is not an action and power is fixed; both smuggle the transmit-power
axis back in. Aiming carries one step of latency, which is required rather than
tolerated: aiming at *this* step's chain would be circular.

---

## 5. Gates

🔒 Every gate is judged on the **worst seed**, at >= 5 seeds, with the rule
declared before the run and never edited afterwards.

| gate | serves | question | verdict |
|---|---|---|---|
| **A** | control (closed) | velocity setpoints as the action space | ⛔ **not met** — 18.3 pp cost, disjoint. [`gate_a.md`](results/gate_a.md) |
| **B** | **RQ1** | is the heuristic more exploitable? | ✅ **confirmed** — and survived its own `capable_no_division` control. [`gate_b.md`](results/gate_b.md) |
| **C** | **RQ3** | does quantisation hurt coordination more than control? | ⛔ **not run** |
| **D** | the instrument | is the learned policy **optimisation-limited** rather than credit-limited? | ⛔ **NULL / REGRESSION, 2026-09-06** — λ is null-to-harmful and 10x the budget costs **32 pp**. [`capability_gates.md`](results/capability_gates.md) |
| **E** | the instrument | does **per-drone credit** (`D_i = G − G_{−i}`) produce roles? | ⛔ **NULL, 2026-09-06** — 7 weights over a 12x range, whole axis inside single-cell noise; `role_entropy` and `observer_range_m` get **worse**. ⭐ And the signal *reached the gradient*: 35.5 % differentiable share against a 0.04–0.16 % control class |
| **F** | the instrument | is the **observation** lying to the policy? | ⛔ **not run** — same file, lowest prior |
| Φ v2 | the instrument | does a steeper potential move the observer? | ⚠️ **killed — and confounded.** 11.8 m of a needed 20 m, measured under ~5,900 Adam steps. §3 |
| k = 2 | the instrument | does one step of history buy link repair? | ⚠️ **inconclusive** — +1.94 pp, worst seed −1.25 |

⚠️ **Gates D, E and F re-open §3.** Every branch of each is declared before its
run and each partitions the outcome space — ⛔ Gate A and
[`trainer_validation.md`](results/trainer_validation.md) *each* recorded a rule
that did not, and that is now a standing requirement rather than a lesson.

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
| *Learned control beats the scripted baseline* | 8 nulls, then §3's five lines | 2026-09-02 — ⚠️ **partially reinstated 2026-09-04.** The eight nulls are *confounded*, not refuted: all were measured at ~5,900 Adam steps. §3 |
| *The adversary ladder is non-monotone; adaptivity does not help* | the 5-seed CUDA re-run **reversed** a one-seed CPU result | 2026-09-03 |
| *Exploitability is a cost of **capability*** | the frontier run: `b0-geodesic` is **more capable than every learned policy and less exploitable than all of them** | 2026-09-04 |
| *Four objectives, with the capability programme as an RQ* | superseded by three RQs; the capability programme is an instrument, not an objective | reframed 2026-09-06 |

⚠️ **The last row is a *reframing*, not a refutation.** No measurement killed it —
the organisation of the same evidence changed, and it is recorded here rather than
applied silently. §2 carries the old → new mapping.

🔒 **The third was refuted *before it was declared*,** because it was fitted over
eight policies rather than written down after four. ⛔ Hold the current claim to
the same standard: §7 runs 1 and 2 exist to break it.

---

## 7. The roadmap

🔒 **Ordered by what the thesis is for.** ⚠️ An earlier version of this section
ordered by evidence-per-hour and put the scripted controls first — which leads
somewhere this project does not want to go, because if a B0 variant is the
frontier-breaking policy then the deliverable is *an improved heuristic*.

⭐ **Reordered 2026-09-06** by the restructure recorded in §6. Runs 1–3 are closed
and kept below as the record. What remains is ordered so that the deliverable
which **cannot fail** is first, the instrument is **time-boxed**, and RQ1's
replication lands before anything is written up.

| | what | serves | cost |
|---|---|---|---|
| **1** | **ONNX export, locally** — this week | RQ3 | days, no training |
| **2** | **the redundancy metric** | RQ2 | half a day, no training |
| **3** | **Gates D–F** — 🔒 three-week box, §3 | the instrument | ~4 GPU-hours, inside the box |
| **4** | **the second-environment replication** | RQ1 | scripted controllers, no training |
| **5** | **the exposé** | all three | — |
| **6** | **J4, the Orin work, longer runs** | RQ1 / RQ2 / RQ3 | after the above |

### 1. ONNX export — locally first, then the Orin. RQ3.

**The only deliverable that cannot fail.** PyTorch Geometric may not export; that
risk is **unretired and cheap to close**, and it closes on a laptop before
anything is flashed to a Jetson. ⚠️ A GNN that cannot deploy is a **reported
result** — its measured advantage over DeepSets is a null anyway. Gate C (§5) is
declared, verbatim, and waits on this.

### 2. Run 4 — the redundancy metric. RQ2. Half a day, then zero training.

*"Does a second threshold-clearing path exist that is edge-disjoint from the
chosen one?"* — computable from the capacity matrix `routing.py` already builds.
⛔ Nothing measures this today, and it is **RQ2's candidate mechanism**.

### 3. ⭐ Runs 5–7 — the capability programme. Gates D, E, F. The instrument.

🔒 **Time-boxed at three weeks (§3).** ~4 GPU-hours of compute; the box is on the
search, not the hardware. 📏 **D and E are closed and both NULL** (§5); **F is
what the box has left**, and it carries the lowest prior of the three.

🔒 **Declared in full, before any run:**
[`results/capability_gates.md`](results/capability_gates.md). Ordered so that each
is interpretable given the one before it.

| | question | instrument | why it is first / next / last |
|---|---|---|---|
| **D** | is the policy **optimisation-limited**? | `--mini-batch-size`, `--target-kl`, `--grad-norm-clip-critic`, `--orthogonal-init`, `--min-log-std`, and 📏 **`--gae-lambda`, never swept in this project's history** | It gates everything. Its NULL branch is a real result: it removes the confound from all eight prior nulls and makes [`credit_assignment.md`](results/credit_assignment.md) *stronger* |
| **E** | does **per-drone credit** produce roles? | `--w-difference` — `D_i = G(z) − G(z_{−i})`, the mission term recomputed with drone `i` deleted, exactly | The successor axis `credit_assignment.md` names. ⛔ Only interpretable on a policy that can actually train, hence after D |
| **F** | is the **observation** lying to the policy, and is the **curriculum** teaching a shortcut? | `--cue-mode`, `--curriculum-boundaries` / `--curriculum-mix`, `--mask-broadcast-obs` | ⚠️ **Two arms, very different priors, never pooled.** ⭐ The cue/curriculum arm is now measured (below); the broadcast-feature arm keeps `obs_mask_gate.md`'s null prior |

🔒 **Gate D is a 2 × 4 factorial**, `{shipped budget, new budget} × λ ∈ {0.95,
0.98, 0.99, 0.995}` — amended 2026-09-04 *before any run*, because the first
draft bundled λ into a five-knob arm that could not separate it from the step
count. At ~5 min a run it does not have to.

📏 **Why λ is an axis and not a footnote.** At `γ = 0.997, λ = 0.95` the
advantage weights rewards by `(γλ)^l = 0.947^l` — an effective horizon of **18.9
steps, 7.6 seconds**. B0's observer tenure is **294.7 steps, 118 s**. ⛔ **The
advantage sees 6 % of the behaviour it is supposed to credit.** `λ = 0.99` gives
77 steps and `λ = 0.995` gives 126. `credit_assignment.md` names the same filter
from the other side: *"GAE accumulates the team component coherently over ~19
effective steps while per-drone terms largely cancel"* — so λ gates whether `D_i`
reaches the gradient at all.

⭐ 📏 **`STAGES[0]` is degenerate, and it is measured.** A policy that does
nothing but servo every drone toward `cue_rel` — no sensing, no roles, no
neighbours, no chain reasoning — scores against B0 at F4/J1, 64 envs, one full
episode per stage:

| stage | cue-follower | B0 | ratio |
|---|---|---|---|
| **1** | **81.6 %** | 87.0 % | **0.94x** |
| 2 | 40.0 % | 90.4 % | 0.44x |
| 3 | 10.9 % | 69.8 % | 0.16x |
| **4** | **6.1 %** | 60.0 % | **0.10x** |

☠️ **A one-line policy scores 94 % of the heuristic at stage 1, and 6.1 % at
stage 4 — below random's 10.7 %.** The first 15 % of training is 100 % stage 1.
📏 Integrated over a run, stage 1 is **24.2 % of episodes** but only **9.2 % of
env-steps**, since its episodes are 150 steps against stage 4's 600 — ⚠️ *less*
exposure than an earlier draft of this section implied.

⚠️ It shows the stage is solvable degenerately. It does **not** show the learned
policy is trapped there; that is the inference Gate F tests. 🔍 Two independent
routes to the same fix: `--curriculum-boundaries` shortens the stage, and
`--cue-mode bearing` removes the shortcut *structurally* — a bearing cannot be
servoed to a point, so "fly here and hover" stops being expressible, while
acquisition (which needs only the bearing, and which B0's own fan uses) survives.

🔒 **Excluded by decision, 2026-09-04**, and both are recorded so they are not
quietly re-litigated:

* ⛔ **an agent index or role embedding.** Roles must **emerge**. B0 is granted
  roles-from-index as a documented advantage; the learned arm is not, and that
  asymmetry is what makes Gate E a test of *credit* rather than of labelling.
* ⛔ **DAgger from B0.** [`bc_init.py`](scripts/bc_init.py) exists and has never
  been reported, and `memory_horizon.md` predicts it fixes the 9.4 % clone. Held
  as the fallback if D, E and F all fail. ⚠️ A teacher-initialised policy is a
  **probe**, not a like-for-like **architecture-ladder** or Gate B arm.

### 4. The second environment — RQ1's replication. No training.

📏 RQ1's dose–response is measured on **one map**, and §8 lists `n = 1
environment` as a risk. The resolution is a **minimal abstract relay
environment**: stations, a threshold, a directed emitter, and the same scripted
controllers varied over `repair_amplitude_m`.

🔒 **It needs no learned policy**, which is why it can sit after the instrument's
box without depending on the box's outcome. If the dose–response reproduces there,
the claim is no longer tied to Frankfurt geometry and `n = 1 environment` becomes
an **external-validity argument** rather than a limitation. ⚠️ If it does not
reproduce, that is a result about RQ1 and it is reported as one.

### 5. The exposé, then the paper

🔒 **After items 1–4.** The claim has moved three times in ten days and §3
re-opened on 2026-09-04; let the data settle it before it is promised to anyone.

### 6. Then: J4, the Orin work, longer runs

| | what | why then |
|---|---|---|
| **J4** | learned jammer, opponent pool | Strengthens RQ1 and RQ2. ⚠️ Gate B stands without it — the fallback declared in §8 |
| **RQ3 / Gate C on hardware** | TensorRT on the Orin, after item 1's local export | The export risk is retired first; the Jetson only ever runs a graph that is known to export |
| **longer runs** | more env-steps per cell | ⚠️ Only if Gate D's regression is understood. 📏 10x the gradient budget already cost **32 pp** |

### Closed — runs 1–3, kept as the record

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

⛔ **Not on the roadmap**, and each for a reason that survives §3's re-opening:

| | why not |
|---|---|
| **further reward *shaping*** | 🔒 §3 line 4 closes it **structurally**, and that line is exact. Team terms cancel from `Var_i(A)` by construction. ⚠️ Gate E is not shaping — it changes the **return**, and `D_i` is factored so it cannot move the equilibrium |
| **further action-space work** | Gate A. ⚠️ It is *also* confounded by the optimisation budget — its velocity seeds "learn normally for the first fifth of the run, then decay" over 46 updates — but 📏 `capable \| observed` is already **0.620 vs B0's 0.617**, so control is not the deficit. ⛔ Re-open only if Gate D promotes and the pathologies persist |
| **recurrence** | Bounded at 0.4 pp for *target* memory by the oracle. ⚠️ Role-commitment memory is left open by `memory_horizon.md` — but that is what Gate E attacks, far more cheaply and at a fraction of the bug density |
| **frame stacking beyond k = 2** | A one-step search state needs no longer history |
| **wider or deeper networks** | 📏 The architecture ladder measured architecture at ±1 pp across three rungs. The actor is 137 k parameters against ~5,900 gradient steps: the budget binds long before the width does |

📏 **And one axis is now measured to be nearly free.** `total_power_w` depends on
**speed, not altitude** — climbing costs a transient `W·v_z/η` and staying high is
free — while `ALT_MAX_M = 80` is a *derived* ceiling. So the optimal altitude is
constant, and 📏 B0's mean `|a_z|` is **0.006** with a standard deviation of
**0.053**, against 0.46 / 0.52 on x / y. ⚠️ A scalar exploration σ therefore
spends a third of its budget on a dimension with nothing to explore, and pays for
it twice: `energy` charges climb power, and leaving the ceiling costs sightlines.
🔧 `--initial-log-std` and `--min-log-std` now take a per-dimension vector.
⛔ Collapsing `ACTION_DIM` to 2 is **not** proposed on that evidence alone — it
breaks every checkpoint — and should follow only if a sweep shows `σ_z` wants to
be ~0.

---

## 8. Risks

| risk | mitigation |
|---|---|
| **Run 1 shows the loop's target does not matter** | §1's mechanism is then wrong and RQ1 reverts to a bare correlation. ✅ The J-ladder decomposition and Gate B's number are unaffected — ⭐ and that is still true now the ladder sits *inside* RQ1, because it measures the **adversary**, not the controller |
| **J4 cycles.** Alternating best response's normal failure | 🔒 RQ1 and RQ3 stand without it, and RQ2's off-diagonal result stands as measured against **scripted** rungs; the strongest adversary reached is then scripted and the paper says so |
| **PyTorch Geometric will not export** | ⚠️ Attempt it *locally*, before touching the Jetson. A GNN that cannot deploy is a **reported result** — its measured advantage over DeepSets is a null anyway |
| **TR 36.777 NLoS intercept is wrong.** Every number re-derives | 🔒 One human reading of one table. Close it before the freeze |
| **n = 1 environment** | ✅ **Resolvable, and cheaply — §7 item 4.** A **minimal abstract relay environment** that reproduces the `repair_amplitude_m` dose–response is the resolution. 🔒 It needs **no learned policy**: the arms are scripted controllers, so it does not queue behind the instrument. That converts the objection from a limitation into an **external-validity argument**, and the mechanism is *named and priced* rather than statistical either way |
| **The claim moves a fourth time** | 🔒 Runs 1–2 are pre-declared attempts to break the current one. If it survives them it has been tested, not just fitted |
| ⭐ **Gates D–F succeed and a learned policy beats B0** | ✅ **This strengthens the thesis rather than threatening it.** §1 already dropped the scripted-vs-learned framing, and it *requires* a capable learned policy to test the loop claim at all. B0 remains the protagonist of RQ1's controlled pair, which is scripted on both sides. 🔒 §3 is an instrument, so this outcome changes what RQ1 can be tested *on*, not what RQ1 claims |
| ☠️ **Gates D–F succeed and every prior null has to be re-read** | ⚠️ The honest cost of §3's re-opening. 📏 Eight nulls, Gate A, Φ v2 and the k = 2 gate were all measured at ~5,900 Adam steps. If Gate D promotes, each needs a one-line re-statement — *"measured under a 10x smaller optimisation budget"* — and the cheap ones (Φ v2, k = 2) should be re-run before the freeze |
| ⚠️ **Gate D promotes and destabilises** | Its PARTIAL branch exists for exactly this: helps on the median, hurts the worst seed. 🔒 `AGENTS.md` judges on the worst seed and that is not relaxed for this programme |
| ⚠️ **`D_i`'s signal is circular** | 📏 The differentiable share it produces is policy-dependent — **35.6 %** on B0 (tenure 295) against **7.98 %** on the learned policy (tenure 47) — so the credit grows as roles emerge. Gate E's validity precondition measures the share **on the trained policy**, and its NULL branch is written to be informative if the bootstrap never starts |

---

## 9. Deliberately not built

📏 Each measured, not assumed — `docs/inherited/DECISIONS.md`.

- **A bigger map, a second city, a better channel.** Compute is not the
  constraint; a 10 M-step run costs 2.2 minutes. ⭐ **But the external-validity
  objection is now answered a different way**: §7 item 4's **abstract twin** — a
  minimal relay environment reproducing the dose–response on scripted controllers
  — replaces "a second city" as RQ1's replication, at a fraction of the cost and
  with no training. See §8. ⚠️ Held-out map *tiles* inside the existing box remain
  free and would answer the in-distribution objection.
- **Flying below 40 m.** TR 36.777 stops at 22.5 m.
- **Transmit power or beamwidth as actions.** Three framings, three nulls, and a
  degenerate optimum for the jammer.
- **Training at more than one `N`.** It turns the zero-shot columns into
  in-distribution tests.
- **An agent index, a role embedding, or DAgger from B0.** ⛔ Excluded by
  decision 2026-09-04 — §7. Roles must **emerge**, and a teacher-initialised
  policy is a probe rather than an arm. DAgger is held as the fallback if
  Gates D–F all fail, or if §3's **three-week box** expires without a capable
  learned policy.
- 🔧 **A per-drone value head or a COMA-style counterfactual baseline.** The
  natural successor to Gate E, and deliberately **not** bundled into it. `A_i =
  G_i − V(s)` and `G_i` is ~99.9 % identical across drones *today*, so an
  agent-specific critic has nothing to fit; with `D_i` on, it does. ⚠️ But the
  direction is ambiguous — a critic that learns *"this drone is the observer, so
  its return is high"* would subtract exactly the signal `D_i` adds. That needs
  its own gate, after Gate E has established there is a signal to preserve.
- 🔧 **A permutation-invariant critic.** On `credit_assignment.md`'s own list:
  the critic is a plain MLP over index-ordered `rel_pos.flatten(1)` while the
  DeepSets and GNN actors are permutation-invariant. ⚠️ Low prior — it cannot
  touch `Var_i(A)` at all, and `explained_variance` already runs 0.94–0.98, so
  there is little headroom to recover.
