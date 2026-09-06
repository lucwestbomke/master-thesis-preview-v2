# Contested Relay — the plan

**Rewritten 2026-09-04**, around the claim the frontier run produced. The two
framings before this one are recorded in §6, because a claim that was refuted is
part of the evidence for the one that replaced it.

⚠️ **Amended 2026-09-04, and resolved 2026-09-06.** §3 said *"closed, do not
re-open"*, was re-opened on a **measured confound in the optimiser** that every
number in `results/` shares, and that confound was then tested and went the other
way. ⛔ Both the amendment and its resolution are recorded rather than made
silently, for the same reason §6 keeps the refuted framings — and the full record
now lives in [`docs/HISTORY.md`](docs/HISTORY.md), split out on 2026-09-06 so this
file can say what happens next.

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

📏 **Two of the three are already spent.** D and E are **closed and both NULL**
(§5) — λ null-to-harmful with 10x the budget at **−32 pp**, and per-drone credit
over a **12x** weight range sitting inside a single cell's seed noise. 🔒 The box
is a *budget*, not a schedule: what it has left is **F** and anything the
programme proposes after it.

⛔ **The retitling does not soften the re-opening**, and the re-opening is
correct. 🔒 §3 was titled *"the capability question"* and said *"closed. Do not
re-open."* until 2026-09-04; it was re-opened on a **named, measured confound in
the optimiser**, that confound was tested on 2026-09-06, and it went the *other*
way. The full record — the re-opening, the arithmetic, the correction of a claim
made during it, Gate D's verdict, the five lines with their statuses, and Gate E —
is in [`docs/HISTORY.md`](docs/HISTORY.md). ⛔ **Nothing was deleted**; it moved,
verbatim, so that this file can say what happens next.

### 📏 Where the search stands, in three lines

| | status |
|---|---|
| **the optimiser** | ⛔ **closed.** Gate D: λ null-to-harmful, and 10x the gradient budget costs **32 pp**. *"The policy cannot move"* is not the binding constraint |
| **the advantage** | ⛔ **closed.** Gate E supplied the per-drone credit `credit_assignment.md` measured as absent — **35.5 %** differentiable share against a **0.04–0.16 %** control class — and `role_entropy` (0.492 → 0.603) and `observer_range_m` (192.8 → 211.7 m) got **worse** |
| **the observation and the curriculum** | ⭐ **open, and now the leading hypothesis.** Gate F. See §5 |

🔒 **What survives from the five lines is line 1, and it is the target**: the gap
is `observed` and nothing else — conditioned on a sightline the GNN converts it as
well as B0, **0.620 against 0.617**. All five lines, with their statuses, are in
[`HISTORY.md`](docs/HISTORY.md).

🔒 **Re-opening this section again needs a NEW mechanism and a gate declared
before its run** — not another sweep of a knob. ⚠️ That rule is what
`docs/HISTORY.md` exists to enforce: ten pre-declared interventions have now
failed to move `observer_range_m` (**187–212 m** against B0's **90**),
`observer_tenure` (**40–47** against **295**) or `role_entropy` (**0.49–0.60**
against **0.062**).

### 📏 What has NOT changed

⛔ **B0 is still the strongest policy in this project**, 57.3 % [54.8 – 60.6]
against the best learned 40.7 %. ⛔ **B0's design advantage is still ~10.3 pp**
against a 15.0 pp gap ([`b0_ablation.md`](results/b0_ablation.md)), so acquiring
every B0 component would still not close it — which is why Gates D–F attack the
optimiser, the advantage and the observation rather than trying to clone the
heuristic.

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
| **A** | method (closed) | velocity setpoints as the action space | ⛔ **not met** — 18.3 pp cost, disjoint. [`gate_a.md`](results/gate_a.md) |
| **B** | **RQ1** | is the heuristic more exploitable? | ✅ **confirmed** — and survived its own `capable_no_division` control. [`gate_b.md`](results/gate_b.md) |
| **C** | **RQ3** | does quantisation hurt coordination more than control? | ⛔ **not run** |
| **D** | the instrument | is the learned policy **optimisation-limited** rather than credit-limited? | ⛔ **NULL / REGRESSION, 2026-09-06** — λ is null-to-harmful and 10x the budget costs **32 pp**. [`capability_gates.md`](results/capability_gates.md) |
| **E** | the instrument | does **per-drone credit** (`D_i = G − G_{−i}`) produce roles? | ⛔ **NULL, 2026-09-06** — 7 weights over a 12x range, whole axis inside single-cell noise; `role_entropy` and `observer_range_m` get **worse**. ⭐ And the signal *reached the gradient*: 35.5 % differentiable share against a 0.04–0.16 % control class |
| **F** | the instrument | is the **observation** lying to the policy, and is the **curriculum** teaching a shortcut? | ⭐ **not run — and now the LEADING hypothesis, 2026-09-06.** Promoted from lowest prior: D and E are null, and 📏 a one-line cue-follower scores **94 %** of B0 at stage 1 and **6.1 %** at stage 4, below random's 10.7 %. §7 |
| **G** | the instrument | is the **final checkpoint** the right one to score? | ⛔ **not run** — declared 2026-09-06 in [`capability_gates.md`](results/capability_gates.md). ⚠️ It applies to the shipped configuration as much as to the budget one, so it is *not* a Gate D re-run |
| **D2** | the instrument | is the −32 pp the **step count**, or the four knobs bundled with it? | ⛔ **not run** — declared below |
| Φ v2 | the instrument | does a steeper potential move the observer? | ⚠️ **killed — and confounded.** 11.8 m of a needed 20 m, measured under ~5,900 Adam steps. [`HISTORY.md`](docs/HISTORY.md) |
| k = 2 | the instrument | does one step of history buy link repair? | ⚠️ **inconclusive** — +1.94 pp, worst seed −1.25 |

⚠️ **Gates D, E and F re-opened §3, and two of the three have resolved.** D is a
NULL/REGRESSION and E is a NULL; **F is what is left**, and it is now the leading
hypothesis rather than the lowest prior. Every branch of each is declared before
its run and each partitions the outcome space — ⛔ Gate A and
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

### Gate D2 — is the −32 pp the step count, or the four knobs bundled with it?

🔒 **Declared 2026-09-06, before the run. Branches partition the real line.**

⚠️ **Not a rescue, and no branch of it un-does Gate D.**
[`capability_gates.md`](results/capability_gates.md)'s REGRESSION branch says
*"do not rescue it by re-tuning"*, and this does not re-tune: it asks **which of
five knobs** produced the collapse. 🔒 Same standing as Gate G, which is declared
in that file on the same reasoning.

📏 **Why it is owed.** Gate D's budget arm is
`--mini-batch-size 4096 --target-kl 0.015 --grad-norm-clip-critic 1.0
--orthogonal-init --min-log-std -1.6`, and the file declares it a **screening**
arm: *"the budget arm bundles five knobs, and that is deliberate."* It scored
**13.10 %** against the control's **45.18 %**. ⛔ The ablation was declared owed
*"if it promotes"* — it regressed instead, so nothing was ever attributed, and the
statement the project can currently support is **"that configuration is
harmful"**, not **"more optimisation does not help."**

⛔ **The learning rate was not one of the five.** `--target-kl 0.015` switches on
the KL-adaptive controller in `src/training/ppo.py`; 📏 `lr_actor` rose to
**5.13e-3** and *plateaued* at half of `lr_max`, holding `approx_kl` in
**[0.0027, 0.0107]** against the **0.015** target. The controller's dead band
held, and *"the `--target-kl` controller ran away"* is already recorded as
**refuted** in that file's corrections log. ⚠️ So a clean arm does not need to
pin the LR — it needs to **not set `--target-kl`**, and the actor LR then stays at
the shipped constant by construction.

| | rule |
|---|---|
| **treatment** | `--mini-batch-size 4096` **alone**. 📏 5,888 → ~58,900 Adam steps at essentially unchanged FLOPs |
| **control** | the shipped defaults at the λ Gate D selected. A **re-run**, not a number quoted from another code state |
| **judged on** | `Δ = median(treatment) − median(control)`, capability, train split, **5 seeds**, with the **worst seed** reported beside it |

| branch | rule | consequence |
|---|---|---|
| ☠️ **REGRESSION REPRODUCED** | `Δ ≤ −5 pp` | The step count alone is harmful. ⭐ Gate D's verdict can then be stated **without** the bundle caveat, which is the point of running this |
| ⚠️ **ATTRIBUTED ELSEWHERE** | `Δ ≥ +3 pp` | The collapse came from one of the other four. ⛔ Gate D's verdict narrows to *"that configuration is harmful"* and the four-knob ablation is owed before **any** claim about the budget. 🔧 `--min-log-std -1.6` is the one to drop first — it floors σ at `exp(−1.6) ≈ 0.20` against a shipped default of `−20.0`, i.e. no floor, so it is the only knob that prevents the policy sharpening at all |
| ⛔ **NULL** | `−5 pp < Δ < +3 pp` | The step count neither helps nor hurts. *"The budget does not bind"* stands, and the −32 pp is attributed to the bundle rather than to optimisation |

📏 **Cost.** 10 runs at ~5 min ≈ **1 GPU-hour**.

### The branch after Gate F

🔒 **Declared 2026-09-06, before Gate F runs.** ⚠️ Gate F's **own** decision rule
lives in [`capability_gates.md`](results/capability_gates.md) and is **not**
restated here. This declares what happens to the **programme** on each outcome,
which nothing currently says.

🔒 **The trigger is Gate F resolving, not the calendar.** The three-week box (§3,
to **2026-09-27**) is the backstop for the case where F does not resolve at all —
📏 D and E together cost about a day, so the binding question is the outcome, not
the budget.

| Gate F, F1/F2 arm | what happens to the instrument programme |
|---|---|
| ✅ **moves capability by its declared rule** | The programme continues **inside the box**: ablate which of `--cue-mode`, `--curriculum-boundaries` and `--curriculum-mix` did it. ⛔ A pooled F1 + F2 result is not an attribution, and the file already forbids pooling the arms |
| ⛔ **nulls** | ☠️ **The instrument programme is finished.** No further capability gate is proposed, [`bc_init.py`](scripts/bc_init.py) manufactures the learned-loop arm for RQ1 as a **probe**, and writing begins. ⚠️ The paper then reports that the scripted baseline remains the strongest policy — which §3 already says the thesis survives |

⛔ **The F3 arm does not extend the box either way.** Its prior is low by
measurement: [`obs_mask_gate.md`](results/obs_mask_gate.md) masked nine features
for a null, and that is the base rate for observation surgery here.


## 6. Framings that were refuted, and why they are kept

⛔ **A refuted framing is evidence, not embarrassment.** Each was killed by a run
designed to test it, and the sequence is why the current claim should be trusted.

🔒 **The narrative behind row 1 lives in [`docs/HISTORY.md`](docs/HISTORY.md)** —
the re-opening, its arithmetic, the correction made during it, and Gate D's
verdict. ⛔ Split out on 2026-09-06, verbatim; nothing was deleted.

| framing | killed by | when |
|---|---|---|
| *Learned control beats the scripted baseline* | 8 nulls, then §3's five lines | 2026-09-02 — ⚠️ **partially reinstated 2026-09-04**, then ⭐ **re-refuted 2026-09-06.** The reinstatement said the eight nulls were *confounded*, not refuted, because all were measured at ~5,900 Adam steps. 📏 Gate D tested that confound directly and it went the **other** way — 10x the budget costs 32 pp — so the nulls stand, and now stand *tested*. [`HISTORY.md`](docs/HISTORY.md) |
| *The adversary ladder is non-monotone; adaptivity does not help* | the 5-seed CUDA re-run **reversed** a one-seed CPU result | 2026-09-03 |
| *Exploitability is a cost of **capability*** | the frontier run: `b0-geodesic` is **more capable than every learned policy and less exploitable than all of them** | 2026-09-04 |
| *Four objectives, with the capability programme as an RQ* | superseded by three RQs; the capability programme is an instrument, not an objective | reframed 2026-09-06 |

⚠️ **The last row is a *reframing*, not a refutation.** No measurement killed it —
the organisation of the same evidence changed, and it is recorded here rather than
applied silently. §2 carries the old → new mapping.

🔒 **The third was refuted *before it was declared*,** because it was fitted over
eight policies rather than written down after four. ⛔ Hold the current claim to
the same standard: §7's closed runs 1 and 2 were built to break it, and §7 item 4
is the next attempt.

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

⭐ **Execution detail lives in [`docs/ROADMAP.md`](docs/ROADMAP.md)**, split out
2026-09-06 the way [`docs/HISTORY.md`](docs/HISTORY.md) was split out of §3. This
table stays as the ordering and the rationale; that file carries, per item, the
hypothesis under test, the branch that would refute it, the machine it runs on,
and where its declaration lives. ⛔ It declares nothing of its own.

| | what | serves | cost |
|---|---|---|---|
| **1** | ✅ **ONNX export, locally** — **done 2026-09-06**, all three rungs | RQ3 | days, no training |
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

#### ✅ **Done, 2026-09-06. The risk is retired — and the export found a bug.**

📏 [`scripts/export_onnx.py`](scripts/export_onnx.py), `onnx 1.22.0` /
`onnxruntime 1.29.0`, opset 20, all three rungs checked numerically against the
PyTorch module at three different row counts. Full table in
[`results/capability_log.md`](results/capability_log.md).

* ✅ **All three rungs export and are numerically exact** (max abs error < 1.3e-07)
  under torch 2.13's default `dynamo` exporter. 🔍 **PyTorch Geometric was never
  the risk it looked like**: `RelationalTrunk` is a custom MPNN in plain torch —
  forced by `docs/inherited/MODELS.md`'s rule that the DeepSets rung must be the
  GNN with `e_ij` zeroed — so **PyG is not on the actor's forward path at all.**
* ☠️ **Under the legacy TorchScript exporter the `deepsets` rung is silently
  wrong.** It exports, passes `onnx.checker`, runs at the traced batch size, and
  throws at any other one. 📏 The traced graph carries a `Constant` of shape
  `[5, 7, 2]` — traced batch × neighbour slots × `EDGE_DIM` — which is
  `torch.zeros_like(edge)` under `use_edges=False`, constant-folded at the traced
  shape. ⚠️ `deepsets` is `scripts/train.py`'s **default** architecture and the
  off-N transfer columns run the actor alone at `N ∈ {3, 8}`, so this would have
  been correct at `N = 5` and broken everywhere else, with no error until it ran.
* ⛔ **TensorRT is still unretired.** This closes the ONNX half only.

### 2. Run 4 — the redundancy metric. RQ2. Half a day, then zero training.

*"Does a second threshold-clearing path exist that is edge-disjoint from the
chosen one?"* — computable from the capacity matrix `routing.py` already builds.
⛔ Nothing measures this today, and it is **RQ2's candidate mechanism**.

### 3. ⭐ Runs 5–7 — the capability programme. Gates D, E, F. The instrument.

🔒 **Time-boxed at three weeks (§3).** ~4 GPU-hours of compute; the box is on the
search, not the hardware. 📏 **D and E are closed and both NULL** (§5); **F is
what the box has left**. ⭐ **And F is no longer the lowest prior — it is the
leading hypothesis** (§5): D and E eliminated the optimiser and the advantage, and
the cue-follower measurement below is direct evidence for the remaining one.
📏 [`capability_gates.md`](results/capability_gates.md) already calls the cue and
curriculum arm *"the best-motivated intervention in this file"*; this file had not
caught up.

🔒 **The next two runs, in order**, both declared before they run and both cheap:

| | run | why |
|---|---|---|
| **i** | **Gate D2** — `--mini-batch-size 4096` alone, 5 seeds | ~1 GPU-hour. It converts *"that configuration is harmful"* into *"more optimisation does not help"*, or shows the collapse belongs to one of the four other knobs. §5 |
| **ii** | **Gate F, `--cue-mode bearing`** | ⭐ The structural arm: a bearing cannot be servoed to a point, so *"fly here and hover"* stops being expressible while acquisition — which needs only the bearing, and which B0's own fan uses — survives. ⛔ It removes the shortcut from the hypothesis space rather than tuning around it |

🔒 **Declared in full, before any run:**
[`results/capability_gates.md`](results/capability_gates.md). Ordered so that each
is interpretable given the one before it.

| | question | instrument | why it is first / next / last |
|---|---|---|---|
| **D** | is the policy **optimisation-limited**? | `--mini-batch-size`, `--target-kl`, `--grad-norm-clip-critic`, `--orthogonal-init`, `--min-log-std`, and 📏 **`--gae-lambda`, never swept in this project's history** | It gates everything. Its NULL branch is a real result: it removes the confound from all eight prior nulls and makes [`credit_assignment.md`](results/credit_assignment.md) *stronger* |
| **E** | does **per-drone credit** produce roles? | `--w-difference` — `D_i = G(z) − G(z_{−i})`, the mission term recomputed with drone `i` deleted, exactly | The successor axis `credit_assignment.md` names. ⛔ Only interpretable on a policy that can actually train, hence after D |
| **F** | is the **observation** lying to the policy, and is the **curriculum** teaching a shortcut? | `--cue-mode`, `--curriculum-boundaries` / `--curriculum-mix`, `--mask-broadcast-obs` | ⭐ **Now the leading hypothesis, not the last resort.** ⚠️ **Two arms, very different priors, never pooled.** The cue/curriculum arm is measured below and 🔧 **`--cue-mode bearing` is the instrument to run first** — it removes the shortcut *structurally* rather than tuning around it. The broadcast-feature arm keeps `obs_mask_gate.md`'s null prior |

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

#### 🔍 Added 2026-09-06 — the shortcut hypothesis RETRO-PREDICTS §3's line 1

⚠️ **An inference, not new evidence.** Both halves have been on the record for
days and nothing joined them; this joins them and claims nothing measured.

📏 §3 line 1: *the gap is `observed` and nothing else — conditioned on a sightline
the GNN converts it as well as B0, **0.620 against 0.617***. That is a **very
specific** signature: the learned policy is not worse at closing a chain, it is
worse at **being somewhere it can see from**.

🔍 A policy that learned *servo-to-the-cue-and-hover* predicts exactly that
signature and no other. It fails to **acquire**, because the cue is a position
that goes stale — 📏 median `|cue − hvt|` **984 m** at `t = 599` against a **127 m**
along-street sightline median — while **conversion given a sightline stays
normal**, because closing the chain is the part the shortcut never had to give up.
⭐ And the cue-follower measurement shows the degenerate policy in its pure form
doing precisely this: at stage 1 its `capable` **equals its `observed` exactly**,
81.6 / 81.6.

⛔ **This is a consistency, not a discriminating test, and it does not become
one.** Other mechanisms also produce an `observed`-only gap, and 🔒 **Gate F's
decision rule is declared in [`capability_gates.md`](results/capability_gates.md)
and is not touched by this paragraph.** What it changes is the **prior**: the
leading hypothesis now explains the shape of the deficit and not only the shape of
the curriculum, which is a reason to run Gate F first and not a reason to expect
it to pass.

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

### ⛔ Not on the roadmap

⛔ **Not on the roadmap**, and each for a reason that survives §3's re-opening:

| | why not |
|---|---|
| **further reward *shaping*** | 🔒 [`HISTORY.md`](docs/HISTORY.md)'s line 4 closes it **structurally**, and that line is exact. Team terms cancel from `Var_i(A)` by construction. ⚠️ Gate E is not shaping — it changes the **return**, and `D_i` is factored so it cannot move the equilibrium |
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
| ☠️ **Gate B's surviving confound is never closed** | ⭐ **The largest open risk to RQ1.** The rate-division half is refuted (B0 13.24 → 12.85 pp), but *"a longer chain has more links for the beam to find"* is not, and ⛔ **no policy in this project supplies the hop-matched comparison** that would close it. 🔒 §7 item 4's abstract twin can hold hop count fixed by construction — that is a second reason to build it |
| ⛔ **Gate F nulls too** | The instrument programme is then **finished**, and that branch is declared in §5 rather than decided afterwards: `bc_init.py` supplies RQ1's learned-loop arm as a **probe**, and writing begins. 🔒 §3 already states the thesis does not depend on the outcome |
| **J4 cycles.** Alternating best response's normal failure | 🔒 RQ1 and RQ3 stand without it, and RQ2's off-diagonal result stands as measured against **scripted** rungs; the strongest adversary reached is then scripted and the paper says so |
| **PyTorch Geometric will not export** | ⚠️ Attempt it *locally*, before touching the Jetson. A GNN that cannot deploy is a **reported result** — its measured advantage over DeepSets is a null anyway |
| **TR 36.777 NLoS intercept is wrong.** Every number re-derives | 🔒 One human reading of one table. Close it before the freeze |
| **n = 1 environment** | ✅ **Resolvable, and cheaply — §7 item 4.** A **minimal abstract relay environment** that reproduces the `repair_amplitude_m` dose–response is the resolution. 🔒 It needs **no learned policy**: the arms are scripted controllers, so it does not queue behind the instrument. That converts the objection from a limitation into an **external-validity argument**, and the mechanism is *named and priced* rather than statistical either way |
| **The claim moves a fourth time** | 🔒 It has already survived three pre-declared attempts to break it — the loop's target (null), its amplitude (dose–response) and Gate B's division control. ⚠️ What is *not* yet tested is a **second environment**; §7 item 4 is the next attempt, and it is designed to fail informatively |
| ⭐ **Gate F succeeds and a learned policy beats B0** | ✅ **This strengthens the thesis rather than threatening it.** §1 already dropped the scripted-vs-learned framing, and it *requires* a capable learned policy to test the loop claim at all. B0 remains the protagonist of RQ1's controlled pair, which is scripted on both sides. 🔒 §3 is an instrument, so this outcome changes what RQ1 can be tested *on*, not what RQ1 claims |
| ☠️ **A late promotion forces every prior null to be re-read** | ⚠️ The honest cost of §3's re-opening. 📏 Eight nulls, Gate A, Φ v2 and the k = 2 gate were all measured at ~5,900 Adam steps. ⛔ **Gate D did not promote**, so this has not fired — but **Gate G** would fire it from a different direction, since it asks whether every learned number in `results/` was read at the wrong point. If either promotes, each null needs a one-line re-statement and the cheap ones (Φ v2, k = 2) should be re-run before the freeze |
| ⚠️ **A gate promotes on the median and hurts the worst seed** | Gate D's PARTIAL branch existed for exactly this, and the rule generalises. 🔒 `AGENTS.md` judges on the **worst seed** and that is not relaxed for this programme, Gate D2 and Gate F included |
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
