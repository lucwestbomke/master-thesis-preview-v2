# History — the settled narrative

**Split out of [`../PLAN.md`](../PLAN.md) §3 on 2026-09-06.** ⛔ **Nothing here was
deleted, edited or renumbered** — every block below is verbatim from where it
stood in `PLAN.md`, and the numbers in it were never restated from memory.

🔍 **Why the split.** `PLAN.md`'s own header says *"this file records what happens
next"*, and §3 had grown into a re-opening, a correction of the re-opening, a
refutation of the confound, and Gate E's result — excellent provenance, and
unusable as a plan. The plan keeps the claim, the RQs, the roadmap, the gates and
the open risks. This file keeps the story of how they got there.

🔒 **This is still binding evidence, not colour.** ⛔ Re-opening the capability
question needs a **new mechanism** and a **gate declared before its run** — not
another sweep of a knob. That rule lives in `PLAN.md` §3 and this file is what it
rests on.

📏 measured · 🔒 constraint · 🔧 provisional — see [`../AGENTS.md`](../AGENTS.md).

---

## 1. The capability question: closed, re-opened, and closed again

⭐ **`PLAN.md` §3 was titled "the capability question" until 2026-09-06**, when it
became **the instrument** — not an objective, but the prerequisite for testing RQ1
on learned controllers. What follows is the record of how it was closed on
2026-09-02, re-opened on 2026-09-04, and closed again on 2026-09-06.

⛔ **The retitling does not soften the re-opening.** ⚠️ **This section said
"closed. Do not re-open." until 2026-09-04.** It was re-opened on a named
confound in the optimiser, and ⭐ **on 2026-09-06 that confound was measured and
refuted** — see below. Four of the five lines were untouched throughout; the
fifth is now restored.

🔍 **The re-opening was still worth it.** It converted *"nobody checked"* into
*"checked, and it is not the explanation"*, which is what §3 needed to be able to
say. And it leaves the search pointed exactly where
[`credit_assignment.md`](../results/credit_assignment.md) pointed it: **the return
and the advantage**, not the optimiser.


## 2. The named confound in the optimiser

### 📏 What changed — the optimisation budget was frozen and never examined

[`docs/inherited/BLOCK_G.md`](inherited/BLOCK_G.md) built three cadences
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

📏 [`capability_gates.md`](../results/capability_gates.md), 2 × 4 factorial, 3 seeds,
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
| 5 | **not memory either**: perfect target state is worth **−0.4 pp** | ⚠️ **stands as a bound on TARGET memory, for B0.** [`memory_horizon.md`](../results/memory_horizon.md) itself leaves **role-commitment** memory open |


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


---

## 3. Where this leaves the search

📏 **Ten pre-declared interventions, and the behavioural metrics do not move.**
`observer_range_m` sits at **187–212 m** against B0's **90**, `observer_tenure` at
**40–47** against **295**, and `role_entropy` at **0.49–0.60** against **0.062**.

🔍 **What survives as a live hypothesis is line 1**, and only line 1: the gap is
`observed` and nothing else. ⭐ That is what promoted **Gate F** from lowest prior
to leading hypothesis on 2026-09-06 — see `PLAN.md` §5. The reasoning is in the
plan, because it is about what happens next; the evidence it rests on is here.

---

## 4. Framings that were refuted

⛔ **Kept in [`../PLAN.md`](../PLAN.md) §6**, not here, because a refuted framing
is part of the evidence for the one that replaced it and belongs where the current
claim can be read against it.
