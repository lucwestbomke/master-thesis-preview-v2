# Roadmap — everything left, what it is expected to show, and what would refute it

🔧 Written 2026-09-06. **Execution detail for [`PLAN.md`](../PLAN.md) §7**, split
out the way [`HISTORY.md`](HISTORY.md) was split out of §3. ⛔ `PLAN.md` remains
the canonical statement of the claim, the RQs and the gates; this file says what
to *do*, in what order, and on which machine.

⚠️ **"What we hope it shows" is written here as a hypothesis, never as a target.**
Where a gate exists, **the declared rule governs and this file does not restate
it** — it points at the file that does. Two claims in the predecessor project were
overturned by reading a single run after the fact, and a roadmap that records what
it wants is exactly how that happens.

---

## Where the three RQs stand

| RQ | question | status | can the thesis state an answer today? |
|---|---|---|---|
| **RQ1** | does adaptation create exploitability, and how does it scale? | 🔶 **supported** | ✅ **yes.** The controlled pair and the dose–response are measured and disjoint |
| **RQ2** | can adversarial co-training remove the cost without losing capability? | 🔶 **effect measured, mechanism untested** | ⚠️ **partly.** The effect, yes. *Why*, no |
| **RQ3** | does it survive deployment? | ⛔ **not started** (export risk retired 2026-09-06) | ⛔ **no** |

🔒 **The instrument is not an RQ** and the thesis does not depend on it
(`PLAN.md` §3). It gates only RQ1's *learned* arm, which is a strengthening and
not a requirement — RQ1's evidence is a **scripted** controlled pair.

---

# Track A — runs on the laptop. No GPU, no training. ⭐ Do these first

Both items are the highest scientific return per hour left in the project, and
neither is blocked on anything.

## A1. The abstract relay twin — RQ1's replication

**What.** A minimal abstract relay environment: stations, a rate threshold, a
directed emitter, and the same scripted controllers swept over
`repair_amplitude_m`. No map, no learned policy, no training.

**Hypothesis under test.** The dose–response — 📏 `repair_amplitude_m`
0 / 50 / 100 / 200 m → **7.94 → 13.24 pp** with disjoint endpoints
([`repair_gates.md`](../results/repair_gates.md)) — is a property of *closing a
loop on the attacked quantity*, not of Frankfurt geometry.

**If it reproduces.** ⭐ `n = 1 environment` stops being a limitation in
`PLAN.md` §8 and becomes an **external-validity argument**. This is the single
biggest upgrade available to the thesis's headline claim.

**If it does not.** ⚠️ That is a result about RQ1 and is reported as one. The
claim narrows to *"on this map"*, which is still true and still measured, but the
generality goes.

**⭐ And it can close something the map cannot.** [`gate_b.md`](../results/gate_b.md)
records that **half the chain-length confound survives**: *"a longer chain has
more links for the beam to find"* is not closed, and **no policy in this project
supplies the hop-matched comparison that would close it.** In an abstract
environment the topology is *constructed*, so two controllers with **identical hop
counts** differing only in repair amplitude are buildable by design. 🔍 That makes
A1 do double duty and is why it sits first.

🔒 **Declare before running**: the amplitude levels, the seed count, and the bar
for *"reproduces"* — stated as a relationship (monotone in amplitude, disjoint
endpoints) rather than as a number, since the absolute pp values will not and need
not match the map's.

---

## A2. The redundancy metric — RQ2's mechanism

**What.** *"Does a second threshold-clearing path exist that is edge-disjoint from
the chosen one?"* — computable from the capacity matrix
[`routing.py`](../src/env/routing.py) already builds. ⛔ Nothing measures this
today.

**Hypothesis under test.** `routing.py` picks the widest **single** path and the
jammer has **one** beam. So a policy that maintains a second, edge-disjoint,
threshold-clearing path makes the beam's kill *recoverable*, and that is why
co-training moves exploitability down (📏 11.12 → 7.51 and 10.45 → 7.29 at
unchanged capability, [`gate_b.md`](../results/gate_b.md)).

**⚠️ Two steps, and `PLAN.md` §7 only lists the first.**

| | |
|---|---|
| **A2a** | build the metric. Half a day. It is an **instrument**, not a result |
| **A2b** | ⭐ **the actual mechanism test** — compute it on co-trained against non-co-trained checkpoints and check it moves in the predicted direction |

🔒 **A2b needs a declaration before it is computed**, including the band that
would **refute** the mechanism. With a single-beam jammer and a single-path
router, redundancy is plausible enough that it will be tempting to read whatever
comes out as confirmation. ⛔ `capability_gates.md`'s corrections log has a
precedent for exactly this: Gate E's `> 20 %` gate was *"borrowed from
`measure_credit.py`'s refute band and repurposed as a validity gate without
re-deriving it."*

**If it moves as predicted.** RQ2 goes from *"effect measured"* to *"effect
measured with a named, quantified mechanism"* — the largest single upgrade
available to the weakest RQ.

**If it does not.** ⭐ Also valuable, and honestly so: the effect is real and the
leading candidate mechanism is refuted, which narrows the field to something else
about the geometry the co-trained policies adopt. ⛔ Report it, do not go fishing
for a third candidate in the same session.

---

# Track B — needs CUDA. The instrument's three remaining gates

⚠️ **Do one thing before any of them.** Phase 0 measured `grad_kept` at
**0.035–0.238** on two tasks that reach published reference performance, and at
**0.56–0.89** on the one that learns nothing
([`trainer_benchmark.md`](../results/trainer_benchmark.md)). ⛔ So
`docs/CAPABILITY_BRIEF.md` §4's Block A acceptance test — *"`grad_kept` > 0.8 and
`approx_kl` above 0.005"* — **would reject both configurations that work.** Gate on
`approx_kl` and `clip_fraction` instead. 🔒 That declaration has not been made yet,
so the fix is free now and expensive after.

## B1. ⭐ Gate F — the leading hypothesis

**Where everything for it already lives** — ⛔ nothing needs to be invented:

| what | where |
|---|---|
| the motivating measurements (F1 cue staleness, F2 cue-follower, F3 broadcast features) | [`capability_gates.md`](../results/capability_gates.md) → *"Gate F — is the observation lying to the policy?"* |
| 🔒 **the decision rule** — PROMOTE / NULL / REGRESSION at ±3 pp | same section → *"The decision rule"* |
| 🔒 **the exact commands** | same file → *"The runbook"* → *"Gate F — two axes, never pooled"* |
| what happens to the **programme** on each outcome | [`PLAN.md`](../PLAN.md) §5 → *"The branch after Gate F"* |

**Hypothesis under test.** 📏 A one-line cue-follower — servo every drone at
`cue_rel`, no sensing, no roles, no chain reasoning — scores **81.6 %** at
curriculum stage 1, which is **0.94×** B0, and **6.1 %** at stage 4, *below*
random's 10.7 %. The first **15 % of training is 100 % stage 1**, which is where
the basin is chosen. 🔍 So the policy may be learning a shortcut that is
near-optimal early and worthless later. `--cue-mode bearing` removes the
shortcut **structurally** — a bearing cannot be servoed to a point — while
acquisition, which needs only the bearing, survives.

**⛔ Three constraints from the file, not to be relaxed:**

* the F1/F2 and F3 arms are **run separately and never pooled** — pooling lets
  F3's expected null bury F1/F2, or lets F1/F2 launder F3;
* **B0 must be scored under `cue_mode="position"` and `mask_broadcast_obs=False`** —
  its acquisition fan and its link repair both read those, and a B0 number
  measured under anything else is not the baseline;
* `--curriculum-boundaries` takes **three** values and is not an `--axis`; run it
  as two explicit conditions via `--train-arg`.

**If it promotes.** The programme continues **inside the box**, ablating which of
`--cue-mode`, `--curriculum-boundaries` and `--curriculum-mix` did it. ⛔ A pooled
result is not an attribution.

**If it nulls.** ☠️ **The instrument programme is finished.** No further capability
gate is proposed, [`bc_init.py`](../scripts/bc_init.py) manufactures RQ1's
learned-loop arm as a **probe**, and writing begins. ⚠️ A teacher-initialised
policy is *not* a like-for-like arm for the architecture ladder or for Gate B, and
every result using it must say so.

📏 **Cost.** ~5 min a run; the cue axis is 3 values × 3 seeds.

## B2. Gate G — is the final checkpoint the right one to score?

**Where.** [`capability_gates.md`](../results/capability_gates.md) → *"Gate G"*,
declared 2026-09-06 with its rule and its cost.

**Hypothesis under test.** 📏 Every training curve on record peaks partway through
and decays — **6 of 6** runs — and no mid-run checkpoint has ever been scored,
because none was ever saved. If the peak survives an honest evaluation, **every
learned number in `results/` is an under-report.**

**⚠️ Nearly free.** Gate E's runs were launched with `--train-arg
checkpoint-every`, so `runs/gateE*/sw-*-s*/checkpoint-pNNN.pt` should already
exist **on the CUDA host** — ⛔ they are not on the laptop; `runs/` here holds only
`bc-gnn-s0`, `stage1-control-s0` and `val-gnn-deep-s0..4`. Only the *scoring* is
outstanding.

🔒 **The protocol binds**: select the checkpoint on the **train** split, report
that one checkpoint on **eval**. Picking the best on eval converts the one
generalisation check this project has left into a validation set.

## B3. Gate D2 — was the −32 pp the step count or the bundle?

**Where.** [`PLAN.md`](../PLAN.md) §5 → *"Gate D2"*, declared with all three
branches.

**Hypothesis under test.** Gate D's budget arm bundled **five** knobs and scored
13.10 % against a 45.18 % control. ⛔ Nothing was ever attributed, so the statement
the project can currently support is *"that configuration is harmful"* — **not**
*"more optimisation does not help."* `--mini-batch-size 4096` **alone** separates
them.

**If the regression reproduces.** ⭐ Gate D's verdict can be stated **without** the
bundle caveat, which is the entire point of running it.

**If it attributes elsewhere.** Gate D's verdict narrows, and the four-knob
ablation is owed before any claim about the budget. 🔧 `--min-log-std -1.6` is the
one to drop first — it is the only knob that prevents the policy sharpening at all.

📏 **Cost.** 10 runs, ~1 GPU-hour. ⚠️ Lowest priority of the three: it tidies a
verdict rather than opening a question.

---

# Track C — needs the Jetson Orin Nano. RQ3, end to end

⭐ **Independent of Tracks A and B.** Gate C asks about **relative** degradation
under quantisation, so *any* existing learned checkpoint serves —
`runs/val-gnn-deep-s0..4` are sufficient. ⛔ **RQ3 is not blocked on the
instrument**, and it is the most mechanical work left in the project.

✅ **C0 is done.** The ONNX half of the export risk is retired: all three rungs
export and are numerically exact to ~1e-7 at three different row counts
([`capability_log.md`](../results/capability_log.md)). 🔍 PyTorch Geometric was
never on the actor's forward path, because `RelationalTrunk` is a custom MPNN in
plain torch. ☠️ And the legacy TorchScript exporter silently bakes the traced batch
into the `deepsets` rung — use the default exporter, and re-run
`scripts/export_onnx.py` after any architecture change.

| | what | expected to show |
|---|---|---|
| **C1** | ONNX → TensorRT on the Orin | ⚠️ the unretired half of the export risk. A rung that converts is a deliverable; one that does not is a **reported result** |
| **C2** | latency, **p99 jitter**, power against the **400 ms** control period | the sim-to-real line, which is most of RQ3's value even if C3 nulls |
| **C3** | 🔒 **Gate C** — declared 2026-08-27, reproduced verbatim in `PLAN.md` §5 | does quantisation degrade **coordination** (`role_entropy`, `observer_range_m`) proportionally more than **control** (`mission_capable`)? |

**Hypothesis under test at C3.** Coordination is a *relational* property computed
across drones and may be carried by small activation differences that int8
quantisation flattens, while basic flight competence is robust. ⭐ If so, that is
a new and deployment-relevant claim.

**If it nulls.** Degradation is proportional; report latency and power as an
engineering result and move on. ⛔ Explicitly allowed for by the declaration.

---

# Track D — after the above

| | what | condition |
|---|---|---|
| **D1** | the exposé, then the paper | 🔒 after Tracks A–C. The claim moved three times in ten days; let the data settle it before it is promised to anyone |
| **D2** | **J4** — a learned jammer with an opponent pool | ⚠️ strengthens RQ1 and RQ2; Gate B stands without it. ⛔ RQ2's result currently holds against **scripted** adversaries only |
| **D3** | longer runs | ⚠️ **only if Gate D's regression is understood** — 📏 10× the gradient budget already cost 32 pp. B3 is the prerequisite |

---

# Held, and not scheduled

| | why it is not on the roadmap |
|---|---|
| **gSDE on the relay** | ✅ Implemented, tested and **ships off**. 📏 It converts `MountainCarContinuous-v0` from 2/5 seeds solved to 5/5, and the magnitude confound was tested and refuted. ⛔ That says nothing about the relay, and the base rate here is eight nulls plus Gates A, D and E. 🔍 It earns a gate only if Gate F promotes and saturation is still the limiting behaviour — 📏 the learned policies sit at the speed cap on 57 % of steps and at the boundary on 15–23 %, against B0's 3.1 % and 0.9 %, which is the same *saturation without displacement* signature the failing benchmark cells show |
| a **high-dimensional trainer benchmark** | ⚠️ Phase 0 validated the trainer on 3-D and 4-D observation spaces only. MuJoCo and Box2D are dependency additions `AGENTS.md` requires flagging. 🔧 Worth it only if a reviewer asks |
| a **truncation-bootstrap behavioural test** | ☠️ Genuinely open. `probe.py` misses it (📏 32.4 against 32.7) and both Phase 0 tasks miss it (📏 0.937 against 0.961), all three for the same structural reason — the value at truncation is near zero. It is covered by unit tests only. Closing it needs a task whose optimal cost-to-go is **large** at the horizon |
| everything in `PLAN.md` §9 | deliberately not built; unchanged |

---

# The minimum viable thesis, if everything else fails

🔒 Stated so the roadmap cannot become a hostage. **Today**, with no further runs:

* **RQ1 is answered** on a scripted controlled pair with disjoint ranges and a
  dose–response. ⚠️ Limitations: one map, and half the chain-length confound open.
* **RQ2 reports a measured effect** with an untested mechanism and a scripted
  adversary. ⚠️ Stated as such.
* **RQ3 reports** that all three architectures export cleanly to ONNX, with a
  silent-failure mode in the legacy exporter documented. ⛔ No hardware numbers.
* The **instrument** reports ten pre-declared interventions, a structural
  explanation for why `Var_i(A) = Var_i(G)`, and a trainer validated against
  published PPO baselines — 📏 which is what makes those nulls facts about the
  task rather than about the optimiser.

⭐ **A1 and A2 are what turn that from defensible into good, and both run on the
laptop.**
