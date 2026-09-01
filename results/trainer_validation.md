# Trainer validation and compute re-baseline — declared 2026-08-30, before the runs

`docs/REDUCTION.md` task 5 built `src/training/ppo.py`. This file records the
rules **before** the runs that resolve them, because a rule invented afterwards
is not a rule, and two claims in the predecessor project were overturned by
reading a single run after the fact.

Results are appended below the declaration, never edited into it.

---

## ⚠️ The device problem, stated first because it bounds every reading here

**This machine has no CUDA.** `torch.cuda.is_available()` is `False`; it is an
Apple M-series laptop, torch 2.13.0, MPS. Every inherited number in
`docs/INHERITED.md` is a **CUDA** number.

⛔ `AGENTS.md`: *"Compare a number measured on one device with one measured on
another"* is on the never-do list. `torch.Generator` streams differ per device,
so the same seed draws **different episodes** on MPS than on CUDA. The physics is
identical; the sample is not.

🔍 **So the inherited 40.7 % cannot be literally reproduced here, and this
document does not claim to.** What it does instead, declared in advance:

> **B0 is the device anchor.** B0 is deterministic given its observation, is
> scored through the same `evaluate.py`, and its inherited number on the **train**
> split at 5 seeds under CUDA is **59.6 %** (`docs/inherited/BLOCK_G.md`, G8). B0
> is therefore measured here on **MPS**, same split, same harness, first. The
> gap between B0-MPS and B0-CUDA is the size of the device shift on this task at
> this sample size, and it is what the learned number is allowed to be read
> against.

If B0-MPS lands far from 59.6 %, the learned number cannot be compared with 40.7 %
at all, and this document says so rather than quietly adopting the new value.

---

## The validation condition

Exactly the inherited G8 `ff + shipped` cell, on an **unchanged** env:

| | |
|---|---|
| architecture | GNN (`RelationalTrunk`, `use_edges=True`), hidden 128 |
| cadence | `deep` — `num_envs` 4096, `rollouts` 64, `mini_batches` 32, 4 epochs |
| optimiser | Adam, lr 3e-4, `grad_norm_clip` 0.5 joint over actor+critic |
| PPO | `ratio_clip` 0.2, `value_clip` 0.2, `value_loss_scale` 2.5, `entropy_loss_scale` 0.0 |
| GAE | `gamma` = `core.GAMMA` = 0.997 (imported, never retyped), `lambda` 0.95 |
| exploration | `initial_log_std` −0.5, `min_log_std` −20 (**no floor** — what G8 ran under) |
| shaping | `shipped` — `RewardWeights()` defaults, `PHI_V2` off |
| env | F4, curriculum on (`CurriculumSchedule()` default), **train** split, `N` = 5 |
| budget | 12 M env-steps (G8's length, *not* the 10 M sometimes quoted), 5 seeds |
| scoring | `scripts/eval_policy.py --group --train-routes --stage 4 --fidelity F4`, i.e. `src/baselines/evaluate.py` — the one harness B0 went through |

**Reference (CUDA, 5 seeds, train split):** median **40.7 %**, IQR **3.2**, worst
seed **29.6**, observer tenure **43.2**.

### 🔒 One deliberate departure from the inherited runs, declared here

`PPOConfig.shuffle_minibatches = True`. skrl's `sample_all` chunks the flattened
`(rollouts × rows)` buffer **in order**, so at the `deep` cadence a "minibatch" of
40 960 rows is every environment at **two consecutive timesteps** — maximally
correlated. Shuffling is the textbook choice. `--no-shuffle` reproduces skrl's
ordering and is the first thing to try if the gate below fails.

---

## Gate V — does the trainer reproduce the inherited result?

Judged on the **median across 5 training seeds**, with the **worst seed** reported
alongside, as `AGENTS.md` requires.

| | rule |
|---|---|
| **pass** | median `mission_capable` in **[35.0, 46.0] %** *and* worst seed ≥ **25.0 %**, *and* B0-MPS within **2 pp** of 59.6 % |
| **fail** | median outside that band. The trainer has a bug; ⛔ nothing downstream runs until it is found. `--no-shuffle` and `--no-value-norm` are the two declared first suspects |
| **inconclusive** | B0-MPS more than 2 pp from 59.6 %. The device shift is then too large to read the learned number against a CUDA reference, and the gate is unresolved rather than passed |

**Why that band.** ±5.3 pp is ~1.7× the inherited IQR of 3.2, and it has to cover
the bimodality the predecessor measured and did not diagnose: per-seed
`episode_return` ran 4.8 / 83.8 / 90.5 / 96.8 / 108.9 in one cell — roughly **one
catastrophic seed in five**, not a smooth spread. A tighter band would fail on
noise that is already documented.

⛔ **Not** judged on `hop_mean | observed` (it measures geometry) and **not** on
`chain_occluded` (it confounds with hop count, `corr = 0.963`).

---

## Gates Budget-L / Budget-S / Budget-W — was "10 M steps, 5 seeds, 232 hidden" ever the right budget?

📏 The budget was sized against an estimate of **2.8 h per run**. G1b measured
**2.2 min** on an RTX 5090 — ~76× cheaper — and nobody re-spent the savings. This
has never been checked.

📏 **Measured here first, because it sets what is affordable:** MPS, GNN, `deep`
cadence, learner attached — **9 070 env-steps/s** (1024 envs: 5 729; 2048: 6 477).
So 12 M ≈ **22 min**, 100 M ≈ **3.1 h** per seed on this machine. ⚠️ The
predecessor quoted 20 519 env-steps/s on MPS at `num_envs = 256` with the skrl
loop; that is 2.3× this at 16× fewer environments, which does not scale the way
anything else here does. Treat the two as unrelated measurements on unrelated
hardware, not as a regression.

### Budget-L — length. 12 M vs 100 M, one seed each, everything else fixed.

| | rule |
|---|---|
| **falsified** ("you under-trained") | 100 M's `mission_capable` is **not more than +3 pp** above 12 M's |
| **not falsified** | 100 M gains > +3 pp. Then the run length *was* wrong and the deficit is partly budget |

⚠️ **Naming.** These are *budget* gates. ⛔ They have nothing to do with **B0**,
which in this project is the scripted baseline policy and nothing else. An
earlier draft of this file called them B1/B2/B3, which read as a ladder of
baselines. It is not one.

⚠️ **The confound, declared in advance so it cannot be discovered afterwards.**
The predecessor's `c1` control already ran 10 M → 20 M: +4 pp, and **final action
std 0.061**. With `entropy_loss_scale = 0` and `min_log_std = −20` the Gaussian
collapses and the policy stops exploring, so a null at 100 M means *"more steps of
a policy that cannot change buy nothing"* — **not** *"the budget is settled"*.
`BLOCK_G.md` says exactly this and it still stands. **The final `log_std` is
therefore reported next to every Budget-L number**, and if it has collapsed the null is
reported as conditional on that.

### Budget-S — seeds. 20 seeds at the winning length.

| | rule |
|---|---|
| **5 was enough** | resampling 5 of the 20 seeds, the 5-seed median lands within **±3 pp** of the 20-seed median at least **80 %** of the time |
| **5 was not enough** | it does not. Report the seed count the project should have been using |

3 pp is the bar because it is the size of the effects this project judges
interventions on (`w_hold` +1.65, DeepSets→GNN +0.4, recurrence −1.05 — all
inside it).

### Budget-W — width (capacity). GNN trunk 128 → 512, 5 seeds, everything else fixed.

| | rule |
|---|---|
| **capacity binds** | median improves by ≥ **3 pp** *and* the worst seed improves |
| **capacity is not the constraint** | it does not. The 16.1 pp deficit is not a width problem |

⚠️ Read `docs/INHERITED.md` first: the deficit looks **structural**, not
under-trained — the policy is pinned at the action boundary (57 % of steps at the
25 m/s dash cap, 23 % at the map wall, mean `|a_z|` 0.82 while parked at the 80 m
ceiling). Budget-L and Budget-W exist to **falsify** the two cheapest alternative
explanations for twenty minutes of compute, not because either is expected to
pay.

---

# Results

*(appended below as runs complete; the declaration above is not edited)*

## 📏 Gate V — measured 2026-08-30, MPS, 5 training seeds, train split, F4, stage 4

Scored through `scripts/eval_policy.py --group`, i.e. `src/baselines/evaluate.py`
— the harness B0 went through. Five checkpoints, **paired**: every seed scored on
the same 64 evaluation episodes, so the spread reported is training-seed
variation and nothing else.

| policy | per seed | median | IQR | worst |
|---|---|---|---|---|
| **GNN / deep / shipped** | 13.1 · 33.9 · 36.8 · 42.3 · 43.1 | **36.8 %** | **8.4** | **13.1** |
| B0 (5 eval seeds) | — | **58.0 %** | 2.2 | 51.9 |
| random | — | 13.1 % | — | — |

Reference, **CUDA**, 5 seeds, same split and condition: GNN **40.7 % [3.2]**,
worst **29.6**; B0 **59.6 % [2.0]**.

### Verdict: the median clause passes, the worst-seed clause fails

| clause | rule | measured | |
|---|---|---|---|
| device anchor | B0-MPS within 2 pp of 59.6 % | **58.0 %** (1.6 pp) | ✅ |
| median | in [35.0, 46.0] % | **36.8 %** | ✅ |
| worst seed | ≥ 25.0 % | **13.1 %** | ❌ |

⚠️ **A drafting error in the declaration, recorded rather than resolved
conveniently.** The `pass` row requires all three clauses; the `fail` row is
written as "median outside that band". Those do not partition the outcome space,
and this result landed in the gap. The gate is therefore **not** a clean pass and
**not** the declared failure. It is reported as what it is, and the rule is not
rewritten after the fact to make it one or the other.

### What is reproduced, and what is not

**Reproduced — capability.** 36.8 % against 40.7 %, inside the band, on a device
whose shift is independently measured at 1.6 pp. Dropping the collapsed seed, the
remaining four have a median of **39.6 %**. The secondary columns are in family:

| | measured (median) | inherited |
|---|---|---|
| `observed` | 56.2 % | 64.9 % |
| observer tenure | 39.3 | 43.2 |
| observer range | 224 m | 184 m |
| `hop_mean` | 1.05 | 1.27 |

⛔ These are corroboration only. `hop_mean` **measures geometry** and
`chain_occluded` **confounds with hop count** — neither is a finding here, and
both are quoted only because a trainer with a credit-assignment or bootstrap bug
would not land the *tenure* and *stand-off* signature while hitting the headline.

**Not reproduced — variance.** The seed IQR is **8.4 against 3.2**, and the worst
seed is **13.1 against 29.6**. Seed 0 scores 0.1309 where random scores 0.1310 —
indistinguishable on `mission_capable`, though its `observed` of 31.1 % is well
above random's 24.6 %, so it is a collapsed policy rather than a random one.

🔍 **This is the documented bimodality, and it is now reproducible on demand.**
📏 The predecessor measured "roughly one catastrophic seed in five, not a smooth
spread" (per-seed returns 4.8 / 83.8 / 90.5 / 96.8 / 108.9) and `PLAN.md` lists it
as a live risk with the mitigation *"diagnose once the trainer is ours and
instrumentable."* Four of five seeds here land in 33.9–43.1; one collapses. ⚠️ But
it collapses **harder than any inherited seed did**, and 5 seeds on a different
device cannot separate "the known bimodality, worse here" from "a defect in this
trainer". That question is open and is stated as open.

### The controls, which are what say the loop is not broken

| control | reference | measured | |
|---|---|---|---|
| episode-spanning probe, known optimum | 33.0 | **32.9** | ✅ |
| — same probe under skrl's stale GAE mask | — | **12.0**, degrading | ✅ discriminates |
| stage-1, MLP, CPU, 2 M steps, no exploration floor | 82.7 peak / 76.2 final (4 M, `--min-std 0.2`) | **77.9 peak / 68.9 final** | ✅ consistent |
| — same, through `evaluate.py` at stage 1 | random 35.7 % | **62.0 % [5.4]** | ✅ |
| B0 through the same harness | 59.6 % | **58.0 %** | ✅ |
| actor gradient surviving the joint norm clip | — | **`grad_kept` 0.85–0.99** | ✅ not throttled |

⚠️ The stage-1 control is **consistent, not identical**: half the budget and no
exploration floor where the reference had one. It answers "is the loop broken"
(no); it does not certify a number.

📏 **And it re-measured a knob this session had written off.** The stage-1 curve
peaks at 77.9 % and then *decays* to 68.9 % while `log_std` shrinks −0.73 → −0.80
— the exploration collapse `BLOCK_G.md` describes, observed in our own code on a
task whose ceiling is known. 🔒 **Consequence for Budget-L, declared now:** a
100 M-step run with `min_log_std = −20` measures a policy that has stopped
exploring, so Budget-L must run as a 2×2, {12 M, 100 M} × {floor, no floor}, or
its null is uninterpretable.

### What runs next, and what does not

⛔ Budget-L / Budget-S / Budget-W were **not** run. On this machine they cost
~10.7 h against ~20 min on the RTX 5090 the project already has numbers from, and
every one of them is a comparison against a CUDA reference that this device
cannot make. They move to a GPU session unchanged.

The first thing that session should run is this same 5-seed cell on **CUDA**,
against the real 40.7 %. It resolves two things at once: whether 36.8 % is the
device shift, and whether the worst-seed collapse follows the trainer or the
device. Second is the one declared suspect — `--no-shuffle`, the single
deliberate departure from the inherited runs — at 5 seeds. Both are minutes there.

---

## 🔒 Amendment to Budget-L and Budget-W, declared 2026-08-30 **before** the CUDA session

📏 **The measured seed IQR of 8.4 invalidates the single-seed design.** Budget-L
was declared as "12 M vs 100 M, **one seed each**" on the assumption that a run
length effect would be visible above seed noise. It is not: a spread of
13.1–43.1 across five seeds swamps the +3 pp decision threshold several times
over, so a one-seed comparison would measure which seed was drawn.

**Amended, before either arm runs:** Budget-L runs at **5 seeds per cell**, as a
2×2 of {12 M, 100 M} × {no floor (`--min-log-std -20`), floor
(`--min-log-std -1.6`, i.e. sigma >= 0.20)}. The decision rule is unchanged —
"you under-trained" is falsified if 100 M is not more than +3 pp above 12 M on
the **median**, judged also on the **worst seed** — but it is now applied to the
floored arm, because the unfloored arm cannot distinguish "more steps do not
help" from "the policy stopped exploring".

Budget-W (trunk 128 -> 512) keeps its 5 seeds and its rule unchanged.

⚠️ Cost of the amendment on an RTX 5090, at the measured 75 k env-steps/s:
12 M cells ~14 min each, 100 M cells ~110 min each; Budget-L becomes ~4 h instead
of ~25 min. 📏 `AGENTS.md`: "Compute is not a constraint." Buying an
interpretable answer for four hours of a rented GPU is the trade this project has
repeatedly failed to make.

---

## 📏 The frozen critic — diagnosed and fixed, 2026-09-01, CUDA, 5 seeds

### The A/B, against the rule declared before it ran

> **Rule:** the fix keeps the median (>= 39 %, within IQR of 41.1 %) **and**
> raises the worst seed above 25 %.

| condition | per seed | median | IQR | range | worst |
|---|---|---|---|---|---|
| inherited (CUDA) | — | 40.7 % | 3.2 | — | 29.6 |
| ours, **before** the fix | 1.8 · 39.1 · 41.1 · 43.0 · 43.5 | 41.1 % | 3.9 | 41.7 | **1.8** |
| **A — clip 0.2, corrected reference** | 43.9 · 44.4 · 44.5 · 44.5 · 46.3 | **44.5 %** | **0.1** | **2.4** | **43.9** |
| **B — `--value-clip 0`** | 42.7 · 46.2 · 47.0 · 47.6 · 48.0 | **47.0 %** | 1.4 | 5.3 | 42.7 |
| B0, same device and split | — | 60.0 % | 3.2 | — | 54.7 |

✅ **Both arms pass both clauses outright.** The collapse is gone: the worst seed
moves **1.8 -> 43.9**, and the seed IQR **3.9 -> 0.1**.

### The mechanism, confirmed on the same seed

📏 Seed 0, the seed that collapsed, before and after:

| | `grad_norm_critic` | zeros | `capable` end | `at_boundary` end |
|---|---|---|---|---|
| before | min **0.0000**, frozen from progress 0.20 | 2 of 16 log lines exactly zero | 0.026 | **0.875** |
| after | min **0.357**, never zero | 0 of 16 | **0.446** | **0.054** |

The critic never freezes, and the map-boundary pathology it caused (87.5 % of
steps against the wall) disappears with it.

### ⛔ What this invalidates, and it is not small

skrl has **both** defects, verified in its 2.1.0 source: `record_transition`
stores values raw (`_value_preprocessor(values, inverse=True)`) and `_update`
re-normalises them with statistics that have since moved
(`_value_preprocessor(values, train=True)`); and its value clip is a bare
`torch.clip`, which has exactly zero gradient outside its range. **This is skrl
bug number five, and it is the most consequential** — the other four announced
themselves with a collapse, this one silently depressed every reported number and
inflated every seed spread.

⚠️ **Consequence: every inherited learned-policy number was produced with this
defect live.** MLP 31.2 %, DeepSets 38.1 %, GNN 41.2 %, and the RQ2 finding built
on them ("MLP -> DeepSets +6.9 pp robust; DeepSets -> GNN +0.4 pp, a null") were
all measured through it, as were the six pre-declared interventions that came
back null. Those measurements are not *wrong* — they were run correctly against
the trainer of the time — but they are **no longer the best estimate**, and the
architecture ladder has to be re-run before RQ2 is reported.

⚠️ **What it does NOT change:** B0 is unaffected — it is a scripted policy with
no critic. So the headline comparison survives: B0 **60.0 %** against the GNN's
44.5-47.0 %, a gap of **13-15 pp** where it was 16.1. `PLAN.md`'s premise is
intact and the thesis framing is unchanged.

### 📏 Budget-S is answered as a side effect

The seed-count question was declared as "do 5 seeds place the median within
+-3 pp of 20 seeds?" With a measured IQR of **0.1** (arm A) and **1.4** (arm B)
and a full range of 2.4 and 5.3 pp, five seeds are **ample** — the question
dissolves rather than needing 20 runs. ⚠️ It should be re-asked once the
adversary is in play, because an adaptive jammer may restore the variance the
frozen critic was supplying.

### Which arm ships

**Arm A** (`value_clip = 0.2`, corrected reference) is the default: it wins on
the **worst seed** (43.9 against 42.7), which is the metric `AGENTS.md` judges
on, and its spread is 2x tighter, which makes every downstream worst-seed gate
more sensitive.

⚠️ **But B's median is 2.5 pp higher, and that is larger than several effects
this project has judged interventions on** (`w_hold` +1.65, DeepSets -> GNN
+0.4). At 5 seeds the two overlap and the difference is not significant
(Mann-Whitney U = 19 of 25, short of the n=5 threshold of 21). Recorded as an
open question, not settled by preference. Re-decide it at 10 seeds if it ever
sits on a reported result.
