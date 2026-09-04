# Can a learned policy clear B0? — three gates, declared 2026-09-04

🔒 **Declared before any run, and never edited afterwards.** Results are appended
under each rule. `AGENTS.md`: *"A rule invented after the fact is not a rule."*

⚠️ **This file re-opens an axis [`PLAN.md`](../PLAN.md) §3 closes.** That is
deliberate and the argument is in §0 below. It is *not* a ninth reward
intervention: [`credit_assignment.md`](credit_assignment.md) closed the reward
axis structurally **and named its own successor** — *"What is left is the critic
and the advantage, none of which has been touched."* Gates D and E are on that
successor axis; Gate F is a separate, lower-prior question about the observation.

---

## The reference, and what "clears B0" means

📏 **B0, eval split, stage 4, F4, J1, 5 seeds × 128 episodes**
([`frontier.jsonl`](frontier.jsonl)):

| per seed | median | worst | best |
|---|---|---|---|
| 56.1 · 60.6 · 60.0 · 54.8 · 57.3 | **57.3 %** | **54.8 %** | **60.6 %** |

| | `observed` | `observer_tenure` | `role_entropy` | `observer_range_m` | `hop_mean` |
|---|---|---|---|---|---|
| **B0** | 92.8 % | **294.7** | **0.062** | **90.0 m** | 2.13 |
| best learned (GNN) | 66.5 % | **47.2** | **0.55** | **218.9 m** | 1.20 |

🔒 **The declared bar, in this project's own standard.** Gate A and RQ2 both judge
on *disjoint seed ranges*, so:

| | rule |
|---|---|
| **clears B0** | median `mission_capable` > **57.3 %** |
| ⭐ **beats B0 the way this session was asked to** | **worst seed > 60.6 %** — disjoint above B0's whole range |

⛔ Everything is scored through [`../src/baselines/evaluate.py`](../src/baselines/evaluate.py),
eval split, 5 seeds, judged on the worst seed. A policy scored by any other loop
is not comparable to the table above.

---

## §0 Why this is not a ninth intervention

📏 **The frozen axis.** [`docs/inherited/BLOCK_G.md`](../docs/inherited/BLOCK_G.md)
built three cadences holding *"gradient density constant at 488 optimizer steps
per M env-steps"*, and recorded — without following it up — that this forces
**the minibatch to 40,960 rows in all three**. It also states: *"⛔ Not swept,
deliberately: the learning rate (fixed at 3e-4)."* So at the `deep` cadence a
12 M-step run is

```
12e6 / (4096 * 64)               =     46 PPO updates
46 * 4 epochs * 32 mini-batches  =  5,888 Adam steps, total
```

on a **137 k-parameter** actor. 📏 And `runs/val-gnn-deep-s*/log.jsonl` records
`approx_kl` at **0.002 – 0.004** for entire runs, against PPO's usual 0.01 – 0.02:
total policy movement is ≈ 46 × 0.003 ≈ **0.14 nats of KL, end to end.**

📏 **And three quarters of what remains is discarded.** `grad_kept` was
instrumented for the joint-clip question `BLOCK_G` lists as open and is **NaN in
every log in `runs/`** — it has never been read. First reading, on a 120 k-step
MPS smoke run at the shipped clip: **0.20 – 0.26**, with `grad_norm_actor`
1.8 – 2.4 against `grad_norm_clip = 0.5`.

☠️ **Every number in `results/`** — the 81-run sweep, the RQ2 ladder, Gate A,
Φ v2, all eight interventions, the GNN row of `credit_assignment.md` — **was
measured under that budget.** It does not make any of them wrong. It makes them
all share one uncontrolled variable, which is why Gate D runs first and why its
NULL branch is a real and interesting outcome rather than a formality.

🔒 **What is NOT re-opened.** `credit_assignment.md`'s structural half is exact
and is untouched by anything here: `mission`, `idle`, `battery_variance` and
`shaping` go through `team(x)`, are identical across drones **by construction**,
and cancel from `Var_i(A)` exactly. No amount of optimisation changes that. It is
the reason Gate E changes the **return** rather than adding a ninth knob.

---

## Gate D — is the policy optimisation-limited?

### Conditions

🔒 **Control** — the shipped defaults, unchanged, so this is a paired comparison
against a re-run rather than against a number from a different code state:

```bash
uv run python scripts/train.py --arch deepsets --cadence deep --timesteps 12000000 \
    --device cuda --seeds 0 1 2 3 4 --tag gateD-control
```

🔒 **Treatment** — the optimisation budget only. ⛔ No reward change, no
observation change, no architecture change, so a difference is attributable:

```bash
uv run python scripts/train.py --arch deepsets --cadence deep --timesteps 12000000 \
    --device cuda --seeds 0 1 2 3 4 --tag gateD-budget \
    --mini-batch-size 4096 --target-kl 0.015 --grad-norm-clip-critic 1.0 \
    --orthogonal-init --min-log-std -1.6
```

📏 `--mini-batch-size 4096` takes the run from 5,888 to ~58,900 Adam steps at
**essentially unchanged FLOPs** — the same rows are visited the same number of
times per epoch; only kernel-launch overhead grows.

### 🔒 Validity precondition, checked before the outcome is read

⛔ If median `approx_kl` in the treatment does **not** reach **≥ 0.008**, the
treatment did not do the thing it claims and the arm is **VOID, not null** — the
LR controller failed and the run is rerun, not interpreted. Declared here so a
non-moving policy cannot be reported as evidence that movement does not help.

### 🔒 The decision rule

Δ is treatment − control on median `mission_capable`, eval split, 5 seeds.
⚠️ The branches partition the real line. *(Gate A and `trainer_validation.md`
each recorded a rule that failed to; this one is written not to.)*

| branch | rule | what it means |
|---|---|---|
| ✅ **PROMOTE** | median Δ ≥ **+5 pp** and worst-seed Δ ≥ **0** | the budget was binding. It becomes the control for Gate E, and every prior null is re-labelled *measured under a 10×-smaller optimisation budget* |
| ⚠️ **PARTIAL** | median Δ ≥ +5 pp and worst-seed Δ < 0 | helps on average, destabilises. Report both; do not ship without a stability fix, and judge Gate E on the control that has the better worst seed |
| ⛔ **NULL** | \|median Δ\| < 5 pp | the budget was **not** binding. ⭐ This is a strong result in the opposite direction: it removes the confound from all eight prior nulls and makes `credit_assignment.md` stronger, not weaker |
| ☠️ **REGRESSION** | median Δ ≤ −5 pp | more optimisation makes it worse, which points at the objective rather than the optimiser. Report as such; do **not** rescue it by re-tuning |

### 🔒 Reported whatever the branch

`approx_kl`, `grad_kept`, `grad_norm_actor`, `grad_norm_critic`,
`explained_variance`, `log_std`, `at_boundary`, `at_speed_cap` — median and worst
seed. ⚠️ `grad_kept` has never been read in this project; its first five-seed
value is a result on its own regardless of the branch.

### Result

⛔ **Not yet run.**

---

## Gate E — does per-drone credit produce roles?

### The instrument

`RewardWeights.w_difference` — the difference reward `D_i = G(z) − G(z_{−i})`
(Wolpert & Tumer 2002; Agogino & Tumer 2008), the mission term recomputed with
drone `i` deleted from the routing DP and the observation OR.
[`../src/env/core.py::_capable_without`](../src/env/core.py) computes it exactly,
in one `(B·N, R, R)` routing call, `R = 6`.

🔒 **Why it is not a ninth knob.** `G(z_{−i})` does not depend on `a_i` at all, so
`∂D_i/∂a_i = ∂G/∂a_i` exactly: `D_i` is **factored**, every agent's best response
to fixed others is unchanged, and the equilibrium of the team objective cannot
move. ⚠️ Factoredness is exact *per step*; over a trajectory `G(z_{−i})(s_t)`
depends on a joint state that `i`'s past actions influenced, so the discounted sum
is only approximately factored. That is the standard difference-reward caveat and
it is stated rather than hidden.

🔒 **Why it is not `w_relay` again.** `w_relay` is PBRS, so its return-to-go
**telescopes** to `Φ(s_T) − Φ(s_0)` and the per-drone part largely cancels over a
trajectory — which is why it only ever reached ~2.9 %. `D_i` is a genuine
per-step reward and does not telescope.

### 📏 Pre-run measurement — it reaches the gradient

`scripts/measure_credit.py`, CPU, eval split, stage 4, F4/J1, 48 envs × 200 steps,
1 seed. ⚠️ Not a 5-seed finding and not labelled as one; it is a **sizing**
measurement, made so the weight is not chosen blind.

| `w_difference` | 0.0 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| **B0** — differentiable share | **0.09 %** | 5.24 % | 15.53 % | **35.64 %** |
| **GNN checkpoint** — same | **0.05 %** | — | — | **7.98 %** |
| the `difference` term's own between-drone share | — | 81.2 % | 81.2 % | 78.7 % |

🔍 **Two things this says, and the second is a warning.** The term is ~80 %
between-drone, which nothing else in the reward is. But the share it produces is
**policy-dependent**: 35.6 % on B0, whose observer role is persistent
(tenure 295), against 7.98 % on the learned policy, whose is not (tenure 47).
⚠️ **The signal grows as roles emerge**, so it is partly circular — which is
exactly why Gate E's NULL branch is written to be informative.

### Conditions

🔒 Two stages, the convention `BLOCK_G` established, and the eval split is touched
once. **Stage A** selects `w_difference ∈ {0.5, 1.0, 2.0}` on the **train** split
at 3 seeds, on median `mission_capable`, ties to the smaller IQR.
**Stage B** re-runs the winner at 5 seeds on eval.

Control is **Gate D's promoted configuration** with `--w-difference 0`, same
seeds, same everything else.

```bash
# stage A -- train split, 3 seeds, one axis
uv run python scripts/sweep.py --axis w_difference=0.5,1.0,2.0 --seeds 0 1 2 \
    --train-arg tag=gateE
```

### 🔒 Validity precondition

⛔ `scripts/measure_credit.py --w-difference <winner>` run **on the trained
policy** must report a differentiable share **> 20 %** (the refute band declared
in that script in 2026-09-02). Below it, the term did not reach the gradient at
that weight and the arm is **VOID**, not null.

### 🔒 The decision rule

Primary readout is **`observer_tenure`**, not `mission_capable` — the deficit is
47 against B0's 294.7, and capability could move for unrelated reasons.
**95** is the bar G8's recurrence gate already declared and is reused unchanged.
Δ capable is treatment − control, median, eval, 5 seeds.

| branch | `observer_tenure` | Δ `mission_capable` | reading |
|---|---|---|---|
| ✅ **CONFIRMED** | ≥ 95 | ≥ +3 pp, worst-seed Δ ≥ 0 | per-drone credit produces roles **and** they pay. `credit_assignment.md`'s redirect is vindicated constructively |
| ⚠️ **MECHANISM, NO OUTCOME** | ≥ 95 | < +3 pp | roles emerged and did not pay. ⛔ That contradicts [`b0_ablation.md`](b0_ablation.md)'s pricing of ranked roles at +3.4 pp, and one of the two is wrong |
| ⚠️ **OUTCOME, NO MECHANISM** | < 95 | ≥ +3 pp | it helped by another route. ⛔ Find the route before crediting the advantage; a difference reward also simply *shapes* toward being useful |
| ⛔ **NULL** | < 95 | < +3 pp | ⭐ **the strongest closure available.** Per-drone credit demonstrably reached the gradient (> 20 % share, validity precondition) and changed nothing. That is stronger than `credit_assignment.md`, which showed only that the signal was *absent* |

### Secondary, reported not gated

`role_entropy` (B0 0.062, learned 0.55), `observer_range_m` (B0 90.0 m, learned
218.9 m), `hop_mean` (B0 2.13, learned 1.20), `observed` (B0 92.8 %, learned
66.5 %). 🔍 `observed` is the one that matters mechanically: `capable | observed`
is already **0.620 for the GNN against B0's 0.617**, so the entire 15 pp lives in
`observed` and a treatment that raises capability *without* raising `observed`
has done something this file did not predict.

### Result

⛔ **Not yet run.**

---

## Gate F — is the observation lying to the policy?

⚠️ **Third in order and lowest prior.** [`obs_mask_gate.md`](obs_mask_gate.md)
already masked nine features for a null, so the base rate for observation
surgery in this environment is low. It is built and gated anyway because the two
findings below are of a different kind from a masked feature — one is a
*misleading* input, the other is a *degenerate training stage*.

### 📏 F1 — the cue is stale, and it is never marked as such

`self.cue` is `hvt_pos` at `t = 0` plus noise and is **never refreshed**. Over
the 1,792 training routes:

| t | 50 | 150 | 300 | 599 |
|---|---|---|---|---|
| median \|cue − hvt\| | 116 m | **322 m** | 632 m | **984 m** |
| median bearing error from the MCV | 8.6° | 11.6° | 15.3° | **17.8°** |

🔍 Against a **127 m** along-street sightline median, the cue is useless as a
*position* within ~60 steps — while remaining a full-magnitude 3-vector in ego
dims 4–6 that points a kilometre away at `t = 599`. Its **bearing** survives,
which is what `BLOCK_D.md` meant by *"it decays in range rather than in
direction"*. `EnvConfig.cue_mode` reports `position` / `bearing` / `off` at
unchanged width.

### 📏 F2 — `STAGES[0]` has a closed-form solution

`CurriculumStage(150, speed_scale=0.00, …, cue_sigma_m=0.0)`: the target **does
not move** and the cue points at it **exactly**. Stage 1 is *"fly to the vector
in ego dims 4–6 and hover"* — solvable by a linear policy on `cue_rel`. It is
15 % of training, plus a 20 % mix for the rest of the run.

⚠️ **Consistent with the curves, not established by them.** All five
`runs/val-gnn-deep-s*` seeds peak at progress **0.20–0.33** and end lower
(peaks 0.453 / 0.526 / 0.560 / 0.491 / 0.628; finals 0.156 / 0.454 / 0.398 /
0.335 / 0.455). ⛔ Those peaks are measured on *easier* curriculum stages and are
**not** comparable to a stage-4 eval number. The shape is 5/5; the attribution is
a hypothesis. `--curriculum-boundaries` and `--curriculum-mix` are the knobs.
`BLOCK_G` already lists the schedule as *provisional and never measured*.

### 📏 F3 — two ego features carry no role information at all

`e2e_capacity` (22) and `steps_since_link` (23) are `(B,)` scalars `.expand()`ed
across the drone axis, so they are identical across drones **by construction**.
📏 Measured between-drone standard deviation, 250 steps at stage 4 / F4:
**0.00000** under B0 *and* under a random policy, against 0.42 (`noise_dbm`),
0.42 (`on_path`), 0.48 (`clr_hvt`). ⚠️ And `e2e_capacity` has the **largest total**
standard deviation in the ego block (1.49) — the most salient feature the policy
sees cannot help it decide *which drone* should act.
`--mask-broadcast-obs` zeroes both.

### 🔒 The decision rule

One axis at a time against Gate E's promoted configuration, 3 seeds on the train
split, then the winner at 5 seeds on eval. Δ is median `mission_capable`.

| branch | rule |
|---|---|
| ✅ **PROMOTE** | one arm gives median Δ ≥ **+3 pp** with worst-seed Δ ≥ 0. ⛔ Only that arm ships; the others are reported as measured nulls |
| ⛔ **NULL** | every arm within ±3 pp. The observation was not the problem, and `obs_mask_gate.md`'s null generalises from *jammer-movable* features to *stale and broadcast* ones |
| ☠️ **REGRESSION** | any arm ≤ −3 pp — report it. ⚠️ `cue_mode=off` regressing would **confirm** the cue is load-bearing for acquisition, which is a result about the task, not a failed intervention |

⛔ **B0 must be scored under `cue_mode="position"` and `mask_broadcast_obs=False`.**
B0's acquisition fan and initial belief are both `cue_rel * POS_SCALE_M`, and its
link repair hill-climbs on the edge capacity. A B0 number measured under any
other setting is not the baseline.

### Result

⛔ **Not yet run.**

---

## The runbook

📏 At the measured CUDA throughput a 12 M-step run is ~2.6 min, and
`--mini-batch-size 4096` adds ~1–2 min of optimizer time. Call it **5 min a run**,
so the whole programme below is **~4 GPU-hours**.

```bash
# Gate D -- 2 conditions x 5 seeds
uv run python scripts/train.py --arch deepsets --cadence deep --device cuda     --seeds 0 1 2 3 4 --tag gateD-control
uv run python scripts/train.py --arch deepsets --cadence deep --device cuda     --seeds 0 1 2 3 4 --tag gateD-budget     --mini-batch-size 4096 --target-kl 0.015 --grad-norm-clip-critic 1.0     --orthogonal-init --min-log-std -1.6

uv run python scripts/eval_policy.py runs/gateD-control-s*/checkpoint.pt     --group gateD/control --device cuda --out results/capability.jsonl
uv run python scripts/eval_policy.py runs/gateD-budget-s*/checkpoint.pt     --group gateD/budget --device cuda --out results/capability.jsonl

# Gate E -- stage A on the TRAIN split, then the winner at 5 seeds on eval
uv run python scripts/sweep.py --axis w_difference=0.0,0.5,1.0,2.0 --seeds 0 1 2

# Gate F -- one axis at a time
uv run python scripts/sweep.py --axis cue_mode=position,bearing,off --seeds 0 1 2
uv run python scripts/sweep.py --axis curriculum_mix=0.2,0.05 --seeds 0 1 2
```

⚠️ **Read `n` before quoting anything** — [`README.md`](README.md) records an
interrupted-and-resumed sweep that appended rows unconditionally and reported
`n = 9` where 5 were asked for.

---

## Not in any gate, and why

| | why it is held back |
|---|---|
| **agent index / role embedding** | ⛔ Excluded by decision, 2026-09-04: roles must **emerge**. B0 is granted roles-from-index as a documented advantage and the learned arm is not; that asymmetry is kept deliberately, and it is what makes Gate E a test of *credit* rather than of labelling |
| **DAgger from B0** | Held as the fallback if D, E and F all fail. [`bc_init.py`](../scripts/bc_init.py) exists and has never been reported; [`memory_horizon.md`](memory_horizon.md) predicts DAgger fixes the 9.4 % clone, since the collapse is covariate shift rather than missing memory. ⚠️ A teacher-initialised policy is a **probe**, not a like-for-like RQ2 or Gate B arm |
| **recurrence** | ⛔ [`memory_horizon.md`](memory_horizon.md) closed *target* memory with a hard oracle bound (perfect target state is worth **−0.4 pp**). It explicitly leaves **role-commitment** memory open — but that is what Gate E attacks, far more cheaply and with a lower bug density |
| **wider / deeper networks** | 📏 RQ2 measured architecture at ±1 pp across three rungs and MLP → DeepSets → GNN at 35.6 / 42.5 / 45.1. Capacity is not the suspect, and the actor is 137 k parameters against ~5,900 gradient steps — the budget binds long before the width does |
| **velocity action space** | ⛔ Gate A. ⚠️ Its own kill branch named exploration as the next suspect and `entropy_loss_scale` has been 0.0 throughout; `--entropy` and `--min-log-std` are now sweepable, so that follow-up is reachable — but it is Gate A's, not this file's |
