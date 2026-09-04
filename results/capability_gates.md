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

🔒 **A 2 × 4 factorial: `{shipped budget, new budget} × λ ∈ {0.95, 0.98, 0.99,
0.995}`**, 3 seeds on the train split, then the winner at 5 fresh seeds on eval.
⛔ The exact commands are in **The runbook** below and are not duplicated here —
a second copy is a second thing to get out of step.

⚠️ **Amended 2026-09-04, before any run.** The first draft of this gate was a
single treatment arm that bundled `--gae-lambda 0.99` with the four optimiser
knobs. It could not have separated λ from the step count, and at ~5 min a run it
does not have to. 🔒 Recorded rather than silently replaced: nothing had been run
against the first version, so this is an amendment to a declaration, not a rule
changed after seeing a result.

* 🔒 **Control** — the shipped defaults at λ = 0.95, i.e. the `(shipped, 0.95)`
  cell. A **re-run**, not a number quoted from a different code state.
* 🔒 **Budget arm** — `--mini-batch-size 4096 --target-kl 0.015
  --grad-norm-clip-critic 1.0 --orthogonal-init --min-log-std -1.6`. ⛔ No reward
  change and no observation change, so a difference is attributable to *the
  optimiser*.

📏 `--mini-batch-size 4096` takes a run from 5,888 to ~58,900 Adam steps at
**essentially unchanged FLOPs** — the same rows are visited the same number of
times per epoch; only kernel-launch overhead grows.

⚠️ **The budget arm bundles five knobs, and that is deliberate.** It is a
**screening** arm: the question is whether the budget binds at all, and five
one-at-a-time arms cost 5× to answer a yes/no. 🔒 If it promotes, the ablation is
owed before anything is claimed about *which* knob mattered, and
`--orthogonal-init` is the one to drop first — it is the only one that changes
the network rather than the optimiser.

### ⭐ 📏 Why `--gae-lambda` is in the treatment and not in a footnote

At `γ = 0.997, λ = 0.95` the advantage weights rewards by `(γλ)^l = 0.94715^l`:

| λ | `γλ` | half-life | effective horizon `1/(1−γλ)` | in seconds at `dt = 0.4` |
|---|---|---|---|---|
| **0.95** (shipped) | 0.94715 | 12.8 steps | **18.9 steps** | **7.6 s** |
| 0.98 | 0.97706 | 29.9 | 43.6 | 17.4 s |
| **0.99** | 0.98703 | 53.1 | **77.1** | 30.8 s |
| 0.995 | 0.99201 | 86.5 | 125.2 | 50.1 s |

☠️ **B0's `observer_tenure` is 294.7 steps — 118 seconds. The advantage sees 6 %
of the behaviour it is supposed to credit.** Committing to the observer role pays
off over hundreds of steps; at λ = 0.95 essentially none of that reaches the
gradient. ⛔ `scripts/train.py` records that λ has **never been swept in this
project's history**.

🔍 And [`credit_assignment.md`](credit_assignment.md) names the same filter from
the other side: *"GAE accumulates the team component coherently over ~19 effective
steps (λ = 0.95) while per-drone terms largely cancel."* So λ gates whether Gate
E's `D_i` reaches the gradient at all — which is why it is set **here**, in the
control Gate E is measured against, rather than varied simultaneously with it.

⚠️ **λ = 0.99 is a choice, not a measurement.** If Gate D lands NULL or
REGRESSION, `--gae-lambda 0.95,0.98,0.99,0.995` is the first follow-up axis, one
variable at a time. Higher λ trades bias for variance, and the variance is
affordable here: the advantage is computed once per rollout over all 1.31 M rows.

### 🔒 Validity precondition, checked before the outcome is read

⛔ If median `approx_kl` in the treatment does **not** reach **≥ 0.008**, the
treatment did not do the thing it claims and the arm is **VOID, not null** — the
LR controller failed and the run is rerun, not interpreted. Declared here so a
non-moving policy cannot be reported as evidence that movement does not help.

### 🔒 The decision rule

🔒 **Δ is the best cell minus the `(shipped budget, λ = 0.95)` control**, on
median `mission_capable`, eval split, 5 fresh seeds — the confirmation run, not
the search score. ⚠️ The search score is biased upward by selection over 8 cells;
`sweep.py` prints both side by side and 🔒 **if they disagree, the disagreement is
the finding.** That is exactly the `45.1 %` cell `BLOCK_G` records as the winner's
curse.

⚠️ The branches partition the real line. *(Gate A and `trainer_validation.md`
each recorded a rule that failed to; this one is written not to.)*

📏 **Reported alongside, and it is the interesting half**: the λ main effect
within each budget, and whether they interact. The claim under test is that they
are **complementary** — λ decides what signal exists in the advantage, the budget
decides whether the policy can move on it — so *"λ helps only under the new
budget"* is a prediction this design can confirm or refute.

| branch | rule | what it means |
|---|---|---|
| ✅ **PROMOTE** | median Δ ≥ **+5 pp** and worst-seed Δ ≥ **0** | the budget was binding. It becomes the control for Gate E, and every prior null is re-labelled *measured under a 10×-smaller optimisation budget* |
| ⚠️ **PARTIAL** | median Δ ≥ +5 pp and worst-seed Δ < 0 | helps on average, destabilises. Report both; do not ship without a stability fix, and judge Gate E on the control that has the better worst seed |
| ⛔ **NULL** | \|median Δ\| < 5 pp | the budget was **not** binding. ⭐ This is a strong result in the opposite direction: it removes the confound from all eight prior nulls and makes `credit_assignment.md` stronger, not weaker |
| ☠️ **REGRESSION** | median Δ ≤ −5 pp | more optimisation makes it worse, which points at the objective rather than the optimiser. Report as such; do **not** rescue it by re-tuning |

### 🔒 Reported whatever the branch

`approx_kl`, `grad_kept`, `grad_norm_actor`, `grad_norm_critic`,
`explained_variance`, `log_std`, `lr_actor`, `at_boundary`, `at_speed_cap` —
median and worst seed. ⚠️ `explained_variance` is expected to **fall** in any arm
that also turns on `w_difference`, and that is not a regression: the critic sees
one global state per env while `G_i` now genuinely differs across drones, so the
between-drone spread is irreducible error by construction. `ppo.py` says so and
`return_spread_between_drones` measures it. ⚠️ `grad_kept` has never been read in this project; its first five-seed
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

⛔ Commands in **The runbook**. 🔒 Gate D's winning λ *and* its winning budget
are forwarded as `--train-arg`, or the comparison is against a policy that cannot
train — and `--train-arg tag=…` is now **refused** by `sweep.py`, because the
draft that used it would have destroyed the sweep.

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

⚠️ **Two arms with very different priors, and they must not be pooled.**

* ⭐ **F1 + F2 — the cue and the curriculum.** 📏 Now the best-motivated
  intervention in this file: a cue-following policy scores **94 % of B0 at stage
  1** and **6.1 % at stage 4**, and the first 15 % of training is entirely stage
  1. This is *shortcut learning*, not feature clutter.
* ⛔ **F3 — the broadcast features.** Low prior:
  [`obs_mask_gate.md`](obs_mask_gate.md) already masked nine features for a null,
  so the base rate for observation surgery here is poor.

🔒 Run them as separate axes. Pooling them would let F3's expected null bury
F1/F2, or let F1/F2 launder F3.

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

### ⭐ 📏 F2 — `STAGES[0]` is degenerate, and it is now measured

`CurriculumStage(150, speed_scale=0.00, …, cue_sigma_m=0.0)`: the target **does
not move** and the cue points at it **exactly**.

📏 **Scored directly.** A policy that does nothing but servo every drone toward
`cue_rel` — no sensing, no roles, no neighbour awareness, no chain reasoning,
using B0's own velocity law on one input — against B0, F4/J1, 64 envs, one full
episode per stage:

| stage | cue-follower `capable` | B0 `capable` | ratio |
|---|---|---|---|
| **1** | **81.6 %** | 87.0 % | **0.94x** |
| 2 | 40.0 % | 90.4 % | 0.44x |
| 3 | 10.9 % | 69.8 % | 0.16x |
| **4** | **6.1 %** | 60.0 % | **0.10x** |

☠️ **A one-line policy scores 94 % of the heuristic at stage 1 — and 6.1 % at
stage 4, which is BELOW random's 10.7 %.** 🔍 At stage 1 the cue-follower's
`capable` equals its `observed` exactly (81.6 / 81.6): all five drones pile onto
one point above a stationary target and the chain still closes, because the
MCV→HVT separation at `t = 0` has a median of only 404 m and one hop covers it.

📏 **The exposure.** Integrating `CurriculumSchedule.weights()` over a run, stage
1 is **24.2 % of episodes** but — because its episodes are 150 steps against
stage 4's 600 — only **9.2 % of env-steps**. ⚠️ That is *less* than an earlier
draft of this file implied and is recorded as the correction it is. But the
**first 15 % of training is 100 % stage 1**, which is where the basin is chosen.

⚠️ **What this does NOT show.** It shows the stage is solvable degenerately. It
does **not** show the learned policy is trapped there — that is the inference, and
it is what the run tests. Corroborating but not decisive: all five
`runs/val-gnn-deep-s*` seeds peak at progress **0.20–0.33** and end lower
(peaks 0.453 / 0.526 / 0.560 / 0.491 / 0.628; finals 0.156 / 0.454 / 0.398 /
0.335 / 0.455). ⛔ Those peaks are measured on *easier* stages and are **not**
comparable to a stage-4 eval number.

🔍 **Two independent routes to the same fix, and they should be run separately.**
`--curriculum-boundaries 0.05 0.30 0.55` shortens the degenerate stage;
`--cue-mode bearing` removes the shortcut's key input *structurally* — a bearing
cannot be servoed to a point, so "fly here and hover" stops being expressible,
while acquisition (which needs only the bearing, and which B0's fan uses) is
preserved. `BLOCK_G` already lists the schedule as *provisional and never
measured*.

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
`--mini-batch-size 4096` adds ~1–2 min of optimizer time. Call it **5 min a run**.
`scripts/sweep.py` is the tool: it searches on the **train** split, ranks on the
**worst** seed, re-runs the winner at **fresh** seeds on eval, and is resumable.

### ☠️ Two traps, both found by dry-running these commands

⛔ **`--train-arg tag=…` is refused, and a first draft of this file used it.**
`build_train_cmd` already passes `--tag`; a second one wins silently, every cell
trains into one directory, and `train_one`'s resume check still looks under the
sweep's own per-cell tag. `OWNED_FLAGS` now rejects it. Use `--run-root`.

⛔ **Two sweeps over the same axis need different `--out` AND different
`--run-root`.** The cell key is built from the swept values alone, so
`gae_lambda=0.95` under the shipped budget and under the new budget were the same
key: the second sweep skipped every cell as *"already recorded"*, and had it run
it would have scored the first sweep's checkpoints. `context_suffix()` now hashes
the `--train-arg` set into the key. 🔒 Empty when there are none, so every row
already in `results/sweep_summary.jsonl` still resumes.

### Gate D — λ × budget, as a 2 × 4

⚠️ **Amended 2026-09-04, before any run.** The first draft bundled
`--gae-lambda 0.99` into a five-knob screening arm, which could not separate λ
from the step count. Compute is ~5 min a run, so it does not have to.

🔍 **Why a factorial and not one-at-a-time.** The claim is that the two are
**complementary**: λ decides what signal exists in the advantage, the budget
decides whether the policy can move on it. That is an interaction, and a
one-at-a-time sweep cannot see it. Two axes is inside `sweep.py`'s own *"a grid
is not a search strategy for more than ~3 axes"*.

```bash
# λ against the SHIPPED optimisation budget
uv run python scripts/sweep.py --axis gae_lambda=0.95,0.98,0.99,0.995 \
    --seeds 0 1 2 --device cuda \
    --run-root runs/gateD-shipped --out results/gateD_shipped.jsonl

# λ against the NEW budget. ⛔ Different --run-root and --out, per the trap above.
uv run python scripts/sweep.py --axis gae_lambda=0.95,0.98,0.99,0.995 \
    --seeds 0 1 2 --device cuda \
    --train-arg mini-batch-size=4096 --train-arg target-kl=0.015 \
    --train-arg grad-norm-clip-critic=1.0 --train-arg orthogonal-init \
    --train-arg min-log-std=-1.6 \
    --run-root runs/gateD-budget --out results/gateD_budget.jsonl
```

📏 24 runs, ~2 GPU-hours. ✅ `--train-arg min-log-std=-1.6` is verified to survive
argparse's negative-number handling into the `nargs="+"` flag.

⚠️ **The budget arm still bundles five knobs.** That is a **screening** arm on
purpose: the question is whether the budget binds at all. 🔒 If it promotes, the
ablation is owed before anything is claimed about *which* knob mattered, and
`--orthogonal-init` is the one to drop first — it is the only one that changes
the network rather than the optimiser.

### Gate E — the difference reward

🔒 Held at Gate D's winning λ **and** its winning budget, forwarded as
`--train-arg`, or the comparison is against a policy that cannot train.

```bash
uv run python scripts/sweep.py --axis w_difference=0.0,0.5,1.0,2.0 \
    --seeds 0 1 2 --device cuda \
    --train-arg gae-lambda=<gateD winner> \
    --train-arg mini-batch-size=4096 --train-arg target-kl=0.015 \
    --train-arg grad-norm-clip-critic=1.0 --train-arg orthogonal-init \
    --train-arg min-log-std=-1.6 \
    --run-root runs/gateE --out results/gateE.jsonl

# the ablation: does the RELAY half of D_i's credit matter?
uv run python scripts/sweep.py --axis difference_on=capable,observed \
    --seeds 0 1 2 --device cuda --train-arg w-difference=<gateE winner> \
    --run-root runs/gateE-target --out results/gateE_target.jsonl
```

### Gate F — two axes, never pooled

```bash
# F1/F2 -- the cue and the curriculum. ⭐ The better-motivated arm.
uv run python scripts/sweep.py --axis cue_mode=position,bearing,off \
    --seeds 0 1 2 --device cuda --run-root runs/gateF-cue --out results/gateF_cue.jsonl
uv run python scripts/sweep.py --axis curriculum_mix=0.2,0.05 \
    --seeds 0 1 2 --device cuda --run-root runs/gateF-mix --out results/gateF_mix.jsonl
# ⚠️ --curriculum-boundaries takes THREE values and is not an --axis (axes pass
# one value); run it as two explicit conditions via --train-arg.

# F3 -- the broadcast features. ⛔ Low prior; a bare switch takes no value.
uv run python scripts/sweep.py --axis gae_lambda=<winner> --seeds 0 1 2 \
    --device cuda --train-arg mask-broadcast-obs \
    --run-root runs/gateF-bcast --out results/gateF_bcast.jsonl
```

⚠️ **Read `n` before quoting anything** — [`README.md`](README.md) records an
interrupted-and-resumed sweep that appended rows unconditionally and reported
`n = 9` where 5 were asked for. And ⛔ a cell whose training failed is written
with `"status": "failed"` and excluded from ranking; check for those first.

---

## Not in any gate, and why

| | why it is held back |
|---|---|
| **agent index / role embedding** | ⛔ Excluded by decision, 2026-09-04: roles must **emerge**. B0 is granted roles-from-index as a documented advantage and the learned arm is not; that asymmetry is kept deliberately, and it is what makes Gate E a test of *credit* rather than of labelling |
| **DAgger from B0** | Held as the fallback if D, E and F all fail. [`bc_init.py`](../scripts/bc_init.py) exists and has never been reported; [`memory_horizon.md`](memory_horizon.md) predicts DAgger fixes the 9.4 % clone, since the collapse is covariate shift rather than missing memory. ⚠️ A teacher-initialised policy is a **probe**, not a like-for-like RQ2 or Gate B arm |
| **recurrence** | ⛔ [`memory_horizon.md`](memory_horizon.md) closed *target* memory with a hard oracle bound (perfect target state is worth **−0.4 pp**). It explicitly leaves **role-commitment** memory open — but that is what Gate E attacks, far more cheaply and with a lower bug density |
| **wider / deeper networks** | 📏 RQ2 measured architecture at ±1 pp across three rungs and MLP → DeepSets → GNN at 35.6 / 42.5 / 45.1. Capacity is not the suspect, and the actor is 137 k parameters against ~5,900 gradient steps — the budget binds long before the width does |
| **velocity action space** | ⛔ Gate A. ⚠️ Its own kill branch named exploration as the next suspect and `entropy_loss_scale` has been 0.0 throughout; `--entropy` and `--min-log-std` are now sweepable, so that follow-up is reachable — but it is Gate A's, not this file's |
