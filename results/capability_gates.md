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

⚠️ **`grad_kept` — a claim made here on 2026-09-04 and CORRECTED 2026-09-06.**
It was instrumented for the joint-clip question `BLOCK_G` lists as open and was
**NaN in every log in `runs/`**. This section originally read *"three quarters of
what remains is discarded"*, from a **120 k-step MPS smoke run at 128 envs**:
`grad_kept` 0.20–0.26, `grad_norm_actor` 1.8–2.4.

📏 **That does not hold on a real run.** `runs/gateD-shipped/sw-gae_lambda0p95-s0`,
CUDA, 4096 envs, 12 M steps: `grad_kept` **0.54 – 0.91** and `grad_norm_actor`
**0.032 – 0.060** — more than 8x below the 0.5 clip. The actor's own gradient
never reaches the clip; the throttling is the *critic* dominating the joint norm,
and it costs a factor of ~1.1–1.8, not ~4. ⛔ A toy-config number was quoted as
though it described the condition under study. The `approx_kl` figure is
unaffected and independently reproduced at **0.0021 – 0.0028** on the same run.

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

### Result — ⚠️ **PARTIAL, 2026-09-04. Arm 1 is clean; arm 2 collapsed.**

#### 📏 Arm 1 — λ against the shipped budget. Valid, and it refutes the λ argument.

Train split, 3 seeds, ranked on the worst seed. [`gateD_shipped.jsonl`](gateD_shipped.jsonl).

| λ | worst | median | per seed |
|---|---|---|---|
| **0.95** (shipped) | **44.28 %** | 45.18 % | 44.3 · 48.5 · 45.2 |
| 0.98 | 41.17 % | 43.82 % | 45.0 · 41.2 · 43.8 |
| 0.99 | 41.39 % | 42.50 % | 41.4 · 44.2 · 42.5 |
| 0.995 | 36.13 % | 39.94 % | 39.9 · 36.1 · 41.8 |

⛔ **Raising λ does not help, and 0.995 clearly hurts.** ⚠️ The honest statement is
*"λ is a null between 0.95 and 0.99, and harmful at 0.995"* — 0.95's range
[44.3, 48.5] overlaps 0.98's [41.2, 45.0] and 0.99's [41.4, 44.2]. Only 0.995
separates on the worst seed.

☠️ **The argument for raising it was mine and it was wrong.** The horizon
arithmetic is right — at λ = 0.95 the advantage sees 18.9 steps against B0's
294.7-step observer tenure — but the conclusion did not follow. Higher λ buys
horizon by *reducing bias and raising variance*, and at ~5,900 gradient steps
there is no budget to average that variance away. 🔍 The prediction that λ would
pay **only under the larger budget** is exactly the interaction the 2 × 4 was
built to test, and it is still open — arm 2 has to be repaired before it can
answer.

#### 📏 Arm 2 — REGRESSION, as declared. And the diagnosis I gave was wrong.

| λ | worst | median | per seed |
|---|---|---|---|
| 0.95 | 13.08 % | 13.10 % | 16.7 · 13.1 · 13.1 |
| 0.98 | 9.22 % | 12.80 % | 15.7 · 9.2 · 12.8 |
| 0.99 | 12.19 % | 15.09 % | 15.1 · 12.2 · 16.1 |
| 0.995 | 4.80 % | 9.15 % | 11.0 · 9.1 · 4.8 |

🔒 **Δ = −32 pp against the control. That is the REGRESSION branch**, and its rule
reads: *"more optimisation makes it worse… Report as such; do **not** rescue it by
re-tuning."* Recorded as declared.

☠️ **My named prime suspect was wrong and the log refutes it.**
`runs/gateD-budget/sw-gae_lambda0p95__7a3bf4-s0`:

| progress | capable | lr_actor | approx_kl | grad_kept | expl_var | at_boundary |
|---|---|---|---|---|---|---|
| 0.044 | 0.371 | 3.75e-04 | 0.0027 | 1.00 | 0.081 | 0.047 |
| 0.175 | 0.502 | 4.27e-03 | 0.0078 | 1.00 | 0.534 | 0.025 |
| **0.306** | **0.536** | 5.13e-03 | 0.0092 | 1.00 | 0.670 | 0.015 |
| 0.437 | 0.399 | 5.13e-03 | 0.0095 | 1.00 | 0.813 | 0.017 |
| 0.699 | 0.287 | 5.13e-03 | 0.0099 | 0.98 | 0.913 | 0.028 |
| 0.961 | 0.240 | 5.13e-03 | 0.0107 | 0.95 | 0.947 | 0.058 |

⛔ **The LR did not run away.** It rose to **5.13e-3** and *plateaued* there — half
of `lr_max` — holding `approx_kl` in **[0.0027, 0.0107]** against a 0.015 target.
The controller's dead band (`target/2` to `2*target`) is what held it. 🔒 So the
amended two-sided precondition **[0.005, 0.05]** is **met**: the arm is a valid
regression, not void. ⚠️ I proposed voiding it on a mechanism the data does not
support, and the amendment stands on its own merits rather than as a way out.

⚠️ **CORRECTED 2026-09-06, once the shipped arm's curve existed.** This section
first read *"it learned better, then forgot"* and *"the budget arm fixes the
boundary pathology"*. 📏 Both are wrong, and both came from comparing against the
`gnn` runs in `runs/val-gnn-deep-s*` rather than against the `deepsets` control
that was sitting in the same sweep:

| progress | curriculum focus | shipped | budget | leads |
|---|---|---|---|---|
| 0.044 | stage 1 | 0.348 | 0.371 | budget |
| 0.175 | stage 2 | 0.507 | 0.502 | tie |
| 0.306 | stage 2 | **0.636** | 0.536 | shipped |
| 0.437 | stage 3 | 0.520 | 0.399 | shipped |
| 0.699 | stage 4 | 0.465 | 0.287 | shipped |
| 0.961 | stage 4 | 0.482 | 0.240 | shipped |

⛔ **The budget arm never learns better.** It leads only at progress 0.044, ties at
0.175, and is behind from 0.306 to the end. And `at_boundary` is 0.059 → 0.067 on
the *shipped* arm against 0.047 → 0.058 on the budget arm — essentially the same,
so nothing was "fixed".

🔍 **And the peak-then-decay shape is largely the CURRICULUM, not forgetting.**
Both arms peak at progress 0.306–0.350, which is exactly where the focus is
**stage 2** — half speed, **no jammer**, exact cue. The metric falls when the task
gets harder. In the window where the task mix is constant (progress ≥ 0.60,
stage-4 focus) the shipped arm reads 0.465 / 0.374 / 0.482 against a final eval of
0.452, and the budget arm 0.287 / 0.147 / 0.240 against 0.131. ⚠️ Three log points
per window cannot establish a trend either way, which is what Gate G is for.

📏 **What is solid**: the budget arm's final policy is, on every structural
metric, **indistinguishable from random**:

| | budget λ=0.95 | shipped λ=0.95 | random |
|---|---|---|---|
| `mission_capable` | 13.1 % | 45.2 % | 10.7 % |
| `observed` | 26.4 % | 64.7 % | 21.9 % |
| `hop_mean` | **0.38** | 1.28 | 0.41 |
| `observer_tenure` | 15.7 | 45.4 | 16.3 |
| `episode_return` | **−147** | +116 | — |

⛔ **Not attributed.** Five knobs moved together; the training curve says the
failure is in *retention*, not in *learning*, but which knob destroys retention is
untested. 🔒 The declared rule forbids re-tuning this arm, and that is honoured:
the follow-up below is a **different question**, declared before its own run.

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

### Result — ⛔ **NULL on both readouts, 2026-09-06.** ⚠️ Verdict pending the validity check.

Search: train split, 3 seeds, ranked on worst. Confirmation: eval split, **fresh**
seeds 100–104. [`gateE.jsonl`](gateE.jsonl).

| `w_difference` | worst | median | per seed |
|---|---|---|---|
| **0.0** (control) | 44.28 % | 45.18 % | 44.3 · 48.5 · 45.2 |
| 0.5 | 44.22 % | 45.44 % | 45.4 · 44.2 · 47.4 |
| 1.0 | 43.93 % | 46.46 % | 43.9 · 49.3 · 46.5 |
| **2.0** | **45.73 %** | **46.88 %** | 45.7 · 52.5 · 46.9 |

✅ **The control reproduces Gate D's `(shipped, λ = 0.95)` cell to the digit** —
44.28 / 45.18 in both. The harness is sound and the two gates are on one scale.

📏 **Δ = +1.70 pp median, +1.45 pp worst, and 3/3 paired seeds improve**
(+1.45, +3.99, +1.69). Small, consistent, and **below the declared +3 pp**.

#### ⛔ The primary readout did not move

| | control | `w = 2.0` | declared bar | B0 |
|---|---|---|---|---|
| **`observer_tenure`** | 45.4 | **46.6** | **≥ 95** | 294.7 |
| `role_entropy` | 0.492 | **0.566** | — | 0.062 |
| `observer_range_m` | 192.8 | 193.9 | — | 90.0 |
| `observed` | 64.7 % | 68.8 % | — | 92.8 % |
| `hop_mean` | 1.277 | 1.337 | — | 2.13 |

☠️ **`role_entropy` got *worse*** — 0.492 → 0.566, further from B0's 0.062. So
whatever the +1.7 pp is, **it is not role differentiation.** `observed` is the
metric that moved most (+4.1 pp), which is the right axis but a twentieth of the
distance to B0.

🔒 Against the declared rule — `observer_tenure` **46.6 < 95** and Δ **+1.70 <
+3 pp** — this is the **NULL** branch.

#### ☠️ VOID, by the declared precondition. 2026-09-06.

📏 `scripts/measure_credit.py` on the trained `w = 2.0` policy, CUDA, eval split,
stage 4, F4/J1, 64 envs, 3 seeds:

| | share |
|---|---|
| seed 0 / 1 / 2 | 14.04 % · 14.25 % · 14.60 % |
| **median** | **14.25 %** |
| the `difference` term's own between-drone share | 52.18 % |
| control class ([`credit_assignment.md`](credit_assignment.md), any policy) | 0.04 – 0.16 % |

🔒 **The declared precondition was `> 20 %`. Measured 14.25 %. The arm is VOID.**
⛔ Recorded as declared. A rule invented after the fact is not a rule, and the
temptation here is precisely to relabel this NULL because the capability numbers
are already in hand.

⚠️ **And the precondition was badly drawn, which is a separate thing to own.** It
borrowed the `> 20 %` figure from `measure_credit.py`'s *refute* band — a
threshold built to answer *"is there ample drone-differentiating signal?"* — and
repurposed it as a **validity** gate without re-deriving it. 14.25 % sits in that
script's own **INCONCLUSIVE** band (5–20 %), and *"the term did not reach the
gradient"* is not an honest description of a **~90–350x** rise over the control
class. ⛔ The rule still binds; the criticism is recorded beside it, not instead
of it.

🔍 **Why the share is 14 % here and 35.6 % on B0, mechanistically.** The
`difference` term is **52.18 %** between-drone on this policy against ~80 % on B0.
`D_i`'s *return-to-go* concentrates on a drone only if that drone stays pivotal;
with `observer_tenure` at **46.6** against B0's **294.7**, the credit is spread
across drones by role churn. ☠️ **The signal is diluted by exactly the churn it
exists to fix** — the circularity this file declared before the run, now measured.

#### 📏 The repaired arm, `w ∈ {3, 4, 6}` — and the axis is a NULL

Declared before it ran, decision rule unchanged. [`gateE2.jsonl`](gateE2.jsonl).
The whole axis, both rounds, search / train split, 3 seeds:

| `w_difference` | 0.0 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 6.0 | **B0** |
|---|---|---|---|---|---|---|---|---|
| **median** | 45.18 | 45.44 | 46.46 | 46.88 | **47.79** | 45.11 | 46.60 | **57.30** |
| worst | 44.28 | 44.22 | 43.93 | 45.73 | 45.86 | 42.73 | 45.07 | 54.80 |
| **`observer_tenure`** | 45.4 | 40.3 | 42.8 | 46.6 | 41.4 | 40.6 | 41.0 | **294.7** |
| **`role_entropy`** | 0.492 | 0.556 | 0.585 | 0.566 | 0.585 | 0.554 | **0.603** | **0.062** |
| `observer_range_m` | 192.8 | 200.1 | 187.1 | 193.9 | 205.3 | 203.1 | **211.7** | **90.0** |
| `observed` | 64.7 | 64.3 | 65.9 | 68.8 | 64.8 | 64.4 | 63.7 | 92.8 |

☠️ **The entire axis is smaller than the noise inside one cell.**

| | |
|---|---|
| median spread over `w = 0 → 6` (a 12x range) | **2.68 pp** |
| typical **within-cell** seed range | **4.23 pp** |

⛔ **And the mechanism moves the wrong way.** At `w = 6.0` — a term worth **six
times** the mission weight — `role_entropy` (0.603) and `observer_range_m`
(211.7 m) are the **worst on the board**. Whatever the term does, it is not
producing roles.

⚠️ **The `+1.70 pp` at `w = 2.0`, with 3/3 paired seeds, was noise.** The
extension shows +2.6 at `w = 3`, **−0.07** at `w = 4`, +1.4 at `w = 6`. 🔒 Three
paired seeds moving together is not a dose–response, and this is the second time
in this project that a monotone-looking 3-point trend did not survive extension.

#### 🔒 What is still owed before this can be called NULL rather than VOID

⛔ The `w = 2.0` arm measured a **14.25 %** differentiable share against a
declared **20 %**, so it is VOID. The extension does not inherit validity — it
has to earn it. 📏 The share scales with the weight (B0: 5.2 / 15.5 / 35.6 % at
`w` = 0.5 / 1 / 2), so `w = 6.0` is the cell most likely to clear the bar:

```bash
uv run python scripts/measure_credit.py \
    --policy runs/gateE2/sw-w_difference6p0__e54208-s0/checkpoint.pt \
    --device cuda --num-envs 64 --seeds 3 --w-difference 6.0
```

| outcome | verdict |
|---|---|
| **≥ 20 %** | ✅ **NULL, earned.** Per-drone credit demonstrably occupied a fifth or more of the differentiable advantage, and role differentiation got **worse**. That is the *"strongest closure available"* branch, and it closes the credit axis on evidence rather than on exhaustion |
| **< 20 %** | ⛔ **The instrument cannot be brought into range in this environment.** Reported as an instrument limitation — the credit axis stays **untested**, not closed, and `D_i` is not the tool that can test it here |

#### ⚠️ Both confirmations landed ~5 pp below their search scores

| cell | search worst / median | confirmation (eval, fresh seeds 100–104) |
|---|---|---|
| `w = 2.0` | 45.73 / 46.88 | **34.76 / 42.51** |
| `w = 3.0` | 45.86 / 47.79 | **37.46 / 42.09** |

🔍 **Consistent across both**, which makes it a property of the train → eval step
plus selection rather than a fluke of one cell. ⛔ Do not quote a search score.
📏 Both land at ~42 % eval median, against B0's **57.3 %**.

#### ⚠️ Search 46.88 / 45.73 → confirmation 42.51 / 34.76

Fresh seeds 100–104 on eval: **34.8 · 38.9 · 42.5 · 43.0 · 43.4**. `sweep.py`'s
own rule is *"if they disagree, the disagreement is the finding"*. ⚠️ Two effects
are confounded in that gap — selection over four cells, and train → eval — and
the eval seed range is **8.6 pp**, which is this project's known seed variance.
⛔ Do not quote 46.88 %.

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

## Gate G — is the FINAL checkpoint the right one to score? Declared 2026-09-06

🔒 **A new question, not a re-run of Gate D.** Gate D's REGRESSION branch forbids
rescuing its treatment by re-tuning, and this does not: it asks nothing about the
budget bundle. It asks whether **every number in `results/` was read at the wrong
point**, and it applies to the shipped configuration exactly as much as to the
budget one.

### ☠️ Why it exists

📏 [`scripts/train.py`](../scripts/train.py) saves **one** checkpoint, after the
last round. Every learned number this project has ever reported is the *final*
state of a run. And every training curve on record peaks partway through and
decays:

| run | peak `capable` (progress) | final |
|---|---|---|
| `val-gnn-deep-s0` | 0.453 (0.197) | 0.156 |
| `val-gnn-deep-s1` | 0.526 (0.328) | 0.454 |
| `val-gnn-deep-s2` | 0.560 (0.328) | 0.398 |
| `val-gnn-deep-s3` | 0.491 (0.197) | 0.335 |
| `val-gnn-deep-s4` | 0.628 (0.328) | 0.455 |
| Gate D budget λ=0.95 s0 | **0.536 (0.306)** | 0.172 |

**6 of 6.** ⛔ No mid-run checkpoint has ever been scored, because none was ever
saved. `--checkpoint-every` now saves one per log line.

### ⚠️ The trap this gate must not fall into

⛔ **A training-log peak is not an eval score.** It is measured on *stochastic*
actions and a *curriculum mix*, so a peak at progress 0.31 sits on stage 2
(speed 0.5, no jammer) while eval is stage 4 / F4 / J1 on the deterministic mean.
The curve cannot answer this; only `evaluate.py` can.

⛔ **And picking the best checkpoint on the eval split is selection on the test
set.** 🔒 So the protocol is fixed here, before the run: **select the checkpoint on
the TRAIN split, report that one checkpoint on eval.** The eval number is then an
honest estimate of a decision made without it. Anything else quietly converts the
one generalisation check this project has left into a validation set.

### Conditions

⚠️ **Amended 2026-09-06, before running, on evidence and to cut cost.** 📏 The
shipped arm's training peak sits at progress 0.350 — exactly the stage-2 focus,
which has **no jammer**, half speed and an exact cue. So the peak is measured on
an easier task and the prediction for the shipped arm is **NULL**. It is still
worth confirming, because it bears on every learned number in `results/`, but it
does not deserve its own runs.

🔒 **So Gate G rides along on Gate E's runs.** `sweep.py --train-arg
checkpoint-every` saves the checkpoints at no training cost; only the *scoring*
is extra, and it happens after Gate E has its answer. ⛔ If Gate E is never run,
Gate G needs its own 3 seeds of the shipped configuration.

### 🔒 The decision rule

`Δ = (best mid-run checkpoint, selected on TRAIN) − (final checkpoint)`, both
scored on **eval**, median over 3 seeds. ⚠️ Branches partition the real line.

| branch | rule | consequence |
|---|---|---|
| ✅ **CONFIRMED** | `Δ_shipped ≥ +3 pp` | ☠️ **Every learned number in `results/` is an under-report.** The reporting protocol changes to train-selected checkpoints, and the RQ2 ladder, Gate A, Φ v2 and the eight nulls each need re-reading |
| ⚠️ **BUDGET-ONLY** | `Δ_shipped < +3 pp` **and** `Δ_budget ≥ +3 pp` | Gate D's arm 2 failed at *retention*, not at learning. ⛔ Still does not un-do Gate D's REGRESSION — it reframes it, and any budget arm returns under a **new** gate with a retention fix |
| ⛔ **NULL** | both `< +3 pp` | The decay is a curriculum-mix artefact of the training log. Scoring the final checkpoint is correct, and this closes a question that has been open by accident since the project began |

### 📏 Cost

6 runs (~30 min training) plus ~48 checkpoint evaluations. ⚠️ Score the **train**
split first to select, then the eval split **once**, on the selected checkpoint
only.

### Result

⛔ **Not yet run.**

---

## Not in any gate, and why

| | why it is held back |
|---|---|
| **agent index / role embedding** | ⛔ Excluded by decision, 2026-09-04: roles must **emerge**. B0 is granted roles-from-index as a documented advantage and the learned arm is not; that asymmetry is kept deliberately, and it is what makes Gate E a test of *credit* rather than of labelling |
| **DAgger from B0** | Held as the fallback if D, E and F all fail. [`bc_init.py`](../scripts/bc_init.py) exists and has never been reported; [`memory_horizon.md`](memory_horizon.md) predicts DAgger fixes the 9.4 % clone, since the collapse is covariate shift rather than missing memory. ⚠️ A teacher-initialised policy is a **probe**, not a like-for-like RQ2 or Gate B arm |
| **recurrence** | ⛔ [`memory_horizon.md`](memory_horizon.md) closed *target* memory with a hard oracle bound (perfect target state is worth **−0.4 pp**). It explicitly leaves **role-commitment** memory open — but that is what Gate E attacks, far more cheaply and with a lower bug density |
| **wider / deeper networks** | 📏 RQ2 measured architecture at ±1 pp across three rungs and MLP → DeepSets → GNN at 35.6 / 42.5 / 45.1. Capacity is not the suspect, and the actor is 137 k parameters against ~5,900 gradient steps — the budget binds long before the width does |
| **velocity action space** | ⛔ Gate A. ⚠️ Its own kill branch named exploration as the next suspect and `entropy_loss_scale` has been 0.0 throughout; `--entropy` and `--min-log-std` are now sweepable, so that follow-up is reachable — but it is Gate A's, not this file's |
