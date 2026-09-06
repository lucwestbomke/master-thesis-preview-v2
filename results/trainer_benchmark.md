# Does the trainer work? — external benchmarks, declared 2026-09-06 before the runs

🔒 **Declared before any five-seed run, and not edited afterwards.** Results are
appended under the rule. `AGENTS.md`: *"A rule invented after the fact is not a
rule."*

**Phase 0 of [`docs/CAPABILITY_BRIEF.md`](../docs/CAPABILITY_BRIEF.md)**, whose
instruction is unambiguous: *"The custom PPO in `src/training/ppo.py` has never
been checked against a known reference. Until it is, a flat learning curve cannot
be attributed to the task rather than the trainer… If it does not reach reference
performance, stop and fix the trainer. Nothing downstream is interpretable."*

---

## §0 Why this is not covered by what already exists

Two instruments in this repository already say the trainer is fine, and **neither
can rule out a trainer that is consistently wrong**:

| instrument | what it clears | what it cannot clear |
|---|---|---|
| [`trainer_validation.md`](trainer_validation.md) | that the de-skrl'd trainer reproduces the **inherited** 40.7 % | that the inherited number was itself produced by a working optimiser. It is a reproduction, not a validation |
| [`../src/training/probe.py`](../src/training/probe.py) | the GAE episode mask, against a known optimum that spans the episode — 📏 32.7 correct against 12.0 with skrl's mask | anything this repository did not think to write a probe for. ⚠️ Its own docstring records that it does **not** detect a missing truncation bootstrap: 📏 32.4 against 32.7, indistinguishable |

⭐ A **published** PPO number is the one reference that was not produced by this
project, on a task this project did not design. That is the whole point of it.

---

## ⚠️ The device, stated first

📏 **This machine has no CUDA.** `torch.cuda.is_available()` is `False`; Apple
M-series, torch 2.13.0, arm64. Every run below is **CPU**, and `AGENTS.md`'s rule
about mixing devices does not bind here for a reason worth stating: the comparison
is against an *external* published number, not against another number in this
repository. ⛔ No number in this file may be compared with any number in
`results/` — different task, different device, different everything.

---

## What is under test

🔒 **`src/training/ppo.py`, unmodified.** `PPOTrainer`, `compute_gae`,
`RunningScalar`, the truncation bootstrap, the value clip, the advantage
normalisation, the two optimizers and the minibatch shuffling are the code the
mission runs. [`../scripts/bench_trainer.py`](../scripts/bench_trainer.py) imports
them.

⛔ **The actor cannot be the mission actor.** `SwarmActor`'s trunk unpacks a
108-wide swarm observation through `core.unpack_flat` and Pendulum has three
numbers. `BenchActor` **subclasses `SwarmActor` and replaces only the trunk**, so
`forward` / `distribution` / `act` / `evaluate` — the unclipped Normal, the
state-independent per-dimension `log_std`, the `tanh_mean` switch, the
`min_log_std` floor — are literally the same code. `SwarmCritic` is used as-is.

⚠️ **Three declared departures**, each because the trainer does not have the
feature and adding one to pass a benchmark would defeat the exercise:

* **no gSDE.** Both `rl-baselines3-zoo` entries quoted below set `use_sde: True`.
* **advantages are normalised over the whole batch**, not per minibatch as SB3
  does. That is `ppo.py`'s stated choice and it is under test, not patched out.
* **`MountainCarContinuous-v0` runs 8 envs × 8 rollouts** where the tuned entry
  runs 1 env × 8 steps; `PPOTrainer` collects `rollouts` steps of `num_envs` envs
  and 8 rows per rollout is not a workable minibatch.

---

## The three tasks

### 1. `lqr` — ⭐ a reference that is **derived, not cited**

⛔ `AGENTS.md`: *"Cite constants an AI produced"* is on the never-do list, and a
published benchmark number is a citation however carefully it is fetched. So the
first task carries a reference this repository computes.

A 2-D double integrator, `dt = 0.1`, `x = (px, vx, py, vy)`, `u = (ax, ay)`,
reward `-(x'Qx + u'Ru)` with `Q = I`, `R = 0.1 I`, `x_0 ~ U[-1,1]^4`, horizon
**100 steps, truncating and never terminating**. `gamma = 0.99`, and the reported
quantity is the **discounted** return — the objective PPO actually maximises.

🔒 The optimum is the exact finite-horizon discounted Riccati recursion,
`P_H = 0`, `P_t = Q + g A'P_{t+1}A - g^2 A'P_{t+1}B S^{-1} B'P_{t+1}A`, evaluated
in closed form as `J* = tr(P_0 * Sigma)` with `Sigma = E[x_0 x_0'] = (1/3) I`.
**No sampling and no citation.**

☠️ **A correction made during construction, before the gate was written.** The
first draft solved the *steady-state, infinite-horizon* DARE and scored the
*undiscounted* return against it. 📏 That "optimum" is beaten by scaling the same
gain up — `K * 1.10` scored **−17.63** against `K`'s **−18.81** — because two
things were mismatched at once, an infinite horizon against a finite one and a
discounted objective against an undiscounted readout. A second draft then divided
the gain by `ACT_SCALE` twice and detuned the controller 3×. Both are recorded
here rather than in a corrections log because neither survived into a declaration.

📏 **The two independent checks the reference passes**, before any policy was
trained:

| check | value |
|---|---|
| exact analytic optimum, `tr(P_0 Sigma)` | **−11.898** |
| the same optimum **simulated**, `-K_t x` through the same rollout loop, 512 episodes | **−12.161 ± 0.326** (0.8 se from the analytic value) |
| fraction of optimal-control actions hitting the `[-1, 1]` box | **0.44 %** — so the unconstrained solution is very nearly feasible and the analytic number is a bound |
| the do-nothing floor, `u = 0`, in closed form | **−1139.2** |

**Readout: `cost_ratio` = policy cost / optimal cost, PAIRED.** `evaluate` and
`evaluate_lqr_optimum` both open with `env.reset()`, which re-seeds the episode
stream, so the policy and the optimal controller are scored on the **same initial
states** and the state distribution contributes no noise to the ratio. 1.0 is
optimal.

### 2. `pendulum` — `Pendulum-v1`, against two published PPO numbers

📏 **Tuned.** `rl-baselines3-zoo/benchmark.md`, `ppo | Pendulum-v1 |
mean_reward -172.225 | std_reward 104.159 | n_timesteps 100k | eval_episodes 750`.
Hyperparameters from the same repository's `hyperparams/ppo.yml`: `n_envs 4,
n_steps 1024, gamma 0.9, gae_lambda 0.95, n_epochs 10, lr 1e-3, clip_range 0.2,
ent_coef 0.0, use_sde True`. All reproduced except `use_sde`.

📏 **Untuned.** `docs.cleanrl.dev/rl-algorithms/rpo`, `ppo_continuous_action` on
`Pendulum-v1`: **−1141.98 ± 135.55**, 8 M steps, 10 seeds. ⭐ Worth stating
plainly: vanilla PPO on Pendulum is a **documented failure case**, which is the
whole reason the RPO paper uses it. So the two references bracket the answer
rather than agreeing on it.

⚠️ **The published `std` is across evaluation *episodes* of one trained policy**,
not across seeds. It is used below as a tolerance band, and that is a choice, not
a statistic.

### 3. `mountaincar` — `MountainCarContinuous-v0`, ⛔ **reported, not gated**

📏 `benchmark.md`: `ppo | MountainCarContinuous-v0 | 88.343 | 2.572 | 20k |
633 episodes`, under `normalize: true`, `log_std_init -3.29` and **`use_sde:
True`**.

🔒 **The prediction is declared here, before the run: this task FAILS, at
approximately 0.** The tuned entry's exploration is gSDE at `sigma ~ 0.037`, and a
diagonal Gaussian at that scale on a task whose reward is `+100` on success and
`-0.1 * u^2` everywhere else does not reach the goal. ⛔ A failure here is
**not** evidence about the trainer, which is why it is ungated — it is included
because a suite in which every task passes tells you nothing about what the suite
can see.

---

## 🔒 The negative control — *a benchmark that cannot fail is not a benchmark*

Every gated task is also run with `--control frozen`: **both learning rates set to
zero, everything else identical.** The run still collects, computes GAE and calls
`optimizer.step()`; it simply cannot move. 🔒 A PASS is only readable against the
control's score, and the control is reported beside every result.

---

## 🔒 The decision rule

⚠️ Branches partition the real line. *(Gate A and `trainer_validation.md` each
recorded a rule that failed to; this one is written not to.)* 5 seeds each, judged
on the **worst** seed where a worst-seed clause is given, per `AGENTS.md`.

### `lqr` — gated on `cost_ratio`

| branch | rule |
|---|---|
| ✅ **PASS** | worst-seed `cost_ratio` ≤ **1.25** |
| ⚠️ **PARTIAL** | worst-seed `cost_ratio` in (1.25, 2.0] — controlling the plant but not near-optimally |
| ☠️ **FAIL** | worst-seed `cost_ratio` > 2.0 |

### `pendulum` — gated on median eval return over 5 seeds

| branch | rule | reading |
|---|---|---|
| ✅ **PASS** | median ≥ **−276.4** (tuned reference mean − 1 published sd) **and** worst seed ≥ **−1006.4** (untuned reference mean + 1 published sd) | reaches published *tuned* PPO performance without gSDE |
| ⚠️ **PARTIAL** | median in [−1006.4, −276.4) | learns, but not to the tuned reference. Attributable to the missing gSDE rather than to the algorithm, and the `lqr` arm then decides |
| ☠️ **FAIL** | median < −1006.4 | no better than a reference that is itself a documented PPO failure |

### The overall verdict this file exists to produce

| branch | rule | consequence |
|---|---|---|
| ✅ **VALIDATED** | `lqr` PASS **and** `pendulum` PASS or PARTIAL | the trainer reaches reference performance. ⭐ A flat learning curve on the mission is then attributable to the task, and the eight nulls plus Gates D and E keep their reading. Phase 1 proceeds |
| ⚠️ **QUALIFIED** | `lqr` PASS **and** `pendulum` FAIL | arithmetically correct on a task with a derived optimum, not competitive on a standard benchmark. ⛔ Do not proceed to Phase 2 before naming which of the four suspects below it is |
| ☠️ **FAILED** | `lqr` FAIL | ⛔ **Stop.** Nothing downstream is interpretable. Suspects in the order `docs/CAPABILITY_BRIEF.md` §2 gives them: advantage normalisation, the value-loss clip, the GAE bootstrap at truncation vs termination, observation normalisation |

---

## 🔒 Reported whatever the branch

Per `docs/CAPABILITY_BRIEF.md` §3, for every task and both controls: **total Adam
steps**, `approx_kl`, `grad_kept`, `clip_fraction`, `explained_variance`,
`grad_norm_actor`, `grad_norm_critic`, per-dimension `sigma`, and wall-clock.

⚠️ `clip_fraction`, per-dimension `sigma`, per-dimension action saturation and
`adam_steps` **did not exist before this programme** — they are Phase 1 of the
brief and were added to `ppo.py` in the same session. 🔒 They are read-only
diagnostics: they consume no RNG and touch no parameter, and the full suite
(**460 passed, 4 skipped**) is unchanged by them.

⭐ **And one thing that is a result on its own, whatever the branch.** The brief's
§0 says `grad_kept` is *"NaN in every log in `runs/`"*. It is not NaN in any run
in this file, and its value is reported for each — because if the actor's gradient
is being throttled by the critic through the joint norm clip on a task PPO is
known to solve, that is a property of `ppo.py`'s default configuration and not of
the relay mission.

---

# Result — ✅ **VALIDATED, 2026-09-06.** Both gated tasks PASS.

📏 CPU, arm64, torch 2.13.0, 5 seeds each, deterministic mean-action evaluation.
Raw rows in [`trainer_benchmark.jsonl`](trainer_benchmark.jsonl).

## `pendulum` — ✅ **PASS**, and it beats the published *tuned* reference

| | median | worst | best | per seed |
|---|---|---|---|---|
| **this trainer**, 100 k steps, 100 eval episodes | **−164.56** | **−220.02** | −149.21 | −220.0 · −171.0 · −164.6 · −149.9 · −149.2 |
| 📏 `rl-baselines3-zoo` PPO, 100 k steps, **tuned, with gSDE** | −172.23 ± 104.16 | | | |
| 📏 `cleanrl` `ppo_continuous_action`, 8 M steps, 10 seeds | −1141.98 ± 135.55 | | | |
| 🔒 **negative control** — same code, `lr = 0`, 3 seeds | **−1167.09** | −1186.68 | | |

✅ **Median −164.56 ≥ −276.4 and worst seed −220.02 ≥ −1006.4.** Both clauses of
the PASS rule are met, and the median is **7.7 points better than the tuned
reference mean** at the same step budget, **without gSDE**.

☠️ **And read the control row against the untuned reference.** A trainer with its
learning rate set to zero scores **−1167.09**; the published `ppo_continuous_action`
number is **−1141.98 ± 135.55** at 80× the budget. They are indistinguishable.
🔍 That is not a criticism of `cleanrl` — Pendulum is the documented failure case
the RPO paper was written about — but it does mean the untuned reference was the
right thing to put at the FAIL boundary, and it says something about how little a
"published baseline" guarantees on its own.

## `lqr` — ✅ **PASS** against a reference with no citation in it

| | median | worst | per seed (discounted) |
|---|---|---|---|
| **`cost_ratio`** (1.0 = optimal, paired) | **1.092** | **1.154** | 1.127 · 1.107 · 1.098 · 1.068 · 0.978 |
| discounted return | −13.06 | −13.41 | −13.41 · −13.16 · −13.06 · −12.70 · −11.64 |
| 🔒 exact analytic optimum, `tr(P_0 Sigma)` | **−11.898** | | |
| 🔒 do-nothing floor, closed form | −1139.2 | | |
| 🔒 **negative control**, `lr = 0`, 3 seeds | | | `cost_ratio` **96.2** |

✅ **Worst-seed `cost_ratio` 1.154 ≤ 1.25.** The policy pays 15 % more cost than
the optimal controller on its worst seed and 9 % on its median, against a
do-nothing floor at 96×.

⚠️ **Seed 4's 0.978 is not "better than optimal".** The analytic value is an
expectation over `x_0 ~ U[-1,1]^4` while a seed's eval is 256 sampled episodes; the
paired *simulated* optimum on the same states is what makes the ratio meaningful,
and a single seed can sit slightly under 1.0 on it. It is inside the same noise
that puts the simulated optimum 0.8 se from the analytic one.

## `mountaincar` — ⛔ fails at ≈ 0, **exactly as declared before the run**

📏 median **−0.016**, worst **−0.035**, against the tuned reference's **88.343 ±
2.572**. The declaration predicted *"this task FAILS, at approximately 0"* and
gave the mechanism: a diagonal Gaussian at `sigma ≈ 0.032` never reaches the goal,
where gSDE's temporally-correlated noise does. ⛔ Ungated, and it stays ungated.

⭐ It earns its place anyway: it is the arm that shows the suite can distinguish
*"the trainer is fine"* from *"everything passes"*.

## 🔒 Diagnostics, reported whatever the branch

| | `pendulum` (PASS) | `lqr` (PASS) | `mountaincar` (fails) | brief §3 "healthy" |
|---|---|---|---|---|
| **Adam steps** | 16,000 | 78,240 | 3,130 | — (📏 the mission runs 5,888) |
| `approx_kl` | 0.0143 – 0.0340 | 0.0199 – 0.0275 | 0.0008 – 0.0023 | 0.01 – 0.02 |
| `clip_fraction` | 0.104 – 0.257 | 0.133 – 0.215 | 0.039 – 0.125 | 0.05 – 0.20 |
| ☠️ **`grad_kept`** | **0.152 – 0.238** | **0.035 – 0.040** | 0.564 – 0.889 | **> 0.8** |
| `grad_norm_actor` | 1.11 – 1.70 | 18.3 – 22.9 | 5.10 – 10.01 | — |
| `grad_norm_critic` | 2.34 – 3.90 | 0.0023 – 0.0079 | 0.0046 – 0.0159 | — |
| `explained_variance` | 0.9990 – 0.9996 | 0.868 – 0.984 | 0.997 – 0.999 | do not let it fall |
| final `sigma` | 0.25 – 0.37 | 0.0081 – 0.0096 | 0.031 – 0.032 | — |
| wall clock / seed | 17 s | 79 s | 3 s | — |

### ☠️ The finding that changes an instruction in the brief

📏 **Both tasks that PASS have `grad_kept` far below 0.8, and the one that fails
has the highest `grad_kept` in the table.**

* `lqr` reaches **within 15 % of an exactly-computed optimum** while the joint norm
  clip discards **96 %** of the actor's gradient (`grad_norm_actor` 18–23 against a
  0.5 clip).
* `pendulum` **beats a published tuned baseline** while discarding **76 – 85 %**.
* `mountaincar` keeps **56 – 89 %** of its gradient and learns nothing.

⛔ So `grad_kept` is **not** a health check, and
`docs/CAPABILITY_BRIEF.md` §4's Block A acceptance test — *"`grad_kept` > 0.8 and
`approx_kl` above 0.005"* — **would have rejected both configurations that reach
reference performance.** 🔍 The mechanism is not mysterious: `clip_grad_norm_`
rescales, it does not truncate, so under Adam a uniformly rescaled gradient is
very nearly the same update. What the clip actually costs is the *relative*
weighting between the actor and the critic in the joint norm, and both PASS runs
show that costing nothing measurable.

🔒 **Recommendation, for Phase 2's declaration and not applied here:** raise
`grad_norm_clip` if `grad_norm_actor` says the clip binds, but do **not** gate
Block A on `grad_kept`. `approx_kl` and `clip_fraction` together already say
whether the policy is moving, and on the shipped mission configuration they say it
is not — `approx_kl` 0.002–0.004 against 0.014–0.034 here.

⚠️ **What this does NOT say.** It does not say the mission's `grad_kept` is
harmless; it says a low `grad_kept` is not by itself evidence of a broken run, so
it cannot carry the Block A gate. The mission's binding constraint is still
`approx_kl`, which is an order of magnitude below every value in this table.

## ⚠️ Limitations, stated rather than left to be found

* ⛔ **The truncation bootstrap is not discriminated by these tasks.** 📏 Measured:
  `lqr` at 500 k steps with `time_limit_bootstrap=False` scores `cost_ratio`
  **0.937** against the correct **0.961** — indistinguishable. 🔍 The reason is
  structural: an LQR policy drives the state to the origin, so `V(x_final) ≈ 0` and
  there is nothing for the bootstrap to add. Pendulum truncates at 200 steps with a
  large ongoing negative reward and `gamma = 0.9`, which also suppresses it.
  `src/training/probe.py` records the same blind spot from the other side
  (📏 32.4 against 32.7). ⭐ **The truncation bootstrap in `ppo.py` remains covered
  only by its unit tests**, and this file does not change that.
* ⛔ **No locomotion task.** The brief asks for *"a Gym/Brax locomotion task"*.
  MuJoCo and Box2D are not installed and both are dependency additions that
  `AGENTS.md` requires flagging; `Pendulum-v1` and `MountainCarContinuous-v0` are
  what `gymnasium` provides out of the box here. ⚠️ So the trainer is validated on
  4-D and 3-D observation spaces and **not** on a high-dimensional one.
* ⚠️ **`BenchActor` is not `SwarmActor`'s trunk.** The relational trunks, the
  observation unpacking and the max-N padding are not exercised here. They are
  exercised by `src/training/probe.py`, which pads into the real `FLAT_DIM`.
* ⚠️ **CPU, single machine.** ⛔ No number here is comparable with anything in
  `results/`.

## 🔒 Consequence for the programme

✅ **The trainer reaches reference performance, so a flat learning curve on the
mission is attributable to the task and not to the optimiser.** Gates D and E keep
their readings, the eight prior nulls keep theirs, and
[`credit_assignment.md`](credit_assignment.md)'s structural finding is not
weakened by an implementation doubt.

⛔ **It does not settle the budget question.** `docs/CAPABILITY_BRIEF.md` §0's
diagnosis is about `approx_kl` at 0.002–0.004 and ~5,888 Adam steps, and this file
says nothing about either — it says the code that would consume a larger budget is
correct. 📏 Gate D already ran that experiment and returned REGRESSION.

---

# Addendum — does gSDE work, and does *correlation* do the work? Declared 2026-09-06 before the runs

🔒 **A new declaration under the same standard**, appended rather than edited into
the section above. ⛔ Nothing above changes: the Phase 0 verdict stands and the
`mountaincar` row above remains the no-gSDE measurement it was declared as.

## Why

📏 Phase 0 left one arm failing at ≈ 0 against a published **88.343 ± 2.572**, and
named the difference: the tuned entry uses **gSDE** and this trainer had no such
distribution. gSDE is now implemented in `SwarmActor` (`_init_sde`), OFF by
default. This addendum asks whether the implementation does what gSDE is for.

⚠️ **Beating the published number is not the interesting question, and on its own
it would not be evidence about the mechanism.** gSDE changes *two* things at once
against the shipped Gaussian: the noise becomes temporally correlated, **and** its
effective magnitude changes, because `sigma_hat(s) = ||phi(s)|| * sigma` rather
than `sigma`. A win against the reference config alone cannot separate them.

## 🔒 The design: a 2 × 2, not a comparison against a paper

**`{gSDE on, gSDE off} × {initial_log_std −3.29, −1.0}`**, 5 seeds each, all four
cells reported. Everything else is the tuned `MountainCarContinuous-v0` entry:
`normalize: true`, `gamma 0.9999`, `gae_lambda 0.9`, `lr 7.77e-5`, `clip_range
0.1`, `ent_coef 0.00429`, `vf_coef 0.19`, `max_grad_norm 5`, `n_epochs 10`,
`ortho_init False`, 20 k steps.

* **The gSDE main effect at a fixed `initial_log_std`** is the readout. Same
  nominal scale, same everything else, one factor changed — so a difference is
  attributable to *correlation*.
* **Two levels of the scale**, because `initial_log_std` does **not** mean the
  same thing in the two arms and cannot be made to. 📏 `||phi(s)|| = 2.10` on a
  64-wide trunk, so the gSDE arm's effective deviation is ~2.1× its nominal one
  while the Gaussian arm's is exactly nominal. ⛔ Running one level would confound
  the main effect with that offset; two levels bracket it.
* **`sigma_x` is reported for every cell.** It is now read from a real forward
  pass under gSDE rather than from a parameter, so the *effective* deviation each
  cell actually explored at is on the record and the confound is measurable rather
  than argued about.

☠️ **A correction made during construction, before this was declared.** The first
version of `_init_sde` subtracted `0.5*log(latent_dim)` from `initial_log_std` so
the flag would keep naming the effective deviation. 📏 That assumed
`||phi||^2 ~ latent_dim`; the measured `||phi||` is **2.10 against an assumed
8.0**, and it is not a constant — it moves with the task and with training. The
correction is removed and the convention now matches the reference
implementation, with the diagnostic carrying the truth instead.

## 🔒 The decision rule

⚠️ Branches partition the real line. Δ is median `eval_return_mean` over 5 seeds,
gSDE on minus gSDE off, **at the same `initial_log_std`**.

| branch | rule | reading |
|---|---|---|
| ✅ **MECHANISM CONFIRMED** | Δ ≥ **+50** at either level, worst-seed Δ ≥ 0 | correlated exploration solves a task white noise cannot, at matched scale. The implementation does what gSDE is for |
| ⚠️ **PARTIAL** | Δ ≥ **+10** at either level | it reaches the goal sometimes. Correlation helps; the published 88.3 needs something else this reproduction does not have |
| ⛔ **NULL** | \|Δ\| < 10 at both levels | ⭐ **A real result, and the one that matters for the relay**: gSDE is not the difference between 0 and 88.3 here, so either the reproduction is missing something else, or the mechanism does not transfer. ⛔ In that case gSDE does **not** go to the mission on this evidence |
| ☠️ **REGRESSION** | Δ ≤ **−10** at either level | report it |

🔒 **What this gate does NOT license, whatever it says.** ⛔ A pass is evidence
that the *implementation* is correct and that correlated noise can matter on
*some* task. It is **not** evidence that gSDE helps the relay. That would be a
Gate, on CUDA, at 5 seeds, against B0 through `evaluate.py` — and this project's
base rate for a new knob is eight pre-declared nulls, Gate A, Gate D and Gate E.

## 🔒 Reported whatever the branch

Per cell: median / worst / per-seed `eval_return_mean`, `sigma_x` (the
**effective** deviation), `approx_kl`, `clip_fraction`, `grad_kept`,
`explained_variance`, `entropy`, and total Adam steps.

---

## ⚠️ Amendment, declared 2026-09-06 after reading `sigma_x` and BEFORE the control run

🔒 **Recorded rather than silently replacing the design above.** The 2 × 2 ran and
its rule resolves cleanly. But its **stated purpose** — *"a difference is
attributable to correlation"* — is **not delivered by the cells as run**, and the
column added to check that is what shows it:

| cell | effective `sigma_x` |
|---|---|
| `−3.29`, gSDE off | 0.0325 – 0.0373 |
| `−3.29`, gSDE on | 0.0190 – 0.0407 |
| `−1.0`, gSDE off | **0.3417 – 0.3499** |
| `−1.0`, gSDE on | **0.4323 – 0.7881** |

⛔ At the level where gSDE wins, the two arms are **not at matched effective
scale**: gSDE explored 1.3 – 2.3× wider. The declaration predicted this offset
and claimed two levels would "bracket" it. 📏 They do not — there is no gSDE-off
cell anywhere in the 0.43 – 0.79 band, so *magnitude* remains an unexcluded
explanation for the win. The bracketing argument was wrong and this amendment
says so.

### 🔒 The control, declared before it runs

**gSDE OFF at three scales spanning the gSDE arm's measured band**:
`initial_log_std ∈ {−0.8, −0.5, −0.25}`, i.e. `sigma ≈ 0.449 / 0.607 / 0.779`.
5 seeds each, everything else identical.

| branch | rule | reading |
|---|---|---|
| ✅ **CORRELATION** | every gSDE-off cell's median stays below **+10** | white noise fails across the whole band the gSDE arm explored at, so the win is not magnitude |
| ⛔ **MAGNITUDE** | any gSDE-off cell's median reaches **+48.6** (the gSDE arm's *worst* seed) | ☠️ the 2 × 2's win is a scale effect and gSDE's correlation is not doing the work. The MECHANISM CONFIRMED verdict above would then be **correct on its rule and wrong in its reading**, and it would be recorded that way |
| ⚠️ **PARTIAL** | any cell's median in [+10, +48.6) | white noise at the right scale gets part of the way. Report both and attribute nothing |
