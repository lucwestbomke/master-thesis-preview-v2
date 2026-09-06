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
