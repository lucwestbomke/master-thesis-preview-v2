# What was carried over that still has to come out

**Written 2026-08-27, at the seeding commit.**

This repository was seeded by **copying** the predecessor project's physics,
geometry and measurement code rather than retyping it. That was deliberate: those
files are where four real bugs were found and fixed, and a rewrite would
re-introduce them.

The cost of that choice is that some carried-over code is **known dead or known
wrong for this project** and has to be removed. This file is the list, in order.

> ⚠️ **Task 5 comes first, and the numbering below is not the running order.**
> Tasks are numbered by subject, not by sequence. Every gate in `../PLAN.md` needs a
> trainer and **this repository has none** — `train.py` was deliberately left
> behind. So the real order is:
>
> 1. **Task 5** — build the trainer, and validate it by reproducing the inherited
>    **40.7 %** at 5 seeds on the train split with the env *unchanged*. A new
>    trainer that cannot reproduce a known number has an unknown bug in it.
> 2. **Task 1** — the velocity action space. The env change is unit-testable
>    without a trainer; **Gate A is not**.
> 3. Tasks 2–4, then 6, then 7.

> 🔒 **The seeding commit is green — 333 passed, 4 skipped.** That is the point of
> it. Do each task below as its own commit, re-run the suite, and keep the
> history bisectable. A reduction you cannot verify against a working baseline is
> a rewrite wearing a diff.

⚠️ **This is also how you learn the codebase.** The predecessor's real problem was
that nobody could hold it in their head. Shrinking a file is a better way to
learn it than reading it.

---

## 1. Action space → velocity setpoints — ✅ **DONE 2026-09-01**

> ✅ `core._advance_drones` takes a **velocity setpoint** scaled by
> `DRONE_DASH_MS`; the airframe closes on it rate-limited by `MAX_ACCEL_MS2`,
> which is retained as a property of the airframe rather than of the action.
> B0's `_servo` became `_velocity_command` and dropped `own_vel`.
>
> 🔒 **B0 is bit-identical across the change**, because the env's rate limit is
> per component and `vel + ((want-vel)/4).clamp(-1,1)*4 == vel + (want-vel).clamp(-4,4)`.
> `test_core.py::test_b0s_velocity_command_reproduces_the_old_servo_exactly`
> pins it, so every inherited B0 number stays valid.
>
> ⚠️ `test_fidelity.py`'s `fly` fixture had to be repaired: a uniform action
> resampled per step was a random walk in *velocity* under the old interface and
> is a zero-mean command under the new one. 📏 Swarm spread fell 219 m → 56 m and
> `chain_occluded` went to exactly 0. The assertions are unchanged; the fixture
> now steers the swarm into a relay chain along the MCV→HVT axis and is asserted
> to produce identical geometry at every rung.
>
> ⛔ **Gate A is NOT yet resolved** — that needs the 5-seed run.

**Where:** `src/env/core.py::_advance_drones`, and the `VEL_SCALE_MS` /
`MAX_ACCEL_MS2` constants.

**Why.** 📏 The learned policy sits pinned at the 25 m/s dash cap on **57 %** of
steps, presses against the map boundary on **23 %**, and commands mean
`|a_z| = 0.82` while parked at the 80 m ceiling — after 12 M steps it has not
learned that pushing up into a hard clamp does nothing. B0 scores 3.1 %, 0.9 %
and 0.005 on the same three.

And B0 does not solve the problem the learner faces: it computes a **desired
velocity** and converts it to acceleration with a proportional servo at the last
moment (`b0.py::_servo`). The learner is handed a double integrator with
saturation and has to discover that inner loop itself.

Commanding raw acceleration to a multirotor is also the *less* faithful choice.
PX4 and ArduPilot offboard control consume velocity setpoints over MAVLink.

**Gate A is declared in [`../PLAN.md`](../PLAN.md) — read it before running.** The
readout is deliberately **not** `mission_capable`: keep the change if the two
pathologies resolve (speed cap < 20 %, boundary < 5 %), because velocity
setpoints are the more faithful interface regardless.

**Expect to change:** `test_core.py`'s kinematics tests, and B0's servo (which
becomes a pass-through).

---

## 2. Fidelity ladder: five rungs → two

**Where:** `src/env/core.py` (`Fidelity`, `Rung`, `channel_occlusion`,
`binary_capacity`, `channel_jammer`, `reuse_limit`), `src/env/test_fidelity.py`,
`scripts/eval_fidelity.py`.

**Why.** The five-rung ladder existed to answer "which physical effects must a
channel model include for learned policies to transfer?" — a question whose
answer (occlusion dominates, in an urban canyon at 3.5 GHz) every reviewer
already believes. Two rungs (occlusion on / off) carry the same argument at a
fraction of the run budget and the code weight.

**Keep the one surprising result:** 📏 F1 is *harder* than F4 (27.9 % against
56.0 % under B0). The ladder is non-monotone in difficulty, which is
counter-intuitive and worth reporting as a secondary finding. Keep enough of
`eval_fidelity.py` to regenerate it.

⛔ **Do not use fidelity as a curriculum axis**, and do not gate the sensor or the
diagnostics on it. 📏 `observed` is 92.0 % at all five rungs *because* the sensor
runs on true geometry at every rung; a gated diagnostic would report
`chain_occluded` as 0.0 % at F0 by construction.

---

## 3. Reward: strip the dead terms, promote `PHI_V2`

**Where:** `src/env/reward.py`, `src/env/test_reward.py`.

**Remove**, all measured nulls at >= 5 seeds:

| term | result |
|---|---|
| `w_hold` — the `Φ_observe` hold factor | +1.65 pp on a 6.8 IQR. Structurally capped: a *factor* on `Φ_observe` can never be worth more than `w_observe · w_hold` |
| `w_relay` — a per-drone potential on `on_path` | raised between-drone advantage variance **71×** and moved behaviour not at all (`hop \| observed` 1.88–1.93 against a control of 1.91) |
| `Snapshot.on_path`, `agent_specific_state` | only read by the above |

**Promote** `PHI_V2` to the default and delete the shipped weights. 📏 The audit
that justifies it: across the closing band the shipped `Φ` moves **0.320 in
total** — 0.0133 per 8 m step against the **0.0544** the energy term pays for
cruising — and it is *exactly constant in four drones out of five*, because every
component is a hard `min` / `max` / routing reduction. Moving a drone that holds
no role 8 m homeward is worth **0.0000**, at any distance.

🔒 **Keep the sizing rule**: the five component weights sum to 1.0 and
`potential_scale` stays 10. `Φ` is redistributed, never inflated — PBRS charges
`(γ−1)·Φ` per step for *holding* a state, a drag proportional to `Φ` and
therefore largest for the best policy.

⛔ **Do not touch the objective weights** (`mission`, `idle`, `energy`,
`battery_variance`, `effort`). Only `λ` is sweepable, and the behavioural
orderings in `weight_constraints_satisfied()` are what pin the rest.

---

## 4. Remove recurrence — ✅ **DONE 2026-08-30**, with task 5

> ✅ `recurrence.py`, `test_recurrence.py`, `SwarmActorRNN`, `SwarmCriticRNN` and
> both `blob["recurrent"]` branches are gone. `eval_policy.py` and
> `viz/episode.py` now **refuse** a recurrent checkpoint rather than silently
> loading it as a feedforward policy, which would score a different network.
> Done here because task 5 rewrote both model files anyway and carrying the GRU
> through that rewrite would have been work spent on something already killed.

**Where:** `src/models/recurrence.py`, `src/models/test_recurrence.py`,
`SwarmActorRNN` in `actor.py`, `SwarmCriticRNN` in `critic.py`, and the
`blob["recurrent"]` branches in `scripts/eval_policy.py` and `src/viz/episode.py`.

**Why.** 📏 Killed on its own pre-declared rule at 5 seeds: −1.05 pp, observer
tenure 36.8 against a required 95, and the seed IQR *widened* 4.7 → 6.9.

⚠️ Distinguish this from "recurrence does not train" — it trains, and it reaches
feedforward parity on the easy curriculum stage. It simply does not help. Do not
re-propose it for observer tenure without a new mechanism.

It is carried over only because `actor.py` and `critic.py` import it and the
seeding commit had to be green.

---

## 5. Replace skrl with an own PPO / MAPPO — ✅ **DONE 2026-08-30**

> ✅ `src/training/ppo.py` (~380 lines), `src/training/probe.py`,
> `scripts/train.py`, `src/training/test_ppo.py`,
> `src/training/test_train_cli.py`. skrl is out of `pyproject.toml` and out of
> the venv; `test_actor.py` asserts no framework base class can creep back.
> The validation gate was declared before the runs and the result is in
> [`../results/trainer_validation.md`](../results/trainer_validation.md).
>
> 📏 The probe **demonstrably catches** bug 2 below: under the stale
> `terminated`-only mask it reaches 12.0 against a known optimum of 33.0, and
> degrades over training exactly as the predecessor's runs did. ⚠️ It does *not*
> catch a missing truncation bootstrap on its own — with no bootstrap there is no
> doubled continuation term for the leak to double-count, so **the two bugs
> cancel** and the probe looks healthy while two things are wrong. That is why
> the bootstrap has its own unit tests rather than relying on the probe.

**Where:** the `skrl` dependency in `pyproject.toml`; `src/models/actor.py` and
`critic.py`, which inherit from `GaussianMixin` / `DeterministicMixin` / `Model`.

**Why.** ☠️ skrl has produced **four silent bugs** in this project, every one of
which cost real time and none of which raised an error:

1. `GaussianMixin(clip_actions=True)` clamps the sampled action and then
   evaluates its log-probability under the *unclamped* Normal, which **inverts
   learning** — the policy fell 30 % → 4.6 %.
2. `skrl/agents/torch/ppo/ppo_rnn.py` is an un-migrated copy of an older PPO
   carrying its own stale `compute_gae`, which masks on `terminated` alone. At
   every truncation the recursion runs *through* the reset. Recurrent training
   collapsed for a week and the GRU was blamed for it.
3. `MAPPO_CFG` cannot be constructed with its own defaults — `Config.expand()`
   rejects any dict whose keys are a strict subset of `possible_agents`, which an
   empty dict always is.
4. Handing skrl one shared `Model` under `N` agent ids builds `N` optimizers over
   the same parameters and runs `N` stale sequential updates.

**What to build.** A single-file PPO with a centralized state-based critic and
parameters shared across homogeneous agents — that *is* MAPPO for a homogeneous
swarm. Roughly 300 lines you understand completely. The models become plain
`nn.Module`s returning `(mean, log_std)` and `(value,)`.

Three things the replacement must get right, because each failed silently before:

* 🔒 **`time_limit_bootstrap`** — episodes truncate at a fixed horizon that is not
  part of the task. Treating truncation as termination teaches the critic that
  the world ends at 600 steps, and at `γ ≈ 0.997` that bias is large and silent.
* 🔒 **Bootstrap off `env.final_observations()` / `final_states()`**, not what
  `step()` returns. With auto-reset the returned tensors are already a fresh
  episode's opening.
* 🔒 **The swarm is ONE parameter-shared agent** over `num_envs * N` rows, not `N`
  agents.

### 🔒 One test must come back with the trainer

`test_every_PBRS_safe_reward_knob_is_settable_from_the_command_line` was dropped
with `test_train.py`, and it has to be rebuilt against the new entry point. It
derives its list from `RewardWeights` itself — every field that is not an
objective weight and not a physical reference lives inside `Φ`, is
optimum-preserving, and must therefore be settable from the CLI, because a knob
that cannot be set cannot be swept.

📏 It has caught two real misses of exactly the same shape, and it is cheap:

* `--w-relay` shipped with its `TrainConfig` field, its `build_weights` wiring
  and its call site — and **no `add_argument`**. It failed on a GPU box as
  `unrecognized arguments`, one command into a 5-seed sweep.
* `w_approach` / `w_observe` / `w_link` were *documented as free* while being
  reachable from nowhere: no `build_weights` branch and no flag. A whole session
  recommended tuning them.

⚠️ `PHI_V2` adds `n_cover_samples`, an `int` rather than a `float`, which the
predecessor's flag loop would not have covered. Derive the list; do not hand-list
it.

⚠️ Add a probe with a known optimum that *spans the episode* before trusting the
loop. 📏 The predecessor's probe used a pure per-step action cost, which has
almost no cross-episode structure — it cleared the plumbing while bug 2 above was
live, and hid it.

---

## 6. Trim `evaluate.py` to the metrics that will be reported

**Where:** `src/baselines/evaluate.py`.

It accumulated ~35 metrics across the predecessor's three research questions.
Keep the mission group, `observer_range_m`, `off_axis_m`, `role_entropy`,
`relay_entropy` and the hop histogram; drop the rest as they stop being cited.

⛔ **Two of the inherited metrics are not usable as defined** and must not be
carried into a result without being fixed first:

* `hop_mean | observed` **measures geometry, not behaviour.** 📏 random 1.83,
  every learned policy 1.86–1.93, B0 2.26 — hop count follows from where the
  observer stands against `R` = 524 m. Three interventions were judged on it.
* `chain_occluded` **confounds with hop count** (`corr = 0.963`). It is a
  per-chain statistic, so it rises with the number of edges. Report the
  **per-edge** rate instead, which is hop-count-invariant.

---

## 7. Re-freeze a behavioural trace, once — and only once the env stabilises

The predecessor's frozen trace (`data/f4_golden.pt.gz`) was **not** carried over.
It records what the env did before the fidelity ladder existed, and the action
space change in task 1 invalidates it by construction — keeping it would have
meant a test failing for the right reason and being read as the wrong one.

After tasks 1–3 land, capture a fresh trace and pin it. 🔒 Then the rule that made
the old one valuable applies again: **never re-capture it to make a test pass.**
A failure means the environment changed, and the question is which number moved.

---

## Not carried over at all

Listed so that "where did X go?" has an answer.

| Left behind | Why |
|---|---|
| `src/training/train.py`, `skrl_wrapper.py`, `recurrent_ppo.py` + tests | replaced by task 5 |
| `src/env/golden.py`, `test_golden.py`, `data/f4_golden.pt.gz`, `scripts/capture_f4_golden.py` | task 7 |
| `src/env/swarm_env.py`, `test_swarm_env.py` | PettingZoo adapter, for API-compliance tests only. Training never used it |
| `scripts/probe_credit.py`, `sweep.py`, `dedupe_summary.py`, `cuda_session.sh` | tied to the removed trainer, or to the predecessor's block structure |
| `configs/*` | encode the five-rung matrix and the removed trainer's flags. Rewrite for the J ladder |
| `results/*` | ~5 MB of the predecessor's runs. Their numbers live in `docs/INHERITED.md`; the old repository keeps the artefacts |
| `docs/ROADMAP.md`, `THESIS_PLAN.md`, `BLOCK_D.md`, `BLOCK_G_PLAN.md` | superseded plans. Keeping a superseded plan is how a repository stops being legible |

Everything else from `docs/` is archived read-only under `docs/inherited/`.
