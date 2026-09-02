# AGENTS.md — Contested Relay

Entry point for any agent or human working in this repo. Deliberately short.
Detail lives in `docs/` and is read **on demand**.

## How to read these documents

Every claim here and in `docs/` is one of three kinds, and they are **not equally
binding**:

| tag | means | how to change it |
|---|---|---|
| 📏 **MEASURED** | a number that came out of a script, with device and seed count | measure it better, then overwrite. Data is not a rule |
| 🔒 **CONSTRAINT** | protects a result's interpretability, not the model's score | change it **deliberately**, in `docs/inherited/DECISIONS.md`, and re-derive what it invalidates |
| 🔧 **PROVISIONAL** | a default nobody has tested | change it freely, on evidence, no ceremony |

⚠️ Untagged prose is 🔧 by default.

---

## The claim

> 🔍 **Exploitability, not capability, is the right axis on which to compare
> learned and scripted multi-agent policies.**

A swarm of `N = 5` UAVs observes a moving ground target and relays the feed to a
command vehicle over a multi-hop chain at >= 15 Mbps, while a jammer degrades
links. 📏 **The scripted baseline B0 wins the static task by 15.0 pp** (55.7 %
against the GNN's 40.7 %, eval split, 5 seeds) — and that is a **settled premise
of this work, not an open question.** Heuristics win static, fully specified
problems, and [`PLAN.md`](PLAN.md) §3 closes the axis on three independent lines
of measured evidence:

* 📏 the gap is **`observed` and nothing else** — conditioned on a sightline the
  GNN converts it as well as B0 does, 0.620 against 0.617;
* 📏 **B0 wins the reward too**, 222.9 against 85.8 `episode_return`, and
  `episode_return` rank-correlates with `mission_capable` at **ρ = 0.987** over
  20 rows — so the objective is not misspecified;
* 📏 **eight pre-declared interventions, eight nulls**, the last with a
  *measured-adequate* gradient.

⛔ **So stop asking whether learned control beats the heuristic on the static
task.** The protagonist is the strongest available policy and it happens to be
scripted; the question asked of it is how much of that capability an adversary
can take away.

**Four objectives, one arc** — an adversary that adapts → a policy co-trained
against it → running on the hardware that has to fly it. Full text in
[`PLAN.md`](PLAN.md) §2:

1. **RQ1 — does the heuristic's advantage survive an adversary that adapts?**
   The exploitability gap, J1 → the strongest rung reached. Gate B.
2. **RQ2 — is adversary capability monotone in adversarial pressure?**
   📏 **No.** J2 (a parked beam) beats J3B (per-step best response) and J3
   (greedy) — 41.8 % against 42.2 % and 44.5 %. Committing beats re-optimising.
   ⚠️ Smoke-measured at one seed; the 5-seed CUDA re-run is the next thing that
   happens. [`results/j_ladder.md`](results/j_ladder.md).
3. **RQ3 — does adversarial co-training produce robustness or opponent-overfit?**
   J4, a learned jammer with an opponent pool, then the full cross-product.
   🔒 The **off-diagonal** is the result. ⛔ J4 is **not built**.
4. **RQ4 — does it survive the airframe?** ONNX → TensorRT on a Jetson Orin
   Nano. Latency, p99 jitter, power — and *does quantisation degrade
   coordination more than control?* Gate C. 🔧 Pure Python.

⚠️ **The velocity action space is no longer a contribution.** Gate A resolved
against it: 📏 the speed-cap pathology vanished (26.1 % → 0.4 %) and boundary
occupancy quadrupled (14.5 % → 72.6 %) at a cost of 18.3 pp with disjoint seed
ranges. It ships off, as a negative result with a mechanism —
[`results/gate_a.md`](results/gate_a.md).

---

## Where things are

| Path | What | State |
|---|---|---|
| `data/frankfurt_box.npz` | buildings + road graph as tensors. **Committed on purpose** — it IS the environment | 🔒 frozen |
| `src/env/occlusion.py` | batched segment-vs-**oriented**-box (slab method), 2.5D. `torch.compile`d | ✅ inherited, unchanged |
| `src/env/channel.py` | path loss by link class, SINR, Shannon with a modulation cap | ✅ inherited, unchanged |
| `src/env/core.py::_beam_gain_db` | the **directional jammer**, 3GPP TR 38.901 element pattern. J0–J3B | ✅ own |
| `src/env/routing.py` | hop-limited widest-path DP, `min(C_i)/min(n, 3)` | ✅ inherited, unchanged |
| `src/env/energy.py` | rotary-wing propulsion power (U-shaped), radio DC draw | ✅ inherited, unchanged |
| `src/env/reward.py` | mission term + potential-based shaping. Carries `PHI_V2` | ⚠️ inherited, **to reduce** |
| `src/env/core.py` | the batched env, leading `num_envs` dimension. THE training path | ⚠️ inherited, **to reduce** |
| `src/baselines/b0.py` | the scripted baseline — the comparison everything is measured against | ✅ inherited |
| `src/baselines/evaluate.py` | **the one rollout harness.** Every number goes through it | ✅ inherited |
| `src/models/` | MLP / DeepSets / GNN actors, one shared critic. Plain `nn.Module`s | ✅ de-skrl'd |
| `src/training/ppo.py` | **the trainer.** One-file MAPPO: shared actor, centralized critic | ✅ own, validated |
| `src/training/probe.py` | known-optimum probe whose optimum **spans the episode** | ✅ own |
| `scripts/train.py` | the only entry point into the trainer. Φ flags are derived | ✅ own |
| `src/training/curriculum.py` | fixed step-count schedule, a pure function of training progress | ✅ inherited |
| `src/viz/` | scene drawing, episode figures and videos | ✅ inherited |
| `scripts/` | offline data prep, measurement, evaluation | ✅ inherited |
| `docs/inherited/` | predecessor documentation, **read-only history** | 📚 |

✅ **The trainer landed 2026-08-30** — `docs/REDUCTION.md` task 5. skrl is gone
from `pyproject.toml`; `src/training/ppo.py` replaces it in ~380 lines with no
framework. Validated against the inherited number before anything was changed —
[`results/trainer_validation.md`](results/trainer_validation.md) carries the gate,
declared before the runs, and the result.

⚠️ Three things in it are 🔒 and each failed **silently** before: the GAE mask is
`terminated | truncated`, the truncation bootstrap reads `extras["final_state"]`
(not what `step()` returns), and the swarm is **one** parameter-shared agent over
`num_envs * N` rows. 📏 `src/training/probe.py` demonstrably catches the first —
32.7 → 12.0 against a known optimum of 33.0 — which the predecessor's per-step
probe did not.

---

## Read before you change things

| File | Read it when |
|---|---|
| [`PLAN.md`](PLAN.md) | **start here** — the claim, the phases, the gates declared before the runs |
| [`docs/REDUCTION.md`](docs/REDUCTION.md) | **second** — what was carried over that still has to come out, in order |
| [`docs/INHERITED.md`](docs/INHERITED.md) | quoting any constant. Every measured number that carries forward, with provenance |
| [`docs/inherited/DECISIONS.md`](docs/inherited/DECISIONS.md) | before proposing anything — every entry was proposed then killed on evidence |
| [`docs/inherited/PHYSICS.md`](docs/inherited/PHYSICS.md) | touching channel / routing / energy / scenario parameters |
| [`docs/inherited/REWARD.md`](docs/inherited/REWARD.md) | touching the reward or its weights. Carries the `Φ` audit |
| [`docs/inherited/BLOCK_B.md`](docs/inherited/BLOCK_B.md) | consuming `frankfurt_box.npz`, or touching geometry/routes |
| [`docs/inherited/BLOCK_G.md`](docs/inherited/BLOCK_G.md) | the record of six pre-declared interventions and why each failed |

---

## Hard rules

**Device.** Training tensors live on `cuda:0`. **Never call `.cpu()`, `.numpy()`
or `.item()` inside env `step()` or the training hot loop** — `.item()` forces a
GPU sync and is the easy one to miss. Local dev is Apple Silicon (CPU/MPS), toy
configs only. Guard device selection; never silently degrade a real run to CPU.

**Batched core, thin adapter.** The env core carries a leading `num_envs`
dimension. 📏 3.17 M env-steps/s for occlusion at `num_envs = 1024` on an
RTX 5090; 75 k env-steps/s end-to-end with a learner attached, so a 10 M-step run
costs ~2.2 minutes. **Compute is not a constraint** — `num_envs` is chosen on
*learning* grounds.

**Geometry offline.** `osmnx`/`shapely` are CPU-and-NumPy-only, used **only** in
`scripts/prep_osm.py` to bake buildings into a tensor. Runtime occlusion is
vectorized segment-vs-**oriented**-box in pure torch. Axis-aligned boxes were
measured and rejected — they fill 94 % of the Frankfurt box.

**B0 sees only `obs["flat"]`.** The scripted baseline is a pure function of the
same tensor the actors consume, plus its own carried state — it never reads
`env.hvt_pos`. That is what makes "does learning earn its keep?" a question about
*control* rather than about information, and it is asserted by a test that runs a
decoy scenario underneath a replay (`src/baselines/test_b0.py`). 📏 The measured
cost of the restriction is **0.6 pp**. Do not relax it to make a number look
better.

**Formulas are traceable.** Do not change path-loss / SINR / capacity / energy
formulas without updating the hand-computed tests and checking the cited
standard. They appear in the methodology chapter.

**Multi-seed.** >= 5 seeds for anything reported as a finding. Median + IQR,
never mean ± std — RL returns are not normally distributed. Never report single
runs, and **judge on the worst seed** (this project has been misled twice by
medians).

**Declare the gate before the run.** A rule invented after the fact is not a
rule. Two claims in the predecessor project were overturned by reading a single
run after the fact.

**Results are committed, checkpoints are not.** `results/` is tracked and `runs/`
is gitignored, on purpose: the summary is a result, the checkpoints are
regenerable. The numbers stay versioned with the commit that produced them.

---

## ⛔ Never do these

Each is inherited and each was **measured**, not assumed — full reasoning in
[`docs/inherited/DECISIONS.md`](docs/inherited/DECISIONS.md).

- ⛔ 📏 **Reintroduce transmit power as an action.** Three framings, three nulls.
  Action space is motion only; Ptx is fixed at 30 dBm. Applies to the **jammer**
  too: a barrage jammer with no cost always plays maximum, so the power axis has
  a degenerate optimum. Steer the beam instead.
- ⛔ 📏 **Make the jammer's beamwidth an action.** It smuggles the power axis back
  in — a wide beam *is* a barrage jammer and a narrow one is a spotlight.
- ⛔ 📏 **Raise the altitude ceiling above 80 m.** It is where the scenario stops
  being a swarm problem: a best-placed *single* drone is mission-capable 3.3 % of
  the time at 80 m and 57 % at 120 m. Raising it also *weakens* the study —
  A2A blockage falls 31 % → 25 % → 10 % at 80 / 120 / 180 m.
- ⛔ 📏 **Lower the rate requirement toward 5 Mbps.** At 5 the link never binds:
  the chain carries 8× the bar and `mission_capable` collapses onto `observed`.
- ⛔ 📏 **Raise the Ptx ceiling.** At 40 dBm a *blocked* A2A link carries 15 Mbps
  over 2.8 km — one drone spans the map and the relay chain becomes pointless.
- ⛔ 🔒 **Move to mmWave.** It makes blockage trivial (mmWave is textbook
  blockage-limited), needs beam-pointing modelling coupled to the motion policy,
  and is the wrong band for the tactical MANET radios Ptx is derived from.
- ⛔ 📏 **Terminate the episode on mission failure.** The policy learns never to
  acquire, and a random initial policy never reaches the tracking phase.
- ⛔ 🔒 **Train at more than one `N`.** It turns the zero-shot transfer columns
  into in-distribution tests.
- ⛔ 🔒 **Use adaptive curriculum advancement in a reported run.** It hands easier
  conditions more experience and confounds the comparison unrecoverably.
  `curriculum.weights()` is a pure function of training progress so that it
  *cannot* see the condition.
- ⛔ 🔒 **Compare a number measured on one device with one measured on another.**
  `torch.Generator` streams differ per device, so the same seed draws *different
  episodes* on MPS than on CPU. The physics is identical; the sample is not.
- ⛔ 🔒 **Cite constants an AI produced.** `TODO(verify)` markers in `channel.py`
  and `energy.py` mean exactly that. 📏 TR 36.777 is 3/4 closed; **the NLoS
  intercept `32.4` is not**, and it happens to equal TR 38.901's *terrestrial*
  UMi LoS intercept — the neighbouring constant a transcription slip lands on.
  One human reading of the UMi-AV table closes it. **Everything downstream rests
  on this.**
- ⛔ 🔒 **Add heavy dependencies** (sim engines, RL frameworks) without flagging.
  📏 The one that was here produced four silent bugs and has been **removed**;
  `src/training/ppo.py` is the replacement. Adding skrl, RLlib or
  stable-baselines3 back is re-acquiring all four — see `docs/REDUCTION.md`
  task 5.

---

## Build / test

```bash
uv sync --extra dev                # `dev` is an EXTRA -- plain `uv sync` gives
                                   # you neither pytest nor ruff
uv run pytest                      # 363 passed (+ 7 CUDA-gated, 4 skip on arm64)
uv run ruff check . && uv run ruff format .
```

Offline data prep (needs network; the artefact is committed, so this is only for
regenerating it deliberately):

```bash
uv run python scripts/prep_osm.py --plot          # bake data/frankfurt_box.npz
uv run python scripts/measure_sightlines.py --plot
```

**Look at the map before trusting it.** 📏 Every geometry bug in the predecessor
project was found by happening to compute the right statistic, and one render of
route 12 overturned four hypotheses drawn from aggregate statistics:

```bash
uv run python scripts/view_episode.py --worst --polygons   # boxes vs footprints
uv run python scripts/render_episode.py --compare --route 12
uv run python scripts/bench_occlusion.py                   # re-run this on CUDA
```

Measurement — regenerates the numbers the design rests on:

```bash
uv run python scripts/measure_envelope.py    # altitude band, sensor envelope, link budget
uv run python scripts/eval_baseline.py       # every B0 number, 5 seeds
uv run python scripts/calibrate_r.py --seeds 8 --num-envs 64 --device mps
uv run python scripts/measure_potential.py --policy b0 --device mps --bank-dir .banks
```

⚠️ **`--device mps` is ~17× on Apple silicon** and the physics is identical — but
it draws *different episodes* for the same seed, so never mix devices within a
comparison.
