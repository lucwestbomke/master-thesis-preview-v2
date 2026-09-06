# Capability programme — working log

🔧 The dated log [`docs/CAPABILITY_BRIEF.md`](../docs/CAPABILITY_BRIEF.md) §8 asks
for: configuration, seeds, diagnostics, and what was concluded.

🔒 **Exploration runs go here and nowhere else** (brief §1). A number in this file
is **not** a result unless it is also written into a gated file under a rule
declared before its run. Confirmation runs live in
[`trainer_benchmark.md`](trainer_benchmark.md) and
[`capability_gates.md`](capability_gates.md).

⚠️ **Every run in this file is CPU on Apple silicon.** There is no CUDA on this
machine. ⛔ Nothing here is comparable with any number in `results/`.

---

## 2026-09-06 — Phase 0 (trainer validation), Phase 1 (instrumentation), §7 (export)

### What was done

| brief section | state | where |
|---|---|---|
| §2 **Phase 0** — validate the trainer against published references | ✅ **done, VALIDATED** | [`trainer_benchmark.md`](trainer_benchmark.md) |
| §3 **Phase 1** — the missing instrumentation | ✅ **done** | `src/training/ppo.py`, `scripts/train.py` |
| §7 — *"run that export locally this week, before this programme starts"* | ✅ **done, and it found something** | [`../scripts/export_onnx.py`](../scripts/export_onnx.py) |
| §4 **Phase 2** — exploration | ⛔ **not started.** Needs CUDA; see "What is blocked" |
| §5 **Phase 3** — confirmation | ⛔ not started |
| §6 — behaviour-cloning fallback | ⛔ not started |

---

### Phase 0 — ✅ VALIDATED

📏 Full declaration, rule and numbers in [`trainer_benchmark.md`](trainer_benchmark.md).
In one line: **`pendulum` median −164.56 against a published tuned PPO reference of
−172.23, and `lqr` worst-seed cost ratio 1.154 against an exactly-computed
optimum.** Both PASS. A `lr = 0` negative control scores −1167.09 and 96.2×.

🔒 The brief's stop condition — *"if it does not reach reference performance, stop
and fix the trainer"* — **does not fire.** Phases 1–3 are unblocked, and the eight
prior nulls plus Gates D and E keep their readings.

⭐ **The one finding that changes an instruction in the brief**: `grad_kept` is not
a health check. Both PASSing tasks run at `grad_kept` **0.035–0.238**, far below
the brief's `> 0.8`, and the task that fails has the *highest* `grad_kept` in the
suite. ⛔ **Block A's acceptance test as written would reject both configurations
that reach reference performance.** Full argument and the mechanism in
`trainer_benchmark.md`; the recommendation is to gate Block A on `approx_kl` and
`clip_fraction` instead, which is a change to a **declaration that has not been
made yet** and therefore costs nothing.

---

### Phase 1 — instrumentation

Added to `src/training/ppo.py`, all read-only, no RNG consumed, no parameter
touched. 📏 `uv run pytest`: **465 passed, 4 skipped** — 460 before, plus the five that pin
the new columns.

| brief §3 asks for | state before | state now |
|---|---|---|
| `approx_kl` | ✅ present | unchanged |
| `grad_kept` | ⚠️ **present and correct** — see below | unchanged, and now read |
| `clip_fraction` | ⛔ absent | ✅ `((ratio − 1).abs() > ratio_clip).mean()`, per minibatch |
| `explained_variance` | ✅ present | unchanged |
| `sigma` **per action dimension** | ⛔ only the mean, as `log_std` | ✅ `sigma_x` / `sigma_y` / `sigma_z`, after the `min_log_std` floor and the `max_log_std` clamp |
| action saturation **per dimension** | ⛔ absent | ✅ `sat_x` / `sat_y` / `sat_z` / `sat_any`, on the **sampled** action |
| boundary occupancy | ✅ present as `at_boundary` | unchanged |
| observer tenure | ⛔ existed only in `evaluate.py`, i.e. only *after* a run | ✅ `observer_run`, via the new `MissionDiagnostics`. ⚠️ **A proxy, not comparable with the 294.7** |
| **total Adam steps** | ⛔ had to be computed by hand | ✅ `adam_steps`, a column in every log line |

`scripts/train.py`'s printed `WATCH` table was reordered to put optimisation
before behaviour, per the brief's 🔒 *"watch `approx_kl` and `grad_kept` before
`mission_capable`"*. ⚠️ It is the **printed** subset only — `log.jsonl` still
carries every column, so `scripts/inspect_run.py` loses nothing.

#### ☠️ A correction to the brief's §0: `grad_kept` was never NaN in this code

The brief says *"currently NaN in every log in `runs/`. Fix it"*. 📏 There was
nothing to fix. `_grad_kept` computes the right thing, and
[`capability_gates.md`](capability_gates.md) §0 already records the correction
(dated 2026-09-06): the NaN belongs to logs written **before** the diagnostic
existed, and the real CUDA run reads **0.54–0.91**. It read 0.152–0.238 and
0.035–0.040 on the two benchmark tasks here, on the first attempt, with no change
to the function. ⚠️ The brief's §0 predates that correction and its `grad_kept`
lines should be read as superseded — as should the *"three quarters of the
gradient is discarded"* framing, which came from a 120 k-step MPS toy run.

---

### §7 — the ONNX export, locally. ⭐ It retires one risk and finds another

`uv run python scripts/export_onnx.py`, both exporters × all three rungs,
`onnx 1.22.0` / `onnxruntime 1.29.0` / `onnxscript 0.7.1`, opset 20.

| exporter | rung | exports? | `onnx.checker` | matches PyTorch at 5 / 15 / 24 rows |
|---|---|---|---|---|
| dynamo | mlp | ✅ | ✅ | ✅ 1.3e-07 |
| dynamo | deepsets | ✅ | ✅ | ✅ 1.0e-07 |
| dynamo | **gnn** | ✅ | ✅ | ✅ 9.7e-08 |
| torchscript | mlp | ✅ | ✅ | ✅ 1.3e-07 |
| torchscript | **deepsets** | ✅ | ✅ | ☠️ **FAILS at 15 rows** |
| torchscript | gnn | ✅ | ✅ | ✅ 1.0e-07 |

✅ **`PLAN.md`'s named export risk is retired.** The risk register reads *"PyTorch
Geometric will not export"* — and it does not have to. `RelationalTrunk` is a
custom MPNN in plain torch (`docs/inherited/MODELS.md`'s rule that the DeepSets
rung must be the GNN with `e_ij` zeroed is what forced that), so **PyG is never on
the actor's forward path.** All three rungs export and all three are numerically
exact under the default exporter. ⛔ TensorRT is still unretired; this is the ONNX
half only.

☠️ **And the failure is exactly this project's favourite shape — silent.** Under
the legacy TorchScript exporter the `deepsets` rung exports without error, passes
`onnx.checker.check_model`, runs correctly at the traced batch size, and then
**throws at any other batch size**:

    Concat node '/actor/trunk/Concat_1':
    Non concat axis dimensions must match: Axis 0 has mismatched dimensions of 5 and 15

📏 **Mechanism, confirmed rather than guessed.** The traced graph contains a
`Constant` of shape **`[5, 7, 2]`** — the traced batch, the 7 neighbour slots and
`EDGE_DIM` — and the `gnn` graph contains no such constant. 🔍 It is
`RelationalTrunk.forward`'s `torch.zeros_like(edge)` under `use_edges=False`: the
tracer constant-folds it at the traced shape, so the one line that makes DeepSets
*be* the GNN with `e_ij` zeroed is also the one line that freezes `N`.

⚠️ **Why this matters beyond a stale exporter.** `deepsets` is
`scripts/train.py`'s **default architecture**, and the off-N transfer columns run
the actor alone at `N ∈ {3, 8}`. A deployment built with the legacy exporter would
have been correct at `N = 5` and broken at every other swarm size, with no error
until it ran. 🔒 `scripts/export_onnx.py` checks three row counts for exactly this
reason and exits non-zero on any rung that fails.

🔧 **Dependency flag**, per `AGENTS.md`: `onnx`, `onnxruntime` and `onnxscript`
were added to the **`dev` extra**, not to the runtime dependencies. They are not
simulators and not RL frameworks, and RQ3 cannot proceed without them.

---

### 🔧 Checked and found already in place — no work needed

* **The §5 capacity-margin readout.** The brief asks Phase 3 to *"report median
  end-to-end capacity margin alongside `mission_capable`"*, on the argument that
  every learned policy sits within **1.1 Mbps of the 15 Mbps bar**.
  `src/baselines/evaluate.py` already carries `capacity_mean` and `capacity_p5`, so
  the margin is `capacity_mean − CAPACITY_THRESHOLD_MBPS` and needs no new code.
* **Block A's four flags.** `--orthogonal-init`, `--no-tanh-mean`, per-dimension
  `--min-std` and `--grad-norm-clip` all exist and all ship OFF;
  [`capability_gates.md`](capability_gates.md) records the bit-identity check over
  207,879 parameters that proves they cannot have moved an inherited number. ⛔ So
  Block A is a *run*, not an implementation task.

---

### ⛔ What is blocked, and why

**Phase 2 and Phase 3 cannot run on this machine.** Not a judgement call:

* 📏 Gate F's control is *"Gate E's promoted configuration"*, measured on **CUDA**.
  ⛔ `AGENTS.md`: comparing a number measured on one device with one measured on
  another is on the never-do list, and `torch.Generator` streams differ per device
  so the same seed draws different episodes. A Gate F arm run here would not be
  comparable with its own control.
* 📏 The brief budgets *"~150 k env-steps/s"* and a three-week programme. This
  machine has no CUDA at all.

🔒 **The correct next action is to run Phase 2 on the CUDA host**, with Block A's
acceptance criterion amended per Phase 0's finding, and with Gate F's runbook in
[`capability_gates.md`](capability_gates.md) — which is already written, already
declared, and explicitly says *"do not redesign it"*.

⚠️ **Also worth knowing before that run**: Gate G's checkpoints *"likely already
exist"* under `runs/gateE*/`, and scoring them is cheap. This machine's `runs/`
holds only `bc-gnn-s0`, `stage1-control-s0` and `val-gnn-deep-s0..4` — ⛔ **no
`gateD*` or `gateE*` directories at all**, so those checkpoints are on the CUDA
host and Gate G cannot be closed from here either.

---

### 🔒 The paragraph the brief §8 asks for — **not yet answerable**

The brief ends: *"At the end of three weeks, write one paragraph answering: was
the policy optimisation-limited?"* ⛔ That paragraph is not written here, because
Phase 0 does not answer it and Phases 2–3 have not run. What Phase 0 establishes
is narrower and is the precondition for the question being meaningful at all:
**the optimiser is correct, so a null on the mission is a fact about the mission.**
📏 The evidence that bears directly on the question is still Gate D's REGRESSION
(−32 pp for a 10× budget) and Gate E's NULL, both measured on CUDA and both
recorded in [`capability_gates.md`](capability_gates.md).
