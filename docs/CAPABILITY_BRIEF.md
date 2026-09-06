# Agent brief — the capability programme

**Purpose.** Produce a learned policy with real capacity headroom, so that RQ1's
loop claim can be tested on learned controllers and not only scripted ones.

**This is an instrument, not a research question.** The thesis stands if it fails.
Read `RESTRUCTURE_BRIEF.md` §3 before starting, and read
`results/capability_gates.md` for the declarations that already exist.

**Budget: three weeks.** At the end, stop regardless of outcome.

---

## 0. The diagnosis this brief acts on

📏 Every number in `results/` was measured under an optimisation budget that was
frozen and never examined:

- **5,888 Adam steps** total, on a 137 k-parameter actor
- `approx_kl` at **0.002–0.004** against PPO's usual 0.01–0.02 — about **0.14 nats
  of total policy movement, end to end**
- `grad_kept` at **0.20–0.26**: three quarters of the surviving gradient is
  discarded by the norm clip
- **no orthogonal initialisation anywhere in the project's history**
- `λ = 0.95` against an observer tenure of **294.7 steps** — the advantage sees
  **6 %** of the behaviour it is meant to credit
- the learning rate is recorded as **deliberately never swept**

🔍 The hypothesis is that the policy never left its initialisation, and that the
eight prior nulls are therefore confounded rather than refuted. This programme
tests that.

⚠️ **Compute is not the constraint.** At ~150 k env-steps/s a 1 B-step run is under
two hours. The shipped 12 M-step runs are ~80 seconds of simulation. Budget
generously.

---

## 1. Exploration and confirmation are separated

⚠️ This project's standard is that a rule is declared before a run and judged on the
worst of ≥5 seeds. That standard is for **claims**. It is the wrong instrument for
**tuning**, and applying it to every knob is part of why the budget was never
examined.

So, explicitly:

- **Phase 2 is exploration.** One or two seeds, no pre-declared rule, read the
  diagnostics not the headline metric. Nothing from this phase may be cited as a
  result.
- **Phase 3 is confirmation.** Declared rule, ≥5 seeds, judged on the worst seed,
  written into `results/` under the existing standard.

🔒 Do not report a Phase 2 number anywhere outside the working log.

---

## 2. Phase 0 — validate the trainer. Two days. Do this first.

The custom PPO in `src/training/ppo.py` has never been checked against a known
reference. Until it is, a flat learning curve cannot be attributed to the task
rather than the trainer.

1. Run the trainer unchanged on 2–3 standard continuous-control benchmarks
   (e.g. a Gym/Brax locomotion task and a simple pendulum-class task).
2. Compare final return against published PPO baselines for the same task and step
   budget.
3. Record in `results/trainer_benchmark.md`: task, steps, seeds, return,
   published reference, and the diagnostics from §3.

**If it does not reach reference performance, stop and fix the trainer.** Nothing
downstream is interpretable. Suspect, in order: advantage normalisation, the
value-loss clip, the GAE bootstrap at truncation vs termination, and observation
normalisation.

⭐ This also has standalone value for the thesis. "A PPO implementation written
after finding four silent bugs in a published library, validated against reference
benchmarks" is a methods appendix, and the four bugs in
`docs/REDUCTION.md` are worth stating.

---

## 3. Phase 1 — instrumentation. Half a day.

Fix and log these every update. Several are currently missing or broken.

| metric | why | healthy |
|---|---|---|
| `approx_kl` | policy movement per update | 0.01–0.02 |
| `grad_kept` | ☠️ **currently NaN in every log in `runs/`.** Fix it | > 0.8 |
| `clip_fraction` | how much of the ratio is being clipped | 0.05–0.20 |
| `explained_variance` | critic fit | already 0.94–0.98; watch it does not fall |
| `σ` **per action dimension** | exploration, separately on x / y / z | x,y not collapsing; z free to shrink |
| action saturation rate per dim | 📏 B0 saturates ≥1 axis on 32.6 % of steps | comparable to B0 |
| boundary occupancy | 📏 learned 15–23 % vs B0's 0.9 % | falling toward B0 |
| observer tenure | 📏 B0 = 294.7 steps, learned = 47 | rising |
| total Adam steps | the confound itself | log it explicitly |

🔒 **Watch `approx_kl` and `grad_kept` before `mission_capable`.** If the policy is
not moving, the headline metric cannot tell you anything.

---

## 4. Phase 2 — exploration. One week, one or two seeds.

### Block A — restore the PPO defaults. Apply together, not one at a time.

These are not tuning. They are restoring standard practice that was never applied,
and separating them wastes the budget.

| flag | value | source |
|---|---|---|
| `--orthogonal-init` | on, `head_gain = 0.01` | absent from the entire project history |
| `--tanh-mean` | **off** | B0 saturates an axis on 32.6 % of steps; a tanh mean cannot express those actions |
| `--min-log-std` | per-dimension, ≈ `-1.6 -1.6 -3.0` | 📏 mean \|a_z\| = 0.006, sd 0.053 — z has nothing to explore and must be allowed to collapse; x/y must not |
| `--grad-norm-clip` | raise until `grad_kept` > 0.8 | currently discarding ~75 % |

⚠️ `min_log_std` is `persistent=False` for checkpoint compatibility. Do not change
that.

**Acceptance for Block A:** `grad_kept` > 0.8 and `approx_kl` above 0.005. If not,
the clip or the learning rate is still binding.

### Block B — the two real axes. This is Gate D's existing 2 × 4 factorial.

`{shipped budget, new budget} × λ ∈ {0.95, 0.98, 0.99, 0.995}`, already declared in
`results/capability_gates.md`. Do not redesign it.

**New budget** means: minibatch **well below** the shipped 40,960 rows, more epochs,
and enough total steps that Adam steps land in the 10⁵–10⁶ range rather than 10³.
Target `approx_kl` in 0.01–0.02 by adjusting learning rate and epochs.

📏 λ is the highest-prior single knob and has never been swept. At `γ = 0.997,
λ = 0.95` the effective horizon is **18.9 steps**; `λ = 0.99` gives 77 and
`λ = 0.995` gives 126, against an observer tenure of 295.

⛔ **Do not sweep the reward weights.** §3 line 4 closes that structurally: team
terms cancel from `Var_i(A)` by construction. Nothing here is a shaping knob.

### Block C — only if A and B leave capability short.

In order, one at a time:

1. **`--cue-mode bearing`.** 📏 A one-line cue-follower scores **81.6 %** at
   curriculum stage 1 (0.94× B0) and **6.1 %** at stage 4 — below random. The stage
   is solvable degenerately. A bearing cannot be servoed to a point, so the
   shortcut stops being expressible while acquisition survives. Cheap, structural,
   and the highest-prior arm of Gate F.
2. **`--curriculum-boundaries`** — shorten stage 1. Same target, weaker instrument.
3. **`--w-difference`** (Gate E) and **`PHI_V2` / `--w-cover`**. ⚠️ Only after the
   policy demonstrably trains. Both were designed against a real structural
   deficit — Φ is a function of one drone and constant in the other four — but
   neither is interpretable on a policy that does not move. Gate E's acceptance
   test is `probe_credit.py`'s **advantage** column, not its value column.

---

## 5. Phase 3 — confirmation. One week, ≥5 seeds.

Take the single best configuration from Phase 2. Declare the rule in
`results/capability_gates.md` **before running**, in the existing format, with every
branch partitioning the outcome space.

🔒 The bar is already declared and is not relaxed:
*clears B0* = median above **57.3 %**; *beats B0* = worst seed above **60.6 %**.

⭐ **The threshold that actually matters for the thesis is lower.** RQ1 needs a
learned policy with **capacity headroom**, not one that beats B0. Every current
learned policy sits within **1.1 Mbps of the 15 Mbps bar**, which is why they
measure damage rather than exploitation. Report median end-to-end capacity margin
alongside `mission_capable`, and treat a policy with ≥3 Mbps of headroom as
sufficient for RQ1's learned arm **even if it does not clear B0**.

### If Gate D promotes

📏 Eight prior nulls, Gate A, the Φ v2 gate and the k = 2 gate were all measured at
~5,900 Adam steps. Each needs a one-line restatement — *"measured under a 10×
smaller optimisation budget"* — and the cheap ones (Φ v2, k = 2) should be re-run
before the freeze. Record this in `PLAN.md` §3, not by editing the results files.

### If Gate D nulls

That is a real result and it makes `results/credit_assignment.md` **stronger**: the
optimisation confound is removed and the structural finding survives. Write it up
that way.

---

## 6. Fallback — behaviour cloning from B0

If Phases 2 and 3 do not produce a policy with headroom, use `scripts/bc_init.py`,
which exists and has never been reported.

- Clone B0, then fine-tune with PPO.
- ⚠️ Clip regression targets to ±0.995 — this is why `tanh_mean=False` matters.
- ⚠️ **A teacher-initialised policy is a probe, not an arm.** It is not a
  like-for-like comparison for the architecture ladder (RQ2's old ladder) or for
  Gate B, and every result using it must say so.
- ✅ For RQ1's learned-loop arm it is perfectly adequate: the requirement is
  capacity headroom, not provenance.
- "How much does RL improve over the heuristic it was initialised from" is itself a
  measurable, reportable quantity.

---

## 7. Constraints

- ⛔ **No agent index or role embedding.** Excluded by decision 2026-09-04. Roles
  must emerge; B0 is granted roles-from-index as a documented advantage and the
  learned arm is not.
- ⛔ **No transmit power or beamwidth as actions.** Three framings, three nulls.
- ⛔ **No recurrence.** Killed on its own pre-declared rule at 5 seeds.
- ⛔ **No wider or deeper networks.** 📏 Architecture measured at ±1 pp across three
  rungs; the budget binds long before the width does.
- ⚠️ **Do not break ONNX export.** RQ3 depends on it and PyTorch Geometric may not
  export at all. Any architecture change must be re-checked against the export
  path — and run that export locally **this week**, before this programme starts.
- ⚠️ Do not collapse `ACTION_DIM` to 2 on the z-axis evidence alone. It breaks every
  checkpoint. Only if a sweep shows `σ_z` wants to be ~0.

---

## 8. Reporting

Keep a dated working log at `results/capability_log.md`: configuration, seeds,
diagnostics from §3, and what was concluded. Exploration runs go here and **nowhere
else**.

At the end of three weeks, write one paragraph answering: **was the policy
optimisation-limited?** That paragraph goes in the thesis whichever way it reads.
