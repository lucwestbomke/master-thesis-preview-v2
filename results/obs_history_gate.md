# Frame stacking — declared 2026-09-02, **before the runs**

`--obs-history k`, one flag, 5 seeds against 5 seeds. `EnvConfig.obs_history`
ships at **1** and every existing number was measured there.

## Why this, and why it is not intervention #9

📏 [`b0_ablation.md`](b0_ablation.md) decomposed B0's +10.1 pp design advantage at
5 seeds × 128 paired episodes:

| component | paired median | seeds won |
|---|---|---|
| **local link repair** | **+6.90 pp** | 5/5 |
| ranked roles + belief | +3.39 pp | 5/5 |
| target information (oracle) | +0.35 pp | 3/5 |

🔍 `_update_repair` is a **gradient-free hill climb** carrying `prev_score` and
`lat_dir` — *"slides perpendicular to the chain and keeps going while the worst of
those improves, reversing when it does not"*. That is **one step** of search
state. A policy that sees the previous frame can form the same difference; one
that sees only the current frame cannot.

⛔ **This is not the memory [`memory_horizon.md`](memory_horizon.md) closed.**
That file ruled out *target* memory on two independent arguments — perfect target
state is worth −0.4 pp, and the gaps a belief would bridge are p50 35 / p90 320
steps with the target ~85 m stale on arrival. **Search memory is a different
quantity**: horizon 1 step, not bounded by the oracle (the oracle supplies target
state, not "which way did I move and did it help"), and worth up to 6.9 pp.

⛔ **And it is not a reward intervention.** [`credit_assignment.md`](credit_assignment.md)
closed the reward axis structurally. This changes the observation, not the
objective; `Var_i(A)` is untouched.

## Condition

Everything fixed except the flag: `gnn` / `deep` cadence, F4, **J1**, curriculum
on, 12 M env-steps, `value_clip` 0.2, lr 3e-4, `initial_log_std` −0.5, N = 5,
5 seeds, eval split. Control is `obs_history=1`; the treatment is `k = 2` and
nothing else.

🔒 **`k = 2`, not more.** The state B0 carries is one step deep. A larger `k`
widens the trunk and confounds "history helped" with "more parameters helped" —
📏 a `k = 2` gnn already carries more parameters than the control, which is a
confound this gate accepts as small and a `k = 8` gate would not.

## 🔒 Decision rule

Primary readout is **`mission_capable`**, unusually for this project, because the
mechanism predicts a change in the *outcome* rather than in a specific behavioural
statistic — link repair improves chain quality, and chain quality is what
`mission_capable` measures once a sightline is held.

| | rule |
|---|---|
| **promote** | median `mission_capable` rises by **≥ 3 pp** *and* the worst seed does not regress. 3 pp is a little under half the 6.9 pp the mechanism targets, which is the most that should be expected from supplying the *input* to a behaviour rather than the behaviour |
| **kill** | median moves **< 1 pp**, or any seed collapses. The one-step-memory hypothesis is then closed and ⛔ no larger `k` is tried — a longer history cannot supply a state that is one step deep |
| **inconclusive** | 1–3 pp. Report and do not build on it |

📏 Secondary, reported either way: **`chain_occluded`** and **`capacity_p5`** are
what link repair actually acts on (it hill-climbs on clearance / bottleneck
capacity), so they should move *before* `mission_capable` does. ⚠️ If
`mission_capable` rises and neither of those moves, the gain is not the declared
mechanism and must not be reported as it.

## ⚠️ Declared risk: this may buy nothing, for a reason already measured

🔒 `_update_repair` is gated on `is_relay = (rank > 0) & (rank <= n_relay)` —
**only relays hill-climb**. So the behaviour this flag supplies the *input* for is
**role-conditional**, and [`credit_assignment.md`](credit_assignment.md) measured
role signal at **0.04–0.16 %** of the advantage. A policy that cannot tell which
drone it is may be unable to use the history even when it has it.

⛔ **Recorded before the run so a null cannot later be explained away as "we
should have known".** If this kills, that is a *second* measured consequence of
the credit-assignment finding, and it strengthens that result rather than
producing a loose end.

⚠️ **And a feedforward policy is not barred from the behaviour without history.**
B0 hill-climbs because it is gradient-free and cannot see the map; a learned
policy could instead learn the direct local-geometry → direction map, which needs
no memory and would be strictly better. 📏 So a null here is evidence about
*optimisation*, not about representation.

## Ceiling, declared now

📏 The full measured budget — repair 6.9 + roles 3.4 + belief ~0 = **~10.3 pp** —
against a **15.0 pp** gap to B0 at J3B. ⛔ Even a complete success here does not
close the gap, and this gate must not be read as a route to beating B0.
`PLAN.md` §3 is why.

## Commands

```bash
for s in 0 1 2 3 4; do
  uv run python scripts/train.py --arch gnn --cadence deep --obs-history 2 \
      --device cuda:0 --seeds $s --tag fs-k2 --out-root runs
done
uv run python scripts/eval_policy.py runs/fs-k2-s*/checkpoint.pt \
    --group "gnn/deep k=2" --obs-history 2 --device cuda:0 --seeds 5 --num-envs 128 \
    --out results/obs_history.jsonl
```

Control is the existing `rq2-gnn-deep` cell (40.7 % [36.0–43.9], eval split).

## Results

*(appended below; the declaration is not edited)*
