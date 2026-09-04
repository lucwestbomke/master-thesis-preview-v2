# Results

**Tracked on purpose.** `results/` is in git and `runs/` is not: the summary is a
result, the checkpoints are regenerable. That keeps every number versioned with
the commit that produced it, so provenance cannot drift from the code.

One JSONL row per scored configuration, written by `scripts/eval_policy.py` and
`scripts/eval_baseline.py` via `--out`. ⚠️ Rows record **per-seed** values, not
just `median [IQR]` — every gate in `PLAN.md` is declared on the *worst* seed, and
a median alone cannot be judged against one.

⚠️ Guard against double-counting. An interrupted-and-resumed sweep in the
predecessor project appended eval rows unconditionally and reported `n = 9` where
5 were asked for. Check `n` before quoting anything.

The predecessor's results are **not** carried over; the old repository keeps them,
and the numbers that matter are consolidated in `docs/INHERITED.md`.

---

## The record, in the order it was made

Every file below declares its decision rule **before** the run that resolved it,
and the declaration is never edited afterwards — results are appended under it.

| file | question | verdict |
|---|---|---|
| [`trainer_validation.md`](trainer_validation.md) | does the own-PPO trainer reproduce the inherited number? | ✅ and it found the **frozen critic** |
| [`rq2_ladder.md`](rq2_ladder.md) | MLP → DeepSets → GNN, re-run on the fixed trainer | +9.1 pp disjoint, then **+0.9 pp null** |
| [`gate_a.md`](gate_a.md) | are the pathologies artefacts of the action space? | ⛔ **not met** — velocity setpoints cost 18.3 pp |
| [`phi_v2_gate.md`](phi_v2_gate.md) | does a steeper potential move the observer? | ⛔ **killed** — 11.8 m of a 130 m gap |
| [`credit_assignment.md`](credit_assignment.md) | can the advantage tell one drone from another? | ⛔ **0.04–0.16 %** of return variance. Closes the reward axis *structurally* |
| [`memory_horizon.md`](memory_horizon.md) | would memory help, and over what horizon? | ⛔ perfect target state is worth **−0.4 pp**. Do not rebuild recurrence |
| [`b0_ablation.md`](b0_ablation.md) | what is B0's +10.1 pp actually made of? | 📏 link repair **+6.9**, ranked roles **+3.4**, belief ~0 |
| [`j_ladder.md`](j_ladder.md) | is the adversary ladder monotone, and where is its power? | 📏 monotone; **directionality −10.6 pp vs adaptivity −2.9 pp** |
| [`obs_history_gate.md`](obs_history_gate.md) | does one step of history buy link repair? | ⚠️ **inconclusive** — +1.94 pp, worst seed −1.25 |
| [`gate_b.md`](gate_b.md) | **is the heuristic more exploitable than a co-trained policy?** | ✅ **CONFIRMED** — 13.24 pp vs 7.51 / 7.29. ⚠️ carries an unresolved **chain-length confound** |

⚠️ **Read `gate_b.md` to the end.** Its verdict stands as declared, but a competing
explanation was found afterwards and the control for it has not been run.
