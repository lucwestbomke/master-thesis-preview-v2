# Memory is not the deficit. ⛔ Do not rebuild recurrence.

**Declared and measured 2026-09-02.** `scripts/measure_memory_horizon.py` carries
the hypothesis and the three-way decision rule, written before the numbers
existed. `tests/test_memory_horizon.py` pins the run-length logic.

## Why this was run

📏 The BC probe ([`../scripts/bc_init.py`](../scripts/bc_init.py)) showed a
memoryless clone of B0 scoring **9.4 %** against B0's **58.0 %**, and B0 carries
`belief_rel` / `belief_vel` — a target-belief filter with dead reckoning — that
the feedforward actors cannot represent. That suggested a structural
disadvantage: **B0 remembers where the target was; the learned policy cannot,
because `obs["flat"]` zeroes `rel_hvt * sees` when a drone cannot see it.**

Recurrence was tried once (G8, 2026-08-25) and killed at −1.05 pp. ⚠️ But that
test ran on a trainer whose **critic could freeze** — a defect not diagnosed
until 2026-09-01, and the same defect for which
[`rq2_ladder.md`](rq2_ladder.md) re-ran the *entire* 45-run architecture grid.
Recurrence was not re-run; it was deleted (`REDUCTION` task 4). Its failure
signature also matches a noisy critic: the mean moved −1.05 pp while the seed IQR
**widened** 4.7 → 6.9. Plus `BLOCK_G` records a second untested defect —
`grad_norm_clip` applied jointly to policy and value parameters, flagged as
*"real, plausible under a GRU, never tested."*

So the re-proposal was methodologically legitimate. The question was whether the
prize justified rebuilding sequence-aware rollout storage, BPTT and cross-boundary
hidden state — the highest bug-density change available in this repo.

---

## ☠️ The prize does not exist, and the number was already in the repo

📏 [`docs/inherited/BLOCK_E.md`](../docs/inherited/BLOCK_E.md), 5 seeds, eval split:

| policy | mission-capable | observed |
|---|---|---|
| `B0-geodesic` (roles fixed by index, **no belief filter**) | 47.1 % [3.1] | 86.0 % |
| **`B0`** (ranked roles + belief filter + spare posts + link repair) | **57.2 % [3.5]** | 93.0 % |
| `B0-oracle` (**+ ground-truth target state**) | 56.8 % [3.2] | 93.9 % |

🔒 **`B0` → `B0-oracle` is −0.4 pp.** BLOCK_E's own words: *"Perfect knowledge of
the target's position is worth nothing measurable in this task."*

⛔ **That is a hard upper bound on memory.** A belief filter, a GRU, or any
recurrent state can at best *reconstruct an estimate* of target state. The oracle
*is* target state, exactly and for free. **Memory cannot be worth more than the
oracle, and the oracle is worth −0.4 pp.**

🔍 **So B0's +10.1 pp over `B0-geodesic` is not its memory.** The bundle also
contains **ranked roles from the sees bits**, spare observation posts and local
link repair — and `B0-geodesic` already has a belief-free version of the rest.
Dynamic *role assignment* is the plausible carrier of that 10.1 pp, which is
exactly the capability [`credit_assignment.md`](credit_assignment.md) measured the
learned policy as structurally unable to express.

---

## 📏 And the independent measurement agrees

`scripts/measure_memory_horizon.py`, MPS, eval split, stage 4, F4/J1, 64 envs ×
600 steps, 3 seeds. Unseen runs **preceded by a sighting** — the only intervals a
belief could bridge — with the target's displacement over each.

| policy | per-drone sees | swarm observed | gap p50 | gap p90 | covered by k=8 (gaps / **blind time**) | hvt moved p50 |
|---|---|---|---|---|---|---|
| **B0** | 27–29 % | 90–94 % | **30–36** | **302–330** | 21 % / **0.9 %** | **82–89 m** |
| GNN | 10–14 % | 26–31 % | 31–41 | 444–478 | 15 % / **0.6 %** | 88–102 m |
| random | 5.6 % | 22 % | 50–65 | 403–420 | 18 % / **0.5 %** | 111–126 m |

Against the declared three-way rule:

| branch | rule | measured |
|---|---|---|
| frame stacking | p90 ≤ 8 steps **and** most blind time short | ⛔ p90 is **302–330**, and k=8 covers **0.9 %** of blind time |
| a filter is justified | long gaps **and** *small* displacement | ⛔ displacement p50 is **82–89 m**, against B0's own 89 m stand-off and the 127 m sightline median |
| **memory is not the deficit** | long gaps **and** large displacement | ✅ **this one** |

🔒 **Both branches that would have justified building something fail, and the
branch that says stop is the one the data lands on.** Frame stacking covers under
1 % of blind time, so the cheap option is dead; and a belief is ~85 m stale by the
time the gap closes — roughly the entire stand-off distance — so the expensive
option is aimed at information that has decayed.

⚠️ **One caveat, in the honest direction.** Raw displacement is an *upper bound*
on a dead-reckoning filter's error: B0 extrapolates with `belief_vel`, and a
target on a road graph is partly predictable. So a good filter would be less stale
than 85 m. That weakens this measurement on its own — but it cannot rescue
recurrence, because the oracle bound above is not a proxy: it *is* perfect target
state, and it is worth −0.4 pp.

---

## The verdict, and a correction

⛔ **Do not rebuild recurrence for target memory.** Two independent arguments, one
of them a hard bound:

1. Perfect target information is worth **−0.4 pp** (5 seeds, eval split). Memory
   cannot exceed it.
2. Gaps are long (p50 ~35 steps, p90 ~320) and the belief is ~85 m stale on
   arrival, so neither frame stacking nor a filter reaches the horizon that would
   matter.

⚠️ **This retracts the case made earlier the same day.** The frozen-critic and
`grad_norm_clip` confounds in the G8 recurrence test are real, and the argument
that recurrence was killed on defective evidence still stands *as an argument
about method*. It is simply aimed at a prize the oracle bound shows is not there.
📏 Re-running recurrence on the fixed trainer would be measuring, at 5 seeds, a
quantity bounded above by 0.4 pp.

⚠️ **It also corrects the BC probe's attribution.** `bc_init.py` reports the
clone's failure as partly structural — B0's belief is not in the student's input.
That remains true about *action prediction*: B0's action depends on belief, so a
memoryless student mispredicts it. But since belief is worth ~0 in **reward**,
the 9.4 % collapse is better explained by **compounding error / covariate shift**
than by missing memory. 🔍 Which means **DAgger would plausibly fix the clone**,
where the earlier reading implied nothing could.

## What survives, and what it points at

🔍 **One hypothesis about memory is untouched by the oracle bound**, and it should
be recorded rather than quietly inherited: the oracle bounds *memory for target
tracking*, not **memory for role commitment** — a recurrent policy remembering
*"I am the observer"* across steps. That is a different use, it is not target
information, and the oracle has nothing to say about it.

⛔ But it is **not** the hypothesis that motivated this measurement ("B0 has
memory and the policy does not"), it is entangled with
[`credit_assignment.md`](credit_assignment.md)'s finding that role signal is
0.04–0.16 % of the advantage, and G8's `observer_tenure` did not move (36.8
against a required 95). Anyone pursuing it needs a new gate, declared before its
own run, and should read the credit measurement first.

📏 **Where the evidence actually points:** B0's advantage is not memory — the
belief filter accounts for at most 0.4 pp of the +10.1 pp.

⛔ **CORRECTED 2026-09-02, same day:** this section originally attributed the
remaining ~10 pp to **role assignment**. That was inference, not measurement, and
[`b0_ablation.md`](b0_ablation.md) measured it: **local link repair is +6.9 pp
and ranked roles are +3.4 pp** (5/5 paired seeds each). Roles are a *precondition*
for repair — `_update_repair` is gated on `is_relay` — so they are load-bearing
for ~10.3 of the 11.0 pp, but they only *score* 3.4.

⚠️ That file also reopens a question this one closed, for a different mechanism:
B0's hill climb carries **one step** of search state (`prev_score`, `lat_dir`),
which the oracle bound does not cover and which `k = 2` frame stacking supplies.
🔒 The verdict here stands as written — it closed **target** memory, at a p50 of
35 steps and a p90 of 320, and that closure is intact.

## Regenerate

```bash
uv run python scripts/measure_memory_horizon.py --policy b0 --device mps --num-envs 64 --seeds 3
uv run python scripts/measure_memory_horizon.py --policy runs/val-gnn-deep-s0/checkpoint.pt \
    --device mps --num-envs 64 --seeds 3
```

⚠️ `torch.Generator` streams differ per device. Raw output:
[`memory_horizon.jsonl`](memory_horizon.jsonl).
