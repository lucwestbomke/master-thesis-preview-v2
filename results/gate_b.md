# Gate B — ✅ **CONFIRMED.** Measured 2026-09-03, CUDA, eval split, 5 seeds.

Rule declared **2026-08-27**, reproduced verbatim in [`../PLAN.md`](../PLAN.md) §5
and unchanged since:

> **confirm** — B0's exploitability gap exceeds the adversarially-trained
> policy's, at 5 seeds, on the worst seed.
> **report** — the full cross-product, not just the diagonal.

📏 `results/gate_b_crossproduct.jsonl`, `cuda:0`, eval split, stage 4, F4,
N = 5, 5 seeds × 128 episodes. B0's rows come from
[`j_ladder.jsonl`](j_ladder.jsonl), same device and harness.

---

## 📏 The exploitability gap, J1 → J3B

Paired within each policy: the same seed draws the same episodes at both rungs.

| policy | J1 | J3B | **gap** | worst | range | relative |
|---|---|---|---|---|---|---|
| **B0** | 57.3 % | 43.8 % | **13.24 pp** | 11.42 | [11.42 – 13.58] | **23.1 %** |
| gnn/deep, trained at J1 | 39.9 % | 28.7 % | 11.12 pp | 10.11 | [10.11 – 13.58] | 27.9 % |
| **gnn/deep, trained at J2** | 40.2 % | 31.8 % | **7.51 pp** | 7.12 | [7.12 – 10.90] | **18.7 %** |
| **gnn/deep, trained at J3B** | 39.3 % | 30.9 % | **7.29 pp** | 6.50 | [6.50 – 9.06] | **18.6 %** |

✅ **B0's gap exceeds both adversarially-trained policies' on every reading the
gate and its amendments require:**

| | B0 vs advtrain-J2 | B0 vs advtrain-J3B |
|---|---|---|
| absolute, median | 13.24 > **7.51** ✅ | 13.24 > **7.29** ✅ |
| absolute, **worst seed** | 11.42 > **7.12** ✅ | 11.42 > **6.50** ✅ |
| relative | 23.1 % > **18.7 %** ✅ | 23.1 % > **18.6 %** ✅ |
| ranges disjoint | ✅ (by 0.52 pp) | ✅ (by 2.36 pp) |

🔒 **Amendment 2 required both normalisations to AGREE, and they do.** That
amendment was written on 2026-09-02 *after* seeing the two disagree on the
J1-trained control, precisely so the verdict could not be chosen after the fact.
On the adversarially-trained policies the disagreement disappears.

⚠️ **advtrain-J2's disjointness is marginal — 0.52 pp**, on one seed
(B0's worst 11.42 against advtrain-J2's best 10.90). `advtrain-J3B` is the
cleaner confirm at 2.36 pp. ⛔ Do not quote the J2 row alone.

⚠️ **The seed semantics differ between the B0 and learned rows**, and this is a
real caveat rather than a formality. B0 is deterministic, so its five seeds vary
only the **evaluation episodes**; the learned rows' five seeds are **training
runs** and carry both sources. "Worst seed" therefore does not mean the same
thing in the two blocks.

---

## 📏 The cross-product — and the off-diagonal is the result

`mission_capable`, median over 5 seeds. Rows = trained at, columns = evaluated at.

| trained at ↓ / evaluated at → | **J1** | **J2** | **J3B** |
|---|---|---|---|
| J1 (the control) | 39.9 % | 30.0 % | 28.7 % |
| **J2** | **40.2 %** | **34.3 %** | **31.8 %** |
| J3B | 39.3 % | 33.1 % | 30.9 % |

### ✅ This is robustness, not opponent-overfitting

🔒 Gate B declared the off-diagonal as the readout because *"a policy robust only
to the adversary it trained against has overfitted to one opponent, and the
off-diagonal is the only place that shows."*

📏 **At J3B — an adversary it has never seen — `advtrain-J2` scores 31.8 %,
BEATING `advtrain-J3B`'s 30.9 % on its own training opponent.** The same holds at
J2. **`advtrain-J2` dominates the entire table.** The diagonal is not the best
cell anywhere, which is the opposite of the overfitting signature.

### 🔍 And the secondary finding rhymes with RQ2

Training against the **weaker but committed** adversary (J2, a parked beam)
generalises *better* than training against the **stronger but re-optimising** one
(J3B). 📏 RQ2 measured the same asymmetry on the attack side — a parked beam
forces a persistent detour while a per-step best response hands a hill-climber a
moving problem. **The same property that makes an adversary damaging makes it a
better teacher**: a consistent pressure is learnable, a re-optimising one is a
moving target.

⚠️ 🔧 That is a *hypothesis with one supporting measurement*, not a result. It
would need its own gate — two rungs is not a trend.

### ✅ Adversarial training is free on the clean rung

| trained at | J1 score | worst |
|---|---|---|
| J1 (control) | 39.9 % | 38.4 % |
| **J2** | **40.2 %** | 38.0 % |
| J3B | 39.3 % | 37.2 % |

📏 **+0.3 pp**, well inside the seed range. There is **no robustness/performance
trade-off here** — the usual cost of adversarial training does not appear.

---

## ⛔ What this does NOT show

📏 The gap to B0 by rung:

| rung | B0 | control | gap | advtrain-J2 | gap |
|---|---|---|---|---|---|
| J1 | 57.3 % | 39.9 % | 17.4 | 40.2 % | **17.2** |
| J2 | 46.7 % | 30.0 % | 16.7 | 34.3 % | **12.4** |
| J3B | 43.8 % | 28.7 % | 15.0 | 31.8 % | **11.9** |

⛔ **B0 still wins by 11.9 pp under the strongest adversary.** Adversarial
training closes 3.1 pp of the 15.0 pp gap, and the remainder is not closing —
consistent with the ~10.3 pp design-component budget measured in
[`b0_ablation.md`](b0_ablation.md) and with `PLAN.md` §3.

🔒 **The claim Gate B confirms is about the DERIVATIVE, not the level.** B0 is
more capable and *more exploitable*; the adversarially-trained policy is less
capable and *more robust*. That is the thesis's claim and it is now measured. It
is **not** a claim that learned control beats the heuristic, and must not be
written up as one.

---

## What is still owed

⚠️ **J4 does not exist.** Gate B's declaration allows "J3 at minimum, J4 if it
trains", so this is a valid Gate B — but the strongest adversary reached is a
*scripted* one. A learned jammer could find a response the scripted rungs do not,
and the gap could look different against it. RQ3 in `PLAN.md` §2.

⚠️ **One architecture, one cadence.** `gnn`/`deep` only. The result is not known
to hold for DeepSets or MLP.

⛔ **Two rungs of adversarial training.** "Committed adversaries are better
teachers" rests on J2 vs J3B alone.
