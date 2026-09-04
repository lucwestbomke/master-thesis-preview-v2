# ⛔ Exploitability is not a cost of capability. It is a cost of **adaptivity**.

**Measured 2026-09-04**, `results/frontier.jsonl`, cuda:0, eval split, stage 4,
F4, 5 seeds × 128 episodes. Eight policies spanning 10.7 – 57.3 % capability,
each scored at **J1** and at **J3B**; the gap is paired per seed.

## Why it was run

[`gate_b.md`](gate_b.md) confirmed that B0 degrades more than a co-trained policy,
then found that the gap tracked **chain length** across all four policies tested.
The `capable_no_division` control refuted the rate-division half of that confound
but not the other half — *"a longer chain has more links for the beam to find"* —
and that half was itself confounded with B0 simply being **better at the task**.

The proposed reading was a **frontier**: *exploitability is a cost of capability,
because holding the target needs a forward observer, a forward observer needs a
long chain, and a long chain is what an adversary attacks.* This run was to fit
that frontier over eight points before declaring it.

## 📏 The frontier does not exist

| policy | capability (J1) | **gap J1 → J3B** | worst–best | hops |
|---|---|---|---|---|
| random | 10.7 % | 2.05 pp | [1.85, 2.51] | 0.41 |
| mlp | 29.3 % | 8.60 pp | [6.03, 9.26] | 0.90 |
| gnn co-trained @ J3B | 39.3 % | 7.29 pp | [6.50, 9.06] | 1.09 |
| gnn @ J1 | 39.9 % | 11.12 pp | [10.11, 13.58] | 1.20 |
| gnn co-trained @ J2 | 40.2 % | 7.51 pp | [7.12, 10.90] | 1.11 |
| deepsets @ J1 | 40.5 % | 10.45 pp | [9.70, 12.70] | 1.16 |
| **`b0-geodesic`** | **45.6 %** | **6.39 pp** | **[5.68, 6.54]** | **2.00** |
| **B0** | **57.3 %** | **13.24 pp** | [11.42, 13.58] | 2.13 |

⛔ **Not monotone, and the counterexample is decisive.** `b0-geodesic` is **more
capable than every learned policy** (45.6 % against 29–40 %) and **less
exploitable than all of them** (6.39 pp against 7.29–11.12) — while running
**2.00-hop chains**, essentially B0's 2.13.

🔒 So capability does not determine exploitability, and **neither does chain
length**: geodesic and B0 have the same chain length and geodesic has *half* the
gap. Both explanations `gate_b.md` was weighing are refuted by one row.

## 🔍 The controlled comparison, and what it isolates

`b0-geodesic` and `B0` are the **same scripted family, same code path, same
stations, same chain length**. `VARIANTS` differ by exactly three things —
ranked roles, the target-belief filter, and **local link repair**:

| | capability | hops | `observed` | **gap** |
|---|---|---|---|---|
| `b0-geodesic` | 45.6 % | 2.00 | 85.5 % | **6.39 pp** [5.68 – 6.54] |
| `B0` | 57.3 % | 2.13 | 92.8 % | **13.24 pp** [11.42 – 13.58] |

✅ **Ranges are disjoint** — geodesic's worst-case best is 6.54, B0's best is
11.42.

📏 [`b0_ablation.md`](b0_ablation.md) already priced those three additions:
**link repair +6.90 pp** of capability, ranked roles +3.39, belief ~0. The
exploitability difference is **13.24 − 6.39 = 6.85 pp**.

☠️ **Link repair buys ~6.9 pp of capability and costs ~6.9 pp of exploitability.
Almost exactly one for one.**

## 🔍 The mechanism, and it is a named subroutine

`b0.py::_update_repair` is *"one step of a 1-D hill climb on observable
clearance… it slides perpendicular to the chain and keeps going while the worst of
those improves, reversing when it does not."*

**It is a closed feedback loop on precisely the quantity the jammer attacks.**
The beam degrades a link → repair hill-climbs to recover it → J3B re-optimises
against the new geometry → repair chases again. The adversary is not merely
degrading a signal; **it has a control input into the defender's behaviour.**

`b0-geodesic` returns from `_update_repair` immediately. It flies to stations
fixed by index. There is no loop to drive, and 📏 it loses 6.39 pp where B0 loses
13.24.

🔍 **And the rest of the table follows the same variable.** `random` adapts to
nothing and has the smallest gap of all (2.05 pp). The learned policies are
reactive but weakly so, and sit between. **Co-training moves them down** — 11.12
→ 7.51 and 10.45 → 7.29 — which is what adversarial training is supposed to do to
an exploitable loop, and is now RQ3's *mechanism* rather than an effect with no
explanation.

## ⚠️ What this does NOT say

⛔ **Geodesic is not the better policy.** At J3B, B0 still scores **43.8 %**
against geodesic's **39.7 %** — it wins by 4.1 pp *under the strongest adversary
that exists*. Adaptivity is worth having on net: it buys 11.7 pp and costs 6.85.
The exploitability is a **real, priced cost of the loop**, not a reason to remove
it.

⛔ **This is not about scripted versus learned**, and that framing should now be
dropped entirely. The two extremes of the exploitability range are **both
scripted** (geodesic 6.39, B0 13.24) and every learned policy sits between them.
The variable is whether the policy runs a closed loop on the jammed quantity — and
if it does, whether that loop was trained against an adversary.

⚠️ **One environment, one adversary family, N = 5.** The mechanism is
mechanistically clean and measured on a controlled pair, but it is one task.

## 🔒 What this replaces

⛔ The frontier claim — *"exploitability is a cost of capability"* — is **refuted
before it was declared**, which is the whole reason it was fitted over eight
points before being written down rather than after four.

✅ The claim it becomes:

> **Exploitability is a cost of *adaptivity*, not of capability.** A policy that
> closes a feedback loop on the quantity an adversary attacks hands that adversary
> a control input. The loop is worth paying for — it buys more capability than it
> costs — but the cost is real, measurable, and separable from the capability it
> buys.

📏 Supported by a within-family controlled pair with disjoint ranges, priced
against an independent ablation of the same subroutine, and consistent with all
eight points including both extremes being scripted.
