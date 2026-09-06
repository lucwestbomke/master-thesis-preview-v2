# Glossary

**Written 2026-09-06**, with the restructure recorded in [`../PLAN.md`](../PLAN.md)
§6. One definition each, and they are used consistently everywhere else in this
repository.

⚠️ **This file defines terms; it does not carry results.** Every number lives in
[`../results/`](../results/) or in `PLAN.md`, and `results/` is never edited.

📏 measured · 🔒 constraint · 🔧 provisional — see [`../AGENTS.md`](../AGENTS.md).

---

| term | definition |
|---|---|
| **adaptation** | the controller repositioning in response to what it senses. In this project: `_update_repair` |
| **loop amplitude** | how far the loop is willing to reposition. `repair_amplitude_m` |
| **loop target** | the quantity the loop scores — clearance or capacity. `B0Config.repair_score` |
| **exploitability** | capability lost between the isotropic emitter (J1) and the strongest adversary (J3B), in pp of `mission_capable` |
| **damage** | capability lost because the link crosses the threshold, with no behavioural response involved |
| **exploitation** | capability lost because the adversary induced the controller to move somewhere worse |
| **adversarial co-training** | training the swarm with an emitter present |
| **off-diagonal** | train against emitter A, evaluate against emitter B. Distinguishes robustness from opponent-overfit |
| **dose–response** | one factor varied across >= 3 levels, everything else held fixed. ⚠️ **Amended 2026-09-06** — see the note below |
| **disjoint** | seed ranges that do not overlap. This project's bar for a claim |
| **coordination** (RQ3) | role structure and observer geometry — `role_entropy`, `observer_range_m` |
| **control** (RQ3) | basic flight competence and on-station behaviour — `mission_capable`. ⚠️ Collides with *control* (experimental); see the note below |
| **control** (experimental) | the arm a treatment is measured against. `capable_no_division` is a control; Gate A is a methodological one |

## The words the methodology rests on

| term | definition |
|---|---|
| **gate** | a question with its **decision rule declared before the run**, whose branches partition the outcome space and are never edited afterwards. ⛔ A rule invented after the fact is not a rule |
| **instrument** | work done to make a measurement *possible*, not to answer a question. The capability programme is one — `PLAN.md` §3 |
| **arm** | a condition in a comparison, like for like with its control: same architecture, same cadence, same seeds |
| **probe** | a policy or measurement that answers *"is the signal there?"* but is **not** like-for-like, so it cannot stand as an arm. A teacher-initialised policy from `bc_init.py` is a probe |
| **worst seed** | the judging statistic for every gate here, at >= 5 seeds. 🔒 Medians have misled this project twice |

## The units

| term | definition |
|---|---|
| **`mission_capable`** | the fraction of steps the swarm both observes the target and relays it end-to-end at >= 15 Mbps. The project's headline metric |
| **capability** | `mission_capable` on the rung under discussion. Unqualified, it means at **J1** |
| **pp** | percentage points — an *absolute* difference between two percentages, never a relative one |
| **headroom** / **capacity margin** | how far the chain's bottleneck sits above the 15 Mbps bar. 🔍 Small headroom is what makes a gap **damage** rather than **exploitation** |

## Named things, defined elsewhere

⛔ **Not redefined here**, so there is one authority for each:

| | where |
|---|---|
| **B0**, **`b0-geodesic`**, **random** | `PLAN.md` §1 — the scripted family and what separates its members |
| **J0 – J4** | `PLAN.md` §4 — the adversary ladder, one rung per row |
| **Gates A – G, D2** | `PLAN.md` §5, and [`../results/capability_gates.md`](../results/capability_gates.md) for D/E/F/G's full declarations |

---

## ⚠️ "Vulnerability" is not a term in this project

It was being used interchangeably with **exploitability**, and that is a defect:
exploitability is a **measured quantity with a definition** — the J1 → J3B gap in
pp of `mission_capable` — while "vulnerability" is a mood. ⛔ **Write
exploitability.** This note is the only place the other word appears.

## 🔒 Why `damage` and `exploitation` are separate words

📏 `PLAN.md` §1 decomposes the gap as `f(threshold proximity) + g(loop amplitude)`.
The first term is **damage** and the second is **exploitation**, and they are not
interchangeable: every learned policy in this project sits within **1.1 Mbps of
the 15 Mbps threshold**, where each dB the jammer removes crosses the bar with no
behavioural response at all. ⛔ A gap measured on such a policy is mostly damage,
which is why a capable learned policy is a **prerequisite** for testing RQ1 on
learned controllers (`PLAN.md` §3, the instrument).

## ⚠️ `control` means two things, and both are load-bearing

📏 **RQ3's sense** — *coordination versus control* — is frozen in Gate C's
declaration (`PLAN.md` §5), which is reproduced verbatim and cannot be edited:
*"Coordination is then more quantisation-sensitive than control."* 🔒 **The
experimental sense** — a control arm — is the one every gate in §5 uses.

⛔ **So the collision cannot be removed by renaming**, because renaming the RQ3
sense would desynchronise this file from a declaration that is not editable. ✅ It
is removed by **always qualifying**: write *"control (RQ3)"* or *"a control arm"*,
never a bare *"control"* where either could be read. ⚠️ Gate A's row in §5 said
*"control (closed)"* and now says *"method (closed)"* for exactly this reason.

## ⚠️ `dose–response`, amended 2026-09-06

**It read:** *"one factor varied across >= 3 levels with the effect monotone in
it."*

⛔ **That bakes the finding into the definition.** Monotonicity is what
`repair_amplitude_m` **measured** — 7.94 → 13.24 pp across 0 / 50 / 100 / 200 m —
not what makes a design a dose–response. Defining it in means *"the dose–response
was confirmed"* cannot fail, which is exactly the circularity this project's gates
exist to prevent.

✅ **It now reads:** one factor varied across >= 3 levels, everything else held
fixed. **Monotonicity is the result**, reported separately, and a non-monotone
outcome is a finding rather than a failed definition.

## 🔒 Why the RQ numbers in `results/` do not match `PLAN.md`

⛔ `results/` records rules declared before runs and is **never edited**, so it
predates the 2026-09-06 renumbering. A file there that says *"RQ3"* for
co-training means what `PLAN.md` §2 now calls **RQ2**, and
[`../results/rq2_ladder.md`](../results/rq2_ladder.md) keeps the predecessor
project's numbering entirely — its "RQ2" is the **architecture ladder**.
**Read the question, not the number.** `PLAN.md` §2 carries the old → new mapping.
