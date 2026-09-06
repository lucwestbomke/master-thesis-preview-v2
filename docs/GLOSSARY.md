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
| **dose–response** | one factor varied across >= 3 levels with the effect monotone in it |
| **disjoint** | seed ranges that do not overlap. This project's bar for a claim |
| **coordination** (RQ3) | role structure and observer geometry |
| **control** (RQ3) | basic flight competence and on-station behaviour |

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

## 🔒 Why the RQ numbers in `results/` do not match `PLAN.md`

⛔ `results/` records rules declared before runs and is **never edited**, so it
predates the 2026-09-06 renumbering. A file there that says *"RQ3"* for
co-training means what `PLAN.md` §2 now calls **RQ2**, and
[`../results/rq2_ladder.md`](../results/rq2_ladder.md) keeps the predecessor
project's numbering entirely — its "RQ2" is the **architecture ladder**.
**Read the question, not the number.** `PLAN.md` §2 carries the old → new mapping.
