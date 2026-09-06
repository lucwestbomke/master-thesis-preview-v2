# Agent brief — restructure the project around three research questions

**Purpose.** The thesis framing has been settled. This brief tells you exactly what
it is and which files to change. The scientific content does not change; the
*organisation* of it does.

**Read `PLAN.md`, `docs/INHERITED.md` and the index of `results/` before editing
anything.**

---

## 0. Hard constraints — read these first

⛔ **Never edit anything in `results/`.** Every file there records a rule declared
before a run. Declarations, amendments and verdicts are evidence. You may *link*
to them and *quote* them; you may not change a number, a rule or a verdict.

⛔ **Never delete a refuted framing.** `PLAN.md` §6 exists because a refuted claim
is part of the evidence for the one that replaced it. This restructure is itself a
framing change and must be recorded there, not applied silently.

⛔ **Do not restate any measured number from memory.** Every figure you write must
be copied from `PLAN.md` or from the `results/` file it came from. If you cannot
find the source for a number, omit it and leave a `TODO(source)` marker.

✅ **Preserve the existing house style**: the emoji markers (📏 measured, 🔒
structural, ⛔ closed, ⚠️ caveat, ⭐ notable, 🔍 mechanism, ☠️ severe, 🔧 tooling),
the tables, and the practice of recording amendments rather than editing in place.

✅ **One commit per file** with a message naming the section changed.

---

## 1. The framing, in full

Copy the following into `PLAN.md` verbatim where §1 and §2 are indicated below.

### Working title

> **The cost of adaptation: measuring and removing exploitability in multi-agent
> relay under directed jamming.**

### Goal

> Show that a controller's own repositioning loop is what makes it exploitable,
> measure how that cost scales, and demonstrate that adversarial co-training
> removes it.

### RQ1 — Does adaptation create exploitability, and how does it scale?

Do controllers that reposition in response to the attacked quantity lose more
capability to an *aiming* adversary than controllers that do not — and does the
loss scale with how far they are willing to reposition?

- **Status:** supported.
- **Evidence:** the `b0-geodesic` / `B0` controlled pair (disjoint ranges); the
  `repair_amplitude_m` dose–response; the null on the loop's *target*; the
  mechanism named as an explicit chase (degrade → repair → re-optimise → chase).
- **Supporting result:** the J-ladder decomposition of adversary power
  (directionality vs adaptivity). This is **no longer its own RQ** — it justifies
  the adversary ladder and belongs inside RQ1.
- **Open:** replication of the dose–response in a second, minimal environment, so
  the claim is not tied to one map. Requires **no training** — it runs on scripted
  controllers.

### RQ2 — Can adversarial co-training remove the cost without losing capability?

If the loop is the vulnerability, can the loop be kept but trained so that it
cannot be led?

- **Status:** effect measured, mechanism untested.
- **Evidence:** both learned policies move down the exploitability axis at
  unchanged capability and no cost on the clean rung; the off-diagonal shows
  robustness rather than opponent-overfit.
- **Open:** the redundancy metric (candidate mechanism: a second
  threshold-clearing, edge-disjoint path; computable from the capacity matrix
  `routing.py` already builds); and **J4**, a learned jammer, to show the result
  holds against a stronger adversary than a scripted one.

### RQ3 — Does it survive deployment?

ONNX → TensorRT on the Jetson Orin Nano: latency, p99 jitter and power against the
400 ms control period. Then: **does quantisation degrade coordination more than it
degrades control?**

- **Status:** not started. The export risk is unretired.
- **Coordination** = role structure and observer geometry (`role_entropy`,
  `observer_range_m`). **Control** = basic competence (`mission_capable`,
  on-station behaviour).
- Gate C's declaration already covers this and is **reproduced verbatim, not
  rewritten**.

---

## 2. Mapping from the old objectives

| old | new home |
|---|---|
| RQ1 — is exploitability a cost of adaptivity | **RQ1**, unchanged in substance |
| RQ2 — where does an adversary's power come from | **folded into RQ1** as a supporting result. Keep every number and the link to its results file; demote the heading |
| RQ3 — does co-training reduce exploitability | **RQ2** |
| RQ4 — does it survive the airframe | **RQ3** |
| §3 the capability question | **no longer an RQ.** Becomes §3 "The instrument", see below |

---

## 3. §3 becomes "The instrument", not a research question

This is the most important change in the restructure, so make it explicitly.

Rewrite the §3 heading and opening to state that the capability programme is an
**instrument**, not an objective. The justification is already in the document and
must be preserved: every learned policy sits within **1.1 Mbps of the 15 Mbps
threshold**, where the adversary's *damage* term dominates and no behavioural
response is being measured — so a capable learned policy is a **prerequisite for
testing RQ1 on learned controllers**, and nothing more.

Add explicitly:

> 🔒 **The thesis does not depend on this section.** RQ1's evidence is a *scripted*
> controlled pair. If Gates D–F all null, RQ1, RQ2 and RQ3 stand unchanged and the
> paper reports that the scripted baseline remains the strongest policy.

Add a **time box**:

> 🔒 **Gates D–F are budgeted at three weeks.** At the end of that budget the
> programme stops regardless of outcome and writing begins. If no capable learned
> policy exists by then, `scripts/bc_init.py` is used to manufacture one for RQ1's
> learned-loop arm. ⚠️ A teacher-initialised policy remains a **probe, not an arm**
> — it is not a like-for-like comparison for the architecture ladder or for Gate B,
> and any result using it must say so.

Keep §3's five-line table exactly as it is, including the ☠️ confounded status of
line 3. Keep the named optimiser confound and its arithmetic. **Do not soften the
re-opening** — it is correct.

Gate D's instrument list already names `--gae-lambda` as never swept in this
project's history, with the effective-horizon arithmetic. Keep that; it is the
highest-prior single knob.

---

## 4. Per-file changes

### `PLAN.md`

1. **§1** — keep the claim and all its evidence. Add the working title and the
   one-sentence goal at the top, before the claim.
2. **§2** — replace the four objectives with the three RQs from §1 of this brief.
   Preserve every existing link into `results/`. Where an old RQ is demoted, keep
   its numbers under its new parent rather than dropping them.
3. **§3** — retitle to "The instrument" and apply §3 of this brief.
4. **§4** — unchanged.
5. **§5** — unchanged content, but retag each gate with the RQ it now serves:
   Gate A (control, closed), Gate B (RQ1), Gate C (RQ3), Gates D/E/F (instrument).
6. **§6** — **append a row** recording this restructure:
   `Four objectives with the capability programme as RQ | reframed 2026-09-06 | superseded by three RQs; the capability programme is an instrument, not an objective`
   Do not remove any existing row.
7. **§7** — reorder the roadmap to:
   - ONNX export **locally**, this week (the only deliverable that cannot fail)
   - the redundancy metric (half a day, no training)
   - Gates D–F, three-week box
   - the second-environment replication for RQ1
   - exposé
   - then J4, the Orin work, longer runs
8. **§8** — update the `n = 1 environment` row. It currently says "not resolvable
   within this project". Replace with: a minimal abstract relay environment
   reproducing the dose–response is the resolution, it needs no learned policy, and
   it converts the objection into an external-validity argument.
9. **§9** — unchanged, except the "bigger map / second city" entry, which should now
   point at the abstract-twin plan in §8 rather than reading as a flat exclusion.

### `docs/` and `README`

- Update any file that enumerates the four objectives to the three RQs.
- Do not rewrite `docs/INHERITED.md`. It records what was already known.
- If a doc's framing is now stale, add a dated note at the top pointing at the new
  §2 rather than rewriting the body.

### New file: `docs/GLOSSARY.md`

Create it, with these definitions. They must be used consistently everywhere else.

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
| **dose–response** | one factor varied across ≥3 levels with the effect monotone in it |
| **disjoint** | seed ranges that do not overlap. This project's bar for a claim |
| **coordination** (RQ3) | role structure and observer geometry |
| **control** (RQ3) | basic flight competence and on-station behaviour |

⚠️ **"Vulnerability" is not a term in this project.** Search the repo for it and
replace every occurrence with **exploitability**. They were being used
interchangeably and that is a defect.

---

## 5. What must NOT change

- Any number, rule, declaration, amendment or verdict in `results/`.
- Gate C's declaration — reproduced verbatim, never edited.
- §3's five lines, including the ☠️ confounded status of line 3 and the ✅ status of
  lines 1, 2, 4 and the ⚠️ of line 5.
- §6's existing rows.
- The exclusions in §7 and §9 — agent index, role embedding, transmit power as an
  action, recurrence, wider networks. Each was excluded on measured grounds.
- The rule that gates are judged on the **worst seed** at ≥5 seeds.
- The bar in `results/capability_gates.md`: *clears B0* = median above 57.3 %,
  *beats B0* = worst seed above 60.6 %.

---

## 6. Verification checklist

After editing, confirm and report:

- [ ] `PLAN.md` §2 lists exactly three RQs; the old RQ2's numbers survive under RQ1
- [ ] §3 is titled as an instrument, carries the three-week box and the explicit
      statement that the thesis does not depend on it
- [ ] §6 has one new row and no removed rows
- [ ] `git diff --stat` shows **no changes under `results/`**
- [ ] `docs/GLOSSARY.md` exists and every term in it is used consistently
- [ ] zero occurrences of "vulnerability" outside the glossary's own note
- [ ] every number in the edited text traces to `PLAN.md` or a `results/` file; any
      that does not is marked `TODO(source)`
- [ ] every link that existed before still resolves

Report anything in the existing documents that **contradicts** this brief rather
than resolving it yourself. A contradiction is more likely to mean the brief is
wrong than the repo is.
