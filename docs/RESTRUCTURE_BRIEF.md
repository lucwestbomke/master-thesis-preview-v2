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

---

## 7. Amendments — resolved 2026-09-06, after the restructure was applied

🔒 **Recorded, not edited in place.** The brief above is unchanged; this section
says where it was wrong and what was done instead. ⛔ Every resolution below
follows the brief's own last line — *a contradiction is more likely to mean the
brief is wrong than the repo is* — and each one was checked against `results/`,
which is unchanged throughout.

### 1. ☠️ §5 asked to keep a status that a measurement had already replaced

**The brief says:** keep *"the ☠️ confounded status of line 3"* of §3's five lines.

**The repo says:** line 3 reads **✅ RESTORED 2026-09-06**. It was raised as
confounded on 2026-09-04 — all eight nulls measured at ~5,900 Adam steps — and
📏 Gate D then tested that confound directly and it went the *other* way: **10x
the gradient budget costs 32 pp**.
[`../results/capability_gates.md`](../results/capability_gates.md).

⛔ **Resolved in the repo's favour, by doing nothing.** The five-line table is
byte-identical to what it was before the restructure. ⚠️ The brief was written
against a snapshot that predates Gate D's verdict; ✅ its *intent* — do not soften
the re-opening — is satisfied, and §3 now says so explicitly.

### 2. ⚠️ §1's verbatim text contains a word §4 forbids

**The brief says both:** copy RQ2 in verbatim, *"If the loop is the
vulnerability…"*; and **"vulnerability" is not a term in this project**, with the
checklist demanding zero occurrences outside the glossary's note.

⛔ **Resolved toward the glossary**, which is the brief's own reasoning: the two
words *"were being used interchangeably and that is a defect"*. `PLAN.md` §2 RQ2
reads **"If the loop is what makes a controller exploitable, can the loop be kept
but trained so that it cannot be led?"** — the same question, in the project's own
vocabulary.

📏 **Worth recording: the repo had zero occurrences to begin with.** The only
place the word appeared was this brief. The defect it names is real but was
already absent from the documents.

### 3. 🔒 The RQ numbers in `results/` are now permanently offset, and that is correct

**The conflict:** §4 asks that the glossary's terms be *"used consistently
everywhere else"*, and §0 forbids editing `results/`. But
[`gate_b.md`](../results/gate_b.md), [`obs_mask_gate.md`](../results/obs_mask_gate.md)
and [`j_ladder.md`](../results/j_ladder.md) say **"RQ3"** for co-training, which
this restructure renamed **RQ2** — and
[`rq2_ladder.md`](../results/rq2_ladder.md) keeps the *predecessor's* numbering
entirely, where "RQ2" is the **architecture ladder**.

⛔ **Not resolvable by renaming, and it should not be.** A file in `results/`
records a rule declared before a run; renumbering it after the fact is exactly the
edit §0 forbids. ✅ **Resolved by a pointer instead**, in three places — `PLAN.md`
§2, `AGENTS.md`, and [`GLOSSARY.md`](GLOSSARY.md): **read the question, not the
number.**

⚠️ **This is a permanent property of the repo now**, not a defect to be cleaned up
later. Any future renumbering inherits it.

### 4. ⚠️ Two of the three time-boxed gates were already closed when the box was written

**The brief says:** *"Gates D–F are budgeted at three weeks"* and *"if Gates D–F
all null"*, as though all three were pending.

**The repo says:** 📏 **D and E are closed and both NULL** — λ null-to-harmful and
10x the budget at −32 pp; per-drone credit over a **12x** weight range inside a
single cell's seed noise. `PLAN.md` §5,
[`capability_gates.md`](../results/capability_gates.md).

✅ **Resolved by keeping the box verbatim and stating what is left under it.** The
box is a *budget*, and a budget survives its first two line items being spent.
`PLAN.md` §3 and §7 item 3 now name **F** as what the box has left.

### 5. ⚠️ "§4 unchanged" could not survive §2's own renumbering

**The brief says** §4 is unchanged. But §4 tagged **J4** as *"RQ3's stretch"* —
the old RQ3, which §2 of the brief renames **RQ2**. Left alone, the tag would
point at deployment.

✅ **Resolved as a cross-reference, not content.** `| **J4** | … | RQ2's stretch |`.
🔒 The rung, the beam pattern, the `θ_3dB = 25°` note and the ⛔ exclusions in §4
are untouched. The same applies to four other stale cross-references in §3, §7,
§8 and §9, each rewritten to name **the J-ladder** or **the architecture ladder**
rather than a number that has moved.

### 6. 🔧 §6's row was given in a different column order than §6's table

**The brief's row:** `framing | reframed 2026-09-06 | superseded by three RQs…`.
**The table's header:** `framing | killed by | when`.

✅ **Resolved by placing the brief's cells to match the header**, and by adding one
note under the table: ⚠️ the new row is a **reframing**, not a refutation — no
measurement killed it, the organisation of the same evidence changed. §6 exists
for framings *"killed by a run designed to test it"*, and this one was not.

### 7. ☠️ Two claims in `AGENTS.md` that the restructure surfaced as wrong

Not contradictions with the brief — contradictions the brief's instruction to
rewrite that section exposed. Both corrected against
[`gate_b.md`](../results/gate_b.md), which is unchanged.

| claim | status |
|---|---|
| *the `capable_no_division` control is declared and **not yet run*** | ⛔ **wrong.** It ran 2026-09-04. 📏 B0's gap moves **13.24 → 12.85 pp** while every learned policy's collapses to **0.44 – 5.46 pp** — the division confound is **refuted**. ⚠️ The other half — *a longer chain has more links for the beam to find* — is **not** closed, and `AGENTS.md` now says so |
| *a per-hop normalisation **reverses** the headline* | ⛔ **dropped.** 📏 No file in `results/` or `PLAN.md` carries it, so under §0 it cannot be restated; and the control it warned about has since run |

### ✅ What the brief got right, and is worth keeping

🔒 **The three-RQ structure, the instrument framing, the time box and the
glossary all stand exactly as written.** Every amendment above is a stale
cross-reference or a wording collision — ⛔ **none of them touches the framing
itself**, and none required a change to a number, a rule, a declaration or a
verdict.
