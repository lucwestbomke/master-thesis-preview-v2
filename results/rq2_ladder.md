# RQ2 — the architecture ladder, re-run. Declared 2026-09-01, before the runs

## Why it is being re-run at all

📏 The frozen-critic defect (`trainer_validation.md`) was live for every
inherited learned-policy number. MLP **31.2 %**, DeepSets **38.1 %**, GNN
**41.2 %**, and the finding built on them — *"MLP → DeepSets +6.9 pp, robust;
DeepSets → GNN +0.4 pp, a null"* — were all measured through a trainer in which
the critic's gradient could go structurally zero and roughly one seed in five
collapsed. Fixing it moved the GNN 41.1 → 44.5 % and the seed IQR 3.9 → 0.1.

⚠️ Those measurements are **not wrong** — each was run correctly against the
trainer of its day. They are simply no longer the best estimate, and RQ2 is a
comparison *between* architectures, so a defect that hits them unequally changes
the ordering rather than shifting it.

⛔ **The cadence choices are suspect for the same reason.** `base`/`wide`/`deep`
were selected under the same trainer, so re-running only the previous winners
would carry the confound forward. The grid is re-run whole.

## Design

🔒 **One variable.** Shaping stays `shipped` throughout. The predecessor measured
the shaping axis as noise (1.2 pp against 6 pp of seed range); with the seed
range now ~0.1–1.4 that measurement deserves redoing, but **not in the same
experiment as the architecture**. `PHI_V2` is `docs/REDUCTION.md` task 3 and is
a separate decision.

| | |
|---|---|
| grid | 3 architectures × 3 cadences × **5 seeds** = 45 runs |
| architectures | `mlp` (232 hidden), `deepsets` (128), `gnn` (128) |
| cadences | `base` · `wide` · `deep` |
| fixed | F4, curriculum on, 12 M env-steps, lr 3e-4, `value_clip` 0.2 (arm A), `shipped` shaping, `N` = 5 |
| stage A | **train** split, selects each architecture's cadence |
| stage B | **eval** split, winners only, `N` ∈ {3, 5, 8} |

🔒 **The search runs per architecture.** Tuning on the GNN and applying its
winner to the other two is the unequal budget `MODELS.md` rule 2 forbids, and it
is the tempting shortcut because the GNN is ahead. 📏 It matters here: the
predecessor measured the MLP as the only rung preferring `base`, so a deep-only
comparison carries an architecture × cadence interaction.

🔒 **Equal capacity.** Parameter counts are within 20 % by construction and
`src/models/test_actor.py` asserts it, so the contrast cannot become
capacity-vs-capacity.

🔒 **The eval split is touched exactly once**, after cadence selection is final.

⛔ **Train at one `N` only.** `N` ∈ {3, 8} are *zero-shot* evaluations. Training
at more than one `N` turns the transfer columns into in-distribution tests.

## Selection rule — stage A

Each architecture's cadence is the one with the highest **median**
`mission_capable` across 5 seeds on the **train** split, ties broken by the
smaller IQR, ties then broken by the better **worst seed**. Declared before any
result is seen.

## Decision rules — stage B, on the eval split

Judged on **non-overlap of the seed ranges**, not on a p-value. With n = 5 a
significance test is theatre; with the measured IQR of 0.1–1.4 non-overlap is
both achievable and assumption-free.

| contrast | confirmed | null |
|---|---|---|
| **MLP → DeepSets** (permutation invariance) | medians differ by ≥ **2 pp** *and* DeepSets' **worst** seed exceeds the MLP's **best** seed | the ranges overlap |
| **DeepSets → GNN** (relational structure — RQ2's actual claim) | medians differ by ≥ **2 pp** *and* the GNN's **worst** seed exceeds DeepSets' **best** seed | the ranges overlap |

📏 2 pp is chosen against the measured spread, not against the old one. The
inherited effects were +6.9 pp and +0.4 pp on a 6 pp seed range; the same effects
against a 0.1–1.4 range are resolvable several times over.

### The column that decides RQ2, and it has never been measured

⚠️ `BLOCK_G.md`: *"RQ2's informative column is `N` = 8 zero-shot and this sweep
does not touch it."* The MLP is **not size-agnostic** — its neighbour slots are
position-specific weights — so `N` = 8 is where permutation invariance and
relational structure should pay if they pay anywhere. It is free to measure and
it is reported alongside `N` = 5 whatever the N=5 result says.

| | rule |
|---|---|
| **report** | the full `N` ∈ {3, 5, 8} table for all three rungs, against B0 and random at the same sizes |
| **transfer finding** | an architecture whose `N` = 8 score falls by less than another's, on the worst seed |

⛔ Not judged on `hop_mean | observed` (it measures geometry) and not on
`chain_occluded` (it confounds with hop count, `corr = 0.963`).

## Results

*(appended below as they land; the declaration above is not edited)*

## 📏 Stage A — measured 2026-09-01, CUDA, train split, 5 seeds per cell

| architecture | cadence | per seed | median | IQR | worst |
|---|---|---|---|---|---|
| **mlp** | **base** ⬅ | 31.3 · 33.1 · 33.6 · 35.6 · 36.3 | **33.6** | 2.46 | 31.3 |
| mlp | deep | 23.5 · 29.9 · 30.5 · 32.1 · 33.8 | 30.5 | 2.26 | 23.5 |
| mlp | wide | 21.1 · 29.0 · 29.3 · 32.5 · 32.9 | 29.3 | 3.51 | 21.1 |
| **deepsets** | **deep** ⬅ | 41.0 · 43.6 · 44.1 · 44.6 · 48.2 | **44.1** | 1.00 | 41.0 |
| deepsets | wide | 35.6 · 37.8 · 38.2 · 39.9 · 46.1 | 38.2 | 2.09 | 35.6 |
| deepsets | base | 32.4 · 33.4 · 36.7 · 39.5 · 39.8 | 36.7 | 6.17 | 32.4 |
| **gnn** | **deep** ⬅ | 43.9 · 44.4 · 44.5 · 44.5 · 46.3 | **44.5** | **0.14** | **43.9** |
| gnn | wide | 35.5 · 38.0 · 40.8 · 41.0 · 42.3 | 40.8 | 2.97 | 35.5 |
| gnn | base | 32.8 · 35.9 · 36.0 · 38.3 · 40.9 | 36.0 | 2.37 | 32.8 |

**Cadence selection**, by the rule declared above (highest median; ties on IQR,
then worst seed). No cell tied, so the rule resolved without its tiebreakers:
**mlp → `base`, deepsets → `deep`, gnn → `deep`.**

### 🔍 The cadence finding replicates exactly, on a fixed trainer

📏 The predecessor measured the MLP as **the only rung preferring `base`**, with
DeepSets and GNN preferring `deep`, and `wide` losing everywhere. All three
reproduce here. That is an independent replication under a trainer whose critic
does not freeze, and it retires the worry that the cadence grid was an artefact
of the defect.

⚠️ **It also means the deep-only contrast is still confounded.** The 33.6 % MLP
figure is its `base` cell; comparing architectures on `deep` alone would compare
the MLP at its *worst* cadence and inflate the first contrast. Stage B uses each
architecture's own winner, which is what `MODELS.md` rule 2 requires.

### Preview of the contrasts — ⚠️ train split, not the reported number

| contrast | effect | ranges | declared verdict |
|---|---|---|---|
| MLP → DeepSets | **+10.5 pp** | [31.3, 36.3] vs [41.0, 48.2] — **disjoint** | **confirmed** |
| DeepSets → GNN | **+0.4 pp** | [41.0, 48.2] vs [43.9, 46.3] — overlapping | **null** |

🔍 **Both inherited conclusions survive the trainer fix, and the second
replicates to the decimal.** The predecessor measured +6.9 pp and **+0.4 pp**;
this measures +10.5 pp and **+0.4 pp**. So the frozen critic did not change RQ2's
ordering — it added noise around it. The permutation-invariance effect is now
resolved with **completely disjoint seed ranges**, which the inherited 6 pp seed
spread could never have shown.

### ⚠️ One observation that is NOT a finding, recorded so it is not lost

At the same median, the GNN is far more *stable* than DeepSets: IQR **0.14
against 1.00**, range **2.4 pp against 7.2 pp**, and its worst seed is **43.9
against 41.0 — 2.9 pp better**.

⛔ This is **not** promoted to a result. The rule declared before the run judges
the contrast on non-overlap of ranges, and by that rule it is a null. Reading a
worst-seed advantage out of the data *after* seeing it is exactly the post-hoc
rule-invention this project forbids, and `AGENTS.md`'s "judge on the worst seed"
is a rule for gates that were declared that way — not a licence to re-judge a
gate that was not.

It is recorded because it is a *hypothesis worth pre-declaring next time*: **the
relational rung may buy variance reduction rather than mean improvement.** If
anyone wants that, it needs its own gate, declared before its own run.

## 📏 Stage B — measured 2026-09-01, CUDA, **eval split**, 5 training seeds, N ∈ {3, 5, 8}

Each architecture at its own stage-A cadence. Median [min–max across the 5
training seeds], scored through `src/baselines/evaluate.py`.

| policy | N = 3 | **N = 5** | **N = 8** |
|---|---|---|---|
| mlp / `base` | 14.9 [13.1–17.0] | **30.7** [29.3–31.4] | **18.0** [15.9–33.2] |
| deepsets / `deep` | 22.3 [20.9–26.8] | **39.8** [36.5–45.8] | **52.5** [52.2–56.7] |
| gnn / `deep` | 24.1 [23.0–25.8] | **40.7** [36.0–43.9] | **53.8** [46.1–56.9] |
| **B0** | 35.2 [32.8–35.8] | **55.7** [52.3–61.2] | **76.0** [71.3–76.4] |
| random | 5.7 [4.9–6.5] | 10.6 [9.6–11.7] | 19.4 [18.7–19.9] |

### The declared contrasts, at N = 5

| contrast | effect | ranges | verdict |
|---|---|---|---|
| **MLP → DeepSets** | **+9.1 pp** | [29.3, 31.4] vs [36.5, 45.8] — **disjoint** | ✅ **confirmed** |
| **DeepSets → GNN** | **+0.9 pp** | overlapping, and below the 2 pp threshold | ⛔ **null** |

🔍 **The inherited RQ2 result is vindicated, not overturned.** Inherited: MLP
31.2, DeepSets 38.1, GNN 41.2, B0 57.3, random 10.7. Measured now: **30.7, 39.8,
40.7, 55.7, 10.6** — every column within ~1.5 pp, on a trainer whose critic does
not freeze. ⚠️ Earlier in this session I wrote that the frozen critic meant the
ladder "has to be re-run before RQ2 is reported". It did, and the re-run says the
predecessor's conclusions were right. That is the outcome a re-run is *for*.

⚠️ **The fix moved the train split and not the eval split.** GNN train 41.1 →
44.5 %, GNN eval 41.2 → 40.7 %. So the ~3.5 pp the critic fix bought is
**train-split only**, and the 3.8 pp train→eval gap it opens is new. Held-out
routes through the same buildings are a weak generalisation test to begin with
(`PLAN.md` §7); this is a reason to take that criticism more seriously, not less.

### 🔍 The column that was never measured — and it is RQ2's real answer

📏 Zero-shot transfer, N = 5 → N = 8, on the **worst** seed as well as the median:

| policy | median | worst seed |
|---|---|---|
| **mlp** | 30.7 → **18.0** (**−12.8 pp**) | 29.3 → 15.9 (**−13.5 pp**) |
| deepsets | 39.8 → 52.5 (**+12.6 pp**) | 36.5 → 52.2 (**+15.7 pp**) |
| gnn | 40.7 → 53.8 (**+13.1 pp**) | 36.0 → 46.1 (+10.0 pp) |
| B0 | 55.7 → 76.0 (+20.2 pp) | 52.3 → 71.3 (+19.0 pp) |
| random | 10.6 → 19.4 (+8.8 pp) | 9.6 → 18.7 (+9.0 pp) |

☠️ **The MLP is the only policy in the table that gets WORSE with more drones**,
and at N = 8 its median (**18.0 %**) sits **below random's (19.4 %)**. Everything
else gains — random gains 8.8 pp simply because more drones cover more ground.

The mechanism is structural and was predicted: the MLP's neighbour slots are
**position-specific weights**, and slots 5–7 are zero-padded at every training
step. At N = 8 they carry real data the network has never seen non-zero. DeepSets
and the GNN pool over valid neighbours and cannot have this failure.

🔍 So permutation invariance is not worth "+9.1 pp" — **in-distribution it is
worth +9.1 pp, and out-of-distribution it is the difference between transferring
and not transferring at all.** That is a much stronger statement of RQ2's first
contrast than the N = 5 column supports, and it is the column `BLOCK_G.md`
flagged as informative and never ran.

**DeepSets → GNN remains null at every swarm size** — +1.8 pp at N = 3, +0.9 at
N = 5, +1.3 at N = 8, ranges overlapping throughout. ⚠️ On the *worst* seed at
N = 8, DeepSets is **better** (52.2 against 46.1), driven by one weak GNN seed.
The relational rung does not earn its keep on this task at any `N` tested.

### ⚠️ B0's advantage grows with the swarm, and that is uncomfortable

| | N = 3 | N = 5 | N = 8 |
|---|---|---|---|
| B0 − best learned | **11.1 pp** | **15.0 pp** | **22.2 pp** |

B0 converts extra drones into mission capability better than any learned policy
does. Since the whole premise is that coordination is what learning should buy,
a gap that *widens* with the number of agents is a finding that belongs in the
thesis rather than in a footnote. ⛔ It is also a reason not to read the N = 8
transfer result as "the learned policies scale well" — they scale, and B0 scales
better.
