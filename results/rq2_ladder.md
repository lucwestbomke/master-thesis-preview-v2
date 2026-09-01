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
