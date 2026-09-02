# B0's +10.1 pp is mostly **link repair**, not role assignment

**Declared and measured 2026-09-02.** `scripts/measure_b0_ablation.py` carries the
hypothesis and the three-way rule, written before the numbers existed.

## Why it was run

📏 `BLOCK_E.md` measures `B0-geodesic` 47.1 % → `B0` 57.2 % and calls the
difference "design effort". [`credit_assignment.md`](credit_assignment.md) and
[`memory_horizon.md`](memory_horizon.md) both **inferred** that the carrier was
*ranked roles*. Inference is not measurement, and this project has twice this
week been wrong by reasoning past a number it could have taken.

## 📏 The decomposition — MPS, eval split, stage 4, 5 seeds × 128 episodes

| arm | `mission_capable` | worst | `observed` | `observer_range_m` |
|---|---|---|---|---|
| **`b0`** (control) | **56.8 %** [3.9] | 51.2 | 92.0 % | 91.2 m |
| `b0-norepair` (`repair_amplitude_m = 0`) | **49.3 %** [5.4] | 45.9 | 92.3 % | 90.8 m |
| `geodesic` (roles by index, no belief, no repair) | **45.7 %** [1.9] | 42.5 | 86.4 % | 99.4 m |
| `oracle` (+ ground-truth target state) | 56.4 % [1.5] | 55.2 | 94.0 % | 84.4 m |

🔒 **B0 is deterministic, so the same seed draws the same 128 episodes and the
arms can be compared PAIRED** — which matters, because the raw ranges overlap and
the paired deltas do not:

| component | seeds won | per-seed pp | paired median |
|---|---|---|---|
| **local link repair** | **5/5** | +7.35 +5.29 +5.83 +7.47 +6.90 | **+6.90 pp** |
| **ranked roles + belief** | **5/5** | +2.84 +3.39 +6.40 +3.57 +2.51 | **+3.39 pp** |
| target information (oracle) | 3/5 | −0.66 −4.01 +1.71 +0.35 +0.62 | **+0.35 pp** |

✅ **The oracle re-confirms `BLOCK_E`'s −0.4 pp** on an independent run: perfect
target state is worth nothing measurable, 3/5 seeds, straddling zero.

⚠️ **The declared rule straddles its own threshold and that is reported, not
resolved by picking.** The rule was `R ≥ 7 pp` ⇒ *"repair carries it"*,
`3 < R < 7` ⇒ *"shared"*. Difference-of-medians gives **7.47** (repair carries
it); median-of-paired-differences gives **6.90** (shared, barely). ⛔ Neither
estimator is chosen after the fact. **The robust statement both support is that
link repair is roughly twice what ranked roles are worth**, and that is what
should be quoted.

## ☠️ Two things this corrects

**1. The role story was over-claimed.** Ranked roles are worth **~3.0–3.4 pp**
(3.39 minus the belief's ≤0.4), not the ~10 pp that
[`credit_assignment.md`](credit_assignment.md) and
[`memory_horizon.md`](memory_horizon.md) implied by attributing the whole bundle
to them. Against a **15.0 pp** gap to B0 at J3B, perfect role assignment closes
**about a fifth of it**.

**2. But roles are a *precondition*, not merely a term.** `_update_repair` is
gated on `is_relay = (rank > 0) & (rank <= n_relay)` — **only relays hill-climb**.
So the 6.9 pp of repair is *unavailable* without role assignment; `geodesic`
returns from `_update_repair` immediately. 🔍 Roles are worth 3.4 pp directly and
**enable** a further 6.9 pp, so they are load-bearing for ~10.3 of the 11.0 pp
even though they only *score* 3.4.

## 🔍 What link repair actually is, and why it reopens a question

From its own docstring: *"One step of a 1-D hill climb on observable clearance…
It slides perpendicular to the chain and keeps going while the worst of those
improves, reversing when it does not. Gradient-free, batched."*

⛔ **It is stateful — and the state is one step deep.** The climb carries
`prev_score` (last step's bottleneck), `lat_dir` (which way it is sliding) and
`lat_m` (how far), all cleared in `reset()`. A gradient-free hill climb *cannot*
work without remembering which way it went and whether that helped.

☠️ **This is a different memory from the one [`memory_horizon.md`](memory_horizon.md)
ruled out, and the ruling does not cover it.**

| | target memory | search memory |
|---|---|---|
| what is remembered | where the target was | which way I moved, and did it help |
| horizon | p50 **35 steps**, p90 **320** | **1 step** |
| bounded by the oracle? | ✅ yes — perfect target state is worth −0.4 pp | ⛔ **no** — the oracle supplies target state, not search state |
| measured worth | **~0 pp** | up to **6.9 pp** (role-gated) |

🔒 **So `memory_horizon.md`'s verdict stands as written** — it closed *target*
memory, and that closure is intact. It did not measure search memory, and the
oracle bound it rests on says nothing about it.

⚠️ **But this is NOT a licence to rebuild the GRU.** The state needed is **one
step deep**, so it is supplied by `k = 2` frame stacking — a ~10-line observation
change with no BPTT, no cross-boundary hidden state, and none of the bug surface
that killed the last attempt. ⛔ A recurrent network is the wrong instrument for a
one-step memory.

⚠️ **And a feedforward policy is not actually barred from the behaviour.** B0 hill
climbs because it is gradient-free and cannot see the map. A learned policy could
instead learn the *direct* map from local geometry to a good direction — a harder
function to learn, but one that needs no memory at all and would be strictly
better if learned. 📏 That it has not is a fact about optimisation, not
representation. ⛔ Do not assert that memory is required here; assert that B0's
route to the behaviour requires it.

## Where this leaves the search

📏 The measured budget, against a 15.0 pp gap at J3B:

| component | worth | available to a parameter-shared policy? |
|---|---|---|
| local link repair | **6.9 pp** | gated on roles; needs either 1-step state or a learned direct gradient |
| ranked roles | **3.0–3.4 pp** | ⛔ blocked — role signal is 0.04–0.16 % of the advantage ([`credit_assignment.md`](credit_assignment.md)) |
| target belief | **~0 pp** | irrelevant ([`memory_horizon.md`](memory_horizon.md)) |
| **total** | **~10.3 pp** | **and the gap is 15.0 pp** |

⛔ **Even acquiring every one of B0's design components would not close the gap**,
which is the third independent line of evidence for `PLAN.md` §3's premise. The
remaining ~5 pp is not in the variant ladder at all.

🔍 **The cheapest untested lever is now `k = 2` frame stacking**, motivated by a
measured 6.9 pp rather than by analogy. It needs its own gate, declared before its
own run, at 5 seeds, judged on the worst.

## Regenerate

```bash
uv run python scripts/measure_b0_ablation.py --seeds 5 --num-envs 128 --device mps \
    --out results/b0_ablation.jsonl
```

⚠️ `torch.Generator` streams differ per device. Raw output:
[`b0_ablation.jsonl`](b0_ablation.jsonl).
