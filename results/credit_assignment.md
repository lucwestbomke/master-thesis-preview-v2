# The advantage cannot tell one drone from another

**Declared and measured 2026-09-02.** `scripts/measure_credit.py`, whose module
docstring carries the hypothesis and the decision rule **written before the
numbers existed**. `tests/test_measure_credit.py` pins the decomposition.

## Why this was run

`PLAN.md` §3 records the deficit — the swarm never differentiates into an observer
and a relay chain — and eight pre-declared interventions that failed to move it,
the last (`PHI_V2`) with a closing gradient *measured adequate* before its run.
Every one of the eight was a **reward** knob.

`src/training/ppo.py` has named a different suspect since the trainer was written,
in a comment nobody had cashed: the critic sees one global state per env and the
reward is per-drone, so `N` rows share one input and carry `N` different targets.

🔍 The observation that turns that comment into an experiment: **the between-drone
spread is two things at once.** It is the ceiling on `explained_variance`, and it
is the *entire budget of drone-differentiating credit*. One value per global state
broadcast across `N` rows means `A[t,b,i] = G[t,b,i] − V[t,b]`, so

```
Var_i(A)  =  Var_i(G)      exactly.
```

Whatever the policy gradient knows about **which drone should do what**, it knows
through that variance and through nothing else.

---

## The structural half — exact, and readable from the source

`reward_terms()` returns `mission`, `idle`, `battery_variance` and `shaping`
through `team(x)`, which broadcasts one `(B,)` tensor across drones. Those terms
are **identical for every drone by construction** and therefore cancel out of
`Var_i` *exactly*, not approximately. Only `energy`, `effort` and `relay` are
per-drone, and 📏 `w_relay` ships at **0.0** — so `relay` is identically zero.

⛔ **The only reward components that can differ between drones are two motion
costs.** This is not a measurement and does not depend on device, seed or policy.
It is pinned by
`test_the_per_drone_term_list_matches_the_reward_it_claims_to_describe`, which
fails if a future term breaks the property.

🔒 **Consequence, stated plainly: no shaping intervention can move role credit.**
`w_hold`, `d_ref`, `potential_scale` and `PHI_V2` all scale team terms. Scaling a
quantity that is identical across drones leaves `Var_i` at exactly zero, whatever
the scale factor is.

## The empirical half — the magnitude

Decomposing the per-drone discounted return-to-go by the law of total variance
over the `(t, b)` grouping:

```
Var_total  =  E_{t,b}[ Var_i(G) ]   +   Var_{t,b}( E_i[G] )
              between-drone             team component
              "differentiable"          what a shared critic predicts
```

📏 **MPS, eval split, stage 4, F4/J1, N = 5, 64 envs × 600 steps, 3 seeds.**
The GNN row is `runs/val-gnn-deep-s0/checkpoint.pt` (12 M steps, `deep`).

| policy | between-drone var | team var | **differentiable share** |
|---|---|---|---|
| random | 0.83 | 2418 | **0.04 %** |
| **B0** | 12.17 | 11568 | **0.11 %** [0.10 – 0.11] |
| **GNN** | 6.65 | 4051 | **0.16 %** [0.13 – 0.21] |

🔒 **The declared rule was `< 5 %` confirms, `> 20 %` refutes.** Measured
**0.04 – 0.16 %**, two orders of magnitude inside the confirm band, on every
policy including the scripted one and the random one.

**99.84 – 99.96 % of the return variance is identical across drones.**

### It is not an artefact of the discount horizon

GAE weights rewards by `(γλ)^l`, not `γ^l`, so the advantage sees a ~19-step
effective horizon rather than a 600-step one. Re-measured at `γλ = 0.94715`
(📏 `GAMMA` 0.997 × `gae_lambda` 0.95):

| policy | share at `γ = 0.999` | share at `γλ = 0.94715` |
|---|---|---|
| B0 | 0.11 % | **0.08 %** |
| GNN | 0.16 % | **0.17 %** |

### Which terms carry it — and it is exactly the two the source predicts

📏 B0, `γ = 0.999`. Between-drone standard deviation of each term's own
return-to-go:

| term | per-drone? | between-drone std |
|---|---|---|
| `mission` | team | **0.00000** |
| `idle` | team | **0.00000** |
| `battery_variance` | team | **0.00000** |
| `shaping` | team | **0.00000** |
| `relay` | yes, but `w_relay = 0` | **0.00000** |
| **`effort`** | **yes** | 1.06645 |
| **`energy`** | **yes** | 2.65357 |

✅ Four exact zeros, measured, matching the structural argument term for term.

---

## ☠️ What this does to eight nulls

It converts them from a list of disappointments into **one mechanism with a
prediction**, and the prediction retro-fits every one of them:

> An intervention that modifies only team reward terms cannot change role
> differentiation, because team terms contribute exactly zero to the only
> variance the policy gradient can use to distinguish drones.

`w_hold`, `d_ref` 1500→400, `potential_scale` 10→30, `PHI_V2` — all team terms,
all predicted nulls. Recurrence and the agent-specific critic changed capacity,
not the signal. That is six of the eight explained by one structural fact.

### ⚠️ And it reframes `w_relay`, which is the one that looks like a counterexample

`docs/inherited/BLOCK_G.md` records `w_relay` raising per-drone advantage variance
**71×** (0.00041 → 0.02931) with no behavioural change, and read that as evidence
that per-drone credit does not help.

📏 Against PPO's unit-normalised advantage, 71× took the differentiating share
from ~0.04 % to ~2.9 % — **still under 3 %.** The honest reading is *"raised it to
3 %, still negligible"*, not *"per-drone credit does not help"*.

🔒 **This reframing was declared in `measure_credit.py` before the run**, precisely
so it could not be invented afterwards.

---

## What it does not show

⚠️ **`differentiable_share` is an upper bound on the advantage's share, not an
estimate of it.** GAE's `δ` carries `γV(s_{t+1}) − V(s_t)`, which varies over
`(t, b)` but is identical across drones — so those terms add to the *team*
component only. Excluding them makes the denominator smaller and the share
larger. The true figure is lower than reported, which is the conservative
direction.

⚠️ **One GNN checkpoint, three evaluation seeds, MPS.** The B0 and random rows
bracket it and the structural half is exact, but this is not a 5-training-seed
result and is not labelled as one. ⛔ Do not quote the GNN row as a policy-level
finding.

⛔ **It does not show that fixing the critic would close the 15 pp gap.** It shows
that one specific mechanism is unavailable to the current architecture. That is a
statement about what *cannot* work, which is weaker than a fix and more durable
than one.

---

## What it redirects the search to

The reward axis is now closed **structurally** rather than by exhaustion — a
better reason than "we tried eight things". What is left is the **critic and the
advantage**, none of which has been touched:

| candidate | why it is in this class |
|---|---|
| a per-drone value head, or a counterfactual baseline (COMA-style) | changes `Var_i(A)` directly rather than scaling a term that cancels |
| a permutation-invariant critic | 📏 the critic is a plain MLP over `rel_pos.flatten(1)`, index-ordered, while the DeepSets/GNN actors are permutation-invariant. And the one permutation-sensitive *actor*, the MLP, is the only policy that gets **worse** with more drones (−12.8 pp at N = 8) |
| per-drone reward decomposition | the team terms are what they are; a decomposition that attributes them is a different objective, and ⚠️ `REWARD.md` has an objection to per-drone potentials that must be read first |

⛔ **This is not a licence for a ninth intervention.** It is a pre-declared
mechanism, and anything built on it needs its own gate declared before its own
run, at 5 seeds, judged on the worst.

## Regenerate

```bash
uv run python scripts/measure_credit.py --policy b0     --device mps --num-envs 64 --seeds 3
uv run python scripts/measure_credit.py --policy random --device mps --num-envs 64 --seeds 3
uv run python scripts/measure_credit.py --policy runs/val-gnn-deep-s0/checkpoint.pt \
    --device mps --num-envs 64 --seeds 3
```

⚠️ `torch.Generator` streams differ per device, so a run on CUDA draws different
episodes. The rows above are comparable with each other and with nothing else.
Raw output: [`credit.jsonl`](credit.jsonl).
