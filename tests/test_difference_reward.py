"""`RewardWeights.w_difference` — per-drone credit that a shared critic cannot erase.

`results/credit_assignment.md` closed the reward axis *structurally*: every term
`reward_terms()` returns through `team(x)` is identical across drones **by
construction**, so it cancels from `Var_i(A)` exactly and no shaping knob can move
role differentiation. 📏 The measured between-drone share of return variance is
**0.04–0.16 %** on B0, on the GNN and on a random policy alike.

`D_i = G(z) − G(z_{−i})` is the one instrument that changes the *return* rather
than scaling a term that cancels. The properties that make it usable are all
checkable, and each is checked here because each would fail **silently**:

1. it is **exact** — the vectorised `(B·N, R, R)` form must equal an explicit
   per-drone loop, or the credit signal is confidently wrong;
2. it is **non-negative** — a consequence of `best_relay_capacity` maximising over
   sub-chains, not of a clamp. ⚠️ It would break under a fixed-hop router;
3. it is **zero for a redundant drone and one for a pivotal one**, which is the
   entire content of the term;
4. it is **off by default, byte-identically** — every number in `results/` was
   measured with it off, so the shipped reward must not move by one bit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines.b0 import B0Policy
from src.env import routing
from src.env.core import CAPACITY_THRESHOLD_MBPS, BatchedSwarmEnv, EnvConfig
from src.env.reward import DEFAULT_WEIGHTS, RewardWeights, difference_reward, reward, reward_terms


def make_env(w_difference: float, num_envs: int = 16, num_drones: int = 5) -> BatchedSwarmEnv:
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs,
            num_drones=num_drones,
            device="cpu",
            seed=3,
            fidelity="F4",
            jammer="J1",
            auto_reset=False,
            compile_occlusion=False,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
        ),
        weights=RewardWeights(w_difference=w_difference),
    )


def roll(env: BatchedSwarmEnv, steps: int = 80):
    """Fly B0 for `steps`, then return the snapshot and aux of the state reached.

    B0 rather than a random policy on purpose: the interesting structure —
    one pivotal observer, several redundant drones — only exists once a chain
    has actually formed.
    """
    b0 = B0Policy(num_envs=env.cfg.num_envs, num_drones=env.cfg.num_drones, device="cpu")
    b0.reset()
    obs = env.reset()
    for _ in range(steps):
        obs, *_ = env.step(b0.act(obs["flat"]))
    return env._evaluate()


# --------------------------------------------------------------------------- #
# 1. Exactness
# --------------------------------------------------------------------------- #


def test_the_vectorised_counterfactual_equals_an_explicit_per_drone_loop():
    """🔒 `_capable_without` folds N deletions into ONE `(B*N, R, R)` routing call.

    A masking bug there is silent: the credit signal is still per-drone, still
    plausible, and attributes the mission to the wrong drone. So it is checked
    against the obvious slow version rather than against intuition.
    """
    env = make_env(1.0)
    snap, aux = roll(env)
    capacity, sees = aux["capacity_mbps"], aux["sees_hvt"]
    r = env.cfg.n_radio

    expected = torch.zeros_like(snap.capable_without)
    for i in range(env.cfg.num_drones):
        cap = capacity.clone()
        cap[:, i, :] = 0.0  # a deleted node can neither forward ...
        cap[:, :, i] = 0.0  # ... nor receive
        src = torch.cat([sees.clone(), torch.zeros_like(sees[:, :1])], dim=1)
        src[:, i] = False
        e2e = routing.best_relay_capacity(
            cap, src, dst_index=env.mcv_idx, max_hops=r - 1, reuse_limit=env.cfg.reuse_limit
        )
        expected[:, i] = (src.any(-1) & (e2e >= CAPACITY_THRESHOLD_MBPS)).float()

    assert torch.equal(snap.capable_without, expected)


def test_deleting_a_drone_also_deletes_its_sighting_not_only_its_links():
    """⚠️ The failure mode that survives a links-only mask.

    Zeroing drone `i`'s row and column but leaving `sees[i]` in the source mask
    leaves a ghost: the router still believes someone is watching the target, so
    `observed_{-i}` stays true and the sole observer scores `D = 0`. That inverts
    the term's meaning for exactly the drone it exists to credit, and every
    number downstream still looks reasonable.
    """
    env = make_env(1.0)
    snap, aux = roll(env)
    sees = aux["sees_hvt"]

    lone = (sees.sum(dim=-1) == 1) & snap.observed
    if not bool(lone.any()):
        pytest.skip("no environment reached a single-observer state in this rollout")
    idx = int(lone.float().argmax())
    observer = int(sees[idx].float().argmax())
    # Delete the only drone holding the target: nobody is observing, so the
    # counterfactual mission MUST be a failure whatever the radio links do.
    assert snap.capable_without[idx, observer] == 0.0


# --------------------------------------------------------------------------- #
# 2. Sign, which is a property of the router
# --------------------------------------------------------------------------- #


def test_the_difference_reward_is_non_negative_because_the_router_maximises():
    """🔒 `best_relay_capacity` maximises over every chain up to `max_hops`,
    *including shorter ones*, so deleting a node only removes options. Hence
    `e2e(z) >= e2e(z_-i)` and `D_i >= 0`.

    ⚠️ This is NOT a clamp and it is NOT an assumption about the physics. It
    would fail under a router that fixed the hop count, because `min(hops, 3)`
    would then let an extra relay cost rate. Pinned so that a future routing
    change surfaces here rather than as an unexplained sign in a reward trace.
    """
    w = RewardWeights(w_difference=1.0)
    env = make_env(1.0)
    snap, _ = roll(env)
    d = difference_reward(snap, w)
    assert (d >= 0.0).all(), f"{int((d < 0).sum())} negative entries"
    assert d.max() <= w.w_difference + 1e-6


# --------------------------------------------------------------------------- #
# 3. Semantics: what the term actually says
# --------------------------------------------------------------------------- #


def test_a_pivotal_drone_scores_the_weight_and_a_redundant_one_scores_zero():
    """🔍 The whole content of the term, stated as a test.

    `mission` is broadcast identically to all N drones whenever the swarm is
    capable. `D` is on only for the drones without which it would not be.
    """
    w = RewardWeights(w_difference=1.0)
    env = make_env(1.0)
    snap, _ = roll(env)
    d = difference_reward(snap, w)
    capable = (snap.observed & (snap.e2e_capacity_mbps >= CAPACITY_THRESHOLD_MBPS)).float()

    # every nonzero D sits in an env that is capable ...
    assert torch.equal((d > 0).any(dim=-1) & True, (d > 0).any(dim=-1) & (capable > 0))
    # ... and equals exactly the weight there
    assert torch.equal(d[d > 0], torch.full_like(d[d > 0], w.w_difference))
    # ... and the redundant drones of a capable env score exactly zero, which is
    # the between-drone spread `mission` cannot produce
    assert bool((d.sum(dim=-1) < capable * env.cfg.num_drones).any())


def test_the_term_carries_between_drone_variance_where_mission_carries_none():
    """📏 The measurement `results/credit_assignment.md` says nothing else can make.

    `mission` is a `team(x)` broadcast, so its between-drone standard deviation
    is **exactly** 0.0. `difference` is the same event, attributed.
    """
    env = make_env(1.0)
    snap, _ = roll(env)
    terms = reward_terms(snap, snap, RewardWeights(w_difference=1.0), gamma=env.cfg.gamma)
    assert terms["mission"].std(dim=-1).max() == 0.0
    assert terms["difference"].std(dim=-1).max() > 0.0


# --------------------------------------------------------------------------- #
# 4. Off by default, byte-identically
# --------------------------------------------------------------------------- #


def test_the_shipped_reward_is_unchanged_bit_for_bit_when_the_weight_is_zero():
    """⛔ Every number in `results/` was measured with this off. If the default
    reward moves by one bit, none of them are reproducible any more."""
    assert DEFAULT_WEIGHTS.w_difference == 0.0
    env = make_env(0.0)
    snap, _ = roll(env)
    assert snap.capable_without is None, "the counterfactual must not even be computed"

    off = reward(snap, snap, RewardWeights(), gamma=env.cfg.gamma)
    explicit_zero = reward(snap, snap, RewardWeights(w_difference=0.0), gamma=env.cfg.gamma)
    assert torch.equal(off, explicit_zero)
    assert torch.equal(difference_reward(snap, RewardWeights()), torch.zeros_like(snap.battery))


def test_a_hand_built_snapshot_without_the_counterfactual_fails_loudly():
    """⚠️ `Snapshot.capable_without` is optional so the reward's own tests can
    build one. With the weight on, a missing field must raise rather than
    silently contribute zero credit — that would look exactly like a null."""
    env = make_env(0.0)
    snap, _ = roll(env)
    assert snap.capable_without is None
    with pytest.raises(ValueError, match="capable_without"):
        difference_reward(snap, RewardWeights(w_difference=1.0))


def test_the_per_drone_term_list_in_measure_credit_still_matches_the_reward():
    """🔒 `scripts/measure_credit.py` labels each term per-drone or team, and the
    label is what the published table is read through. A new per-drone term that
    is not on that list would be reported as a team term and silently understate
    the credit it supplies."""
    from scripts.measure_credit import PER_DRONE_TERMS

    env = make_env(1.0)
    snap, _ = roll(env)
    terms = reward_terms(snap, snap, RewardWeights(w_difference=1.0), gamma=env.cfg.gamma)
    per_drone = {name for name, value in terms.items() if float(value.std(dim=-1).max()) > 0.0}
    assert per_drone <= set(PER_DRONE_TERMS), f"unlabelled per-drone terms: {per_drone}"
    assert "difference" in PER_DRONE_TERMS
