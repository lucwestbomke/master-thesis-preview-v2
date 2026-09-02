"""The variance decomposition `scripts/measure_credit.py` reports.

The whole argument rests on one identity -- that `E[Var_i] + Var(E_i)` really is
the total variance, so `differentiable_share` is a share of something and not two
unrelated numbers divided by each other. That is worth pinning: the script's
conclusion is a *structural* claim about what a shared critic can express, and a
silently wrong denominator would make it unfalsifiable rather than wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.measure_credit import PER_DRONE_TERMS, decompose, returns_to_go


def test_the_decomposition_is_the_law_of_total_variance() -> None:
    """`E[Var_i] + Var(E_i)` == `Var(all)`, which is what makes it a *share*."""
    g = torch.randn(37, 11, 5)
    d = decompose(g)
    # `pytest.approx` rather than `==`: `decompose` sums in float32 and converts,
    # while the right-hand side converts each part to float64 and sums, so the two
    # round differently in the last bit. The identity is exact; the arithmetic is not.
    assert d["total_var"] == pytest.approx(d["between_drone_var"] + d["team_var"], rel=1e-6)
    # Population variance over every element, which is what the two parts sum to.
    torch.testing.assert_close(
        torch.tensor(d["total_var"]),
        g.reshape(-1, g.shape[-1]).var(unbiased=False),
        rtol=1e-4,
        atol=1e-5,
    )
    assert 0.0 <= d["differentiable_share"] <= 1.0


def test_identical_drones_have_no_differentiable_signal() -> None:
    """A team-only reward is broadcast across drones, so `Var_i` is exactly zero.

    This is the degenerate case the script exists to detect: it is what every
    `team(x)` term in `reward_terms()` contributes, and it is why no shaping knob
    can move role credit.
    """
    shared = torch.randn(20, 7, 1).expand(20, 7, 5).contiguous()
    d = decompose(shared)
    assert d["between_drone_var"] == 0.0
    assert d["differentiable_share"] == 0.0
    assert d["team_var"] > 0.0


def test_pure_per_drone_noise_is_entirely_differentiable() -> None:
    """The opposite pole, so the statistic is pinned at both ends."""
    g = torch.randn(400, 9, 5)
    g = g - g.mean(dim=-1, keepdim=True)  # remove the team component exactly
    d = decompose(g)
    assert d["team_var"] < 1e-6
    assert d["differentiable_share"] > 0.99


def test_returns_to_go_discounts_backwards() -> None:
    """Hand-computed against a three-step episode, gamma = 0.5."""
    rewards = torch.tensor([[[1.0]], [[2.0]], [[4.0]]])  # (T=3, B=1, N=1)
    got = returns_to_go(rewards, gamma=0.5)
    # t=2: 4 ; t=1: 2 + .5*4 = 4 ; t=0: 1 + .5*4 = 3
    torch.testing.assert_close(got.flatten(), torch.tensor([3.0, 4.0, 4.0]))


def test_undiscounted_return_to_go_is_the_reversed_cumulative_sum() -> None:
    rewards = torch.randn(15, 3, 4)
    got = returns_to_go(rewards, gamma=1.0)
    want = rewards.flip(0).cumsum(0).flip(0)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_the_per_drone_term_list_matches_the_reward_it_claims_to_describe() -> None:
    """⛔ The structural half of the argument, asserted rather than asserted-in-prose.

    `measure_credit.py` claims only `energy`, `effort` and `relay` differ between
    drones; every other term in `reward_terms()` goes through `team(x)` and is
    broadcast identically. If a future term breaks that, the script's conclusion
    silently stops following and this test is what says so.
    """
    from src.env.core import BatchedSwarmEnv, EnvConfig
    from src.env.reward import reward_terms

    env = BatchedSwarmEnv(EnvConfig(num_envs=4, num_drones=5, device="cpu", seed=0))
    env.reset()
    snap = env.snap
    terms = reward_terms(snap, snap)

    for name, value in terms.items():
        spread = float(value.var(dim=-1, unbiased=False).max())
        if name in PER_DRONE_TERMS:
            continue  # may or may not vary on any given state; not pinned here
        assert spread == 0.0, f"{name} varies between drones but is not in PER_DRONE_TERMS"

    assert set(PER_DRONE_TERMS) <= set(terms), "PER_DRONE_TERMS names a term that is gone"
