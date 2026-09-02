"""Frame stacking: `EnvConfig.obs_history` and `SwarmActor(obs_history=...)`.

📏 Motivated by [`results/b0_ablation.md`](../results/b0_ablation.md): local link
repair is **+6.9 pp** of B0's design advantage and `_update_repair` is a
gradient-free hill climb carrying **one step** of search state (`prev_score`,
`lat_dir`). A policy that can see the previous frame can form the same
difference; one that cannot, cannot.

Three things here fail *silently* rather than loudly, which is why each has a
test: the history advancing twice per step under `auto_reset`, a fresh episode
inheriting the previous one's frames, and the relational rungs losing their
permutation structure when the stack is flattened the lazy way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.core import FLAT_DIM, BatchedSwarmEnv, EnvConfig
from src.models.actor import SwarmActor, unpack_stacked

K = 3


def make(obs_history: int, auto_reset: bool = False, num_envs: int = 4):
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs,
            num_drones=5,
            device="cpu",
            seed=0,
            obs_history=obs_history,
            auto_reset=auto_reset,
        )
    )


def test_off_by_default_and_flat_is_never_touched() -> None:
    """🔒 B0 and `test_golden.py` consume `flat`; stacking must be invisible to them."""
    assert EnvConfig(num_envs=1).obs_history == 1
    env = make(1)
    obs = env.reset()
    assert "flat_history" not in obs
    assert obs["flat"].shape == (4, 5, FLAT_DIM)

    env_k = make(K)
    obs_k = env_k.reset()
    assert obs_k["flat"].shape == (4, 5, FLAT_DIM), "flat must stay one frame wide"
    assert obs_k["flat_history"].shape == (4, 5, K, FLAT_DIM)


def test_a_fresh_episode_repeats_its_first_frame_rather_than_reading_zeros() -> None:
    """⚠️ Zero-padding at t=0 is a huge spurious jump to a hill climber."""
    env = make(K)
    obs = env.reset()
    hist = obs["flat_history"]
    for i in range(K):
        torch.testing.assert_close(hist[:, :, i], obs["flat"])


def test_the_newest_frame_is_last_and_the_history_actually_shifts() -> None:
    env = make(K)
    obs = env.reset()
    first = obs["flat"].clone()

    seen = [first]
    for _ in range(2):
        obs, *_ = env.step(torch.zeros(4, 5, 3))
        seen.append(obs["flat"].clone())
        torch.testing.assert_close(obs["flat_history"][:, :, -1], obs["flat"])

    hist = obs["flat_history"]
    # After two steps: slots are [first, second, third] with third == newest.
    torch.testing.assert_close(hist[:, :, -1], seen[2])
    torch.testing.assert_close(hist[:, :, -2], seen[1])
    torch.testing.assert_close(hist[:, :, -3], seen[0])


def test_the_history_advances_exactly_once_per_step_under_auto_reset() -> None:
    """⛔ The silent one. `_observe` runs TWICE per `step()` when auto_reset is on.

    Rolling inside `_observe` would advance the history twice for every
    environment that did not terminate, so "the previous frame" would silently
    mean *two* steps ago -- for exactly the transitions the policy learns from.
    """
    env = make(K, auto_reset=True)
    obs = env.reset()
    prev_flat = obs["flat"].clone()

    obs, *_ = env.step(torch.zeros(4, 5, 3))
    hist = obs["flat_history"]

    # One step: newest is the current frame, and the slot behind it is the frame
    # from immediately before -- not two steps back.
    torch.testing.assert_close(hist[:, :, -1], obs["flat"])
    torch.testing.assert_close(hist[:, :, -2], prev_flat)


def test_unpack_stacked_keeps_each_neighbours_history_attached_to_that_neighbour() -> None:
    """🔒 The structure `docs/MODELS.md` requires DeepSets and the GNN to have.

    Flattening `(k, 108)` into one 324-vector would scramble which frame belonged
    to which neighbour slot, and the off-N transfer columns would then be
    measuring the loss of permutation structure rather than the architecture.
    """
    stacked = torch.randn(2, 5, K, FLAT_DIM)
    parts = unpack_stacked(stacked, K)

    assert parts["ego"].shape == (2, 5, 24 * K)
    assert parts["neighbour"].shape == (2, 5, 7, 9 * K)
    assert parts["edge"].shape == (2, 5, 7, 2 * K)
    # `valid` is the max-N padding mask, fixed for an episode: newest frame only.
    assert parts["valid"].shape == (2, 5, 7)

    from src.env.core import unpack_flat

    for i in range(K):
        single = unpack_flat(stacked[..., i, :])
        torch.testing.assert_close(parts["ego"][..., i * 24 : (i + 1) * 24], single["ego"])
        torch.testing.assert_close(
            parts["neighbour"][..., i * 9 : (i + 1) * 9], single["neighbour"]
        )


def test_every_architecture_accepts_a_stacked_observation_flattened() -> None:
    """The actor's public input stays 2-D; it owns the unflatten."""
    rows = torch.randn(6, K * FLAT_DIM)
    for arch in ("mlp", "deepsets", "gnn"):
        actor = SwarmActor(architecture=arch, obs_history=K)
        mean, log_std = actor(rows)
        assert mean.shape == (6, 3)
        assert torch.isfinite(mean).all()
        assert log_std.shape == (3,)


def test_a_stacked_actor_has_more_parameters_than_an_unstacked_one() -> None:
    """Cheap guard that `frames` actually reached the trunk rather than being dropped."""
    from src.models.actor import parameter_count

    for arch in ("mlp", "deepsets", "gnn"):
        assert parameter_count(SwarmActor(architecture=arch, obs_history=K)) > parameter_count(
            SwarmActor(architecture=arch)
        )


def test_deepsets_stays_permutation_invariant_with_history() -> None:
    """The property the relational rungs exist for must survive stacking."""
    b, n_nb = 3, 7
    stacked = torch.randn(b, K, FLAT_DIM)
    actor = SwarmActor(architecture="deepsets", obs_history=K)
    base = actor(stacked.reshape(b, -1))[0]

    # Permute the neighbour slots identically in every frame, and the edge slots
    # with them; a permutation-invariant trunk must not notice.
    perm = torch.randperm(n_nb)
    parts = unpack_stacked(stacked.unsqueeze(1), K)
    nb = parts["neighbour"][:, :, perm]
    ed = parts["edge"][:, :, perm]
    va = parts["valid"][:, :, perm]
    rebuilt = torch.cat(
        [
            torch.cat(
                [
                    parts["ego"][..., i * 24 : (i + 1) * 24],
                    nb[..., i * 9 : (i + 1) * 9].flatten(-2),
                    ed[..., i * 2 : (i + 1) * 2].flatten(-2),
                    va,
                ],
                dim=-1,
            )
            for i in range(K)
        ],
        dim=-1,
    ).squeeze(1)
    torch.testing.assert_close(actor(rebuilt)[0], base, rtol=1e-4, atol=1e-5)
