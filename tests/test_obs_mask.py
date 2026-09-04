"""`EnvConfig.mask_jammed_obs` — the learned analogue of clearance-repair.

`PLAN.md` §1 claims exploitability is a cost of closing a feedback loop on the
quantity an adversary attacks. `B0Config.repair_score = "clearance"` tests that on
the scripted side; this flag tests it on the learned side, by removing exactly the
features the emitter can move and leaving geometry intact.

⛔ The whole experiment is void if the mask hits the wrong columns, and a wrong
column is silent — the policy still trains, just deprived of something the claim
never said to remove. So the index derivation is checked against `unpack_flat`
rather than against the numbers a human wrote down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.core import (
    EDGE_DIM,
    EGO_DIM,
    FLAT_DIM,
    JAMMED_EDGE_IDX,
    JAMMED_EGO_IDX,
    N_MAX,
    BatchedSwarmEnv,
    EnvConfig,
    jammed_flat_indices,
    unpack_flat,
)


def make(mask: bool, num_envs: int = 4):
    return BatchedSwarmEnv(
        EnvConfig(num_envs=num_envs, num_drones=5, device="cpu", seed=0, mask_jammed_obs=mask)
    )


def test_off_by_default() -> None:
    """⛔ Every number in results/ was measured with this off."""
    assert EnvConfig(num_envs=1).mask_jammed_obs is False
    env = make(False)
    obs = env.reset()
    assert obs["flat"][..., list(jammed_flat_indices())].abs().max() > 0


def test_the_indices_land_where_unpack_flat_says_they_do() -> None:
    """🔒 Derived from the layout, not written down. A drifted index is silent.

    Builds a probe vector that is 1.0 at exactly the claimed jammed positions and
    0.0 everywhere else, then unpacks it and asserts the ones land on the ego
    features and the edge sub-feature the emitter actually moves.
    """
    probe = torch.zeros(1, 1, FLAT_DIM)
    probe[..., list(jammed_flat_indices())] = 1.0
    parts = unpack_flat(probe)

    ego = parts["ego"][0, 0]
    assert set(torch.nonzero(ego).flatten().tolist()) == set(JAMMED_EGO_IDX)

    # Every neighbour slot's capacity entry, and no clearance entry.
    edge = parts["edge"][0, 0]  # (7, 2)
    for slot in range(N_MAX - 1):
        for i in range(EDGE_DIM):
            want = 1.0 if i in JAMMED_EDGE_IDX else 0.0
            assert edge[slot, i].item() == want, f"edge slot {slot} feature {i}"

    # Nothing in the neighbour block or the padding mask.
    assert parts["neighbour"].abs().sum() == 0
    assert parts["valid"].abs().sum() == 0


def test_geometry_is_kept_and_only_the_jammed_columns_are_zeroed() -> None:
    """The point of the flag: adapt on geometry, not on the attacked quantity."""
    idx = list(jammed_flat_indices())
    keep = [i for i in range(FLAT_DIM) if i not in idx]
    env = make(True)
    obs = env.reset()
    for _ in range(6):
        obs, *_ = env.step(torch.zeros(4, 5, 3))
    flat = obs["flat"]
    assert flat[..., idx].abs().max() == 0.0
    assert flat[..., keep].abs().max() > 0.0


def test_clearance_survives_the_mask() -> None:
    """🔒 `clr_hvt` (19) and `clr_mcv` (20) are building occlusion, which the
    emitter cannot touch. Masking them would deprive the learned arm of exactly
    what the scripted analogue is allowed to use."""
    assert 19 not in JAMMED_EGO_IDX
    assert 20 not in JAMMED_EGO_IDX
    # And the clearance half of every edge pair.
    assert 1 not in JAMMED_EDGE_IDX


def test_on_path_and_link_timeout_are_deliberately_left_in() -> None:
    """⚠️ Routing-derived, so reachable indirectly -- but B0's clearance-repair
    reads `nb_onpath` too, and removing them would make the learned arm strictly
    more deprived than the comparison it exists to match."""
    assert 21 not in JAMMED_EGO_IDX  # on_path
    assert 23 not in JAMMED_EGO_IDX  # steps_since_link


def test_the_mask_reaches_every_frame_of_the_history() -> None:
    """⛔ The silent one. Masking downstream of the history buffer would leave the
    previous frames unmasked and hand the policy its loop back through time."""
    env = BatchedSwarmEnv(
        EnvConfig(
            num_envs=4, num_drones=5, device="cpu", seed=0, mask_jammed_obs=True, obs_history=3
        )
    )
    obs = env.reset()
    for _ in range(5):
        obs, *_ = env.step(torch.zeros(4, 5, 3))
    assert obs["flat_history"][..., list(jammed_flat_indices())].abs().max() == 0.0


def test_masking_changes_nothing_but_the_masked_columns() -> None:
    """Same seed, same episodes: the mask is an observation change, not a physics
    change. If the drones ended up somewhere else the flag is doing more than it
    claims."""
    a, b = make(False), make(True)
    oa, ob = a.reset(), b.reset()
    for _ in range(8):
        oa, *_ = a.step(torch.zeros(4, 5, 3))
        ob, *_ = b.step(torch.zeros(4, 5, 3))
    torch.testing.assert_close(a.drone_pos, b.drone_pos)
    torch.testing.assert_close(a.hvt_pos, b.hvt_pos)
    keep = [i for i in range(FLAT_DIM) if i not in list(jammed_flat_indices())]
    torch.testing.assert_close(oa["flat"][..., keep], ob["flat"][..., keep])


def test_the_derivation_survives_a_layout_change() -> None:
    """`jammed_flat_indices` must be computed, never hardcoded."""
    k = N_MAX - 1
    base = EGO_DIM + k * 9  # NEIGHBOUR_DIM
    want = set(JAMMED_EGO_IDX) | {
        base + s * EDGE_DIM + i for s in range(k) for i in JAMMED_EDGE_IDX
    }
    assert set(jammed_flat_indices()) == want
    assert max(jammed_flat_indices()) < FLAT_DIM
