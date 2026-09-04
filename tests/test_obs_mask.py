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

import pytest
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


def _env(num_envs: int = 8, **kw) -> BatchedSwarmEnv:
    """A deterministic stage-4 env for the observation-content tests below."""
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs,
            num_drones=5,
            device="cpu",
            seed=0,
            auto_reset=False,
            compile_occlusion=False,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
            **kw,
        )
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


# --------------------------------------------------------------------------- #
# `mask_broadcast_obs` and `cue_mode`, added 2026-09-04.
#
# Same discipline as the jammed mask above: the whole experiment is void if the
# wrong columns move, and a wrong column is silent.
# --------------------------------------------------------------------------- #


def test_broadcast_indices_are_the_features_that_are_provably_identical_per_drone():
    """🔒 Derived from `unpack_flat`, not hand-written, and checked against the
    property that defines them: `ego[b, i, k] == ego[b, j, k]` for every pair of
    drones, at every state, because both are `(B,)` scalars `.expand()`ed across
    the drone axis."""
    from src.env.core import BROADCAST_EGO_IDX, broadcast_flat_indices

    env = _env()
    obs = env.reset()
    for _ in range(20):
        obs, *_ = env.step(torch.zeros(env.cfg.num_envs, env.cfg.num_drones, 3))
    ego = unpack_flat(obs["flat"])["ego"]

    spread = ego.std(dim=1)  # across drones, (B, EGO_DIM)
    identical = {k for k in range(EGO_DIM) if float(spread[:, k].max()) == 0.0}
    assert set(BROADCAST_EGO_IDX) <= identical, (
        f"declared broadcast features that are NOT identical across drones: "
        f"{set(BROADCAST_EGO_IDX) - identical}"
    )
    # the ego block sits at the front of `_pack`, so the flat indices coincide
    assert broadcast_flat_indices() == BROADCAST_EGO_IDX
    assert all(i < EGO_DIM for i in broadcast_flat_indices())


def test_the_broadcast_mask_zeroes_those_columns_and_only_those():
    from src.env.core import broadcast_flat_indices

    plain, masked = _env(), _env(mask_broadcast_obs=True)
    a, b = plain.reset()["flat"], masked.reset()["flat"]
    idx = list(broadcast_flat_indices())
    assert (b[..., idx] == 0.0).all()
    others = [i for i in range(a.shape[-1]) if i not in idx]
    assert torch.equal(a[..., others], b[..., others])


def test_cue_mode_changes_only_the_cue_block_and_keeps_the_width():
    """🔒 Three wide in every mode, so `EGO_DIM`, `FLAT_DIM`, `unpack_flat`, the
    rollout buffer and every checkpoint are untouched."""
    egos = {}
    for mode in ("position", "bearing", "off"):
        env = _env(cue_mode=mode)
        egos[mode] = unpack_flat(env.reset()["flat"])["ego"]

    for mode, ego in egos.items():
        assert ego.shape[-1] == EGO_DIM, mode
        others = [i for i in range(EGO_DIM) if not 4 <= i < 7]
        # At reset every mode starts from the same state, so only 4:7 may differ.
        assert torch.equal(ego[..., others], egos["position"][..., others]), mode

    assert (egos["off"][..., 4:7] == 0.0).all()
    # `bearing` is a horizontal UNIT vector with z dropped
    bearing = egos["bearing"][..., 4:7]
    assert (bearing[..., 2] == 0.0).all()
    assert torch.allclose(
        bearing[..., :2].norm(dim=-1), torch.ones_like(bearing[..., 0]), atol=1e-5
    )
    # and it points the same way the position vector does
    pos_xy = egos["position"][..., 4:6]
    unit = pos_xy / pos_xy.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    assert torch.allclose(unit, bearing[..., :2], atol=1e-4)


def test_an_unknown_cue_mode_is_refused_at_construction():
    """A typo must not silently fall through to the shipped behaviour and be
    reported as a measured arm."""
    with pytest.raises(ValueError, match="cue_mode"):
        EnvConfig(num_envs=1, cue_mode="bearings")
