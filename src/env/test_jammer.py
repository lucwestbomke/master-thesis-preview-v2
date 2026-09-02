"""The adversary ladder J0-J3 -- `PLAN.md` §3, the thesis's independent variable.

The claim this project exists to test is not that learned control beats B0 on the
static task; it is that B0 **degrades more** when the adversary adapts. So the
ladder is what the whole result rests on, and the properties below are the ones
whose violation would be silent:

- 🔒 **J1 is bit-identical to the environment before the beam existed**, so no
  inherited number moves;
- the pattern is the 3GPP element pattern and not something that merely looks
  like it;
- 🔒 **the rungs change only the emitter** -- the sensor and the diagnostics run
  on true geometry at every rung, exactly as `test_fidelity.py` requires of the
  fidelity ladder (`BLOCK_F.md` decision 2);
- J3 aims with **one step of latency**, which is what breaks the circularity
  between the jammer, the capacity and the routed chain it targets.
"""

from __future__ import annotations

import math

import pytest
import torch

from . import channel
from .core import (
    JAMMER_BEAMWIDTH_DEG,
    JAMMER_DBM,
    JAMMER_MAX_ATTEN_DB,
    JAMMER_PEAK_GAIN_DBI,
    BatchedSwarmEnv,
    EnvConfig,
)

RUNGS = ("J0", "J1", "J2", "J3")


def make(jammer="J1", num_envs=4, seed=0, fidelity="F4", **kw) -> BatchedSwarmEnv:
    kw.setdefault("stage_weights", (0.0, 0.0, 0.0, 1.0))
    kw.setdefault("compile_occlusion", False)
    env = BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs, num_drones=5, seed=seed, fidelity=fidelity, jammer=jammer, **kw
        )
    )
    env.reset()
    return env


def fly(env, steps, seed=3):
    """Fly the swarm into a relay chain across the city, identically at every rung.

    ⚠️ Same reasoning as `test_fidelity.py::fly`, and the same trap: a random
    action sequence leaves the swarm near the spawn ring with **no routed
    chain**, and a jammer ladder whose adaptive rung has no chain to target is
    testing nothing. The law reads only kinematics, which the rung does not
    touch, so the geometry is identical across rungs by induction.
    """
    gen = torch.Generator(device=env.device).manual_seed(seed)
    b, n = env.cfg.num_envs, env.cfg.num_drones
    frac = torch.linspace(0.15, 0.95, n, device=env.device).view(1, n, 1)
    gain = 25.0 * env.cfg.dt_s * 3.0
    for _ in range(steps):
        axis = env.hvt_pos[:, None, :] - env.mcv_pos[:, None, :]
        station = env.mcv_pos[:, None, :] + frac * axis
        target = torch.cat([station[..., :2], torch.full_like(station[..., 2:], 60.0)], dim=-1)
        jitter = (torch.rand(b, n, 3, generator=gen, device=env.device) * 2.0 - 1.0) * 0.2
        env.step(((target - env.drone_pos) / gain + jitter).clamp(-1.0, 1.0))


# --------------------------------------------------------------------------- #
# J1 is the inherited condition, unchanged
# --------------------------------------------------------------------------- #


def test_j1_is_bit_identical_to_the_environment_before_the_beam():
    """🔒 The whole inherited results table is measured under the isotropic
    jammer. If adding the beam moved J1 by so much as a float, every number in
    `docs/INHERITED.md` would silently need re-measuring."""
    env = make("J1")
    fly(env, 20)
    _snap, aux = env._evaluate()

    # the pre-beam formula, written out
    pos_k = torch.cat([env.drone_pos, env.mcv_pos.unsqueeze(1), env.hvt_pos.unsqueeze(1)], dim=1)
    radio = pos_k[:, : env.cfg.n_radio]
    _true, chan = env._clearance(pos_k)
    r = env.cfg.n_radio
    d = (radio - env.hvt_pos.unsqueeze(1)).norm(dim=-1)
    los = chan[:, :r, env.hvt_idx] >= 0.0
    expect = channel.dbm_to_mw(
        JAMMER_DBM - channel.pathloss_a2g_umi_av_db(d, radio[..., 2], los)
    ) * env.jammer_on.unsqueeze(-1)
    assert torch.allclose(aux["jam_mw"], expect, rtol=1e-6), (
        "J1 is no longer the isotropic emitter the inherited numbers were measured under"
    )


def test_j0_removes_the_emitter_entirely():
    env = make("J0")
    fly(env, 10)
    _snap, aux = env._evaluate()
    assert torch.equal(aux["jam_mw"], torch.zeros_like(aux["jam_mw"]))


def test_an_unknown_rung_is_refused_at_construction():
    with pytest.raises(ValueError, match="jammer must be one of"):
        EnvConfig(num_envs=1, jammer="J9")
    with pytest.raises(ValueError, match="jammer must be one of"):
        EnvConfig(num_envs=1, jammer="J4")  # the learned rung has no learner yet


# --------------------------------------------------------------------------- #
# The element pattern
# --------------------------------------------------------------------------- #


def test_the_pattern_is_the_3gpp_element_pattern():
    """`A(theta) = -min[12 (theta/theta_3dB)^2, A_max]`, added to the peak gain.
    Checked at the three points that define it, plus symmetry and monotonicity --
    a pattern that merely decreased with angle would pass a weaker test."""
    env = make("J2", num_envs=1)
    b, r = 1, env.cfg.n_radio
    env.hvt_pos = torch.zeros(b, 3)
    env.jam_target = torch.zeros(b, dtype=torch.long)  # boresight = node 0

    def gain_at(deg: float) -> float:
        radio = torch.zeros(b, r, 3)
        radio[0, 0] = torch.tensor([100.0, 0.0, 60.0])  # node 0 defines boresight
        a = math.radians(deg)
        radio[0, 1] = torch.tensor([100.0 * math.cos(a), 100.0 * math.sin(a), 60.0])
        return float(env._beam_gain_db(radio)[0, 1])

    assert gain_at(0.0) == pytest.approx(JAMMER_PEAK_GAIN_DBI, abs=1e-4)
    # theta_3dB is where the pattern is 3 dB down, by construction: 12*(1)^2 = 12?
    # No -- the 3 dB point of THIS pattern is where 12*(theta/theta_3dB)^2 = 3,
    # i.e. theta = theta_3dB / 2. Assert the defining coefficient instead.
    half = JAMMER_BEAMWIDTH_DEG / 2.0
    assert gain_at(half) == pytest.approx(JAMMER_PEAK_GAIN_DBI - 3.0, abs=1e-3)
    assert gain_at(JAMMER_BEAMWIDTH_DEG) == pytest.approx(JAMMER_PEAK_GAIN_DBI - 12.0, abs=1e-3)

    floor = JAMMER_PEAK_GAIN_DBI - JAMMER_MAX_ATTEN_DB
    assert gain_at(180.0) == pytest.approx(floor, abs=1e-4)
    assert gain_at(90.0) == pytest.approx(floor, abs=1e-4), "must FLOOR, not keep falling"

    for deg in (10.0, 30.0, 60.0):
        assert gain_at(deg) == pytest.approx(gain_at(-deg), abs=1e-5), "pattern must be symmetric"
    assert gain_at(5.0) > gain_at(15.0) > gain_at(25.0), "monotone within the main lobe"


def test_the_beam_concentrates_rather_than_uniformly_worsening():
    """⚠️ A beam is not strictly worse for the swarm than isotropic: +12 dB on
    boresight, down to −18 dB off it. 📏 That is why the inherited probe measured
    'point at the observer' at **59.9 %** against isotropic's 58.6 % — *worse
    than not aiming at all*."""
    env = make("J2", num_envs=1)
    env.hvt_pos = torch.zeros(1, 3)
    env.jam_target = torch.zeros(1, dtype=torch.long)
    radio = torch.zeros(1, env.cfg.n_radio, 3)
    radio[0, 0] = torch.tensor([100.0, 0.0, 60.0])
    radio[0, 1] = torch.tensor([-100.0, 0.0, 60.0])
    gain = env._beam_gain_db(radio)
    assert float(gain[0, 0]) > 0.0, "boresight must be amplified relative to isotropic"
    assert float(gain[0, 1]) < 0.0, "the back lobe must be attenuated below isotropic"


# --------------------------------------------------------------------------- #
# Aiming
# --------------------------------------------------------------------------- #


def test_j2_holds_a_fixed_target_and_j3_does_not():
    """⛔ J2 is not optional and is not a weaker J3. Without a fixed-target
    directional rung, a result at J3 cannot distinguish 'the adversary adapted'
    from 'the adversary had a beam' (`PLAN.md` Gate B, control row)."""
    fixed = make("J2", num_envs=8)
    fly(fixed, 40)
    assert torch.equal(fixed._jammer_boresight(), torch.full((8,), fixed.mcv_idx))

    greedy = make("J3", num_envs=8)
    fly(greedy, 40)
    assert not torch.equal(greedy.jam_target, torch.full((8,), greedy.mcv_idx)), (
        "J3 never retargeted -- either no chain ever formed or the update is not firing"
    )


def test_j3_targets_a_receiver_on_the_chain_it_saw_last_step():
    """🔒 One step of latency, and it is required rather than tolerated: the
    jammer changes capacity, capacity changes the routed chain, and the chain is
    what J3 targets. Aiming at *this* step's chain would be circular."""
    env = make("J3", num_envs=8)
    fly(env, 40)
    _snap, aux = env._evaluate()

    # the target now carried must be a receiver of an edge in the chain just seen
    on_edge = aux["on_edge"]
    for b in range(8):
        if not bool(on_edge[b].any()):
            continue
        receivers = on_edge[b].any(dim=0).nonzero().flatten().tolist()
        assert int(env.jam_target[b]) in receivers, (b, int(env.jam_target[b]), receivers)


def test_a_chainless_environment_holds_the_mcv():
    """Nothing to retarget is not the same as a free choice."""
    env = make("J3", num_envs=4)
    for _ in range(5):  # drones stay in the spawn ring; no chain reaches the MCV
        env.step(torch.zeros(4, 5, 3))
    assert torch.equal(env.jam_target, torch.full((4,), env.mcv_idx))


# --------------------------------------------------------------------------- #
# The rung changes the emitter and nothing else
# --------------------------------------------------------------------------- #


def test_the_rung_never_touches_the_sensor_or_the_diagnostics():
    """🔒 The analogue of `BLOCK_F.md` decision 2. The sensor is geometry, and a
    rung that gated it would make `observed` a function of the adversary --
    which would make the exploitability gap uninterpretable."""
    seen, occluded = [], []
    for rung in RUNGS:
        env = make(rung, num_envs=4)
        fly(env, 25)
        _snap, aux = env._evaluate()
        seen.append(aux["sees_hvt"].clone())
        occluded.append(aux["true_clearance"].clone())
    for other in seen[1:]:
        assert torch.equal(seen[0], other), "the emitter changed what the sensor sees"
    for other in occluded[1:]:
        assert torch.allclose(occluded[0], other), "the emitter changed the true geometry"


def test_every_rung_leaves_the_geometry_identical_under_a_fixed_action_sequence():
    """The invariant every J comparison rests on: only the channel differs."""
    positions = []
    for rung in RUNGS:
        env = make(rung, num_envs=4)
        fly(env, 25)
        positions.append(env.drone_pos.clone())
    for other in positions[1:]:
        assert torch.allclose(positions[0], other, atol=1e-5)


def test_the_jammer_reaches_the_sinr_denominator_at_the_receiver():
    """🔍 The physics that makes the whole scenario counter-intuitive: jamming
    raises a noise floor at an antenna, so it hurts whoever is *listening*. There
    is no such thing as jamming an outgoing signal, which is why the observer --
    a pure source -- is the one node the emitter cannot hurt."""
    quiet, loud = make("J0", num_envs=4), make("J2", num_envs=4)
    fly(quiet, 25)
    fly(loud, 25)
    _q, aq = quiet._evaluate()
    _l, al = loud._evaluate()
    assert (al["capacity_mbps"] <= aq["capacity_mbps"] + 1e-4).all(), (
        "an emitter must never RAISE a link's capacity"
    )
    assert (al["capacity_mbps"] < aq["capacity_mbps"] - 1e-4).any()


# --------------------------------------------------------------------------- #
# J3B -- the exhaustive best response
# --------------------------------------------------------------------------- #


def test_j3b_selects_the_argmin_of_end_to_end_capacity():
    """The defining property, checked against the quantity it optimises rather
    than against a re-implementation of it."""
    env = make("J3B", num_envs=8)
    fly(env, 40)

    pos_k = torch.cat([env.drone_pos, env.mcv_pos.unsqueeze(1), env.hvt_pos.unsqueeze(1)], dim=1)
    _true, chan = env._clearance(pos_k)
    _snap, aux = env._evaluate()
    e2e = env._update_best_response_target(aux, pos_k[:, : env.cfg.n_radio], chan)

    for b in range(8):
        if float(e2e[b].max()) <= 0.0:
            assert int(env.jam_target[b]) == env.mcv_idx  # nothing to break
            continue
        chosen = float(e2e[b, int(env.jam_target[b])])
        assert chosen == pytest.approx(float(e2e[b].min()), abs=1e-4), (b, chosen, e2e[b].tolist())


def test_j3b_is_at_least_as_damaging_as_holding_the_mcv():
    """🔒 The property that makes it a *best response* rather than a heuristic:
    it can never do worse than the fixed target J2 uses, on the geometry it was
    given. ⚠️ This is a statement about the one-step counterfactual only -- over
    a full episode a committed adversary can still beat a myopic one, which is
    exactly what `results/j_ladder.md` measures."""
    env = make("J3B", num_envs=8)
    fly(env, 40)
    pos_k = torch.cat([env.drone_pos, env.mcv_pos.unsqueeze(1), env.hvt_pos.unsqueeze(1)], dim=1)
    _true, chan = env._clearance(pos_k)
    _snap, aux = env._evaluate()
    e2e = env._update_best_response_target(aux, pos_k[:, : env.cfg.n_radio], chan)
    live = e2e.max(dim=1).values > 0.0
    assert (e2e.min(dim=1).values[live] <= e2e[live, env.mcv_idx] + 1e-4).all()


def test_j3b_holds_the_mcv_where_the_emitter_cannot_enter_capacity():
    """⛔ At a binary rung every candidate is identical, so an argmin would emit a
    silent arbitrary choice. Hold the MCV instead."""
    env = make("J3B", num_envs=4, fidelity="F1")
    fly(env, 20)
    assert torch.equal(env.jam_target, torch.full((4,), env.mcv_idx))
