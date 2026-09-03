"""The jammer as drawn: does the beam point where the physics points it?

A rendered beam is only worth having if it agrees with `core.py`. A wedge that
confidently points at the wrong node is worse than no wedge, because it would be
believed -- and `docs/inherited/BLOCK_G.md` credits one render with overturning
four hypotheses drawn from aggregate statistics, so these figures get believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.core import JAMMER_BEAMWIDTH_DEG
from src.viz.episode import beam_nodes, beam_wedge, fly

STEPS = 40


@pytest.fixture(scope="module")
def traces() -> dict:
    return {r: fly(12, policy="b0", steps=STEPS, jammer=r) for r in ("J0", "J1", "J2", "J3B")}


def test_j0_draws_no_emitter_at_all(traces) -> None:
    """⚠️ `env.jammer_on` is the CURRICULUM's switch, not the rung's.

    At J0 there is no emitter, but the curriculum flag is still 1 -- so reading it
    alone would draw an emitter marker for an adversary that does not exist.
    """
    tr = traces["J0"]
    assert not np.asarray(tr.jammer_on).any()
    assert (np.asarray(tr.jam_target) < 0).all()
    assert all(beam_wedge(tr, i) is None for i in range(STEPS))


def test_an_isotropic_emitter_has_no_boresight(traces) -> None:
    """J1 radiates but has no direction, so there is nothing to point a wedge at."""
    tr = traces["J1"]
    assert np.asarray(tr.jammer_on).all()
    assert (np.asarray(tr.jam_target) < 0).all()
    assert all(beam_wedge(tr, i) is None for i in range(STEPS))


def test_j2_holds_the_mcv_and_only_the_mcv(traces) -> None:
    """🔒 The fixed-target control rung. If it ever moves, it is not a control."""
    tr = traces["J2"]
    tgts = np.asarray(tr.jam_target)
    mcv_idx = tr.pos.shape[1]  # drones 0..N-1, MCV at N
    assert set(tgts.tolist()) == {mcv_idx}


def test_j3b_actually_retargets(traces) -> None:
    """A best response that never changes target would be J2 wearing a costume."""
    assert len(set(np.asarray(traces["J3B"].jam_target).tolist())) > 1


@pytest.mark.parametrize("rung", ["J2", "J3B"])
def test_the_wedge_points_at_the_node_the_physics_targets(traces, rung: str) -> None:
    """⛔ The one that matters: bearing must agree with the targeted node exactly.

    `beam_nodes` must use `core.py`'s index order -- drones `0..N-1`, MCV at `N` --
    or the wedge lands on a confidently wrong drone.
    """
    tr = traces[rung]
    checked = 0
    for i in range(STEPS):
        spec = beam_wedge(tr, i)
        if spec is None:
            continue
        apex, half, bearing = spec
        node = beam_nodes(tr, i)[int(tr.jam_target[i])]
        d = node - apex
        want = float(np.degrees(np.arctan2(d[1], d[0])))
        # Wrapped difference, so 179.9 deg vs -179.9 deg is 0.2 and not 359.8.
        assert abs((bearing - want + 180) % 360 - 180) < 1e-6
        assert half == JAMMER_BEAMWIDTH_DEG
        checked += 1
    assert checked > 0, "no beam was drawn at all, so nothing was verified"


def test_the_emitter_rides_the_target(traces) -> None:
    """`core.py`: bearings are measured from the HVT, so the apex is the HVT."""
    tr = traces["J3B"]
    for i in range(0, STEPS, 7):
        spec = beam_wedge(tr, i)
        if spec is not None:
            assert np.allclose(spec[0], tr.hvt[i])


def test_beam_nodes_uses_the_core_index_order(traces) -> None:
    tr = traces["J2"]
    nodes = beam_nodes(tr, 0)
    assert nodes.shape == (tr.pos.shape[1] + 1, 2)
    assert np.allclose(nodes[:-1], tr.pos[0, :, :2])
    assert np.allclose(nodes[-1], tr.mcv)


def test_the_rung_is_carried_on_the_trace(traces) -> None:
    """A figure of a J2 episode that does not say so is a figure of a beam with
    no explanation attached -- the same reason `fidelity` is carried."""
    for rung, tr in traces.items():
        assert tr.jammer == rung
