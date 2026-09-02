"""The run-length logic behind `scripts/measure_memory_horizon.py`.

The measurement decides whether to rebuild recurrence -- a change
`docs/REDUCTION.md` task 4 removed and which carries the highest bug density
available in this repo. So the statistic it turns on is worth pinning, and the
fiddly part is not the arithmetic but **which runs count**: an unseen run before
a drone's first sighting is not an interval memory could have bridged, and
including it would inflate the horizon with time where memory is definitionally
useless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.measure_memory_horizon import gaps


def _series(bits: list[int], hvt: list[float] | None = None):
    """One drone, one env. `hvt` is an x-position per step; y is held at 0."""
    sees = np.array(bits, dtype=bool).reshape(-1, 1, 1)
    x = np.arange(len(bits), dtype=float) if hvt is None else np.asarray(hvt, dtype=float)
    xy = np.stack([x, np.zeros_like(x)], axis=-1).reshape(-1, 1, 2)
    return sees, xy


def test_a_gap_before_the_first_sighting_is_not_counted() -> None:
    """🔒 The restriction the whole measurement rests on."""
    lengths, _ = gaps(*_series([0, 0, 0, 1, 1]))
    assert lengths.size == 0, "an unseen run with nothing remembered is not bridgeable"


def test_one_bracketed_gap_is_measured_by_its_length() -> None:
    lengths, _ = gaps(*_series([1, 0, 0, 0, 1]))
    assert lengths.tolist() == [3]


def test_a_gap_running_to_the_end_of_the_episode_still_counts() -> None:
    """It is blind time the belief would have had to cover, reacquired or not."""
    lengths, _ = gaps(*_series([1, 0, 0]))
    assert lengths.tolist() == [2]


def test_several_gaps_are_reported_separately() -> None:
    lengths, _ = gaps(*_series([1, 0, 1, 0, 0, 0, 1, 0]))
    assert sorted(lengths.tolist()) == [1, 1, 3]


def test_a_drone_that_always_sees_has_no_gaps() -> None:
    lengths, _ = gaps(*_series([1, 1, 1, 1]))
    assert lengths.size == 0


def test_displacement_is_measured_from_the_last_sighting_to_reacquisition() -> None:
    """⚠️ This is what stops gap LENGTH alone over-selling recurrence.

    A long gap over a short displacement leaves a usable belief; a long gap over a
    large one leaves a stale belief and a reacquisition problem instead.
    """
    # seen at t=0 (x=0), blind t=1..3, reacquired t=4 (x=100).
    _, disps = gaps(*_series([1, 0, 0, 0, 1], hvt=[0, 10, 40, 70, 100]))
    assert disps.tolist() == [100.0]


def test_displacement_of_a_stationary_target_is_zero_however_long_the_gap() -> None:
    lengths, disps = gaps(*_series([1, 0, 0, 0, 0, 1], hvt=[5, 5, 5, 5, 5, 5]))
    assert lengths.tolist() == [4]
    assert disps.tolist() == [0.0]


def test_drones_and_envs_are_walked_independently() -> None:
    """Runs must not be joined across the batch or the swarm dimension."""
    sees = np.zeros((4, 2, 2), dtype=bool)
    sees[:, 0, 0] = [1, 0, 0, 1]  # one gap of 2
    sees[:, 1, 1] = [1, 0, 1, 1]  # one gap of 1
    xy = np.zeros((4, 2, 2), dtype=float)
    lengths, _ = gaps(sees, xy)
    assert sorted(lengths.tolist()) == [1, 2]
