"""The one test `docs/REDUCTION.md` requires to come back with the trainer.

`test_every_PBRS_safe_reward_knob_is_settable_from_the_command_line` was dropped
with the predecessor's `test_train.py`, and it is genuinely load-bearing. Every
field of `RewardWeights` that is not an objective weight and not a physical
reference lives inside `Phi`, is optimum-preserving by the PBRS proof, and must
therefore be settable from the CLI -- **because a knob that cannot be set cannot
be swept.**

📏 It has caught two real misses of exactly the same shape:

* `--w-relay` shipped with its config field, its `build_weights` wiring and its
  call site -- and **no `add_argument`**. It failed on a GPU box as
  `unrecognized arguments`, one command into a 5-seed sweep.
* `w_approach` / `w_observe` / `w_link` were *documented as free* while being
  reachable from nowhere: no `build_weights` branch and no flag. A whole session
  recommended tuning them.

⚠️ `PHI_V2` adds `n_cover_samples`, an `int` where every previous knob was a
`float`, which the predecessor's hand-written flag loop would not have covered.
The list is derived; it is never hand-listed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from ..env.reward import (
    OBJECTIVE_WEIGHTS,
    PHYSICAL_REFERENCES,
    RewardWeights,
    pbrs_safe_fields,
)


def _train_module():
    """Import `scripts/train.py` as a module, the way the shell invokes it."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_train_entry"] = module
    spec.loader.exec_module(module)
    return module


def test_every_pbrs_safe_reward_knob_is_settable_from_the_command_line():
    """The whole point: presence AND effect, for every derived field."""
    train = _train_module()
    parser = train.build_parser()
    base = RewardWeights()

    for name in pbrs_safe_fields():
        current = getattr(base, name)
        # A value that is definitely different, in the field's own type.
        probe = int(current) + 3 if isinstance(current, int) else float(current) + 0.125

        args = parser.parse_args([train.flag(name), str(probe)])
        weights = train.build_weights(args)

        assert getattr(weights, name) == probe, (
            f"{train.flag(name)} parses but does not reach RewardWeights.{name} -- "
            "this is the --w-relay failure, which shipped everywhere except the flag"
        )
        assert isinstance(getattr(weights, name), type(current)), (
            f"{train.flag(name)} changed the type of {name}: "
            f"{type(getattr(weights, name))} not {type(current)}"
        )


def test_the_knob_list_is_derived_from_the_dataclass_and_not_hand_written():
    """A hand-written list is how `n_cover_samples` would be missed. Adding a
    field to `RewardWeights` must land it in the swept set automatically."""
    every = {f.name for f in dataclasses.fields(RewardWeights)}
    derived = set(pbrs_safe_fields())
    assert derived == every - OBJECTIVE_WEIGHTS - PHYSICAL_REFERENCES
    assert "n_cover_samples" in derived
    assert isinstance(RewardWeights().n_cover_samples, int)


#: 🔒 The objective weights that MAY be set from the command line, each with the
#: argument that licenses it. ⛔ This is a closed list, and adding to it is a
#: design decision that belongs in `docs/inherited/DECISIONS.md` -- not a
#: convenience. Every other objective weight is set by the behavioural orderings
#: in `weight_constraints_satisfied()` and changing one changes the mission.
SWEEPABLE_OBJECTIVE_WEIGHTS: dict[str, str] = {
    "battery_variance": (
        "lambda -- the single weight docs/REWARD.md's method leaves free, because "
        "the orderings bound it rather than pinning it"
    ),
    "w_difference": (
        "the difference reward D_i = G(z) - G(z_-i). Licensed by a DIFFERENT "
        "argument from lambda's, and the distinction is the point: G(z_-i) does "
        "not depend on a_i at all, so d(D_i)/d(a_i) = d(G)/d(a_i) exactly. D_i is "
        "FACTORED (Wolpert & Tumer 2002), so every agent's best response to fixed "
        "others is unchanged and the equilibrium of the team objective cannot "
        "move. It is not optimum-preserving by the PBRS proof -- it is not inside "
        "Phi -- which is why it is an objective weight and not a derived flag"
    ),
}


def test_objective_weights_get_no_flags_except_the_ones_the_design_permits():
    """⛔ Objective weights change what is OPTIMAL. They are set by the
    behavioural orderings in `weight_constraints_satisfied()`, not swept.

    ⚠️ There are now **two** exceptions and they are licensed by two different
    arguments -- see `SWEEPABLE_OBJECTIVE_WEIGHTS`. The guard is kept as a closed
    allow-list rather than relaxed to "objective weights may have flags", because
    the whole value of it is that a third addition has to be argued for.
    """
    train = _train_module()
    parser = train.build_parser()
    known = {action.dest for action in parser._actions}

    for name in OBJECTIVE_WEIGHTS | PHYSICAL_REFERENCES:
        if name in SWEEPABLE_OBJECTIVE_WEIGHTS:
            assert name in known, (
                f"{name} is licensed by SWEEPABLE_OBJECTIVE_WEIGHTS but has no flag: "
                f"{SWEEPABLE_OBJECTIVE_WEIGHTS[name]}"
            )
            continue
        assert name not in known, f"{name} is an objective weight and must not have a flag"
        assert f"phi_{name}" not in known

    # 🔒 The allow-list may not drift away from the weights it is about.
    assert set(SWEEPABLE_OBJECTIVE_WEIGHTS) <= OBJECTIVE_WEIGHTS


def test_a_reward_that_breaks_a_behavioural_ordering_is_refused():
    """The orderings are how the objective weights were SET. A run that violates
    one is optimising a different mission, and it should stop at the CLI rather
    than at the reading of its results."""
    train = _train_module()
    parser = train.build_parser()
    # potential_scale outside (3, 50) breaks `potential_guides_without_dominating`
    args = parser.parse_args(["--potential-scale", "1000.0"])
    with pytest.raises(SystemExit, match="behavioural orderings"):
        train.build_weights(args)


def test_phi_v2_is_reachable_and_is_not_the_default():
    """🔒 `PHI_V2` ships OFF: `w_standoff = w_cover = 0` reproduces the shipped
    potential bitwise. Promoting it is `docs/REDUCTION.md` task 3, a separate
    decision from building the trainer."""
    train = _train_module()
    parser = train.build_parser()
    assert train.build_weights(parser.parse_args([])).w_cover == 0.0
    assert train.build_weights(parser.parse_args(["--phi-v2"])).w_cover == 0.4


def test_the_device_is_never_silently_downgraded():
    """⛔ `AGENTS.md`: guard device selection; never silently degrade a real run
    to CPU. A run that quietly moved to CPU would also draw *different episodes*
    for the same seed, so its numbers would not compare with anything."""
    import torch

    train = _train_module()
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="CUDA is not available"):
            train.resolve_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(SystemExit, match="MPS is not available"):
            train.resolve_device("mps")
    assert train.resolve_device("cpu").type == "cpu"
