"""`eval_policy.py --b0-config` — the harness for PLAN.md §7 runs 2 and 3.

Both runs vary a `B0Config` field and score the result through the same harness
every other number came from. Two things must hold or the rows are worse than
useless: an unknown field has to raise rather than silently score the shipped B0,
and an overridden arm must never share a row name with the baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_policy import b0_config, b0_label
from src.baselines.b0 import B0Config


def test_no_overrides_is_the_shipped_config() -> None:
    assert b0_config([]) == B0Config()


def test_a_float_field_is_typed_off_the_dataclass() -> None:
    cfg = b0_config(["repair_amplitude_m=50"])
    assert cfg.repair_amplitude_m == 50.0
    assert isinstance(cfg.repair_amplitude_m, float)
    # everything else untouched
    assert cfg.repair_score == B0Config().repair_score


def test_a_string_field_is_not_coerced() -> None:
    assert b0_config(["repair_score=clearance"]).repair_score == "clearance"


def test_several_overrides_compose() -> None:
    cfg = b0_config(["repair_score=clearance", "repair_amplitude_m=100"])
    assert cfg.repair_score == "clearance"
    assert cfg.repair_amplitude_m == 100.0


def test_an_unknown_field_raises() -> None:
    """⛔ Silently ignoring it would score the SHIPPED B0 under an ablation's name.

    That row would then look like an arm and be the control -- the worst possible
    failure for a comparison, and invisible in the output.
    """
    with pytest.raises(SystemExit):
        b0_config(["repair_scores=clearance"])
    with pytest.raises(SystemExit):
        b0_config(["typo=1"])


def test_an_overridden_arm_never_shares_a_name_with_the_baseline() -> None:
    assert b0_label("b0", []) == "b0"
    assert b0_label("b0", ["repair_score=clearance"]) != "b0"
    assert "repair_score=clearance" in b0_label("b0", ["repair_score=clearance"])


def test_the_label_is_order_independent() -> None:
    """Two invocations of the same arm must produce the same row name, or a
    resumed or re-run comparison silently splits into two conditions."""
    a = b0_label("b0", ["repair_score=clearance", "repair_amplitude_m=50"])
    b = b0_label("b0", ["repair_amplitude_m=50", "repair_score=clearance"])
    assert a == b


def test_amplitude_zero_disables_the_loop() -> None:
    """Run 3's floor: 0 is the no-repair arm, matching b0-geodesic's mechanism."""
    assert b0_config(["repair_amplitude_m=0"]).repair_amplitude_m == 0.0
