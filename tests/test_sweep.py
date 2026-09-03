"""Guards on `scripts/sweep.py`.

A sweep is the easiest way in this repo to manufacture a result that does not
reproduce — 📏 `docs/inherited/BLOCK_G.md` records a 45.1 % headline from a 3-seed
cell that was the winner's curse. The three rules that prevent it (train split
only, worst-seed ranking, fresh-seed confirmation) are only worth anything if
they cannot be switched off by accident, so each is pinned here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sweep import AXES, build_score_cmd, build_train_cmd, config_key, parse_axis

SWEEP = ROOT / "scripts" / "sweep.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SWEEP), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_an_unknown_axis_fails_loudly() -> None:
    """⛔ A typo must not be forwarded to argparse and silently ignored.

    A misspelled axis that reached `train.py` as an unknown flag would either
    crash every run or -- if it happened to collide with a real prefix -- sweep a
    grid of identical configurations that all look distinct in the output.
    """
    out = run("--axis", "learnrate=1e-4", "--out", "/tmp/nope.jsonl")
    assert out.returncode != 0
    assert "unknown axis" in out.stdout + out.stderr


def test_confirm_seeds_may_not_overlap_search_seeds() -> None:
    """🔒 The confirmation measures the winner on seeds the selection never saw."""
    out = run(
        "--axis",
        "lr=1e-4",
        "--seeds",
        "0",
        "1",
        "--confirm-seeds",
        "1",
        "2",
        "--out",
        "/tmp/nope.jsonl",
    )
    assert out.returncode != 0
    assert "overlaps" in out.stdout + out.stderr


def test_a_fixed_flag_may_not_also_be_swept() -> None:
    """Both would be appended, the axis would win, and every cell would train alike."""
    out = run("--axis", "lr=1e-3", "--train-arg", "lr=3e-4", "--out", "/tmp/nope.jsonl")
    assert out.returncode != 0
    assert "also sweeps" in out.stdout + out.stderr


def _ns(**kw):
    import argparse

    base = {
        "device": "cpu",
        "timesteps": 1000,
        "run_root": Path("runs/sweep"),
        "train_arg": [],
        "eval_envs": 8,
        "stage": 4,
        "fidelity": "F4",
        "eval_jammer": "J1",
        "obs_history_for_eval": 1,
    }
    return argparse.Namespace(**(base | kw))


def test_the_search_never_trains_on_the_eval_split() -> None:
    """⛔ By construction: the built command cannot carry --eval-routes.

    `train.py` defaults to the train split, and the eval split is this project's
    only generalisation check -- already weak, and spent if a whole grid is scored
    against it. So the search has no way to reach it.
    """
    cmd = build_train_cmd(_ns(), {"lr": "3e-4"}, seed=0, tag="t")
    assert "--eval-routes" not in cmd
    assert "--lr" in cmd and "3e-4" in cmd


def test_fixed_flags_are_forwarded_and_bare_switches_take_no_value() -> None:
    cmd = build_train_cmd(
        _ns(train_arg=["num-envs=512", "no-curriculum"]), {"lr": "1e-4"}, seed=2, tag="t"
    )
    assert cmd[cmd.index("--num-envs") + 1] == "512"
    assert "--no-curriculum" in cmd
    # A bare switch must not swallow the next token as its value.
    assert cmd[cmd.index("--no-curriculum") + 1].startswith("--")


def test_the_search_scores_on_train_and_the_confirmation_on_eval() -> None:
    """🔒 One boolean carries the whole train/eval discipline, so it is pinned."""
    search = build_score_cmd(_ns(), [Path("a.pt")], "k", True, Path("o.jsonl"))
    confirm = build_score_cmd(_ns(), [Path("a.pt")], "k", False, Path("o.jsonl"))
    assert "--train-routes" in search
    assert "--train-routes" not in confirm


def test_scoring_passes_every_seed_as_one_grouped_condition() -> None:
    """--group scores each checkpoint on the SAME episodes, so the spread is seeds."""
    cmd = build_score_cmd(
        _ns(), [Path("a.pt"), Path("b.pt"), Path("c.pt")], "k", True, Path("o.jsonl")
    )
    assert "--group" in cmd
    assert cmd[cmd.index("--seeds") + 1] == "3"
    for p in ("a.pt", "b.pt", "c.pt"):
        assert p in cmd


def test_a_stacked_checkpoint_is_scored_at_a_matching_env_width() -> None:
    plain = build_score_cmd(_ns(), [Path("a.pt")], "k", True, Path("o.jsonl"))
    stacked = build_score_cmd(
        _ns(obs_history_for_eval=2), [Path("a.pt")], "k", True, Path("o.jsonl")
    )
    assert "--obs-history" not in plain
    assert stacked[stacked.index("--obs-history") + 1] == "2"


def test_ranking_is_on_the_worst_seed() -> None:
    """⛔ Not the median. A median-ranked sweep stacks two selection effects."""
    src = SWEEP.read_text()
    assert 'ok.sort(key=lambda r: r["worst"], reverse=True)' in src
    assert '"worst": min(mc)' in src


def test_a_failed_configuration_is_excluded_rather_than_partly_scored() -> None:
    """Scoring the seeds that survived would rank a 2-seed cell against 3-seed cells."""
    src = SWEEP.read_text()
    assert 'r.get("status") == "ok"' in src
    assert '"status": "failed"' in src


@pytest.mark.parametrize(
    ("spec", "want"),
    [
        ("lr=1e-4,3e-4", ("lr", ["1e-4", "3e-4"])),
        ("entropy= 0.0 , 0.01 ", ("entropy", ["0.0", "0.01"])),
    ],
)
def test_axis_parsing(spec: str, want: tuple[str, list[str]]) -> None:
    assert parse_axis(spec) == want


def test_axis_parsing_rejects_a_spec_with_no_values() -> None:
    with pytest.raises(SystemExit):
        parse_axis("lr=")
    with pytest.raises(SystemExit):
        parse_axis("lr")


def test_config_keys_are_stable_and_filesystem_safe() -> None:
    """The key becomes a run-directory name and the resume identity."""
    a = config_key({"lr": "3e-4", "entropy": "0.0"})
    b = config_key({"entropy": "0.0", "lr": "3e-4"})
    assert a == b, "key must not depend on dict order, or resume misses rows"
    assert "." not in a and "/" not in a and " " not in a
    assert config_key({"lr": "1e-3"}) != config_key({"lr": "3e-4"})


def test_every_axis_maps_to_a_flag_train_py_actually_accepts() -> None:
    """⛔ The whitelist must not drift away from the trainer it drives."""
    help_text = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    ).stdout
    for name, flag in AXES.items():
        assert flag in help_text, f"axis {name!r} maps to {flag}, which train.py does not accept"
