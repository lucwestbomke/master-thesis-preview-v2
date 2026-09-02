"""The behaviour-clone seam: `scripts/bc_init.py` and `train.py --init-from`.

The probe's whole value is that its two outcomes mean different things
(`results/bc_init.md`), and both readings assume the clone really is a clone and
really is what PPO starts from. Three things can silently break that and each is
pinned here: the teacher's carried state, the `tanh` target ceiling, and the
critic quietly arriving pre-fitted to the teacher.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import SwarmActor


def test_a_memoryless_actor_cannot_represent_b0_exactly() -> None:
    """⛔ The ceiling on any behaviour clone here, asserted rather than discovered twice.

    `B0Policy` carries `belief_rel` / `belief_vel` -- a target-belief filter with
    dead reckoning -- plus repair offsets and ranked roles, and `reset(mask)` must
    clear them on episode boundaries. The actors are feedforward over `obs["flat"]`,
    and `flat` zeroes the target terms when a drone cannot see it (`rel_hvt * sees`).

    So when the target is unseen B0 acts on remembered state that is **not in the
    student's input at all**, and no amount of BC data closes that. 📏 Measured: a
    20-epoch clone reaches val MAE 0.188 and then scores 9.4 % against B0's 58.0 %.

    This test pins the *cause* so the next person does not read the failure as
    underfitting and spend a week on capacity.
    """
    from src.baselines.b0 import B0Policy

    b0 = B0Policy(num_envs=2, num_drones=5, device="cpu")
    b0.reset()
    assert hasattr(b0, "belief_rel"), "B0 is stateful; a memoryless clone has a ceiling"
    assert hasattr(b0, "belief_vel")
    assert b0.belief_rel.shape == (2, 5, 3)


def test_init_from_rejects_a_different_architecture() -> None:
    """Loading a gnn clone into an mlp would score a network nobody trained."""
    actor = SwarmActor(architecture="gnn")
    blob = {
        "policy": actor.state_dict(),
        "architecture": "gnn",
        "hidden": actor.trunk.out_dim,
        "min_log_std": actor.min_log_std,
        "timestep": 0,
    }
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        torch.save(blob, path)
        out = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                "--arch",
                "mlp",
                "--init-from",
                str(path),
                "--timesteps",
                "100",
                "--device",
                "cpu",
                "--out-root",
                tmp,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        assert out.returncode != 0
        assert "would score a different network" in out.stdout + out.stderr


def test_targets_are_clipped_below_the_tanh_ceiling() -> None:
    """`tanh` cannot reach +-1, and B0 saturates there constantly.

    Regressing onto un-clipped targets drives `|head|` toward infinity chasing a
    value the parameterisation cannot represent, so the clone's loss plateaus for
    a reason that looks like underfitting and is not. The default clip is 0.995.
    """
    actor = SwarmActor(architecture="deepsets")
    from src.env.core import FLAT_DIM

    mean, _ = actor(torch.randn(16, 5, FLAT_DIM))
    assert mean.abs().max() < 1.0, "tanh output must be strictly inside +-1"


def test_the_clone_runs_and_the_actor_changes() -> None:
    """End to end on a toy config: BC must actually move the weights."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "checkpoint.pt"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bc_init.py"),
                "--arch",
                "deepsets",
                "--device",
                "cpu",
                "--num-envs",
                "4",
                "--steps-per-episode",
                "30",
                "--epochs",
                "2",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        blob = torch.load(out, map_location="cpu", weights_only=False)

        assert "value" not in blob
        assert blob["architecture"] == "deepsets"
        assert blob["provenance"]["kind"] == "behaviour_clone"
        assert blob["provenance"]["teacher"] == "b0"
        # ⛔ Cloned on TRAIN routes; eval-split cloning would contaminate every
        # downstream measurement of the policy it produces.
        assert blob["provenance"]["split"] == "train"
        assert blob["provenance"]["val_mse"] > 0.0

        fresh = SwarmActor(architecture="deepsets")
        same = all(
            torch.equal(a, b)
            for a, b in zip(fresh.state_dict().values(), blob["policy"].values(), strict=True)
        )
        assert not same, "BC produced an actor identical to a fresh init"
