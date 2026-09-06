"""Export every actor rung to ONNX and check it numerically. RQ3 / Gate C.

    uv run python scripts/export_onnx.py                    # all three rungs
    uv run python scripts/export_onnx.py --checkpoint runs/val-gnn-deep-s0/checkpoint.pt

⭐ **`PLAN.md` §7 item 1: the only deliverable that cannot fail.** The export risk
is unretired, and `PLAN.md`'s risk table names the specific fear — *"PyTorch
Geometric will not export"* — with the instruction to attempt it **locally, before
touching the Jetson**. This script is that attempt.

🔍 **What it actually answers.** The Orin pipeline is ONNX → TensorRT, and the
first of those two steps is the one that can fail for architectural reasons. So:

1. does each rung export at all,
2. does the exported graph compute the **same function** as the PyTorch module,
3. and does it stay correct at a **different `N`** — the off-N transfer columns
   are evaluated with the actor alone, so an export that bakes in `N = 5` would
   silently invalidate them.

⚠️ **Only the actor is exported, and that is not a shortcut.** `SwarmCritic` is
discarded at evaluation (`src/models/critic.py`: *"zero-shot transfer to N in
{3, 8} runs the actor alone and the critic is discarded"*), so it never flies.

🔒 **The exported signature is `flat -> mean`, deterministic.** `act` samples and
`evaluate` needs an action argument; neither is what a flight controller calls.
`evaluate.py` scores the deterministic mean, so the mean is what deploys, and
`log_std` never leaves the training host.

⚠️ **`torch.onnx.export` needs `onnx`, `onnxruntime` and `onnxscript`**, which are
in the **`dev` extra** and not in the runtime dependencies. `uv sync --extra dev`.
They are not on `AGENTS.md`'s heavy-dependency list — no simulator and no RL
framework — but they are an addition and this comment is the flag.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.core import FLAT_DIM
from src.models import ARCHITECTURES, SwarmActor, parameter_count


def build(architecture: str, checkpoint: Path | None) -> tuple[SwarmActor, int]:
    """The actor, and the observation width it consumes.

    🔒 Reads `obs_history`, `tanh_mean` and `layer_norm` off the checkpoint's top
    level, exactly as `scripts/eval_policy.py` does. All three change what the
    network *computes* without changing the shape of its state dict, so a loader
    that misses them exports a different function and `load_state_dict` raises
    nothing.
    """
    if checkpoint is None:
        actor = SwarmActor(architecture=architecture)
        return actor.eval(), FLAT_DIM
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    frames = int(blob.get("obs_history", 1))
    actor = SwarmActor(
        architecture=blob["architecture"],
        hidden=blob.get("hidden"),
        obs_history=frames,
        tanh_mean=bool(blob.get("tanh_mean", True)),
        layer_norm=bool(blob.get("layer_norm", False)),
        sde=bool(blob.get("sde", False)),
    )
    actor.load_state_dict(blob["policy"])
    return actor.eval(), FLAT_DIM * frames


class DeterministicActor(torch.nn.Module):
    """`flat -> mean`. What deploys; see the module docstring."""

    def __init__(self, actor: SwarmActor):
        super().__init__()
        self.actor = actor

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        return self.actor(flat)[0]


def export_one(
    architecture: str, checkpoint: Path | None, out_dir: Path | None, dynamo: bool = False
) -> dict:
    import onnx
    import onnxruntime as ort

    actor, width = build(architecture, checkpoint)
    module = DeterministicActor(actor).eval()
    row: dict = {
        "exporter": "dynamo" if dynamo else "torchscript",
        "architecture": actor.architecture,
        "actor_params": parameter_count(actor),
        "obs_width": width,
    }

    # 🔒 A batch of 5 rows to trace with, and 15 to check with. `PPOTrainer` folds
    # the drone axis into the batch axis, so the actor never sees `N` -- it sees
    # `num_envs * N` independent rows. A dynamic batch axis is therefore all that
    # off-N transfer needs, and checking at a different row count is what proves
    # `N` was not baked in.
    sample = torch.randn(5, width)
    target = out_dir or Path(tempfile.mkdtemp())
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"actor-{actor.architecture}-{'dynamo' if dynamo else 'ts'}.onnx"

    try:
        torch.onnx.export(
            module,
            (sample,),
            str(path),
            input_names=["flat"],
            output_names=["mean"],
            dynamic_axes={"flat": {0: "rows"}, "mean": {0: "rows"}},
            dynamo=dynamo,
        )
    except Exception as exc:  # noqa: BLE001 -- the failure IS the result here
        row |= {"exported": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        return row

    row["exported"] = True
    row["bytes"] = path.stat().st_size
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    row["opset"] = model.opset_import[0].version
    row["nodes"] = len(model.graph.node)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for rows in (5, 15, 24):  # N = 5 traced; 3 and 8 drones at 3 envs
        probe = torch.randn(rows, width)
        with torch.no_grad():
            expected = module(probe).numpy()
        try:
            got = session.run(["mean"], {"flat": probe.numpy()})[0]
        except Exception as exc:  # noqa: BLE001 -- the failure IS the result
            row |= {
                "numerically_equal": False,
                "failed_at_rows": rows,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
            return row
        worst = max(worst, float(np.abs(expected - got).max()))
    row["max_abs_error"] = worst
    # float32 through two different kernels; 1e-5 is the standard tolerance and
    # anything above it means the graph is not the module.
    row["numerically_equal"] = worst < 1e-5
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", nargs="+", default=list(ARCHITECTURES), choices=ARCHITECTURES)
    ap.add_argument("--checkpoint", type=Path, default=None, help="export a trained policy instead")
    ap.add_argument("--out-dir", type=Path, default=None, help="keep the .onnx files here")
    ap.add_argument(
        "--exporter",
        nargs="+",
        default=["torchscript", "dynamo"],
        choices=["torchscript", "dynamo"],
        help="⚠️ Both, by default. torch 2.13 defaults to the dynamo exporter and "
        "the two do NOT produce the same graph -- the TorchScript tracer bakes the "
        "traced batch size into `expand`, which is exactly the failure mode the "
        "relational rungs hit",
    )
    a = ap.parse_args()

    rows = [
        export_one(arch, a.checkpoint, a.out_dir, dynamo=(exporter == "dynamo"))
        for exporter in a.exporter
        for arch in a.arch
    ]
    for row in rows:
        print(json.dumps(row))
    ok = all(r.get("exported") and r.get("numerically_equal") for r in rows)
    print(f"\n{'ALL RUNGS EXPORT' if ok else 'AT LEAST ONE RUNG FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
