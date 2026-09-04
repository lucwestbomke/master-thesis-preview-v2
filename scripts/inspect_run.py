"""Print a training run's diagnostics as a table. Read a collapse, don't guess at it.

    uv run python scripts/inspect_run.py runs/gateD-budget/sw-gae_lambda0p95__*-s0

⚠️ **This exists because a five-knob screening arm collapsed to 13 % and there was
no quick way to see which knob did it.** `scripts/train.py` prints these columns
while it runs and then they are only in `log.jsonl`, which nobody reads by hand.

🔒 The columns that diagnose a PPO collapse, in the order you should read them:

* `lr_actor`     -- runaway adaptive LR. ⚠️ `--target-kl` raises it 1.5x per round
                    whenever the round's MEAN KL is under half the target, and the
                    mean is taken over ALL epochs including epoch 0, where the
                    ratio is 1 and the KL is ~0 by construction. That biases the
                    controller UPWARD.
* `approx_kl`    -- how far the policy moved. Healthy PPO is 0.01-0.02 per round.
                    Under 0.005 is a policy that cannot move; over ~0.05 is one
                    that is being thrown around.
* `grad_kept`    -- fraction of the ACTOR gradient surviving the norm clip.
* `log_std`      -- exploration. Monotone decay to the floor is normal; a RISE is
                    the signature of a policy being pushed off its optimum.
* `explained_variance` -- critic health. A collapse here is a different failure
                    from a policy collapse and they are easy to confuse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = (
    ("progress", "prog", "{:6.3f}"),
    ("mission_capable", "capable", "{:8.4f}"),
    ("observed", "observed", "{:9.4f}"),
    ("lr_actor", "lr_actor", "{:9.2e}"),
    ("approx_kl", "approx_kl", "{:10.5f}"),
    ("log_std", "log_std", "{:8.3f}"),
    ("grad_kept", "grad_kept", "{:10.3f}"),
    ("grad_norm_actor", "g_actor", "{:8.3f}"),
    ("explained_variance", "expl_var", "{:9.3f}"),
    ("at_boundary", "boundary", "{:9.3f}"),
    ("return_spread_between_drones", "drone_sd", "{:9.4f}"),
)


def show(path: Path, every: int) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    prov = next((r["provenance"] for r in rows if "provenance" in r), {})
    rows = [r for r in rows if "progress" in r]
    if not rows:
        print(f"  {path}: no logged rows")
        return

    interesting = (
        "lr",
        "gae_lambda",
        "mini_batch_size",
        "target_kl",
        "grad_norm_clip_critic",
        "orthogonal_init",
        "min_log_std",
        "tanh_mean",
        "layer_norm",
        "cue_mode",
        "mask_broadcast_obs",
    )
    cfg = {k: prov[k] for k in interesting if k in prov and prov[k] not in (None, False)}
    print(f"\n=== {path.parent.name} ===")
    print(f"  {cfg}")
    print("  " + "".join(f"{label:>10}" for _, label, _ in COLUMNS))
    for r in rows[:: max(1, every)]:
        cells = []
        for key, _, fmt in COLUMNS:
            v = r.get(key)
            cells.append(f"{'':>10}" if v is None else f"{fmt.format(v):>10}")
        print("  " + "".join(cells))

    peak = max(rows, key=lambda r: r.get("mission_capable", -1))
    print(
        f"  peak capable {peak.get('mission_capable', float('nan')):.4f} "
        f"at progress {peak['progress']:.3f}   ->   final "
        f"{rows[-1].get('mission_capable', float('nan')):.4f}"
    )
    kls = [r["approx_kl"] for r in rows if "approx_kl" in r]
    lrs = [r["lr_actor"] for r in rows if "lr_actor" in r]
    if kls:
        print(f"  approx_kl  min {min(kls):.5f}  max {max(kls):.5f}")
    if lrs:
        print(f"  lr_actor   min {min(lrs):.2e}  max {max(lrs):.2e}  (started 3.00e-04)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories or log.jsonl paths")
    ap.add_argument("--every", type=int, default=1, help="print every Nth logged row")
    a = ap.parse_args()
    for target in a.runs:
        path = target if target.suffix == ".jsonl" else target / "log.jsonl"
        if path.exists():
            show(path, a.every)
        else:
            print(f"  {path}: not found")


if __name__ == "__main__":
    main()
