r"""Behaviour-clone B0 into an actor, so PPO can be started from B0's basin.

⛔ **This is a falsification test, not a rescue attempt**, and the distinction is
what makes it worth running at all. `results/credit_assignment.md` measured that
**0.04-0.16 %** of the return variance differs between drones, so the advantage
is ~99.9 % identical across the swarm and role credit has almost no channel to
travel through. That predicts something sharp and testable:

> 🔒 A policy placed *inside* B0's role-differentiated basin has essentially no
> gradient signal to **stay** differentiated, and should decay back toward the
> symmetric 218 m stand-off under continued PPO.

If it decays, the credit finding is confirmed behaviourally and the critic work
`PLAN.md` §3 names is warranted. If it holds, 0.16 % was apparently enough, the
credit finding is much less binding than it looks, and ⛔ the critic axis closes
before anyone builds a counterfactual baseline. **Both outcomes are results.**

## Why cloning B0 is legitimate here rather than circular

🔒 `AGENTS.md`'s constraint is that **B0 sees only `obs["flat"]`** -- the same
tensor the actors consume, plus its own carried state, never `env.hvt_pos`. So
the teacher is a function of exactly the student's input, and the clone is a pure
imitation problem with no information leak. 📏 The measured cost of that
restriction is 0.6 pp.

⚠️ A BC-initialised policy is **not** a like-for-like entry in any RQ2 or Gate B
table -- it has seen a scripted teacher and the other arms have not. It is a
probe, and anything it produces is labelled as one.

## The two things that make this a clone and not a fit to noise

**`B0Policy` is stateful.** It carries a target belief and a repair timer, and
`reset(mask)` must be called on episode boundaries. `BLOCK_E` §14 records
forgetting that as the failure mode. This script runs with `auto_reset=False` and
one episode per environment, so the only reset is the first.

**The actor's mean is `tanh(...)`, which cannot reach the +-1 B0 saturates at.**
Regressing onto raw B0 actions therefore drives `|head|` to infinity chasing a
target the parameterisation cannot represent. Targets are clipped to +-`--clip`
(default 0.995) for that reason, which costs 0.4 % of the action range and keeps
the gradient finite.

Usage:

    uv run python scripts/bc_init.py --device mps --steps 40000 --out runs/bc-gnn/checkpoint.pt
    uv run python scripts/train.py --init-from runs/bc-gnn/checkpoint.pt --tag bc-probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.b0 import B0Policy
from src.env.core import STAGES, BatchedSwarmEnv, EnvConfig
from src.models import ARCHITECTURES, SwarmActor, parameter_count


def make_env(a, seed: int) -> BatchedSwarmEnv:
    weights = tuple(1.0 if i == a.stage - 1 else 0.0 for i in range(len(STAGES)))
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=a.num_envs,
            num_drones=a.num_drones,
            device=a.device,
            seed=seed,
            fidelity=a.fidelity,
            jammer=a.jammer,
            # ⛔ TRAIN routes. This is an initialisation step, not a reported
            # number, and cloning on the eval split would contaminate every
            # downstream measurement of the policy it produces.
            eval_routes=False,
            auto_reset=False,
            stage_weights=weights,
            compile_occlusion=a.device != "cpu",
        )
    )


@torch.no_grad()
def collect(a, seed: int) -> tuple[Tensor, Tensor]:
    """Roll B0 out and return `(observations, actions)` flattened to rows.

    One episode per environment, so `B0Policy.reset()` is called once and the
    carried belief is never stale -- the `BLOCK_E` §14 failure mode.
    """
    env = make_env(a, seed)
    steps = a.steps_per_episode or STAGES[a.stage - 1].episode_steps
    b0 = B0Policy(num_envs=env.cfg.num_envs, num_drones=env.cfg.num_drones, device=env.device)
    b0.reset()

    obs = env.reset()
    xs, ys = [], []
    for _ in range(steps):
        flat = obs["flat"]
        action = b0.act(flat)
        xs.append(flat.reshape(-1, flat.shape[-1]).clone())
        ys.append(action.reshape(-1, action.shape[-1]).clone())
        obs, _rew, _term, _trunc, _extras = env.step(action)
    return torch.cat(xs), torch.cat(ys)


def clone(a) -> tuple[SwarmActor, dict[str, float]]:
    torch.manual_seed(a.seed)
    env_probe = make_env(a, a.seed)
    actor = SwarmActor(
        architecture=a.arch,
        hidden=a.hidden,
        initial_log_std=a.initial_log_std,
        min_log_std=a.min_std,
    ).to(a.device)
    del env_probe

    x, y = collect(a, a.seed)
    y = y.clamp(-a.clip, a.clip)  # tanh cannot reach +-1; see the module docstring
    n = x.shape[0]
    print(f"  collected {n:,} (obs, action) rows from B0 on the TRAIN split")

    # Hold out a slice so the reported fit is not the one that was optimised.
    perm = torch.randperm(n, device=x.device)
    cut = int(n * 0.9)
    tr, va = perm[:cut], perm[cut:]

    opt = torch.optim.Adam(actor.parameters(), lr=a.lr)
    stats: dict[str, float] = {}
    for epoch in range(1, a.epochs + 1):
        actor.train()
        order = tr[torch.randperm(tr.numel(), device=x.device)]
        total = 0.0
        for i in range(0, order.numel(), a.batch):
            idx = order[i : i + a.batch]
            mean, _ = actor(x[idx])
            loss = torch.nn.functional.mse_loss(mean, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * idx.numel()
        actor.eval()
        with torch.no_grad():
            mean, _ = actor(x[va])
            val = float(torch.nn.functional.mse_loss(mean, y[va]))
            mae = float((mean - y[va]).abs().mean())
        stats = {"train_mse": total / cut, "val_mse": val, "val_mae": mae}
        print(
            f"  epoch {epoch:>3}  train MSE {stats['train_mse']:.5f}   "
            f"val MSE {val:.5f}   val MAE {mae:.5f}"
        )
    return actor, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gnn", choices=sorted(ARCHITECTURES))
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--steps-per-episode", type=int, default=None)
    ap.add_argument("--fidelity", default="F4")
    ap.add_argument("--jammer", default="J1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.995, help="target clip; tanh cannot reach 1")
    ap.add_argument("--initial-log-std", type=float, default=-0.5)
    ap.add_argument("--min-std", type=float, default=-20.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    print(f"\n  behaviour-cloning B0 -> {a.arch}  (device={a.device}, seed={a.seed})\n")
    actor, stats = clone(a)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": actor.state_dict(),
            "architecture": actor.architecture,
            "hidden": actor.trunk.out_dim,
            "min_log_std": actor.min_log_std.tolist(),
            # 🔒 Top level for the same reason as `obs_history`: both change
            # what the network COMPUTES without changing the shape of its
            # state dict, so a loader that misses them scores a different
            # function and `load_state_dict` raises nothing.
            "tanh_mean": bool(getattr(actor, "tanh_mean", True)),
            "layer_norm": bool(getattr(actor, "layer_norm", False)),
            "timestep": 0,
            # ⚠️ No "value": a BC checkpoint has no critic. `train.py --init-from`
            # loads the actor only and trains a fresh critic, which is correct --
            # a critic fitted to B0's returns is not the critic PPO needs.
            "provenance": {
                "kind": "behaviour_clone",
                "teacher": "b0",
                "split": "train",
                "seed": a.seed,
                "device": str(a.device),
                "architecture": a.arch,
                "num_envs": a.num_envs,
                "num_drones": a.num_drones,
                "stage": a.stage,
                "fidelity": a.fidelity,
                "jammer": a.jammer,
                "epochs": a.epochs,
                "lr": a.lr,
                "target_clip": a.clip,
                "actor_params": parameter_count(actor),
                **stats,
            },
        },
        a.out,
    )
    print(f"\n  -> {a.out}")
    print(f"     {json.dumps(stats)}\n")
    print("  ⚠️  This is a PROBE checkpoint. It has seen a scripted teacher and is")
    print("     not a like-for-like entry in any RQ2 or Gate B table.\n")


if __name__ == "__main__":
    main()
