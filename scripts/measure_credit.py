r"""How much of the advantage signal can tell one drone from another?

⛔ **Declared before the numbers existed** (2026-09-02), because a rule invented
after the fact is not a rule and this script is aimed at a hypothesis that eight
pre-declared interventions have failed to move.

## The question

`PLAN.md` §3 records the measured deficit: the swarm never differentiates into an
observer and a relay chain. 📏 `role_entropy` 0.51 against B0's 0.10, observer
range 218.9 m against 88.7 m, `observed` 59.6 % against 93.6 %. Every proposed
mechanism so far has been a reward knob, and all eight were nulls -- the last
(`PHI_V2`) with a gradient *measured adequate* before the run.

`src/training/ppo.py` names a different suspect, in a comment nobody has cashed:

> The global state is repeated per drone, but `reward()` is per-drone (team terms
> plus INDIVIDUAL energy and effort costs), so N rows share one input and carry N
> different targets. The critic can only learn their mean; the between-drone
> spread is irreducible error that goes straight into advantage noise.

🔍 That spread is **two things at once**, and the second is what matters here:

1. it is the ceiling on `explained_variance` -- a shared critic cannot predict it;
2. it is the **entire budget of drone-differentiating credit**. With one value per
   global state broadcast across N rows, `A[t,b,i] = G[t,b,i] - V[t,b]`, so
   `Var_i(A) = Var_i(G)` exactly. Whatever the policy gradient knows about *which
   drone should do what*, it knows through that variance and through nothing else.

## The structural half, readable from the source rather than measured

`reward_terms()` returns `mission`, `idle`, `battery_variance` and `shaping`
through `team(x)`, which broadcasts one `(B,)` tensor across drones -- **identical
for every drone by construction**. Only `energy`, `effort` and `relay` are
per-drone, and 📏 `w_relay` ships at **0.0** (checkpoint provenance), so `relay`
is exactly zero.

⛔ **So the only reward components that differ between drones are two motion
costs.** No shaping intervention can change this: `w_hold`, `d_ref`,
`potential_scale` and `PHI_V2` all scale team terms, which are broadcast
identically and therefore cancel out of `Var_i` exactly.

## What this script measures

The empirical half: the *magnitude*. Decompose the per-drone discounted
return-to-go `G[t,b,i]` by the law of total variance over the `(t,b)` grouping:

    Var_total  =  E_{t,b}[ Var_i(G) ]   +   Var_{t,b}( E_i[G] )
                  \_______________/         \_______________/
                   between-drone             team component
                   "differentiable"          what a shared critic predicts

and report `differentiable_share = between_drone / total`.

## 🔒 Declared reading, before the run

| | rule |
|---|---|
| **confirms the mechanism** | `differentiable_share` < 5 %. The advantage is then ~95 % identical across drones, and role credit is not something this architecture can express at any reward setting. That closes the reward axis *structurally* rather than by exhaustion, and redirects the search to the critic |
| **refutes it** | `differentiable_share` > 20 %. There is then ample drone-differentiating signal and the failure is elsewhere -- exploration, or capacity |
| **inconclusive** | 5-20 %. Report and do not build on it |

⚠️ **One number already bears on this and must be read alongside.**
`docs/inherited/BLOCK_G.md` records `w_relay 0.5` raising per-drone advantage
variance **71x** (0.00041 -> 0.02931) with no behavioural change. Against PPO's
unit-normalised advantage that is a move from ~0.04 % to ~2.9 % -- so if this
script returns a single-digit share, the honest reading of `w_relay`'s null is
**"raised it to 3 %, still negligible"**, not "per-drone credit does not help".
⛔ That reframing is declared here so it cannot be invented afterwards.

⚠️ Provenance rules apply: `torch.Generator` streams differ per device, so a run
here is comparable only with another run of this script on the same device.

Usage:

    uv run python scripts/measure_credit.py --policy b0 --device mps
    uv run python scripts/measure_credit.py --policy runs/val-gnn-deep-s0/checkpoint.pt
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
from src.env.reward import RewardWeights
from src.models.actor import SwarmActor

#: The per-drone reward terms. Everything else in `reward_terms()` goes through
#: `team(x)` and is identical across drones by construction, so it cancels out of
#: `Var_i` exactly and cannot carry role credit.
PER_DRONE_TERMS = ("energy", "effort", "relay", "difference")


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
            eval_routes=not a.train_routes,
            auto_reset=False,  # one episode per environment
            stage_weights=weights,
            training_extras=True,  # this is the whole point: per-term rewards
            compile_occlusion=a.device != "cpu",
        ),
        # ⚠️ The difference reward is the one intervention this script exists to
        # SCORE rather than to explain away, so it has to be settable here.
        # Everything else is left at `DEFAULT_WEIGHTS`.
        weights=RewardWeights(w_difference=a.w_difference),
    )


def load_actor(path: Path, env: BatchedSwarmEnv) -> SwarmActor:
    blob = torch.load(path, map_location=env.device, weights_only=False)
    if blob.get("recurrent"):
        raise SystemExit(f"{path} is a recurrent checkpoint; see eval_policy.py")
    actor = SwarmActor(
        architecture=blob["architecture"],
        hidden=blob.get("hidden"),
        min_log_std=blob.get("min_log_std", -20.0),
        # ⚠️ Defaults are the PRE-change behaviour, so an old checkpoint
        # still loads as the network it was trained as.
        tanh_mean=blob.get("tanh_mean", True),
        layer_norm=blob.get("layer_norm", False),
    ).to(env.device)
    actor.load_state_dict(blob["policy"])
    actor.eval()
    return actor


def returns_to_go(rewards: Tensor, gamma: float) -> Tensor:
    """`(T, B, N)` discounted return-to-go, computed backwards in place.

    No bootstrap and no truncation handling: `auto_reset=False` and `steps` is the
    episode length, so every row is one complete episode and the tail really is
    the end of it.
    """
    out = torch.zeros_like(rewards)
    acc = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


def decompose(g: Tensor) -> dict[str, float]:
    """Law of total variance over the `(t, b)` grouping of `(T, B, N)` returns.

    `between_drone` is `E_{t,b}[Var_i(G)]` -- what a shared critic cannot predict
    AND the only part of the advantage that differs between drones.
    `team` is `Var_{t,b}(E_i[G])` -- exactly what a shared critic does predict.
    """
    flat = g.reshape(-1, g.shape[-1])  # (T*B, N)
    within = flat.var(dim=-1, unbiased=False).mean()  # E[Var_i]
    between = flat.mean(dim=-1).var(unbiased=False)  # Var(E_i)
    total = within + between
    return {
        "between_drone_var": float(within),
        "team_var": float(between),
        "total_var": float(total),
        "differentiable_share": float(within / total.clamp_min(1e-12)),
        "between_drone_std": float(within.sqrt()),
        "total_std": float(total.sqrt()),
    }


def rollout(a, seed: int) -> dict[str, object]:
    env = make_env(a, seed)
    steps = a.steps or STAGES[a.stage - 1].episode_steps
    b, n = env.cfg.num_envs, env.cfg.num_drones
    dev = env.device

    b0 = actor = None
    if a.policy == "b0":
        b0 = B0Policy(num_envs=b, num_drones=n, device=dev)
        b0.reset()
    elif a.policy == "random":
        gen = torch.Generator(device=dev).manual_seed(seed)
    else:
        actor = load_actor(Path(a.policy), env)

    obs = env.reset()
    rewards = torch.zeros(steps, b, n, device=dev)
    terms: dict[str, Tensor] = {}

    for t in range(steps):
        if b0 is not None:
            action = b0.act(obs["flat"])
        elif actor is not None:
            with torch.no_grad():
                action = actor(obs["flat"])[0]
        else:
            action = torch.empty(b, n, 3, device=dev).uniform_(-1, 1, generator=gen)

        obs, rew, _term, _trunc, extras = env.step(action)
        rewards[t] = rew
        for key, value in extras.items():
            if key.startswith("reward/"):
                name = key.split("/", 1)[1]
                if name not in terms:
                    terms[name] = torch.zeros(steps, b, n, device=dev)
                terms[name][t] = value

    g = returns_to_go(rewards, a.gamma)
    out: dict[str, object] = {"total": decompose(g)}
    out["terms"] = {
        name: decompose(returns_to_go(value, a.gamma)) | {"per_drone": name in PER_DRONE_TERMS}
        for name, value in sorted(terms.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="b0", help="b0 | random | path/to/checkpoint.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None, help="default: the stage's length")
    ap.add_argument("--fidelity", default="F4")
    ap.add_argument("--jammer", default="J1")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument(
        "--w-difference",
        type=float,
        default=0.0,
        help="weight on the difference reward D_i = G - G_without_i. 0.0 (default) "
        "reproduces every row already in results/credit.jsonl. 🔒 The pre-declared "
        "reading in this file's docstring applies unchanged: >20 %% refutes the "
        "mechanism, <5 %% confirms it",
    )
    ap.add_argument("--train-routes", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="append one JSON line here")
    a = ap.parse_args()

    runs = [rollout(a, seed) for seed in range(a.seeds)]
    shares = [float(r["total"]["differentiable_share"]) for r in runs]  # type: ignore[index]
    med = float(torch.tensor(shares).median())

    print(f"\n  policy={a.policy}  device={a.device}  stage={a.stage}  {a.fidelity}/{a.jammer}")
    print(f"  {a.seeds} seeds x {a.num_envs} envs x N={a.num_drones}, gamma={a.gamma}\n")
    print(f"  {'':<22}{'between-drone':>15}{'team':>12}{'differentiable':>16}")
    for seed, r in enumerate(runs):
        d = r["total"]  # type: ignore[index]
        print(
            f"  seed {seed:<17}{d['between_drone_var']:>15.4f}{d['team_var']:>12.2f}"
            f"{d['differentiable_share'] * 100:>15.2f} %"
        )
    print(f"\n  📏 median differentiable share = {med * 100:.2f} %")
    verdict = (
        "CONFIRMS the mechanism (< 5 %)"
        if med < 0.05
        else "REFUTES it (> 20 %)"
        if med > 0.20
        else "INCONCLUSIVE (5-20 %)"
    )
    print(f"  🔒 declared reading: {verdict}\n")

    print(f"  {'term':<20}{'per-drone':>11}{'between-drone std':>20}{'share of its own':>18}")
    for name, d in runs[0]["terms"].items():  # type: ignore[index]
        flag = "yes" if d["per_drone"] else "team"
        print(
            f"  {name:<20}{flag:>11}{d['between_drone_std']:>20.5f}"
            f"{d['differentiable_share'] * 100:>17.2f} %"
        )
    print()

    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "policy": a.policy,
                        "device": a.device,
                        "stage": a.stage,
                        "fidelity": a.fidelity,
                        "jammer": a.jammer,
                        "num_envs": a.num_envs,
                        "num_drones": a.num_drones,
                        "seeds": a.seeds,
                        "gamma": a.gamma,
                        "split": "train" if a.train_routes else "eval",
                        "median_differentiable_share": med,
                        "per_seed": shares,
                        "runs": runs,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
