r"""Over what horizon would memory have to operate to be worth anything here?

**Declared 2026-09-02, before the numbers existed.** This decides a design
question and is deliberately not a gate on a policy: `docs/REDUCTION.md` task 4
deleted recurrence, and rebuilding it means sequence-aware rollout storage, BPTT
and hidden state carried across the rollout boundary -- the highest bug-density
change available in this repo, and the surface that produced four silent bugs
last time. ⛔ Measure the requirement before paying that.

## The quantity, and why it is the right one

B0 carries `belief_rel` / `belief_vel` -- a target-belief filter with dead
reckoning. That memory is cashed in exactly one situation: **a drone saw the
target, lost it, and keeps acting on what it remembers.** The actors are
feedforward over `obs["flat"]`, which zeroes `rel_hvt * sees` when a drone cannot
see the target, so during precisely those intervals B0 has information the
student does not.

📏 So the measurement is the length of **unseen runs preceded by a sighting**,
per drone. Three things make that correct rather than merely plausible:

1. 🔒 **Runs never preceded by a sighting are excluded.** A drone that has never
   seen the target has nothing to remember, and counting those intervals would
   inflate the horizon with time where memory is *definitionally* useless.
2. 🔒 **Target displacement over the gap is measured alongside its length.**
   Dead reckoning decays: a 50-step gap over 20 m is memory-useful, a 50-step gap
   over 400 m is not -- the belief is stale and the real deficit is reacquisition,
   not memory. **Length alone would over-sell recurrence.**
3. 🔒 **Both the fraction of GAPS and the fraction of blind TIME are reported.**
   They can disagree sharply. If 90 % of gaps are short but 90 % of blind time
   sits in a few long ones, frame stacking covers most events and almost none of
   the problem.

## 🔒 Declared reading, before the run

`k` below is the frame-stacking depth that would cover a gap: `k` stacked
observations carry `k-1` steps of history.

| | rule | what it implies |
|---|---|---|
| **frame stacking** | p90 gap <= **8 steps** *and* most blind time in short gaps | a ~10-line observation change buys the horizon. ⛔ Do not rebuild the GRU |
| **a filter is justified** | p90 gap >> 8 steps *and* median displacement over a gap is **small** relative to the sensor envelope | remembered position is still usable after the gap; recurrence has something to represent |
| **memory is not the deficit** | long gaps *and* large displacement | the belief would be stale on arrival. The problem is reacquisition, and ⛔ recurrence should not be rebuilt on this evidence |

⚠️ **What this does NOT show.** It bounds the *design requirement*, not the
*benefit*. A policy given memory would reposition and change its own gap
distribution, so this cannot say memory will help. It answers
frame-stacking-vs-GRU, and it can return a clean negative.

⚠️ Provenance: `torch.Generator` streams differ per device; rows are comparable
with each other and with nothing else.

Usage:

    uv run python scripts/measure_memory_horizon.py --policy b0 --device mps
    uv run python scripts/measure_memory_horizon.py --policy runs/val-gnn-deep-s0/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.b0 import B0Policy
from src.env.core import STAGES, BatchedSwarmEnv, EnvConfig
from src.models.actor import SwarmActor


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
            auto_reset=False,
            stage_weights=weights,
            compile_occlusion=a.device != "cpu",
        )
    )


@torch.no_grad()
def rollout(a, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`(sees, hvt_xy)` as `(T, B, N)` bool and `(T, B, 2)` metres."""
    env = make_env(a, seed)
    steps = a.steps or STAGES[a.stage - 1].episode_steps

    b0 = actor = None
    if a.policy == "b0":
        b0 = B0Policy(num_envs=env.cfg.num_envs, num_drones=env.cfg.num_drones, device=env.device)
        b0.reset()
    elif a.policy == "random":
        gen = torch.Generator(device=env.device).manual_seed(seed)
    else:
        blob = torch.load(Path(a.policy), map_location=env.device, weights_only=False)
        actor = SwarmActor(
            architecture=blob["architecture"],
            hidden=blob.get("hidden"),
            min_log_std=blob.get("min_log_std", -20.0),
        ).to(env.device)
        actor.load_state_dict(blob["policy"])
        actor.eval()

    obs = env.reset()
    sees, hvt = [], []
    for _ in range(steps):
        if b0 is not None:
            action = b0.act(obs["flat"])
        elif actor is not None:
            action = actor(obs["flat"])[0]
        else:
            action = torch.empty(
                env.cfg.num_envs, env.cfg.num_drones, 3, device=env.device
            ).uniform_(-1, 1, generator=gen)
        # ⚠️ hvt_pos read BEFORE the step, so it pairs with the `sees` bit that
        # the same state produced. Reading it after would offset the displacement
        # by one tick against the sighting it is measured from.
        hvt.append(env.hvt_pos[:, :2].clone())
        obs, _r, _t, _tr, extras = env.step(action)
        sees.append(extras["sees_hvt"].bool().clone())
    return (
        torch.stack(sees).cpu().numpy(),
        torch.stack(hvt).cpu().numpy(),
    )


def gaps(sees: np.ndarray, hvt_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unseen runs **preceded by a sighting**, as `(lengths, displacements_m)`.

    🔒 The "preceded by a sighting" restriction is the whole point: a drone that
    has never seen the target carries no belief, so an unseen run before its first
    sighting is not an interval memory could have bridged. Those are dropped.

    Displacement is `‖hvt(reacquire or episode end) − hvt(last seen)‖` -- how far
    the remembered position has gone stale by the time the gap closes.
    """
    t, b, n = sees.shape
    lengths, disps = [], []
    for e in range(b):
        for d in range(n):
            s = sees[:, e, d]
            seen_yet = False
            i = 0
            while i < t:
                if s[i]:
                    seen_yet = True
                    i += 1
                    continue
                if not seen_yet:  # never sighted -> nothing to remember
                    i += 1
                    continue
                start = i
                while i < t and not s[i]:
                    i += 1
                # `start - 1` is the last step the target WAS seen; `i` is the
                # reacquisition step, or `t` if the episode ended blind.
                last_seen, end = start - 1, min(i, t - 1)
                lengths.append(i - start)
                disps.append(float(np.linalg.norm(hvt_xy[end, e] - hvt_xy[last_seen, e])))
    return np.asarray(lengths), np.asarray(disps)


def report(name: str, sees: np.ndarray, hvt_xy: np.ndarray, k: int) -> dict[str, float]:
    lengths, disps = gaps(sees, hvt_xy)
    if lengths.size == 0:
        print(f"  {name}: no bridgeable gaps at all")
        return {}
    blind = lengths.sum()
    short = lengths <= k
    out = {
        "per_drone_sees_rate": float(sees.mean()),
        "swarm_observed_rate": float(sees.any(axis=-1).mean()),
        "n_gaps": int(lengths.size),
        "gap_p50": float(np.percentile(lengths, 50)),
        "gap_p90": float(np.percentile(lengths, 90)),
        "gap_p99": float(np.percentile(lengths, 99)),
        "gap_max": float(lengths.max()),
        "frac_gaps_short": float(short.mean()),
        "frac_blind_time_short": float(lengths[short].sum() / blind),
        "disp_p50_m": float(np.percentile(disps, 50)),
        "disp_p90_m": float(np.percentile(disps, 90)),
        "disp_p50_short_m": float(np.percentile(disps[short], 50)) if short.any() else float("nan"),
    }
    print(f"\n  --- {name} ---")
    print(
        f"  per-drone sees {out['per_drone_sees_rate'] * 100:5.1f} %   "
        f"swarm observed {out['swarm_observed_rate'] * 100:5.1f} %   "
        f"{out['n_gaps']:,} bridgeable gaps"
    )
    print(
        f"  gap steps      p50 {out['gap_p50']:>6.0f}   p90 {out['gap_p90']:>6.0f}   "
        f"p99 {out['gap_p99']:>6.0f}   max {out['gap_max']:>6.0f}"
    )
    print(
        f"  covered by k={k}   {out['frac_gaps_short'] * 100:5.1f} % of gaps   "
        f"but only {out['frac_blind_time_short'] * 100:5.1f} % of blind TIME"
    )
    print(
        f"  hvt moved      p50 {out['disp_p50_m']:>6.0f} m  p90 {out['disp_p90_m']:>6.0f} m"
        f"   (short gaps: p50 {out['disp_p50_short_m']:.0f} m)"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="b0", help="b0 | random | path/to/checkpoint.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--fidelity", default="F4")
    ap.add_argument("--jammer", default="J1")
    ap.add_argument("--k", type=int, default=8, help="frame-stacking depth to score against")
    ap.add_argument("--train-routes", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    print(f"\n  policy={a.policy}  device={a.device}  stage={a.stage}  {a.fidelity}/{a.jammer}")
    rows = []
    for seed in range(a.seeds):
        sees, hvt = rollout(a, seed)
        rows.append(report(f"{a.policy}  seed {seed}", sees, hvt, a.k))

    if rows and a.out is not None:
        agg = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
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
                        "seeds": a.seeds,
                        "k": a.k,
                        "split": "train" if a.train_routes else "eval",
                        "median": agg,
                        "per_seed": rows,
                    }
                )
                + "\n"
            )
        print(f"\n  -> {a.out}\n")


if __name__ == "__main__":
    main()
