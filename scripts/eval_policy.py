"""Score a trained checkpoint through the harness B0 was scored with.

    uv run python scripts/eval_policy.py runs/F4-mlp-s0/checkpoint.pt --device mps
    uv run python scripts/eval_policy.py --policy random --stage 1 --device mps
    uv run python scripts/eval_policy.py runs/*/checkpoint.pt --n 3 5 8

⚠️ This deliberately adds no metrics of its own. `src/baselines/evaluate.py`
already produces every number the thesis reports -- mission-capable,
`chain_occluded`, the hop histogram, capacity quantiles, the rate-division
counterfactual, the RQ3 anticipation lead -- and B0's 57.2 % came out of it. A
learned policy scored by a *different* loop is not comparable to B0, and that
comparison is the entire reason B0 exists.

Two rules the defaults enforce:

* **Evaluation is on the held-out routes** (`eval_routes=True`), which is the
  only generalisation check left now that the second city is cut.
* **Every RQ1 number is measured under `--fidelity F4`**, whatever the policy
  was trained under. That is the whole design of the ladder: train on a rung,
  sit the same exam.

Aggregation follows Block E and F: **means across episodes within a seed**
(`RolloutMetrics.summary`), **median + IQR across seeds**. A median within a
seed reports 0.0 % for every rare-event metric.

⚠️ A device is part of a measurement's provenance. `torch.Generator` streams
differ per device, so the same seed draws *different episodes* on MPS than on
CPU. Never compare a number from one device with a number from another.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.b0 import B0Policy
from src.baselines.evaluate import RolloutMetrics, rollout
from src.env.core import STAGES, BatchedSwarmEnv, EnvConfig
from src.models import SwarmActor

#: Metrics stored as a fraction in [0, 1] and read as a percentage. ⚠️ Keep this
#: in step with REPORT: `capable_last_third` shipped without it and printed as
#: "0.4", one decimal of a fraction -- a +-5 pp resolution on the column the
#: late-episode question is decided by.
_AS_PERCENT = frozenset(
    {
        "mission_capable",
        "observed",
        "link_alive",
        "chain_occluded",
        "capable_no_division",
        "capable_last_third",
        "observed_last_third",
        "capable_share_high",
        "capable_share_low",
        # ⚠️ Reads BACKWARDS from the obvious intuition, and that is the point. The
        # rotary-wing power curve is U-shaped: cruising at 13.3 m/s costs 0.638 of
        # hover. So a drone that wanders burns LESS battery than one holding station,
        # and `battery_end` HIGHER than B0's is evidence the swarm is cruising to
        # stay cheap rather than holding a useful position.
        "battery_end",
    }
)

REPORT = (
    "mission_capable",
    "observed",
    "link_alive",
    "chain_occluded",
    "capacity_mean",
    "capacity_p5",
    # ⚠️ The CONTROL for the chain-length confound in results/gate_b.md.
    # `capable_no_division` re-scores at `reuse_limit = 1`, i.e. with the
    # `min(C_i)/min(n, 3)` rate-division penalty removed. A longer chain is more
    # exploitable for two separable reasons -- more links to jam, and a bigger
    # division penalty -- and this column is what separates them.
    "capable_no_division",
    "episode_return",
    "hop_mean",
    # RQ3's own metrics, reported here because they turned out to be the
    # diagnosis rather than a side result: a swarm that cannot hold the observer
    # role hands it over constantly and leaves gaps where nobody is looking.
    "handoffs",
    # Mean steps one drone holds the observer role: observed steps / tenures.
    # The clearest single number for "does anyone commit?", and it is what
    # separates the learned policies from B0 by an order of magnitude.
    "observer_tenure",
    # Role emergence (Block G): normalised entropy of observer identity, 0 = one
    # drone owns the role and 1 = every drone holds it equally. `standoff_gap_m`
    # is how much further back the swarm sits than its closest member -- large
    # means one drone went in and the rest held back to relay.
    "role_entropy",
    "standoff_gap_m",
    # ⚠️ The number the Block G diagnosis rests on: B0 parks its observer at
    # ~79 m, every learned policy loiters at ~291 m, and everything else
    # (observed, tenure, hop count) is a view of that one gap.
    "observer_range_m",
    "capable_last_third",
    "observer_range_last_third",
    "off_axis_m",
    "capable_share_high",
    "capable_share_low",
    # ⚠️ Reads BACKWARDS from the obvious intuition, and that is the point. The
    # rotary-wing power curve is U-shaped: cruising at 13.3 m/s costs 0.638 of
    # hover. So a drone that wanders burns LESS battery than one holding station,
    # and `battery_end` HIGHER than B0's is evidence the swarm is cruising to
    # stay cheap rather than holding a useful position.
    "battery_end",
)


def med_iqr(values: list[float]) -> tuple[float, float]:
    """Median and inter-quartile range. AGENTS.md: never mean +- std."""
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.median()), float(t.quantile(0.75) - t.quantile(0.25))


def make_env(a, seed: int, num_drones: int) -> BatchedSwarmEnv:
    weights = tuple(1.0 if i == a.stage - 1 else 0.0 for i in range(len(STAGES)))
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=a.num_envs,
            num_drones=num_drones,
            device=a.device,
            seed=seed,
            fidelity=a.fidelity,
            action_space=a.action_space,
            jammer=a.jammer,
            eval_routes=not a.train_routes,
            obs_history=a.obs_history,
            auto_reset=False,  # one episode per environment: clean metric rows
            stage_weights=weights,
            compile_occlusion=a.device != "cpu",
        )
    )


def load_actor(path: Path, env: BatchedSwarmEnv) -> tuple[SwarmActor, dict]:
    """Rebuild the actor a checkpoint describes.

    ⛔ The `blob["recurrent"]` branch is gone with `docs/REDUCTION.md` task 4:
    recurrence was killed on its own pre-declared rule (−1.05 pp, tenure 36.8
    against a required 95, seed IQR *widened* 4.7 → 6.9). A checkpoint carrying
    that key is a predecessor artefact and is refused rather than silently
    loaded as a feedforward policy, which would score a different network.
    """
    blob = torch.load(path, map_location=env.device, weights_only=False)
    if blob.get("recurrent"):
        raise SystemExit(
            f"{path} is a recurrent checkpoint; recurrence was removed in "
            "docs/REDUCTION.md task 4 and there is nothing here that can load it"
        )
    actor = SwarmActor(
        architecture=blob["architecture"],
        hidden=blob.get("hidden"),
        min_log_std=blob.get("min_log_std", -20.0),
        obs_history=blob.get("obs_history", 1),
    ).to(env.device)
    if blob.get("obs_history", 1) != env.cfg.obs_history:
        raise SystemExit(
            f"checkpoint wants obs_history={blob.get('obs_history', 1)} but the env has "
            f"{env.cfg.obs_history}; scoring it would feed the actor the wrong input width"
        )
    actor.load_state_dict(blob["policy"])
    actor.eval()
    return actor, blob


def score(a, name: str, checkpoint: Path | None, num_drones: int) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    steps = STAGES[a.stage - 1].episode_steps

    for seed in range(a.seeds):
        env = make_env(a, seed, num_drones)
        b, n = env.cfg.num_envs, env.cfg.num_drones
        on_reset = None

        if name == "random":
            gen = torch.Generator(device=env.device).manual_seed(seed)

            def policy(_obs, _b=b, _n=n, _gen=gen, _env=env):
                return torch.empty(_b, _n, 3, device=_env.device).uniform_(-1, 1, generator=_gen)

        elif name == "b0":
            pol = B0Policy(b, n, variant="b0", device=env.device, action_space=env.cfg.action_space)
            on_reset = pol.reset

            def policy(obs, _pol=pol):
                return _pol.act(obs["flat"])

        else:
            actor, _blob = load_actor(checkpoint, env)

            @torch.no_grad()
            def policy(obs, _actor=actor, _b=b, _n=n):
                # The deterministic action: the Gaussian's mean, not a sample.
                # Evaluating a stochastic policy by sampling would report the
                # exploration noise as part of the result.
                key = "flat_history" if "flat_history" in obs else "flat"
                mean, _ = _actor(obs[key].reshape(_b * _n, -1))
                return mean.view(_b, _n, 3)

        metrics: RolloutMetrics = rollout(env, policy, steps, on_reset=on_reset)
        summary = metrics.summary()
        summary["observer_tenure"] = summary["observed"] * steps / (summary["handoffs"] + 1.0)
        summary["hop_mean"] = float(
            (metrics.hop_hist * torch.arange(metrics.hop_hist.shape[-1])).sum(-1).mean()
        )
        for key in REPORT:
            cols.setdefault(key, []).append(summary[key])
    return cols


def _append(path: Path, row: dict) -> None:
    """One JSONL row, appended. Same contract as `sweep.py`'s summary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="*", type=Path)
    ap.add_argument("--policy", nargs="*", default=[], choices=["random", "b0"])
    ap.add_argument("--fidelity", default="F4", choices=["F0", "F1", "F2", "F3", "F4"])
    ap.add_argument(
        "--jammer",
        default="J1",
        choices=["J0", "J1", "J2", "J3", "J3B"],
        help="the adversary rung to evaluate UNDER, PLAN.md §4. J1 is the inherited "
        "isotropic emitter and every pre-2026-09-02 number in results/ was measured "
        "at it. ⛔ Orthogonal to --fidelity, which decides whether the emitter is in "
        "the SINR denominator at all: a rung is a beam PATTERN, not a fidelity level.",
    )
    ap.add_argument("--action-space", default="acceleration", choices=["acceleration", "velocity"])
    ap.add_argument("--stage", type=int, default=4, help="curriculum stage to evaluate at")
    ap.add_argument(
        "--obs-history",
        type=int,
        default=1,
        help="frames of observation history the env supplies (1 = off, the default). "
        "⛔ Must match the checkpoint: `load_actor` refuses a mismatch rather than "
        "scoring an actor at the wrong input width",
    )
    ap.add_argument("--n", nargs="*", type=int, default=[5], help="swarm sizes (zero-shot)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-routes", action="store_true", help="tune on these, never report them")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="ALSO append one JSONL row per (policy, N) here, carrying the PER-SEED "
        "values and not just the median. Point it at results/ (tracked) rather than "
        "runs/ (gitignored): the summary is a result, the checkpoints are regenerable, "
        "so the machine that ran it can `git push` and the machine that analyses it "
        "can `git pull`. Per-seed matters -- BLOCK_G_PLAN's gate rules are declared "
        "on the WORST seed, and a median alone cannot be judged against them.",
    )
    ap.add_argument(
        "--group",
        default=None,
        help="treat the given checkpoints as SEEDS OF ONE CONDITION: score each on the "
        "same episodes and report median [IQR] across them. This is the aggregation "
        "AGENTS.md asks for -- means across episodes within a seed, median across seeds "
        "-- and it is the only one that reports training-seed variation. Without it, "
        "--seeds varies the EVALUATION episodes of a single policy, which says nothing "
        "about whether a second training run would have landed anywhere near the first.",
    )
    a = ap.parse_args()

    if a.seeds < 5:
        print(f"⚠️  {a.seeds} seeds. AGENTS.md requires >=5 for anything reported as a finding.\n")

    split = "train" if a.train_routes else "eval"
    print(
        f"device={a.device}  fidelity={a.fidelity}  jammer={a.jammer}  stage={a.stage}  routes={split}  "
        f"{a.seeds} seeds x {a.num_envs} episodes; median [IQR] across seeds"
    )
    print("⚠️  device is part of provenance: never compare across devices.\n")

    header = f"{'policy':<24}{'N':>3}" + "".join(f"{k.replace('_', ' '):>18}" for k in REPORT)
    print(header)
    print("-" * len(header))

    jobs: list[tuple[str, Path | None]] = [(p, None) for p in a.policy]
    if a.group:
        jobs.append((a.group, None))  # handled below, from a.checkpoints
    else:
        jobs += [(cp.parent.name, cp) for cp in a.checkpoints]

    for name, checkpoint in jobs:
        for n in a.n:
            if a.group and name == a.group:
                # One row per training seed, all on the SAME episodes (paired), so
                # the spread reported is training-seed variation and nothing else.
                paired = argparse.Namespace(**{**vars(a), "seeds": 1})
                cols = {}
                for cp in a.checkpoints:
                    for key, values in score(paired, cp.parent.name, cp, n).items():
                        cols.setdefault(key, []).extend(values)
            else:
                cols = score(a, name, checkpoint, n)
            row = f"{name:<24}{n:>3}"
            for key in REPORT:
                m, i = med_iqr(cols[key])
                scale = 100.0 if key in _AS_PERCENT else 1.0
                unit = " %" if scale == 100.0 else ""
                row += f"{m * scale:>12.1f}{unit:<2}[{i * scale:.1f}]".rjust(18)
            print(row)

            if a.out:
                _append(
                    a.out,
                    {
                        "policy": name,
                        "n": n,
                        "split": split,
                        "stage": a.stage,
                        "fidelity": a.fidelity,
                        "jammer": a.jammer,
                        "device": a.device,
                        "num_envs": a.num_envs,
                        "grouped": bool(a.group and name == a.group),
                        "checkpoints": [str(cp) for cp in a.checkpoints]
                        if a.group and name == a.group
                        else ([str(checkpoint)] if checkpoint else []),
                        # ⚠️ Per-seed, not just the aggregate. `median [IQR]` cannot be
                        # judged against a rule declared on the worst seed, and every
                        # gate rule in BLOCK_G_PLAN.md is.
                        "seeds": {key: cols[key] for key in REPORT},
                        "median": {key: med_iqr(cols[key])[0] for key in REPORT},
                        "iqr": {key: med_iqr(cols[key])[1] for key in REPORT},
                    },
                )


if __name__ == "__main__":
    main()
