"""Train one policy. The only entry point into `src/training/ppo.py`.

    # the inherited condition, 5 seeds -- GNN, deep cadence, F4, train split
    uv run python scripts/train.py --arch gnn --cadence deep --seeds 0 1 2 3 4 \
        --timesteps 12000000 --device cuda --tag g8-ff-shipped

    # then score them as ONE condition, five training seeds, paired episodes
    uv run python scripts/eval_policy.py runs/g8-ff-shipped-s*/checkpoint.pt \
        --group "gnn/deep/shipped" --policy b0 --train-routes --device cuda

⚠️ **This script reports nothing.** Every number the thesis quotes goes through
`src/baselines/evaluate.py`, which is what B0 was scored with; a learned policy
scored by a different loop is not comparable to B0, and that comparison is the
entire reason B0 exists. What training prints is diagnostics.

⚠️ **A device is part of a measurement's provenance.** `torch.Generator` streams
differ per device, so the same seed draws *different episodes* on MPS than on
CUDA. The physics is identical; the sample is not. The device is written into
every checkpoint and every log row, and `--device` is never silently downgraded.

## The reward flags are derived, not written

🔒 Every field of `RewardWeights` inside `Phi` gets a flag automatically, from
`reward.pbrs_safe_fields()`. Two real misses of exactly the shape a hand-written
flag list produces are recorded in `docs/REDUCTION.md`, and
`src/training/test_train_cli.py` asserts the coverage still holds.

⛔ Objective weights get no flags. They change what is *optimal*, and the
behavioural orderings in `weight_constraints_satisfied()` are what set them.
`--battery-variance` is the one exception the design permits, and it is spelled
out rather than derived so that the exception stays visible.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.core import STAGES, BatchedSwarmEnv, EnvConfig
from src.env.reward import PHI_V2, RewardWeights, pbrs_safe_fields, weight_constraints_satisfied
from src.models import ARCHITECTURES, SwarmActor, SwarmCritic, parameter_count
from src.training.curriculum import CurriculumSchedule
from src.training.ppo import CADENCES, PPOConfig, PPOTrainer, mission_diagnostics

#: What a log line shows. Ordered so a run can be read down the column.
WATCH = (
    "progress",
    "reward",
    "mission_capable",
    "observed",
    "e2e_mbps",
    "speed_ms",
    "at_speed_cap",
    "at_boundary",
    "log_std",
    "approx_kl",
    "grad_norm_actor",
    "grad_norm_critic",
    "grad_kept",
    "value_loss",
    "explained_variance",
    "return_spread_between_drones",
    "steps_per_s",
)


def flag(field: str) -> str:
    return "--" + field.replace("_", "-")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--arch",
        default="deepsets",
        choices=ARCHITECTURES,
        help="📏 `deepsets` is the default since RQ2 stage B: the GNN is a null "
        "against it at every swarm size (+1.8 / +0.9 / +1.3 pp at N = 3/5/8, "
        "ranges overlapping), its worst seed at N = 8 is WORSE (46.1 vs 52.2), "
        "and PLAN.md expects PyTorch Geometric to export badly to TensorRT. "
        "⛔ The `gnn` rung is KEPT, not deleted -- RQ2 reports it as a measured "
        "null and Gate C asks which architectures export at all. A rung you "
        "delete is a result you can no longer state",
    )
    ap.add_argument("--hidden", type=int, default=None, help="actor trunk width")
    ap.add_argument("--critic-hidden", type=int, default=256)
    ap.add_argument(
        "--cadence",
        default="deep",
        choices=sorted(CADENCES),
        help="num_envs / rollouts / mini_batches, holding gradient density at 488 per "
        "M env-steps. 📏 `deep` won the 81-run sweep; `wide` quadrupled the seed "
        "spread it was built to shrink",
    )
    ap.add_argument("--num-envs", type=int, default=None, help="overrides --cadence")
    ap.add_argument("--rollouts", type=int, default=None, help="overrides --cadence")
    ap.add_argument("--mini-batches", type=int, default=None, help="overrides --cadence")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=12_000_000)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="seed the MODEL INITIALISATION and the minibatch permutation separately "
        "from the environment's episode stream. 📏 Needed because seed 0 collapsed on "
        "BOTH mps and cuda, and `torch.manual_seed` runs on CPU before `.to(device)` "
        "-- so init and the permutation are device-independent while the episode "
        "stream is not. Splitting them is what attributes the collapse to one or the "
        "other. Default: the same value as --seeds",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fidelity", default="F4", choices=["F0", "F1", "F2", "F3", "F4"])
    ap.add_argument(
        "--jammer",
        default="J1",
        choices=["J0", "J1", "J2", "J3", "J3B"],
        help="the adversary rung the policy TRAINS under, PLAN.md §3",
    )
    ap.add_argument(
        "--action-space",
        default="acceleration",
        choices=["acceleration", "velocity"],
        help="📏 `acceleration` ships. Gate A rejected velocity setpoints "
        "(results/gate_a.md): 21.5 %% against 39.8 %% at matched exploration, "
        "with disjoint seed ranges, and boundary occupancy 14.5-19.7 %% -> "
        "41.6-72.6 %%. Kept so C3 is reported as a comparison",
    )
    ap.add_argument(
        "--eval-routes",
        action="store_true",
        help="⛔ train on the HELD-OUT routes. Never do this for a reported run",
    )
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--stage", type=int, default=4, help="fixed stage when --no-curriculum")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.0, help="entropy_loss_scale")
    ap.add_argument(
        "--min-log-std",
        type=float,
        default=-20.0,
        dest="min_std",
        help="floor on log(sigma) -- a LOG value, not a standard deviation. "
        "-1.6 gives sigma >= 0.20; -20 is no floor and is what the inherited "
        "40.7 %% ran under. 📏 With entropy 0 the deviation shrinks to 0.061 by "
        "20 M steps and the policy stops exploring, so a long run without a "
        "floor measures a policy that has stopped changing. ⚠️ The predecessor's "
        "flag was spelled `--min-std 0.2` and it is not recorded whether that "
        "was a log value or a sigma; this one is unambiguous",
    )
    ap.add_argument("--initial-log-std", type=float, default=-0.5)
    ap.add_argument("--no-shuffle", action="store_true", help="reproduce skrl's minibatch order")
    ap.add_argument("--no-value-norm", action="store_true")
    ap.add_argument(
        "--value-clip",
        type=float,
        default=0.2,
        help="0 disables value clipping entirely. ☠️ A saturated value clip is "
        "what froze the critic on seed 0 (grad_norm_critic -> 0.000 at progress "
        "0.20, mission_capable 50.6 %% -> 2.6 %%); 0 is the arm that cannot do that",
    )
    ap.add_argument("--phi-v2", action="store_true", help="start from PHI_V2 instead of shipped")
    ap.add_argument(
        "--battery-variance",
        type=float,
        default=None,
        help="lambda -- the ONE objective weight the design permits sweeping",
    )
    ap.add_argument("--tag", default=None, help="run directory prefix under --out-root")
    ap.add_argument("--out-root", type=Path, default=Path("runs"))
    ap.add_argument("--log-lines", type=int, default=20)

    phi = ap.add_argument_group(
        "Phi (PBRS-safe, optimum-preserving)",
        "Derived from RewardWeights; every one of these is inside the potential.",
    )
    base = RewardWeights()
    for name in pbrs_safe_fields():
        kind = type(getattr(base, name))
        phi.add_argument(flag(name), type=kind, default=None, dest=f"phi_{name}")
    return ap


def build_weights(a: argparse.Namespace) -> RewardWeights:
    """Start from the shipped potential (or `PHI_V2`) and apply every flag set."""
    weights = PHI_V2 if a.phi_v2 else RewardWeights()
    overrides = {
        name: getattr(a, f"phi_{name}")
        for name in pbrs_safe_fields()
        if getattr(a, f"phi_{name}") is not None
    }
    if a.battery_variance is not None:
        overrides["battery_variance"] = a.battery_variance
    weights = dataclasses.replace(weights, **overrides)

    failed = [k for k, ok in weight_constraints_satisfied(weights).items() if not ok]
    if failed:
        raise SystemExit(
            "these behavioural orderings no longer hold, so the reward no longer "
            f"ranks known policy pairs correctly: {failed}"
        )
    return weights


def resolve_device(name: str) -> torch.device:
    """⛔ Never silently degrade a real run to CPU (`AGENTS.md`)."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested and CUDA is not available; refusing to run")
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested and MPS is not available; refusing to run")
    return torch.device(name)


def run_one(a: argparse.Namespace, seed: int, weights: RewardWeights) -> Path:
    cadence = dict(CADENCES[a.cadence])
    num_envs = a.num_envs or cadence["num_envs"]
    rollouts = a.rollouts or cadence["rollouts"]
    mini_batches = a.mini_batches or cadence["mini_batches"]

    stage_weights = (
        (0.0,) * (a.stage - 1) + (1.0,) + (0.0,) * (len(STAGES) - a.stage)
        if a.no_curriculum
        else (1.0, 0.0, 0.0, 0.0)  # the schedule takes over from the first update
    )
    env = BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs,
            num_drones=a.num_drones,
            device=str(a.device),
            seed=seed,
            fidelity=a.fidelity,
            action_space=a.action_space,
            jammer=a.jammer,
            eval_routes=a.eval_routes,
            auto_reset=True,
            training_extras=True,  # 🔒 the truncation bootstrap needs final_state
            stage_weights=stage_weights,
            compile_occlusion=str(a.device) != "cpu",
        ),
        weights=weights,
    )

    init_seed = seed if a.init_seed is None else a.init_seed
    torch.manual_seed(init_seed)
    actor = SwarmActor(
        architecture=a.arch,
        hidden=a.hidden,
        initial_log_std=a.initial_log_std,
        min_log_std=a.min_std,
    ).to(a.device)
    critic = SwarmCritic(env.cfg.state_dim, hidden=a.critic_hidden).to(a.device)

    ppo = PPOConfig(
        rollouts=rollouts,
        learning_epochs=a.epochs,
        mini_batches=mini_batches,
        learning_rate=a.lr,
        entropy_loss_scale=a.entropy,
        value_clip=a.value_clip,
        shuffle_minibatches=not a.no_shuffle,
        normalise_values=not a.no_value_norm,
    )
    trainer = PPOTrainer(
        env,
        actor,
        critic,
        ppo,
        total_timesteps=a.timesteps,
        curriculum=None if a.no_curriculum else CurriculumSchedule(),
        seed=init_seed,
        diagnostics=mission_diagnostics,
    )

    tag = a.tag or f"{a.fidelity}-{a.arch}-{a.cadence}"
    out = a.out_root / (f"{tag}-s{seed}" if a.init_seed is None else f"{tag}-s{seed}i{init_seed}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    provenance = {
        "seed": seed,
        "init_seed": init_seed,
        "device": str(a.device),
        "architecture": a.arch,
        "cadence": a.cadence,
        "num_envs": num_envs,
        "rollouts": rollouts,
        "mini_batches": mini_batches,
        "epochs": a.epochs,
        "timesteps": a.timesteps,
        "fidelity": a.fidelity,
        "action_space": a.action_space,
        "jammer": a.jammer,
        "split": "eval" if a.eval_routes else "train",
        "curriculum": not a.no_curriculum,
        "stage": None if not a.no_curriculum else a.stage,
        "lr": a.lr,
        "min_log_std": a.min_std,
        "entropy_loss_scale": a.entropy,
        "shuffle_minibatches": not a.no_shuffle,
        "value_clip": a.value_clip,
        "actor_params": parameter_count(actor),
        "critic_params": parameter_count(critic),
        "weights": dataclasses.asdict(weights),
        "torch": torch.__version__,
    }
    print(f"\n=== {out}  {a.arch}/{a.cadence}  seed {seed}  device {a.device} ===")
    print(
        f"    {provenance['actor_params']:,} actor params, "
        f"{provenance['critic_params']:,} critic params, "
        f"{num_envs} envs x {rollouts} rollouts, {a.timesteps:,} env-steps"
    )
    print("    " + "".join(f"{k.replace('_', ' '):>14}" for k in WATCH))

    with log_path.open("w") as handle:
        handle.write(json.dumps({"provenance": provenance}) + "\n")

        def on_log(row: dict) -> None:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print("    " + "".join(f"{row.get(k, float('nan')):>14.4g}" for k in WATCH))

        started = time.perf_counter()
        trainer.train(a.timesteps, on_log=on_log, log_lines=a.log_lines)

    checkpoint = out / "checkpoint.pt"
    trainer.save(checkpoint, extra={"provenance": provenance})
    print(f"    -> {checkpoint}  ({time.perf_counter() - started:.1f} s)")
    return checkpoint


def main() -> None:
    # Line-buffered so a redirected run is watchable while it runs, not after.
    sys.stdout.reconfigure(line_buffering=True)
    a = build_parser().parse_args()
    a.device = resolve_device(a.device)
    weights = build_weights(a)
    if a.eval_routes:
        print("⛔ --eval-routes: training on the HELD-OUT split. Not a reportable run.\n")
    if len(a.seeds) < 5:
        print(
            f"⚠️  {len(a.seeds)} seed(s). AGENTS.md requires >=5 for anything reported "
            "as a finding, judged on the WORST seed.\n"
        )
    for seed in a.seeds:
        run_one(a, seed, weights)


if __name__ == "__main__":
    main()
