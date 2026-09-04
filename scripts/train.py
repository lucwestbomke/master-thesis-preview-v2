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
    ap.add_argument(
        "--mask-jammed-obs",
        action="store_true",
        help="zero the 9 observation features the jammer can move (noise_dbm, "
        "e2e_capacity, per-edge capacity), leaving geometry and the sensor. "
        "📏 PLAN.md §7: the learned analogue of B0Config.repair_score='clearance' "
        "-- a policy that can still adapt, but not on what it is attacked through",
    )
    ap.add_argument(
        "--mask-broadcast-obs",
        action="store_true",
        help="zero the ego features that are TEAM BROADCASTS (e2e_capacity, "
        "steps_since_link). 📏 Both have a between-drone standard deviation of "
        "EXACTLY 0.00000 under B0 and under a random policy alike -- they are "
        "`(B,)` scalars `.expand()`ed across the drone axis -- so they cannot "
        "break symmetry, which is the measured deficit. ⚠️ e2e_capacity is also "
        "the highest-variance ego feature there is (total std 1.49)",
    )
    ap.add_argument(
        "--cue-mode",
        default="position",
        choices=["position", "bearing", "off"],
        help="what the persistent briefed cue in ego dims 4-6 reports. 📏 The cue "
        "is hvt_pos at t=0 and is NEVER refreshed: median |cue - hvt| is 322 m at "
        "t=150 and 984 m at t=599, against a 127 m along-street sightline median "
        "-- so as a POSITION it is stale within ~60 steps. Its BEARING survives "
        "(17.8 deg median error at t=599). `bearing` reports the horizontal unit "
        "vector only; `off` zeroes it. ⛔ B0 reads this block, so it must be "
        "scored under `position`",
    )
    ap.add_argument(
        "--obs-history",
        type=int,
        default=1,
        help="frames of observation history the actor sees (1 = off, the default). "
        "📏 Motivated by results/b0_ablation.md: local link repair is +6.9 pp of "
        "B0's design advantage and is a hill climb carrying ONE step of search "
        "state, which k=2 supplies. ⛔ Not the target memory memory_horizon.md "
        "closed -- that needed 320 steps and was worth ~0",
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="start the ACTOR from this checkpoint instead of a fresh init -- the "
        "seam scripts/bc_init.py's behaviour clone plugs into. ⛔ The CRITIC is "
        "always fresh: a critic fitted to the teacher's returns is not the critic "
        "PPO needs, and loading one would silently start training with a value "
        "function for a different policy. ⚠️ A run started this way has seen a "
        "scripted teacher and is a PROBE, not a like-for-like RQ2 or Gate B arm",
    )
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
    ap.add_argument(
        "--curriculum-boundaries",
        type=float,
        nargs=3,
        default=None,
        help="progress fractions at which stages 2/3/4 become the focus; default "
        "(0.15, 0.35, 0.60). ⚠️ docs/inherited/BLOCK_G.md lists the schedule as "
        "PROVISIONAL and never measured. 📏 And STAGES[0] is degenerate: "
        "speed_scale 0.0 and cue_sigma_m 0.0 means the target does not move and "
        "the cue points at it EXACTLY, so stage 1 is solvable by a linear policy "
        "on cue_rel -- 15 %% of training plus a 20 %% mix thereafter, teaching a "
        "shortcut that is a median 984 m wrong by t = 599",
    )
    ap.add_argument(
        "--curriculum-mix",
        type=float,
        default=None,
        help="share of episodes drawn from EARLIER stages once past stage 1; "
        "default 0.20",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE(lambda). ⚠️ Never swept in this project's history",
    )
    ap.add_argument(
        "--grad-norm-clip",
        type=float,
        default=0.5,
        help="⚠️ Applied JOINTLY to policy and value parameters (ppo.py). "
        "docs/inherited/BLOCK_G.md flags that as 'real, plausible under a GRU, "
        "never tested' -- exposed here so it can finally be tested",
    )
    ap.add_argument("--entropy", type=float, default=0.0, help="entropy_loss_scale")
    ap.add_argument(
        "--mini-batch-size",
        type=int,
        default=None,
        help="ROWS per gradient step, set directly instead of via --mini-batches. "
        "☠️ The axis this project froze. docs/inherited/BLOCK_G.md held 'gradient "
        "density constant at 488 optimizer steps per M env-steps' across all three "
        "cadences, which forces the minibatch to 40,960 rows in every one of them. "
        "At `deep`, a 12 M-step run is 46 PPO updates and ~5,900 Adam steps total on "
        "a 137 k-parameter actor. 📏 runs/val-gnn-deep-s*/log.jsonl show approx_kl at "
        "0.002-0.004 throughout, against PPO's usual 0.01-0.02. Shrinking this costs "
        "almost no FLOPs and multiplies the gradient-step count by the same factor",
    )
    ap.add_argument(
        "--lr-critic",
        type=float,
        default=None,
        help="critic learning rate; default = --lr, which is the inherited single "
        "Adam over both parameter sets",
    )
    ap.add_argument(
        "--grad-norm-clip-critic",
        type=float,
        default=None,
        help="separate gradient-norm clip for the critic. Default (unset) keeps the "
        "inherited JOINT clip, which docs/inherited/BLOCK_G.md lists as open and "
        "untested: with value_loss_scale 2.5 a large value gradient throttles the "
        "policy gradient. ⚠️ `grad_kept` is instrumented for it and is NaN in every "
        "log in runs/ -- it has never been read",
    )
    ap.add_argument(
        "--target-kl",
        type=float,
        default=0.0,
        help="adaptive actor LR targeting this per-round KL (0 = off, the inherited "
        "behaviour). 0.015 is the usual MAPPO target",
    )
    ap.add_argument(
        "--w-difference",
        type=float,
        default=None,
        help="weight on the DIFFERENCE REWARD D_i = G - G_without_i, the mission term "
        "recomputed with drone i deleted (Wolpert & Tumer 2002). 0.0 ships and the "
        "reward is then byte-identical. 🔍 The one instrument that changes the RETURN "
        "rather than scaling a team term that cancels exactly: results/"
        "credit_assignment.md closed the reward axis structurally, and D_i is "
        "factored (G(z_-i) does not depend on a_i) so it cannot move the equilibrium. "
        "📏 Measured on B0 with scripts/measure_credit.py: differentiable_share "
        "0.09 %% -> 5.2 / 15.5 / 35.6 %% at w = 0.5 / 1.0 / 2.0. ⚠️ An OBJECTIVE "
        "weight, not a Phi term, which is why it is spelled out here like "
        "--battery-variance rather than derived",
    )
    ap.add_argument(
        "--min-log-std",
        type=float,
        nargs="+",
        default=[-20.0],
        dest="min_std",
        help="floor on log(sigma) -- a LOG value, not a standard deviation. "
        "-1.6 gives sigma >= 0.20; -20 is no floor and is what the inherited "
        "40.7 %% ran under. 📏 With entropy 0 the deviation shrinks to 0.061 by "
        "20 M steps and the policy stops exploring, so a long run without a "
        "floor measures a policy that has stopped changing. ⚠️ The predecessor's "
        "flag was spelled `--min-std 0.2` and it is not recorded whether that "
        "was a log value or a sigma; this one is unambiguous",
    )
    ap.add_argument(
        "--initial-log-std",
        type=float,
        nargs="+",
        default=[-0.5],
        help="one value, or one PER ACTION DIMENSION (x y z). 📏 The z axis is "
        "very nearly dead -- B0's mean |a_z| is 0.006 with std 0.053 against "
        "0.46 / 0.52 on x / y -- because altitude has a constant optimum at the "
        "derived ALT_MAX_M ceiling. A scalar sigma spends a third of the "
        "exploration budget there and pays twice: `energy` charges climb power "
        "and leaving the ceiling costs sightlines",
    )
    ap.add_argument(
        "--no-tanh-mean",
        action="store_true",
        help="emit the head's RAW output as the Gaussian mean; core._advance_drones "
        "already clamps to [-1, 1], so the bound is unchanged and only the density "
        "moves. 📏 B0 saturates at least one axis on 32.6 %% of steps (x 15.0, "
        "y 19.8), which a tanh mean reaches only asymptotically -- the same "
        "obstacle that made scripts/bc_init.py clip its targets to +-0.995",
    )
    ap.add_argument(
        "--orthogonal-init",
        action="store_true",
        help="the PPO reference initialisation (orthogonal, gain sqrt(2); policy "
        "head at 0.01). ⚠️ Absent from every run in this project's history",
    )
    ap.add_argument(
        "--layer-norm",
        action="store_true",
        help="LayerNorm before each hidden activation in the actor trunk",
    )
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
    if a.w_difference is not None:
        overrides["w_difference"] = a.w_difference
    weights = dataclasses.replace(weights, **overrides)

    failed = [k for k, ok in weight_constraints_satisfied(weights).items() if not ok]
    if failed:
        raise SystemExit(
            "these behavioural orderings no longer hold, so the reward no longer "
            f"ranks known policy pairs correctly: {failed}"
        )
    return weights


def _schedule(a: argparse.Namespace) -> CurriculumSchedule:
    """The shipped schedule unless a flag moves it. ⛔ Still a pure function of
    training progress -- `AGENTS.md` forbids adaptive advancement in a reported
    run, and nothing here gives the schedule a channel to the return."""
    kw = {}
    if a.curriculum_boundaries is not None:
        kw["boundaries"] = tuple(a.curriculum_boundaries)
    if a.curriculum_mix is not None:
        kw["mix"] = a.curriculum_mix
    return CurriculumSchedule(**kw)


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
            obs_history=a.obs_history,
            mask_jammed_obs=a.mask_jammed_obs,
            mask_broadcast_obs=a.mask_broadcast_obs,
            cue_mode=a.cue_mode,
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
        obs_history=a.obs_history,
        tanh_mean=not a.no_tanh_mean,
        orthogonal_init=a.orthogonal_init,
        layer_norm=a.layer_norm,
    ).to(a.device)
    init_from = None
    if a.init_from is not None:
        blob = torch.load(a.init_from, map_location=a.device, weights_only=False)
        if blob["architecture"] != a.arch:
            raise SystemExit(
                f"--init-from is a {blob['architecture']!r} checkpoint but --arch is "
                f"{a.arch!r}; loading it would score a different network"
            )
        actor.load_state_dict(blob["policy"])
        init_from = {"path": str(a.init_from), **blob.get("provenance", {})}
        print(f"    actor initialised from {a.init_from} (critic is fresh)")

    critic = SwarmCritic(env.cfg.state_dim, hidden=a.critic_hidden).to(a.device)

    ppo = PPOConfig(
        rollouts=rollouts,
        learning_epochs=a.epochs,
        gae_lambda=a.gae_lambda,
        grad_norm_clip=a.grad_norm_clip,
        mini_batches=mini_batches,
        learning_rate=a.lr,
        entropy_loss_scale=a.entropy,
        value_clip=a.value_clip,
        shuffle_minibatches=not a.no_shuffle,
        normalise_values=not a.no_value_norm,
        mini_batch_size=a.mini_batch_size,
        learning_rate_critic=a.lr_critic,
        grad_norm_clip_critic=a.grad_norm_clip_critic,
        target_kl=a.target_kl,
    )
    trainer = PPOTrainer(
        env,
        actor,
        critic,
        ppo,
        total_timesteps=a.timesteps,
        curriculum=None if a.no_curriculum else _schedule(a),
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
        "curriculum_boundaries": None if a.no_curriculum else list(_schedule(a).boundaries),
        "curriculum_mix": None if a.no_curriculum else _schedule(a).mix,
        "stage": None if not a.no_curriculum else a.stage,
        "lr": a.lr,
        "lr_critic": a.lr_critic,
        "mini_batch_size": a.mini_batch_size,
        "grad_norm_clip_critic": a.grad_norm_clip_critic,
        "target_kl": a.target_kl,
        "gae_lambda": a.gae_lambda,
        "grad_norm_clip": a.grad_norm_clip,
        "min_log_std": a.min_std,
        "initial_log_std": a.initial_log_std,
        "tanh_mean": not a.no_tanh_mean,
        "orthogonal_init": a.orthogonal_init,
        "layer_norm": a.layer_norm,
        "entropy_loss_scale": a.entropy,
        "shuffle_minibatches": not a.no_shuffle,
        "value_clip": a.value_clip,
        "init_from": init_from,
        "obs_history": a.obs_history,
        "mask_jammed_obs": a.mask_jammed_obs,
        "mask_broadcast_obs": a.mask_broadcast_obs,
        "cue_mode": a.cue_mode,
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
