r"""Grid sweep with the winner's curse designed out of it.

    uv run python scripts/sweep.py --axis lr=1e-4,3e-4,1e-3 --axis entropy=0.0,0.003 \
        --seeds 0 1 2 --device cuda:0 --out results/sweep_lr_entropy.jsonl

## Why this exists rather than a shell loop

📏 This project has already been bitten once: `docs/inherited/BLOCK_G.md` records a
**45.1 %** headline from a `dref400_k30` cell at 3 seeds that did **not** reproduce
— *"the shaping axis is noise, so that cell was the winner's curse, exactly as the
stage-A analysis predicted."* A sweep turns that from an occasional accident into
a systematic one: the more configurations you score, the more certain it becomes
that the best-looking cell is the luckiest rather than the best.

🔒 **So three rules are enforced by this tool rather than left to discipline**,
because a shell loop enforces none of them:

1. **The search runs on the TRAIN split.** ⛔ There is no flag to search on eval.
   The eval split is the one generalisation check this project has left, and it is
   already weak (`PLAN.md` §9: held-out routes through the same buildings). Scoring
   200 configurations against it consumes it.
2. **Configurations are ranked on the WORST seed**, never the median. `AGENTS.md`
   requires it, and a median-ranked sweep hands you the luckiest seed of the
   luckiest configuration — two selection effects stacked.
3. **The winner is re-run at FRESH seeds on the eval split**, and both numbers are
   reported side by side. The search score is biased upward by selection; the
   confirmation is the unbiased estimate. ⚠️ **If they disagree, the disagreement
   is the finding** — that is what the 45.1 % cell would have shown.

## What it does not do

⛔ **It does not decide anything.** A sweep is a *tuning* instrument; promoting its
winner into a reported result still needs a gate declared before its own run, at
>= 5 seeds, judged on the worst (`AGENTS.md`). This script deliberately prints the
confirmation next to the search score and stops there.

⚠️ **A grid is not a search strategy for more than ~3 axes.** For anything larger,
sweep one axis at a time against a fixed control — the interaction terms this grid
would spend its budget on are almost certainly smaller than the seed spread.

## Resumability and honesty about failures

Rows are appended as each configuration finishes, and a re-run **skips
configurations already present in `--out`**, so a crash costs one cell rather than
the sweep. ⛔ A configuration whose training or scoring fails is recorded with
`"status": "failed"` and is **excluded from ranking** rather than being scored on
whatever seeds happened to survive — silently ranking a 3-seed cell against 5-seed
cells is exactly the kind of comparison this file exists to prevent.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Axis name -> the `scripts/train.py` flag it sets. ⛔ Deliberately a whitelist:
#: a typo in `--axis` should fail loudly here rather than be silently forwarded to
#: argparse and ignored, which would sweep a grid of identical configurations.
AXES: dict[str, str] = {
    "lr": "--lr",
    "entropy": "--entropy",
    "value_clip": "--value-clip",
    "epochs": "--epochs",
    "gae_lambda": "--gae-lambda",
    "grad_norm_clip": "--grad-norm-clip",
    "initial_log_std": "--initial-log-std",
    "min_std": "--min-log-std",
    # -- the axes nothing in this project has ever swept, 2026-09-04 ----------
    #: ☠️ The optimisation budget. docs/inherited/BLOCK_G.md held gradient
    #: density at 488 steps/M across all three cadences, which pins the minibatch
    #: at 40,960 rows in every one of them -- ~5,900 Adam steps for a 12 M run.
    "mini_batch_size": "--mini-batch-size",
    "target_kl": "--target-kl",
    "grad_norm_clip_critic": "--grad-norm-clip-critic",
    #: 🔍 The difference reward, `results/capability_gates.md` Gate E.
    "w_difference": "--w-difference",
    #: 📏 Observation content, `results/capability_gates.md` "not in either gate".
    #: ⛔ `--mask-broadcast-obs` is deliberately NOT here: it is a `store_true`
    #: flag and `run_one` emits `[flag, value]` pairs, so an axis over it would
    #: pass an unrecognised positional and die. Every axis in this table must
    #: name a flag that TAKES a value. (📏 `min_std` pointed at `--min-std`,
    #: which `train.py` never defined -- that axis had been dead since it was
    #: written, and it is why this rule is now stated rather than assumed.)
    "cue_mode": "--cue-mode",
    #: ⚠️ BLOCK_G lists the curriculum schedule as provisional and never measured.
    "curriculum_mix": "--curriculum-mix",
    "cadence": "--cadence",
    "arch": "--arch",
    "obs_history": "--obs-history",
    "num_drones": "--num-drones",
    "jammer": "--jammer",
    "fidelity": "--fidelity",
}


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def parse_axis(spec: str) -> tuple[str, list[str]]:
    if "=" not in spec:
        raise SystemExit(f"--axis wants name=v1,v2,... ; got {spec!r}")
    name, values = spec.split("=", 1)
    name = name.strip()
    if name not in AXES:
        raise SystemExit(f"unknown axis {name!r}; known axes: {', '.join(sorted(AXES))}")
    vals = [v.strip() for v in values.split(",") if v.strip()]
    if not vals:
        raise SystemExit(f"axis {name!r} has no values")
    return name, vals


def config_key(cfg: dict[str, str]) -> str:
    """Stable identity for a configuration, used for tags and for resume."""
    return "_".join(f"{k}{cfg[k]}" for k in sorted(cfg)).replace(".", "p").replace("-", "m")


def build_train_cmd(a, cfg: dict[str, str], seed: int, tag: str) -> list[str]:
    """The `scripts/train.py` invocation for one cell. Pure, so it can be tested.

    ⛔ Never carries `--eval-routes`. `train.py` defaults to the train split and
    the search must stay there; the eval split is spent once, on the winner.
    """
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--device",
        a.device,
        "--seeds",
        str(seed),
        "--timesteps",
        str(a.timesteps),
        "--tag",
        tag,
        "--out-root",
        str(a.run_root),
    ]
    # ⛔ No --eval-routes. The search trains and scores on the train split; see
    # the module docstring. train.py defaults to the train split.
    for extra in a.train_arg:
        flag, _, value = extra.partition("=")
        cmd += [f"--{flag.lstrip('-')}"] + ([value] if value else [])
    for name, value in cfg.items():
        cmd += [AXES[name], value]
    return cmd


def train_one(a, cfg: dict[str, str], seed: int, tag: str) -> Path | None:
    """One training run. Returns the checkpoint path, or None if it failed."""
    ckpt = a.run_root / f"{tag}-s{seed}" / "checkpoint.pt"
    if ckpt.exists() and not a.retrain:
        return ckpt
    cmd = build_train_cmd(a, cfg, seed, tag)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    if proc.returncode != 0 or not ckpt.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        print(f"      ⛔ seed {seed} FAILED: {' | '.join(tail)[:200]}", flush=True)
        return None
    return ckpt


def score(a, checkpoints: list[Path], label: str, train_routes: bool) -> dict | None:
    """Score a set of seeds as one condition through `eval_policy.py`.

    🔒 Everything goes through that script, which goes through
    `src/baselines/evaluate.py` -- the one harness B0's numbers came from. A
    learned policy scored by a different loop is not comparable to B0, and that
    comparison is the entire reason B0 exists (`AGENTS.md`).
    """
    tmp = Path(tempfile.mkdtemp()) / "score.jsonl"
    cmd = build_score_cmd(a, checkpoints, label, train_routes, tmp)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    try:
        if proc.returncode != 0 or not tmp.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            print(f"      ⛔ scoring FAILED: {' | '.join(tail)[:200]}", flush=True)
            return None
        return json.loads(tmp.read_text().strip().splitlines()[-1])
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def build_score_cmd(a, checkpoints, label: str, train_routes: bool, out: Path) -> list[str]:
    """The `eval_policy.py` invocation. Pure, so the split choice can be tested.

    🔒 `--train-routes` is passed for the SEARCH and omitted for the confirmation;
    that one boolean is the whole train/eval discipline, so it is a parameter
    rather than a global.
    """
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_policy.py"),
        *[str(c) for c in checkpoints],
        "--group",
        label,
        "--device",
        a.device,
        "--num-envs",
        str(a.eval_envs),
        "--seeds",
        str(len(checkpoints)),
        "--stage",
        str(a.stage),
        "--fidelity",
        a.fidelity,
        "--jammer",
        a.eval_jammer,
        "--out",
        str(out),
    ]
    if train_routes:
        cmd.append("--train-routes")
    if a.obs_history_for_eval > 1:
        cmd += ["--obs-history", str(a.obs_history_for_eval)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--axis",
        action="append",
        default=[],
        metavar="NAME=V1,V2",
        help=f"sweepable: {', '.join(sorted(AXES))}",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2], help="training seeds for the SEARCH"
    )
    ap.add_argument(
        "--confirm-seeds",
        nargs="+",
        type=int,
        default=[100, 101, 102, 103, 104],
        help="FRESH seeds for the winner's confirmation. ⛔ Must not overlap --seeds",
    )
    ap.add_argument("--timesteps", type=int, default=12_000_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-envs", type=int, default=128)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--fidelity", default="F4")
    ap.add_argument("--eval-jammer", default="J1", help="rung to SCORE at (not to train at)")
    ap.add_argument(
        "--obs-history-for-eval",
        type=int,
        default=1,
        help="set when sweeping obs_history so the env matches the checkpoint",
    )
    ap.add_argument("--run-root", type=Path, default=Path("runs/sweep"))
    ap.add_argument("--out", type=Path, required=True, help="JSONL, one row per config")
    ap.add_argument(
        "--train-arg",
        action="append",
        default=[],
        metavar="FLAG=VALUE",
        help="forward a fixed scripts/train.py flag to every run, e.g. "
        "--train-arg num-envs=512. Use --axis for anything that VARIES; this is for "
        "what is held constant. Bare switches take no value: --train-arg no-curriculum",
    )
    ap.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument(
        "--no-confirm",
        action="store_true",
        help="skip the fresh-seed confirmation. ⚠️ Then the winner is a "
        "biased estimate and must not be quoted",
    )
    a = ap.parse_args()

    if not a.axis:
        raise SystemExit("no --axis given; nothing to sweep")
    overlap = set(a.seeds) & set(a.confirm_seeds)
    if overlap and not a.no_confirm:
        raise SystemExit(
            f"--confirm-seeds overlaps --seeds at {sorted(overlap)}. The confirmation "
            "exists to measure the winner on seeds the selection never saw; reusing one "
            "re-introduces exactly the bias it is there to remove."
        )

    axes = dict(parse_axis(s) for s in a.axis)
    fixed = {e.partition("=")[0].lstrip("-").replace("-", "_") for e in a.train_arg}
    clash = fixed & set(axes)
    if clash:
        raise SystemExit(
            f"--train-arg fixes {sorted(clash)} which --axis also sweeps. The axis value "
            "would be appended after the fixed one and silently win, so every cell would "
            "look distinct and train identically."
        )
    names = sorted(axes)
    grid = [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(*(axes[n] for n in names))
    ]

    per_run_s = 2.7 * 60 * (a.timesteps / 12_000_000)
    total = len(grid) * len(a.seeds)
    print(f"\n  {len(grid)} configurations x {len(a.seeds)} seeds = {total} runs", flush=True)
    print("  axes: " + "; ".join(f"{n}={','.join(axes[n])}" for n in names), flush=True)
    print("  search: TRAIN split, ranked on the WORST seed", flush=True)
    print(
        f"  confirm: {'OFF ⚠️' if a.no_confirm else f'EVAL split, fresh seeds {a.confirm_seeds}'}",
        flush=True,
    )
    print(
        f"  ~{total * per_run_s / 3600:.1f} h of training at {a.timesteps:,} steps/run "
        f"(reference: 2.7 min per 12 M on an RTX 5090)\n"
    )
    if a.dry_run:
        for i, cfg in enumerate(grid, 1):
            print(f"    {i:>3}. {config_key(cfg)}", flush=True)
        return

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line).get("key"))
        if done:
            print(f"  resuming: {len(done)} configuration(s) already in {a.out}\n", flush=True)

    sha = git_sha()
    for i, cfg in enumerate(grid, 1):
        key = config_key(cfg)
        if key in done:
            print(f"  [{i}/{len(grid)}] {key}  — already recorded, skipping", flush=True)
            continue
        print(f"  [{i}/{len(grid)}] {key}", flush=True)
        t0 = time.time()
        ckpts = [train_one(a, cfg, s, f"sw-{key}") for s in a.seeds]
        row: dict = {
            "key": key,
            "config": cfg,
            "seeds": a.seeds,
            "timesteps": a.timesteps,
            "device": a.device,
            "split": "train",
            "stage": a.stage,
            "fidelity": a.fidelity,
            "eval_jammer": a.eval_jammer,
            "git_sha": sha,
            "elapsed_s": None,
            "status": "ok",
        }
        if any(c is None for c in ckpts):
            # ⛔ Excluded from ranking. Scoring the seeds that survived would put a
            # 2-seed cell next to 3-seed cells in the same ranking.
            row |= {
                "status": "failed",
                "reason": "a training seed failed",
                "elapsed_s": round(time.time() - t0, 1),
            }
        else:
            res = score(a, [c for c in ckpts if c], key, train_routes=True)
            if res is None:
                row |= {"status": "failed", "reason": "scoring failed"}
            else:
                mc = res["seeds"]["mission_capable"]
                row |= {
                    "mission_capable": mc,
                    "worst": min(mc),
                    "median": sorted(mc)[len(mc) // 2],
                    "metrics": res["median"],
                }
            row["elapsed_s"] = round(time.time() - t0, 1)
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if row["status"] == "ok":
            print(
                f"      worst {row['worst'] * 100:.2f} %   median {row['median'] * 100:.2f} %"
                f"   ({row['elapsed_s'] / 60:.1f} min)"
            )

    rows = [json.loads(l) for l in a.out.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    if not ok:
        raise SystemExit("\n  no configuration completed; nothing to rank\n")

    ok.sort(key=lambda r: r["worst"], reverse=True)
    print(f"\n  === RANKED ON THE WORST SEED — train split, {len(a.seeds)} seeds ===\n", flush=True)
    print(f"  {'':<3}{'worst':>9}{'median':>9}   configuration", flush=True)
    for rank, r in enumerate(ok[:15], 1):
        print(
            f"  {rank:<3}{r['worst'] * 100:>8.2f}%{r['median'] * 100:>8.2f}%   {r['key']}",
            flush=True,
        )
    if failed:
        print(
            f"\n  ⛔ {len(failed)} configuration(s) excluded: "
            + ", ".join(f"{r['key']} ({r.get('reason')})" for r in failed[:5])
        )

    if a.no_confirm:
        print(
            "\n  ⚠️  --no-confirm: the winner above is a BIASED estimate, selected on the "
            "same data that scored it. Do not quote it.\n"
        )
        return

    win = ok[0]
    print(
        f"\n  === CONFIRMING {win['key']} on FRESH seeds {a.confirm_seeds}, EVAL split ===\n",
        flush=True,
    )
    cfg = win["config"]
    ckpts = [train_one(a, cfg, s, f"sw-{win['key']}") for s in a.confirm_seeds]
    if any(c is None for c in ckpts):
        raise SystemExit("  ⛔ a confirmation seed failed to train; not reporting a number\n")
    res = score(a, [c for c in ckpts if c], f"{win['key']}-confirm", train_routes=False)
    if res is None:
        raise SystemExit("  ⛔ confirmation scoring failed\n")

    mc = res["seeds"]["mission_capable"]
    conf = {
        "key": win["key"],
        "config": cfg,
        "seeds": a.confirm_seeds,
        "split": "eval",
        "mission_capable": mc,
        "worst": min(mc),
        "median": sorted(mc)[len(mc) // 2],
        "metrics": res["median"],
        "git_sha": sha,
        "status": "confirmation",
        "search_worst": win["worst"],
        "search_median": win["median"],
    }
    with a.out.open("a") as fh:
        fh.write(json.dumps(conf) + "\n")

    d_worst = (conf["worst"] - win["worst"]) * 100
    d_med = (conf["median"] - win["median"]) * 100
    print(f"  {'':<12}{'worst':>9}{'median':>9}", flush=True)
    print(
        f"  {'search':<12}{win['worst'] * 100:>8.2f}%{win['median'] * 100:>8.2f}%   "
        f"(train split, seeds {a.seeds})"
    )
    print(
        f"  {'confirm':<12}{conf['worst'] * 100:>8.2f}%{conf['median'] * 100:>8.2f}%   "
        f"(EVAL split, seeds {a.confirm_seeds})"
    )
    print(f"  {'delta':<12}{d_worst:>+8.2f}{'':>1}{d_med:>+8.2f}\n", flush=True)
    if d_worst < -2.0:
        print(
            "  ⚠️  The winner lost more than 2 pp on fresh seeds. That is the winner's", flush=True
        )
        print(
            "      curse, and the DISAGREEMENT is the finding -- not the search score.\n",
            flush=True,
        )
    print(
        "  ⛔ A sweep tunes; it does not decide. Promoting this needs a gate declared", flush=True
    )
    print("     before its own run, at >= 5 seeds, judged on the worst (AGENTS.md).\n", flush=True)


if __name__ == "__main__":
    main()
