r"""What is B0's +10.1 pp actually made of?

**Declared 2026-09-02, before the numbers existed.**

## Why this has to be measured rather than inferred

📏 `docs/inherited/BLOCK_E.md` measures `B0-geodesic` **47.1 %** → `B0`
**57.2 %** = **+10.1 pp**, and calls the difference "design effort". The b0
variant adds four things at once:

> ranked roles from the sees bits, target-belief filter with dead reckoning,
> spare observation posts, local link repair

📏 Two of those are already accounted for:

* the **belief filter** is bounded above by **0.4 pp** — `B0` → `B0-oracle` is
  −0.4 pp, and perfect target state is strictly more than any filter can
  reconstruct ([`results/memory_horizon.md`](../results/memory_horizon.md));
* **spare observation posts are OFF in the shipped config** — `B0Config.max_spares
  = 0`, so they contribute exactly nothing to the reported 57.2 %.

⛔ **So the +10.1 pp is split between RANKED ROLES and LOCAL LINK REPAIR, and
nobody has measured which.** `results/credit_assignment.md` and
`results/memory_horizon.md` both *infer* it is roles. Inference is not
measurement, and this project has been wrong twice this week by reasoning past a
number it could have taken.

⚠️ This matters because it sizes a prize. If ranked roles carry ~9 of the 10 pp,
the credit-assignment work has a target worth building for. If **link repair**
carries most of it, then B0's advantage is a *link-maintenance heuristic*, the
role story is wrong, and the critic work is aimed at the wrong deficit.

## The ablation

🔒 **Nothing about the shipped B0 changes.** Each arm varies exactly one field of
`B0Config` from the shipped default, or selects an existing variant. `b0` is
re-run here as the control so every arm is scored on the same episodes.

| arm | what it removes | isolates |
|---|---|---|
| `b0` | — (control) | |
| `b0-norepair` | `repair_amplitude_m = 0` | **local link repair** |
| `geodesic` | ranked roles → roles fixed by index, and the belief filter | **roles + belief together** |
| `oracle` | — (adds ground truth) | the belief bound, re-confirmed |

📏 `repair_along` is already 0.0 and `max_spares` already 0 in the shipped
config, so neither needs an arm.

## 🔒 Declared reading, before the run

Let `R` = `b0` − `b0-norepair` (what link repair is worth).

| | rule | consequence |
|---|---|---|
| **roles carry it** | `R` <= **3 pp** | ~7+ pp is roles-and-belief, belief is <= 0.4, so **roles**. The credit-assignment work has a measured target |
| **repair carries it** | `R` >= **7 pp** | B0's edge is link maintenance, not role assignment. ⛔ The role story in `credit_assignment.md` and `memory_horizon.md` is **wrong** and both need correcting |
| **shared** | 3 pp < `R` < 7 pp | report the split and do not attribute the deficit to one mechanism |

⚠️ **A ceiling, declared now.** Whatever `R` turns out to be, the role component
is at most `10.1 − R − 0.4` pp, against a **15.0 pp** gap to B0 at J3B. ⛔ So even
a policy that acquired role assignment *perfectly* would not close the gap on this
evidence, and "will it surpass B0" is not a question this decomposition can answer
yes to. Recorded before the run so the result cannot be read as more than it is.

Usage:

    uv run python scripts/measure_b0_ablation.py --seeds 5 --num-envs 128 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_baseline import make_env, med_iqr, run
from src.baselines import B0Config

#: Each arm: (label, variant name for `run`, B0Config override or None).
ARMS: tuple[tuple[str, str, dict], ...] = (
    ("b0", "b0", {}),
    ("b0-norepair", "b0", {"repair_amplitude_m": 0.0}),
    ("geodesic", "geodesic", {}),
    ("oracle", "oracle", {}),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-routes", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    if a.seeds < 5:
        print(f"⚠️  {a.seeds} seeds. AGENTS.md requires >=5 for anything reported as a finding.\n")

    base = B0Config()
    results: dict[str, list[float]] = {}
    extra: dict[str, dict[str, list[float]]] = {}

    print(
        f"\n  device={a.device}  {a.seeds} seeds x {a.num_envs} episodes  "
        f"split={'train' if a.train_routes else 'eval'}  stage 4\n"
    )

    for label, variant, override in ARMS:
        caps, obs_, rng, roles = [], [], [], []
        for seed in range(a.seeds):
            env = make_env(a.num_envs, a.num_drones, seed, not a.train_routes, a.device != "cpu")
            cfg = replace(base, **override) if override else None
            m = run(variant, env, cfg).summary()
            caps.append(m["mission_capable"])
            obs_.append(m["observed"])
            rng.append(m["observer_range_m"])
            roles.append(m["role_entropy"])
        results[label] = caps
        extra[label] = {"observed": obs_, "observer_range_m": rng, "role_entropy": roles}
        med, iqr = med_iqr(caps)
        print(
            f"  {label:<14} {med * 100:>6.1f} % [{iqr * 100:.1f}]   worst {min(caps) * 100:>5.1f} %"
            f"   observed {med_iqr(obs_)[0] * 100:>5.1f} %"
            f"   obs_range {med_iqr(rng)[0]:>6.1f} m"
            f"   role_H {med_iqr(roles)[0]:.2f}"
        )

    b0_med = med_iqr(results["b0"])[0]
    r = (b0_med - med_iqr(results["b0-norepair"])[0]) * 100
    geo = (b0_med - med_iqr(results["geodesic"])[0]) * 100
    orc = (b0_med - med_iqr(results["oracle"])[0]) * 100

    print(f"\n  📏 link repair      R = {r:+.2f} pp")
    print(f"  📏 roles + belief       {geo:+.2f} pp   (belief bounded at 0.4 by the oracle)")
    print(f"  📏 target information   {orc:+.2f} pp   (re-confirming BLOCK_E's −0.4)")
    verdict = (
        "ROLES carry it (R <= 3 pp)"
        if r <= 3.0
        else "REPAIR carries it (R >= 7 pp) -- the role story is WRONG"
        if r >= 7.0
        else "SHARED (3 < R < 7 pp) -- do not attribute to one mechanism"
    )
    print(f"  🔒 declared reading: {verdict}\n")

    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "device": a.device,
                        "seeds": a.seeds,
                        "num_envs": a.num_envs,
                        "num_drones": a.num_drones,
                        "split": "train" if a.train_routes else "eval",
                        "mission_capable": results,
                        "other": extra,
                        "link_repair_pp": r,
                        "roles_plus_belief_pp": geo,
                        "target_information_pp": orc,
                        "verdict": verdict,
                    }
                )
                + "\n"
            )
        print(f"  -> {a.out}\n")


if __name__ == "__main__":
    main()
