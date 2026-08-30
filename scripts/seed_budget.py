"""How many training seeds does a 3 pp effect actually need?

    uv run python scripts/seed_budget.py results/trainer_validation.jsonl \
        --policy gnn/deep/shipped --tolerance 3.0

📏 **The question has never been asked in this project, and it is cheap.** The
run budget (10 M steps) and the seed count (5) were sized against an *estimate*
of 2.8 h per run. G1b then measured **2.2 min** — ~76x cheaper — and nobody
re-spent the savings. Meanwhile every intervention this project has judged was
judged at 5 seeds, on effects of 0.4-1.7 pp, against a measured seed IQR of
3.2-6.9 and a *bimodal* distribution (per-seed `episode_return` ran
4.8 / 83.8 / 90.5 / 96.8 / 108.9 — roughly one catastrophic seed in five).

This script takes the per-training-seed `mission_capable` values that
`eval_policy.py --group --out` writes and answers one question by exhaustive or
Monte-Carlo resampling:

> Drawing `k` of the measured seeds, how often does the `k`-seed median land
> within `tolerance` of the full-sample median?

⚠️ It is a statement about **this** measured seed distribution, not a general
power calculation. With 20 seeds the tail is still thin, so read the numbers as
"at least this bad" rather than as a precise coverage.

⛔ It reports the **worst seed** alongside, because `AGENTS.md` judges gates on
the worst seed and this project has been misled by medians twice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path


def coverage(values: list[float], k: int, tolerance: float, cap: int = 200_000) -> float:
    """Fraction of `k`-subsets whose median is within `tolerance` of the full one."""
    truth = statistics.median(values)
    combos = itertools.combinations(range(len(values)), k)
    hits = total = 0
    for idx in itertools.islice(combos, cap):
        if abs(statistics.median([values[i] for i in idx]) - truth) <= tolerance:
            hits += 1
        total += 1
    return hits / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--policy", required=True, help="the `policy` field to select")
    ap.add_argument("--metric", default="mission_capable")
    ap.add_argument("--tolerance", type=float, default=3.0, help="in percentage points")
    a = ap.parse_args()

    rows = [json.loads(line) for line in a.jsonl.read_text().splitlines() if line.strip()]
    matching = [r for r in rows if r["policy"] == a.policy and r.get("grouped")]
    if not matching:
        raise SystemExit(
            f"no grouped row with policy={a.policy!r} in {a.jsonl}. "
            "`--group` is what makes a row's seeds TRAINING seeds rather than "
            "evaluation episodes, and only the former answers this question."
        )
    row = matching[-1]
    values = [v * 100.0 for v in row["seeds"][a.metric]]
    n = len(values)

    print(
        f"{a.policy}  {a.metric}  device={row['device']}  split={row['split']}  {n} training seeds"
    )
    print("⚠️  device is part of provenance: never compare across devices.\n")
    print("  per seed: " + "  ".join(f"{v:.1f}" for v in sorted(values)))
    print(
        f"  median {statistics.median(values):.1f}   "
        f"IQR {statistics.quantiles(values, n=4)[2] - statistics.quantiles(values, n=4)[0]:.1f}   "
        f"worst {min(values):.1f}   best {max(values):.1f}\n"
    )

    print(f"  coverage: P(|median of k seeds - median of {n}| <= {a.tolerance} pp)")
    print(f"  {'k':>4}{'coverage':>12}{'worst-seed spread':>22}")
    for k in range(3, n + 1):
        cov = coverage(values, k, a.tolerance)
        # How far the WORST seed of a k-draw ranges -- the quantity gates are
        # actually judged on.
        worsts = [
            min(values[i] for i in idx)
            for idx in itertools.islice(itertools.combinations(range(n), k), 20000)
        ]
        print(f"  {k:>4}{cov:>11.0%}{min(worsts):>12.1f} - {max(worsts):<8.1f}")


if __name__ == "__main__":
    main()
