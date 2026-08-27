# Results

**Tracked on purpose.** `results/` is in git and `runs/` is not: the summary is a
result, the checkpoints are regenerable. That keeps every number versioned with
the commit that produced it, so provenance cannot drift from the code.

One JSONL row per scored configuration, written by `scripts/eval_policy.py` and
`scripts/eval_baseline.py` via `--out`. ⚠️ Rows record **per-seed** values, not
just `median [IQR]` — every gate in `PLAN.md` is declared on the *worst* seed, and
a median alone cannot be judged against one.

⚠️ Guard against double-counting. An interrupted-and-resumed sweep in the
predecessor project appended eval rows unconditionally and reported `n = 9` where
5 were asked for. Check `n` before quoting anything.

The predecessor's results are **not** carried over; the old repository keeps them,
and the numbers that matter are consolidated in `docs/INHERITED.md`.
