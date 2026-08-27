# Contested Relay

Adversarially robust multi-agent UAV relay under contested spectrum, from GPU
simulation to embedded deployment.

A swarm of `N` UAVs must observe a moving ground target in Frankfurt, relay the
sensor feed to a command vehicle over a multi-hop chain at >= 15 Mbps, and
survive on finite batteries while a jammer degrades links near the target.
Buildings block line of sight, so the relay chain is geometrically necessary.

**The question is not whether learned control beats a scripted baseline on the
static task.** It does not, and a heuristic should win a static, fully-specified
problem. The question is how far each policy degrades when the adversary
*adapts* to it. See [`PLAN.md`](PLAN.md).

## Provenance

This repository is seeded from a predecessor project
(`lucwestbomke/master-thesis-preview`) and is **not** a fresh start. The channel
model, occlusion kernel, routing DP, energy model and their tests are inherited
unchanged: they are where that project's bugs were found and fixed, and retyping
them would re-introduce those bugs. What is deliberately left behind, and what
still has to be stripped out of what was carried over, is recorded in
[`docs/REDUCTION.md`](docs/REDUCTION.md).

Predecessor documentation is archived read-only under
[`docs/inherited/`](docs/inherited/). The measured constants that carry forward,
each with its provenance, are consolidated in
[`docs/INHERITED.md`](docs/INHERITED.md).

## Getting started

```bash
uv sync --extra dev      # `dev` is an EXTRA -- plain `uv sync` gives no pytest
uv run pytest -q         # 333 passed, 4 skipped (7 CUDA-gated tests skip on arm64)
uv run ruff check . && uv run ruff format .
```

`data/frankfurt_box.npz` is **committed on purpose**, not a build product. OSM
and the Hessen LoD2 service both change, so re-running `scripts/prep_osm.py` in
2027 would silently produce a different map. The file in git *is* the
environment; the script only documents how it was made.

Start with [`AGENTS.md`](AGENTS.md).
