# Masking the jammed observations — ⚠️ **NULL**, and it scopes the claim

**Measured 2026-09-04.** `results/obs_mask.jsonl`, cuda:0, eval split, stage 4,
F4, `gnn`/`deep`, 5 seeds × 128 episodes. Control is `rq2-gnn-deep`, same
architecture, cadence and budget, flag off.

🔒 **The three branches were committed in `1f631e8` before the run**, in
`PLAN.md` §7 — *"the claim holds constructively" / "the claim holds, expensively"
/ "null: neither moves"*, the last with its reason already stated.

## 📏 Result

| | capability (J1) | gap J1 → J3B | gap range |
|---|---|---|---|
| control (unmasked) | 39.89 % | **11.12 pp** | [10.11 – 13.58] |
| **masked** | **38.38 %** | **10.14 pp** | [7.99 – 12.20] |
| Δ | **−1.51 pp** | **−0.98 pp** | heavily overlapping |

⚠️ **Neither moved.** The third branch, as written: *"null: neither moves."*

## 🔍 Why — and this is the useful part

📏 **The masked features are worth 1.51 pp of capability.** That is the whole
size of the learned policy's loop on the jammed quantity. ⛔ **You cannot extract
11 pp of exploitability from a 1.5 pp loop.** The treatment removed something the
policy barely used, so the experiment did not *test* the claim — it measured how
little there was to remove.

📏 For scale, [`frontier.md`](frontier.md) prices B0's repair loop at **+6.90 pp
of capability and +6.85 pp of exploitability**. The learned loop is **~4.5×
smaller**, and prices out at a broadly similar ratio (1.51 : 0.98) — though ⚠️
both terms here sit inside the seed ranges and that ratio must not be quoted as a
measurement.

## ☠️ What this reveals: the learned policies' exploitability is not loop-driven

If the loop is worth ~1 pp, the other **~10 pp** of `rq2-gnn-deep`'s gap is
something else. 📏 Sorted by capacity headroom over the 15 Mbps bar:

| policy | mean capacity @ J1 | headroom | gap |
|---|---|---|---|
| random | 4.7 | −10.3 | 2.05 |
| mlp | 11.0 | −4.0 | 8.60 |
| **masked** | 13.9 | **−1.1** | 10.14 |
| deepsets | 14.2 | **−0.8** | 10.45 |
| advtrain-J2 | 14.3 | **−0.7** | 7.51 |
| gnn | 14.5 | **−0.5** | 11.12 |
| advtrain-J3B | 14.7 | **−0.3** | 7.29 |
| `b0-geodesic` | 18.3 | +3.3 | 6.39 |
| **B0** | **21.6** | **+6.6** | **13.24** |

🔍 **Every learned policy sits within 1.1 Mbps of the threshold.** On that knife
edge, each dB the jammer removes crosses the bar, so `mission_capable` falls
without any behavioural response at all. **That is damage, not exploitation.**

## 🔒 The decomposition this forces, and it explains all nine rows

> **gap = f(threshold proximity) + g(loop strength)**

| policy | f — how close to the bar | g — the loop | gap |
|---|---|---|---|
| random | ~0, nothing to lose | 0 | 2.05 |
| learned cluster | **large**, sitting on the bar | ~1 | 7.3 – 11.1 |
| `b0-geodesic` | small, 3.3 Mbps of headroom | **0** | 6.39 |
| B0 | smallest, 6.6 Mbps of headroom | **large** | 13.24 |

✅ **The claim survives where it is actually tested**, and the test is
conservative: B0 has **more** headroom than geodesic (+6.6 against +3.3), so its
`f` term is *smaller* — and its gap is still **twice** geodesic's. The 6.85 pp is
therefore a **lower bound** on what the loop contributes.

✅ **And co-training is not headroom either.** `advtrain-J2` (−0.7) against
`deepsets` (−0.8) and `gnn` (−0.5): the same knife edge, gaps 7.51 against 10.45
and 11.12. Same `f`, different outcome. RQ3 stands.

## ⛔ What must change in the write-up

`PLAN.md` §1 claims exploitability *is* a cost of adaptivity. 📏 That is **too
general**. The measured claim is:

> Closing a feedback loop on the attacked quantity adds exploitability **on top of
> the damage a policy's operating point already exposes it to**. B0's repair loop
> adds ≥6.85 pp. It is not the only term, and for policies sitting on the
> capability threshold it is not the dominant one.

⚠️ **The learned policies in this project cannot test the claim**, because they
have no loop worth removing. Testing it on the learned side needs a policy with
**capacity headroom** — i.e. one that is actually good — which is the thing this
project has not managed to train.

## What would test it next

⛔ **Not a bigger mask.** `on_path` and `steps_since_link` are the remaining
indirect channels, but a 1.5 pp loop does not become an 11 pp loop by adding two
routing-derived scalars.

🔧 The honest options, in order:

1. **Report the decomposition as the finding.** It explains nine policies with two
   terms, one of which is measured on a controlled pair. That is a stronger result
   than the single claim it replaces.
2. **Run 2 (`repair_score="clearance"`) and run 3 (`repair_amplitude_m` dose–
   response)** — both zero-training, both on the *only* policies in this project
   with a loop big enough to measure.
3. ⚠️ A learned policy with real headroom would test it properly, and that is the
   15 pp capability gap `PLAN.md` §3 closed on five lines of evidence. **Do not
   re-open that to rescue this.**
