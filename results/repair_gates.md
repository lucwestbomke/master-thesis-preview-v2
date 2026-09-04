# The loop's **amplitude** is what costs. Its **target** is not.

**Measured 2026-09-04**, `results/repair_score.jsonl` and
`results/repair_dose.jsonl`. cuda:0, eval split, stage 4, F4, 5 seeds × 128
episodes, through `eval_policy.py` — the harness every other number came from.
🔒 The shipped B0 is untouched; each arm is one `B0Config` field.

Two factors, varied separately: run 2 changed **what the repair loop is scored
on** at fixed amplitude; run 3 changed **how far it may move** at fixed score.

---

## ⛔ Run 2 — the target does **not** matter. The sharp claim is refuted.

`_update_repair` hill-climbs on either the chain's bottleneck **capacity**
(SINR-derived, so the jammer moves it) or its **clearance** (building occlusion,
which the jammer *cannot* touch).

| arm | capability (J1) | gap J1 → J3B | range |
|---|---|---|---|
| `b0-geodesic` — no loop at all | 45.6 % | **6.39 pp** | [5.68 – 6.54] |
| **b0, repair on CLEARANCE** | 54.0 % | **13.37 pp** | [12.60 – 16.36] |
| b0, repair on CAPACITY (shipped) | 57.3 % | **13.24 pp** | [11.42 – 13.58] |

☠️ **Pointing the loop at a quantity the jammer cannot move changes nothing** —
13.37 against 13.24, overlapping ranges, if anything slightly *worse*.

⛔ **So `PLAN.md` §1's sharp clause — *"a loop on the quantity an adversary
attacks"* — is wrong.** It is not about the quantity.

🔍 **The mechanism, and it was already written down.** `_update_repair` scores
only the edges it is *carrying*: `torch.where(nb_onpath, edge_clr, _BIG_M)`.
`on_path` comes from the router, the router picks the widest path, and widest
depends on capacity — **which the jammer sets.** So the adversary drives the loop
through **routing**, whatever the loop is scored on.

⚠️ This is exactly the caveat recorded in `core.py` when the observation mask was
built: *"`on_path` (21) and `steps_since_link` (23) are routing-derived and
therefore reachable indirectly."* It was left unmasked deliberately; here it turns
out to be the whole story.

---

## 📏 Run 3 — the amplitude **does** matter, and the shipped value is wrong

`repair_amplitude_m` bounds how far a relay may slide from its station.

| `repair_amplitude_m` | capability (J1) | **at J3B** | gap | range |
|---|---|---|---|---|
| **0** — loop disabled | 50.9 % | 42.3 % | **7.94 pp** | [7.50 – 8.66] |
| 50 | 55.0 % | 45.5 % | 9.59 pp | [9.12 – 10.26] |
| **100** | 56.9 % | **47.3 %** | 9.54 pp | [9.23 – 11.94] |
| **200** — shipped | **57.3 %** | 43.8 % | **13.24 pp** | [11.42 – 13.58] |

✅ **A dose–response, with disjoint endpoints**: 0 → 50 is disjoint
(8.66 < 9.12), and 0 → 200 is disjoint by a wide margin. ⚠️ 50 and 100 are flat
and overlapping, so the curve is a step rather than a ramp — 📏 four points, and
the shape should not be over-read.

☠️ **Capability saturates while the cost keeps climbing.** The last doubling,
100 → 200, buys **+0.4 pp** of capability and costs **+3.7 pp** of exploitability.

## ⭐ And it produces a policy that is both more robust and better under attack

📏 **At J3B, `repair_amplitude_m = 100` scores 47.3 % against the shipped B0's
43.8 % — winning on 5 of 5 paired seeds, +3.58 pp median, for 0.46 pp of J1
capability.**

🔍 **The shipped amplitude is tuned for an unjammed world.** 200 m is the value
that maximises capability when nothing is attacking; under the strongest adversary
built it is roughly **twice too large**. A policy that is willing to be led 200 m
out of position is led 200 m out of position.

---

## 🔒 The claim these two runs leave

> **Exploitability scales with the *amplitude* of a policy's positional response,
> not with what that response is computed from.** An adversary exploits by leading
> a policy out of position; the further it is willing to be led, the more it
> loses. Capability saturates in that amplitude long before the cost does.

| supported by | |
|---|---|
| loop vs no loop | geodesic **6.39** / amp-0 **7.94** against any loop **~13.3**, disjoint |
| target is irrelevant | clearance **13.37** vs capacity **13.24**, overlapping |
| amplitude is the variable | dose–response with disjoint endpoints, 7.94 → 13.24 |
| and it is mis-set | amp 100 beats amp 200 at J3B on **5/5** paired seeds |

⚠️ **What it is not.** This is measured on **one scripted family** in **one
environment**. `results/obs_mask_gate.md` established that the *learned* policies
here cannot test it — their loop is worth 1.51 pp and they sit on the capability
threshold, so their gap is damage rather than response. ⛔ The claim is about
policies with enough capability to *have* a response worth exploiting, and this
project has exactly one such family.

⚠️ **And amp 100 is an improved heuristic, not a thesis deliverable.** It is
reported because it *prices the finding* — the mis-set amplitude is the evidence
that capability and exploitability separate — not because tuning B0 is the work.
