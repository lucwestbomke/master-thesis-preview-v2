# Inherited measurements

**Every number this project may quote that was measured before it existed**,
with where it came from and what regenerates it. Consolidated 2026-08-27 from
eleven predecessor documents, which are archived in full under
[`inherited/`](inherited/).

⚠️ **A device is part of a measurement's provenance.** `torch.Generator` streams
differ per device, so the same seed draws *different episodes* on MPS than on
CPU. The physics is identical; the sample is not. Never compare across devices.

📏 measured · 🔒 constraint · 🔧 provisional — see [`../AGENTS.md`](../AGENTS.md).

---

## Settled scenario parameters

| | Value | Basis |
|---|---|---|
| City / area | Frankfurt, **1500 m** box, centre **50.11200 N, 8.67040 E** (UTM 32N) | heterogeneous: low fabric gives a workable observation envelope, towers block air-to-air. Centre chosen by sweep — `scripts/choose_box.py` |
| Buildings | Hessen **LoD2** via INSPIRE WFS, **oriented** boxes | 100 % height coverage. Axis-aligned boxes fill 94 % of the box and were rejected |
| Target roads | all surface streets, speed capped **13.9 m/s** | the constraint is speed, not road class |
| Ptx | **30 dBm fixed** ⛔ | UAV tactical MANET radios are 0.5–2 W |
| Jammer | 30 dBm in-band, rides the target | vehicle-mounted counter-UAS barrage emitter |
| Carrier / bandwidth | 3.5 GHz / **10 MHz** | so the rate target actually binds |
| Rate target | **15 Mbps** end-to-end ⛔ | dual EO/IR feed at low latency. Raised from 5, where the link never bound |
| Flight altitude | band **40–80 m**, ceiling = nominal ⛔ | both ends *derived*, not chosen — see below |
| Drone speed | 20 m/s cruise, 25 m/s dash | **1.4×** the fastest permitted road class, **~3.4×** the route bank's realised median of 5.8 m/s |
| Episode | **600 steps × 0.4 s** = 240 s | covers the 1 → 2 → 3 hop escalation. At 120 s *no* route enters the 3-hop regime (0.0 % against 36.8 %) |
| Swarm | `N = 5` trained; 3/5/8 evaluated 🔒 | trained at one `N` only, or zero-shot transfer becomes an in-distribution test |
| Discount | **γ ≈ 0.997–0.999** | the default 0.99 has a 100-step horizon and is blind to the hard end of the episode |

### Why the altitude band is 40–80 m, and why it cannot move

📏 `scripts/measure_envelope.py`. **Floor:** below 40 m, 8–37 % of positions sit
inside a building box, and TR 36.777 stops at 22.5 m. **Ceiling:** a best-placed
*single* drone is mission-capable **3.3 %** of the time at 80 m, 23 % at 100 m and
**57 %** at 120 m — above 80 m one drone can do the mission and the scenario stops
being a swarm problem.

⛔ The ceiling is *also* the primary control on how much air-to-air occlusion
exists at all: **31.2 %** of A2A links are blocked at 80 m, 24.6 % at 120 m,
~10 % at 180 m. Raising it deletes the effect under study. B0 scores 56.6 % at
80 m and 74.5 % at 120 m while `observed` barely moves.

### Why the rate target is 15 Mbps, not 5

📏 B0 at `N = 5` on the eval split, as the requirement moves:

```
requirement    5      10     15     20     30     40   Mbps
capable      93.3%  69.6%  54.7%  44.3%  19.4%  11.2%
sensor-only ceiling 93.4 %
```

At 5 Mbps the chain's bottleneck carries a median 37.6 Mbps — **8× the bar** — so
`mission_capable` reduces to `observed`, the whole mission collapses to "put one
drone over the car", and a scripted baseline reaches 93.2 % with the metric
saturated. At 15 the binding constraint moves from the **sensor** to the **relay
chain**, which is the problem this project studies.

---

## Measured geometry

📏 `scripts/measure_sightlines.py`, on `data/frankfurt_box.npz`. Each of these
**replaced an assumption**, which is why they are worth re-quoting:

| | measured | had been assumed |
|---|---|---|
| street width, median | **21 m** | 20 m |
| canyon ratio `H_b/W`, median | **0.93** | 1.10 |
| across-street envelope at 80 m | median **43 m**, p10–p90 24–88 | a flat 36 m |
| along-street sightline | median **127 m**, p90 387 | — |

🔍 **The 830 m sensor range never binds** — 99.8 % of sightlines are shorter — so
occlusion is the constraint everywhere. That is what keeps the study about
channel physics rather than about sensor specification.

⚠️ **The 127 m along-street median is load-bearing.** B0's observer stands at
**88.8 m**, *inside* it; every learned policy stands at **184 m**, *outside* it.
That threshold is the whole `observed` gap, and it is why `Φ_standoff` centres
its logistic there rather than ramping to a distant reference.

---

## Measured channel

| | value | status |
|---|---|---|
| `R`, median link range | **524 m** | 📏 `scripts/calibrate_r.py`, cross-checked by degree matching at 418 m. ⚠️ "median link range" has two readings differing 2× — see `inherited/BLOCK_F.md` before quoting |
| A2A `blockage_db` | **20.0** | 📏 **closed by sensitivity, not by citation.** The physically-motivated value is ~40 dB, but B0's headline moves 59.7 → 59.5 % across a 20–40 dB sweep, because 83 % of occluded chain edges are air-to-ground. `scripts/verify_blockage.py` |
| TR 36.777 LoS intercept, slope, height correction | ✅ | corroborated against secondary sources |
| TR 36.777 NLoS slope | ✅ | two independent sources give `4.32 − 0.76 log10(h)` |
| TR 36.777 **NLoS intercept `32.4`** | ⛔ **NOT VERIFIED** | it also equals TR 38.901's *terrestrial* UMi **LoS** intercept — exactly the neighbouring constant a transcription slip lands on. **One human reading of the UMi-AV table closes it, and every number in this project is downstream** |
| Is there anything newer? | no | 📏 TR 38.901 adds no aerial UE heights through Rel-19 (June 2025), so TR 36.777 is still current |

---

## Measured performance — the predecessor's final position

📏 Eval split, F4, curriculum stage 4, **CUDA**, 5 seeds, one harness:

| policy | capable | observed | observer range | hops |
|---|---|---|---|---|
| random | 10.7 % [0.2] | 21.9 % | 319 m | 0.4 |
| MLP | 31.2 % [1.2] | 53.8 % | — | 1.00 |
| DeepSets | 38.1 % [1.0] | 65.3 % | — | 1.23 |
| GNN | 41.2 % [3.8] | 66.5 % | **184 m** | 1.27 |
| **B0** | **57.3 % [3.9]** | **92.8 %** | **88.8 m** | 2.1 |

🔍 **Conditioned on holding a sightline the GNN converts it exactly as well as
B0** (0.620 against 0.617). The entire 16.1 pp gap is `observed`.

⚠️ **Two of these columns do not mean what they look like:**
`hop_mean | observed` **measures geometry** — hop count follows from where the
observer stands against `R` = 524 m — and `chain_occluded` **confounds with hop
count** (`corr = 0.963`). Three interventions were judged on the first.

### Architecture, cadence and shaping

| finding | value |
|---|---|
| MLP → DeepSets | **+6.9 pp**, robust |
| DeepSets → GNN | **+0.4 pp — a null** once shaping is held fixed |
| rollout length 64 (`deep`) | wins; `wide` **quadrupled** the seed spread it was built to shrink |
| the shaping axis | **noise** — 1.2 pp spread against 6 pp of within-cell seed range, and three architectures picked three different winners |

⚠️ The long-quoted **45.1 %** headline was that noise. The same configuration at
5 seeds gives **40.7 %**.

### The fidelity ladder is non-monotone in difficulty

📏 B0 under each rung: **F1 27.9 %**, F4 **56.0 %**. F1 is *harder* than F4. F0, F2
and F3 all collapse `mission_capable` onto `observed`, because `reuse_limit = 1`
below F4. 🔍 Worth reporting: adding physical realism does not monotonically
increase task difficulty.

### Throughput — compute is not a constraint

📏 RTX 5090: compiled occlusion runs **3.17 M env-steps/s** at `num_envs = 1024`,
and occlusion is ~99 % of the step. With a learner attached, **75,252
env-steps/s**, so a 10 M-step run costs **2.2 minutes** end-to-end. GPU
utilisation was 33 % at 3 GiB of 32 GiB.

`torch.compile` is 110–150× on CUDA and free — but eager *also* clears the
throughput gate on CUDA (28.9 k env-steps/s). ⚠️ The 17.6 GB figure sometimes
quoted is memory **traffic** per call, not live allocation; peak VRAM stays under
4 GB in both paths.

---

## Measured 2026-08-27 — the `Φ` audit and the beam probe

📏 `scripts/measure_potential.py`, eval split, stage 4, F4, **MPS**, 32 envs,
seed 0, against a 12 M-step feedforward GNN.

**`Φ` is switched off where it matters.** Along the closing band — observer
250 → 60 m, ray clear, chain at 25 Mbps:

| | swing | per 8 m step | vs the energy term's 0.0544 |
|---|---|---|---|
| shipped `Φ` | **+0.320** | 0.0133 | **0.25×** |
| `PHI_V2` | +1.717 | 0.0774 | **1.42×** |

**And it is exactly constant in four drones out of five.** Moving a drone that
holds no role 8 m back toward the axis: **0.0000** at 200 m off-axis, at 500 m,
and at 800 m. Under `PHI_V2`: 0.0078 / 0.0010 / 0.0003 per step, and +0.34 to
+0.47 for the whole trip home.

**Behavioural pathologies**, per drone per step:

| | speed p50 | steps > 24 m/s | energy term | at the map wall | mean \|a_z\| |
|---|---|---|---|---|---|
| B0 | 5.81 m/s | 3.1 % | −0.1250 | 0.9 % | 0.005 |
| GNN | **24.71 m/s** | **56.7 %** | **−0.1333** | **23.1 %** | **0.821** |
| MLP | — | — | — | 15.6 % | 0.626 |
| random | 17.13 m/s | 13.9 % | −0.1158 | — | — |

☠️ **The learned policy is not collecting the energy bonus.** It flies at the dash
cap where `P/P_hover ≈ 0.99` and pays **more** than B0. 0.0544/step remains the
right *bar to size `Φ` against* — it is the largest per-step force the objective
can apply — but it is **not the mechanism** behind the 184 m stand-off.

**The adversary window.** 📏 B0 under a 25° / 12 dBi jammer beam:

| beam pointed at | capable | mean e2e |
|---|---|---|
| isotropic (the inherited jammer) | 58.6 % | 22.3 Mbps |
| the MCV | 50.5 % | 17.2 Mbps |
| **the observer** | **59.9 %** | 23.5 Mbps |
| best target each step | **47.7 %** | 16.7 Mbps |

🔍 **Jamming the observer is worse than not aiming at all.** The jammer enters
SINR at the *receiver*, and the observer is a pure source — it never listens.
And the MCV is not a degenerate target: isotropic jammer power received is
**−58 to −61 dBm** at the drones against **−91.9 dBm** at the MCV, 31 dB of
shelter, because the drones are airborne with line of sight while the MCV sits in
ground clutter.

---

## Six interventions, six nulls

All pre-declared, all at >= 5 seeds. Full entries in
[`inherited/DECISIONS.md`](inherited/DECISIONS.md). ⛔ **Do not re-propose any of
these without a new mechanism.**

| | result |
|---|---|
| recurrence (GRU actor) | −1.05 pp, tenure 36.8 against a required 95, seed IQR *widened* 4.7 → 6.9 |
| `w_hold` — the `Φ_observe` hold factor | +1.65 pp on a 6.8 IQR |
| `w_relay` — a per-drone potential on `on_path` | between-drone advantage variance **71×**, behaviour unmoved |
| agent-specific critic (Yu et al.) | actively hurt — 34.1 %, one seed collapsing to 0.3 % |
| `d_ref` 1500 → 400 | null across an 81-run sweep |
| `potential_scale` 10 → 30 | null across the same sweep |

🔍 **The audit above retro-explains the shape of all six.** Each scaled a
potential whose directional gradient in the operating regime was 0.013–0.03 per
step. The reason they failed is arithmetic, not mechanism.

---

## Three nulls on transmit power

📏 [`inherited/NEGATIVE_RESULTS.md`](inherited/NEGATIVE_RESULTS.md). ⛔ Do not
reintroduce Ptx as an action.

| framing | result | reason |
|---|---|---|
| energy saving | ~1.6 % of power draw | raising the ceiling to fix it destroys the mission instead |
| interference management | **0.0 %** against a fair baseline | one flow, routing-aware MAC, <= 3 hops ⇒ the reuse schedule never runs two transmitters at once |
| detectability / EMCON | 0.1–1.1 % | exposure saturates: the observer must sit in the threat's line of sight and is always detected |

The one untested route is **multiple concurrent flows**, which would create real
contention — future work, because it turns the project into a distributed
link-scheduling study.
