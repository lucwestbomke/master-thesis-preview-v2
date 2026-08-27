# Inherited documentation — read-only history

These files are the predecessor project's documentation, copied **verbatim** and
frozen at the seeding commit (2026-08-27). They are kept for provenance: when a
constant, a decision or a measurement needs to be traced, this is where the
reasoning lives.

⚠️ **Do not treat them as live.** They describe a project with a different
research question, a five-rung fidelity ladder, an acceleration action space and
an skrl-based trainer. Cross-references between them are intact; cross-references
*out* of them may point at files this repository does not have.

**Read [`../INHERITED.md`](../INHERITED.md) first** — it consolidates every
measured number that still carries, with a pointer to the file it came from. Come
here only when you need the full argument behind one of them.

| file | what it still answers |
|---|---|
| `DECISIONS.md` | every direction that was proposed and then killed on evidence. **Read before proposing anything** |
| `PHYSICS.md` | channel, routing, energy and scenario parameters — the methodology chapter's source |
| `REWARD.md` | the reward's structure, the weight-setting method, and the 2026-08-27 `Φ` audit |
| `BLOCK_B.md` | what is inside `data/frankfurt_box.npz`, and how the box and the routes were chosen |
| `BLOCK_C.md` | the occlusion kernel: oriented boxes, the slab method, the 2.5D convention |
| `BLOCK_E.md` | why the rate target is 15 Mbps, and what moved when it was raised from 5 |
| `BLOCK_F.md` | the fidelity ladder's definitions and `R` = 524 m |
| `BLOCK_G.md` | the record of six pre-declared interventions and why each failed |
| `MODELS.md` | actor and critic architecture rationale |
| `ENVIRONMENT.md` | episodes, the cue, the curriculum, the observation layout |
| `NEGATIVE_RESULTS.md` | the three transmit-power nulls in full |

Superseded plans — `ROADMAP.md`, `THESIS_PLAN.md`, `BLOCK_D.md`,
`BLOCK_G_PLAN.md` — were **not** carried over. Keeping a superseded plan is how a
repository stops being legible; [`../../PLAN.md`](../../PLAN.md) replaces them.
