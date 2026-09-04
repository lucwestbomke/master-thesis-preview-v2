"""The three actor rungs of RQ2's ladder, on one observation contract.

    | rung     | neighbours read as        | perm-invariant | size-agnostic | edges |
    | mlp      | concatenated, max-N padded| no             | no            | no    |
    | deepsets | shared embed, then pooled | yes            | yes           | no    |
    | gnn      | same, messages see e_ij   | yes            | yes           | yes   |

MLP -> DeepSets isolates permutation invariance; DeepSets -> GNN isolates the
*relational* part, which is RQ2's actual claim (`docs/inherited/MODELS.md`).

**DeepSets is the GNN with `e_ij` zeroed** -- literally the same class, the same
code path, the same parameter count, the same optimiser, one input masked. That
is what makes the ablation exact and is why the layer is a custom MPNN
(Gilmer et al., 2017: `message(x_i, x_j, e_ij) = MLP([x_i, x_j, e_ij])`) rather
than two differently-named PyG layers, which would always invite "maybe GATv2 is
just a better layer".

⛔ `SAGEConv` is never an option here: it cannot ingest edge features at all, so
it would silently collapse the GNN rung into the DeepSets rung and RQ2 would
measure nothing. It is also the layer people reach for by default.

## One message-passing layer, not two -- and why that is not a shortcut

`docs/inherited/MODELS.md` says two message-passing layers is the ceiling the
graph justifies, reasoning from a graph of `N` drone nodes. **The actor does not
hold that graph.** Its observation is `(B, 108)`: its own ego block plus 7
neighbour slots -- a *star* centred on itself. A second layer over the true swarm
graph would give drone `i` access to `j`'s aggregate of `k`, which `i` does not
possess and could only obtain by exchanging embeddings with its neighbours. That
would hand the GNN rung strictly more information than the MLP and DeepSets
rungs get, and RQ2's comparison would confound architecture with information --
the exact confound the zeroed-`e_ij` design exists to rule out.

So: one message-passing layer over the local star, with depth living in the
message and update MLPs, where it is capacity rather than reach.

Every rung consumes `obs["flat"]` and unpacks it with `core.unpack_flat`, so the
max-N padding is identical across rungs by construction rather than by
discipline, and all three accept `N` in {3, 5, 8} without reshaping.

## Plain `nn.Module`, since `docs/REDUCTION.md` task 5

These used to inherit from skrl's `Model` / `GaussianMixin`, which is what made
the four silent bugs in that document reachable. The actor is now an ordinary
module: `forward(flat) -> (mean, log_std)`, with `act` / `evaluate` supplying the
two things PPO needs from a Gaussian policy. `src/training/ppo.py` owns the
algorithm; this file owns the architecture, and neither knows anything about the
other's internals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from ..env.core import ACTION_DIM, EDGE_DIM, EGO_DIM, FLAT_DIM, NEIGHBOUR_DIM, unpack_flat

ARCHITECTURES = ("mlp", "deepsets", "gnn")

#: Widths chosen so the three rungs land within 20 % of each other on parameter
#: count (`docs/inherited/MODELS.md` rule 3: the comparison must not be
#: capacity-vs-capacity). `test_actor.py` asserts the spread, so these cannot
#: drift apart silently. They are a starting point for the equal-budget search,
#: not findings.
DEFAULT_HIDDEN = 128
DEFAULT_MLP_HIDDEN = 232


def _mlp(sizes: list[int], layer_norm: bool = False) -> nn.Sequential:
    """Tanh MLP. `layer_norm` inserts `LayerNorm` before each hidden activation.

    ⛔ `layer_norm=False` is the default and reproduces every inherited number
    bit for bit -- it adds no parameters and no ops when off.
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _per_dim(value: float | Sequence[float], name: str) -> Tensor:
    """A scalar broadcast across `ACTION_DIM`, or a per-dimension vector as given."""
    if isinstance(value, (int, float)):
        return torch.full((ACTION_DIM,), float(value))
    values = [float(v) for v in value]
    # A one-element sequence broadcasts, so `--min-log-std -1.6` and
    # `--min-log-std -1.6 -1.6 -3.0` are both valid and mean what they look like.
    if len(values) == 1:
        return torch.full((ACTION_DIM,), values[0])
    if len(values) != ACTION_DIM:
        raise ValueError(f"{name} must be a scalar or {ACTION_DIM} values, got {values}")
    return torch.tensor(values)


def orthogonal_init_(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Orthogonal weights, zero bias -- the PPO reference initialisation.

    ⚠️ **Absent from every run in this project's history.** `_mlp` builds plain
    `nn.Linear`s, so the trunk ships with PyTorch's default Kaiming-uniform
    weights, which is not what the PPO recipe assumes. It matters most early,
    which is where `runs/val-gnn-deep-s*/log.jsonl` shows the whole run happening:
    at ~5,900 Adam steps and `approx_kl` 0.002-0.004 the policy never travels far
    from wherever it was initialised.

    `gain = sqrt(2)` for hidden layers is the standard choice under `tanh`/`relu`;
    the policy head is initialised separately at a much smaller gain so the
    initial action distribution is near-uniform in the mean rather than
    committing to a direction before any data has arrived.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)


class FlatTrunk(nn.Module):
    """The MLP rung. Reads the padded 108-vector as one flat block.

    Not permutation-invariant and not size-agnostic: neighbour slot 3 is a
    different set of weights from slot 4, so relabelling the swarm changes the
    output. That is the property RQ2's first contrast measures, so it is a
    faithful implementation rather than a weak one -- and the max-N padding plus
    validity bits are what let it be *evaluated* off-N at all.
    """

    def __init__(self, hidden: int = DEFAULT_MLP_HIDDEN, frames: int = 1, layer_norm: bool = False):
        super().__init__()
        self.frames = frames
        self.net = _mlp([FLAT_DIM * frames, hidden, hidden, hidden], layer_norm)
        self.out_dim = hidden

    def forward(self, flat: Tensor) -> Tensor:
        # The MLP rung has no relational structure to preserve, so a stacked
        # observation `(..., k, FLAT_DIM)` is simply flattened back into one
        # vector. ⚠️ Without this, `nn.Linear` would silently fold `k` into the
        # BATCH dimension and multiply against the wrong width.
        return torch.tanh(self.net(flat.flatten(-2) if self.frames > 1 else flat))


def unpack_stacked(flat: Tensor, frames: int) -> dict[str, Tensor]:
    """`unpack_flat` over `(..., k, FLAT_DIM)`, concatenating history per ENTITY.

    🔒 The point is that a stacked observation must not be flattened into one long
    vector for the relational rungs. `ego` becomes `(..., k*EGO_DIM)` and each
    neighbour slot becomes `(..., 7, k*NEIGHBOUR_DIM)`, so drone `j`'s history
    stays attached to drone `j`. Flattening instead would destroy the permutation
    structure that `docs/MODELS.md` requires DeepSets and the GNN to have, and the
    off-N transfer columns would then be measuring its loss.

    `valid` is taken from the NEWEST frame only: it is the max-N padding mask,
    fixed for the whole episode at a given `N`, so stacking it would add `k`
    identical copies.
    """
    parts = [unpack_flat(flat[..., i, :]) for i in range(frames)]
    return {
        "ego": torch.cat([p["ego"] for p in parts], dim=-1),
        "neighbour": torch.cat([p["neighbour"] for p in parts], dim=-1),
        "edge": torch.cat([p["edge"] for p in parts], dim=-1),
        "valid": parts[-1]["valid"],
    }


class RelationalTrunk(nn.Module):
    """The DeepSets and GNN rungs. One class, one flag, one masked input.

    `use_edges=False` zeroes `e_ij` on the way into the message function and
    changes nothing else -- the weights that would multiply it still exist and
    are still trained, so the two rungs have identical parameter counts by
    construction. Pooling is a **mean over valid neighbours**, not a sum: a sum
    scales with `N` and would make the off-N transfer columns measure a
    normalisation artefact instead of the architecture.
    """

    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN,
        use_edges: bool = True,
        frames: int = 1,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.use_edges = use_edges
        self.frames = frames
        self.ego = _mlp([EGO_DIM * frames, hidden, hidden], layer_norm)
        self.neighbour = _mlp([NEIGHBOUR_DIM * frames, hidden, hidden], layer_norm)
        self.message = _mlp([2 * hidden + EDGE_DIM * frames, hidden, hidden], layer_norm)
        self.update = _mlp([2 * hidden, hidden, hidden], layer_norm)
        self.out_dim = hidden

    def forward(self, flat: Tensor) -> Tensor:
        parts = unpack_flat(flat) if self.frames == 1 else unpack_stacked(flat, self.frames)
        valid = parts["valid"].unsqueeze(-1)  # (..., 7, 1)

        h_i = torch.tanh(self.ego(parts["ego"]))  # (..., H)
        h_j = torch.tanh(self.neighbour(parts["neighbour"]))  # (..., 7, H)

        edge = parts["edge"]
        if not self.use_edges:
            edge = torch.zeros_like(edge)

        pair = torch.cat([h_i.unsqueeze(-2).expand_as(h_j), h_j, edge], dim=-1)
        msg = torch.tanh(self.message(pair)) * valid
        # Mean over *valid* neighbours; `clamp_min(1)` keeps N=1 finite rather
        # than dividing by zero, and the padded slots contribute nothing.
        agg = msg.sum(dim=-2) / valid.sum(dim=-2).clamp_min(1.0)

        return torch.tanh(self.update(torch.cat([h_i, agg], dim=-1)))


def build_trunk(
    architecture: str,
    hidden: int | None = None,
    frames: int = 1,
    layer_norm: bool = False,
) -> nn.Module:
    if architecture == "mlp":
        return FlatTrunk(hidden or DEFAULT_MLP_HIDDEN, frames=frames, layer_norm=layer_norm)
    if architecture == "deepsets":
        return RelationalTrunk(
            hidden or DEFAULT_HIDDEN, use_edges=False, frames=frames, layer_norm=layer_norm
        )
    if architecture == "gnn":
        return RelationalTrunk(
            hidden or DEFAULT_HIDDEN, use_edges=True, frames=frames, layer_norm=layer_norm
        )
    raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")


class SwarmActor(nn.Module):
    """Agent-local stochastic actor. Reads the flat observation, never the state.

    Reading the critic's global state here would quietly turn CTDE into
    centralized execution and invalidate the whole framing
    (`docs/inherited/ENVIRONMENT.md` -> Observations), so the separation is
    structural: this class takes one tensor and it is `obs["flat"]`.

    The action is motion only -- 3-dim. ⛔ Transmit power is not an action and
    never becomes one: three framings, three nulls
    (`docs/inherited/NEGATIVE_RESULTS.md`).

    ⚠️ **The distribution is never clipped, and that is not a detail.** skrl's
    `GaussianMixin(clip_actions=True)` clamped the sampled action to the action
    space AND then evaluated its log-probability under the *unclamped* Normal.
    Every sample from the tails was recorded as if it had landed exactly on the
    boundary, piling empirical mass on ±1 that the density used in the PPO ratio
    does not account for. Measured here: the action standard deviation rose
    monotonically with no entropy bonus anywhere in the config, actions
    saturated at the corners, and mission-capable fell from ~30 % to ~5 % on a
    reward whose only term IS mission-capable.

    The standard treatment is to let the distribution stay unbounded and clip in
    the environment, so the log-probability belongs to the action that was
    actually sampled. `core._advance_drones` already opens with
    `actions.clamp(-1.0, 1.0)`, so the bound is enforced either way and only the
    density changes. There is deliberately no flag for the other behaviour.
    """

    def __init__(
        self,
        architecture: str = "mlp",
        hidden: int | None = None,
        initial_log_std: float | Sequence[float] = -0.5,
        min_log_std: float | Sequence[float] = -20.0,
        max_log_std: float = 2.0,
        obs_history: int = 1,
        tanh_mean: bool = True,
        orthogonal_init: bool = False,
        layer_norm: bool = False,
        head_gain: float = 0.01,
    ):
        super().__init__()
        self.architecture = architecture
        self.obs_history = obs_history
        self.layer_norm = bool(layer_norm)
        self.trunk = build_trunk(architecture, hidden, frames=obs_history, layer_norm=layer_norm)
        self.head = nn.Linear(self.trunk.out_dim, ACTION_DIM)
        # 🔒 **Whether the mean is squashed.** `True` reproduces every inherited
        # number; `False` returns the head's raw output and lets
        # `core._advance_drones`'s existing `actions.clamp(-1, 1)` be the only
        # bound. ⚠️ It is not cosmetic: `tanh` reaches +-1 only asymptotically,
        # and 📏 **B0 saturates at least one action axis on 32.6 % of steps**
        # (x 15.0 %, y 19.8 %, stage 4 / F4, 64 envs x 400 steps). A tanh mean
        # therefore cannot express a third of the expert's actions without
        # driving `|head|` toward infinity -- which is exactly the failure
        # `scripts/bc_init.py` had to clip its regression targets to +-0.995 to
        # avoid. The distribution stays unclipped either way, so the
        # log-probability still belongs to the action that was sampled.
        self.tanh_mean = bool(tanh_mean)

        # One state-independent log-std per action dimension, the PPO standard.
        # -0.5 (sigma ~ 0.61) against a tanh-bounded mean in [-1, 1]: exploratory
        # without spending most of its mass on the clip boundary.
        #
        # ⚠️ **Scalars broadcast, sequences do not, and the difference is worth a
        # flag.** 📏 The z axis is very nearly dead: B0's mean |a_z| is **0.006**
        # with a standard deviation of **0.053**, and it saturates 0.1 % of the
        # time -- against 0.46 / 0.52 mean |a| on x / y. Altitude has a constant
        # optimum here (`ALT_MAX_M`, and the ceiling is *derived* -- see
        # `core.ALT_MAX_M`), so a scalar sigma spends a third of the exploration
        # budget on a dimension with nothing to explore, and pays for it twice:
        # `energy` charges climb power and leaving the ceiling costs sightlines.
        # A per-dimension floor is the cheap fix and needs the vector form.
        self.log_std = nn.Parameter(_per_dim(initial_log_std, "initial_log_std"))
        # `min_log_std` floors exploration. skrl's default (-20) is effectively
        # no floor, and with `entropy_loss_scale = 0` the standard deviation then
        # shrinks monotonically as the policy grows confident: measured on the
        # full mission it reached **0.061** by 20 M steps, at which point the
        # policy is deterministic, stops exploring, and sits in whatever local
        # behaviour it found. A floor is preferable to an entropy bonus for that
        # job because it bounds the policy CLASS instead of adding a term to the
        # objective -- an entropy bonus large enough to matter here (0.01)
        # instead drove the deviation UP, to 1.11, which is its own failure.
        #
        # ⚠️ The default is -20 because that is what the inherited 40.7 % ran
        # under. It is a knob, not a recommendation.
        # ☠️ `persistent=False`, and it is NOT a detail. A plain buffer joins
        # `state_dict()`, and every checkpoint written before this field existed
        # then fails to load with `Missing key(s) in state_dict: "min_log_std"` --
        # which is every checkpoint in `runs/`, and every loader in `scripts/`.
        # It is a configuration constant, not a learned parameter: it already
        # travels in the checkpoint blob under its own key and is passed back
        # through the constructor.
        self.register_buffer("min_log_std", _per_dim(min_log_std, "min_log_std"), persistent=False)
        self.max_log_std = float(max_log_std)

        if orthogonal_init:
            orthogonal_init_(self.trunk)
            # A small head gain keeps the initial mean near zero, so the policy
            # does not commit to a direction before it has seen anything.
            nn.init.orthogonal_(self.head.weight, head_gain)
            nn.init.zeros_(self.head.bias)

    def forward(self, flat: Tensor) -> tuple[Tensor, Tensor]:
        """`(mean, log_std)`. `log_std` broadcasts against `mean`.

        🔒 The public input is always **2-D in the feature dim**: `(..., FLAT_DIM)`
        with no history, `(..., k * FLAT_DIM)` with it. Callers -- the trainer's
        rollout buffer, `evaluate.py`, an ONNX export -- never handle a `k` axis.
        The unflatten to `(..., k, FLAT_DIM)` happens here, once, at the boundary.
        """
        if self.obs_history > 1:
            flat = flat.unflatten(-1, (self.obs_history, FLAT_DIM))
        raw = self.head(self.trunk(flat))
        mean = torch.tanh(raw) if self.tanh_mean else raw
        # ⚠️ Two steps rather than one `clamp`: `min_log_std` is a per-dimension
        # BUFFER and `max_log_std` a scalar, and `torch.clamp` refuses that mix.
        log_std = self.log_std.clamp(max=self.max_log_std)
        return mean, torch.maximum(log_std, self.min_log_std)

    def distribution(self, flat: Tensor) -> Normal:
        mean, log_std = self(flat)
        return Normal(mean, log_std.exp().expand_as(mean))

    def act(self, flat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample. Returns `(action, log_prob, mean)`, log-prob summed over dims.

        The action is returned unclipped and its log-probability is the
        log-probability of *that* value -- see the class docstring.
        """
        dist = self.distribution(flat)
        action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.mean

    def evaluate(self, flat: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """`(log_prob, entropy)` of `actions` under the current parameters."""
        dist = self.distribution(flat)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def gaussian_entropy(log_std: Tensor) -> Tensor:
    """Closed-form differential entropy of a diagonal Gaussian. Diagnostic."""
    return (log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum()
