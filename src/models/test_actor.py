"""What keeps RQ2's architecture comparison honest.

Each test here pins one of `docs/MODELS.md`'s rules. They are cheap and they all
guard silent failures: an architecture that is secretly another architecture, a
comparison that is capacity-vs-capacity, or a rung that cannot be evaluated
off-`N` at all.
"""

from __future__ import annotations

import pytest
import torch

from ..env.core import ACTION_DIM, EDGE_DIM, EGO_DIM, FLAT_DIM, N_MAX, NEIGHBOUR_DIM
from .actor import ARCHITECTURES, RelationalTrunk, SwarmActor, build_trunk, parameter_count
from .critic import SwarmCritic


def flat_batch(rows: int = 16, num_drones: int = 5, seed: int = 0) -> torch.Tensor:
    """A `(rows, 108)` observation with `num_drones - 1` valid neighbour slots."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, FLAT_DIM, generator=g)
    k = N_MAX - 1
    real = num_drones - 1
    valid = torch.zeros(rows, k)
    valid[:, :real] = 1.0
    x[..., FLAT_DIM - k :] = valid
    # padded slots carry zeros, exactly as `_pack` writes them
    nb = x[..., EGO_DIM : EGO_DIM + k * NEIGHBOUR_DIM].unflatten(-1, (k, NEIGHBOUR_DIM))
    eg = x[..., EGO_DIM + k * NEIGHBOUR_DIM : FLAT_DIM - k].unflatten(-1, (k, EDGE_DIM))
    nb[:, real:] = 0.0
    eg[:, real:] = 0.0
    return x


def permute_neighbours(flat: torch.Tensor, order: list[int]) -> torch.Tensor:
    k = N_MAX - 1
    a, b = EGO_DIM, EGO_DIM + k * NEIGHBOUR_DIM
    c = b + k * EDGE_DIM
    out = flat.clone()
    idx = torch.tensor(order)
    out[..., a:b] = flat[..., a:b].unflatten(-1, (k, NEIGHBOUR_DIM))[:, idx].flatten(-2)
    out[..., b:c] = flat[..., b:c].unflatten(-1, (k, EDGE_DIM))[:, idx].flatten(-2)
    out[..., c:] = flat[..., c:][:, idx]
    return out


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_accepts_every_swarm_size(architecture):
    """RQ2 trains at N=5 and evaluates zero-shot at N in {3,5,8}. A rung that
    cannot take the off-N observation cannot be in the comparison at all."""
    actor = SwarmActor(architecture=architecture)
    for n in (3, 5, 8):
        mean, log_std = actor(flat_batch(num_drones=n))
        assert mean.shape == (16, ACTION_DIM)
        assert torch.isfinite(mean).all()
        assert log_std.shape == (ACTION_DIM,)


def test_parameter_counts_are_within_twenty_percent():
    """docs/MODELS.md rule 3. Otherwise the comparison is capacity-vs-capacity
    and the result says nothing about relational structure."""
    counts = {a: parameter_count(build_trunk(a)) for a in ARCHITECTURES}
    spread = max(counts.values()) / min(counts.values())
    assert spread <= 1.2, counts


def test_deepsets_is_the_gnn_with_the_edge_features_zeroed():
    """The ablation has to be exact: same class, same code path, same parameter
    count, one input masked. Two differently-named layers would always invite
    'maybe the GNN layer is just better'."""
    gnn = build_trunk("gnn")
    deepsets = build_trunk("deepsets")
    assert isinstance(gnn, RelationalTrunk) and isinstance(deepsets, RelationalTrunk)
    assert parameter_count(gnn) == parameter_count(deepsets)
    assert gnn.use_edges and not deepsets.use_edges

    deepsets.load_state_dict(gnn.state_dict())
    x = flat_batch()
    k = N_MAX - 1
    b = EGO_DIM + k * NEIGHBOUR_DIM
    zeroed = x.clone()
    zeroed[..., b : b + k * EDGE_DIM] = 0.0

    assert torch.allclose(deepsets(x), gnn(zeroed), atol=1e-6)
    assert not torch.allclose(gnn(x), gnn(zeroed), atol=1e-4), "edges must actually matter"


def test_the_relational_rungs_are_permutation_invariant_and_the_mlp_is_not():
    """The first contrast of the ladder, MLP -> DeepSets, *is* this property."""
    x = flat_batch(num_drones=5)
    order = [3, 0, 2, 1, 4, 5, 6]  # the four valid slots shuffled among themselves
    y = permute_neighbours(x, order)

    for architecture in ("deepsets", "gnn"):
        trunk = build_trunk(architecture)
        assert torch.allclose(trunk(x), trunk(y), atol=1e-6), architecture

    mlp = build_trunk("mlp")
    assert not torch.allclose(mlp(x), mlp(y), atol=1e-4)


def test_padded_neighbour_slots_do_not_reach_the_pooling():
    """Off-N transfer is only meaningful if the padding is inert. A padded slot
    that leaked into the mean would make N=3 a different task, not a smaller
    one."""
    for architecture in ("deepsets", "gnn"):
        trunk = build_trunk(architecture)
        x = flat_batch(num_drones=3)
        polluted = x.clone()
        k = N_MAX - 1
        a, b = EGO_DIM, EGO_DIM + k * NEIGHBOUR_DIM
        nb = polluted[..., a:b].unflatten(-1, (k, NEIGHBOUR_DIM))
        nb[:, 2:] = 7.0  # garbage in every invalid slot
        polluted[..., a:b] = nb.flatten(-2)
        assert torch.allclose(trunk(x), trunk(polluted), atol=1e-6), architecture


def test_the_actor_never_reads_the_critic_state():
    """CTDE is a claim this project makes, and a wrapper that quietly handed the
    actor the global state would invalidate it silently."""
    import inspect

    actor = SwarmActor()
    # Structural, not behavioural: `forward` takes ONE tensor and it is the
    # agent-local observation. There is no argument a global state could enter
    # through, so centralized execution is not reachable by accident.
    assert list(inspect.signature(SwarmActor.forward).parameters) == ["self", "flat"]
    mean, _ = actor(flat_batch())
    again, _ = actor(flat_batch())
    assert torch.equal(mean, again)


def test_actions_are_not_clipped_inside_the_distribution():
    """⚠️ Regression pin, and the failure it guards is silent and severe.

    skrl's `clip_actions=True` clamped the sample to the action space and then
    evaluated its log-probability under the *unclamped* Normal, so every tail
    draw was recorded as if it landed exactly on the boundary. Measured on this
    task: the action standard deviation rises monotonically with no entropy
    bonus anywhere in the config, actions saturate at the corners, and
    mission-capable falls from ~30 % to ~5 % under a reward whose only term is
    mission-capable. `core._advance_drones` clamps actions itself, so the bound
    is enforced either way and only the density changes.
    """
    torch.manual_seed(0)
    actor = SwarmActor()
    flat = flat_batch(rows=4096, seed=1)

    actions, log_prob, _ = actor.act(flat)
    assert (actions.abs() > 1.0).any(), "an unclipped Gaussian must leave the box sometimes"

    # The recorded log-probability must be the one of the action returned, or
    # the PPO ratio is computed against a density the sample did not come from.
    replayed, _ = actor.evaluate(flat, actions)
    assert torch.allclose(log_prob, replayed, atol=1e-5)


def test_the_log_std_floor_bounds_the_policy_class():
    """📏 With `entropy_loss_scale = 0` the deviation shrinks monotonically --
    0.061 by 20 M steps, at which point the policy is deterministic and has
    stopped exploring. The floor bounds the policy CLASS instead of adding a
    term to the objective, which is why it is preferred to an entropy bonus
    (0.01 drove the deviation UP, to 1.11)."""
    actor = SwarmActor(initial_log_std=-9.0, min_log_std=-1.0)
    _, log_std = actor(flat_batch(rows=4))
    assert torch.allclose(log_std, torch.full((ACTION_DIM,), -1.0))
    with torch.no_grad():
        stddev = float(actor.distribution(flat_batch(rows=4)).stddev.min())
    assert stddev == pytest.approx(float(torch.tensor(-1.0).exp()))


def test_the_mean_is_bounded_but_the_sample_is_not():
    """The mean is `tanh`-bounded so the deterministic policy the evaluator runs
    is always inside the action box; the sample is not, so the density is
    honest. Both properties are load-bearing and they are not the same one."""
    actor = SwarmActor(initial_log_std=1.0)
    mean, _ = actor(flat_batch(rows=512, seed=3))
    assert (mean.abs() <= 1.0).all()
    actions, _, returned_mean = actor.act(flat_batch(rows=512, seed=3))
    assert torch.equal(mean, returned_mean)
    assert (actions.abs() > 1.0).any()


def test_the_critic_is_one_design_with_no_architecture_argument():
    """docs/inherited/MODELS.md: if both the actor and the critic vary, RQ2 is
    confounded."""
    import inspect

    assert "architecture" not in inspect.signature(SwarmCritic.__init__).parameters
    critic = SwarmCritic(state_dim=54)
    assert critic(torch.randn(16, 54)).shape == (16, 1)


def test_the_models_are_plain_modules_with_no_framework_base_class():
    """🔒 `docs/REDUCTION.md` task 5. skrl produced four silent bugs in this
    project, three of them reachable only through its `Model` / mixin
    inheritance. A base class creeping back in is how they return."""
    for cls in (SwarmActor, SwarmCritic):
        bases = {b.__module__.split(".")[0] for b in cls.__mro__} - {"builtins"}
        assert bases <= {"torch", "src", "models"}, (cls, bases)
    assert "skrl" not in {m.split(".")[0] for m in list(__import__("sys").modules)}


# --------------------------------------------------------------------------- #
# The optimisation-recipe knobs added 2026-09-04. Every one ships OFF, and the
# tests here are mostly about that: a default build must be the network every
# number in `results/` was measured on.
# --------------------------------------------------------------------------- #


def test_every_new_knob_ships_off_so_the_default_build_is_the_inherited_one():
    """⛔ `tanh_mean`, `orthogonal_init` and `layer_norm` all change what the
    network computes. If any defaulted ON, every inherited number would silently
    belong to a different function."""
    actor = SwarmActor()
    assert actor.tanh_mean is True
    assert actor.layer_norm is False
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in actor.modules())

    torch.manual_seed(0)
    plain = SwarmActor(architecture="gnn")
    torch.manual_seed(0)
    explicit = SwarmActor(
        architecture="gnn", tanh_mean=True, orthogonal_init=False, layer_norm=False
    )
    for a, b in zip(plain.parameters(), explicit.parameters(), strict=True):
        assert torch.equal(a, b)


def test_the_unsquashed_mean_can_reach_the_corners_the_squashed_one_cannot():
    """📏 B0 saturates at least one action axis on 32.6 % of steps. `tanh` gets
    there only asymptotically, so the squashed mean must push `|head|` toward
    infinity to imitate it -- the obstacle `scripts/bc_init.py` clips around."""
    flat = flat_batch()
    squashed, raw = SwarmActor(tanh_mean=True), SwarmActor(tanh_mean=False)
    raw.load_state_dict(squashed.state_dict())  # same weights, one flag apart

    with torch.no_grad():
        # Scale the head so its largest raw output is exactly 3.0 -- a realistic
        # magnitude, and deliberately NOT one that saturates `tanh` to float32
        # 1.0, which would make the comparison vacuous rather than informative.
        scale = 3.0 / raw(flat)[0].abs().max()
        for actor in (squashed, raw):
            actor.head.weight.mul_(scale)
            actor.head.bias.mul_(scale)
    m_squashed, m_raw = squashed(flat)[0], raw(flat)[0]
    assert m_squashed.abs().max() < 1.0
    assert m_raw.abs().max() > 1.0
    assert torch.allclose(m_squashed, torch.tanh(m_raw))

    # 🔍 The cost, stated as the arithmetic that drives it: `atanh` is what the
    # head has to reach for a given mean, and it diverges. B0's saturated
    # actions are at 1.0 exactly, so a squashed policy can only chase them.
    head_needed = torch.atanh(torch.tensor([0.9, 0.99, 0.999, 0.9999]))
    assert head_needed.tolist() == pytest.approx([1.472, 2.646, 3.800, 4.952], abs=1e-3)
    # ⚠️ The BOUND is unchanged either way -- `core._advance_drones` clamps -- so
    # only the density the PPO ratio uses moves.


def test_the_log_std_floor_may_be_set_per_action_dimension():
    """📏 The z axis is nearly dead (B0: mean |a_z| 0.006, std 0.053) because
    altitude has a constant optimum at the derived ceiling. A scalar floor
    spends a third of the exploration budget there; a vector floor does not."""
    actor = SwarmActor(initial_log_std=[-0.5, -0.5, -3.0], min_log_std=[-1.6, -1.6, -5.0])
    _, log_std = actor(flat_batch())
    assert log_std.shape == (3,)
    assert torch.allclose(log_std, torch.tensor([-0.5, -0.5, -3.0]))
    # a scalar still broadcasts, and so does a one-element sequence
    assert torch.allclose(SwarmActor(min_log_std=-1.6).min_log_std, torch.full((3,), -1.6))
    assert torch.allclose(SwarmActor(min_log_std=[-1.6]).min_log_std, torch.full((3,), -1.6))
    with pytest.raises(ValueError, match="min_log_std"):
        SwarmActor(min_log_std=[-1.0, -2.0])


def test_the_per_dimension_floor_binds_dimension_by_dimension():
    """A floor that only ever applied the tightest of the three would silently
    make the vector form useless."""
    actor = SwarmActor(initial_log_std=-9.0, min_log_std=[-1.0, -2.0, -8.0])
    _, log_std = actor(flat_batch())
    assert torch.allclose(log_std, torch.tensor([-1.0, -2.0, -8.0]))


def test_orthogonal_init_gives_the_policy_head_a_small_gain():
    """The head is initialised separately and much smaller than the trunk, so the
    initial mean sits near zero rather than committing to a direction before any
    data has arrived."""
    actor = SwarmActor(architecture="gnn", orthogonal_init=True, head_gain=0.01)
    assert actor.head.weight.std() < 0.01
    assert torch.equal(actor.head.bias, torch.zeros_like(actor.head.bias))
    # The trunk is orthogonal at gain sqrt(2). ⚠️ A `(out, in)` weight with
    # `out > in` can only be orthonormal along the SMALLER axis -- `W @ W.T` is
    # rank-deficient there -- so the Gram matrix is taken over `in`.
    first = next(m for m in actor.trunk.modules() if isinstance(m, torch.nn.Linear))
    w = first.weight
    gram = w.T @ w if w.shape[0] >= w.shape[1] else w @ w.T
    assert torch.allclose(gram, torch.eye(gram.shape[0]) * 2.0, atol=1e-4)  # gain^2


def test_layer_norm_is_inserted_before_each_hidden_activation_and_nowhere_else():
    """It must not land on the OUTPUT layer of a trunk MLP -- that would
    normalise away the scale the next block reads."""
    actor = SwarmActor(architecture="gnn", layer_norm=True)
    layers = list(actor.trunk.ego)
    assert isinstance(layers[-1], torch.nn.Linear), "no norm or activation after the output"
    norms = [i for i, m in enumerate(layers) if isinstance(m, torch.nn.LayerNorm)]
    assert norms and all(isinstance(layers[i + 1], torch.nn.Tanh) for i in norms)
