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
