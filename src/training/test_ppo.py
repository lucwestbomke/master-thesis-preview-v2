"""What has to be true of the trainer before any number it produces is read.

Each test here pins one of the failures `docs/REDUCTION.md` task 5 records. They
are ordered by how silent the failure is, worst first: the GAE episode mask cost
a week and was blamed on a GRU, and the truncation bootstrap is invisible at
`gamma = 0.997` on returns of order 300.
"""

from __future__ import annotations

import torch

from ..models import SwarmActor, SwarmCritic
from . import ppo as ppo_module
from .ppo import PPOConfig, PPOTrainer, RunningScalar, _grad_norm, compute_gae
from .probe import HORIZON, PAID_FROM, BeaconEnv, ProbeConfig, probe_diagnostics

# --------------------------------------------------------------------------- #
# GAE -- hand-computed, because this is where the week went
# --------------------------------------------------------------------------- #


def _reference_gae(rewards, values, terminated, truncated, last_values, gamma, lam):
    """The recursion written out in Python, one step at a time. Deliberately not
    vectorised: it is the thing `compute_gae` is checked against."""
    t = len(rewards)
    advantages = [0.0] * t
    carry = 0.0
    for i in reversed(range(t)):
        nxt = values[i + 1] if i < t - 1 else last_values
        not_done = 0.0 if (terminated[i] or truncated[i]) else 1.0
        carry = rewards[i] - values[i] + gamma * not_done * (nxt + lam * carry)
        advantages[i] = carry
    return [a + v for a, v in zip(advantages, values, strict=True)], advantages


def _call(rewards, values, terminated, truncated, last_values, gamma=0.9, lam=0.8):
    def to(x, d=torch.float32):
        return torch.tensor(x, dtype=d).unsqueeze(-1)

    ret, adv = compute_gae(
        to(rewards),
        to(values),
        to(terminated, torch.bool),
        to(truncated, torch.bool),
        torch.tensor([last_values]),
        gamma,
        lam,
    )
    return ret.squeeze(-1).tolist(), adv.squeeze(-1).tolist()


def test_gae_matches_a_hand_written_recursion_with_no_episode_boundary():
    args = ([1.0, 2.0, 3.0, 4.0], [0.5, 0.6, 0.7, 0.8], [0] * 4, [0] * 4, 0.9)
    ret, adv = _call(*args)
    want_ret, want_adv = _reference_gae(*args[:4], args[4], 0.9, 0.8)
    assert torch.allclose(torch.tensor(adv), torch.tensor(want_adv), atol=1e-6)
    assert torch.allclose(torch.tensor(ret), torch.tensor(want_ret), atol=1e-6)


def test_a_truncation_stops_the_recursion_exactly_as_a_termination_does():
    """🔒 The bug that collapsed recurrent training for a week.

    skrl's `ppo_rnn.py` carried a stale `compute_gae` that masked on `terminated`
    alone. At a truncation `not_terminated` is True, so the recursion ran
    *through* the reset and the step collected `gamma * (V_next + lam * A_next)`
    from the **next episode** -- on top of a bootstrap already folded into the
    reward. Double-counted, and the next episode's advantage propagated backwards
    at `(gamma*lam)^k`.
    """
    rewards, values = [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]
    boundary_at = 1

    trunc = [0, 0, 0, 0]
    trunc[boundary_at] = 1
    term = [0, 0, 0, 0]
    term[boundary_at] = 1

    _, adv_truncated = _call(rewards, values, [0] * 4, trunc, 0.9)
    _, adv_terminated = _call(rewards, values, term, [0] * 4, 0.9)
    assert torch.allclose(torch.tensor(adv_truncated), torch.tensor(adv_terminated), atol=1e-9)

    # And the boundary must actually stop it: the step before the boundary must
    # not see the step after it.
    _, adv_none = _call(rewards, values, [0] * 4, [0] * 4, 0.9)
    assert adv_truncated[0] != adv_none[0]
    assert adv_truncated[boundary_at] == rewards[boundary_at] - values[boundary_at]


def test_the_last_step_bootstraps_off_last_values_and_not_off_zero():
    """A rollout boundary is not an episode boundary. Cutting the value there
    would teach the critic the world ends every `rollouts` steps."""
    _, adv_a = _call([0.0], [0.0], [0], [0], 10.0)
    _, adv_b = _call([0.0], [0.0], [0], [0], 0.0)
    assert adv_a[0] == 9.0 and adv_b[0] == 0.0


def test_returns_are_advantages_plus_values():
    ret, adv = _call([1.0, 2.0], [0.3, 0.4], [0, 0], [0, 0], 0.5)
    assert abs(ret[0] - (adv[0] + 0.3)) < 1e-6
    assert abs(ret[1] - (adv[1] + 0.4)) < 1e-6


# --------------------------------------------------------------------------- #
# The truncation bootstrap, and where it reads its state from
# --------------------------------------------------------------------------- #


class _FakeEnv:
    """Two envs, one drone, truncating on a schedule this test controls.

    The critic here is a fixed linear function of the state, so `V(final_state)`
    is known in closed form and the recorded reward can be checked against it.
    """

    def __init__(self, truncate_at: int):
        from .probe import ProbeConfig

        self.cfg = ProbeConfig(num_envs=2, num_drones=1)
        self.device = torch.device("cpu")
        self.truncate_at = truncate_at
        self.t = 0

    def _obs(self, value: float):
        return {
            "flat": torch.full((2, 1, 108), value),
            "state": torch.full((2, 6), value),
        }

    def reset(self, seed=None):
        self.t = 0
        return self._obs(0.0)

    def step(self, actions):
        self.t += 1
        truncated = torch.tensor([self.t == self.truncate_at, False])
        reward = torch.ones(2, 1)
        # `final_state` is the PRE-reset state; what `step` returns is already
        # the next episode's opening, and carries a different value on purpose.
        extras = {"final_state": torch.full((2, 6), 7.0)}
        return self._obs(-3.0), reward, torch.zeros(2, dtype=torch.bool), truncated, extras


def test_truncation_bootstraps_off_the_pre_reset_state_not_off_what_step_returns():
    """🔒 With `auto_reset` the tensors `step()` returns are a FRESH episode's
    opening. Bootstrapping on them values an unrelated state, silently, at
    `gamma = 0.997` on returns of order 300."""
    torch.manual_seed(0)
    env = _FakeEnv(truncate_at=2)
    actor = SwarmActor(architecture="mlp")
    critic = SwarmCritic(state_dim=6, hidden=16)

    cfg = PPOConfig(rollouts=3, normalise_values=False, time_limit_bootstrap=True)
    trainer = PPOTrainer(env, actor, critic, cfg)
    trainer.collect()

    # The two states the env can present. `final_state` is 7.0 everywhere and
    # the post-reset observation is -3.0 everywhere, so a bootstrap off the
    # wrong one lands on a different number.
    with torch.no_grad():
        v_final = float(critic(torch.full((1, 6), 7.0)).squeeze())
        v_next_obs = float(critic(torch.full((1, 6), -3.0)).squeeze())
    assert abs(v_final - v_next_obs) > 1e-3, "the fixture must distinguish the two states"

    rewards = trainer.buf["reward"]
    truncated_step, ordinary_step = 1, 0  # env 0 truncates on the second step
    assert abs(float(rewards[ordinary_step, 0]) - 1.0) < 1e-5
    assert abs(float(rewards[truncated_step, 0]) - (1.0 + cfg.discount_factor * v_final)) < 1e-4
    # env 1 never truncates and must be untouched
    assert abs(float(rewards[truncated_step, 1]) - 1.0) < 1e-5


def test_the_bootstrap_can_be_switched_off_and_then_changes_the_reward():
    env = _FakeEnv(truncate_at=2)
    kw = {"rollouts": 3, "normalise_values": False}
    on = PPOTrainer(
        env,
        SwarmActor(architecture="mlp"),
        SwarmCritic(6),
        PPOConfig(time_limit_bootstrap=True, **kw),
    )
    off = PPOTrainer(
        env,
        SwarmActor(architecture="mlp"),
        SwarmCritic(6),
        PPOConfig(time_limit_bootstrap=False, **kw),
    )
    on.collect()
    off.collect()
    assert not torch.allclose(on.buf["reward"], off.buf["reward"])


# --------------------------------------------------------------------------- #
# One parameter-shared agent, not N agents
# --------------------------------------------------------------------------- #


def test_the_swarm_is_one_shared_agent_over_num_envs_times_n_rows():
    """🔒 Handing a framework one shared model under N agent ids builds N
    optimizers over the same parameters and runs N stale sequential updates.
    Here there is one optimizer and the drone axis is folded into the batch."""
    env = BeaconEnv(ProbeConfig(num_envs=8, num_drones=3))
    trainer = PPOTrainer(
        env, SwarmActor(architecture="gnn"), SwarmCritic(env.state_dim), PPOConfig(rollouts=4)
    )
    assert trainer.rows == 8 * 3
    assert trainer.buf["obs"].shape == (4, 24, 108)
    # 🔒 The invariant is "no PER-AGENT bookkeeping", not "one optimizer object".
    # There are two -- one per NETWORK, so the critic can have its own LR and its
    # own gradient clip -- and each holds a single parameter group covering that
    # network exactly once. N = 3 drones must not produce 3 of anything.
    assert len(trainer.actor_optimizer.param_groups) == 1
    assert len(trainer.critic_optimizer.param_groups) == 1
    actor_ids = {id(p) for p in trainer.actor_optimizer.param_groups[0]["params"]}
    critic_ids = {id(p) for p in trainer.critic_optimizer.param_groups[0]["params"]}
    assert actor_ids == {id(p) for p in trainer.actor.parameters()}
    assert critic_ids == {id(p) for p in trainer.critic.parameters()}
    assert not (actor_ids & critic_ids)

    # the critic's state is the SAME row repeated per drone -- the measured
    # property `scripts/probe_credit.py` found, kept deliberately
    flat, state = trainer._flatten(env.reset())
    assert torch.equal(state[0], state[1]) and torch.equal(state[1], state[2])
    assert not torch.equal(flat[0], flat[1])


def test_the_actor_never_sees_the_critic_state():
    """CTDE is a claim this project makes. The buffers keep them apart by
    construction: `obs` is 108-wide and `state` is not."""
    env = BeaconEnv(ProbeConfig(num_envs=4, num_drones=2))
    trainer = PPOTrainer(env, SwarmActor(), SwarmCritic(env.state_dim), PPOConfig(rollouts=2))
    assert trainer.buf["obs"].shape[-1] == 108
    assert trainer.buf["state"].shape[-1] == env.state_dim != 108


def test_it_refuses_an_env_that_cannot_supply_the_pre_reset_state():
    import dataclasses

    import pytest

    env = BeaconEnv(ProbeConfig(num_envs=4, num_drones=2))
    env.cfg = dataclasses.replace(env.cfg, training_extras=False)
    with pytest.raises(ValueError, match="training_extras"):
        PPOTrainer(env, SwarmActor(), SwarmCritic(env.state_dim), PPOConfig(rollouts=2))


# --------------------------------------------------------------------------- #
# Value normalisation
# --------------------------------------------------------------------------- #


def test_the_running_scalar_recovers_the_batch_moments_and_round_trips():
    scaler = RunningScalar("cpu")
    x = torch.randn(4096) * 37.0 + 300.0  # returns are of order 300 at gamma=0.997
    for chunk in x.chunk(8):
        scaler.update(chunk)
    assert abs(float(scaler.mean) - float(x.mean())) < 1.0
    assert abs(float(scaler.var.sqrt()) - float(x.std(unbiased=False))) < 1.0

    y = torch.linspace(200.0, 400.0, 64)
    assert torch.allclose(scaler.denormalise(scaler.normalise(y)), y, atol=1e-2)


# --------------------------------------------------------------------------- #
# The probe: a known optimum that SPANS the episode
# --------------------------------------------------------------------------- #


def test_the_probe_optimum_is_reachable_inside_the_unpaid_half():
    """If the beacon could not be reached before payment starts, the optimum
    would not be `HORIZON - PAID_FROM + 1` and the assertions below would be
    measuring something else."""
    env = BeaconEnv(ProbeConfig(num_envs=256, num_drones=2, seed=0))
    env.reset()
    steps_needed = 0
    for _ in range(HORIZON // 2):
        towards = torch.sign(env.goal - env.pos)
        _, _, _, _, extras = env.step(towards)
        if not bool(extras["on_target"].all()):
            steps_needed += 1
    assert steps_needed < HORIZON // 2, steps_needed
    assert BeaconEnv.optimal_return() == HORIZON - PAID_FROM + 1


def _probe_run(gae=None, **over) -> list[float]:
    """One short probe run. Returns mean episodic return per drone, per log line."""
    torch.manual_seed(0)
    env = BeaconEnv(ProbeConfig(num_envs=64, num_drones=2, seed=0))
    actor = SwarmActor(architecture="mlp", hidden=64)
    critic = SwarmCritic(env.state_dim, hidden=64)
    cfg = PPOConfig(rollouts=HORIZON, learning_epochs=4, mini_batches=4, learning_rate=3e-3, **over)
    trainer = PPOTrainer(env, actor, critic, cfg, seed=0, diagnostics=probe_diagnostics)
    original = ppo_module.compute_gae
    if gae is not None:
        ppo_module.compute_gae = gae
    try:
        history = trainer.train(64 * HORIZON * 120, log_lines=6)
    finally:
        ppo_module.compute_gae = original
    return [h["reward"] * HORIZON for h in history]


def test_ppo_learns_the_episode_spanning_probe():
    """⚠️ The test the predecessor's probe should have been.

    Reward is paid only in the second half, so a step-0 action is worth
    something only through rewards 32-64 steps later. The probe it replaces used
    a pure per-step action cost, which has almost no cross-episode structure --
    it cleared the plumbing while the GAE boundary bug below was live and hid it
    for a week.
    """
    curve = _probe_run()
    optimal = BeaconEnv.optimal_return()
    assert curve[-1] > 0.9 * optimal, f"reached {curve[-1]:.1f} of {optimal:.1f}; {curve}"


def _stale_mask_gae(rewards, values, terminated, truncated, last_values, gamma, lam):
    """skrl `ppo_rnn.py`'s mask, verbatim in effect: `terminated` alone."""
    advantages = torch.zeros_like(rewards)
    not_done = (~terminated).to(rewards.dtype)
    carry = torch.zeros_like(last_values)
    for t in range(rewards.shape[0] - 1, -1, -1):
        nxt = values[t + 1] if t < rewards.shape[0] - 1 else last_values
        carry = rewards[t] - values[t] + gamma * not_done[t] * (nxt + lam * carry)
        advantages[t] = carry
    return advantages + values, advantages


def test_the_probe_would_have_caught_the_gae_boundary_bug():
    """🔒 A probe that cannot fail is not a probe.

    This is the whole reason `probe.py` exists: the predecessor's probe passed
    while `compute_gae` masked on `terminated` alone and recursed straight
    through every reset. Run the same probe under that mask and it must fail --
    and fail with the historical signature, improving and then *degrading* for
    the rest of training.
    """
    broken = _probe_run(gae=_stale_mask_gae)
    optimal = BeaconEnv.optimal_return()
    assert broken[-1] < 0.6 * optimal, f"the broken mask reached {broken[-1]:.1f}; {broken}"
    assert broken[-1] < 0.7 * max(broken), (
        f"expected the historical signature -- improve, then degrade -- got {broken}"
    )


# --------------------------------------------------------------------------- #
# The value clip -- the trap that cost a collapsed seed in five
# --------------------------------------------------------------------------- #


def test_the_value_clip_reference_equals_the_prediction_before_any_gradient_step():
    """☠️ The regression this file exists for most.

    A value clip is only safe while its reference is *the critic's own current
    output*. At epoch 0, before any optimizer step, `predicted` must equal the
    stored `values` exactly -- then the clip cannot be saturated and the value
    loss always has gradient. Once the reference drifts, a clip saturates on
    every row and the critic's gradient can go **structurally zero**, and it
    cannot recover because it is clipped against a value it cannot move toward.

    📏 Measured on CUDA, seed 0: `grad_norm_critic` 0.195 -> 0.005 -> **0.000**
    at progress 0.20, one boundary past the curriculum's 0.15, and zero for the
    remaining 80 % of the run. `mission_capable` decayed 50.6 % -> 2.6 % and the
    swarm spent 90 % of steps against the map boundary.

    ⚠️ The `max` form does NOT by itself guarantee a gradient -- if the clipped
    loss is the larger of the two it is selected and it is constant in
    `predicted`. It bounds the damage; **the reference being correct is the
    actual fix.** That is what this test pins.
    """
    env = BeaconEnv(ProbeConfig(num_envs=8, num_drones=2))
    critic = SwarmCritic(env.state_dim, hidden=16)
    trainer = PPOTrainer(
        env, SwarmActor(architecture="mlp"), critic, PPOConfig(rollouts=2, value_clip=0.2)
    )
    trainer.collect()

    # Whatever the scaler has done in between, the stored reference must still be
    # what the critic outputs right now.
    states = trainer.buf["state"].reshape(-1, trainer.buf["state"].shape[-1])
    with torch.no_grad():
        predicted = critic(states).squeeze(-1)
    stored = trainer.buf["value_norm"].reshape(-1)
    assert torch.allclose(predicted, stored, atol=1e-5), (
        "the value-clip reference is not the critic's own output -- the clip can "
        "saturate on every row and kill the critic's gradient permanently"
    )

    # And with a correct reference the value loss does produce gradient.
    trainer.update(torch.zeros(trainer.rows, device=trainer.device))
    assert float(_grad_norm(critic.parameters())) > 0.0


def test_the_stored_value_reference_is_the_critics_own_output_not_a_round_trip():
    """🔒 `normalise_new(denormalise_old(x)) != x` once the scaler's statistics
    have moved, and they move most at a curriculum boundary. Feeding that stale
    reference to the value clip is what set the trap above."""
    env = BeaconEnv(ProbeConfig(num_envs=8, num_drones=2))
    critic = SwarmCritic(env.state_dim, hidden=16)
    trainer = PPOTrainer(env, SwarmActor(architecture="mlp"), critic, PPOConfig(rollouts=2))

    _flat, state = trainer._flatten(env.reset())
    raw, normalised = trainer._values_both(state)
    with torch.no_grad():
        assert torch.allclose(normalised, critic(state).squeeze(-1), atol=1e-6)

    # Move the scaler, exactly as a curriculum boundary does, and the raw value
    # no longer round-trips -- while the network's own output is unaffected.
    trainer.scaler.update(torch.full((4096,), 300.0))
    trainer.scaler.update(torch.full((4096,), -50.0))
    assert not torch.allclose(trainer.scaler.normalise(raw), normalised, atol=1e-3)
