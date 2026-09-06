"""Phase 0 of `docs/CAPABILITY_BRIEF.md`: run `src/training/ppo.py` **unchanged**
on tasks whose answer is known from outside this project.

    uv run python scripts/bench_trainer.py --task lqr      --seeds 0 1 2 3 4
    uv run python scripts/bench_trainer.py --task pendulum --seeds 0 1 2 3 4
    uv run python scripts/bench_trainer.py --task mountaincar --seeds 0 1 2 3 4

⚠️ **Why this exists.** Every capability null in `results/` is read as a statement
about the task. That reading is only available once the trainer is known to work,
and this project has never checked it against anything but itself:
`results/trainer_validation.md` reproduced an *inherited* number, and
`src/training/probe.py` is a probe **this repository wrote**. Both can be cleared
by a trainer that is consistently wrong in the same way. A published PPO baseline
cannot.

## What is under test, and what is not

🔒 **`src/training/ppo.py` is imported and not modified.** `PPOTrainer`,
`compute_gae`, `RunningScalar`, the truncation bootstrap, the value clip, the
advantage normalisation and the two optimizers are the code the mission runs.

⛔ **The actor is NOT the mission actor**, and cannot be: `SwarmActor`'s trunk
unpacks a 108-wide swarm observation through `core.unpack_flat`, and Pendulum has
three numbers. `BenchActor` therefore replaces the *trunk* only -- it **inherits
`forward` / `distribution` / `act` / `evaluate` from `SwarmActor`**, so the
distribution semantics under test (unclipped Normal, state-independent
per-dimension `log_std`, the `tanh_mean` switch, the `min_log_std` floor) are
literally the same code the mission runs. `SwarmCritic` is used as-is; it is
already generic in `state_dim`.

## The environment contract

`PPOTrainer` duck-types `BatchedSwarmEnv`, and `src/training/probe.py` already
duck-types it for the beacon probe. The two adapters here do the same:

* `cfg.num_envs`, `cfg.num_drones`, `cfg.training_extras`, `cfg.auto_reset`
* `reset() -> {"flat": (B, N, obs), "state": (B, state)}`
* `step(a: (B, N, act)) -> (obs, reward (B, N), terminated (B,), truncated (B,),
  extras)` with `extras["final_state"]` the **pre-reset** state

🔒 `num_drones = 1` throughout. The row folding is exercised by the beacon probe,
which runs `num_drones = 2`; what is exercised here is the algorithm.

⚠️ **Gymnasium's autoreset mode is load-bearing.** `SyncVectorEnv` defaults to
`AutoresetMode.NEXT_STEP`, which spends a whole step doing nothing and returns the
*final* observation on the terminating step -- the opposite of the contract
`PPOTrainer` asserts. `AutoresetMode.SAME_STEP` returns the fresh episode's
opening and puts the final observation in `info["final_obs"]`, which is exactly
`extras["final_state"]`. Getting this wrong would silently mis-bootstrap every
truncation, i.e. it would fake the bug this whole file exists to rule out.

## Actions

The mission's convention is that the policy is unbounded, `core._advance_drones`
opens with `actions.clamp(-1, 1)`, and the log-probability belongs to the
unclipped sample (`SwarmActor`'s docstring: skrl clipped the sample and evaluated
it under the unclipped Normal, and the deviation rose monotonically with no
entropy bonus anywhere). The adapters keep that convention: clamp to [-1, 1], then
affinely map onto the task's own action box. Nothing is clipped inside the
distribution.

## The reward normalisation in `mountaincar`

📏 `rl-baselines3-zoo`'s tuned entry for `MountainCarContinuous-v0` sets
`normalize: true`, i.e. SB3's `VecNormalize` over **observations and rewards**.
That is an environment wrapper, not an algorithm change, so it lives here in
`RunningNorm` and `ppo.py` is untouched. ⚠️ Episodic returns are always reported on
the **raw** reward, as `benchmark.md` reports them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import SwarmActor, SwarmCritic, parameter_count
from src.models.actor import orthogonal_init_
from src.training.ppo import PPOConfig, PPOTrainer

# --------------------------------------------------------------------------- #
# The actor: the mission's distribution over a benchmark-shaped trunk
# --------------------------------------------------------------------------- #


class BenchActor(SwarmActor):
    """`SwarmActor` with a plain MLP trunk and an arbitrary observation width.

    🔒 `nn.Module.__init__` rather than `super().__init__()`, deliberately: every
    method that defines the *policy distribution* is inherited unchanged, and only
    the field construction is replaced. So `forward`, `distribution`, `act` and
    `evaluate` here are the same bytes the mission runs, and a bug in any of them
    shows up on Pendulum.

    The default trunk is SB3's `MlpPolicy` shape -- two `tanh` layers of 64 -- so
    the comparison is not architecture-versus-architecture.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 64,
        initial_log_std: float = 0.0,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        tanh_mean: bool = False,
        orthogonal_init: bool = True,
        head_gain: float = 0.01,
        sde: bool = False,
        sde_sample_freq: int = 4,
    ):
        nn.Module.__init__(self)
        self.architecture = "bench-mlp"
        self.obs_history = 1
        self.layer_norm = False
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.trunk.out_dim = hidden  # `PPOTrainer.save` reads it
        self.head = nn.Linear(hidden, act_dim)
        # ⛔ `False` by default here, unlike the mission. A `tanh` mean cannot
        # reach the edge of the action box, and Pendulum's optimal control
        # saturates at +-2 through most of the swing-up.
        self.tanh_mean = bool(tanh_mean)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(initial_log_std)))
        self.register_buffer(
            "min_log_std", torch.full((act_dim,), float(min_log_std)), persistent=False
        )
        self.max_log_std = float(max_log_std)
        # 🔒 The same method the mission actor calls, with this task's action
        # width. gSDE is a property of the distribution, so it has to be the same
        # code here or the benchmark validates something else.
        self._init_sde(sde, sde_sample_freq, initial_log_std, act_dim)
        if orthogonal_init:
            orthogonal_init_(self.trunk, math.sqrt(2.0))
            nn.init.orthogonal_(self.head.weight, head_gain)
            nn.init.zeros_(self.head.bias)


# --------------------------------------------------------------------------- #
# Observation / reward normalisation, SB3 `VecNormalize`-shaped
# --------------------------------------------------------------------------- #


class RunningNorm:
    """Running mean/variance of a vector, updated on the batch axis.

    ⛔ Not `RunningScalar` from `ppo.py`: that one is the *value target* scaler and
    is part of the algorithm under test. This is an environment wrapper and is
    kept separate so the two cannot be confused in a diagnosis.
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, n = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m_a, m_b = self.var * self.count, batch_var * n
        self.var = (m_a + m_b + delta**2 * self.count * n / total) / total
        self.count = total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)


# --------------------------------------------------------------------------- #
# The env adapters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BenchEnvConfig:
    """The four fields `PPOTrainer.__init__` asserts on, and the two shapes it reads."""

    num_envs: int
    num_drones: int = 1
    training_extras: bool = True
    auto_reset: bool = True


class GymAdapter:
    """A `gymnasium` vector env behind the `BatchedSwarmEnv` contract."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        device: str = "cpu",
        normalise_obs: bool = False,
        normalise_reward: bool = False,
        gamma: float = 0.99,
    ):
        import gymnasium as gym
        from gymnasium.vector import AutoresetMode, SyncVectorEnv

        self.env_id = env_id
        self.cfg = BenchEnvConfig(num_envs=num_envs)
        self.device = torch.device(device)
        self.vec = SyncVectorEnv(
            [lambda: gym.make(env_id) for _ in range(num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        self.obs_dim = int(np.prod(self.vec.single_observation_space.shape))
        self.act_dim = int(np.prod(self.vec.single_action_space.shape))
        self.low = self.vec.single_action_space.low.astype(np.float32)
        self.high = self.vec.single_action_space.high.astype(np.float32)
        self.seed = seed
        self.gamma = gamma
        self.obs_norm = RunningNorm((self.obs_dim,)) if normalise_obs else None
        # SB3 normalises the reward by the standard deviation of the *discounted
        # return*, not of the reward, and does not re-centre it.
        self.ret_norm = RunningNorm((1,)) if normalise_reward else None
        self._disc = np.zeros(num_envs, dtype=np.float64)
        # Episodic bookkeeping. ⚠️ Host-side, and that is fine: this file is not
        # the mission's hot loop and `AGENTS.md`'s no-`.item()` rule is about a
        # CUDA sync in `step()`.
        self._running = np.zeros(num_envs, dtype=np.float64)
        self._last = np.zeros(num_envs, dtype=np.float64)
        self.finished: deque[float] = deque(maxlen=200)

    # -- contract ------------------------------------------------------- #

    @property
    def state_dim(self) -> int:
        return self.obs_dim

    def _pack(self, obs: np.ndarray) -> dict[str, Tensor]:
        if self.obs_norm is not None:
            obs = self.obs_norm(obs)
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return {"flat": t.unsqueeze(1), "state": t}

    def reset(self) -> dict[str, Tensor]:
        obs, _ = self.vec.reset(seed=self.seed)
        if self.obs_norm is not None:
            self.obs_norm.update(np.asarray(obs, dtype=np.float64))
        return self._pack(obs)

    def step(self, actions: Tensor):
        b = self.cfg.num_envs
        # The mission's convention: unbounded policy, env clamps. See the module
        # docstring.
        a = actions.reshape(b, self.act_dim).clamp(-1.0, 1.0).detach().cpu().numpy()
        scaled = self.low + (a + 1.0) * 0.5 * (self.high - self.low)
        obs, reward, terminated, truncated, info = self.vec.step(scaled.astype(np.float32))

        done = np.logical_or(terminated, truncated)
        self._running += reward
        if done.any():
            self._last = np.where(done, self._running, self._last)
            self.finished.extend(self._running[done].tolist())
            self._running = np.where(done, 0.0, self._running)

        # `final_obs` holds the pre-reset observation on done rows and `None`
        # elsewhere; everywhere else the current observation IS the final one.
        final = np.asarray(obs, dtype=np.float64).copy()
        if done.any():
            raw_final = info["final_obs"]
            for i in np.flatnonzero(done):
                final[i] = np.asarray(raw_final[i], dtype=np.float64)

        if self.obs_norm is not None:
            self.obs_norm.update(np.asarray(obs, dtype=np.float64))

        shaped = reward.astype(np.float64)
        if self.ret_norm is not None:
            self._disc = self._disc * self.gamma * (~done) + shaped
            self.ret_norm.update(self._disc.reshape(-1, 1))
            shaped = np.clip(shaped / np.sqrt(self.ret_norm.var[0] + 1e-8), -10.0, 10.0)

        extras = {
            "final_state": self._pack(final)["state"],
            # ⚠️ The reward `step` RETURNS may be `VecNormalize`-scaled. Episodic
            # returns must be reported raw, so the raw one rides in `extras` and
            # `evaluate` reads it from there rather than from the return value.
            "raw_reward": torch.as_tensor(
                reward.astype(np.float32), dtype=torch.float32, device=self.device
            ),
            "episodic_return": torch.as_tensor(
                float(self._last.mean()), dtype=torch.float32, device=self.device
            ),
        }
        return (
            self._pack(obs),
            torch.as_tensor(shaped, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.as_tensor(terminated, device=self.device),
            torch.as_tensor(truncated, device=self.device),
            extras,
        )

    def close(self) -> None:
        self.vec.close()


class LQREnv:
    """A batched linear-quadratic regulator whose optimum is **derived, not cited**.

    ⭐ This is the reference `AGENTS.md` prefers: *"⛔ Cite constants an AI
    produced"* is a real rule here, and a published benchmark number is a citation
    however carefully it is fetched. The optimal gain of a discounted LQR follows
    from the discrete algebraic Riccati equation, which is solved below by value
    iteration on `P` and converges to machine precision in a few hundred sweeps.
    So the bar this task is judged against is **computed in-repo, on the same
    seeds, through the same rollout loop as the policy under test.**

    The plant is a 2-D double integrator at `dt = 0.1`:

        x = (px, vx, py, vy),  u = (ax, ay)
        reward = -(x' Q x + u' R u)

    ⚠️ It **truncates and never terminates**, which is deliberate: the truncation
    bootstrap is invariant 2 of `ppo.py` and the failure it guards against is
    silent. A trainer that treats truncation as termination learns that the world
    ends at `HORIZON`, and on a task whose optimal cost-to-go is *quadratic and
    unbounded* that bias is easy to see.
    """

    DT = 0.1
    HORIZON = 100
    ACT_SCALE = 3.0  # the box the policy's [-1, 1] maps onto

    def __init__(self, num_envs: int, seed: int, device: str = "cpu", gamma: float = 0.99):
        self.cfg = BenchEnvConfig(num_envs=num_envs)
        self.device = torch.device(device)
        self.gamma = gamma
        d = self.DT
        self.A = torch.tensor(
            [[1.0, d, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, d], [0.0, 0.0, 0.0, 1.0]],
            device=self.device,
        )
        self.B = torch.tensor([[0.0, 0.0], [d, 0.0], [0.0, 0.0], [0.0, d]], device=self.device)
        self.Q = torch.eye(4, device=self.device)
        self.R = 0.1 * torch.eye(2, device=self.device)
        self.obs_dim, self.act_dim = 4, 2
        self.seed = seed
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.x = torch.zeros(num_envs, 4, device=self.device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._running = torch.zeros(num_envs, device=self.device)
        self._last = torch.zeros(num_envs, device=self.device)
        self.finished: list[float] = []

    # -- contract ------------------------------------------------------- #

    @property
    def state_dim(self) -> int:
        return 4

    def _sample(self, mask: Tensor) -> None:
        fresh = torch.rand(self.cfg.num_envs, 4, device=self.device, generator=self.gen) * 2.0 - 1.0
        self.x = torch.where(mask.unsqueeze(-1), fresh, self.x)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)

    def _observe(self) -> dict[str, Tensor]:
        return {"flat": self.x.unsqueeze(1), "state": self.x}

    def reset(self) -> dict[str, Tensor]:
        # 🔒 Re-seeds the episode stream. `evaluate` and `evaluate_lqr_optimum`
        # both open with a `reset`, so the policy and the derived optimum are
        # scored on the **same** initial states -- which is what makes the gap
        # between them a measurement rather than a comparison of two samples.
        self.gen.manual_seed(self.seed)
        self._sample(torch.ones(self.cfg.num_envs, dtype=torch.bool, device=self.device))
        return self._observe()

    def step(self, actions: Tensor):
        b = self.cfg.num_envs
        u = actions.reshape(b, 2).clamp(-1.0, 1.0) * self.ACT_SCALE
        cost = (self.x @ self.Q * self.x).sum(-1) + (u @ self.R * u).sum(-1)
        reward = -cost
        self.x = self.x @ self.A.T + u @ self.B.T
        self.t = self.t + 1

        self._running = self._running + reward
        truncated = self.t >= self.HORIZON
        terminated = torch.zeros(b, dtype=torch.bool, device=self.device)
        if bool(truncated.any()):
            self._last = torch.where(truncated, self._running, self._last)
            self.finished.extend(self._running[truncated].tolist())
            self._running = torch.where(truncated, torch.zeros_like(self._running), self._running)

        final = self._observe()["state"]
        extras = {
            "final_state": final,
            "raw_reward": reward,
            "episodic_return": self._last.mean(),
        }
        self._sample(truncated)
        return self._observe(), reward.unsqueeze(1), terminated, truncated, extras

    def close(self) -> None:
        pass

    # -- the derived optimum ---------------------------------------------- #

    def riccati(self) -> tuple[Tensor, list[Tensor]]:
        """The **exact finite-horizon** solution: `(P_0, [K_0 .. K_{H-1}])`.

        ☠️ **The steady-state DARE is the optimum of a different problem, and
        using it here would have been wrong.** A first draft of this file solved
        the infinite-horizon discounted DARE and scored the *undiscounted*
        100-step return against the gain it produced. 📏 That "optimum" is beaten
        by simply scaling the same gain up: `K * 1.10` scores **-17.63** against
        `K`'s **-18.81**. Two things were mismatched at once -- an infinite horizon
        against a finite one, and a discounted objective against an undiscounted
        readout -- and either alone is enough to make the reference not a bound.

        🔒 So the reference is the exact optimum of **the objective PPO is
        actually maximising**: the discounted 100-step return at this `gamma`.
        Backward recursion, `P_H = 0`, cost charged at `t = 0 .. H-1`:

            S   = R + g B'P_{t+1} B
            K_t = g S^-1 B'P_{t+1} A
            P_t = Q + g A'P_{t+1} A - g^2 A'P_{t+1}B S^-1 B'P_{t+1}A

        ⚠️ `B` and `R` are rescaled by `ACT_SCALE`, because the policy's action box
        is `[-1, 1]` and the plant sees `ACT_SCALE * u`. `K_t` comes back in policy
        units, so `-K_t x` is directly what the actor's mean is compared against.
        """
        g = self.gamma
        A = self.A.double()
        B = (self.B * self.ACT_SCALE).double()
        Q = self.Q.double()
        R = (self.R * (self.ACT_SCALE**2)).double()
        P = torch.zeros_like(Q)
        gains: list[Tensor] = []
        for _ in range(self.HORIZON):
            S = R + g * B.T @ P @ B
            K = g * torch.linalg.solve(S, B.T @ P @ A)
            P = Q + g * A.T @ P @ A - g * A.T @ P @ B @ K
            gains.append(K.float())
        gains.reverse()  # built backwards; `gains[t]` is the gain used at step t
        return P.float(), gains

    def optimal_return(self) -> float:
        """The exact expected optimal **discounted** episodic return. No sampling.

        ⭐ `J* = E[x_0' P_0 x_0] = tr(P_0 * Sigma)` with `Sigma = E[x_0 x_0'] =
        (1/3) I` for `x_0 ~ U[-1, 1]^4` componentwise. So the bar this task is
        judged against is a **closed form**, not a measurement of a reference
        controller -- there is no seed noise in it at all.

        ⚠️ It is a bound only while the action box does not bind.
        `optimal_saturation()` measures that and it is reported beside the number.
        """
        p0, _ = self.riccati()
        sigma = torch.eye(4) / 3.0
        return float(-(p0 * sigma).sum())

    @torch.no_grad()
    def optimal_saturation(self, steps: int | None = None) -> float:
        """Fraction of optimal-control actions that hit the `[-1, 1]` policy box.

        🔒 The precondition on `optimal_return` being a bound. If this is not ~0,
        the unconstrained Riccati solution is not feasible for the policy and the
        analytic number is an over-estimate of what any policy could reach.
        """
        _, gains = self.riccati()
        obs = self.reset()
        b = self.cfg.num_envs
        hits = 0.0
        for t in range(steps or self.HORIZON):
            # ☠️ No `/ ACT_SCALE`. `riccati` already rescales `B` and `R`, so
            # `K_t` maps state to a **policy-unit** action in [-1, 1]. Dividing
            # again silently detunes the controller by 3x -- which is how a first
            # draft of this file produced an "optimum" that `K * 1.10` beat.
            u = -(obs["state"] @ gains[t % self.HORIZON].T)
            hits += float((u.abs() >= 1.0).to(torch.float32).mean())
            obs, _, _, _, _ = self.step(u.view(b, 1, -1))
        return hits / float(steps or self.HORIZON)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    """One benchmark: how to build it, what to train it with, what to beat."""

    name: str
    make: Callable[[int, str], Any]
    ppo: PPOConfig
    total_timesteps: int
    hidden: int = 64
    critic_hidden: int = 64
    initial_log_std: float = 0.0
    orthogonal_init: bool = True
    eval_episodes: int = 100
    sde: bool = False
    gated: bool = True
    reference: dict[str, Any] = field(default_factory=dict)


def _pendulum(seed: int, device: str) -> GymAdapter:
    # 📏 `rl-baselines3-zoo/hyperparams/ppo.yml`, `Pendulum-v1`: n_envs 4,
    # n_steps 1024, gamma 0.9, n_epochs 10, lr 1e-3, gae_lambda 0.95,
    # clip_range 0.2, ent_coef 0.0, 1e5 steps. ⚠️ `use_sde: True` is NOT
    # reproduced -- gSDE is a different exploration distribution and this project
    # does not have one. That is a declared departure, not an oversight.
    return GymAdapter("Pendulum-v1", num_envs=4, seed=seed, device=device, gamma=0.9)


def _mountaincar(seed: int, device: str) -> GymAdapter:
    # 📏 Same source, `MountainCarContinuous-v0`: normalize true, n_envs 1,
    # n_steps 8, batch 256, gamma 0.9999, lr 7.77e-5, ent_coef 0.00429,
    # clip_range 0.1, n_epochs 10, gae_lambda 0.9, max_grad_norm 5, vf_coef 0.19,
    # log_std_init -3.29, ortho_init False, 2e4 steps.
    # ⚠️ `n_steps: 8` at `n_envs: 1` is 8 rows per rollout; `PPOTrainer` collects
    # `rollouts` steps of `num_envs` envs, so this is run at 8 envs x 8 rollouts =
    # 64 rows to keep the arithmetic sane, and that is a departure.
    return GymAdapter(
        "MountainCarContinuous-v0",
        num_envs=8,
        seed=seed,
        device=device,
        normalise_obs=True,
        normalise_reward=True,
        gamma=0.9999,
    )


def _lqr(seed: int, device: str) -> LQREnv:
    return LQREnv(num_envs=64, seed=seed, device=device, gamma=0.99)


TASKS: dict[str, Task] = {
    # ⭐ The strongest reference in this file: computed, not cited.
    "lqr": Task(
        name="LQR-4 (double integrator, dt=0.1, H=100)",
        make=_lqr,
        ppo=PPOConfig(
            # ⚠️ 64, not 100. At `rollouts = HORIZON` every rollout would end
            # exactly on a truncation and the bootstrap would only ever be
            # exercised on the buffer's last row -- which is the one row
            # `last_values` covers anyway. 64 makes truncations land mid-buffer,
            # where `compute_gae`'s mask has to do the work.
            rollouts=64,
            learning_epochs=10,
            mini_batch_size=256,
            discount_factor=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            value_loss_scale=0.5,
            value_clip=0.0,
            grad_norm_clip=0.5,
            normalise_values=True,
        ),
        total_timesteps=2_000_000,
        initial_log_std=-0.5,
        eval_episodes=256,
        reference={
            "source": "derived in-repo: exact finite-horizon discounted Riccati "
            "recursion, `LQREnv.optimal_return` = tr(P_0 * Sigma). No citation "
            "and no sampling",
            "kind": "known optimum",
        },
    ),
    "pendulum": Task(
        name="Pendulum-v1",
        make=_pendulum,
        ppo=PPOConfig(
            rollouts=1024,
            learning_epochs=10,
            mini_batch_size=64,
            discount_factor=0.9,
            gae_lambda=0.95,
            learning_rate=1e-3,
            value_loss_scale=0.5,
            value_clip=0.0,
            grad_norm_clip=0.5,
            entropy_loss_scale=0.0,
            normalise_values=False,
        ),
        total_timesteps=100_000,
        eval_episodes=100,
        reference={
            "tuned": "-172.225 +- 104.159, PPO, 100k steps, 750 eval episodes "
            "(rl-baselines3-zoo benchmark.md)",
            "untuned": "-1141.98 +- 135.55, cleanrl ppo_continuous_action, "
            "8M steps, 10 seeds (docs.cleanrl.dev/rl-algorithms/rpo)",
        },
    ),
    "mountaincar": Task(
        name="MountainCarContinuous-v0",
        make=_mountaincar,
        ppo=PPOConfig(
            rollouts=8,
            learning_epochs=10,
            mini_batch_size=64,
            discount_factor=0.9999,
            gae_lambda=0.9,
            learning_rate=7.77e-5,
            ratio_clip=0.1,
            value_loss_scale=0.19,
            value_clip=0.0,
            grad_norm_clip=5.0,
            entropy_loss_scale=0.00429,
            normalise_values=False,
        ),
        total_timesteps=20_000,
        initial_log_std=-3.29,
        # 📏 The tuned entry sets `ortho_init: False`. ⚠️ Missed in Phase 0 and
        # corrected here: under gSDE the initialisation is not cosmetic, because
        # the effective deviation scales with `||phi(s)||`.
        orthogonal_init=False,
        eval_episodes=100,
        gated=False,
        reference={
            "tuned": "88.343 +- 2.572, PPO, 20k steps, 633 eval episodes "
            "(rl-baselines3-zoo benchmark.md); uses gSDE, which this trainer "
            "does not have",
        },
    ),
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def rollout(
    env,
    control: Callable[[dict[str, Tensor], Tensor], Tensor],
    episodes: int,
    gamma: float = 1.0,
    max_steps: int = 200_000,
) -> tuple[list[float], list[float]]:
    """`(undiscounted, discounted)` episodic return under a deterministic control.

    ⚠️ Deterministic because that is what `benchmark.md` reports
    (`evaluate_policy(..., deterministic=True)`). A stochastic evaluation of the
    same policy on Pendulum is worth tens of points and the two are not
    comparable.

    🔒 Both returns are computed from `extras["raw_reward"]`, never from the reward
    `step` returns: on a task with `normalize: true` the returned one is
    `VecNormalize`-scaled and a return built from it is not comparable with any
    published number.

    `control` is handed `(obs, elapsed)` so a time-varying gain can index itself.
    """
    obs = env.reset()
    b = env.cfg.num_envs
    plain: list[float] = []
    disc: list[float] = []
    running = torch.zeros(b, device=env.device)
    running_d = torch.zeros(b, device=env.device)
    elapsed = torch.zeros(b, dtype=torch.long, device=env.device)
    steps = 0
    while len(plain) < episodes and steps < max_steps:
        action = control(obs, elapsed)
        obs, _, terminated, truncated, extras = env.step(action.view(b, 1, -1))
        reward = extras["raw_reward"].reshape(b)
        running = running + reward
        running_d = running_d + (gamma ** elapsed.to(reward.dtype)) * reward
        elapsed = elapsed + 1
        done = terminated | truncated
        if bool(done.any()):
            plain.extend(running[done].tolist())
            disc.extend(running_d[done].tolist())
            zero = torch.zeros_like(running)
            running = torch.where(done, zero, running)
            running_d = torch.where(done, zero, running_d)
            elapsed = torch.where(done, torch.zeros_like(elapsed), elapsed)
        steps += 1
    return plain[:episodes], disc[:episodes]


def evaluate(env, actor: BenchActor, episodes: int, gamma: float = 1.0):
    """`rollout` driven by the actor's **mean** action."""
    b = env.cfg.num_envs
    return rollout(env, lambda obs, _t: actor(obs["flat"].reshape(b, -1))[0], episodes, gamma)


def evaluate_lqr_optimum(env: LQREnv, episodes: int):
    """`rollout` driven by the exact finite-horizon gains `-K_t x`.

    ⭐ Its discounted return is a **cross-check on `optimal_return()`**, which is a
    closed form. If the simulated optimal controller and the analytic `tr(P_0
    Sigma)` disagree, the Riccati recursion or the plant is wrong, and that is
    worth catching before anything is read off this task.
    """
    _, gains = env.riccati()
    stacked = torch.stack(gains)

    def control(obs: dict[str, Tensor], elapsed: Tensor) -> Tensor:
        k = stacked[elapsed.clamp(max=len(gains) - 1)]  # (B, act, state)
        # ☠️ No `/ ACT_SCALE`. See `optimal_saturation`.
        return -torch.einsum("bas,bs->ba", k, obs["state"])

    return rollout(env, control, episodes, env.gamma)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_one(
    task: Task, seed: int, device: str, verbose: bool = True, control: str = "none"
) -> dict:
    """One seed, end to end. `control` selects a **negative control** instead.

    ⚠️ *"A benchmark that cannot fail is not a benchmark."* `control="frozen"` sets
    both learning rates to zero and changes nothing else, so the run collects,
    computes GAE and calls `optimizer.step()` exactly as usual and simply cannot
    move. Its score is what this harness reports for a trainer that does not
    learn, and every PASS below is only meaningful against it.
    """
    torch.manual_seed(seed)
    if control == "frozen":
        task = replace(task, ppo=replace(task.ppo, learning_rate=0.0, learning_rate_critic=0.0))
    env = task.make(seed, device)
    actor = BenchActor(
        env.obs_dim,
        env.act_dim,
        hidden=task.hidden,
        initial_log_std=task.initial_log_std,
        orthogonal_init=task.orthogonal_init,
        sde=task.sde,
        sde_sample_freq=task.ppo.sde_sample_freq,
    ).to(device)
    critic = SwarmCritic(env.state_dim, hidden=task.critic_hidden).to(device)

    trainer = PPOTrainer(
        env,
        actor,
        critic,
        cfg=task.ppo,
        total_timesteps=task.total_timesteps,
        curriculum=None,
        seed=seed,
        diagnostics=lambda e, x: {"episodic_return": x["episodic_return"]},
    )
    started = time.perf_counter()
    history = trainer.train(task.total_timesteps, log_lines=10)
    elapsed = time.perf_counter() - started

    gamma = task.ppo.discount_factor
    scores, discounted = evaluate(env, actor, task.eval_episodes, gamma)
    row = {
        "task": task.name,
        "control": control,
        "sde": task.sde,
        "seed": seed,
        "device": device,
        "timesteps": trainer.timestep,
        "adam_steps": adam_steps(task, trainer),
        "eval_episodes": len(scores),
        "eval_return_mean": float(np.mean(scores)),
        "eval_return_std": float(np.std(scores)),
        # ⚠️ The **discounted** return is what PPO maximises, and it is the column
        # `lqr` is gated on -- its bar is the exact optimum of that objective. The
        # undiscounted one is what published benchmarks report, and it is the
        # column `pendulum` and `mountaincar` are read against. They are different
        # quantities and both are recorded so neither can be quoted as the other.
        "eval_discounted_mean": float(np.mean(discounted)),
        "train_s": round(elapsed, 1),
        "actor_params": parameter_count(actor),
        "critic_params": parameter_count(critic),
    }
    tail = history[-1]
    for key in (
        "approx_kl",
        "grad_kept",
        "clip_fraction",
        "grad_norm_actor",
        "grad_norm_critic",
        "explained_variance",
        "entropy",
        "log_std",
        "episodic_return",
        # 🔒 The EFFECTIVE deviation. Under gSDE `log_std` is an unused parameter
        # and `sigma_*` is read off a real forward pass, which is the only reason
        # the two arms of the 2 x 2 can be compared at matched exploration.
        "sigma_x",
        "sigma_y",
        "sigma_z",
        "sat_any",
    ):
        if key in tail:
            row[key] = round(float(tail[key]), 6)
    if isinstance(env, LQREnv):
        opt_plain, opt_disc = evaluate_lqr_optimum(env, task.eval_episodes)
        row |= {
            # The closed form, and the simulated optimal controller that checks it.
            "optimal_discounted_exact": env.optimal_return(),
            "optimal_discounted_simulated": float(np.mean(opt_disc)),
            "optimal_return_mean": float(np.mean(opt_plain)),
            "optimal_saturation": env.optimal_saturation(),
            # 🔒 The gated readout: 1.0 is optimal, 0.0 is the do-nothing policy.
            # Scale-free, so the bar does not have to be re-derived if `Q`, `R` or
            # the horizon ever move.
            "normalised_score": normalised_score(
                float(np.mean(discounted)), env.optimal_return(), zero_return(env)
            ),
            # 🔒 **The gated readout**, and it is PAIRED. `evaluate` and
            # `evaluate_lqr_optimum` both open with `env.reset()`, which re-seeds
            # the episode stream, so the policy and the optimal controller are
            # scored on the **same initial states** and the ratio carries no
            # sampling noise from the state distribution. 1.0 is optimal; 1.25 is
            # a policy paying 25 % more cost than the best controller there is.
            "cost_ratio": float(np.mean(discounted)) / float(np.mean(opt_disc)),
        }
    env.close()
    if verbose:
        print(json.dumps(row))
    return row


def zero_return(env: LQREnv) -> float:
    """The exact expected discounted return of `u = 0`, in closed form.

    The floor the normalised score is measured from. With no control the plant is
    `x_{t+1} = A x_t`, so the cost-to-go matrix is the Lyapunov-style recursion
    `P_t = Q + g A' P_{t+1} A` and the same `tr(P_0 Sigma)` applies. ⚠️ Not a
    *random* policy: an exploratory Gaussian is charged `u'Ru` on top and scores
    worse than doing nothing, so "zero" is the honest floor for this task.
    """
    A, Q, g = env.A.double(), env.Q.double(), env.gamma
    P = torch.zeros_like(Q)
    for _ in range(env.HORIZON):
        P = Q + g * A.T @ P @ A
    return float(-(P.float() * (torch.eye(4) / 3.0)).sum())


def normalised_score(score: float, optimal: float, floor: float) -> float:
    """`0.0` at the do-nothing floor, `1.0` at the optimum."""
    return (score - floor) / (optimal - floor)


def adam_steps(task: Task, trainer: PPOTrainer) -> int:
    """Total gradient steps. 📏 The confound `docs/CAPABILITY_BRIEF.md` §0 names."""
    b = trainer.env.cfg.num_envs
    rounds = max(1, math.ceil(task.total_timesteps / (b * task.ppo.rollouts)))
    rows = task.ppo.rollouts * trainer.rows
    n_batches = (
        max(1, rows // task.ppo.mini_batch_size)
        if task.ppo.mini_batch_size is not None
        else task.ppo.mini_batches
    )
    return rounds * task.ppo.learning_epochs * n_batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--sde",
        default=None,
        choices=["on", "off"],
        help="override the task's gSDE setting. ⭐ The point of the override is the "
        "MATCHED PAIR: `--sde on` and `--sde off` at an identical "
        "`initial_log_std` isolate noise *correlation* from noise *magnitude*, "
        "which comparing against a published gSDE config alone cannot",
    )
    ap.add_argument(
        "--initial-log-std",
        type=float,
        default=None,
        help="override the task's initial exploration scale. ⚠️ Under gSDE this "
        "project's convention is that it names the EFFECTIVE deviation, not the "
        "noise-matrix entries -- see `SwarmActor._init_sde` -- so it is not "
        "interchangeable with SB3's `log_std_init`",
    )
    ap.add_argument(
        "--control",
        default="none",
        choices=["none", "frozen"],
        help="`frozen` runs the negative control: lr = 0 on both optimizers, "
        "everything else identical. A PASS is only readable against it",
    )
    ap.add_argument("--out", type=Path, default=Path("results/trainer_benchmark.jsonl"))
    a = ap.parse_args()

    task = TASKS[a.task]
    if a.sde is not None:
        task = replace(task, sde=(a.sde == "on"))
    if a.initial_log_std is not None:
        task = replace(task, initial_log_std=a.initial_log_std)
    rows = [run_one(task, seed, a.device, control=a.control) for seed in a.seeds]
    scores = sorted(r["eval_return_mean"] for r in rows)
    disc = sorted(r["eval_discounted_mean"] for r in rows)
    summary = {
        "task": task.name,
        "seeds": a.seeds,
        "device": a.device,
        "median": float(np.median(scores)),
        "worst": scores[0],
        "best": scores[-1],
        "per_seed": scores,
        "discounted_median": float(np.median(disc)),
        "discounted_per_seed": disc,
        "reference": task.reference,
        "gated": task.gated,
        "control": a.control,
        "sde": task.sde,
        "initial_log_std": task.initial_log_std,
    }
    if "normalised_score" in rows[0]:
        norms = sorted(r["normalised_score"] for r in rows)
        summary |= {
            "optimal_discounted_exact": rows[0]["optimal_discounted_exact"],
            "optimal_saturation": rows[0]["optimal_saturation"],
            "cost_ratio_median": float(np.median([r["cost_ratio"] for r in rows])),
            "cost_ratio_worst": max(r["cost_ratio"] for r in rows),
            "normalised_median": float(np.median(norms)),
            "normalised_worst": norms[0],
            "normalised_per_seed": norms,
        }
    print(json.dumps(summary, indent=2))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.write(json.dumps({"summary": summary}) + "\n")


if __name__ == "__main__":
    main()
