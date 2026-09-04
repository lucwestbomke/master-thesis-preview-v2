"""MAPPO for a homogeneous swarm: one parameter-shared actor, one centralized
critic, in one file that fits in a head.

`docs/REDUCTION.md` task 5 is why this exists. The predecessor used skrl, which
produced **four silent bugs**, none of which raised an error and each of which
cost real time. Three of the four are structural properties of the interface
rather than of the algorithm, so they are pinned here as invariants:

1. 🔒 **The swarm is ONE parameter-shared agent over `num_envs * N` rows, not `N`
   agents.** Handing a framework one shared model under `N` agent ids builds `N`
   optimizers over the same parameters and runs `N` stale sequential updates.
   Here the drone axis is folded into the batch axis in `_flatten` and there is
   no per-agent bookkeeping anywhere -- one optimizer, one update, correct
   ratios. It is also what makes zero-shot evaluation at `N = 8` possible at all.
2. 🔒 **`time_limit_bootstrap`.** Episodes truncate at a fixed horizon that is
   not part of the task, so a truncation is not a terminal state. Treating it as
   one teaches the critic the world ends at 600 steps, and at `gamma = 0.997`
   that bias is large and silent. Truncation adds `gamma * V(s_final)` to the
   reward AND masks the GAE recursion; termination masks it and adds nothing.
3. 🔒 **Bootstrap off `extras["final_state"]`, not off what `step()` returns.**
   With `auto_reset` the returned tensors are already a fresh episode's opening,
   so bootstrapping on them values an unrelated state. Requires
   `EnvConfig(training_extras=True)`, which is off by default; `PPOTrainer`
   refuses an env without it rather than silently doing the wrong thing.

The fourth (`clip_actions=True` inverting learning) lives in
`src/models/actor.py`, because it is a property of the distribution.

## What is deliberately NOT here

⛔ No framework, no registry, no callback system, no config file format. The one
dependency already in this project produced the four bugs above; adding another
is on `AGENTS.md`'s never-do list.

⛔ No adaptive curriculum advancement. `curriculum.CurriculumSchedule.weights()`
is a pure function of training progress and is never handed the return, the
fidelity rung or anything else it could adapt to. Adaptive advancement hands
easier conditions more experience and confounds RQ1 unrecoverably.

## Device discipline

`AGENTS.md`: no `.cpu()`, `.numpy()` or `.item()` in the hot loop -- `.item()`
forces a GPU sync and is the easy one to miss. Every per-step diagnostic here
accumulates into a device tensor and is read once, at a logging boundary, by
`_drain`. The rollout buffer is preallocated once and written in place.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from ..env.core import BOX_HALF_M, GAMMA, BatchedSwarmEnv
from ..models import SwarmActor, SwarmCritic
from .curriculum import CurriculumCallback, CurriculumSchedule

# --------------------------------------------------------------------------- #
# Value normalisation
# --------------------------------------------------------------------------- #


class RunningScalar(nn.Module):
    """Running mean/variance of a scalar, for normalising the value target.

    📏 Returns are of order **300** at `gamma = 0.997`, and an unnormalised
    critic has to fit that scale from an initialisation near zero. Closed in the
    predecessor's Block G as `value_preprocessor=RunningStandardScaler`; this is
    the same thing, without the framework.

    ⚠️ **Accumulator dtype is chosen by device, and it is a real constraint.**
    Over a 100 M-step run the parallel-variance update adds `delta * count /
    total` with `total` near 1e8, where a float32 increment can vanish entirely.
    So the moments are float64 -- except on MPS, which cannot allocate float64 at
    all, where float32 is the only option and the drift is accepted and stated.
    """

    def __init__(self, device: torch.device | str, epsilon: float = 1e-8, clip: float = 5.0):
        super().__init__()
        dev = torch.device(device)
        dtype = torch.float32 if dev.type == "mps" else torch.float64
        self.epsilon = epsilon
        self.clip = clip
        self.register_buffer("mean", torch.zeros((), dtype=dtype, device=dev))
        self.register_buffer("var", torch.ones((), dtype=dtype, device=dev))
        self.register_buffer("count", torch.ones((), dtype=dtype, device=dev))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Chan et al.'s parallel variance, on the flattened batch."""
        x = x.reshape(-1).to(self.mean.dtype)
        n = torch.tensor(float(x.numel()), dtype=self.mean.dtype, device=x.device)
        delta = x.mean() - self.mean
        total = self.count + n
        self.var.copy_(
            (self.var * self.count + x.var(unbiased=False) * n + delta**2 * self.count * n / total)
            / total
        )
        self.mean.copy_(self.mean + delta * n / total)
        self.count.copy_(total)

    def normalise(self, x: Tensor) -> Tensor:
        std = self.var.to(x.dtype).sqrt()
        return ((x - self.mean.to(x.dtype)) / (std + self.epsilon)).clamp(-self.clip, self.clip)

    def denormalise(self, x: Tensor) -> Tensor:
        std = self.var.to(x.dtype).sqrt()
        return std * x.clamp(-self.clip, self.clip) + self.mean.to(x.dtype)


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    last_values: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """GAE(lambda) over `(T, rows)` tensors. Returns `(returns, advantages)`.

    🔒 **The mask is `terminated | truncated`, and that is the whole point.**
    skrl's `ppo_rnn.py` masked on `terminated` alone, so at every truncation the
    recursion ran *through* the reset: the step collected `gamma * (V_{i+1} +
    lam * A_{i+1})` from the **next episode**, on top of a bootstrap already
    folded into the reward. Double-counted, with the next episode's advantage
    propagating backwards at `(gamma*lam)^k = 0.947` per step. Recurrent
    training collapsed for a week and the GRU was blamed for it.

    The truncation bootstrap itself is NOT applied here -- the caller has already
    added `gamma * V(final_state)` into `rewards` at truncated steps, because
    only the caller has the pre-reset state. See `PPOTrainer.collect`.

    ⚠️ `values` must be in **raw return units**, not normalised ones.
    """
    advantages = torch.zeros_like(rewards)
    not_done = (~(terminated | truncated)).to(rewards.dtype)
    advantage = torch.zeros_like(last_values)
    for t in range(rewards.shape[0] - 1, -1, -1):
        next_values = values[t + 1] if t < rewards.shape[0] - 1 else last_values
        advantage = rewards[t] - values[t] + gamma * not_done[t] * (next_values + lam * advantage)
        advantages[t] = advantage
    return advantages + values, advantages


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: 📏 The three cadences the predecessor's 81-run sweep compared, unchanged.
#: Each holds gradient density constant at **488 optimizer steps per M
#: env-steps** and varies what the extra samples buy. `deep` won; `wide`
#: *quadrupled* the seed spread it was built to shrink. `docs/INHERITED.md`.
#:
#: ⚠️ `deep` confounds `num_envs` with `rollouts` -- the predecessor recorded this
#: as open and it still is.
CADENCES: dict[str, dict[str, int]] = {
    "base": {"num_envs": 1024, "rollouts": 32, "mini_batches": 4},
    "wide": {"num_envs": 4096, "rollouts": 32, "mini_batches": 16},
    "deep": {"num_envs": 4096, "rollouts": 64, "mini_batches": 32},
}


@dataclass
class PPOConfig:
    """Everything the algorithm reads. Defaults reproduce the inherited runs.

    ⛔ `discount_factor` defaults to `core.GAMMA` by import, never by retyping:
    the env's PBRS shaping uses the same constant and the invariance proof
    requires the two to be identical.
    """

    rollouts: int = 64
    learning_epochs: int = 4
    mini_batches: int = 32
    discount_factor: float = GAMMA
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    value_loss_scale: float = 2.5
    #: ⚠️ The entropy here is **summed over the 3 action dimensions** and then
    #: averaged over rows; skrl averaged over both, so the same number is 3x
    #: stronger here. 📏 The inherited "0.01 drove the deviation from 0.64 to
    #: 1.11 and the actions became noise" was measured under skrl's convention.
    #: Divide by 3 before quoting that experiment against this flag.
    entropy_loss_scale: float = 0.0
    grad_norm_clip: float = 0.5
    kl_threshold: float = 0.0  # 0 disables early stopping

    # -- the optimisation budget. 🔒 Every default here reproduces the inherited
    # runs exactly; each is a knob that has NEVER been varied in this project. --
    #
    # ☠️ **The frozen axis.** `docs/inherited/BLOCK_G.md` chose three cadences
    # holding "gradient density constant at 488 optimizer steps per M env-steps"
    # and noted, without following it up, that this forces
    # **the minibatch to 40,960 rows in all three**. At the `deep` cadence a
    # 12 M-step run is therefore
    #
    #     12e6 / (4096 * 64)              =    46 PPO updates
    #     46 * 4 epochs * 32 minibatches  = 5,888 Adam steps, total
    #
    # on a 137 k-parameter actor, at a fixed `lr = 3e-4` that BLOCK_G declares
    # explicitly out of the sweep. 📏 The measured consequence is in
    # `runs/val-gnn-deep-s*/log.jsonl`: `approx_kl` sits at **0.002-0.004** for
    # the whole run against PPO's usual 0.01-0.02, so total policy movement is
    # ~0.14 nats of KL end to end. Every result in `results/` -- the 81-run
    # sweep, the RQ2 ladder, all eight interventions -- was measured under it.
    #
    #: Rows per gradient step, set directly. `None` keeps `mini_batches`, which
    #: is what ties the minibatch to the cadence. ⚠️ Setting this costs almost no
    #: FLOPs: the same rows are visited the same number of times per epoch, so
    #: only kernel-launch overhead grows while the gradient-step count rises by
    #: the same factor the minibatch shrinks.
    mini_batch_size: int | None = None
    #: Critic learning rate. `None` = `learning_rate`, i.e. today's single Adam
    #: over the union of both parameter sets (Adam is per-parameter, so two
    #: optimizers at equal LR are mathematically identical to one).
    learning_rate_critic: float | None = None
    #: Separate gradient-norm clip for the critic. `None` reproduces the
    #: inherited **joint** clip over actor and critic parameters, which
    #: `docs/inherited/BLOCK_G.md` lists as open and untested: with
    #: `value_loss_scale = 2.5` a large value gradient scales the policy gradient
    #: down by the same factor. ⚠️ `grad_kept` is instrumented for exactly this
    #: and is **NaN in every logged run in `runs/`** -- it has never been read.
    grad_norm_clip_critic: float | None = None
    #: Adaptive LR on the measured KL, the MAPPO/SB3 rule. `0.0` disables it and
    #: is the inherited behaviour. When set, the LR is multiplied by 1/1.5 if the
    #: update round's KL exceeds `2 * target_kl` and by 1.5 if it falls below
    #: `target_kl / 2`, once per round rather than per minibatch.
    target_kl: float = 0.0
    #: Bounds for the adaptive rule, so it cannot run away in either direction.
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    #: 🔒 Never False in a reported run. See the module docstring.
    time_limit_bootstrap: bool = True
    normalise_values: bool = True
    #: ⚠️ **A deliberate departure from the inherited runs.** skrl's `sample_all`
    #: chunks the flattened `(T * rows)` buffer *in order*, so at the `deep`
    #: cadence a "minibatch" of 40 960 rows is every environment at two
    #: consecutive timesteps -- maximally correlated. Shuffling is the textbook
    #: choice and is the default here. Set False to reproduce skrl's ordering.
    shuffle_minibatches: bool = True


@dataclass
class TrainConfig:
    """The run. Env, models and schedule, plus every reward knob inside `Phi`.

    🔒 Every field of `RewardWeights` that is not an objective weight and not a
    physical reference lives inside `Phi`, is optimum-preserving, and must be
    settable from the command line -- because a knob that cannot be set cannot be
    swept. `scripts/train.py` derives its flags from `RewardWeights` itself and
    `test_train_cli.py` asserts the coverage; two real misses of exactly that
    shape are recorded in `docs/REDUCTION.md`.
    """

    architecture: str = "deepsets"
    hidden: int | None = None
    critic_hidden: int = 256
    num_envs: int = 4096
    num_drones: int = 5
    timesteps: int = 12_000_000
    seed: int = 0
    device: str = "cpu"
    fidelity: str = "F4"
    #: ⛔ `False` in every reported run: the eval split is the only
    #: generalisation check left. Tuning decisions are made on the train split.
    eval_routes: bool = False
    curriculum: bool = True
    stage: int = 4  # only consulted when `curriculum` is False
    initial_log_std: float = -0.5
    min_log_std: float = -20.0
    ppo: PPOConfig = field(default_factory=PPOConfig)
    log_every: int = 10
    out_dir: Path | None = None


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #


class PPOTrainer:
    """Collect `rollouts` steps, compute GAE, run `epochs x mini_batches`, repeat.

    The env is handed in already built, so the same class trains the real mission
    and the known-optimum probe in `probe.py` without knowing which is which.
    """

    def __init__(
        self,
        env: BatchedSwarmEnv,
        actor: SwarmActor,
        critic: SwarmCritic,
        cfg: PPOConfig | None = None,
        total_timesteps: int | None = None,
        curriculum: CurriculumSchedule | None = None,
        seed: int = 0,
        diagnostics: Callable[[BatchedSwarmEnv, dict[str, Tensor]], dict[str, Tensor]]
        | None = None,
    ):
        if not env.cfg.training_extras:
            raise ValueError(
                "PPOTrainer needs EnvConfig(training_extras=True): the truncation "
                "bootstrap must read extras['final_state'], and with auto_reset the "
                "state step() returns is already a fresh episode's opening"
            )
        if not env.cfg.auto_reset:
            raise ValueError("PPOTrainer needs auto_reset=True; training never waits on reset()")

        self.env = env
        self.actor = actor
        self.critic = critic
        self.cfg = cfg or PPOConfig()
        self.device = env.device
        self.rows = env.cfg.num_envs * env.cfg.num_drones
        self.total_timesteps = total_timesteps or 0
        self.diagnostics = diagnostics
        # 🔒 Two optimizers, always. Adam is per-parameter, so at equal learning
        # rates this is mathematically identical to the single Adam over the
        # union that the inherited runs used -- `test_ppo.py` pins that. What it
        # buys is the ability to give the critic its own LR and its own gradient
        # clip, neither of which was reachable before.
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.cfg.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=self.cfg.learning_rate_critic or self.cfg.learning_rate
        )
        self.scaler = RunningScalar(self.device) if self.cfg.normalise_values else None
        self.curriculum = (
            CurriculumCallback(env, max(self.total_timesteps, 1), curriculum)
            if curriculum is not None
            else None
        )
        # ⚠️ CPU generator on purpose: the minibatch permutation must not consume
        # the device stream the env draws episodes from, or changing
        # `mini_batches` would silently change which episodes a seed sees.
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.timestep = 0

        # Widths come from one real observation rather than from the env's
        # config, so the known-optimum probe in `probe.py` and the mission share
        # this class without either knowing about the other.
        self.obs = env.reset()
        flat, state = self._flatten(self.obs)
        t, rows = self.cfg.rollouts, self.rows
        self.buf = {
            "obs": torch.zeros(t, rows, flat.shape[-1], device=self.device),
            "state": torch.zeros(t, rows, state.shape[-1], device=self.device),
            "action": torch.zeros(t, rows, actor.head.out_features, device=self.device),
            "log_prob": torch.zeros(t, rows, device=self.device),
            "value": torch.zeros(t, rows, device=self.device),
            "value_norm": torch.zeros(t, rows, device=self.device),
            "reward": torch.zeros(t, rows, device=self.device),
            "terminated": torch.zeros(t, rows, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(t, rows, dtype=torch.bool, device=self.device),
        }
        # Diagnostics accumulate on device and are read ONCE, at a logging
        # boundary. `.item()` inside the loop is a GPU sync and is the easy one
        # to miss (`AGENTS.md`), so `_drain` is the only place that syncs.
        self.stats: dict[str, Tensor] = {}
        self.stat_counts: dict[str, float] = {}

    # -- helpers ---------------------------------------------------------- #

    def _flatten(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """`(rows, obs_dim)` and `(rows, state_dim)`.

        🔒 The global state is **repeated per drone**. That is what makes this one
        parameter-shared agent over `num_envs * N` rows rather than `N` agents,
        and it is also why `max |V_i - V_j| = 0` -- a measured property of the
        design, not a bug. `scripts/probe_credit.py` found it and `w_relay` was
        the intervention; both are recorded as nulls in `docs/INHERITED.md`.
        """
        b, n = self.env.cfg.num_envs, self.env.cfg.num_drones
        # 🔒 `flat_history` when the env stacks frames, `flat` otherwise. The
        # actor was built for whichever this env produces (`scripts/train.py`
        # passes `obs_history` to both), so a mismatch is a construction error
        # and not something to paper over here.
        key = "flat_history" if "flat_history" in obs else "flat"
        # Flattened to `(rows, k * FLAT_DIM)`: the rollout buffer, the minibatch
        # indexing and an ONNX export all want a plain 2-D input, so the actor --
        # not the trainer -- owns the unflatten back to `(rows, k, FLAT_DIM)`.
        flat = obs[key].reshape(b * n, -1)
        state = obs["state"].unsqueeze(1).expand(b, n, -1).reshape(b * n, -1)
        return flat, state

    def _accumulate(self, values: dict[str, Tensor]) -> None:
        """Add one scalar per key, on device. Each key counts its own samples,
        so collection stats (once per env-step) and update stats (once per
        gradient step) can share one dict without being divided by each other's
        denominator."""
        for key, value in values.items():
            prev = self.stats.get(key)
            self.stats[key] = value.detach() if prev is None else prev + value.detach()
            self.stat_counts[key] = self.stat_counts.get(key, 0.0) + 1.0

    def _drain(self) -> dict[str, float]:
        """The ONE host synchronisation per logging boundary."""
        out = {k: float(v) / self.stat_counts[k] for k, v in self.stats.items()}
        self.stats, self.stat_counts = {}, {}
        return out

    @torch.no_grad()
    def _values(self, state: Tensor) -> Tensor:
        """Critic value in **raw return units**, whatever the normaliser holds."""
        return self._values_both(state)[0]

    @torch.no_grad()
    def _values_both(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """`(raw, normalised)`. GAE needs raw; the value clip needs normalised.

        ☠️ **Returning only the raw value and re-normalising it later is a bug,
        and it cost a collapsed seed in five.** The scaler's statistics move
        between collection and the update, so
        `normalise_new(denormalise_old(x)) != x` -- and the gap is largest
        exactly at a curriculum boundary, where the return distribution shifts.
        Feed that stale reference to a `clamp`-based value clip and the critic's
        gradient can go **structurally zero**, permanently. 📏 Measured: `g_crit`
        0.195 -> 0.005 -> 0.000 at progress 0.20, one boundary after the
        curriculum's 0.15, and zero for the remaining 80 % of the run.

        Keeping the network's own output removes the round trip entirely.
        """
        out = self.critic(state).squeeze(-1)
        if self.scaler is None:
            return out, out
        return self.scaler.denormalise(out), out

    # -- collection ------------------------------------------------------- #

    @torch.no_grad()
    def collect(self) -> Tensor:
        """Fill the buffer. Returns `last_values`, `(rows,)`, in raw units."""
        b, n = self.env.cfg.num_envs, self.env.cfg.num_drones
        for t in range(self.cfg.rollouts):
            if self.curriculum is not None:
                self.curriculum.update(self.timestep)

            flat, state = self._flatten(self.obs)
            action, log_prob, _ = self.actor.act(flat)
            value, value_norm = self._values_both(state)

            obs, reward, terminated, truncated, extras = self.env.step(action.view(b, n, -1))

            # (B,) episode flags broadcast to every drone of that environment.
            term = terminated.unsqueeze(-1).expand(b, n).reshape(-1)
            trunc = truncated.unsqueeze(-1).expand(b, n).reshape(-1)
            rew = reward.reshape(-1)

            # 🔒 Truncation bootstrap, off the PRE-RESET state. `extras` carries
            # it precisely because `obs` above is already the next episode.
            if self.cfg.time_limit_bootstrap:
                final = extras["final_state"].unsqueeze(1).expand(b, n, -1).reshape(b * n, -1)
                rew = rew + self.cfg.discount_factor * self._values(final) * trunc.to(rew.dtype)

            self.buf["obs"][t] = flat
            self.buf["state"][t] = state
            self.buf["action"][t] = action
            self.buf["log_prob"][t] = log_prob
            self.buf["value"][t] = value
            self.buf["value_norm"][t] = value_norm
            self.buf["reward"][t] = rew
            self.buf["terminated"][t] = term
            self.buf["truncated"][t] = trunc

            self.obs = obs
            self.timestep += b
            row = {"reward": reward.mean(), "action_abs": action.abs().mean()}
            if self.diagnostics is not None:
                row |= self.diagnostics(self.env, extras)
            self._accumulate(row)

        _, state = self._flatten(self.obs)
        return self._values(state)

    # -- update ----------------------------------------------------------- #

    def update(self, last_values: Tensor) -> None:
        cfg = self.cfg
        returns, advantages = compute_gae(
            self.buf["reward"],
            self.buf["value"],
            self.buf["terminated"],
            self.buf["truncated"],
            last_values,
            cfg.discount_factor,
            cfg.gae_lambda,
        )
        # 📏 Explained variance -- the standard "is the critic any good?"
        # diagnostic, and the one this project never had. `1 - Var(R - V)/Var(R)`:
        # 1.0 is a perfect value function, 0.0 is no better than predicting the
        # mean return, negative is worse than that.
        #
        # ⚠️ There is a CEILING on it here, and the ceiling is structural. The
        # global state is repeated per drone, but `reward()` is per-drone (team
        # terms plus INDIVIDUAL energy and effort costs), so N rows share one
        # input and carry N different targets. The critic can only learn their
        # mean; the between-drone spread is irreducible error that goes straight
        # into advantage noise. `return_spread_between_drones` measures it, so
        # the ceiling can be read off rather than guessed at.
        b, n = self.env.cfg.num_envs, self.env.cfg.num_drones
        resid = returns - self.buf["value"]
        self._accumulate(
            {
                "explained_variance": 1.0 - resid.var() / returns.var().clamp_min(1e-8),
                "return_spread_between_drones": returns.view(-1, b, n).std(dim=-1).mean(),
                "return_std": returns.std(),
            }
        )

        # Normalise advantages once over the whole batch, as PPO does -- not per
        # minibatch, which would rescale each chunk by its own noise.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        values = self.buf["value"]
        if self.scaler is not None:
            # ⚠️ The running moments are fitted on the RETURNS only. skrl fitted
            # them on the values and then again on the returns in the same
            # update, which mixes two distributions into one scale for no stated
            # reason. The difference is small and it is recorded here rather
            # than left to be rediscovered as a discrepancy.
            self.scaler.update(returns)
            returns = self.scaler.normalise(returns)
            # 🔒 The critic's OWN output, not a re-normalised raw value. See
            # `_values_both` for the seed this cost.
            values = self.buf["value_norm"]

        def flat(x: Tensor) -> Tensor:
            return x.reshape(-1, *x.shape[2:])

        obs, state = flat(self.buf["obs"]), flat(self.buf["state"])
        action, old_log_prob = flat(self.buf["action"]), flat(self.buf["log_prob"])
        returns, advantages, values = flat(returns), flat(advantages), flat(values)

        kl_sum = torch.zeros((), device=self.device)
        kl_n = 0.0
        total = obs.shape[0]
        # ☠️ The frozen axis -- see `PPOConfig.mini_batch_size`. `mini_batches`
        # ties the gradient-step count to the cadence, which is how every run in
        # this project ended up at 40,960 rows per step regardless of preset.
        if cfg.mini_batch_size is not None:
            n_batches = max(1, total // cfg.mini_batch_size)
        else:
            n_batches = cfg.mini_batches
        size = total // n_batches
        stop = False
        for _ in range(cfg.learning_epochs):
            if stop:
                break
            order = (
                torch.randperm(total, generator=self.gen).to(self.device)
                if cfg.shuffle_minibatches
                else torch.arange(total, device=self.device)
            )
            for i in range(n_batches):
                idx = order[i * size : (i + 1) * size]

                log_prob, entropy = self.actor.evaluate(obs[idx], action[idx])
                log_ratio = log_prob - old_log_prob[idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # Schulman's k3 estimator; unbiased and non-negative.
                    kl = ((ratio - 1.0) - log_ratio).mean()
                    kl_sum = kl_sum + kl
                    kl_n += 1.0
                if cfg.kl_threshold and bool(kl > cfg.kl_threshold):
                    stop = True
                    break

                surrogate = advantages[idx] * ratio
                clipped = advantages[idx] * ratio.clamp(1 - cfg.ratio_clip, 1 + cfg.ratio_clip)
                policy_loss = -torch.min(surrogate, clipped).mean()

                predicted = self.critic(state[idx]).squeeze(-1)
                squared = (predicted - returns[idx]).pow(2)
                if cfg.value_clip > 0:
                    # ☠️ **The max form, not a bare `clamp`.** A `clamp` has
                    # EXACTLY zero gradient outside its range, so once the
                    # critic drifts more than `value_clip` from the stored
                    # reference on every row of a minibatch, the value loss
                    # stops producing gradient and the critic can never move
                    # back -- it is clipped against a value it cannot update
                    # toward. That is a trap, and it is self-reinforcing.
                    # Schulman's PPO2 form takes the LARGER of the clipped and
                    # unclipped losses. ⚠️ It does NOT by itself guarantee a
                    # gradient -- when the clipped loss is the larger it is
                    # selected, and it is constant in `predicted`. It bounds the
                    # damage; **the actual fix is that `values` is now the
                    # critic's own output** (`_values_both`), so at epoch 0 the
                    # clip cannot be saturated at all.
                    clipped = values[idx] + (predicted - values[idx]).clamp(
                        -cfg.value_clip, cfg.value_clip
                    )
                    squared = torch.max(squared, (clipped - returns[idx]).pow(2))
                value_loss = cfg.value_loss_scale * squared.mean()

                entropy_loss = -cfg.entropy_loss_scale * entropy.mean()
                loss = policy_loss + value_loss + entropy_loss

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # ⚠️ Instrumented because `docs/inherited/BLOCK_G.md` lists it as
                # OPEN: the norm clip is applied to actor and critic parameters
                # JOINTLY, so a large value-loss gradient scales the policy
                # gradient down by the same factor. If `grad_norm_critic`
                # dominates `grad_norm_actor`, the policy is being throttled by
                # the critic and the mean will barely move -- which is visible
                # as a small `approx_kl` and an `action_abs` that stays at what
                # pure exploration noise would give.
                g_actor = _grad_norm(self.actor.parameters())
                g_critic = _grad_norm(self.critic.parameters())
                if cfg.grad_norm_clip_critic is not None:
                    # ✅ Separate norms. `policy_loss` and `entropy_loss` are
                    # functions of the actor's parameters alone and `value_loss`
                    # of the critic's, so the two gradients are disjoint and
                    # clipping them apart is exact rather than an approximation.
                    if cfg.grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_norm_clip)
                    if cfg.grad_norm_clip_critic > 0:
                        nn.utils.clip_grad_norm_(
                            self.critic.parameters(), cfg.grad_norm_clip_critic
                        )
                elif cfg.grad_norm_clip > 0:
                    # ⚠️ Applied JOINTLY to actor and critic parameters, which is
                    # what the inherited runs did. `docs/inherited/BLOCK_G.md`
                    # flags it as open: a large value-loss gradient can throttle
                    # the policy gradient through the shared norm.
                    nn.utils.clip_grad_norm_(
                        [*self.actor.parameters(), *self.critic.parameters()],
                        cfg.grad_norm_clip,
                    )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self._accumulate(
                    {
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "approx_kl": kl,
                        "entropy": entropy.mean(),
                        "log_std": self.actor.log_std.mean(),
                        "lr_actor": torch.tensor(
                            self.actor_optimizer.param_groups[0]["lr"], device=self.device
                        ),
                        "grad_norm_actor": g_actor,
                        "grad_norm_critic": g_critic,
                        # What fraction of the ACTOR's gradient survives the
                        # clip. 1.0 = untouched.
                        #
                        # ☠️ **The denominator depends on which clip is in
                        # force**, and getting it wrong makes the number quietly
                        # meaningless -- which is how this diagnostic came to sit
                        # at NaN in every log in `runs/` without anyone noticing.
                        # Under the JOINT clip the actor is scaled by
                        # `clip / ||[g_actor, g_critic]||`, so the critic's
                        # gradient throttles the policy; under SEPARATE clips it
                        # is scaled by `clip / ||g_actor||` and the critic cannot
                        # reach it at all. Reporting the joint form while running
                        # separate clips would understate retention by exactly
                        # the factor the split was made to remove.
                        "grad_kept": _grad_kept(cfg, g_actor, g_critic),
                    }
                )

        self._adapt_learning_rate(kl_sum / max(kl_n, 1.0))

    # -- adaptive learning rate ------------------------------------------- #

    def _adapt_learning_rate(self, kl: Tensor) -> None:
        """MAPPO/SB3's KL-targeting rule, once per update round.

        ⛔ Off unless `target_kl > 0`, and then it moves the ACTOR's LR only --
        the critic is fitting a regression whose difficulty has nothing to do
        with how far the policy moved, so tying its step size to the policy's KL
        would couple two unrelated schedules.

        📏 Why the rule is here at all: `runs/val-gnn-deep-s*/log.jsonl` records
        `approx_kl` at **0.002-0.004** for entire 12 M-step runs, against PPO's
        usual 0.01-0.02 and a `ratio_clip = 0.2` that corresponds to far more.
        The policy is taking steps roughly an order of magnitude smaller than the
        algorithm is designed for, and `lr` was declared out of the sweep.

        ⚠️ One host sync per update round, deliberately -- `bool(kl > ...)` reads
        a device scalar. That is ~46 syncs in a 12 M-step run at the `deep`
        cadence, outside the hot loop, against `AGENTS.md`'s rule about
        `.item()` in `step()`. The alternative is a tensor-valued LR, which Adam
        does not accept.
        """
        cfg = self.cfg
        if cfg.target_kl <= 0.0:
            return
        lr = self.actor_optimizer.param_groups[0]["lr"]
        measured = float(kl)
        if measured > 2.0 * cfg.target_kl:
            lr = max(cfg.lr_min, lr / 1.5)
        elif measured < 0.5 * cfg.target_kl:
            lr = min(cfg.lr_max, lr * 1.5)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr

    # -- driver ----------------------------------------------------------- #

    def train(self, total_timesteps: int, on_log=None, log_lines: int = 20) -> list[dict]:
        """Run to `total_timesteps` env-steps. Returns the log history.

        One env-step is one tick of one environment, so a round of collection is
        `num_envs * rollouts` of them -- the unit `docs/INHERITED.md` quotes
        throughput and run length in.
        """
        b = self.env.cfg.num_envs
        self.total_timesteps = total_timesteps
        if self.curriculum is not None:
            self.curriculum.total_timesteps = total_timesteps
        rounds = max(1, math.ceil(total_timesteps / (b * self.cfg.rollouts)))
        every = max(1, rounds // max(1, log_lines))

        history: list[dict] = []
        started = time.perf_counter()
        for r in range(1, rounds + 1):
            self.update(self.collect())
            if r % every == 0 or r == rounds:
                elapsed = time.perf_counter() - started
                row = self._drain() | {
                    "round": r,
                    "timestep": self.timestep,
                    "elapsed_s": elapsed,
                    "steps_per_s": self.timestep / max(elapsed, 1e-9),
                    "progress": self.timestep / max(total_timesteps, 1),
                }
                history.append(row)
                if on_log is not None:
                    on_log(row)
        return history

    def save(self, path: Path, extra: dict | None = None) -> None:
        """The checkpoint contract `scripts/eval_policy.py` reads.

        🔒 `results/` is tracked and `runs/` is gitignored on purpose: the summary
        is a result, the checkpoints are regenerable.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.actor.state_dict(),
                "value": self.critic.state_dict(),
                "architecture": self.actor.architecture,
                "hidden": self.actor.trunk.out_dim,
                # 🔒 Top level, not buried in provenance: a loader that misses it
                # builds an actor of the wrong input width and either crashes or
                # -- worse -- silently scores a different network.
                "obs_history": getattr(self.actor, "obs_history", 1),
                "mask_jammed_obs": bool(getattr(self.env.cfg, "mask_jammed_obs", False)),
                "min_log_std": self.actor.min_log_std.tolist(),
                # 🔒 Top level for the same reason as `obs_history`: both change
                # what the network COMPUTES without changing the shape of its
                # state dict, so a loader that misses them scores a different
                # function and `load_state_dict` raises nothing.
                "tanh_mean": bool(getattr(self.actor, "tanh_mean", True)),
                "layer_norm": bool(getattr(self.actor, "layer_norm", False)),
                "timestep": self.timestep,
                **(extra or {}),
            },
            path,
        )


def _grad_kept(cfg: PPOConfig, g_actor: Tensor, g_critic: Tensor) -> Tensor:
    """Fraction of the actor's gradient that survives `clip_grad_norm_`.

    📏 The first reading of it, on a 120 k-step MPS smoke run at the defaults:
    **0.20-0.26**, with `grad_norm_actor` 1.8-2.4 against `grad_norm_clip = 0.5`.
    Three quarters of the policy gradient is discarded every step -- on top of a
    total budget of ~5,900 Adam steps. Both effects point the same way and they
    multiply.
    """
    if cfg.grad_norm_clip <= 0:
        return torch.ones((), device=g_actor.device)
    norm = (
        g_actor
        if cfg.grad_norm_clip_critic is not None
        else (g_actor.square() + g_critic.square()).sqrt()
    )
    return torch.clamp(cfg.grad_norm_clip / norm.clamp_min(1e-12), max=1.0)


def _grad_norm(params) -> Tensor:
    """L2 norm of one parameter group's gradient, on device (no `.item()`)."""
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))


def mission_diagnostics(env: BatchedSwarmEnv, extras: dict[str, Tensor]) -> dict[str, Tensor]:
    """The per-step behavioural signals worth watching during a mission run.

    ⚠️ `at_speed_cap` and `at_boundary` are here because they are Gate A's
    readout (`PLAN.md`): 📏 the learned policy sits pinned at the 25 m/s dash cap
    on **57 %** of steps and presses against the map boundary on **23 %**, where
    B0 scores 3.1 % and 0.9 %. Watching them during training is how a run that
    is heading there is spotted before it finishes.

    ⛔ Nothing here is a reported number. Every reported number goes through
    `src/baselines/evaluate.py`, on the eval split, at F4 -- a learned policy
    scored by a different loop is not comparable to B0.

    ⚠️ `at_boundary` is defined **here**, as "either horizontal coordinate within
    1 m of the box edge". The predecessor's 23.1 % came out of
    `scripts/measure_potential.py` and its definition was not carried over, so
    treat this column as a trend to watch during a run, **not** as a number
    comparable with that 23.1 %.
    """
    speed = env.drone_vel.norm(dim=-1)
    at_wall = (env.drone_pos[..., :2].abs() >= (BOX_HALF_M - 1.0)).any(dim=-1)
    f = torch.float32
    return {
        "mission_capable": extras["mission_capable"].to(f).mean(),
        "observed": extras["sees_any"].to(f).mean(),
        "e2e_mbps": extras["e2e_capacity_mbps"].mean(),
        "speed_ms": speed.mean(),
        "at_speed_cap": (speed > 24.0).to(f).mean(),
        "at_boundary": at_wall.to(f).mean(),
    }
