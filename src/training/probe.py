"""A learning probe with a known optimum that **spans the episode**.

⚠️ This exists because the predecessor's probe did not span one, and that cost a
week. Its probe zeroed every objective term except `-w_effort*||a||^2`, whose
optimum is `a = 0`. PPO improved it monotonically, which was read as "the loop is
correct" -- wrapper, flattening, bootstrap, GAE, optimiser all cleared. But a
pure per-step action cost has almost **no cross-episode structure**: the optimal
action at step `t` does not depend on any other step, so a GAE recursion that
leaks across an episode boundary costs it nothing. The probe passed while
`compute_gae` was masking on `terminated` alone and recursing straight through
every reset. `docs/inherited/DECISIONS.md`:

> **The probe clears the plumbing; it does not clear credit assignment.** Add a
> probe with a known optimum that *spans* the episode before trusting it again.

## The task

Each drone starts at the origin and must reach a beacon `g` it can see, then hold
station on it. **Reward is paid only in the second half of the episode.** So the
first 32 steps pay exactly nothing and are worth something only through what they
make possible 32-64 steps later -- which is the property the predecessor's probe
lacked and the one that makes a boundary bug visible.

Two things it therefore exercises that a per-step probe does not:

* **long-horizon credit** -- the gradient on a step-0 action arrives from rewards
  at least 32 steps away, through GAE at `gamma = 0.997`;
* **the GAE episode mask** -- with `auto_reset` on, a leak across the boundary
  imports the *next* episode's beacon, which is independent of this one.

📏 **Measured, and it does catch the historical bug.** 120 rounds, 64 envs x 2
drones, mean episodic return per drone against a known optimum of **33.0**:

    correct mask (terminated | truncated)   11.3 -> 31.8 -> 32.8 -> 32.9 -> 32.7
    skrl's stale mask (terminated only)      9.8 -> 26.9 -> 16.9 -> 12.6 -> 12.0
    stale mask AND no truncation bootstrap  12.5 -> 32.2 -> 32.6 -> 32.8 -> 32.9

The middle row reproduces the predecessor's signature exactly: it improves, then
*degrades for the rest of training*. `test_ppo.py` pins the gap.

⚠️ **And the third row is the reason both fixes are pinned separately.** The two
bugs **cancel**: with no bootstrap there is no doubled continuation term for the
leak to double-count, so the probe looks healthy while two things are wrong. This
is the mechanism `docs/inherited/DECISIONS.md` describes ("Double-counted, and the
next episode's advantage propagates backwards") seen from the other side, and it
is why the truncation bootstrap has its own unit tests rather than relying on
this probe -- **the probe does not detect a missing bootstrap on its own**
(measured: 32.4 against 32.7, indistinguishable).

## The optimum, in closed form

Each axis closes at `SPEED` per step under `a = ±1`, and reward needs
`||x - g||_inf <= TOLERANCE`. So a drone arrives at
`t* = ceil((max_i |g_i| - TOLERANCE) / SPEED)` steps, and with `|g_i| <= 0.6`
that is at most **10** -- comfortably inside the unpaid first half. The optimal
episodic return is therefore exactly `HORIZON / 2` per drone, independently of
where the beacon fell, and `optimal_return()` returns it. A policy that does not
move scores ~0.

## Interface

It duck-types `BatchedSwarmEnv` closely enough for `PPOTrainer`, and deliberately
pads its observation to `FLAT_DIM` so the probe runs the **real** `SwarmActor`
and `SwarmCritic` rather than a stand-in. A stand-in would clear a loop the
mission does not use.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..env.core import FLAT_DIM

SPEED = 0.05  # per step, per axis, at |a| = 1
TOLERANCE = 0.1
BEACON_HALF = 0.6  # beacons are drawn from [-0.6, 0.6]^3
HORIZON = 64
PAID_FROM = HORIZON // 2  # reward is paid only from here on


@dataclass(frozen=True)
class ProbeConfig:
    num_envs: int = 64
    num_drones: int = 2
    device: str = "cpu"
    seed: int = 0
    #: Present so `PPOTrainer` can assert on them exactly as it does for the
    #: mission. Both must be True; the probe is a training env.
    training_extras: bool = True
    auto_reset: bool = True


class BeaconEnv:
    """`num_envs x num_drones` independent reach-and-hold tasks, stepped together.

    Every drone carries its own beacon, so the row folding
    (`num_envs * num_drones` rows, one parameter-shared policy) is exercised
    rather than assumed. The critic's state is the concatenation of all drones'
    positions and beacons -- centralized, exactly as in the mission.
    """

    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        b, n = cfg.num_envs, cfg.num_drones
        self.pos = torch.zeros(b, n, 3, device=self.device)
        self.goal = torch.zeros(b, n, 3, device=self.device)
        self.t = torch.zeros(b, dtype=torch.long, device=self.device)

    @property
    def state_dim(self) -> int:
        return 6 * self.cfg.num_drones

    def _sample(self, mask: Tensor) -> None:
        b, n = self.cfg.num_envs, self.cfg.num_drones
        fresh = (
            torch.rand(b, n, 3, device=self.device, generator=self.gen) * 2.0 - 1.0
        ) * BEACON_HALF
        m = mask.view(b, 1, 1)
        self.goal = torch.where(m, fresh, self.goal)
        self.pos = torch.where(m, torch.zeros_like(self.pos), self.pos)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)

    def _observe(self) -> dict[str, Tensor]:
        b, n = self.cfg.num_envs, self.cfg.num_drones
        # Padded into the real observation contract: the first six ego slots
        # carry own position and the vector to the beacon, every neighbour slot
        # is invalid. `unpack_flat` reads it exactly as it reads the mission's.
        flat = torch.zeros(b, n, FLAT_DIM, device=self.device)
        flat[..., 0:3] = self.pos
        flat[..., 3:6] = self.goal - self.pos
        # A clock feature would let the policy read the paid half directly. It
        # is deliberately absent: the credit has to come through the value
        # function, which is the thing under test.
        state = torch.cat([self.pos.flatten(1), self.goal.flatten(1)], dim=-1)
        return {"flat": flat, "state": state}

    def reset(self, seed: int | None = None) -> dict[str, Tensor]:
        if seed is not None:
            self.gen.manual_seed(seed)
        self._sample(torch.ones(self.cfg.num_envs, dtype=torch.bool, device=self.device))
        return self._observe()

    def step(self, actions: Tensor):
        b, n = self.cfg.num_envs, self.cfg.num_drones
        self.pos = (self.pos + actions.clamp(-1.0, 1.0) * SPEED).clamp(-1.0, 1.0)
        self.t = self.t + 1

        on_target = (self.pos - self.goal).abs().amax(dim=-1) <= TOLERANCE
        paid = (self.t >= PAID_FROM).unsqueeze(-1).expand(b, n)
        reward = (on_target & paid).to(torch.float32)

        terminated = torch.zeros(b, dtype=torch.bool, device=self.device)
        truncated = self.t >= HORIZON

        final = self._observe()
        extras = {"final_state": final["state"], "on_target": on_target}
        self._sample(truncated)
        return self._observe(), reward, terminated, truncated, extras

    @staticmethod
    def optimal_return() -> float:
        """Undiscounted episodic return of the optimal policy, per drone.

        `t* = ceil((max_i |g_i| - TOLERANCE) / SPEED) <= 10 < PAID_FROM`, so an
        optimal drone is already holding station when payment starts and
        collects every paid step. `t` is incremented before the reward is
        scored, so the paid steps are `t = PAID_FROM .. HORIZON` inclusive --
        `HORIZON - PAID_FROM + 1` of them, and the off-by-one matters because
        this number is the bar the probe is judged against.
        """
        assert (BEACON_HALF - TOLERANCE) / SPEED < PAID_FROM
        return float(HORIZON - PAID_FROM + 1)


def probe_diagnostics(env: BeaconEnv, extras: dict[str, Tensor]) -> dict[str, Tensor]:
    return {"on_target": extras["on_target"].to(torch.float32).mean()}
