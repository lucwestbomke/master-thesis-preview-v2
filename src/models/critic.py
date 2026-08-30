"""The centralized critic -- one design, identical in all three RQ2 conditions.

`docs/inherited/MODELS.md`: "Keep the critic identical across all three
architecture conditions. If only the actor varies, RQ2 isolates the actor. If
both vary, it is confounded." So this file has no `architecture` argument,
deliberately.

It need not be size-agnostic either: zero-shot transfer to `N in {3, 8}` runs
the actor alone and the critic is discarded at evaluation, so a plain MLP over
the concatenated global state is the right shape. The state is already unit-
scaled by `core._critic_state` (positions MCV-relative and divided by the box
half-width, velocities by the dash speed, capacity in threshold units), which is
why `src/training/ppo.py` normalises the value *output* and sets no state
preprocessor.

⛔ The recurrent variant is gone with `docs/REDUCTION.md` task 4: recurrence was
killed on its own pre-declared rule at 5 seeds (−1.05 pp, observer tenure 36.8
against a required 95, and the seed IQR *widened* 4.7 → 6.9). It trains, and it
reaches feedforward parity on the easy curriculum stage; it simply does not help.
Do not re-propose it for observer tenure without a new mechanism.
"""

from __future__ import annotations

from torch import Tensor, nn

from .actor import _mlp

DEFAULT_CRITIC_HIDDEN = 256


class SwarmCritic(nn.Module):
    """Value of the global state. Training only. `forward(state) -> (rows, 1)`."""

    def __init__(self, state_dim: int, hidden: int = DEFAULT_CRITIC_HIDDEN):
        super().__init__()
        self.state_dim = state_dim
        self.net = _mlp([state_dim, hidden, hidden, 1])

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)
