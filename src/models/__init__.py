"""Actor/critic architectures -- Block G's RQ2 ladder.

`docs/inherited/MODELS.md` is the specification. Three actor rungs isolating one
factor each (MLP -> DeepSets -> GNN), one critic shared by all three so RQ2
measures the actor and nothing else.

Both are plain `nn.Module`s: `SwarmActor.forward` returns `(mean, log_std)` and
`SwarmCritic.forward` returns the value. `src/training/ppo.py` owns the
algorithm. See `docs/REDUCTION.md` task 5 for why skrl's `Model` / mixin
inheritance was removed.
"""

from .actor import (
    ARCHITECTURES,
    SwarmActor,
    build_trunk,
    gaussian_entropy,
    parameter_count,
)
from .critic import SwarmCritic

__all__ = [
    "ARCHITECTURES",
    "SwarmActor",
    "SwarmCritic",
    "build_trunk",
    "gaussian_entropy",
    "parameter_count",
]
