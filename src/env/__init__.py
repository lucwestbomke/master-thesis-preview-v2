"""
Environment package.

Built and unit-tested (see the co-located `test_*.py` for hand-computed values):

    channel.py   path loss by link class, SINR with intra-swarm interference,
                 Shannon rate with a modulation cap
    routing.py   hop-limited widest-path DP; end-to-end rate min(C_i)/min(n,3)
    energy.py    rotary-wing propulsion power (U-shaped), radio DC draw
    reward.py    pure-function reward: mission term + potential-based shaping

    occlusion.py batched segment-vs-oriented-box (slab method), 2.5D
    core.py      the batched env, leading num_envs dimension -- THE training path

⚠️ `channel.py`, `routing.py`, `energy.py` and `occlusion.py` are inherited
UNCHANGED from the predecessor project, tests included. They are where this
project's bugs were found and fixed -- a dBm addition that returned +100 dB
SINR, a double-counted half-duplex cost, an inverted power curve, an endpoint
containment convention. Do not retype them; change them only against the cited
standard and the hand-computed tests.

`core.py` and `reward.py` are inherited and **scheduled for reduction** --
docs/REDUCTION.md names what comes out of each and in what order. Read it
before editing either.
"""
