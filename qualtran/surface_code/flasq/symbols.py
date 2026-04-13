"""Sympy symbols for deferred resolution in FLASQ cost expressions.

These symbols act as placeholders in symbolic cost formulas, allowing
gate volumes and depths to be computed before concrete parameter values
(rotation error, reaction time, cultivation volume) are known. Use
substitute_until_fixed_point from utils.py to resolve them.
"""

import sympy

#: Per-rotation synthesis error epsilon in the paper.
ROTATION_ERROR = sympy.symbols("ROTATION_ERROR")

#: Cultivation spacetime volume v(p_phys, p_cult) in the paper.
V_CULT_FACTOR = sympy.symbols("V_CULT_FACTOR")

#: Reaction time in logical timesteps (t_react in the paper).
T_REACT = sympy.Symbol("t_react")

#: Expected T-count via mixed fallback synthesis (no ceiling — this is an
#: expected value, not an integer count).
MIXED_FALLBACK_T_COUNT = 4.86 - 0.53 * sympy.log(ROTATION_ERROR, 2)
