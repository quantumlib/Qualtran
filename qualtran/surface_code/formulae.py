#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math


def error_at(phys_err: float, *, d: int) -> float:
    """Logical error suppressed with code distance `d` for this physical error rate.

    This is an estimate, see the references section.

    The formula was originally expressed as $p_l = a (b * p_p)^((d+1)/2)$ physical
    error rate $p_p$ and parameters $a$ and $b$. This can alternatively be expressed with
    $p_th = (1/b)$ roughly corresponding to the code threshold. This is sometimes also
    expressed with $lambda = p_th / p_p$. A lambda of 10, for example, would be p_p = 1e-3
    and p_th = 0.01. The pre-factor $a$ has no clear provenance.

    References:
        Low overhead quantum computation using lattice surgery. Fowler and Gidney (2018).
        https://arxiv.org/abs/1808.06709.
        See section XV for introduction of this formula, with citation to below.

        Surface code quantum error correction incorporating accurate error propagation.
        Fowler et. al. (2010). https://arxiv.org/abs/1004.0255.
        Note: this doesn't actually contain the formula from the above reference.
    """
    return 0.1 * (100 * phys_err) ** ((d + 1) / 2)


def code_distance_from_budget(phys_err: float, budget: float) -> int:
    """Get the code distance that keeps one below the logical error `budget`."""

    # See: `error_at()`. p_l = a Λ^(-r) where r = (d+1)/2
    # Which we invert: r = ln(p_l/a) / ln(1/Λ)
    r = math.log(10 * budget) / math.log(100 * phys_err)
    d = 2 * math.ceil(r) - 1
    if d < 3:
        return 3
    return d
