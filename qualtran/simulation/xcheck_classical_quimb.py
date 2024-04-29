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
from typing import cast, Dict, Iterable, Optional, TYPE_CHECKING

import numpy as np

from qualtran import Bloq, BloqBuilder, CompositeBloq, Register, Soquet, SoquetT
from qualtran.bloqs.basic_gates import IntEffect, IntState

if TYPE_CHECKING:
    from qualtran.simulation.classical_sim import ClassicalValT


def _add_classical_kets(
    bb: BloqBuilder, registers: Iterable[Register], vals: Dict[str, 'ClassicalValT']
) -> Dict[str, 'SoquetT']:
    """Use `bb` to add `IntState` for all the `vals`."""
    soqs: Dict[str, 'SoquetT'] = {}
    for reg in registers:
        if reg.shape:
            reg_vals = np.asarray(vals[reg.name])
            soq = np.empty(reg.shape, dtype=object)
            for idx in reg.all_idxs():
                soq[idx] = bb.add(IntState(val=reg_vals[idx], bitsize=reg.bitsize))
        else:
            soq = bb.add(IntState(val=cast(int, vals[reg.name]), bitsize=reg.bitsize))

        soqs[reg.name] = soq
    return soqs


def _add_classical_bras(
    bb: BloqBuilder,
    registers: Iterable[Register],
    vals: Dict[str, 'ClassicalValT'],
    soqs: Dict[str, 'SoquetT'],
) -> None:
    """Use `bb` to add `IntEffect` on `soqs` for all the `vals`."""
    for reg in registers:
        if reg.shape:
            reg_vals = np.asarray(vals[reg.name])
            reg_name = soqs[reg.name]
            if isinstance(reg_name, Soquet):
                raise ValueError(f'soqs {reg.name} must be a numpy array: {soqs[reg.name]}')
            for idx in reg.all_idxs():
                bb.add(IntEffect(val=reg_vals[idx], bitsize=reg.bitsize), val=reg_name[idx])
        else:
            bb.add(
                IntEffect(val=cast(int, vals[reg.name]), bitsize=reg.bitsize), val=soqs[reg.name]
            )


def flank_with_classical_vectors(
    bloq: 'Bloq',
    in_vals: Dict[str, 'ClassicalValT'],
    out_vals: Optional[Dict[str, 'ClassicalValT']] = None,
) -> 'CompositeBloq':
    """Surround `bloq` with computational basis vectors according to the provided values.

    This function is useful for cross-checking classical and quantum simulation protocols.
    If bloq supports classical simulation and `out_vals` are not provided, the values
    will be determined by the classical simulation protocol. The resultant `CompositeBloq`
    can be tensor-contracted to assert its validity. Namely: if the tensor network contracts
    to the scalar value `1.0`, the inputs and outputs match the behavior of `bloq`.

    Args:
        bloq: The bloq to flank. We'll add `IntState`s and `IntEffect`s to all its registers
            according to the `vals` argument. If `out_vals` is not provided, this bloq
            must support the classical simulation protocol.
        in_vals: A mapping from register name to classical values. These will set the values
            of the initial `IntState`s.
        out_vals: A mapping from output register name to classical values. These will set the
            values of the final `IntEffect`s. If not provided, we will use bloq's classical
            simulation on `in_vals` to determine the correct `out_vals`.
    """
    bb = BloqBuilder()

    # Add the initial 'kets' according to the provided values.
    in_soqs = _add_classical_kets(bb, bloq.signature.lefts(), in_vals)

    # Add the bloq itself
    out_soqs = bb.add_d(bloq, **in_soqs)

    # Add the final 'bras' according to either the provided values or the result of the
    # classical simulation protocol on the input values.
    if out_vals is None:
        out_vals = bloq.as_composite_bloq().on_classical_vals(**in_vals)
    _add_classical_bras(bb, bloq.signature.rights(), out_vals, out_soqs)
    return bb.finalize()
