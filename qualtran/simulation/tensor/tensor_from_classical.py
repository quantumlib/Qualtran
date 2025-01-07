#  Copyright 2024 Google LLC
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
import itertools
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from qualtran import Bloq, Register

if TYPE_CHECKING:
    from qualtran.simulation.classical_sim import ClassicalValT


def _bits_to_classical_reg_data(reg: Register, bits) -> 'ClassicalValT':
    if reg.shape == ():
        return reg.dtype.from_bits(bits)
    return reg.dtype.from_bits_array(np.reshape(bits, reg.shape + (reg.dtype.num_qubits,)))


def tensor_from_classical_sim(bloq: Bloq) -> NDArray:
    left_qubit_counts = tuple(reg.total_bits() for reg in bloq.signature.lefts())
    left_qubit_splits = np.cumsum(left_qubit_counts)

    n_qubits_left = sum(left_qubit_counts)
    n_qubits_right = sum(reg.total_bits() for reg in bloq.signature.rights())

    matrix = np.zeros((2,) * (n_qubits_right + n_qubits_left))

    for input_t in itertools.product((0, 1), repeat=n_qubits_left):
        *inputs_t, last = np.split(input_t, left_qubit_splits)
        assert np.size(last) == 0

        input_kwargs = {
            reg.name: _bits_to_classical_reg_data(reg, bits)
            for reg, bits in zip(bloq.signature.lefts(), inputs_t)
        }
        output_args = bloq.call_classically(**input_kwargs)

        if output_args:
            output_t = np.concatenate(
                [
                    reg.dtype.to_bits_array(np.asarray(vals)).flat
                    for reg, vals in zip(bloq.signature.rights(), output_args)
                ]
            )
        else:
            output_t = np.array([])

        matrix[tuple([*np.atleast_1d(output_t), *np.atleast_1d(input_t)])] = 1

    shape: tuple[int, ...]
    if n_qubits_left == 0 and n_qubits_right == 0:
        shape = ()
    elif n_qubits_left == 0 or n_qubits_right == 0:
        shape = (2 ** max(n_qubits_left, n_qubits_right),)
    else:
        shape = (2**n_qubits_right, 2**n_qubits_left)

    return matrix.reshape(shape)
