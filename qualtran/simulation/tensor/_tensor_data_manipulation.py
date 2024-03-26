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

"""Utility methods to generate and manipulate tensor data for Bloqs."""
import itertools
from typing import List, Tuple, Union

import attrs
import numpy as np

from qualtran import CtrlSpec, Side, Signature


def tensor_out_inp_shape_from_signature(
    signature: Signature,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Returns a tuple for tensor data corresponding to signature.

    Tensor data for a bloq with a given `signature` can be expressed as a ndarray of
    shape `out_indices_shape + inp_indices_shape` where

     1. out_indices_shape - A tuple of values `2 ** soq.reg.bitsize` for every soquet `soq`
         corresponding to the RIGHT registers in signature.
     2. inp_indices_shape - A tuple of values `2 ** soq.reg.bitsize` for every soquet `soq`
         corresponding to the LEFT registers in signature.

    This method returns a tuple of (out_indices_shape, inp_indices_shape).
    """
    inp_indices_shape = [2**reg.bitsize for reg in signature.lefts() for _ in reg.all_idxs()]
    out_indices_shape = [2**reg.bitsize for reg in signature.rights() for _ in reg.all_idxs()]
    return tuple(out_indices_shape), tuple(inp_indices_shape)


def tensor_shape_from_signature(signature: Signature) -> Tuple[int, ...]:
    """Returns a tuple for tensor data corresponding to signature.

    Tensor data for a bloq with a given `signature` can be expressed as a ndarray of
    shape `out_indices_shape + inp_indices_shape` where

     1. out_indices_shape - A tuple of values `2 ** soq.reg.bitsize` for every soquet `soq`
         corresponding to the RIGHT registers in signature.
     2. inp_indices_shape - A tuple of values `2 ** soq.reg.bitsize` for every soquet `soq`
         corresponding to the LEFT registers in signature.

    This method returns a tuple of (*out_indices_shape, *inp_indices_shape).
    """
    out_shape, inp_shape = tensor_out_inp_shape_from_signature(signature)
    return out_shape + inp_shape


def active_space_for_ctrl_spec(
    signature: Signature, ctrl_spec: CtrlSpec
) -> Tuple[Union[int, slice], ...]:
    """Returns the "active" subspace corresponding to `signature` and `ctrl_spec`.

    Assumes first n-registers for `signature` are control registers corresponding to `ctrl_spec`.
    Returns a tuple of indices/slices that can be used to address into the ndarray, representing
    tensor data of shape `tensor_shape_from_signature(signature)`, and access the active subspace.
    """
    out_ind, inp_ind = tensor_out_inp_shape_from_signature(signature)
    data_shape = out_ind + inp_ind
    active_idx: List[Union[int, slice]] = [slice(x) for x in data_shape]
    ctrl_idx = 0
    for cv in ctrl_spec.cvs:
        for idx in itertools.product(*[range(sh) for sh in cv.shape]):
            active_idx[ctrl_idx] = int(cv[idx])
            active_idx[ctrl_idx + len(out_ind)] = int(cv[idx])
            ctrl_idx += 1
    return tuple(active_idx)


def _n_qubits(signature: Signature) -> int:
    return sum(reg.total_bits() for reg in signature)


def eye_tensor_for_signature(signature: Signature) -> np.ndarray:
    """Returns an identity tensor with shape `tensor_shape_from_signature(signature)`"""
    return tensor_data_from_unitary_and_signature(
        np.eye(2 ** _n_qubits(signature), dtype=np.complex128), signature
    )


def tensor_data_from_unitary_and_signature(unitary: np.ndarray, signature: Signature) -> np.ndarray:
    """Returns tensor data respecting `signature` corresponding to `unitary`

    For a given input unitary, we extract the action of the unitary on a subspace where
    input qubits corresponding to LEFT registers and output qubits corresponding to RIGHT
    registers in `signature` are 0.

    The input unitary is assumed to act on `_n_qubits(signature)`, and thus is of shape
    `(2 ** _n_qubits(signature), 2 ** _n_qubits(signature))` where `_n_qubits(signature)`
    is `sum(reg.total_bits() for reg in signature)`.

    The shape of the returned tensor matches `tensor_shape_from_signature(signature)`.
    """

    # Reshape the unitary into correct shape assuming all registers are THRU registers.
    assert unitary.shape == (2 ** _n_qubits(signature),) * 2
    signature_ignoring_sides = Signature([attrs.evolve(reg, side=Side.THRU) for reg in signature])
    unitary_shape = tensor_shape_from_signature(signature_ignoring_sides)
    n = len(unitary_shape) // 2
    unitary = unitary.reshape(unitary_shape)

    # Find the subspace corresponding to registers with sides.
    idx: List[Union[int, slice]] = [slice(x) for x in unitary_shape]
    curr_idx = 0
    for reg in signature:
        if reg.side == Side.LEFT:
            for _ in reg.all_idxs():
                # LEFT register ends, extract right subspace that's equivalent to 0.
                idx[curr_idx] = 0
                curr_idx += 1
        if reg.side == Side.RIGHT:
            for _ in reg.all_idxs():
                # Right register begins, extract the left subspace that's equivalent to 0.
                idx[curr_idx + n] = 0
                curr_idx += 1
        if reg.side == Side.THRU:
            curr_idx += int(np.prod(reg.shape))

    # Extract the subspace, assert it has the correct shape corresponding to `signature` and
    # return the result.
    unitary = unitary[tuple(idx)]
    assert unitary.shape == tensor_shape_from_signature(signature)
    return unitary
