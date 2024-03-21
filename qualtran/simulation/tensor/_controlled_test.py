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

import quimb.tensor as qtn

from qualtran.simulation.tensor._controlled import _get_ctrl_tensor, _get_ctrled_tensor_for_bloq, \
    _get_ctrl_tensor_data
import numpy as np
from numpy.typing import NDArray

_PAULI_X = np.array([[0,1], [1,0]])
_PAULI_X = np.array([[0.1,0.2], [0.3,0.4]])

def test_get_ctrl_data():
    def _is_active_1(cv: int):
        return bool(cv)
    td = _get_ctrl_tensor_data(n_ctrl=1, is_active_func=_is_active_1)
    print()
    for cv, not_active in np.where(td):
        if cv:
            assert not_active == 0
        else:
            assert not_active == 1
    print(np.where(td))
    print()

def test_ctrl():
    incoming = {
        'c1': 'c1i',
        'c2': 'c2i',
        'x': 'xi',
    }
    outgoing = {
        'c1': 'c1o',
        'c2': 'c2o',
        'x': 'xo',
    }
    ctrl_reg_names = ['c1', 'c2']

    def _is_active(c1:int, c2: int) -> bool:
        return (c1 == 0 and c2 == 0)

    internal_ctrl_ind = qtn.rand_uuid()
    ctrl_tensor = _get_ctrl_tensor(
        ctrl_reg_names,
        is_active_func=_is_active,
        incoming=incoming,
        outgoing=outgoing,
        internal_ctrl_ind=internal_ctrl_ind,
    )

    left_inds = ['xi']
    right_inds = ['xo']
    tensor = qtn.Tensor(data=_PAULI_X, inds=right_inds+left_inds)
    tensor.new_ind_with_identity(name=internal_ctrl_ind, left_inds=left_inds, right_inds=right_inds)

    res = ctrl_tensor & tensor
    mat = res.to_dense(('c1o', 'c2o', 'xo'), ('c1i', 'c2i', 'xi'))
    print()
    print(mat.astype(float))
    print()


def test_ctrl_shaped():
    incoming = {
        'cc': np.array([['c00i', 'c01i'],['c10i', 'c11i']]),
        'x': 'xi',
    }
    outgoing = {
        'cc': np.array([['c00o', 'c01o'],['c10o', 'c11o']]),
        'x': 'xo',
    }
    ctrl_reg_names = ['cc']

    def _is_active(cc: NDArray[int]) -> bool:
        return np.all(cc)

    internal_ctrl_ind = qtn.rand_uuid()
    ctrl_tensor = _get_ctrl_tensor(
        ctrl_reg_names,
        is_active_func=_is_active,
        incoming=incoming,
        outgoing=outgoing,
        internal_ctrl_ind=internal_ctrl_ind,
    )

    left_inds = ['xi']
    right_inds = ['xo']
    tensor = qtn.Tensor(data=_PAULI_X, inds=right_inds+left_inds)
    tensor.new_ind_with_identity(name=internal_ctrl_ind, left_inds=left_inds, right_inds=right_inds)

    res = ctrl_tensor & tensor
    mat = res.to_dense(('c00i', 'c01i', 'c10i', 'c11i', 'xi'), ('c00o', 'c01o', 'c10o', 'c11o', 'xo'))
    print()
    print(mat.astype(float))
    print()
