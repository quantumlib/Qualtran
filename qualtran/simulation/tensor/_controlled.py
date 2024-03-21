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
import itertools
from typing import Any, Callable, Dict, List, Sequence, Hashable

import numpy as np
import quimb.tensor as qtn
from numpy.typing import NDArray

from qualtran import Bloq, LeftDangle, RightDangle, SoquetT

from ._quimb import cbloq_as_contracted_tensor, cbloq_to_quimb


def _get_ctrl_tensor_data(n_ctrl: int, is_active_func: Callable[[Any, ...], bool]) -> NDArray:
    """Get an ndarray for a factorized control tensor.

    This tensor copies `n_ctrl` input legs to their same output values and activates the
    orthogonal `internal_ctrl_ind` leg where the control values are active (relative to the provided
    activation function).

    NOTE! the `is_active` leg is *active low* to correspond to
    `qtn.Tensor.new_ind_with_identity()`. A `True` return from `is_active_func` corresponds
    to the index value 0.

    The ndarray has indices ordered:
    (ctrl1_in, ... cntrln_in, ctrl1_out, ... cntrln_out, internal_ctrl_ind).
    """
    # TODO: this must use the ctrl_spec data types and shapes.

    # Each ctrl line has an input and output
    # plus the orthogonal activation leg.
    n_dim = 2 * n_ctrl + 1

    ctrl_tensor = np.zeros((2,) * n_dim, dtype=np.complex128)
    for ctrl_vals in itertools.product([0, 1], repeat=n_ctrl):
        is_active = 1 - int(is_active_func(*ctrl_vals))
        ctrl_tensor[ctrl_vals + ctrl_vals + (is_active,)] = 1.0

    return ctrl_tensor


def _get_ctrl_tensor(
    ctrl_reg_names: Sequence[str],
    is_active_func: Callable[[Any, ...], bool],
    incoming: Dict[str, Hashable],
    outgoing: Dict[str, Hashable],
    internal_ctrl_ind: str,
) -> qtn.Tensor:
    """Produce a factorized control tensor.

    This needs to be joined with an appropriate bloq tensor along `internal_ctrl_ind` to
    represent a full gate.

    Args:
        ctrl_reg_names: The register names that correspond to the control registers. Used
            to index into `incoming` and `outgoing.`
        is_active_func: A predicate that reports whether a given set of control values
            correspond to an activated bloq.
        incoming: A superset of incoming indices by register name. We'll select out those
            that correspond to controls using `ctrl_reg_names`.
        outgoing: A superset of outgoing indices by register name. We'll select out those
            that correspond to controls using `ctrl_reg_names`.
        internal_ctrl_ind: The name of the additional (internal) index. This index is active low!
            An index value of 0 corresponds to an activated tensor.
    """
    # i_ctrls = tuple(itertools.chain.from_iterable(incoming[creg_name].reshape(-1) for creg_name in ctrl_reg_names))
    # o_ctrls = tuple(itertools.chain.from_iterable(outgoing[creg_name].reshape(-1) for creg_name in ctrl_reg_names))
    i_ctrls = tuple(incoming[creg_name] for creg_name in ctrl_reg_names)
    o_ctrls = tuple(outgoing[creg_name] for creg_name in ctrl_reg_names)
    ctrl_tensor = _get_ctrl_tensor_data(n_ctrl=len(i_ctrls), is_active_func=is_active_func)
    return qtn.Tensor(data=ctrl_tensor, inds=i_ctrls + o_ctrls + (internal_ctrl_ind,))


def _get_ctrled_tensor_for_bloq(
    bloq: 'Bloq',
    incoming: Dict[str, 'SoquetT'],
    outgoing: Dict[str, 'SoquetT'],
    internal_ctrl_ind: str,
) -> qtn.Tensor:
    """Get a tensor for a bloq with one additional control-like index.

    This needs to be joined with an appropriate control tensor along `internal_ctrl_ind` to
    represent a full gate.

    Args:
        bloq: The bloq to contract and control.
        incoming: A superset of incoming indices by register name. We'll select out the
            indices for the (uncontrolled) `bloq`.
        outgoing: A superset of outgoing indices by register name. We'll select out the
            indices for the (uncontrolled) `bloq`.
        internal_ctrl_ind: The name of the additional (internal) index. This index is active low!
            An index value of 0 corresponds to an activated tensor.
    """
    tensor = cbloq_as_contracted_tensor(
        bloq.as_composite_bloq(), incoming=incoming, outgoing=outgoing
    )
    left_inds = [soq for soq in tensor.inds if soq.binst is LeftDangle]
    right_inds = [soq for soq in tensor.inds if soq.binst is RightDangle]
    tensor.new_ind_with_identity(name=internal_ctrl_ind, left_inds=left_inds, right_inds=right_inds)
    return tensor
