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
import collections.abc as abc
from collections import defaultdict
from typing import cast, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np
import sympy

from qualtran import Adjoint, Bloq, Controlled
from qualtran.resource_counting.generalizers import (
    ignore_alloc_free,
    ignore_cliffords,
    ignore_split_join,
)
from qualtran.symbolics import is_symbolic

if TYPE_CHECKING:
    from qualtran.resource_counting import GeneralizerT


def _get_basic_bloq_classification() -> Dict[str, str]:
    """High level classification of bloqs by the module name."""
    bloq_classifier = {
        'qualtran.bloqs.arithmetic': 'arithmetic',
        'qualtran.bloqs.rotations': 'rotations',
        'qualtran.bloqs.basic_gates.rotation': 'rotations',
        'qualtran.bloqs.state_preparation': 'state_preparation',
        'qualtran.bloqs.data_loading': 'data_loading',
        'qualtran.bloqs.mcmt': 'multi_control_pauli',
        'qualtran.bloqs.multiplexers': 'multiplexers',
        'qualtran.bloqs.swap_network': 'swaps',
        'qualtran.bloqs.basic_gates.swap': 'swaps',
        'qualtran.bloqs.reflection': 'reflection',
        'qualtran.bloqs.basic_gates.toffoli': 'toffoli',
        'qualtran.bloqs.basic_gates.t_gate': 'tgate',
    }
    return bloq_classifier


def classify_bloq(bloq: Bloq, bloq_classification: Dict[str, str]) -> str:
    """Classify a bloq given a bloq_classification.

    Args:
        bloq: The bloq to classify
        bloq_classification: A dictionary mapping a classification to a tuple of
            bloqs in that classification.
    Returns:
        classification: The matching key in bloq_classification. Returns other if not classified.
    """
    if 'adjoint' in bloq.__module__:
        mod_name = cast(Adjoint, bloq).subbloq.__module__
    else:
        mod_name = bloq.__module__
    for k, v in bloq_classification.items():
        if k in mod_name:
            return v
    return 'other'


def classify_t_count_by_bloq_type(
    bloq: Bloq,
    bloq_classification: Optional[Dict[str, str]] = None,
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
) -> Dict[str, Union[int, sympy.Expr]]:
    """Classify (bin) the T count of a bloq's call graph by type of operation.

    Args:
        bloq: the bloq to classify.
        bloq_classification: An optional dictionary mapping bloq_classifications to bloq types.
        generalizer: If provided, run this function on each (sub)bloq to replace attributes
            that do not affect resource estimates with generic sympy symbols. If the function
            returns `None`, the bloq is omitted from the counts graph. If a sequence of
            generalizers is provided, each generalizer will be run in order.

    Returns
        classified_bloqs: dictionary containing the T count for different types of bloqs.
    """
    from qualtran.resource_counting import get_cost_value, QECGatesCost

    if bloq_classification is None:
        bloq_classification = _get_basic_bloq_classification()
    keeper = lambda bloq: classify_bloq(bloq, bloq_classification) != 'other'
    basic_generalizer: List['GeneralizerT'] = [
        ignore_split_join,
        ignore_alloc_free,
        ignore_cliffords,
    ]
    if generalizer is not None:
        if isinstance(generalizer, abc.Sequence):
            basic_generalizer.extend(generalizer)
        else:
            basic_generalizer.append(generalizer)
    _, sigma = bloq.call_graph(generalizer=basic_generalizer, keep=keeper)
    classified_bloqs: Dict[str, Union[int, sympy.Expr]] = defaultdict(int)
    for k, v in sigma.items():
        classification = classify_bloq(k, bloq_classification)
        t_counts = get_cost_value(k, QECGatesCost()).total_t_count()
        if t_counts > 0:
            classified_bloqs[classification] += v * t_counts
    return dict(classified_bloqs)


_CLIFFORD_ANGLES = np.array(
    [
        np.pi,
        np.pi / 2,
        3 * np.pi / 2,
        -np.pi,
        -np.pi / 2,
        -3 * np.pi / 2,
        0.0,
        -2 * np.pi,
        2 * np.pi,
    ]
)
_CLIFFORD_EXPONENTS = np.array([1.0, 0.5, 1.5, -1.0, -0.5, -1.5, 0.0, -2, 2])
_T_ANGLES = np.array([np.pi / 4, -np.pi / 4])
_T_EXPONENTS = np.array([0.25, -0.25])
_ANGLE_ATOL = 1e-12


def bloq_is_t_like(b: Bloq) -> bool:
    """Whether a bloq should be counted as a T gate.

    This will return `True` for any instance of `TGate`. It will also consider
    single-qubit rotations and return True if the angle corresponds to a T gate
    (up to clifford reference frame).
    """
    from qualtran.bloqs.basic_gates import Rx, Ry, Rz, TGate, XPowGate, YPowGate, ZPowGate

    if isinstance(b, TGate):
        return True

    if isinstance(b, (Rz, Rx, Ry)):
        if is_symbolic(b.angle):
            return False  # Symbolic rotation
        if np.any(np.abs(b.angle - _T_ANGLES) < _ANGLE_ATOL):
            return True  # T hidden in a rotation bloq
        return False

    if isinstance(b, (ZPowGate, XPowGate, YPowGate)):
        if is_symbolic(b.exponent):
            return False  # Symbolic rotation
        if np.any(np.abs(b.exponent - _T_EXPONENTS) < _ANGLE_ATOL):
            return True  # T hidden in a rotation bloq
        return False

    return False


def bloq_is_clifford(b: Bloq) -> bool:
    """Whether the bloq represents a clifford operation.

    This checks against an explicit list of clifford bloqs in the Qualtran standard library,
    so it may return `False` for an unknown gate.

    This inspects single qubit rotations. If the angles correspond to Clifford angles, this
    returns `True`.
    """
    from qualtran.bloqs.basic_gates import (
        CNOT,
        CYGate,
        CZ,
        Hadamard,
        SGate,
        TwoBitSwap,
        XGate,
        YGate,
        ZGate,
    )
    from qualtran.bloqs.basic_gates.rotation import Rx, Ry, Rz, XPowGate, YPowGate, ZPowGate
    from qualtran.bloqs.bookkeeping import ArbitraryClifford

    if isinstance(b, Adjoint):
        b = b.subbloq

    if isinstance(
        b, (TwoBitSwap, Hadamard, XGate, ZGate, YGate, ArbitraryClifford, CNOT, CYGate, CZ, SGate)
    ):
        return True

    if isinstance(b, (Rz, Rx, Ry)):
        if is_symbolic(b.angle):
            return False  # Symbolic rotation
        if np.any(np.abs(b.angle - _CLIFFORD_ANGLES) < _ANGLE_ATOL):
            return True  # Clifford hidden in a rotation bloq
        return False

    if isinstance(b, (ZPowGate, XPowGate, YPowGate)):
        if is_symbolic(b.exponent):
            return False  # Symbolic rotation
        if np.any(np.abs(b.exponent - _CLIFFORD_EXPONENTS) < _ANGLE_ATOL):
            return True  # Clifford hidden in a rotation bloq
        return False

    return False


def bloq_is_rotation(b: Bloq) -> bool:
    """Whether a bloq represents a rotation operation.

    This inspects the single qubit rotation bloqs and returns `True` unless the angle
    represents a clifford or T-gate angle.

    This function has a shim for counting Controlled[Rotation] gates as a rotation, which
    will be remediated when the Qualtran standard library gains a bespoke bloq for each CRot.
    """
    from qualtran.bloqs.basic_gates import SGate, TGate
    from qualtran.bloqs.basic_gates.rotation import (
        CZPowGate,
        Rx,
        Ry,
        Rz,
        XPowGate,
        YPowGate,
        ZPowGate,
    )

    if isinstance(b, Controlled):
        if b.ctrl_spec.num_qubits > 1:
            return False

        # TODO https://github.com/quantumlib/Qualtran/issues/878
        #      explicit representation of all two-qubit rotations.
        if isinstance(b.subbloq, (SGate, TGate)):
            return True

        # For historical reasons, this hacky solution for controlled rotations does *not*
        # do clifford, T angle simplification.
        return isinstance(b.subbloq, (Rx, Ry, Rz, XPowGate, YPowGate, ZPowGate))

    if isinstance(b, CZPowGate):
        return True

    if isinstance(b, (Rz, Rx, Ry)):
        if is_symbolic(b.angle):
            return True
        if np.any(np.abs(b.angle - _CLIFFORD_ANGLES) < _ANGLE_ATOL):
            return False  # Clifford
        if np.any(np.abs(b.angle - _T_ANGLES) < _ANGLE_ATOL):
            return False  # T gate
        return True

    if isinstance(b, (ZPowGate, XPowGate, YPowGate)):
        if is_symbolic(b.exponent):
            return True
        if np.any(np.abs(b.exponent - _CLIFFORD_EXPONENTS) < _ANGLE_ATOL):
            return False  # Clifford
        if np.any(np.abs(b.exponent - _T_EXPONENTS) < _ANGLE_ATOL):
            return False  # T gate
        return True

    return False


def bloq_is_state_or_effect(b: Bloq) -> bool:
    from qualtran.bloqs.basic_gates.x_basis import _XVector
    from qualtran.bloqs.basic_gates.z_basis import _ZVector

    return isinstance(b, (_XVector, _ZVector))
