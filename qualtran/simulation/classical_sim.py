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

"""Functionality for the `Bloq.call_classically(...)` protocol."""
import itertools
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import networkx as nx
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqInstance,
    DanglingT,
    LeftDangle,
    Register,
    RightDangle,
    Signature,
    Soquet,
)
from qualtran._infra.composite_bloq import _binst_to_cxns

if TYPE_CHECKING:
    from qualtran import CompositeBloq, QCDType

ClassicalValT = Union[int, np.integer, NDArray[np.integer]]
ClassicalValRetT = Union[int, np.integer, NDArray[np.integer]]


def _numpy_dtype_from_qlt_dtype(dtype: 'QCDType') -> Type:
    # TODO: Move to a method on QCDType. https://github.com/quantumlib/Qualtran/issues/1437.
    from qualtran._infra.data_types import CBit, QBit, QInt, QUInt

    if isinstance(dtype, QUInt):
        if dtype.bitsize <= 8:
            return np.uint8
        elif dtype.bitsize <= 16:
            return np.uint16
        elif dtype.bitsize <= 32:
            return np.uint32
        elif dtype.bitsize <= 64:
            return np.uint64

    if isinstance(dtype, QInt):
        if dtype.bitsize <= 8:
            return np.int8
        elif dtype.bitsize <= 16:
            return np.int16
        elif dtype.bitsize <= 32:
            return np.int32
        elif dtype.bitsize <= 64:
            return np.int64

    if isinstance(dtype, (QBit, CBit)):
        return np.uint8

    return object


def _empty_ndarray_from_reg(reg: Register) -> np.ndarray:
    from qualtran._infra.data_types import QGF

    if isinstance(reg.dtype, QGF):
        return reg.dtype.gf_type.Zeros(reg.shape)

    return np.empty(reg.shape, dtype=_numpy_dtype_from_qlt_dtype(reg.dtype))


def _get_in_vals(
    binst: Union[DanglingT, BloqInstance], reg: Register, soq_assign: Dict[Soquet, ClassicalValT]
) -> ClassicalValT:
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    if not reg.shape:
        return soq_assign[Soquet(binst, reg)]

    arg = _empty_ndarray_from_reg(reg)
    for idx in reg.all_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


class ClassicalSimState:
    """A mutable class for classically simulating composite bloqs.

    Consider using the public method `Bloq.call_classically(...)` for a simple interface
    for classical simulation.

    The `.step()` and `.finalize()` methods provide fine-grained control over the progress
    of the simulation; or the `.simulate()` method will step through the entire composite bloq.

    Args:
        signature: The signature of the composite bloq.
        binst_graph: The directed-graph form of the composite bloq. Consider constructing
            this class with the `.from_cbloq` constructor method to correctly generate the
            binst graph.
        vals: A mapping of input register name to classical value to serve as inputs to the
            procedure.

    Attributes:
        soq_assign: An assignment of soquets to classical values. We store the classical state
            of each soquet (wire connection point in the compute graph) for debugging and/or
            visualization. After stepping through each bloq instance, the right-dangling soquet
            are assigned the output classical values
        last_binst: A record of the last bloq instance we processed during simulation. This
            can be used in concert with `.step()` for debugging.

    """

    def __init__(
        self,
        signature: 'Signature',
        binst_graph: nx.DiGraph,
        vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
    ):
        self._signature = signature
        self._binst_graph = binst_graph
        self._binst_iter = nx.topological_sort(self._binst_graph)

        # Keep track of each soquet's bit array. Initialize with LeftDangle
        self.soq_assign: Dict[Soquet, ClassicalValT] = {}
        self._update_assign_from_vals(self._signature.lefts(), LeftDangle, dict(vals))

        self.last_binst: Optional['BloqInstance'] = None

    @classmethod
    def from_cbloq(
        cls, cbloq: 'CompositeBloq', vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]]
    ) -> 'ClassicalSimState':
        """Initiate a classical simulation from a CompositeBloq.

        Args:
            cbloq: The composite bloq
            vals: A mapping of input register name to classical value to serve as inputs to the
                procedure.

        Returns:
            A new classical sim state.

        """
        return cls(signature=cbloq.signature, binst_graph=cbloq._binst_graph, vals=vals)

    def _update_assign_from_vals(
        self,
        regs: Iterable[Register],
        binst: Union[DanglingT, BloqInstance],
        vals: Union[Dict[str, Union[sympy.Symbol, ClassicalValT]], Dict[str, ClassicalValT]],
    ) -> None:
        """Update `self.soq_assign` using `vals`.

        This helper function is responsible for error checking. We use `regs` to make sure all the
        keys are present in the vals dictionary. We check the classical value shapes, types, and
        ranges.
        """
        for reg in regs:
            debug_str = f'{binst}.{reg.name}'
            try:
                val = vals[reg.name]
            except KeyError as e:
                raise ValueError(f"{binst} requires a {reg.side} register named {reg.name}") from e

            if reg.shape:
                # `val` is an array
                val = np.asanyarray(val)
                if val.shape != reg.shape:
                    raise ValueError(
                        f"Incorrect shape {val.shape} received for {debug_str}. "
                        f"Want {reg.shape}."
                    )
                reg.dtype.assert_valid_classical_val_array(val, debug_str)

                for idx in reg.all_idxs():
                    soq = Soquet(binst, reg, idx=idx)
                    self.soq_assign[soq] = val[idx]

            elif isinstance(val, sympy.Expr):
                # `val` is symbolic
                soq = Soquet(binst, reg)
                self.soq_assign[soq] = val  # type: ignore[assignment]

            else:
                # `val` is one value.
                reg.dtype.assert_valid_classical_val(val, debug_str)
                soq = Soquet(binst, reg)
                self.soq_assign[soq] = val

    def _binst_on_classical_vals(self, binst, in_vals) -> None:
        """Call `on_classical_vals` on a given bloq instance."""
        bloq = binst.bloq

        out_vals = bloq.on_classical_vals(**in_vals)
        if not isinstance(out_vals, dict):
            raise TypeError(
                f"{bloq.__class__.__name__}.on_classical_vals should return a dictionary."
            )
        self._update_assign_from_vals(bloq.signature.rights(), binst, out_vals)

    def _binst_basis_state_phase(self, binst, in_vals) -> None:
        """Call `basis_state_phase` on a given bloq instance.

        This base simulation class will raise an error if the bloq reports any phasing.
        This method is overwritten in `PhasedClassicalSimState` to support phasing.
        """
        bloq = binst.bloq
        bloq_phase = bloq.basis_state_phase(**in_vals)
        if bloq_phase is not None:
            raise ValueError(
                f"{bloq} imparts a phase, and can't be simulated purely classically. Consider using `do_phased_classical_simulation`."
            )

    def step(self) -> 'ClassicalSimState':
        """Advance the simulation by one bloq instance.

        After calling this method, `self.last_binst` will contain the bloq instance that
        was just simulated. `self.soq_assign` and any other state variables will be updated.

        Returns:
            self
        """
        binst = next(self._binst_iter)
        self.last_binst = binst
        if isinstance(binst, DanglingT):
            return self
        pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=self._binst_graph)

        # Track inter-Bloq name changes
        for cxn in pred_cxns:
            self.soq_assign[cxn.right] = self.soq_assign[cxn.left]

        def _in_vals(reg: Register):
            # close over binst and `soq_assign`
            return _get_in_vals(binst, reg, soq_assign=self.soq_assign)

        bloq = binst.bloq
        in_vals = {reg.name: _in_vals(reg) for reg in bloq.signature.lefts()}

        # Apply methods
        self._binst_on_classical_vals(binst, in_vals)
        self._binst_basis_state_phase(binst, in_vals)
        return self

    def finalize(self) -> Dict[str, 'ClassicalValT']:
        """Finish simulating a composite bloq and extract final values.

        Returns:
            final_vals: The final classical values, keyed by the RIGHT register names of the
                composite bloq.

        Raises:
            KeyError if `.step()` has not been called for each bloq instance.
        """

        # Track bloq-to-dangle name changes
        if len(list(self._signature.rights())) > 0:
            final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=self._binst_graph)
            for cxn in final_preds:
                self.soq_assign[cxn.right] = self.soq_assign[cxn.left]

        # Formulate output with expected API
        def _f_vals(reg: Register):
            return _get_in_vals(RightDangle, reg, soq_assign=self.soq_assign)

        final_vals = {reg.name: _f_vals(reg) for reg in self._signature.rights()}
        return final_vals

    def simulate(self) -> Dict[str, 'ClassicalValT']:
        """Simulate the composite bloq and return the final values."""
        try:
            while True:
                self.step()
        except StopIteration:
            return self.finalize()


class PhasedClassicalSimState(ClassicalSimState):
    """A mutable class for classically simulating composite bloqs with phase tracking.

    The convenience function `do_phased_classical_simulation` will simulate a bloq. Use this
    class directly for more fine-grained control.

    This simulation scheme supports a class of circuits containing only:
     - classical operations corresponding to permutation matrices in the computational basis
     - phase-like operations corresponding to diagonal matrices in the computational basis.

    Args:
        signature: The signature of the composite bloq.
        binst_graph: The directed-graph form of the composite bloq. Consider constructing
            this class with the `.from_cbloq` constructor method to correctly generate the
            binst graph.
        vals: A mapping of input register name to classical value to serve as inputs to the
            procedure.
        phase: The initial phase. It must be a valid phase: a complex number with unit modulus.

    Attributes:
        soq_assign: An assignment of soquets to classical values.
        last_binst: A record of the last bloq instance we processed during simulation.
        phase: The current phase of the simulation state.
    """

    def __init__(
        self,
        signature: 'Signature',
        binst_graph: nx.DiGraph,
        vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
        *,
        phase: complex = 1.0,
    ):
        super().__init__(signature=signature, binst_graph=binst_graph, vals=vals)
        _assert_valid_phase(phase)
        self.phase = phase

    @classmethod
    def from_cbloq(
        cls, cbloq: 'CompositeBloq', vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]]
    ) -> 'PhasedClassicalSimState':
        """Initiate a classical simulation from a CompositeBloq.

        Args:
            cbloq: The composite bloq
            vals: A mapping of input register name to classical value to serve as inputs to the
                procedure.

        Returns:
            A new classical sim state.
        """
        return cls(signature=cbloq.signature, binst_graph=cbloq._binst_graph, vals=vals)

    def _binst_basis_state_phase(self, binst, in_vals):
        """Call `basis_state_phase` on a given bloq instance.

        If this method returns a value, the current phase will be updated. Otherwise, we
        leave the phase as-is.
        """
        bloq = binst.bloq
        bloq_phase = bloq.basis_state_phase(**in_vals)
        if bloq_phase is not None:
            _assert_valid_phase(bloq_phase)
            self.phase *= bloq_phase
        else:
            # Purely classical bloq; phase of 1
            pass


def call_cbloq_classically(
    signature: Signature,
    vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
    binst_graph: nx.DiGraph,
) -> Tuple[Dict[str, ClassicalValT], Dict[Soquet, ClassicalValT]]:
    """Propagate `on_classical_vals` calls through a composite bloq's contents.

    While we're handling the plumbing, we also do error checking on the arguments; see
    `_update_assign_from_vals`.

    Args:
        signature: The cbloq's signature for validating inputs
        vals: Mapping from register name to classical values
        binst_graph: The cbloq's binst graph.

    Returns:
        final_vals: A mapping from register name to output classical values
        soq_assign: An assignment from each soquet to its classical value. Soquets
            corresponding to thru registers will be mapped to the *output* classical
            value.
    """
    sim = ClassicalSimState(signature, binst_graph, vals)
    final_vals = sim.simulate()
    return final_vals, sim.soq_assign


def _assert_valid_phase(p: complex, atol: float = 1e-8):
    if np.abs(np.abs(p) - 1.0) > atol:
        raise ValueError(f"Phases must have unit modulus. Found {p}.")


def do_phased_classical_simulation(bloq: 'Bloq', vals: Mapping[str, 'ClassicalValT']):
    """Do a phased classical simulation of the bloq.

    This provides a simple interface to `PhasedClassicalSimState`. Advanced users
    may wish to use that class directly.

    Args:
        bloq: The bloq to simulate
        vals: A mapping from input register name to initial classical values. The initial phase is
            assumed to be 1.0.

    Returns:
        final_vals: A mapping of output register name to final classical values.
        phase: The final phase.
    """
    cbloq = bloq.as_composite_bloq()
    sim = PhasedClassicalSimState.from_cbloq(cbloq, vals=vals)
    final_vals = sim.simulate()
    phase = sim.phase
    return final_vals, phase


def get_classical_truth_table(
    bloq: 'Bloq',
) -> Tuple[List[str], List[str], List[Tuple[Sequence[Any], Sequence[Any]]]]:
    """Get a 'truth table' for a classical-reversible bloq.

    Args:
        bloq: The classical-reversible bloq to create a truth table for.

    Returns:
        in_names: The names of the left, input registers to serve as truth table headings for
            the input side of the truth table.
        out_names: The names of the right, output registers to serve as truth table headings
            for the output side of the truth table.
        truth_table: A list of table entries. Each entry is a tuple of (in_vals, out_vals).
            The vals sequences are ordered according to the `in_names` and `out_names` return
            values.
    """
    for reg in bloq.signature.lefts():
        if reg.shape:
            raise NotImplementedError()

    in_names: List[str] = []
    iters = []
    for reg in bloq.signature.lefts():
        in_names.append(reg.name)
        iters.append(reg.dtype.get_classical_domain())
    out_names: List[str] = [reg.name for reg in bloq.signature.rights()]

    truth_table: List[Tuple[Sequence[Any], Sequence[Any]]] = []
    for in_val_tuple in itertools.product(*iters):
        in_val_d = {name: val for name, val in zip(in_names, in_val_tuple)}
        out_val_tuple = bloq.call_classically(**in_val_d)
        # out_val_d = {name: val for name, val in zip(out_names, out_val_tuple)}
        truth_table.append((in_val_tuple, out_val_tuple))
    return in_names, out_names, truth_table


def format_classical_truth_table(
    in_names: Sequence[str],
    out_names: Sequence[str],
    truth_table: Sequence[Tuple[Sequence[Any], Sequence[Any]]],
) -> str:
    """Get a formatted tabular representation of the classical truth table."""
    heading = '  '.join(in_names) + '  |  ' + '  '.join(out_names) + '\n'
    heading += '-' * len(heading)
    entries = [
        ', '.join(f'{v}' for v in invals) + ' -> ' + ', '.join(f'{v}' for v in outvals)
        for invals, outvals in truth_table
    ]
    return '\n'.join([heading] + entries)


def add_ints(a: int, b: int, *, num_bits: Optional[int] = None, is_signed: bool = False) -> int:
    r"""Performs addition modulo $2^\mathrm{num\_bits}$ of (un)signed in a reversible way.

    Addition of signed integers can result in an overflow. In most classical programming languages (e.g. C++)
    what happens when an overflow happens is left as an implementation detail for compiler designers. However,
    for quantum subtraction, the operation should be unitary and that means that the unitary of the bloq should
    be a permutation matrix.

    If we hold `a` constant then the valid range of values of $b \in [-2^{\mathrm{num\_bits}-1}, 2^{\mathrm{num\_bits}-1})$
    gets shifted forward or backward by `a`. To keep the operation unitary overflowing values wrap around. This is the same
    as moving the range $2^\mathrm{num\_bits}$ by the same amount modulo $2^\mathrm{num\_bits}$. That is add
    $2^{\mathrm{num\_bits}-1})$ before addition modulo and then remove it.

    Args:
        a: left operand of addition.
        b: right operand of addition.
        num_bits: optional num_bits. When specified addition is done in the interval [0, 2**num_bits) or
            [-2**(num_bits-1), 2**(num_bits-1)) based on the value of `is_signed`.
        is_signed: boolean whether the numbers are unsigned or signed ints. This value is only used when
            `num_bits` is provided.
    """
    c = a + b
    if num_bits is not None:
        N = 2**num_bits
        if is_signed:
            return (c + N // 2) % N - N // 2
        return c % N
    return c
