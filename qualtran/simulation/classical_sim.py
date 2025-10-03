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
import abc
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

import attrs
import networkx as nx
import numpy as np
import sympy
from numpy.typing import NDArray
from typing_extensions import Self

from qualtran import (
    Bloq,
    BloqInstance,
    DanglingT,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    LeftDangle,
    Register,
    RightDangle,
    Signature,
    Soquet,
)
from qualtran._infra.composite_bloq import _binst_to_cxns, _get_soquet

if TYPE_CHECKING:
    from qualtran import CompositeBloq, QCDType

ClassicalValT = Union[int, np.integer, NDArray[np.integer]]
ClassicalValRetT = Union[int, np.integer, NDArray[np.integer], 'ClassicalValDistribution']


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


@attrs.frozen(hash=False)
class ClassicalValDistribution:
    """This class represents a distribution of classical values.

    Use this if the bloq has performed a measurement or other projection
    that has resulted in a mixed state of purely classical values.

    Args:
        a: An array of choices, or `np.arange` if an integer is given.
            This is the `a` parameter to `np.random.Generator.choice()`.
        p: An array of probabilities. If not supplied, the uniform distribution is assumed.
            This is the `p` parameter to `np.random.Generator.choice()`.
    """

    a: Union[int, np.typing.ArrayLike]
    p: Optional[np.typing.ArrayLike] = None


class _ClassicalValHandler(metaclass=abc.ABCMeta):
    """An internal class for returning a random classical value.

    Implmentors should write the get() function which returns a random
    choice of values."""

    @abc.abstractmethod
    def get(self, binst: 'BloqInstance', distribution: ClassicalValDistribution) -> Any: ...


class _RandomClassicalValHandler(_ClassicalValHandler):
    """Returns a random classical value using a random number generator."""

    def __init__(self, rng: 'np.random.Generator'):
        self._gen = rng

    def get(self, binst, distribution: ClassicalValDistribution):
        return self._gen.choice(distribution.a, p=distribution.p)  # type:ignore[arg-type]


class _FixedClassicalValHandler(_ClassicalValHandler):
    """Returns a random classical value using a fixed value per bloq instance.

    Useful for deterministic testing.

    Args:
        binst_i_to_val: mapping from BloqInstance.i instance indices
            to the fixed classical value.
    """

    def __init__(self, binst_i_to_val: Dict[int, Any]):
        self._binst_i_to_val = binst_i_to_val

    def get(self, binst, distribution: ClassicalValDistribution):
        return self._binst_i_to_val[binst.i]


class _BannedClassicalValHandler(_ClassicalValHandler):
    """Used when random classical value is not able to be performed."""

    def get(self, binst: 'BloqInstance', distribution: ClassicalValDistribution) -> Any:
        raise ValueError(
            f"{binst} has non-deterministic classical action."
            "Cannot simulate with classical values."
        )


@attrs.frozen
class MeasurementPhase:
    """Sentinel value for phases based on measurement outcomes:

    This can be returned from `Bloq.basis_state_phase`
    if a phase should be applied based on a measurement outcome.
    This can be used in special circumstances to verify measurement-based uncomputation (MBUC).

    Args:
        reg_name: Name of the register
        idx: Index of the register wire(s).
    """

    reg_name: str
    idx: Tuple[int, ...] = ()


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
        random_handler: The classical random number handler to use for use in
            measurement-based outcomes (e.g. MBUC).

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
        random_handler: '_ClassicalValHandler' = _BannedClassicalValHandler(),
    ):
        self._signature = signature
        self._binst_graph = binst_graph
        self._binst_iter = nx.topological_sort(self._binst_graph)
        self._random_handler = random_handler

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
                if isinstance(val, ClassicalValDistribution):
                    val = self._random_handler.get(binst, val)

                reg.dtype.assert_valid_classical_val(val, debug_str)
                soq = Soquet(binst, reg)
                self.soq_assign[soq] = val

    def _recurse_impl(self, cbloq: 'CompositeBloq', in_vals):
        """Overridable function to recursively simulate a composite bloq."""
        out_vals, _ = call_cbloq_classically(cbloq.signature, in_vals, cbloq._binst_graph)
        phase = None

        return out_vals, phase

    def _recurse(self, binst: 'BloqInstance', in_vals) -> Self:
        """Recursively simulate a composite bloq.

        This handles decomposing the bloq and using the results of the sub-simulation
        to update the simulator state. Developers should override `_recurse_impl` to
        customize the sub-simulation.
        """
        bloq = binst.bloq
        try:
            cbloq = bloq.decompose_bloq()
            out_vals, bloq_phase = self._recurse_impl(cbloq, in_vals)

        except DecomposeTypeError as e:
            raise NotImplementedError(f"{bloq} is not classically simulable.") from e
        except DecomposeNotImplementedError as e:
            raise NotImplementedError(
                f"{bloq} has no decomposition and does not "
                f"support classical simulation directly"
            ) from e
        except NotImplementedError as e:
            raise NotImplementedError(f"{bloq} does not support classical simulation: {e}") from e

        self._update(binst, out_vals, bloq_phase)
        return self

    def _update(self, binst: 'BloqInstance', out_vals, bloq_phase: Union[complex, None]) -> None:
        """Overridable method to update the current simulator state."""
        self._update_assign_from_vals(binst.bloq.signature.rights(), binst, out_vals)

        if bloq_phase is not None:
            raise ValueError(
                f"{binst.bloq} imparts a phase of {bloq_phase}, and can't be simulated purely classically. "
                f"Consider using `do_phased_classical_simulation`."
            )

    def step(self) -> Self:
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
        bcls_name = bloq.__class__.__name__
        in_vals = {reg.name: _in_vals(reg) for reg in bloq.signature.lefts()}
        out_vals = bloq.on_classical_vals(**in_vals)
        bloq_phase = bloq.basis_state_phase(**in_vals)

        #                     +--+ basis_state_phase
        #   +- on_classical_vals
        #   |                 |
        # dict              None           Use classical values.
        # dict              number         Use classical values and phase only if doing phased sim
        # NotImplemented    None           decompose and use the correct simulator type
        # NotImplemented    number         error

        if out_vals is NotImplemented:
            if bloq_phase is not None:
                raise ValueError(
                    f"`basis_state_phase` defined on {bcls_name}, but not `on_classical_vals`"
                )

            return self._recurse(binst, in_vals)

        if not isinstance(out_vals, dict):
            raise TypeError(f"`{bcls_name}.on_classical_vals` should return a dictionary.")

        self._update(binst, out_vals, bloq_phase)

        return self

    def finalize(self) -> Dict[str, 'ClassicalValT']:
        """Finish simulating a composite bloq and extract final values.

        Returns:
            final_vals: The final classical values, keyed by the RIGHT register names of the
                composite bloq.

        Raises:
            KeyError: if `.step()` has not been called for each bloq instance.
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

    This simulation scheme supports a class of circuits containing only
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
        random_handler: The classical random number handler to use for use in
            measurement-based outcomes (e.g. MBUC).
    """

    def __init__(
        self,
        signature: 'Signature',
        binst_graph: nx.DiGraph,
        vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
        *,
        phase: complex = 1.0,
        random_handler: '_ClassicalValHandler',
    ):
        super().__init__(
            signature=signature, binst_graph=binst_graph, vals=vals, random_handler=random_handler
        )
        _assert_valid_phase(phase)
        self.phase = phase

    @classmethod
    def from_cbloq(
        cls,
        cbloq: 'CompositeBloq',
        vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
        rng: Optional['np.random.Generator'] = None,
        fixed_random_vals: Optional[Dict[int, Any]] = None,
    ) -> 'PhasedClassicalSimState':
        """Initiate a classical simulation from a CompositeBloq.

        Args:
            cbloq: The composite bloq
            vals: A mapping of input register name to classical value to serve as inputs to the
                procedure.
            rng: A random number generator to use for classical random values, such a np.random.
            fixed_random_vals: A dictionary of bloq instances to values to perform fixed calculation
                for classical values.

        Returns:
            A new classical sim state.
        """
        rnd_handler: _ClassicalValHandler
        if rng is not None:
            rnd_handler = _RandomClassicalValHandler(rng=rng)
        elif fixed_random_vals is not None:
            rnd_handler = _FixedClassicalValHandler(binst_i_to_val=fixed_random_vals)
        else:
            rnd_handler = _BannedClassicalValHandler()
        return cls(
            signature=cbloq.signature,
            binst_graph=cbloq._binst_graph,
            vals=vals,
            random_handler=rnd_handler,
        )

    def _recurse_impl(self, cbloq, in_vals):
        """Use phased classical simulation when recursing."""
        sim = PhasedClassicalSimState(
            cbloq.signature, cbloq._binst_graph, in_vals, random_handler=self._random_handler
        )
        final_vals = sim.simulate()
        phase = sim.phase
        return final_vals, phase

    def _update(self, binst: 'BloqInstance', out_vals, bloq_phase: Union[complex, None]) -> None:
        """Update the current simulator state, including phase tracking."""
        self._update_assign_from_vals(binst.bloq.signature.rights(), binst, out_vals)

        if isinstance(bloq_phase, MeasurementPhase):
            # In this special case, there is a coupling between the classical result and the
            # phase result (because the classical result is stochastic). We look up the measurement
            # result and apply a phase if it is `1`.
            meas_result = self.soq_assign[
                _get_soquet(
                    binst=binst,
                    reg_name=bloq_phase.reg_name,
                    right=True,
                    idx=bloq_phase.idx,
                    binst_graph=self._binst_graph,
                )
            ]
            if meas_result == 1:
                # Measurement result of 1, phase of -1
                self.phase *= -1.0
            else:
                # Measurement result of 0, phase of +1
                pass
        elif bloq_phase is not None:
            _assert_valid_phase(bloq_phase)
            self.phase *= bloq_phase
        else:
            # Purely classical bloq; phase of 1
            pass


def call_cbloq_classically(
    signature: Signature,
    vals: Mapping[str, Union[sympy.Symbol, ClassicalValT]],
    binst_graph: nx.DiGraph,
    random_handler: '_ClassicalValHandler' = _RandomClassicalValHandler(
        rng=np.random.default_rng()
    ),
) -> Tuple[Dict[str, ClassicalValT], Dict[Soquet, ClassicalValT]]:
    """Propagate `on_classical_vals` calls through a composite bloq's contents.

    While we're handling the plumbing, we also do error checking on the arguments; see
    `_update_assign_from_vals`.

    Args:
        signature: The cbloq's signature for validating inputs
        vals: Mapping from register name to classical values
        binst_graph: The cbloq's binst graph.
        random_handler: The classical random number handler to use for use in
            measurement-based outcomes (e.g. MBUC).

    Returns:
        final_vals: A mapping from register name to output classical values
        soq_assign: An assignment from each soquet to its classical value. Soquets
            corresponding to thru registers will be mapped to the *output* classical
            value.
    """
    sim = ClassicalSimState(signature, binst_graph, vals, random_handler)
    final_vals = sim.simulate()
    return final_vals, sim.soq_assign


def _assert_valid_phase(p: complex, atol: float = 1e-8):
    if np.abs(np.abs(p) - 1.0) > atol:
        raise ValueError(f"Phases must have unit modulus. Found {p}.")


def do_phased_classical_simulation(
    bloq: 'Bloq',
    vals: Mapping[str, 'ClassicalValT'],
    rng: Optional['np.random.Generator'] = None,
    fixed_random_vals: Optional[Dict[int, Any]] = None,
) -> Tuple[Dict[str, 'ClassicalValT'], complex]:
    """Do a phased classical simulation of the bloq.

    This provides a simple interface to `PhasedClassicalSimState`. Advanced users
    may wish to use that class directly.

    Args:
        bloq: The bloq to simulate
        vals: A mapping from input register name to initial classical values. The initial phase is
            assumed to be 1.0.
        rng: A numpy random generator (e.g. from `np.random.default_rng()`). This function
            will use this generator to supply random values from certain phased-classical operations
            like `MeasureX`. If not supplied, classical measurements will use a random value.
        fixed_random_vals: A dictionary of instance to values to perform fixed calculation
                for classical values.

    Returns:
        final_vals: A mapping of output register name to final classical values.
        phase: The final phase.
    """
    cbloq = bloq.as_composite_bloq()
    sim = PhasedClassicalSimState.from_cbloq(
        cbloq, vals=vals, rng=rng, fixed_random_vals=fixed_random_vals
    )
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
    r"""Classically performs addition modulo $2^n$ of two integers in a reversible way.

    Addition of integers can result in an overflow. In C/C++, overflow behavior is left as an
    implementation detail for compiler designers. However, for quantum programs, the operation
    must be unitary (i.e. reversible). To keep the operation unitary, overflowing values wrap
    around.

    Args:
        a: left operand of addition.
        b: right operand of addition.
        num_bits: When specified, addition is done in the interval `[0, 2**num_bits)` or
            `[-2**(num_bits-1), 2**(num_bits-1))` based on the value of `is_signed`. Otherwise,
            arbitrary-precision Python integer addition is performed.
        is_signed: Whether the numbers are unsigned or signed ints. This value is only used when
            `num_bits` is provided.
    """
    c = a + b
    if num_bits is not None:
        N = 2**num_bits
        if is_signed:
            return (c + N // 2) % N - N // 2
        return c % N
    return c
