#  Copyright 2026 Google LLC
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
"""Python bindings for qlt_fastsim."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from qualtran.simulation.classical_sim import ClassicalValT

if TYPE_CHECKING:
    import qualtran as qlt

from . import _rsqlt

_SUPPORTED_INT_DTYPES = frozenset(("QInt", "QUInt", "QBit", "QAny"))


def compile_l1_to_fastsim(module: _rsqlt.L1Module) -> _rsqlt.CompiledModule:
    """Compile an L1Module AST into a CompiledModule."""
    return _rsqlt.compile(module)


def _convert_output_value(
    val_str: str,
    dtype_str: str,
    name: str,
    n_bits: int,
    shape: Optional[List[int]],
) -> ClassicalValT:
    """Convert a string output value to its appropriate Python type.

    Args:
        val_str: The string representation of the output value.
        dtype_str: The dtype string (e.g. 'QInt', 'QUInt', 'QBit').
        name: The register name, for error messages.
        n_bits: Total number of bits for this register.
        shape: Shape dimensions for nd-array registers, or None for scalars.

    Returns:
        The converted value as int, or as a numpy ndarray for shaped registers.

    Raises:
        ValueError: If the dtype is not supported for conversion.
    """
    if dtype_str not in _SUPPORTED_INT_DTYPES:
        raise ValueError(
            f"Unsupported output dtype '{dtype_str}' for register '{name}'. "
            f"Cannot convert value '{val_str}' to a Python type."
        )

    if shape is None:
        return int(val_str)

    # Shaped register: split the flat integer into per-element values.
    total_elements = 1
    for d in shape:
        total_elements *= d
    assert n_bits % total_elements == 0, (
        f"Register '{name}' has {n_bits} bits but {total_elements} elements; "
        f"bits must be evenly divisible by element count."
    )
    element_bits = n_bits // total_elements

    flat_int = int(val_str)
    # Extract per-element values from the flat integer.
    # The Rust VM packs elements in row-major order with the first element
    # in the most-significant bits: value = e0 * 2^((N-1)*eb) + ... + e_{N-1}.
    flat_values = np.empty(total_elements, dtype=np.uint64)
    element_mask = (1 << element_bits) - 1
    for i in range(total_elements - 1, -1, -1):
        flat_values[i] = flat_int & element_mask
        flat_int >>= element_bits

    return flat_values.reshape(shape)


def _ndarray_to_bits(
    arr: np.ndarray,
    n_bits: int,
    shape: List[int],
    dtype: str,
    name: str,
) -> List[bool]:
    """Convert an np.ndarray input to a flat bit vector.

    The array is flattened in row-major (C) order and each element is
    converted to its bit representation. The resulting bit vectors are
    concatenated.

    Args:
        arr: Input numpy array.
        n_bits: Total number of bits for the register.
        shape: Expected shape dimensions from the register signature.
        dtype: The dtype string (e.g. 'QBit', 'QUInt').
        name: The register name, for error messages.

    Returns:
        A flat list of bools representing the full bit vector.

    Raises:
        ValueError: If the array shape doesn't match the register shape.
        TypeError: If the dtype is not supported for nd-array conversion.
    """
    expected_shape = tuple(shape)
    if arr.shape != expected_shape:
        raise ValueError(
            f"Array shape {arr.shape} for register '{name}' does not match "
            f"expected shape {expected_shape}."
        )

    total_elements = 1
    for d in shape:
        total_elements *= d
    assert n_bits % total_elements == 0, (
        f"Register '{name}' has {n_bits} bits but {total_elements} elements; "
        f"bits must be evenly divisible by element count."
    )
    element_bits = n_bits // total_elements

    if dtype not in _SUPPORTED_INT_DTYPES:
        raise TypeError(
            f"Unsupported dtype '{dtype}' for nd-array register '{name}'."
        )

    bits: List[bool] = []
    for val in arr.flat:
        elem_val = int(val)
        elem_bits = _rsqlt.int_to_bits(elem_val, element_bits)
        bits.extend(elem_bits)

    assert len(bits) == n_bits, (
        f"Expected {n_bits} bits for register '{name}', got {len(bits)}."
    )
    return bits


class QLTFastsim:
    """Virtual machine simulator for executing compiled Qualtran-L1 subroutines.

    This class wraps the Rust VM simulator and presents a stateless
    call-oriented interface. Each call to `call_classically` or
    `simulate` runs the full simulation and returns results in one
    shot. No internal state leaks between calls.
    """

    def __init__(self, compiled_module: _rsqlt.CompiledModule, entrypoint: str):
        self._compiled = compiled_module
        self._entrypoint = entrypoint
        self._sim = _rsqlt.VmSimulator(compiled_module, entrypoint)

        self._sig = compiled_module.get_subroutine_signature(entrypoint)
        self._inputs: Dict[str, Tuple[int, str, Optional[List[int]]]] = {}
        self._outputs: Dict[str, Tuple[int, str, Optional[List[int]]]] = {}
        for name, n_bits, dir_, dtype, shape in self._sig:
            if dir_ in ("Thru", "LeftOnly", "Cast"):
                self._inputs[name] = (n_bits, dtype, shape)
            if dir_ in ("Thru", "RightOnly", "Cast"):
                self._outputs[name] = (n_bits, dtype, shape)

    @classmethod
    def from_bloq(cls, bloq: "qlt.Bloq") -> "QLTFastsim":
        """Construct a QLTFastsim simulator for a specific bloq."""
        from qualtran.l1 import L1ModuleBuilder
        from . import nodes as rust_nodes

        l1_mb = L1ModuleBuilder(nodes=rust_nodes)
        root_bloq_key = l1_mb.add_bloqs(root=bloq)
        l1_mod = l1_mb.finalize()

        bc = compile_l1_to_fastsim(l1_mod)
        return cls(bc, root_bloq_key)

    def _prepare_inputs(self, kwargs: Dict[str, Any]) -> List[Tuple[str, List[bool]]]:
        """Validate and convert keyword arguments to bit-vector inputs.

        Args:
            kwargs: Input register values keyed by register name.

        Returns:
            A list of ``(name, bits)`` pairs in signature order.

        Raises:
            ValueError: If an unexpected or missing register is provided.
            TypeError: If an input value has an unsupported type.
        """
        for key in kwargs:
            if key not in self._inputs:
                raise ValueError(
                    f"Unexpected input register '{key}' "
                    f"for subroutine '{self._entrypoint}'"
                )

        input_values: List[Tuple[str, List[bool]]] = []
        for name, (n_bits, dtype, shape) in self._inputs.items():
            if name not in kwargs:
                raise ValueError(f"Missing required input register: '{name}'")
            val = kwargs[name]
            if isinstance(val, np.ndarray):
                if shape is None:
                    raise TypeError(
                        f"Got np.ndarray for scalar register '{name}'. "
                        f"Use an int or str instead."
                    )
                bits = _ndarray_to_bits(val, n_bits, shape, dtype, name)
            elif isinstance(val, int):
                bits = _rsqlt.int_to_bits(val, n_bits)
            elif isinstance(val, str):
                if dtype == "QInt":
                    bits = _rsqlt.signed_decimal_str_to_bits(val, n_bits)
                else:
                    bits = _rsqlt.decimal_str_to_bits(val, n_bits)
            elif isinstance(val, (list, tuple)):
                bits = list(val)
            else:
                raise TypeError(
                    f"Unsupported input type for register '{name}': {type(val)}"
                )
            input_values.append((name, bits))

        return input_values

    def call_classically(
        self, **kwargs: Any
    ) -> Tuple[ClassicalValT, ...]:
        """Execute a simulation run and return output values.

        Compatible with the `qualtran.Bloq.call_classically` interface:
        returns a tuple of output classical values in right-register
        (signature) order and requires the simulation to be phase-free.

        Args:
            **kwargs: Input register values keyed by register name. Values
                can be `int`, `str` (decimal), `list`/`tuple`
                of bools (raw bit vector), or `np.ndarray` for shaped
                registers.

        Returns:
            A tuple of output classical values in right-register order.
            Shaped registers are returned as ``np.ndarray``.

        Raises:
            ValueError: If the simulation accumulates a non-zero phase exponent,
                or if an unexpected or missing register is provided.
            TypeError: If an input value has an unsupported type.
        """
        input_values = self._prepare_inputs(kwargs)
        raw_outputs, phase_exponent = self._sim.execute_run(input_values)

        if phase_exponent != 0.0:
            raise ValueError(
                f"call_classically is for phase-free simulation only, "
                f"but the simulation accumulated a non-zero phase exponent "
                f"of {phase_exponent}. Use simulate() instead."
            )

        results = []
        for name, val_str, dtype_str in raw_outputs:
            out_n_bits, _, out_shape = self._outputs[name]
            results.append(
                _convert_output_value(val_str, dtype_str, name, out_n_bits, out_shape)
            )
        return tuple(results)

    def simulate(
        self, **kwargs: Any
    ) -> Tuple[Dict[str, ClassicalValT], float]:
        """Execute a simulation run and return outputs as a dict with phase exponent.

        Unlike `call_classically`, this method does not require the
        simulation to be phase-free and returns outputs keyed by register
        name.

        Args:
            **kwargs: Input register values keyed by register name. Values
                can be `int`, `str` (decimal), `list`/`tuple`
                of bools (raw bit vector), or `np.ndarray` for shaped
                registers.

        Returns:
            A tuple `(outputs, phase_exponent)` where `outputs` is a dict
            mapping output register names to their classical values and
            `phase_exponent` is the accumulated value `x` such that the
            global phase is `exp(iπ * x)`. Shaped registers are returned
            as ``np.ndarray``.

        Raises:
            ValueError: If an unexpected or missing register is provided.
            TypeError: If an input value has an unsupported type.
        """
        input_values = self._prepare_inputs(kwargs)
        raw_outputs, phase_exponent = self._sim.execute_run(input_values)

        outputs: Dict[str, ClassicalValT] = {}
        for name, val_str, dtype_str in raw_outputs:
            out_n_bits, _, out_shape = self._outputs[name]
            outputs[name] = _convert_output_value(
                val_str, dtype_str, name, out_n_bits, out_shape
            )

        return outputs, phase_exponent
