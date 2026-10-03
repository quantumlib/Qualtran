# kickmix v0.1 API Reference

## Index
- [`kickmix.Circuit`](#kickmix.Circuit)
    - [`kickmix.Circuit.__add__`](#kickmix.Circuit.__add__)
    - [`kickmix.Circuit.__eq__`](#kickmix.Circuit.__eq__)
    - [`kickmix.Circuit.__init__`](#kickmix.Circuit.__init__)
    - [`kickmix.Circuit.__len__`](#kickmix.Circuit.__len__)
    - [`kickmix.Circuit.__mul__`](#kickmix.Circuit.__mul__)
    - [`kickmix.Circuit.__ne__`](#kickmix.Circuit.__ne__)
    - [`kickmix.Circuit.__repr__`](#kickmix.Circuit.__repr__)
    - [`kickmix.Circuit.__rmul__`](#kickmix.Circuit.__rmul__)
    - [`kickmix.Circuit.__str__`](#kickmix.Circuit.__str__)
    - [`kickmix.Circuit.html_diagram`](#kickmix.Circuit.html_diagram)
    - [`kickmix.Circuit.max_magic`](#kickmix.Circuit.max_magic)
    - [`kickmix.Circuit.num_bits`](#kickmix.Circuit.num_bits)
    - [`kickmix.Circuit.num_qubits`](#kickmix.Circuit.num_qubits)
    - [`kickmix.Circuit.num_registers`](#kickmix.Circuit.num_registers)
    - [`kickmix.Circuit.reaction_depth`](#kickmix.Circuit.reaction_depth)
    - [`kickmix.Circuit.register_data`](#kickmix.Circuit.register_data)
    - [`kickmix.Circuit.text_diagram`](#kickmix.Circuit.text_diagram)
- [`kickmix.CircuitBuilder`](#kickmix.CircuitBuilder)
    - [`kickmix.CircuitBuilder.__init__`](#kickmix.CircuitBuilder.__init__)
    - [`kickmix.CircuitBuilder.alloc_bits`](#kickmix.CircuitBuilder.alloc_bits)
    - [`kickmix.CircuitBuilder.alloc_qubits`](#kickmix.CircuitBuilder.alloc_qubits)
    - [`kickmix.CircuitBuilder.bit_store`](#kickmix.CircuitBuilder.bit_store)
    - [`kickmix.CircuitBuilder.ccx`](#kickmix.CircuitBuilder.ccx)
    - [`kickmix.CircuitBuilder.create_classical_register`](#kickmix.CircuitBuilder.create_classical_register)
    - [`kickmix.CircuitBuilder.create_quantum_register`](#kickmix.CircuitBuilder.create_quantum_register)
    - [`kickmix.CircuitBuilder.cswap`](#kickmix.CircuitBuilder.cswap)
    - [`kickmix.CircuitBuilder.cx`](#kickmix.CircuitBuilder.cx)
    - [`kickmix.CircuitBuilder.cz`](#kickmix.CircuitBuilder.cz)
    - [`kickmix.CircuitBuilder.debug_print`](#kickmix.CircuitBuilder.debug_print)
    - [`kickmix.CircuitBuilder.del_and`](#kickmix.CircuitBuilder.del_and)
    - [`kickmix.CircuitBuilder.del_gf2_div`](#kickmix.CircuitBuilder.del_gf2_div)
    - [`kickmix.CircuitBuilder.del_gf2_div_with_scaffold`](#kickmix.CircuitBuilder.del_gf2_div_with_scaffold)
    - [`kickmix.CircuitBuilder.del_gf2_inverse`](#kickmix.CircuitBuilder.del_gf2_inverse)
    - [`kickmix.CircuitBuilder.del_gf2_inverse_with_scaffold`](#kickmix.CircuitBuilder.del_gf2_inverse_with_scaffold)
    - [`kickmix.CircuitBuilder.del_gf2_mul`](#kickmix.CircuitBuilder.del_gf2_mul)
    - [`kickmix.CircuitBuilder.del_lookup`](#kickmix.CircuitBuilder.del_lookup)
    - [`kickmix.CircuitBuilder.end_auto_free_scope`](#kickmix.CircuitBuilder.end_auto_free_scope)
    - [`kickmix.CircuitBuilder.finish_circuit`](#kickmix.CircuitBuilder.finish_circuit)
    - [`kickmix.CircuitBuilder.flame_chart_svg`](#kickmix.CircuitBuilder.flame_chart_svg)
    - [`kickmix.CircuitBuilder.flip_if_equal`](#kickmix.CircuitBuilder.flip_if_equal)
    - [`kickmix.CircuitBuilder.flip_if_greater_than`](#kickmix.CircuitBuilder.flip_if_greater_than)
    - [`kickmix.CircuitBuilder.flip_if_less_than`](#kickmix.CircuitBuilder.flip_if_less_than)
    - [`kickmix.CircuitBuilder.free`](#kickmix.CircuitBuilder.free)
    - [`kickmix.CircuitBuilder.gf2_iadd`](#kickmix.CircuitBuilder.gf2_iadd)
    - [`kickmix.CircuitBuilder.gf2_idiv`](#kickmix.CircuitBuilder.gf2_idiv)
    - [`kickmix.CircuitBuilder.gf2_imul`](#kickmix.CircuitBuilder.gf2_imul)
    - [`kickmix.CircuitBuilder.gf2_phase_by_product`](#kickmix.CircuitBuilder.gf2_phase_by_product)
    - [`kickmix.CircuitBuilder.hmr`](#kickmix.CircuitBuilder.hmr)
    - [`kickmix.CircuitBuilder.iadd`](#kickmix.CircuitBuilder.iadd)
    - [`kickmix.CircuitBuilder.init_and`](#kickmix.CircuitBuilder.init_and)
    - [`kickmix.CircuitBuilder.init_gf2_div`](#kickmix.CircuitBuilder.init_gf2_div)
    - [`kickmix.CircuitBuilder.init_gf2_div_with_scaffold`](#kickmix.CircuitBuilder.init_gf2_div_with_scaffold)
    - [`kickmix.CircuitBuilder.init_gf2_inverse`](#kickmix.CircuitBuilder.init_gf2_inverse)
    - [`kickmix.CircuitBuilder.init_gf2_inverse_with_scaffold`](#kickmix.CircuitBuilder.init_gf2_inverse_with_scaffold)
    - [`kickmix.CircuitBuilder.init_gf2_mul`](#kickmix.CircuitBuilder.init_gf2_mul)
    - [`kickmix.CircuitBuilder.init_lookup`](#kickmix.CircuitBuilder.init_lookup)
    - [`kickmix.CircuitBuilder.isub`](#kickmix.CircuitBuilder.isub)
    - [`kickmix.CircuitBuilder.ixor_gf2_div`](#kickmix.CircuitBuilder.ixor_gf2_div)
    - [`kickmix.CircuitBuilder.ixor_gf2_mul`](#kickmix.CircuitBuilder.ixor_gf2_mul)
    - [`kickmix.CircuitBuilder.left_rotate`](#kickmix.CircuitBuilder.left_rotate)
    - [`kickmix.CircuitBuilder.max_qubits`](#kickmix.CircuitBuilder.max_qubits)
    - [`kickmix.CircuitBuilder.multi_controlled_x`](#kickmix.CircuitBuilder.multi_controlled_x)
    - [`kickmix.CircuitBuilder.neg`](#kickmix.CircuitBuilder.neg)
    - [`kickmix.CircuitBuilder.neg_if`](#kickmix.CircuitBuilder.neg_if)
    - [`kickmix.CircuitBuilder.num_allocated_qubits`](#kickmix.CircuitBuilder.num_allocated_qubits)
    - [`kickmix.CircuitBuilder.num_free_qubits`](#kickmix.CircuitBuilder.num_free_qubits)
    - [`kickmix.CircuitBuilder.pop_condition`](#kickmix.CircuitBuilder.pop_condition)
    - [`kickmix.CircuitBuilder.push_condition`](#kickmix.CircuitBuilder.push_condition)
    - [`kickmix.CircuitBuilder.reset`](#kickmix.CircuitBuilder.reset)
    - [`kickmix.CircuitBuilder.right_rotate`](#kickmix.CircuitBuilder.right_rotate)
    - [`kickmix.CircuitBuilder.set_max_qubits`](#kickmix.CircuitBuilder.set_max_qubits)
    - [`kickmix.CircuitBuilder.start_auto_free_scope`](#kickmix.CircuitBuilder.start_auto_free_scope)
    - [`kickmix.CircuitBuilder.swap`](#kickmix.CircuitBuilder.swap)
    - [`kickmix.CircuitBuilder.unary_iteration`](#kickmix.CircuitBuilder.unary_iteration)
    - [`kickmix.CircuitBuilder.x`](#kickmix.CircuitBuilder.x)
    - [`kickmix.CircuitBuilder.z`](#kickmix.CircuitBuilder.z)
    - [`kickmix.CircuitBuilder.z_pow`](#kickmix.CircuitBuilder.z_pow)
- [`kickmix.GF2Field`](#kickmix.GF2Field)
    - [`kickmix.GF2Field.__eq__`](#kickmix.GF2Field.__eq__)
    - [`kickmix.GF2Field.__hash__`](#kickmix.GF2Field.__hash__)
    - [`kickmix.GF2Field.__init__`](#kickmix.GF2Field.__init__)
    - [`kickmix.GF2Field.__repr__`](#kickmix.GF2Field.__repr__)
    - [`kickmix.GF2Field.__str__`](#kickmix.GF2Field.__str__)
    - [`kickmix.GF2Field.add`](#kickmix.GF2Field.add)
    - [`kickmix.GF2Field.degree`](#kickmix.GF2Field.degree)
    - [`kickmix.GF2Field.div`](#kickmix.GF2Field.div)
    - [`kickmix.GF2Field.frobenius`](#kickmix.GF2Field.frobenius)
    - [`kickmix.GF2Field.from_galois`](#kickmix.GF2Field.from_galois)
    - [`kickmix.GF2Field.invert`](#kickmix.GF2Field.invert)
    - [`kickmix.GF2Field.is_element`](#kickmix.GF2Field.is_element)
    - [`kickmix.GF2Field.is_irreducible`](#kickmix.GF2Field.is_irreducible)
    - [`kickmix.GF2Field.is_primitive_element`](#kickmix.GF2Field.is_primitive_element)
    - [`kickmix.GF2Field.mod`](#kickmix.GF2Field.mod)
    - [`kickmix.GF2Field.modulus`](#kickmix.GF2Field.modulus)
    - [`kickmix.GF2Field.mul`](#kickmix.GF2Field.mul)
    - [`kickmix.GF2Field.pow`](#kickmix.GF2Field.pow)
    - [`kickmix.GF2Field.primitive_element`](#kickmix.GF2Field.primitive_element)
    - [`kickmix.GF2Field.square`](#kickmix.GF2Field.square)
- [`kickmix.Simulator`](#kickmix.Simulator)
    - [`kickmix.Simulator.__init__`](#kickmix.Simulator.__init__)
    - [`kickmix.Simulator.batch_size`](#kickmix.Simulator.batch_size)
    - [`kickmix.Simulator.clear_for_shot`](#kickmix.Simulator.clear_for_shot)
    - [`kickmix.Simulator.do`](#kickmix.Simulator.do)
    - [`kickmix.Simulator.read_across_shots`](#kickmix.Simulator.read_across_shots)
    - [`kickmix.Simulator.read_phase_flipped_across_shots`](#kickmix.Simulator.read_phase_flipped_across_shots)
    - [`kickmix.Simulator.read_shot_phase`](#kickmix.Simulator.read_shot_phase)
    - [`kickmix.Simulator.read_within_shot`](#kickmix.Simulator.read_within_shot)
    - [`kickmix.Simulator.use_same_registers_as`](#kickmix.Simulator.use_same_registers_as)
    - [`kickmix.Simulator.write_across_shots`](#kickmix.Simulator.write_across_shots)
    - [`kickmix.Simulator.write_shot_phase`](#kickmix.Simulator.write_shot_phase)
    - [`kickmix.Simulator.write_within_shot`](#kickmix.Simulator.write_within_shot)
- [`kickmix.UnaryIterationCursor`](#kickmix.UnaryIterationCursor)
    - [`kickmix.UnaryIterationCursor.__enter__`](#kickmix.UnaryIterationCursor.__enter__)
    - [`kickmix.UnaryIterationCursor.__exit__`](#kickmix.UnaryIterationCursor.__exit__)
    - [`kickmix.UnaryIterationCursor.__init__`](#kickmix.UnaryIterationCursor.__init__)
    - [`kickmix.UnaryIterationCursor.__iter__`](#kickmix.UnaryIterationCursor.__iter__)
    - [`kickmix.UnaryIterationCursor.__next__`](#kickmix.UnaryIterationCursor.__next__)
    - [`kickmix.UnaryIterationCursor.__repr__`](#kickmix.UnaryIterationCursor.__repr__)
    - [`kickmix.UnaryIterationCursor.close`](#kickmix.UnaryIterationCursor.close)
    - [`kickmix.UnaryIterationCursor.cur_address_value`](#kickmix.UnaryIterationCursor.cur_address_value)
    - [`kickmix.UnaryIterationCursor.match_qubit`](#kickmix.UnaryIterationCursor.match_qubit)
    - [`kickmix.UnaryIterationCursor.move_to`](#kickmix.UnaryIterationCursor.move_to)
- [`kickmix.array`](#kickmix.array)
    - [`kickmix.array.__add__`](#kickmix.array.__add__)
    - [`kickmix.array.__eq__`](#kickmix.array.__eq__)
    - [`kickmix.array.__getitem__`](#kickmix.array.__getitem__)
    - [`kickmix.array.__len__`](#kickmix.array.__len__)
    - [`kickmix.array.__ne__`](#kickmix.array.__ne__)
    - [`kickmix.array.__repr__`](#kickmix.array.__repr__)
    - [`kickmix.array.__str__`](#kickmix.array.__str__)
- [`kickmix.b`](#kickmix.b)
    - [`kickmix.b.__eq__`](#kickmix.b.__eq__)
    - [`kickmix.b.__hash__`](#kickmix.b.__hash__)
    - [`kickmix.b.__init__`](#kickmix.b.__init__)
    - [`kickmix.b.__lt__`](#kickmix.b.__lt__)
    - [`kickmix.b.__ne__`](#kickmix.b.__ne__)
    - [`kickmix.b.__repr__`](#kickmix.b.__repr__)
    - [`kickmix.b.__str__`](#kickmix.b.__str__)
    - [`kickmix.b.id`](#kickmix.b.id)
- [`kickmix.q`](#kickmix.q)
    - [`kickmix.q.__eq__`](#kickmix.q.__eq__)
    - [`kickmix.q.__hash__`](#kickmix.q.__hash__)
    - [`kickmix.q.__init__`](#kickmix.q.__init__)
    - [`kickmix.q.__lt__`](#kickmix.q.__lt__)
    - [`kickmix.q.__ne__`](#kickmix.q.__ne__)
    - [`kickmix.q.__repr__`](#kickmix.q.__repr__)
    - [`kickmix.q.__str__`](#kickmix.q.__str__)
    - [`kickmix.q.id`](#kickmix.q.id)
- [`kickmix.r`](#kickmix.r)
    - [`kickmix.r.__eq__`](#kickmix.r.__eq__)
    - [`kickmix.r.__hash__`](#kickmix.r.__hash__)
    - [`kickmix.r.__init__`](#kickmix.r.__init__)
    - [`kickmix.r.__ne__`](#kickmix.r.__ne__)
    - [`kickmix.r.__repr__`](#kickmix.r.__repr__)
    - [`kickmix.r.__str__`](#kickmix.r.__str__)
    - [`kickmix.r.id`](#kickmix.r.id)
- [`kickmix.xb`](#kickmix.xb)
    - [`kickmix.xb.__eq__`](#kickmix.xb.__eq__)
    - [`kickmix.xb.__hash__`](#kickmix.xb.__hash__)
    - [`kickmix.xb.__init__`](#kickmix.xb.__init__)
    - [`kickmix.xb.__lt__`](#kickmix.xb.__lt__)
    - [`kickmix.xb.__ne__`](#kickmix.xb.__ne__)
    - [`kickmix.xb.__repr__`](#kickmix.xb.__repr__)
    - [`kickmix.xb.__str__`](#kickmix.xb.__str__)
    - [`kickmix.xb.id`](#kickmix.xb.id)
- [`kickmix.xbool`](#kickmix.xbool)
    - [`kickmix.xbool.__eq__`](#kickmix.xbool.__eq__)
    - [`kickmix.xbool.__hash__`](#kickmix.xbool.__hash__)
    - [`kickmix.xbool.__init__`](#kickmix.xbool.__init__)
    - [`kickmix.xbool.__lt__`](#kickmix.xbool.__lt__)
    - [`kickmix.xbool.__ne__`](#kickmix.xbool.__ne__)
    - [`kickmix.xbool.__repr__`](#kickmix.xbool.__repr__)
```python
# Types used by the method definitions.
from typing import overload, TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import io
import pathlib
import numpy as np
```

<a name="kickmix.Circuit"></a>
```python
# kickmix.Circuit

# (at top-level in the kickmix module)
class Circuit:
    """A kickmix circuit.
    """
```

<a name="kickmix.Circuit.__add__"></a>
```python
# kickmix.Circuit.__add__

# (in class kickmix.Circuit)
def __add__(
    self,
    arg0: kickmix.Circuit,
) -> kickmix.Circuit:
    """Returns the concatenation of two circuits.

    Note: register instructions are not part of the concatenation.
    This method arbitrarily chooses the register data of the result to correspond
    to the register data of the left hand side of the addition, unless the left
    hand side has no register data, in which case the right hand side is used.

    Examples:
        >>> import kickmix as km
        >>> km.Circuit('CX q0 q1') + km.Circuit('Z q2')
        km.Circuit('''
            CX q0 q1
            Z q2
        ''')

        >>> a = km.Circuit('''
        ...     REGISTER r0 "test"
        ...     APPEND_TO_REGISTER q0 r0
        ...     APPEND_TO_REGISTER q1 r0
        ...     APPEND_TO_REGISTER q2 r0
        ...     CCX q0 q1 q2
        ... ''')
        >>> b = km.Circuit('''
        ...     CZ q0 q1
        ... ''')
        >>> a + b
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            APPEND_TO_REGISTER q1 r0
            APPEND_TO_REGISTER q2 r0
            REGISTER r0 "test"
            CCX q0 q1 q2
            CZ q0 q1
        ''')
        >>> b + a
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            APPEND_TO_REGISTER q1 r0
            APPEND_TO_REGISTER q2 r0
            REGISTER r0 "test"
            CZ q0 q1
            CCX q0 q1 q2
        ''')
        >>> a + a
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            APPEND_TO_REGISTER q1 r0
            APPEND_TO_REGISTER q2 r0
            REGISTER r0 "test"
            CCX q0 q1 q2
            CCX q0 q1 q2
        ''')
        >>> b + b
        km.Circuit('''
            CZ q0 q1
            CZ q0 q1
        ''')
    """
```

<a name="kickmix.Circuit.__eq__"></a>
```python
# kickmix.Circuit.__eq__

# (in class kickmix.Circuit)
def __eq__(
    self,
    arg0: kickmix.Circuit,
) -> bool:
    """Determines if two circuits have identical instructions.
    """
```

<a name="kickmix.Circuit.__init__"></a>
```python
# kickmix.Circuit.__init__

# (in class kickmix.Circuit)
def __init__(
    self,
    circuit_text: str = '',
) -> None:
    """Creates a kickmix Circuit by parsing the given text.
    """
```

<a name="kickmix.Circuit.__len__"></a>
```python
# kickmix.Circuit.__len__

# (in class kickmix.Circuit)
def __len__(
    self,
) -> int:
    """Returns the number of instructions in the circuit.
    """
```

<a name="kickmix.Circuit.__mul__"></a>
```python
# kickmix.Circuit.__mul__

# (in class kickmix.Circuit)
def __mul__(
    self,
    arg0: int,
) -> kickmix.Circuit:
    """Repeats the contents of a circuit the given number of times.

    Note: register data is not repeated.

    Examples:
        >>> import kickmix as km
        >>> km.Circuit('CX q0 q1') * 3
        km.Circuit('''
            CX q0 q1
            CX q0 q1
            CX q0 q1
        ''')

        >>> km.Circuit('CX q0 q1') * 0
        km.Circuit('''
        ''')

        >>> 5 * km.Circuit('''
        ...     REGISTER r0 "test"
        ...     APPEND_TO_REGISTER q0 r0
        ...     APPEND_TO_REGISTER q1 r0
        ...     APPEND_TO_REGISTER q2 r0
        ...     CCX q0 q1 q2
        ...     X q0
        ... ''')
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            APPEND_TO_REGISTER q1 r0
            APPEND_TO_REGISTER q2 r0
            REGISTER r0 "test"
            CCX q0 q1 q2
            X q0
            CCX q0 q1 q2
            X q0
            CCX q0 q1 q2
            X q0
            CCX q0 q1 q2
            X q0
            CCX q0 q1 q2
            X q0
        ''')
    """
```

<a name="kickmix.Circuit.__ne__"></a>
```python
# kickmix.Circuit.__ne__

# (in class kickmix.Circuit)
def __ne__(
    self,
    arg0: kickmix.Circuit,
) -> bool:
    """Determines if two circuits have different instructions.
    """
```

<a name="kickmix.Circuit.__repr__"></a>
```python
# kickmix.Circuit.__repr__

# (in class kickmix.Circuit)
def __repr__(
    self,
) -> str:
    """Returns a description of the circuit.
    """
```

<a name="kickmix.Circuit.__rmul__"></a>
```python
# kickmix.Circuit.__rmul__

# (in class kickmix.Circuit)
def __rmul__(
    self,
    arg0: int,
) -> kickmix.Circuit:
    """Repeats the contents of a circuit the given number of times.

    Note: register data is not repeated.

    Examples:
        >>> import kickmix as km
        >>> km.Circuit('CX q0 q1') * 3
        km.Circuit('''
            CX q0 q1
            CX q0 q1
            CX q0 q1
        ''')

        >>> km.Circuit('CX q0 q1') * 0
        km.Circuit('''
        ''')

        >>> 5 * km.Circuit('''
        ...     REGISTER r0 "test"
        ...     APPEND_TO_REGISTER q0 r0
        ...     APPEND_TO_REGISTER q1 r0
        ...     APPEND_TO_REGISTER q2 r0
        ...     CCX q0 q1 q2
        ...     X q0
        ... ''')
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            APPEND_TO_REGISTER q1 r0
            APPEND_TO_REGISTER q2 r0
            REGISTER r0 "test"
            CCX q0 q1 q2
            X q0
            CCX q0 q1 q2
            X q0
            CCX q0 q1 q2
            X q0
            CCX q0 q1 q2
            X q0
            CCX q0 q1 q2
            X q0
        ''')
    """
```

<a name="kickmix.Circuit.__str__"></a>
```python
# kickmix.Circuit.__str__

# (in class kickmix.Circuit)
def __str__(
    self,
) -> str:
    """Returns a description of the circuit.
    """
```

<a name="kickmix.Circuit.html_diagram"></a>
```python
# kickmix.Circuit.html_diagram

# (in class kickmix.Circuit)
def html_diagram(
    self,
) -> str:
    """Returns an html diagram of the circuit.
    """
```

<a name="kickmix.Circuit.max_magic"></a>
```python
# kickmix.Circuit.max_magic

# (in class kickmix.Circuit)
def max_magic(
    self,
) -> int:
    """Returns an upper bound on the number of CCX/CCZ gates run by the circuit.
    """
```

<a name="kickmix.Circuit.num_bits"></a>
```python
# kickmix.Circuit.num_bits

# (in class kickmix.Circuit)
@property
def num_bits(
    self,
) -> int:
    """Returns the number of bits used by the circuit.
    """
```

<a name="kickmix.Circuit.num_qubits"></a>
```python
# kickmix.Circuit.num_qubits

# (in class kickmix.Circuit)
@property
def num_qubits(
    self,
) -> int:
    """Returns the number of qubits used by the circuit.
    """
```

<a name="kickmix.Circuit.num_registers"></a>
```python
# kickmix.Circuit.num_registers

# (in class kickmix.Circuit)
@property
def num_registers(
    self,
) -> int:
    """Returns the number of registers used by the circuit.
    """
```

<a name="kickmix.Circuit.reaction_depth"></a>
```python
# kickmix.Circuit.reaction_depth

# (in class kickmix.Circuit)
def reaction_depth(
    self,
) -> int:
    """Returns the reaction depth of the circuit, assuming it is powered by CCZ states.
    """
```

<a name="kickmix.Circuit.register_data"></a>
```python
# kickmix.Circuit.register_data

# (in class kickmix.Circuit)
@property
def register_data(
    self,
) -> object:
    """Returns the names and contents of registers in the circuit.
    """
```

<a name="kickmix.Circuit.text_diagram"></a>
```python
# kickmix.Circuit.text_diagram

# (in class kickmix.Circuit)
def text_diagram(
    self,
) -> str:
    """Returns a text diagram of the circuit.
    """
```

<a name="kickmix.CircuitBuilder"></a>
```python
# kickmix.CircuitBuilder

# (at top-level in the kickmix module)
class CircuitBuilder:
    """A kickmix circuit builder.
    """
```

<a name="kickmix.CircuitBuilder.__init__"></a>
```python
# kickmix.CircuitBuilder.__init__

# (in class kickmix.CircuitBuilder)
def __init__(
    self,
) -> None:
    """Initializes a new circuit builder.
    """
```

<a name="kickmix.CircuitBuilder.alloc_bits"></a>
```python
# kickmix.CircuitBuilder.alloc_bits

# (in class kickmix.CircuitBuilder)
def alloc_bits(
    self,
    count: int,
) -> kickmix.array:
```

<a name="kickmix.CircuitBuilder.alloc_qubits"></a>
```python
# kickmix.CircuitBuilder.alloc_qubits

# (in class kickmix.CircuitBuilder)
def alloc_qubits(
    self,
    count: int,
) -> kickmix.array:
    """Allocates clean qubits to use as workspace in a circuit.

    Allocated qubits can be deallocated by using the `CircuitBuilder.free` method.
    Qubits allocated within a `with builder.scope(auto_free=True):` block are
    deallocated automatically upon execution exiting the block.

    Although the qubits returned by this method *should* be clean, nothing *enforces*
    this. If a prior user of the allocated qubit failed to correctly clean some qubits
    before freeing them, the returned qubits may contain their buggy garbage. As such,
    it's recommended that you reset the qubits before using them. Reset gates also
    have the nice side effect of clearly indicating in circuit diagrams when a new
    usage of a qubit has begun.

    Args:
        length: The number of qubits to allocate.

    Returns:
        An array containing the ids of the allocated qubits. Until these ids are
        freed, they won't be returned by any other allocation calls.
    """
```

<a name="kickmix.CircuitBuilder.bit_store"></a>
```python
# kickmix.CircuitBuilder.bit_store

# (in class kickmix.CircuitBuilder)
def bit_store(
    self,
    target: b | km.array,
    value: bool,
    *,
    control: b | bool | km.array = True,
) -> None:
    """Appends operations to perform `if control: target = value`.

    Writes a constant into classical bits. The control must be
    classical, since a classical bit cannot be coherently conditioned
    on a qubit.

    This method broadcasts over km.array arguments.

    Args:
        target: The classical bit (or bits) to write into.
        value: The constant to store.
        control: Defaults to True. Determines if the store occurs.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> a = builder.create_classical_register(1, name='a')[0]
        >>> b = builder.create_classical_register(1, name='b')[0]
        >>> out = builder.create_classical_register(1, name='out')[0]
        >>> # Compute logical OR: out = a | b
        >>> builder.bit_store(out, False)
        >>> builder.bit_store(out, True, control=km.array([a, b]))
        >>> builder.finish_circuit()
        km.Circuit('''
            APPEND_TO_REGISTER b0 r0
            REGISTER r0 "a"
            APPEND_TO_REGISTER b1 r1
            REGISTER r1 "b"
            APPEND_TO_REGISTER b2 r2
            REGISTER r2 "out"
            BIT_STORE0 b2
            BIT_STORE1 b2 if b0
            BIT_STORE1 b2 if b1
        ''')
    """
```

<a name="kickmix.CircuitBuilder.ccx"></a>
```python
# kickmix.CircuitBuilder.ccx

# (in class kickmix.CircuitBuilder)
def ccx(
    self,
    control1: bool | b | q | km.array,
    control2: bool | b | q | km.array,
    target: q | km.array,
) -> None:
    """Appends many doubly-controlled NOT gates to the circuit.

    This method broadcasts over km.array arguments. If any of the arguments is a
    km.array, an append is performed for each array index. When multiple km.array
    arguments are used, they are combined by zipping (i.e. using two arrays of
    length n will produce n instructions rather than n*n instructions).

    The exact kind of operation(s) appended to the circuit depends on the types of
    the controls. For example, if one of the controls is the boolean value True
    instead of a qubit, then a CX instruction will be appended instead of a CCX
    instruction.

    Args:
        control1: One of the controls determining if the NOT gate happens or not.
        control2: The other control determining if the NOT gate happens or not.
        target: The qubit to bit flip when the controls are satisfied.

    Examples:
        >>> import kickmix as km

        >>> # Demo a simple ccx call
        >>> builder = km.CircuitBuilder()
        >>> builder.ccx(km.q(0), km.q(1), km.q(2))
        >>> builder.finish_circuit()
        km.Circuit('''
            CCX q0 q1 q2
        ''')

        >>> # Demo a broadcasting call
        >>> builder = km.CircuitBuilder()
        >>> shared_control = builder.create_quantum_register(1, "shared_control")[0]
        >>> controls = builder.create_quantum_register(3, "controls")
        >>> targets = builder.create_quantum_register(3, "targets")[0]
        >>> builder.ccx(shared_control, controls, targets)
        >>> builder.finish_circuit()
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            REGISTER r0 "shared_control"
            APPEND_TO_REGISTER q1 r1
            APPEND_TO_REGISTER q2 r1
            APPEND_TO_REGISTER q3 r1
            REGISTER r1 "controls"
            APPEND_TO_REGISTER q4 r2
            APPEND_TO_REGISTER q5 r2
            APPEND_TO_REGISTER q6 r2
            REGISTER r2 "targets"
            CCX q0 q1 q4
            CCX q0 q2 q4
            CCX q0 q3 q4
        ''')
    """
```

<a name="kickmix.CircuitBuilder.create_classical_register"></a>
```python
# kickmix.CircuitBuilder.create_classical_register

# (in class kickmix.CircuitBuilder)
def create_classical_register(
    self,
    length: int,
    name: str,
) -> kickmix.array:
    """Creates a classical register.

    Reserves the given number of bits for the register, and emits instructions
    defining the register.

    Args:
        name: A name for the register, which can be used to refer to it.
        length: The number of bits in the register.

    Returns:
        A kickmix.CircuitRegister containing the allocated qubits.
    """
```

<a name="kickmix.CircuitBuilder.create_quantum_register"></a>
```python
# kickmix.CircuitBuilder.create_quantum_register

# (in class kickmix.CircuitBuilder)
def create_quantum_register(
    self,
    length: int,
    name: str,
) -> kickmix.array:
    """Creates a quantum register.

    Reserves the given number of qubits for the register, and emits instructions
    defining the register.

    Args:
        name: A name for the register, which can be used to refer to it.
        length: The number of qubits in the register.

    Returns:
        A kickmix.CircuitRegister containing the allocated qubits.
    """
```

<a name="kickmix.CircuitBuilder.cswap"></a>
```python
# kickmix.CircuitBuilder.cswap

# (in class kickmix.CircuitBuilder)
def cswap(
    self,
    control: q | b | bool,
    target1: km.array,
    target2: km.array,
) -> None:
    """Appends operations to conditionally swap two equal-length registers.

    Performs `if control: target1, target2 = target2, target1`, pairing the
    registers index by index.

    The cost depends on the control. A qubit control expands to a
    cx/ccx/cx triple per index. A classical bit control pushes a
    classical condition around plain swaps, costing no Toffolis. A
    control of True reduces to a plain swap, and False emits nothing.

    Args:
        control: Determines if the swap happens.
        target1: The first register.
        target2: The second register, which must be the same length as the first.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> c = builder.create_quantum_register(1, name='c')[0]
        >>> x = builder.create_quantum_register(2, name='x')
        >>> y = builder.create_quantum_register(2, name='y')
        >>> builder.cswap(c, x, y)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -c[0]---@-----@---
                    |     |
        q1: -x[0]-@-X-@---|---
                  | | |   |
        q2: -x[1]-|-|-|-@-X-@-
                  | | | | | |
        q3: -y[0]-X-@-X-|-|-|-
                        | | |
        q4: -y[1]-------X-@-X-
    """
```

<a name="kickmix.CircuitBuilder.cx"></a>
```python
# kickmix.CircuitBuilder.cx

# (in class kickmix.CircuitBuilder)
def cx(
    self,
    control: object,
    target: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.cz"></a>
```python
# kickmix.CircuitBuilder.cz

# (in class kickmix.CircuitBuilder)
def cz(
    self,
    control1: bool | b | q | km.array,
    control2: bool | b | q | km.array,
) -> None:
    """Appends many CZ gates to the circuit.

    This method broadcasts over km.array arguments. If any of the arguments is a
    km.array, an append is performed for each array index. When multiple km.array
    arguments are used, they are combined by zipping (i.e. using two arrays of
    length n will produce n instructions rather than n*n instructions).

    The exact kind of operation(s) appended to the circuit depends on the types of
    the controls. For example, if one of the controls is the boolean value True
    instead of a qubit, then a Z instruction will be appended instead of a CZ
    instruction.

    Args:
        control1: One of the controls determining if phasing occurs.
        control2: The other controls determining if phasing occurs.
    """
```

<a name="kickmix.CircuitBuilder.debug_print"></a>
```python
# kickmix.CircuitBuilder.debug_print

# (in class kickmix.CircuitBuilder)
def debug_print(
    self,
    arg: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.del_and"></a>
```python
# kickmix.CircuitBuilder.del_and

# (in class kickmix.CircuitBuilder)
def del_and(
    self,
    control1: bool | b | q | km.array,
    control2: bool | b | q | km.array,
    target: q | km.array,
) -> None:
    """Clears a qubit holding `control1 and control2`, using only Cliffords and measurement.

    This is the uncomputation partner of `init_and`, or of a `ccx` into
    a qubit that started at |0>. It applies an `HMR` (X-basis
    measurement and Z-basis reset to |0>) to `target`, followed by a
    phase correction `CZ(control1, control2)` conditioned on the
    measurement result being 1. The `target` qubit is left in |0> and is
    not freed, so the caller still decides when to `free` it.

    This method broadcasts over km.array arguments.

    Args:
        control1: One of the controls that was used to compute the target.
        control2: The other control that was used to compute the target.
        target: The qubit holding the AND, restored to |0>.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> c1 = builder.alloc_qubits(1)[0]
        >>> c2 = builder.alloc_qubits(1)[0]
        >>> target = builder.alloc_qubits(1)[0]
        >>> builder.del_and(c1, c2, target)
        >>> builder.finish_circuit()
        km.Circuit('''
            HMR q2 b0
            CZ q0 q1 if b0
        ''')
    """
```

<a name="kickmix.CircuitBuilder.del_gf2_div"></a>
```python
# kickmix.CircuitBuilder.del_gf2_div

# (in class kickmix.CircuitBuilder)
def del_gf2_div(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    target: km.array,
    field: km.GF2Field | None = None,
) -> None:
    """Clears a GF(2^m) register known to hold `(lhs / rhs) % modulus` to 0.

    This is the uncomputation partner of `init_gf2_div`. Temporary
    workspace is allocated to recompute the inverse. If you still hold
    the scaffold from `init_gf2_div_with_scaffold`, use
    `del_gf2_div_with_scaffold` instead to avoid the recompute.

    Args:
        lhs: The dividend register. Left unchanged.
        rhs: The divisor register. Left unchanged.
        target: The register holding `lhs / rhs`, cleared to 0.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> lhs = builder.create_quantum_register(8, name='lhs')
        >>> rhs = builder.create_quantum_register(8, name='rhs')
        >>> quotient = builder.init_gf2_div(lhs, rhs)
        >>> builder.del_gf2_div(lhs, rhs, target=quotient)
        >>> builder.finish_circuit().max_magic()  # 135 to build, 108 to rebuild
        243
    """
```

<a name="kickmix.CircuitBuilder.del_gf2_div_with_scaffold"></a>
```python
# kickmix.CircuitBuilder.del_gf2_div_with_scaffold

# (in class kickmix.CircuitBuilder)
def del_gf2_div_with_scaffold(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    target: km.array,
    scaffold: km.array,
    field: km.GF2Field | None = None,
) -> None:
    """Clears `target` and its scaffold, using only Cliffords and measurement.

    This is the uncomputation partner of `init_gf2_div_with_scaffold`.
    Because the divisor's inverse is still live, both `target` and
    `scaffold` are cleared to 0 without any Toffolis. Neither register
    is freed, so the caller still decides when to `free` them.

    Args:
        lhs: The dividend register. Left unchanged.
        rhs: The divisor register. Left unchanged.
        target: The register holding `lhs / rhs`, cleared to 0.
        scaffold: The workspace from `init_gf2_div_with_scaffold`, also
            cleared to 0.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> lhs = builder.create_quantum_register(8, name='lhs')
        >>> rhs = builder.create_quantum_register(8, name='rhs')
        >>> quotient, scaffold = builder.init_gf2_div_with_scaffold(lhs, rhs)
        >>> builder.del_gf2_div_with_scaffold(
        ...     lhs, rhs, target=quotient, scaffold=scaffold)
        >>> builder.finish_circuit().max_magic()  # no more than building it once
        135
    """
```

<a name="kickmix.CircuitBuilder.del_gf2_inverse"></a>
```python
# kickmix.CircuitBuilder.del_gf2_inverse

# (in class kickmix.CircuitBuilder)
def del_gf2_inverse(
    self,
    input: km.array,
    *,
    target: km.array,
    field: km.GF2Field | None = None,
) -> None:
    """Clears `target` (holding `input ** -1` in GF(2^m)) back to 0.

    This is the uncomputation partner of `init_gf2_inverse`. Temporary
    workspace is allocated to rebuild and uncompute the addition chain.
    If you still hold the scaffold from
    `init_gf2_inverse_with_scaffold`, use
    `del_gf2_inverse_with_scaffold` instead to avoid the rebuild.

    Args:
        input: The inverted GF(2^m) register. Left unchanged.
        target: The register holding `input ** -1`, cleared to 0.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> inp = builder.create_quantum_register(8, name='inp')
        >>> inv = builder.init_gf2_inverse(inp)
        >>> builder.del_gf2_inverse(inp, target=inv)
        >>> builder.finish_circuit().max_magic()  # 108 to build, 108 to rebuild
        216
    """
```

<a name="kickmix.CircuitBuilder.del_gf2_inverse_with_scaffold"></a>
```python
# kickmix.CircuitBuilder.del_gf2_inverse_with_scaffold

# (in class kickmix.CircuitBuilder)
def del_gf2_inverse_with_scaffold(
    self,
    input: km.array,
    *,
    target: km.array,
    scaffold: km.array,
    field: km.GF2Field | None = None,
) -> None:
    """Clears `target` and its scaffold, using only Cliffords and measurement.

    This is the uncomputation partner of
    `init_gf2_inverse_with_scaffold`. Because the addition chain is
    still live, both `target` and `scaffold` are cleared to 0 without
    any Toffolis. Neither register is freed, so the caller still
    decides when to `free` them.

    Args:
        input: The inverted GF(2^m) register. Left unchanged.
        target: The register holding `input ** -1`, cleared to 0.
        scaffold: The addition chain from
            `init_gf2_inverse_with_scaffold`, also cleared to 0.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> inp = builder.create_quantum_register(8, name='inp')
        >>> inv, scaffold = builder.init_gf2_inverse_with_scaffold(inp)
        >>> builder.del_gf2_inverse_with_scaffold(
        ...     inp, target=inv, scaffold=scaffold)
        >>> builder.finish_circuit().max_magic()  # no more than building it once
        108
    """
```

<a name="kickmix.CircuitBuilder.del_gf2_mul"></a>
```python
# kickmix.CircuitBuilder.del_gf2_mul

# (in class kickmix.CircuitBuilder)
def del_gf2_mul(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    target: km.array,
    field: km.GF2Field | None = None,
    control: q | b | bool = True,
) -> None:
    """Clears a GF(2^m) register known to hold `(lhs * rhs) % modulus`.

    This is the uncomputation partner of `init_gf2_mul`. When
    uncontrolled, uncomputes `target` back to 0 using X-basis
    measurements and classically conditioned CZ fixups (zero Toffolis).

    Args:
        lhs: The first factor register. Left unchanged.
        rhs: The second factor register. Left unchanged.
        target: The GF(2^m) register holding `lhs * rhs`, cleared to 0.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.
        control: Defaults to True. Determines if the operation occurs.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> lhs = builder.create_quantum_register(1, name='lhs')
        >>> rhs = builder.create_quantum_register(1, name='rhs')
        >>> target = builder.init_gf2_mul(lhs, rhs)
        >>> builder.del_gf2_mul(lhs, rhs, target=target)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -lhs[0]-@--------@-----
                    |        |
        q1: -rhs[0]-@--------Z**b0-
                    |
        q2:  |0>----X-HMR=b0
    """
```

<a name="kickmix.CircuitBuilder.del_lookup"></a>
```python
# kickmix.CircuitBuilder.del_lookup

# (in class kickmix.CircuitBuilder)
def del_lookup(
    self,
    *,
    table: Iterable[int],
    address: km.array,
    target: km.array,
) -> None:
    """Appends operations to perform `del target := table[address]`.

    Clears a value created by looking up a value from a classical table, indexed by
    a quantum register.

    Because this is a `del` method, the target qubits will end in the 0 state.

    Args:
        table: The table of classical values, as a list of python integers.
            The integers are interpreted in a little-endian fashion, meaning that
            `table_data[address, out_index] == (table[address] >> out_index) & 1`.
        address: The address used to select a value from the table.
            Qubits are in little-endian order.
        target: The qubits to clear the lookup result from.
    """
```

<a name="kickmix.CircuitBuilder.end_auto_free_scope"></a>
```python
# kickmix.CircuitBuilder.end_auto_free_scope

# (in class kickmix.CircuitBuilder)
def end_auto_free_scope(
    self,
) -> None:
    """Ends a scope that automatically frees allocations.

    Allocations since the corresponding builder.start_auto_free_scope() will be
    freed. If multiple scopes are being used, they operate in LIFO order (i.e. like
    a stack).
    """
```

<a name="kickmix.CircuitBuilder.finish_circuit"></a>
```python
# kickmix.CircuitBuilder.finish_circuit

# (in class kickmix.CircuitBuilder)
def finish_circuit(
    self,
) -> kickmix.Circuit:
    """Returns the circuit built by the circuit builder.
    """
```

<a name="kickmix.CircuitBuilder.flame_chart_svg"></a>
```python
# kickmix.CircuitBuilder.flame_chart_svg

# (in class kickmix.CircuitBuilder)
def flame_chart_svg(
    self,
) -> str:
    """Returns a flame chart of CCX+CCZ and qubit utilization of the circuit being built.
    """
```

<a name="kickmix.CircuitBuilder.flip_if_equal"></a>
```python
# kickmix.CircuitBuilder.flip_if_equal

# (in class kickmix.CircuitBuilder)
def flip_if_equal(
    self,
    lhs: km.array,
    rhs: km.array | int,
    *,
    target: q,
    control: q | b | bool = True,
) -> None:
    """Appends operations to perform `if control: target ^= lhs == rhs`.

    Xors the result of an unsigned integer comparison into a target qubit.

    Args:
        lhs: The little-endian value on the left hand side of the comparison.
        rhs: The little-endian value or non-negative integer constant on the
            right hand side of the comparison.
        target: The qubit to flip when lhs == rhs.
        control: Defaults to True. Determines if the comparisons happens at all.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> t = builder.create_quantum_register(1, name='t')[0]
        >>> a = builder.create_quantum_register(2, name='a')
        >>> builder.flip_if_equal(a, 2, target=t)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -t[0]-------------X-------------------------------
                              |
        q1: -a[0]-X-----@-----|---------------------Z**b0-X---
                        |     |
        q2: -a[1]-X-X---|---@-@--------Z**b0--------------X-X-
                        |   | |        |
        q3:         |0>-X---@-|--------@-----HMR=b0
                            | |
        q4:             |0>-X-@-HMR=b0
        >>> # Quantum-quantum comparison:
        >>> builder = km.CircuitBuilder()
        >>> t = builder.create_quantum_register(1, name='t')[0]
        >>> a = builder.create_quantum_register(2, name='a')
        >>> b = builder.create_quantum_register(2, name='b')
        >>> builder.flip_if_equal(a, b, target=t)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -t[0]---------------X---------------------------------
                                |
        q1: -a[0]-X-X-----@-----|---------------------Z**b0-X-X---
                  |       |     |                             |
        q2: -a[1]-|-X-X---|---@-@--------Z**b0--------------X-|-X-
                  | |     |   | |        |                    | |
        q3: -b[0]-@-|-----|---|-|--------|--------------------@-|-
                    |     |   | |        |                      |
        q4: -b[1]---@-----|---|-|--------|----------------------@-
                          |   | |        |
        q5:           |0>-X---@-|--------@-----HMR=b0
                              | |
        q6:               |0>-X-@-HMR=b0
    """
```

<a name="kickmix.CircuitBuilder.flip_if_greater_than"></a>
```python
# kickmix.CircuitBuilder.flip_if_greater_than

# (in class kickmix.CircuitBuilder)
def flip_if_greater_than(
    self,
    lhs: km.array,
    rhs: km.array | int,
    *,
    target: q,
    or_equal: q | b | bool = false,
    control: q | b | bool = True,
    btol: float = float('inf'),
) -> None:
    """Appends operations to perform `if control: target ^= lhs > rhs - or_equal`.

    Xors the result of an unsigned integer comparison into a target qubit.

    Args:
        lhs: The little-endian value on the left hand side of the comparison.
        rhs: The little-endian value or non-negative integer constant on the
            right hand side of the comparison.
        target: The qubit to flip when lhs > rhs.
        or_equal: Defaults to False. Determines if the comparison is a
            greater-than-or-equal-to test rather than a greater-than test.
        control: Defaults to True. Determines if the comparisons happens at all.
        btol: Defaults to infinity (exact). A finite value leads to a
            cheaper circuit by only comparing the most significant bits,
            since the rest only decide the answer when those tie. In
            exchange, the answer can be wrong, but on no more than a
            2**-btol fraction of inputs, assuming lhs and rhs are
            independent and uniformly random. Holding one of them fixed
            costs about a bit of that guarantee, and adversarial inputs
            get no guarantee at all.

            btol doesn't have to be an integer. Fractional values come up
            naturally because approximation errors add up: doing k
            approximate operations at btol + log2(k) each keeps the total
            under 2**-btol.

            A btol of len(lhs) or more is exact. A btol of 1 or less lets
            the circuit skip the comparison entirely, since never
            flipping the target is already right half the time.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> t = builder.create_quantum_register(1, name='t')[0]
        >>> a = builder.create_quantum_register(2, name='a')
        >>> builder.flip_if_greater_than(a, 1, target=t)
        >>> print(builder.finish_circuit().text_diagram())
                                    push_cond if b0     pop_cond
        q0: -t[0]---------X----------------------------------------X-
                          |         neg
        q1: -a[0]-X-X-----|-------------------------Z-Z----------X-X-
                          |
        q2: -a[1]-X-------@----------------------------------------X-
                          |
        q3:         |0>-X-@-HMR=b0
        >>> # Greater-than-or-equal-to comparison:
        >>> builder = km.CircuitBuilder()
        >>> t = builder.create_quantum_register(1, name='t')[0]
        >>> a = builder.create_quantum_register(2, name='a')
        >>> builder.flip_if_greater_than(a, 1, target=t, or_equal=True)
        >>> print(builder.finish_circuit().text_diagram())
                                      push_cond if b0   pop_cond
        q0: -t[0]-----------X--------------------------------------X-
                            |         neg
        q1: -a[0]-X-X---@---|-------------------------Z----------X-X-
                        |   |
        q2: -a[1]-X-----|---@--------------------------------------X-
                        |   |
        q3:         |0>-X-X-@-HMR=b0
        >>> # Controlled comparison:
        >>> builder = km.CircuitBuilder()
        >>> c = builder.create_quantum_register(1, name='c')[0]
        >>> t = builder.create_quantum_register(1, name='t')[0]
        >>> a = builder.create_quantum_register(2, name='a')
        >>> builder.flip_if_greater_than(a, 1, target=t, control=c)
        >>> print(builder.finish_circuit().text_diagram())
                                               push_cond if b0     pop_cond
        q0: -c[0]-----------@----------@--------------------------------------@-
                            |          |       neg                            |
        q1: -t[0]-----------|-X--------|--------------------------------------X-
                            | |        |
        q2: -a[0]-X-X-------|-|--------|-----------------------Z-Z----------X-X-
                            | |        |
        q3: -a[1]-X---------@-|--------Z**b0----------------------------------X-
                            | |
        q4:         |0>-X---|-@--------HMR=b0
                            | |
        q5:             |0>-X-@-HMR=b0
        >>> # Quantum-quantum comparison:
        >>> builder = km.CircuitBuilder()
        >>> t = builder.create_quantum_register(1, name='t')[0]
        >>> a = builder.create_quantum_register(2, name='a')
        >>> b = builder.create_quantum_register(2, name='b')
        >>> builder.flip_if_greater_than(a, b, target=t)
        >>> print(builder.finish_circuit().text_diagram())
                                                    push_cond if b0     pop_cond
        q0: -t[0]-------------------X---X--------------------------------------------X-
                                    |   |
        q1: -a[0]-X-X-------@-------|---|---------------------------Z-@----------X-X---
                    |       |       |   |                             |          |
        q2: -a[1]-X-|-X-----|-------|---@-----------------------------|----------|-X-X-
                    | |     |       |   |                             |          | |
        q3: -b[0]---@-|---X-@-X-@---|---|---------------------------Z-Z----------@-|---
                      |     |   |   |   |                                          |
        q4: -b[1]-----@-----|---|-@-@-@-|-@----------------------------------------@---
                            |   | | | | | |
        q5:           |0>---X---X-X-X-X-@-X-HMR=b0
        >>> # Bounded tolerance comparison (comparing top bits only):
        >>> builder = km.CircuitBuilder()
        >>> t = builder.create_quantum_register(1, name='t')[0]
        >>> a = builder.create_quantum_register(4, name='a')
        >>> builder.flip_if_greater_than(a, 7, target=t, btol=2)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -t[0]---X-X-
                    |
        q1: -a[0]---|---
                    |
        q2: -a[1]---|---
                    |
        q3: -a[2]---|---
                    |
        q4: -a[3]-X-@-X-
    """
```

<a name="kickmix.CircuitBuilder.flip_if_less_than"></a>
```python
# kickmix.CircuitBuilder.flip_if_less_than

# (in class kickmix.CircuitBuilder)
def flip_if_less_than(
    self,
    lhs: km.array,
    rhs: km.array | int,
    *,
    target: q,
    or_equal: q | b | bool = false,
    control: q | b | bool = True,
    btol: float = float('inf'),
) -> None:
    """Appends operations to perform `if control: target ^= lhs < rhs + or_equal`.

    Xors the result of an unsigned integer comparison into a target qubit.

    Args:
        lhs: The little-endian value on the left hand side of the comparison.
        rhs: The little-endian value or non-negative integer constant on the
            right hand side of the comparison.
        target: The qubit to flip when lhs < rhs.
        or_equal: Defaults to False. Determines if the comparison is a
            less-than-or-equal-to test rather than a less-than test.
        control: Defaults to True. Determines if the comparisons happens at all.
        btol: Defaults to infinity (exact). A finite value leads to a
            cheaper circuit by only comparing the most significant bits,
            since the rest only decide the answer when those tie. In
            exchange, the answer can be wrong, but on no more than a
            2**-btol fraction of inputs, assuming lhs and rhs are
            independent and uniformly random. Holding one of them fixed
            costs about a bit of that guarantee, and adversarial inputs
            get no guarantee at all.

            btol doesn't have to be an integer. Fractional values come up
            naturally because approximation errors add up: doing k
            approximate operations at btol + log2(k) each keeps the total
            under 2**-btol.

            A btol of len(lhs) or more is exact. A btol of 1 or less lets
            the circuit skip the comparison entirely, since never
            flipping the target is already right half the time.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> target = builder.create_quantum_register(1, name="target")[0]
        >>> lhs = builder.create_quantum_register(1, name="lhs")
        >>> rhs = builder.create_quantum_register(1, name="rhs")
        >>> builder.flip_if_less_than(lhs, rhs, target=target)
        >>> builder.finish_circuit()
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            REGISTER r0 "target"
            APPEND_TO_REGISTER q1 r1
            REGISTER r1 "lhs"
            APPEND_TO_REGISTER q2 r2
            REGISTER r2 "rhs"
            CX q2 q1
            CCX q1 q2 q0
            X q1
            X q1
            CX q2 q1
        ''')
    """
```

<a name="kickmix.CircuitBuilder.free"></a>
```python
# kickmix.CircuitBuilder.free

# (in class kickmix.CircuitBuilder)
def free(
    self,
    arg: object,
) -> None:
    """Frees allocated qubits or bits, so that they can be allocated again.
    """
```

<a name="kickmix.CircuitBuilder.gf2_iadd"></a>
```python
# kickmix.CircuitBuilder.gf2_iadd

# (in class kickmix.CircuitBuilder)
def gf2_iadd(
    self,
    offset: km.array | int,
    *,
    target: km.array,
    field: km.GF2Field | None = None,
    control: q | b | bool = True,
) -> None:
    """Appends operations to perform `if control: target ^= offset` in GF(2^m).

    Addition in a binary extension field is bitwise XOR, realized as a
    layer of CNOTs (or Toffolis when quantum-controlled, or NOTs/CNOTs
    when adding a classical constant).

    Args:
        offset: The register or classical constant (int) to add. Left unchanged.
        target: The register to add into in place.
        field: Optional `km.GF2Field`. If provided, its degree must
            equal `len(target)`.
        control: Defaults to True. Determines if the addition occurs.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> target = builder.create_quantum_register(2, name='target')
        >>> offset = builder.create_quantum_register(2, name='offset')
        >>> builder.gf2_iadd(offset, target=target)
        >>> builder.gf2_iadd(1, target=target)
        >>> builder.finish_circuit()
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            APPEND_TO_REGISTER q1 r0
            REGISTER r0 "target"
            APPEND_TO_REGISTER q2 r1
            APPEND_TO_REGISTER q3 r1
            REGISTER r1 "offset"
            CX q2 q0
            CX q3 q1
            X q0
        ''')
    """
```

<a name="kickmix.CircuitBuilder.gf2_idiv"></a>
```python
# kickmix.CircuitBuilder.gf2_idiv

# (in class kickmix.CircuitBuilder)
def gf2_idiv(
    self,
    divisor: int,
    *,
    target: km.array,
    field: km.GF2Field | None = None,
) -> None:
    """Appends operations to perform `target = (target / divisor) % modulus`.

    Divides a GF(2^m) register in place by a non-zero classical divisor.
    This is the exact inverse of `gf2_imul`.

    Currently only classical integer divisors are supported; passing a
    non-classical input raises `NotImplementedError`.

    Args:
        divisor: The non-zero classical field element to divide by.
        target: The GF(2^m) register to divide in place.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> target = builder.create_quantum_register(2, name='target')
        >>> builder.gf2_idiv(2, target=target)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -target[0]-X-@-
                       | |
        q1: -target[1]-@-X-
    """
```

<a name="kickmix.CircuitBuilder.gf2_imul"></a>
```python
# kickmix.CircuitBuilder.gf2_imul

# (in class kickmix.CircuitBuilder)
def gf2_imul(
    self,
    factor: int,
    *,
    target: km.array,
    field: km.GF2Field | None = None,
) -> None:
    """Appends operations to perform `target = (target * factor) % modulus`.

    Multiplies a GF(2^m) register in place by a non-zero classical factor
    using only CNOT and SWAP gates (no ancillas or Toffolis).

    Currently only classical integer factors are supported; passing a
    non-classical input raises `NotImplementedError`.

    Args:
        factor: The non-zero classical field element to multiply by.
        target: The GF(2^m) register to multiply in place.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> target = builder.create_quantum_register(2, name='target')
        >>> builder.gf2_imul(2, target=target)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -target[0]-X-SWAP-
                       | |
        q1: -target[1]-@-SWAP-
    """
```

<a name="kickmix.CircuitBuilder.gf2_phase_by_product"></a>
```python
# kickmix.CircuitBuilder.gf2_phase_by_product

# (in class kickmix.CircuitBuilder)
def gf2_phase_by_product(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    mask: km.array,
    field: km.GF2Field | None = None,
) -> None:
    """Negates amplitudes where `popcount(mask & (lhs * rhs) % modulus)` is odd.

    Kicks back the phase of the product directly without computing it
    into an intermediate register, using zero Toffolis and zero
    ancillas.

    Args:
        lhs: The first factor register. Left unchanged.
        rhs: The second factor register. Left unchanged.
        mask: Classical bits selecting which product bits contribute.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(lhs))`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> lhs = builder.create_quantum_register(2, name='lhs')
        >>> rhs = builder.create_quantum_register(2, name='rhs')
        >>> mask = builder.create_classical_register(2, name='mask')
        >>> builder.gf2_phase_by_product(lhs, rhs, mask=mask)
        >>> print(builder.finish_circuit().text_diagram())
                    mask[0]=b0             b2^=1 if b0       b2^=1 if b0
        q0: -lhs[0]------------@-----------------------------------------
                    mask[1]=b1 |           b2^=1 if b1       b2^=1 if b1
        q1: -lhs[1]------------|-----@-----------------@-----------------
                               |     |                 |
        q2: -rhs[0]------------Z**b0-Z**b1-------------|-----------------
                               |                       |
        q3: -rhs[1]------------Z**b1-------------------Z**b2-------------
    """
```

<a name="kickmix.CircuitBuilder.hmr"></a>
```python
# kickmix.CircuitBuilder.hmr

# (in class kickmix.CircuitBuilder)
def hmr(
    self,
    target: object,
    output: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.iadd"></a>
```python
# kickmix.CircuitBuilder.iadd

# (in class kickmix.CircuitBuilder)
def iadd(
    self,
    offset: km.array | int,
    *,
    target: km.array,
    control: q | b | bool = True,
    btol: float = float('inf'),
) -> None:
    """Appends operations to perform `if control: target += offset`.

    Performs an inplace 2s-complement addition modulo 2**len(target).
    The offset can be a classical integer constant, classical bit
    register, or a quantum register.

    Args:
        offset: The little-endian value or integer to add. Can be
            classical or quantum.
        target: The little-endian quantum register to add into.
        control: Defaults to True. Determines if the addition occurs.
        btol: Defaults to infinity (exact). Only applies to a classical
            offset; passing a finite btol with a quantum offset is an
            error. A finite value can lead to a cheaper circuit by not
            carrying into the leading run of identical bits of offset,
            so there's nothing to save unless that run is longer than
            btol. In exchange, the sum can be wrong, but on no more than
            a 2**-btol fraction of target values, assuming target is
            uniformly random.

            btol doesn't have to be an integer. Fractional values come up
            naturally because approximation errors add up: doing k
            approximate operations at btol + log2(k) each keeps the total
            under 2**-btol.

            A btol of len(target) or more is exact.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> a = builder.create_quantum_register(2, name='a')
        >>> b = builder.create_quantum_register(2, name='b')
        >>> builder.iadd(b, target=a)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -a[0]-@-X---
                  | |
        q1: -a[1]-X-|-X-
                  | | |
        q2: -b[0]-@-@-|-
                      |
        q3: -b[1]-----@-
    """
```

<a name="kickmix.CircuitBuilder.init_and"></a>
```python
# kickmix.CircuitBuilder.init_and

# (in class kickmix.CircuitBuilder)
def init_and(
    self,
    control1: bool | b | q | km.array,
    control2: bool | b | q | km.array,
    target: q | km.array,
) -> None:
    """Initializes a qubit to `control1 and control2`, resetting it first.

    This is the computation partner of `del_and`. It resets `target`
    to |0> and then applies `CCX(control1, control2, target)`.

    This method broadcasts over km.array arguments.

    Args:
        control1: One of the controls of the AND.
        control2: The other control of the AND.
        target: The qubit initialized to the AND result.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> c1 = builder.alloc_qubits(1)[0]
        >>> c2 = builder.alloc_qubits(1)[0]
        >>> target = builder.alloc_qubits(1)[0]
        >>> builder.init_and(c1, c2, target)
        >>> builder.finish_circuit()
        km.Circuit('''
            R q2
            CCX q0 q1 q2
        ''')
    """
```

<a name="kickmix.CircuitBuilder.init_gf2_div"></a>
```python
# kickmix.CircuitBuilder.init_gf2_div

# (in class kickmix.CircuitBuilder)
def init_gf2_div(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    target: km.array | str = "alloc",
    field: km.GF2Field | None = None,
) -> km.array:
    """Initializes a GF(2^m) register to `(lhs / rhs) % modulus`, resetting it first.

    This is the computation partner of `del_gf2_div`. Resets `target`
    to |0> and then writes `lhs / rhs` into it. A zero divisor produces
    a quotient of 0. The divisor's inverse and its addition chain are
    temporary workspace, allocated and cleaned up automatically. Use
    `init_gf2_div_with_scaffold` to keep them instead.

    Args:
        lhs: The dividend register. Left unchanged.
        rhs: The divisor register. Left unchanged.
        target: The GF(2^m) register to initialize. Defaults to "alloc",
            which allocates a register of the field's degree.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(lhs))`.

    Returns:
        The initialized register.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> lhs = builder.create_quantum_register(8, name='lhs')
        >>> rhs = builder.create_quantum_register(8, name='rhs')
        >>> quotient = builder.init_gf2_div(lhs, rhs)
        >>> builder.finish_circuit().max_magic()
        135
    """
```

<a name="kickmix.CircuitBuilder.init_gf2_div_with_scaffold"></a>
```python
# kickmix.CircuitBuilder.init_gf2_div_with_scaffold

# (in class kickmix.CircuitBuilder)
def init_gf2_div_with_scaffold(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    target: km.array | str = "alloc",
    field: km.GF2Field | None = None,
) -> Tuple[km.array, km.array]:
    """Initializes `target := lhs / rhs`, keeping the division workspace.

    Same as `init_gf2_div`, except the divisor's inverse and its
    addition chain are kept rather than uncomputed. Keeping them lets
    `del_gf2_div_with_scaffold` undo the whole thing with zero
    Toffolis, which is cheaper than recomputing the inverse.

    This is the computation partner of `del_gf2_div_with_scaffold`.

    Args:
        lhs: The dividend register. Left unchanged.
        rhs: The divisor register. Left unchanged.
        target: The GF(2^m) register to initialize. Defaults to "alloc",
            which allocates a register of the field's degree.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(lhs))`.

    Returns:
        A tuple of the initialized register and the scaffold register
        holding the divisor's inverse and addition chain.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> lhs = builder.create_quantum_register(8, name='lhs')
        >>> rhs = builder.create_quantum_register(8, name='rhs')
        >>> quotient, scaffold = builder.init_gf2_div_with_scaffold(lhs, rhs)
        >>> len(scaffold)
        32
        >>> builder.finish_circuit().max_magic()
        135
    """
```

<a name="kickmix.CircuitBuilder.init_gf2_inverse"></a>
```python
# kickmix.CircuitBuilder.init_gf2_inverse

# (in class kickmix.CircuitBuilder)
def init_gf2_inverse(
    self,
    input: km.array,
    *,
    target: km.array | str = "alloc",
    field: km.GF2Field | None = None,
) -> km.array:
    """Initializes `target := input ** -1` in GF(2^m) (mapping 0 to 0).

    This is the computation partner of `del_gf2_inverse`. Resets
    `target` to |0> and then writes the inverse into it, using the
    Itoh-Tsujii addition chain. The chain's intermediates are temporary
    workspace, allocated and cleaned up automatically. Use
    `init_gf2_inverse_with_scaffold` to keep them instead, which makes
    the matching uncomputation free of Toffolis.

    Args:
        input: The GF(2^m) register to invert. Left unchanged.
        target: The register to store the inverse into. Defaults to
            "alloc", which allocates a register of the field's degree.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(input))`.

    Returns:
        The initialized register.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> inp = builder.create_quantum_register(2, name='inp')
        >>> inv = builder.init_gf2_inverse(inp)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -inp[0]-@-----
                    |
        q1: -inp[1]-|-@---
                    | |
        q2:  |0>----X-|-X-
                      | |
        q3:  |0>------X-@-
    """
```

<a name="kickmix.CircuitBuilder.init_gf2_inverse_with_scaffold"></a>
```python
# kickmix.CircuitBuilder.init_gf2_inverse_with_scaffold

# (in class kickmix.CircuitBuilder)
def init_gf2_inverse_with_scaffold(
    self,
    input: km.array,
    *,
    target: km.array | str = "alloc",
    field: km.GF2Field | None = None,
) -> Tuple[km.array, km.array]:
    """Initializes `target := input ** -1`, keeping the addition chain.

    Same as `init_gf2_inverse`, except the Itoh-Tsujii addition chain
    intermediates are kept rather than uncomputed. Keeping them lets
    `del_gf2_inverse_with_scaffold` undo the whole thing with zero
    Toffolis, which is cheaper than recomputing the chain.

    This is the computation partner of `del_gf2_inverse_with_scaffold`.

    Args:
        input: The GF(2^m) register to invert. Left unchanged.
        target: The register to store the inverse into. Defaults to
            "alloc", which allocates a register of the field's degree.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(input))`.

    Returns:
        A tuple of the initialized register and the scaffold register
        holding the addition chain.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> inp = builder.create_quantum_register(8, name='inp')
        >>> inv, scaffold = builder.init_gf2_inverse_with_scaffold(inp)
        >>> len(scaffold)
        24
        >>> builder.finish_circuit().max_magic()
        108
    """
```

<a name="kickmix.CircuitBuilder.init_gf2_mul"></a>
```python
# kickmix.CircuitBuilder.init_gf2_mul

# (in class kickmix.CircuitBuilder)
def init_gf2_mul(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    target: km.array | str = "alloc",
    field: km.GF2Field | None = None,
    control: q | b | bool = True,
) -> km.array:
    """Initializes a GF(2^m) register to `(lhs * rhs) % modulus`, resetting it first.

    This is the computation partner of `del_gf2_mul`. Resets `target`
    to |0> and then multiplies `lhs` and `rhs` into `target`.

    Args:
        lhs: The first factor register. Left unchanged.
        rhs: The second factor register. Left unchanged.
        target: The GF(2^m) register to initialize. Defaults to "alloc",
            which allocates a register of the field's degree.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(lhs))`.
        control: Defaults to True. Determines if the operation occurs.

    Returns:
        The initialized register.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> lhs = builder.create_quantum_register(2, name='lhs')
        >>> rhs = builder.create_quantum_register(2, name='rhs')
        >>> target = builder.init_gf2_mul(lhs, rhs)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -lhs[0]---X----@-X---------------@-----
                      |    | |               |
        q1: -lhs[1]---@----|-@------@--------|-----
                           |        |        |
        q2: -rhs[0]---X----@-X------|--------@-----
                      |    | |      |        |
        q3: -rhs[1]---@----|-@------@--------|-----
                           |        |        |
        q4:  |0>----@-SWAP-X-X-SWAP-X-X-SWAP-X-X-@-
                    | |      | |      | |      | |
        q5:  |0>----X-SWAP---@-SWAP---@-SWAP---@-X-
    """
```

<a name="kickmix.CircuitBuilder.init_lookup"></a>
```python
# kickmix.CircuitBuilder.init_lookup

# (in class kickmix.CircuitBuilder)
def init_lookup(
    self,
    *,
    table: Iterable[int],
    address: km.array,
    target: km.array,
) -> None:
    """Appends operations to perform `let target := table[address]`.

    Looks up a value from a classical table, indexed by a quantum register.

    Because this is an `init` method, the target qubits must start in the 0 state.
    The target qubits will be reset before being used, transforming any bit errors
    into probabilistic phase errors.

    Args:
        table: The table of classical values, as a list of python integers.
            The integers are interpreted in a little-endian fashion, meaning that
            `table_data[address, out_index] == (table[address] >> out_index) & 1`.
        address: The address used to select a value from the table.
        target: The clean qubits to store the output into.
    """
```

<a name="kickmix.CircuitBuilder.isub"></a>
```python
# kickmix.CircuitBuilder.isub

# (in class kickmix.CircuitBuilder)
def isub(
    self,
    offset: km.array | int,
    *,
    target: km.array,
    control: q | b | bool = True,
    btol: float = float('inf'),
) -> None:
    """Appends operations to perform `if control: target -= offset`.

    Performs an inplace 2s-complement subtraction modulo 2**len(target).
    The offset can be a classical integer constant, classical bit
    register, or a quantum register.

    Args:
        offset: The little-endian value or integer to subtract. Can be
            classical or quantum.
        target: The little-endian quantum register to subtract from.
        control: Defaults to True. Determines if the subtraction occurs.
        btol: Defaults to infinity (exact). Only applies to a classical
            offset; passing a finite btol with a quantum offset is an
            error. A finite value can lead to a cheaper circuit by not
            borrowing into the leading run of identical bits of offset,
            so there's nothing to save unless that run is longer than
            btol. In exchange, the difference can be wrong, but on no
            more than a 2**-btol fraction of target values, assuming
            target is uniformly random.

            btol doesn't have to be an integer. Fractional values come up
            naturally because approximation errors add up: doing k
            approximate operations at btol + log2(k) each keeps the total
            under 2**-btol.

            A btol of len(target) or more is exact.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> a = builder.create_quantum_register(2, name='a')
        >>> b = builder.create_quantum_register(2, name='b')
        >>> builder.isub(b, target=a)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -a[0]-X-@-X-X---
                    | |
        q1: -a[1]-X-X-|-X-X-
                    | | |
        q2: -b[0]---@-@-|---
                        |
        q3: -b[1]-------@---
    """
```

<a name="kickmix.CircuitBuilder.ixor_gf2_div"></a>
```python
# kickmix.CircuitBuilder.ixor_gf2_div

# (in class kickmix.CircuitBuilder)
def ixor_gf2_div(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    target: km.array,
    field: km.GF2Field | None = None,
) -> None:
    """Appends operations to perform `target ^= (lhs / rhs) % modulus`.

    Out-of-place division accumulating the quotient into `target` via
    bitwise XOR. Does not reset `target` or assume it is in the |0>
    state. A zero divisor produces a quotient of 0. Temporary workspace
    is allocated and cleaned up automatically.

    Args:
        lhs: The dividend register. Left unchanged.
        rhs: The divisor register. Left unchanged.
        target: The GF(2^m) register to XOR the quotient into.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> lhs = builder.create_quantum_register(8, name='lhs')
        >>> rhs = builder.create_quantum_register(8, name='rhs')
        >>> target = builder.create_quantum_register(8, name='target')
        >>> builder.ixor_gf2_div(lhs, rhs, target=target)
        >>> builder.finish_circuit().max_magic()
        135
    """
```

<a name="kickmix.CircuitBuilder.ixor_gf2_mul"></a>
```python
# kickmix.CircuitBuilder.ixor_gf2_mul

# (in class kickmix.CircuitBuilder)
def ixor_gf2_mul(
    self,
    lhs: km.array,
    rhs: km.array,
    *,
    target: km.array,
    field: km.GF2Field | None = None,
    control: q | b | bool = True,
) -> None:
    """Appends operations to perform `if control: target ^= (lhs * rhs) % modulus`.

    Out-of-place multiplication accumulating the product into `target`
    via bitwise XOR. Does not reset `target` or assume it is in the
    |0> state.

    Args:
        lhs: The first factor register. Left unchanged.
        rhs: The second factor register. Left unchanged.
        target: The GF(2^m) register to XOR the product into.
        field: Optional `km.GF2Field`. Defaults to
            `km.GF2Field(len(target))`.
        control: Defaults to True. Determines if the operation occurs.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> target = builder.create_quantum_register(2, name='target')
        >>> lhs = builder.create_quantum_register(2, name='lhs')
        >>> rhs = builder.create_quantum_register(2, name='rhs')
        >>> builder.ixor_gf2_mul(lhs, rhs, target=target)
        >>> print(builder.finish_circuit().text_diagram())
        q0: -target[0]-@-SWAP-X-X-SWAP-X-X-SWAP-X-X-@-
                       | |    | | |    | | |    | | |
        q1: -target[1]-X-SWAP-|-@-SWAP-|-@-SWAP-|-@-X-
                              |        |        |
        q2: -lhs[0]------X----@-X------|--------@-----
                         |    | |      |        |
        q3: -lhs[1]------@----|-@------@--------|-----
                              |        |        |
        q4: -rhs[0]------X----@-X------|--------@-----
                         |      |      |
        q5: -rhs[1]------@------@------@--------------
    """
```

<a name="kickmix.CircuitBuilder.left_rotate"></a>
```python
# kickmix.CircuitBuilder.left_rotate

# (in class kickmix.CircuitBuilder)
def left_rotate(
    self,
    target: object,
    *,
    control: object = True,
) -> None:
```

<a name="kickmix.CircuitBuilder.max_qubits"></a>
```python
# kickmix.CircuitBuilder.max_qubits

# (in class kickmix.CircuitBuilder)
@property
def max_qubits(
    self,
) -> object:
    """Returns the maximum number of qubits allowed when building circuits.
    """
```

<a name="kickmix.CircuitBuilder.multi_controlled_x"></a>
```python
# kickmix.CircuitBuilder.multi_controlled_x

# (in class kickmix.CircuitBuilder)
def multi_controlled_x(
    self,
    controls: object,
    target: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.neg"></a>
```python
# kickmix.CircuitBuilder.neg

# (in class kickmix.CircuitBuilder)
def neg(
    self,
) -> None:
```

<a name="kickmix.CircuitBuilder.neg_if"></a>
```python
# kickmix.CircuitBuilder.neg_if

# (in class kickmix.CircuitBuilder)
def neg_if(
    self,
    condition: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.num_allocated_qubits"></a>
```python
# kickmix.CircuitBuilder.num_allocated_qubits

# (in class kickmix.CircuitBuilder)
@property
def num_allocated_qubits(
    self,
) -> int:
    """Returns the current number of allocated qubits.
    """
```

<a name="kickmix.CircuitBuilder.num_free_qubits"></a>
```python
# kickmix.CircuitBuilder.num_free_qubits

# (in class kickmix.CircuitBuilder)
@property
def num_free_qubits(
    self,
) -> object:
    """Returns the maximum number of qubits allowed when building circuits.
    """
```

<a name="kickmix.CircuitBuilder.pop_condition"></a>
```python
# kickmix.CircuitBuilder.pop_condition

# (in class kickmix.CircuitBuilder)
def pop_condition(
    self,
) -> None:
```

<a name="kickmix.CircuitBuilder.push_condition"></a>
```python
# kickmix.CircuitBuilder.push_condition

# (in class kickmix.CircuitBuilder)
def push_condition(
    self,
    condition: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.reset"></a>
```python
# kickmix.CircuitBuilder.reset

# (in class kickmix.CircuitBuilder)
def reset(
    self,
    target: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.right_rotate"></a>
```python
# kickmix.CircuitBuilder.right_rotate

# (in class kickmix.CircuitBuilder)
def right_rotate(
    self,
    target: object,
    *,
    control: object = True,
) -> None:
```

<a name="kickmix.CircuitBuilder.set_max_qubits"></a>
```python
# kickmix.CircuitBuilder.set_max_qubits

# (in class kickmix.CircuitBuilder)
@staticmethod
def set_max_qubits(
    limit: int | float,
) -> None:
    """Sets the maximum number of qubits allowed when building circuits.

    Attempting to allocate more qubits than the maximum will cause the allocation
    to raise an exception. This includes both qubits returned by
    `CircuitBuilder.alloc_qubits` and by `CircuitBuilder.create_quantum_register`.

    Args:
        limit: The maximum number of qubits that can be allocated.
            Set to `float('inf')` to disable the limit.
    """
```

<a name="kickmix.CircuitBuilder.start_auto_free_scope"></a>
```python
# kickmix.CircuitBuilder.start_auto_free_scope

# (in class kickmix.CircuitBuilder)
def start_auto_free_scope(
    self,
) -> None:
    """Starts a scope that automatically frees allocations.

    Call builder.end_auto_free_scope() to free allocations performed during the
    scope. If multiple scopes are being used, they operate in LIFO order (i.e. like
    a stack).
    """
```

<a name="kickmix.CircuitBuilder.swap"></a>
```python
# kickmix.CircuitBuilder.swap

# (in class kickmix.CircuitBuilder)
def swap(
    self,
    target1: object,
    target2: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.unary_iteration"></a>
```python
# kickmix.CircuitBuilder.unary_iteration

# (in class kickmix.CircuitBuilder)
def unary_iteration(
    self,
    address: km.array,
    address_values: range | None = None,
    *,
    control: q | b | bool = True,
) -> km.UnaryIterationCursor:
    """Returns a cursor for stepping through values of `address`.

    Use `with builder.unary_iteration(...) as it:` to open a pass. Inside the
    `with` block, loop over `it` to visit `(address_value, match_qubit)` pairs
    in order, or call `it.move_to(v)` to jump to any value `v`. At each step,
    `it.match_qubit` is a qubit that is ON in the parts of the superposition
    where `control` is active and `address == v`.

    Opening the `with` block allocates `max(1, len(address))` temporary qubits,
    and exiting the block uncomputes and frees them. Do not modify `address`,
    `control`, or `match_qubit` while the pass is open, and only use
    `match_qubit` as a control.

    For a non-empty `address`, positioning at the first in-range value currently
    costs at most `len(address)` Toffoli gates when `control` is a qubit, or
    `len(address) - 1` when `control` is `True` or a classical bit. Moving
    between distinct in-range values `a` and `b` costs at most
    `(a ^ b).bit_length() - 1` Toffoli gates, and closing the pass costs zero
    Toffoli gates.

    Args:
        address: The little-endian address to match against (at most 63 qubits).
        address_values: Defaults to `range(1 << len(address))`. The step-1 `range`
            of values visited by `for` loops.
        control: Gates the iteration when provided. If `False`, no operations are
            emitted and `for` loops visit zero values. Defaults to `True`.

    Returns:
        A reusable cursor over `address` that can be opened with `with`.

    Examples:
        >>> import kickmix as km
        >>> # Sequential iteration over a range of address values:
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(3, name='address')
        >>> target = builder.create_quantum_register(1, name='target')
        >>> with builder.unary_iteration(address, range(4)) as it:
        ...     for address_value, match_qubit in it:
        ...         if address_value == 1:
        ...             builder.cx(match_qubit, target[0])
        >>> circuit = builder.finish_circuit()
        >>> sim = km.Simulator(batch_size=8)
        >>> sim.use_same_registers_as(circuit)
        >>> for a in range(8):
        ...     sim.write_within_shot('address', a, a)
        >>> sim.do(circuit)
        >>> for a in range(8):
        ...     print(f"{a}: {sim.read_within_shot('target', a, out=int)}")
        0: 0
        1: 1
        2: 0
        3: 0
        4: 0
        5: 0
        6: 0
        7: 0

        >>> # Random-access jumps using move_to:
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(3, name='address')
        >>> target = builder.create_quantum_register(1, name='target')
        >>> with builder.unary_iteration(address) as it:
        ...     for address_value in [5, 2, 7]:
        ...         it.move_to(address_value)
        ...         builder.cx(it.match_qubit, target[0])
        >>> circuit = builder.finish_circuit()
        >>> sim = km.Simulator(batch_size=8)
        >>> sim.use_same_registers_as(circuit)
        >>> for a in range(8):
        ...     sim.write_within_shot('address', a, a)
        >>> sim.do(circuit)
        >>> for a in range(8):
        ...     print(f"{a}: {sim.read_within_shot('target', a, out=int)}")
        0: 0
        1: 0
        2: 1
        3: 0
        4: 0
        5: 1
        6: 0
        7: 1
    """
```

<a name="kickmix.CircuitBuilder.x"></a>
```python
# kickmix.CircuitBuilder.x

# (in class kickmix.CircuitBuilder)
def x(
    self,
    target: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.z"></a>
```python
# kickmix.CircuitBuilder.z

# (in class kickmix.CircuitBuilder)
def z(
    self,
    target: object,
) -> None:
```

<a name="kickmix.CircuitBuilder.z_pow"></a>
```python
# kickmix.CircuitBuilder.z_pow

# (in class kickmix.CircuitBuilder)
def z_pow(
    self,
    target: km.q | km.array | Sequence[km.q],
    exponent: float | int | fractions.Fraction | str,
) -> None:
    """Appends Z_POW gates to the circuit.

    This method broadcasts over km.array arguments. If the target is a sequence
    of qubits, rather than a single qubit, an operation is appended for each qubit.

    Args:
        target: The qubit(s) to apply Z_POW to.
        exponent: The exponent of the Z_POW operation.
            The exponent is converted into a fractions.Fraction, then canonicalized
            into the range [0, 2) and rounded to the nearest multiple of 2**-127.

    Examples:
        >>> import kickmix as km

        >>> # Demo a simple z_pow call
        >>> builder = km.CircuitBuilder()
        >>> builder.z_pow(km.q(0), 0.25)
        >>> builder.finish_circuit()
        km.Circuit('''
            Z_POW q0 0.25
        ''')

        >>> # Demo a broadcasting call
        >>> builder = km.CircuitBuilder()
        >>> targets = builder.create_quantum_register(3, "targets")
        >>> builder.z_pow(targets, -0.5)
        >>> builder.finish_circuit()
        km.Circuit('''
            APPEND_TO_REGISTER q0 r0
            APPEND_TO_REGISTER q1 r0
            APPEND_TO_REGISTER q2 r0
            REGISTER r0 "targets"
            Z_POW q0 1.5
            Z_POW q1 1.5
            Z_POW q2 1.5
        ''')
    """
```

<a name="kickmix.GF2Field"></a>
```python
# kickmix.GF2Field

# (at top-level in the kickmix module)
class GF2Field:
    """A binary extension field GF(2^m) with polynomial basis arithmetic.

    Elements are polynomials of degree less than m over GF(2), represented
    in Python as non-negative integers where bit k is the coefficient of x^k.
    Addition is bitwise XOR, and multiplication is polynomial multiplication
    reduced modulo an irreducible polynomial of degree m.

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.degree
        4
        >>> hex(field.modulus)
        '0x13'
        >>> field.primitive_element
        2
        >>> field.mul(3, 5)
        15
    """
```

<a name="kickmix.GF2Field.__eq__"></a>
```python
# kickmix.GF2Field.__eq__

# (in class kickmix.GF2Field)
def __eq__(
    self,
    other: object,
) -> bool:
    """Returns True if other is a GF2Field with the same degree, modulus,
    and primitive_element.
    """
```

<a name="kickmix.GF2Field.__hash__"></a>
```python
# kickmix.GF2Field.__hash__

# (in class kickmix.GF2Field)
def __hash__(
    self,
) -> int:
    """Returns a hash of the field's degree, modulus, and primitive_element.
    """
```

<a name="kickmix.GF2Field.__init__"></a>
```python
# kickmix.GF2Field.__init__

# (in class kickmix.GF2Field)
@overload
def __init__(
    self,
    degree: int | None = None,
    modulus: int | str | Any | None = None,
    primitive_element: int | str | Any | None = None,
) -> None:
    pass
@overload
def __init__(
    self,
    galois_field: Any,
) -> None:
    pass
def __init__(
    self,
    degree: int | Any | None = None,
    modulus: int | str | Any | None = None,
    primitive_element: int | str | Any | None = None,
) -> None:
    """Creates the binary extension field GF(2^degree).

    Can also be constructed directly from a `galois.GF(2**m)` field class
    (e.g. `km.GF2Field(galois.GF(2**8))` or `km.GF2Field.from_galois(GF)`),
    extracting its degree, irreducible polynomial, and primitive element.

    Args:
        degree: The extension degree m (1 <= m <= 512), or a
            `galois.GF(2**m)` field class. If None, m is inferred from
            the degree of `modulus`.
        modulus: Optional irreducible reduction polynomial of degree m,
            encoded as an int where bit k is the coefficient of x^k, a
            `galois.Poly` over GF(2), or a polynomial string such as
            "x^4 + x + 1" or "0x13". If None, a low-weight default
            irreducible polynomial is chosen.
        primitive_element: Optional multiplicative generator of GF(2^m)*
            (of order 2^m - 1), encoded as an int, a `galois.Poly`, or a
            polynomial string. Validated if provided; if None, the
            smallest primitive element in integer order is used.

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.degree
        4
        >>> hex(field.modulus)
        '0x13'
        >>> field.primitive_element
        2
        >>> aes_field = km.GF2Field(8, modulus=0x11B)
        >>> hex(aes_field.modulus)
        '0x11b'
        >>> aes_field.primitive_element
        3
        >>> str_field = km.GF2Field(modulus="x^4 + x^3 + 1")
        >>> str_field == km.GF2Field(4, modulus=0x19)
        True
    """
```

<a name="kickmix.GF2Field.__repr__"></a>
```python
# kickmix.GF2Field.__repr__

# (in class kickmix.GF2Field)
def __repr__(
    self,
) -> str:
    """Returns a string representation of the field that evaluates to an equal field.
    """
```

<a name="kickmix.GF2Field.__str__"></a>
```python
# kickmix.GF2Field.__str__

# (in class kickmix.GF2Field)
def __str__(
    self,
) -> str:
    """Returns a human-readable summary of the field.
    """
```

<a name="kickmix.GF2Field.add"></a>
```python
# kickmix.GF2Field.add

# (in class kickmix.GF2Field)
def add(
    self,
    a: int,
    b: int,
) -> int:
    """Returns (a + b) mod modulus in GF(2^m), which is bitwise XOR.

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.add(3, 5)
        6
    """
```

<a name="kickmix.GF2Field.degree"></a>
```python
# kickmix.GF2Field.degree

# (in class kickmix.GF2Field)
@property
def degree(
    self,
) -> int:
    """The extension degree m of the field GF(2^m).
    """
```

<a name="kickmix.GF2Field.div"></a>
```python
# kickmix.GF2Field.div

# (in class kickmix.GF2Field)
def div(
    self,
    a: int,
    b: int,
) -> int:
    """Returns (a / b) mod modulus in GF(2^m), or 0 when b is 0.

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> q = field.div(6, 3)
        >>> q
        2
        >>> field.mul(q, 3)
        6
        >>> field.div(6, 0)
        0
    """
```

<a name="kickmix.GF2Field.frobenius"></a>
```python
# kickmix.GF2Field.frobenius

# (in class kickmix.GF2Field)
def frobenius(
    self,
    a: int,
    k: int = 1,
) -> int:
    """Returns a^(2^k) mod modulus in GF(2^m).

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.frobenius(3, 1)
        5
        >>> field.frobenius(3, 4) == 3  # in GF(2^4), a^(2^4) == a
        True
    """
```

<a name="kickmix.GF2Field.from_galois"></a>
```python
# kickmix.GF2Field.from_galois

# (in class kickmix.GF2Field)
@staticmethod
def from_galois(
    galois_field: Any,
) -> km.GF2Field:
    """Creates a `km.GF2Field` from a `galois.GF(2**m)` field class.

    Args:
        galois_field: A binary field class created by `galois.GF(2**m)`.

    Examples:
        >>> import kickmix as km
        >>> import galois  # doctest: +SKIP
        >>> GF = galois.GF(2**8, irreducible_poly=0x11B)  # doctest: +SKIP
        >>> field = km.GF2Field.from_galois(GF)  # doctest: +SKIP
        >>> field  # doctest: +SKIP
        km.GF2Field(8, modulus=0x11B)
        >>> field.primitive_element == int(GF.primitive_element)  # doctest: +SKIP
        True
    """
```

<a name="kickmix.GF2Field.invert"></a>
```python
# kickmix.GF2Field.invert

# (in class kickmix.GF2Field)
def invert(
    self,
    a: int,
) -> int:
    """Returns the multiplicative inverse of a in GF(2^m), or 0 when a is 0.

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> inv = field.invert(3)
        >>> inv
        14
        >>> field.mul(3, inv)
        1
        >>> field.invert(0)
        0
    """
```

<a name="kickmix.GF2Field.is_element"></a>
```python
# kickmix.GF2Field.is_element

# (in class kickmix.GF2Field)
def is_element(
    self,
    a: int,
) -> bool:
    """Returns True if a is a valid element of GF(2^m) (0 <= a < 2**degree).

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.is_element(15)
        True
        >>> field.is_element(16)
        False
        >>> field.is_element(-1)
        False
    """
```

<a name="kickmix.GF2Field.is_irreducible"></a>
```python
# kickmix.GF2Field.is_irreducible

# (in class kickmix.GF2Field)
@staticmethod
def is_irreducible(
    poly: int,
) -> bool:
    """Returns True if poly is irreducible over GF(2).

    Examples:
        >>> import kickmix as km
        >>> km.GF2Field.is_irreducible(0x13)  # x^4 + x + 1
        True
        >>> km.GF2Field.is_irreducible(0x15)  # x^4 + x^2 + 1 = (x^2 + x + 1)^2
        False
    """
```

<a name="kickmix.GF2Field.is_primitive_element"></a>
```python
# kickmix.GF2Field.is_primitive_element

# (in class kickmix.GF2Field)
def is_primitive_element(
    self,
    a: int,
) -> bool:
    """Returns True if a is a primitive element of GF(2^m) (multiplicative
    order 2**degree - 1).

    Examples:
        >>> import kickmix as km
        >>> aes = km.GF2Field(8, modulus=0x11B)
        >>> aes.is_primitive_element(2)
        False
        >>> aes.is_primitive_element(3)
        True
    """
```

<a name="kickmix.GF2Field.mod"></a>
```python
# kickmix.GF2Field.mod

# (in class kickmix.GF2Field)
def mod(
    self,
    a: int,
) -> int:
    """Reduces a polynomial mod the field's irreducible polynomial.

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.mod(0x13)  # reduction modulo modulus (0x13)
        0
        >>> field.mod(0x15)
        6
    """
```

<a name="kickmix.GF2Field.modulus"></a>
```python
# kickmix.GF2Field.modulus

# (in class kickmix.GF2Field)
@property
def modulus(
    self,
) -> int:
    """The irreducible reduction polynomial of degree m, including the x^m bit.
    """
```

<a name="kickmix.GF2Field.mul"></a>
```python
# kickmix.GF2Field.mul

# (in class kickmix.GF2Field)
def mul(
    self,
    a: int,
    b: int,
) -> int:
    """Returns the product (a * b) mod modulus in GF(2^m).

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.mul(3, 5)
        15
    """
```

<a name="kickmix.GF2Field.pow"></a>
```python
# kickmix.GF2Field.pow

# (in class kickmix.GF2Field)
def pow(
    self,
    a: int,
    exponent: int,
) -> int:
    """Returns a**exponent mod modulus in GF(2^m). Negative exponents invert a.

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.pow(3, 2)
        5
        >>> field.pow(3, -1)
        14
        >>> field.mul(3, 14)
        1
    """
```

<a name="kickmix.GF2Field.primitive_element"></a>
```python
# kickmix.GF2Field.primitive_element

# (in class kickmix.GF2Field)
@property
def primitive_element(
    self,
) -> int:
    """The primitive element (multiplicative generator of order 2**m - 1) of GF(2^m).
    """
```

<a name="kickmix.GF2Field.square"></a>
```python
# kickmix.GF2Field.square

# (in class kickmix.GF2Field)
def square(
    self,
    a: int,
) -> int:
    """Returns a^2 mod modulus in GF(2^m).

    Examples:
        >>> import kickmix as km
        >>> field = km.GF2Field(4)
        >>> field.square(3)
        5
    """
```

<a name="kickmix.Simulator"></a>
```python
# kickmix.Simulator

# (at top-level in the kickmix module)
class Simulator:
    """An interactive kickmix simulator.
    """
```

<a name="kickmix.Simulator.__init__"></a>
```python
# kickmix.Simulator.__init__

# (in class kickmix.Simulator)
@staticmethod
def __init__(
    batch_size: int,
):
    """Initializes an interactive simulator with the given batch size.

    Args:
        batch_size: Determines how many simultaneous shots are being tracked by the
            simulator.
    """
```

<a name="kickmix.Simulator.batch_size"></a>
```python
# kickmix.Simulator.batch_size

# (in class kickmix.Simulator)
@property
def batch_size(
    self,
) -> int:
    """The number of shots being tracked by the simulator.
    """
```

<a name="kickmix.Simulator.clear_for_shot"></a>
```python
# kickmix.Simulator.clear_for_shot

# (in class kickmix.Simulator)
def clear_for_shot(
    self,
) -> None:
    """Zeroes all data that should be reset between shots from a circuit.

    When using the simulator to repeatedly sample from a circuit, it is important
    to call clear_for_shot between calls. Otherwise state modified by an earlier
    shot may end up affecting a later shot.

    This method clears:
    - tracked qubit values
    - tracked bit values
    - tracked phase values
    - the condition stack

    Examples:
        >>> import kickmix as km
        >>> sim = km.Simulator(batch_size=8)
        >>> sim.do(km.Circuit('''
        ...     X q0
        ...     PUSH_CONDITION if b0
        ... '''))
        >>> sim.read_across_shots(km.q(0), out=int)
        255

        >>> sim.clear_for_shot()
        >>> sim.read_across_shots(km.q(0), out=int)
        0
    """
```

<a name="kickmix.Simulator.do"></a>
```python
# kickmix.Simulator.do

# (in class kickmix.Simulator)
def do(
    self,
    circuit: km.Circuit,
):
    """Applies the instructions from the circuit to the simulator's state.

    Args:
        circuit: The circuit with instructions to apply.
    """
```

<a name="kickmix.Simulator.read_across_shots"></a>
```python
# kickmix.Simulator.read_across_shots

# (in class kickmix.Simulator)
def read_across_shots(
    self,
    indices: object,
    *,
    out: object = None,
) -> object:
```

<a name="kickmix.Simulator.read_phase_flipped_across_shots"></a>
```python
# kickmix.Simulator.read_phase_flipped_across_shots

# (in class kickmix.Simulator)
def read_phase_flipped_across_shots(
    self,
    *,
    out: object = None,
) -> object:
    """Reads whether or not each shot is phase flipped.

    A shot is considered phase flipped if its tracked phase is closer to a
    half turn than to no turns.

    Args:
        out: Where to store the output. Defaults to None (allocate a new numpy
            array). Can be set to a numpy array to write the results into, or
            to `int` to return the result bit packed into an integer.

    Returns:
        The phase flip information.
    """
```

<a name="kickmix.Simulator.read_shot_phase"></a>
```python
# kickmix.Simulator.read_shot_phase

# (in class kickmix.Simulator)
def read_shot_phase(
    self,
    shot_index: int,
) -> object:
    """Returns the tracked global phase of a shot.

    The phase is affected by operations like Z, HMR, and Z_POW.

    Returns:
        The phase, in half turns, as an int or as a `fractions.Fraction`.

        If the shot is exactly unphased (0 radians), the int 0 is returned.
        If the shot is exactly phase flipped (pi radians), the int 1 is returned.
        If the phase is some other value, a fractions.Fraction with the exact
        phase value (in half turns) is returned.

    Examples:
        >>> import kickmix as km
        >>> sim = km.Simulator(batch_size=5)
        >>> sim.read_shot_phase(shot_index=4)
        0

        >>> sim.do(km.Circuit('''
        ...     X q0
        ...     Z q0
        ... '''))
        >>> sim.read_shot_phase(shot_index=4)
        1

        >>> sim.do(km.Circuit('''
        ...     Z_POW q0 1.25000000000000001387778780781445675529539585113525390625
        ... '''))
        >>> sim.read_shot_phase(shot_index=4)
        Fraction(18014398509481985, 72057594037927936)
    """
```

<a name="kickmix.Simulator.read_within_shot"></a>
```python
# kickmix.Simulator.read_within_shot

# (in class kickmix.Simulator)
def read_within_shot(
    self,
    indices: object,
    shot_index: object,
    *,
    out: object = None,
) -> object:
```

<a name="kickmix.Simulator.use_same_registers_as"></a>
```python
# kickmix.Simulator.use_same_registers_as

# (in class kickmix.Simulator)
def use_same_registers_as(
    self,
    arg0: kickmix.Circuit,
) -> None:
```

<a name="kickmix.Simulator.write_across_shots"></a>
```python
# kickmix.Simulator.write_across_shots

# (in class kickmix.Simulator)
def write_across_shots(
    self,
    index: object,
    new_value: object,
) -> None:
    """Sets the tracked value of the given qubits or bits, across all shots.

    Args:
        index: The qubit id or bit id to write to, or an array of them
            (e.g. a register name).
        new_value: The value to write.

            If `index` is a single qubit or bit, this is an `int` or an
            `np.ndarray` with dtype=np.bool_ and shape=(sim.batch_size,).

            If `index` is an array of length n, this is a sequence of n
            `int`s or an `np.ndarray` with dtype=np.bool_ and
            shape=(n, sim.batch_size).

            Each `int` is bit packed in little-endian order, so the bit
            for shot k is `(new_value >> k) & 1`. This matches the
            layout returned by `read_across_shots(..., out=int)`.

    Raises:
        IndexError: The given index isn't a qubit id or bit id.
        ValueError: The given new_value isn't valid.
    """
```

<a name="kickmix.Simulator.write_shot_phase"></a>
```python
# kickmix.Simulator.write_shot_phase

# (in class kickmix.Simulator)
def write_shot_phase(
    self,
    shot_index: int,
    new_phase_half_turns: object,
) -> None:
    """Sets the tracked global phase of a shot.

    The phase is affected by operations like Z, HMR, and Z_POW.

    Args:
        shot_index: The shot whose phase is being written.
        new_phase_half_turns: The new phase, as a rotation in half turn units.
            This can be an int, a float, or a fractions.Fraction.
            The value will be canonicalized into the range [0, 2) and then
            rounded to the nearest multiple of 2**-127.

    Examples:
        >>> import kickmix as km
        >>> sim = km.Simulator(batch_size=5)

        >>> sim.write_shot_phase(4, 0.125)
        >>> sim.read_shot_phase(4)
        Fraction(1, 8)

        >>> sim.write_shot_phase(4, 1.0)
        >>> sim.read_shot_phase(4)
        1
    """
```

<a name="kickmix.Simulator.write_within_shot"></a>
```python
# kickmix.Simulator.write_within_shot

# (in class kickmix.Simulator)
def write_within_shot(
    self,
    index: object,
    shot_index: object,
    new_value: object,
) -> None:
```

<a name="kickmix.UnaryIterationCursor"></a>
```python
# kickmix.UnaryIterationCursor

# (at top-level in the kickmix module)
class UnaryIterationCursor:
    """A cursor that steps through values of an address register.

    Create a cursor with `builder.unary_iteration(address, ...)` and open a pass
    using a `with` block. Inside the block, iterate with a `for` loop to visit
    `(address_value, match_qubit)` pairs in order, or call `move_to` to jump to
    any value.

    Entering the `with` block allocates `max(1, len(address))` temporary qubits;
    leaving the block uncomputes and frees them. A cursor that is never opened
    adds no operations to the circuit, and the same cursor can be reused across
    multiple `with` blocks.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(2, name='address')
        >>> target = builder.create_quantum_register(1, name='target')
        >>> cursor = builder.unary_iteration(address)
        >>> with cursor:
        ...     for address_value, match_qubit in cursor:
        ...         if address_value == 3:
        ...             builder.cx(match_qubit, target[0])
        >>> # Reusing the same cursor to uncompute:
        >>> with cursor:
        ...     for address_value, match_qubit in cursor:
        ...         if address_value == 3:
        ...             builder.cx(match_qubit, target[0])
        >>> circuit = builder.finish_circuit()
        >>> sim = km.Simulator(batch_size=4)
        >>> sim.use_same_registers_as(circuit)
        >>> for a in range(4):
        ...     sim.write_within_shot('address', a, a)
        >>> sim.do(circuit)
        >>> for a in range(4):
        ...     print(f"{a}: {sim.read_within_shot('target', a, out=int)}")
        0: 0
        1: 0
        2: 0
        3: 0
    """
```

<a name="kickmix.UnaryIterationCursor.__enter__"></a>
```python
# kickmix.UnaryIterationCursor.__enter__

# (in class kickmix.UnaryIterationCursor)
def __enter__(
    self,
) -> km.UnaryIterationCursor:
    """Opens a unary iteration pass and allocates temporary qubits.

    No gates are emitted on entry. The cursor starts at `1 << len(address)` with
    `match_qubit` set to `|0>` until the first `for` step or `move_to` call.

    Returns:
        The open cursor (`self`).

    Raises:
        ValueError: If the cursor is already open.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(2, name='address')
        >>> with builder.unary_iteration(address) as it:
        ...     it.cur_address_value
        4
    """
```

<a name="kickmix.UnaryIterationCursor.__exit__"></a>
```python
# kickmix.UnaryIterationCursor.__exit__

# (in class kickmix.UnaryIterationCursor)
def __exit__(
    self,
    exc_type,
    exc_value,
    traceback,
) -> bool:
    """Closes the active pass, uncomputing and freeing its temporary qubits.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(2, name='address')
        >>> with builder.unary_iteration(address) as it:
        ...     it.move_to(1)
        ...     print(builder.num_allocated_qubits)
        4
        >>> builder.num_allocated_qubits
        2
    """
```

<a name="kickmix.UnaryIterationCursor.__init__"></a>
```python
# kickmix.UnaryIterationCursor.__init__

# (in class kickmix.UnaryIterationCursor)
def __init__(
    self,
    builder: km.CircuitBuilder,
    address: km.array,
    address_values: range | None = None,
    *,
    control: q | b | bool = True,
) -> None:
    """Initializes a reusable unary iteration cursor over `address`.

    Args:
        builder: Where circuit operations and temporary qubit allocations are
            recorded.
        address: The little-endian address to match against (at most 63 qubits).
        address_values: Defaults to `range(1 << len(address))`. The step-1 `range`
            of values visited by `for` loops.
        control: Gates the iteration when provided. If `False`, no operations are
            emitted and `for` loops visit zero values. Defaults to `True`.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(2, name='address')
        >>> target = builder.create_quantum_register(1, name='target')
        >>> cursor = km.UnaryIterationCursor(builder, address)
        >>> with cursor:
        ...     cursor.move_to(2)
        ...     builder.cx(cursor.match_qubit, target[0])
        >>> builder.finish_circuit().max_magic()
        1
    """
```

<a name="kickmix.UnaryIterationCursor.__iter__"></a>
```python
# kickmix.UnaryIterationCursor.__iter__

# (in class kickmix.UnaryIterationCursor)
def __iter__(
    self,
) -> Iterator[Tuple[int, q]]:
    """Yields `(address_value, match_qubit)` pairs for each configured value.

    The cursor must already be open inside a `with` block. Finishing or breaking
    out of the loop leaves the pass open until the `with` block exits.

    Raises:
        ValueError: If the cursor is not open or is already iterating.
    """
```

<a name="kickmix.UnaryIterationCursor.__next__"></a>
```python
# kickmix.UnaryIterationCursor.__next__

# (in class kickmix.UnaryIterationCursor)
def __next__(
    self,
) -> Tuple[int, q]:
    """Advances to the next value and returns `(address_value, match_qubit)`.

    Raises:
        StopIteration: When all values in `address_values` have been visited.
        ValueError: If the cursor is not open.
    """
```

<a name="kickmix.UnaryIterationCursor.__repr__"></a>
```python
# kickmix.UnaryIterationCursor.__repr__

# (in class kickmix.UnaryIterationCursor)
def __repr__(
    self,
) -> str:
    """Returns a string describing the cursor.
    """
```

<a name="kickmix.UnaryIterationCursor.close"></a>
```python
# kickmix.UnaryIterationCursor.close

# (in class kickmix.UnaryIterationCursor)
def close(
    self,
) -> None:
    """Closes the active pass, uncomputing and freeing its temporary qubits.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(2, name='address')
        >>> with builder.unary_iteration(address) as it:
        ...     it.move_to(1)
        ...     print(builder.num_allocated_qubits)
        ...     it.close()
        ...     print(builder.num_allocated_qubits)
        4
        2
    """
```

<a name="kickmix.UnaryIterationCursor.cur_address_value"></a>
```python
# kickmix.UnaryIterationCursor.cur_address_value

# (in class kickmix.UnaryIterationCursor)
@property
def cur_address_value(
    self,
) -> int:
    """Returns the address value the cursor is currently positioned at.

    When a pass is first opened, `cur_address_value` is `1 << len(address)` and
    `match_qubit` is `|0>`.

    Raises:
        ValueError: If the cursor is not open.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(2, name='address')
        >>> with builder.unary_iteration(address) as it:
        ...     it.move_to(3)
        ...     it.cur_address_value
        3
    """
```

<a name="kickmix.UnaryIterationCursor.match_qubit"></a>
```python
# kickmix.UnaryIterationCursor.match_qubit

# (in class kickmix.UnaryIterationCursor)
@property
def match_qubit(
    self,
) -> q:
    """Returns the qubit that is ON in the parts of the superposition where `control`
    is active and `address == cur_address_value`.

    Use this qubit only as a control. When `cur_address_value` is outside
    `range(1 << len(address))`, this qubit is `|0>`.

    Raises:
        ValueError: If the cursor is not open.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(2, name='address')
        >>> target = builder.create_quantum_register(1, name='target')
        >>> with builder.unary_iteration(address) as it:
        ...     it.move_to(2)
        ...     builder.cx(it.match_qubit, target[0])
    """
```

<a name="kickmix.UnaryIterationCursor.move_to"></a>
```python
# kickmix.UnaryIterationCursor.move_to

# (in class kickmix.UnaryIterationCursor)
def move_to(
    self,
    address_value: int,
) -> None:
    """Moves the cursor to `address_value` and updates `match_qubit`.

    For a non-empty `address`, moving from out of range to an in-range value
    currently costs at most `len(address)` Toffoli gates when `control` is a
    qubit, or `len(address) - 1` when `control` is `True` or a classical bit.
    Moving between distinct in-range values `a` and `b` costs at most
    `(a ^ b).bit_length() - 1` Toffoli gates. Moving to any value outside
    `range(1 << len(address))` sets `match_qubit` to `|0>` with zero Toffoli
    gates while recording `cur_address_value == address_value`.

    Args:
        address_value: The target value to position the cursor at.

    Raises:
        ValueError: If the cursor is not open.

    Examples:
        >>> import kickmix as km
        >>> builder = km.CircuitBuilder()
        >>> address = builder.create_quantum_register(2, name='address')
        >>> with builder.unary_iteration(address) as it:
        ...     it.move_to(2)
        ...     it.cur_address_value
        2
    """
```

<a name="kickmix.array"></a>
```python
# kickmix.array

# (at top-level in the kickmix module)
class array:
    """An immutable array of qubit ids, bit ids, and/or boolean values.

    Circuit creation methods operate substantially faster when given a
    km.array,  rather than a normal python list of values, because a
    km.array's values are stored contiguously in memory alongside compressed
    type information.
    """
```

<a name="kickmix.array.__add__"></a>
```python
# kickmix.array.__add__

# (in class kickmix.array)
def __add__(
    self,
    arg0: kickmix.array,
) -> kickmix.array:
    """Returns the concatenation of two arrays.
    """
```

<a name="kickmix.array.__eq__"></a>
```python
# kickmix.array.__eq__

# (in class kickmix.array)
def __eq__(
    self,
    arg0: kickmix.array,
) -> bool:
    """Determines if two km.arrays have identical contents.
    """
```

<a name="kickmix.array.__getitem__"></a>
```python
# kickmix.array.__getitem__

# (in class kickmix.array)
def __getitem__(
    self,
    index: object,
) -> object:
    """Returns an item or slice from the array.

    Arg:
        index: An int or a slice.

    Returns:
         If the index is an int: the item at that index.
         If the index is a slice: a view of the array.
    """
```

<a name="kickmix.array.__len__"></a>
```python
# kickmix.array.__len__

# (in class kickmix.array)
def __len__(
    self,
) -> int:
    """Returns the length of the array.
    """
```

<a name="kickmix.array.__ne__"></a>
```python
# kickmix.array.__ne__

# (in class kickmix.array)
def __ne__(
    self,
    arg0: kickmix.array,
) -> bool:
    """Determines if two km.arrays have different contents.
    """
```

<a name="kickmix.array.__repr__"></a>
```python
# kickmix.array.__repr__

# (in class kickmix.array)
def __repr__(
    self,
) -> str:
    """Returns a parseable text representation of the array.
    """
```

<a name="kickmix.array.__str__"></a>
```python
# kickmix.array.__str__

# (in class kickmix.array)
def __str__(
    self,
) -> str:
    """Returns a text representation of the array.
    """
```

<a name="kickmix.b"></a>
```python
# kickmix.b

# (at top-level in the kickmix module)
class b:
    """A kickmix bit index.
    """
```

<a name="kickmix.b.__eq__"></a>
```python
# kickmix.b.__eq__

# (in class kickmix.b)
def __eq__(
    self,
    arg0: kickmix.b,
) -> bool:
    """Determines if two bit identifiers are equal.
    """
```

<a name="kickmix.b.__hash__"></a>
```python
# kickmix.b.__hash__

# (in class kickmix.b)
def __hash__(
    self,
) -> int:
    """Returns a hash of the bit.
    """
```

<a name="kickmix.b.__init__"></a>
```python
# kickmix.b.__init__

# (in class kickmix.b)
def __init__(
    self,
    arg: int,
    /,
) -> None:
    """Initializes a bit identifier with the given value.
    """
```

<a name="kickmix.b.__lt__"></a>
```python
# kickmix.b.__lt__

# (in class kickmix.b)
def __lt__(
    self,
    arg0: object,
) -> bool:
    """Does a less-than comparison.
    """
```

<a name="kickmix.b.__ne__"></a>
```python
# kickmix.b.__ne__

# (in class kickmix.b)
def __ne__(
    self,
    arg0: kickmix.b,
) -> bool:
    """Determines if two bit identifiers are not equal.
    """
```

<a name="kickmix.b.__repr__"></a>
```python
# kickmix.b.__repr__

# (in class kickmix.b)
def __repr__(
    self,
) -> str:
    """Returns a parseable text representation of the bit.
    """
```

<a name="kickmix.b.__str__"></a>
```python
# kickmix.b.__str__

# (in class kickmix.b)
def __str__(
    self,
) -> str:
    """Returns a description of the bit identifier.
    """
```

<a name="kickmix.b.id"></a>
```python
# kickmix.b.id

# (in class kickmix.b)
@property
def id(
    self,
) -> int:
    """Returns the index of the bit.
    """
```

<a name="kickmix.q"></a>
```python
# kickmix.q

# (at top-level in the kickmix module)
class q:
    """A kickmix qubit index.
    """
```

<a name="kickmix.q.__eq__"></a>
```python
# kickmix.q.__eq__

# (in class kickmix.q)
def __eq__(
    self,
    arg0: kickmix.q,
) -> bool:
    """Determines if two QIds are equal.
    """
```

<a name="kickmix.q.__hash__"></a>
```python
# kickmix.q.__hash__

# (in class kickmix.q)
def __hash__(
    self,
) -> int:
    """Returns a hash of the bit.
    """
```

<a name="kickmix.q.__init__"></a>
```python
# kickmix.q.__init__

# (in class kickmix.q)
def __init__(
    self,
    arg: int,
    /,
) -> None:
    """Returns a q with the given index.
    """
```

<a name="kickmix.q.__lt__"></a>
```python
# kickmix.q.__lt__

# (in class kickmix.q)
@staticmethod
def __lt__(
    *args,
    **kwargs,
):
    """Overloaded function.

    1. __lt__(self: kickmix.q, arg0: object) -> bool

    Does a less-than comparison.


    2. __lt__(self: kickmix.q, arg0: kickmix.q) -> bool

    Compares two QIds.
    """
```

<a name="kickmix.q.__ne__"></a>
```python
# kickmix.q.__ne__

# (in class kickmix.q)
def __ne__(
    self,
    arg0: kickmix.q,
) -> bool:
    """Determines if two QIds are not equal.
    """
```

<a name="kickmix.q.__repr__"></a>
```python
# kickmix.q.__repr__

# (in class kickmix.q)
def __repr__(
    self,
) -> str:
    """Returns a parseable text representation of the q.
    """
```

<a name="kickmix.q.__str__"></a>
```python
# kickmix.q.__str__

# (in class kickmix.q)
def __str__(
    self,
) -> str:
    """Returns a description of the q.
    """
```

<a name="kickmix.q.id"></a>
```python
# kickmix.q.id

# (in class kickmix.q)
@property
def id(
    self,
) -> int:
    """Returns the index of the q.
    """
```

<a name="kickmix.r"></a>
```python
# kickmix.r

# (at top-level in the kickmix module)
class r:
    """A kickmix register index.
    """
```

<a name="kickmix.r.__eq__"></a>
```python
# kickmix.r.__eq__

# (in class kickmix.r)
def __eq__(
    self,
    arg0: kickmix.r,
) -> bool:
    """Determines if two register identifiers are equal.
    """
```

<a name="kickmix.r.__hash__"></a>
```python
# kickmix.r.__hash__

# (in class kickmix.r)
def __hash__(
    self,
) -> int:
    """Returns a hash of the bit.
    """
```

<a name="kickmix.r.__init__"></a>
```python
# kickmix.r.__init__

# (in class kickmix.r)
def __init__(
    self,
    arg: int,
    /,
) -> None:
    """Returns a register identifier with the given index.
    """
```

<a name="kickmix.r.__ne__"></a>
```python
# kickmix.r.__ne__

# (in class kickmix.r)
def __ne__(
    self,
    arg0: kickmix.r,
) -> bool:
    """Determines if two register identifiers are not equal.
    """
```

<a name="kickmix.r.__repr__"></a>
```python
# kickmix.r.__repr__

# (in class kickmix.r)
def __repr__(
    self,
) -> str:
    """Returns a parseable text representation of the bit.
    """
```

<a name="kickmix.r.__str__"></a>
```python
# kickmix.r.__str__

# (in class kickmix.r)
def __str__(
    self,
) -> str:
    """Returns a description of the register identifier.
    """
```

<a name="kickmix.r.id"></a>
```python
# kickmix.r.id

# (in class kickmix.r)
@property
def id(
    self,
) -> int:
    """Returns the index of the bit.
    """
```

<a name="kickmix.xb"></a>
```python
# kickmix.xb

# (at top-level in the kickmix module)
class xb:
    """A kickmix bit index being interpreted as an X basis value.

    X basis values act like controls when given as the targets
    of bit flip operations. For example, cx(BitId(0), XBitId(1)) is equivalent
    to cz(BitId(0), BitId(1)).
    """
```

<a name="kickmix.xb.__eq__"></a>
```python
# kickmix.xb.__eq__

# (in class kickmix.xb)
def __eq__(
    self,
    arg0: kickmix.xb,
) -> bool:
    """Determines if two XBIds are equal.
    """
```

<a name="kickmix.xb.__hash__"></a>
```python
# kickmix.xb.__hash__

# (in class kickmix.xb)
def __hash__(
    self,
) -> int:
    """Returns a hash of the xb.
    """
```

<a name="kickmix.xb.__init__"></a>
```python
# kickmix.xb.__init__

# (in class kickmix.xb)
def __init__(
    self,
    arg: int,
    /,
) -> None:
    """Returns an xb with the given index.
    """
```

<a name="kickmix.xb.__lt__"></a>
```python
# kickmix.xb.__lt__

# (in class kickmix.xb)
def __lt__(
    self,
    arg0: object,
) -> bool:
    """Does a less-than comparison.
    """
```

<a name="kickmix.xb.__ne__"></a>
```python
# kickmix.xb.__ne__

# (in class kickmix.xb)
def __ne__(
    self,
    arg0: kickmix.xb,
) -> bool:
    """Determines if two XBIds are not equal.
    """
```

<a name="kickmix.xb.__repr__"></a>
```python
# kickmix.xb.__repr__

# (in class kickmix.xb)
def __repr__(
    self,
) -> str:
    """Returns a parseable text representation of the xb.
    """
```

<a name="kickmix.xb.__str__"></a>
```python
# kickmix.xb.__str__

# (in class kickmix.xb)
def __str__(
    self,
) -> str:
    """Returns a description of the xb.
    """
```

<a name="kickmix.xb.id"></a>
```python
# kickmix.xb.id

# (in class kickmix.xb)
@property
def id(
    self,
) -> int:
    """Returns the index of the xb.
    """
```

<a name="kickmix.xbool"></a>
```python
# kickmix.xbool

# (at top-level in the kickmix module)
class xbool:
    """A boolean that represents an X basis value.

    If the boolean is True, the X basis value is |->.
    If the boolean is False, the X basis value is |+>.

    X basis values act like controls when given as the targets
    of bit flip operations. For example, cx(a, xbool(b)) is equivalent
    to cz(a, b).
    """
```

<a name="kickmix.xbool.__eq__"></a>
```python
# kickmix.xbool.__eq__

# (in class kickmix.xbool)
def __eq__(
    self,
    arg0: kickmix.xbool,
) -> bool:
    """Determines if two xbools are equal.
    """
```

<a name="kickmix.xbool.__hash__"></a>
```python
# kickmix.xbool.__hash__

# (in class kickmix.xbool)
def __hash__(
    self,
) -> int:
    """Returns a hash of the bit.
    """
```

<a name="kickmix.xbool.__init__"></a>
```python
# kickmix.xbool.__init__

# (in class kickmix.xbool)
def __init__(
    self,
    arg: bool,
    /,
) -> None:
    """Returns an xbool of the given boolean.
    """
```

<a name="kickmix.xbool.__lt__"></a>
```python
# kickmix.xbool.__lt__

# (in class kickmix.xbool)
def __lt__(
    self,
    arg0: object,
) -> bool:
    """Does a less-than comparison.
    """
```

<a name="kickmix.xbool.__ne__"></a>
```python
# kickmix.xbool.__ne__

# (in class kickmix.xbool)
def __ne__(
    self,
    arg0: kickmix.xbool,
) -> bool:
    """Determines if two xbools are not equal.
    """
```

<a name="kickmix.xbool.__repr__"></a>
```python
# kickmix.xbool.__repr__

# (in class kickmix.xbool)
def __repr__(
    self,
) -> str:
    """Returns a parseable text representation of the xbool.
    """
```
