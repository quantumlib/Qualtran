This document describes various conventions used by the kickmix API.

# init/del builder methods

`init_` methods prepare a value given that the target qubits start in a clean state.

`del_` methods uncompute a value, leaving the target qubits in a clean state.

Every `init_` has a matching `del_` (and vice versa).
For example, `builder.init_and` is paired with `builder.del_and`.

`init_` methods enforce that their targets are clean by explicitly reset-ing the qubits being initialized.
For example, `builder.init_and(a, b, t)` is equivalent to `builder.reset(t)` then `builder.ccx(a, b, t)`.

`init_` methods return the target being initialized.
For example, `builder.init_and(a, b, t)` returns `t`.

`init_` methods can have their target set to the special string `"alloc"`,  which means the builder allocates qubits or bits for the target.
For example, `t = builder.init_and(a, b, "alloc")` is equivalent to `t = builder.alloc_qubits(1)[0]` then `builder.init_and(a, b, t)`.
The target argument *may* default to "alloc", so that for example `t = builder.init_and(a, b, "alloc")`
can be shortened to `t = builder.init_and(a, b)`.

`init_` methods can initialize multiple things, including unwanted things.
For example, `result, scaffold = builder.init_gf2_div_with_scaffold(g)` initializes the desired division but also
produces temporary garbage qubits (the scaffold) that will later be uncomputed by the corresponding `del_gf2_div_with_scaffold`.
(It saves Toffolis to keep the scaffold around rather than compute and uncompute it multiple times.)

(NOT DECIDED YET): When methods perform an inplace mutation but produce a scaffold, how are they named? For example, an inplace modular inverse that produces some garbage that's inconvenient to remove until the inverting is undone. `inplace_op_with_init_scaffold`? `init_inplace_op_with_scaffold`? `iinit_op_with_scaffold`? `initscaffold_inplace_op`?


# gate broadcasting

All operations that act on individual qubits can be given arrays instead of single qubits, with the result being numpy-like broadcasting.

For example, `ccx([q0, q1], q2, [q3, q4])` is equivalent to `ccx(q0, q2, q3)` then `ccx(q1, q2, q4)`.

# little endianness

All multi-qubit registers default to being interpreted as little-endian.

For quints, the qubit at offset 0 is the 1s qubit, the qubit at offset 1 is the 2s qubit, the qubit at offset 2 is the 4s qubit, and so forth. The qubit at offset k is the `2**k`'s qubit.

For gf2 polynomials, the qubit at offset k is the coefficient for `x**k`.

# truncation / padding

(NOT DECIDED YET) When are registers automatically grown or shrunk in size in order to make an operation go through? When is an exception raised?

# argument names, types, and styles

Control qubits in the python API should be named "control" and have type "km.q | km.b | bool". In most cases the control should default to "True", meaning it is unused unless the user opts into using it.

Control qubits in C++ implementations have type "control: kickmix.QubitOrTrue" (handle the other cases while crossing the boundary).

The first two arguments to comparison methods are called "lhs" and "rhs" and can be specified positionally.

Prefer named arguments to positional arguments, to maintain future API flexibility. Positional arguments are not banned, they are particularly useful for primitive operations where the order is clear such as "ccx", but think twice before not requiring an argument to be named when calling the method. We can always turn a name-required argument into a position argument, but the reverse direction is not possible while preserving backwards compatibility.

When arguments are specified positionally, match the order of arguments to the order of words in the name of the method. For example, `write_shot_phase` should have the shot index argument before the phase value argument because "shot" comes before "phase" in the name. Similarly, "ccx" takes two control arguments and then the target argument.


# eval-able repr

Custom `__repr__` methods should produce a result that is a valid Python expression that can be parsed.

Concretely, the goal is to satisfy the following check:

`assert eval(repr(val), {'km': km}, {}) == val`

Note this has the side benefit of forcing states to be describeable in the first place (e.g. when initializing a simulator it should be possible to specify the states of the qubits and so forth, rather than needing to achieve them indirectly via a long series of methods).i
