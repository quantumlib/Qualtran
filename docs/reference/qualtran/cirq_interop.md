# Module: cirq_interop


Bi-directional interop between Qualtran & Cirq using Cirq-FT.



isort:skip_file
## Classes

[`class CirqGateAsBloq`](../qualtran/cirq_interop/CirqGateAsBloq.md): A Bloq wrapper around a `cirq.Gate`, preserving signature if gate is a `GateWithRegisters`.

[`class BloqAsCirqGate`](../qualtran/cirq_interop/BloqAsCirqGate.md): A shim for using bloqs in a Cirq circuit.

## Functions

[`cirq_optree_to_cbloq(...)`](../qualtran/cirq_interop/cirq_optree_to_cbloq.md): Convert a Cirq OP-TREE into a `CompositeBloq` with signature `signature`.

[`decompose_from_cirq_op(...)`](../qualtran/cirq_interop/decompose_from_cirq_op.md): Returns a CompositeBloq constructed using Cirq operations obtained via `bloq.as_cirq_op`.

