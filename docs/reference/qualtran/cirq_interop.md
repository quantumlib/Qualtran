# Module: cirq_interop


Bi-directional interop between Qualtran & Cirq.



## Modules

[`decompose_protocol`](../qualtran/cirq_interop/decompose_protocol.md) module

[`t_complexity_protocol`](../qualtran/cirq_interop/t_complexity_protocol.md) module

## Classes

[`class CirqGateAsBloq`](../qualtran/cirq_interop/CirqGateAsBloq.md): An adapter that fulfils the Bloq API by delegating to `cirq.Gate` methods.

[`class CirqGateAsBloqBase`](../qualtran/cirq_interop/CirqGateAsBloqBase.md): A base class to bootstrap a bloq from a `cirq.Gate`.

[`class BloqAsCirqGate`](../qualtran/cirq_interop/BloqAsCirqGate.md): A shim for using bloqs in a Cirq circuit.

## Functions

[`cirq_gate_to_bloq(...)`](../qualtran/cirq_interop/cirq_gate_to_bloq.md): For a given Cirq gate, return an equivalent bloq.

[`cirq_optree_to_cbloq(...)`](../qualtran/cirq_interop/cirq_optree_to_cbloq.md): Convert a Cirq OP-TREE into a `CompositeBloq` with signature `signature`.

[`decompose_from_cirq_style_method(...)`](../qualtran/cirq_interop/decompose_from_cirq_style_method.md): Return a `CompositeBloq` decomposition using a cirq-style decompose method.

## Type Aliases

[`CirqQuregT`](../qualtran/cirq_interop/CirqQuregT.md)

