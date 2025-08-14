# Module: qualtran


The top-level Qualtran module.



Many fundamental objects for expressing quantum programs can be imported from this
top-level namespace like <a href="./qualtran/Bloq.html"><code>qualtran.Bloq</code></a>, <a href="./qualtran/Register.html"><code>qualtran.Register</code></a>, and the various quantum
data types (<a href="./qualtran/QBit.html"><code>qualtran.QBit</code></a>, <a href="./qualtran/QUInt.html"><code>qualtran.QUInt</code></a>, ...).

The standard library of quantum algorithms must be imported from the `qualtran.bloqs` submodule.
A variety of analysis protocols are available in submodules as well like
<a href="./qualtran/resource_counting.html"><code>qualtran.resource_counting</code></a>, <a href="./qualtran/drawing.html"><code>qualtran.drawing</code></a>, <a href="./qualtran/simulation.html"><code>qualtran.simulation</code></a>, and others.
## Modules

[`cirq_interop`](./qualtran/cirq_interop.md): Bi-directional interop between Qualtran & Cirq.

[`drawing`](./qualtran/drawing.md): Draw and visualize bloqs.

[`linalg`](./qualtran/linalg.md): Linear algebra routines for building bloqs.

[`resource_counting`](./qualtran/resource_counting.md): Analysis routines for computing costs and resource counts.

[`serialization`](./qualtran/serialization.md): Functions for protobuf serialization of bloqs.

[`simulation`](./qualtran/simulation.md): Simulators for quantum programs.

[`surface_code`](./qualtran/surface_code.md): Physical cost models for surface code architectures.

[`symbolics`](./qualtran/symbolics.md): Utilities for simultaneous support for Sympy symbolic objects and concrete values.

[`testing`](./qualtran/testing.md): Functions for testing bloqs.

## Classes

[`class Bloq`](./qualtran/Bloq.md): Bloq is the primary abstract base class for all operations.

[`class DecomposeTypeError`](./qualtran/DecomposeTypeError.md): Raised if a decomposition does not make sense in this context.

[`class DecomposeNotImplementedError`](./qualtran/DecomposeNotImplementedError.md): Raised if a decomposition is not yet provided.

[`class BloqError`](./qualtran/BloqError.md): A value error raised when CompositeBloq conditions are violated.

[`class CompositeBloq`](./qualtran/CompositeBloq.md): A bloq defined by a collection of sub-bloqs and dataflows between them

[`class BloqBuilder`](./qualtran/BloqBuilder.md): A builder class for constructing a `CompositeBloq`.

[`class DidNotFlattenAnythingError`](./qualtran/DidNotFlattenAnythingError.md): An exception raised if `flatten_once()` did not find anything to flatten.

[`class QCDType`](./qualtran/QCDType.md): The abstract interface for quantum/classical quantum computing data types.

[`class CDType`](./qualtran/CDType.md): The abstract interface for classical data types.

[`class QDType`](./qualtran/QDType.md): The abstract interface for quantum data types.

[`class QAny`](./qualtran/QAny.md): Opaque bag-of-qubits type.

[`class QBit`](./qualtran/QBit.md): A single qubit. The smallest addressable unit of quantum data.

[`class CBit`](./qualtran/CBit.md): A single classical bit. The smallest addressable unit of classical data.

[`class QInt`](./qualtran/QInt.md): Signed Integer of a given width bitsize.

[`class QIntOnesComp`](./qualtran/QIntOnesComp.md): Signed Integer of a given width bitsize.

[`class QUInt`](./qualtran/QUInt.md): Unsigned integer of a given width bitsize which wraps around upon overflow.

[`class BQUInt`](./qualtran/BQUInt.md): Unsigned integer whose values are bounded within a range.

[`class QFxp`](./qualtran/QFxp.md): Fixed point type to represent real numbers.

[`class QMontgomeryUInt`](./qualtran/QMontgomeryUInt.md): Montgomery form of an unsigned integer of a given width bitsize which wraps around upon overflow.

[`class QGF`](./qualtran/QGF.md): Galois Field type to represent elements of a finite field.

[`class QGFPoly`](./qualtran/QGFPoly.md): Univariate Polynomials with coefficients in a Galois Field GF($p^m$).

[`class QDTypeCheckingSeverity`](./qualtran/QDTypeCheckingSeverity.md): The level of type checking to enforce

[`class Register`](./qualtran/Register.md): A register serves as the input/output quantum data specifications in a bloq's `Signature`.

[`class Signature`](./qualtran/Signature.md): An ordered sequence of `Register`s that follow the rules for a bloq signature.

[`class Side`](./qualtran/Side.md): Denote LEFT, RIGHT, or THRU registers.

[`class BloqInstance`](./qualtran/BloqInstance.md): A unique instance of a Bloq within a `CompositeBloq`.

[`class Connection`](./qualtran/Connection.md): A connection between two `Soquet`s.

[`class DanglingT`](./qualtran/DanglingT.md): The type of the singleton objects `LeftDangle` and `RightDangle`.

[`class Soquet`](./qualtran/Soquet.md): One half of a connection.

[`class GateWithRegisters`](./qualtran/GateWithRegisters.md): `cirq.Gate`s extension with support for composite gates acting on multiple qubit registers.

[`class Adjoint`](./qualtran/Adjoint.md): The standard adjoint of `subbloq`.

[`class Controlled`](./qualtran/Controlled.md): A controlled version of `subbloq`.

[`class CtrlSpec`](./qualtran/CtrlSpec.md): A specification for how to control a bloq.

[`class AddControlledT`](./qualtran/AddControlledT.md): The signature for the `add_controlled` callback part of `ctrl_system`.

[`class BloqExample`](./qualtran/BloqExample.md): An instantiation of a bloq and its metadata.

[`class BloqDocSpec`](./qualtran/BloqDocSpec.md): A collection of bloq examples and specifications for documenting a bloq class.

## Functions

[`bloq_example(...)`](./qualtran/bloq_example.md): Decorator to turn a function into a `BloqExample`.

[`check_dtypes_consistent(...)`](./qualtran/check_dtypes_consistent.md): Check if two types are consistent given our current definition on consistent types.

[`make_ctrl_system_with_correct_metabloq(...)`](./qualtran/make_ctrl_system_with_correct_metabloq.md): The default fallback for `Bloq.make_ctrl_system.

## Type Aliases

[`ConnectionT`](./qualtran/ConnectionT.md)



<h2 class="add-link">Other Members</h2>

LeftDangle<a id="LeftDangle"></a>
: Instance of <a href="./qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>

RightDangle<a id="RightDangle"></a>
: Instance of <a href="./qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>


