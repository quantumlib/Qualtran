# Module: qualtran


qualtran




isort:skip_file
## Modules

[`cirq_interop`](./qualtran/cirq_interop.md): Bi-directional interop between Qualtran & Cirq using Cirq-FT.

[`drawing`](./qualtran/drawing.md): Draw and visualize bloqs

[`linalg`](./qualtran/linalg.md) module

[`resource_counting`](./qualtran/resource_counting.md): Counting resource usage (bloqs, qubits)

[`serialization`](./qualtran/serialization.md) module

[`simulation`](./qualtran/simulation.md) module

[`surface_code`](./qualtran/surface_code.md) module

[`testing`](./qualtran/testing.md) module

## Classes

[`class Bloq`](./qualtran/Bloq.md): Bloq is the primary abstract base class for all operations.

[`class DecomposeTypeError`](./qualtran/DecomposeTypeError.md): Raised if a decomposition does not make sense in this context.

[`class DecomposeNotImplementedError`](./qualtran/DecomposeNotImplementedError.md): Raised if a decomposition is not yet provided.

[`class BloqError`](./qualtran/BloqError.md): A value error raised when CompositeBloq conditions are violated.

[`class CompositeBloq`](./qualtran/CompositeBloq.md): A bloq defined by a collection of sub-bloqs and dataflows between them

[`class BloqBuilder`](./qualtran/BloqBuilder.md): A builder class for constructing a `CompositeBloq`.

[`class DidNotFlattenAnythingError`](./qualtran/DidNotFlattenAnythingError.md): An exception raised if `flatten_once()` did not find anything to flatten.

[`class QDType`](./qualtran/QDType.md): This defines the abstract interface for quantum data types.

[`class QInt`](./qualtran/QInt.md): Signed Integer of a given width bitsize.

[`class QBit`](./qualtran/QBit.md): A single qubit. The smallest addressable unit of quantum data.

[`class QAny`](./qualtran/QAny.md): Opaque bag-of-qbits type.

[`class QFxp`](./qualtran/QFxp.md): Fixed point type to represent real numbers.

[`class QIntOnesComp`](./qualtran/QIntOnesComp.md): Signed Integer of a given width bitsize.

[`class QUInt`](./qualtran/QUInt.md): Unsigned integer of a given width bitsize which wraps around upon overflow.

[`class BoundedQUInt`](./qualtran/BoundedQUInt.md): Unsigned integer whose values are bounded within a range.

[`class Register`](./qualtran/Register.md): A data type describing a register of qubits.

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



<h2 class="add-link">Other Members</h2>

LeftDangle<a id="LeftDangle"></a>
: Instance of <a href="./qualtran/DanglingT.md"><code>qualtran.DanglingT</code></a>

RightDangle<a id="RightDangle"></a>
: Instance of <a href="./qualtran/DanglingT.md"><code>qualtran.DanglingT</code></a>


