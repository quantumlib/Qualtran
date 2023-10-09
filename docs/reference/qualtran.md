# Module: qualtran


qualtran




isort:skip_file
## Modules

[`cirq_interop`](./qualtran/cirq_interop.md): Bi-directional interop between Qualtran & Cirq using Cirq-FT.

[`drawing`](./qualtran/drawing.md): Draw and visualize bloqs

[`resource_counting`](./qualtran/resource_counting.md): Counting resource usage (bloqs, qubits)

[`simulation`](./qualtran/simulation.md) module

## Classes

[`class Bloq`](./qualtran/Bloq.md): Bloq is the primary abstract base class for all operations.

[`class BloqError`](./qualtran/BloqError.md): A value error raised when CompositeBloq conditions are violated.

[`class CompositeBloq`](./qualtran/CompositeBloq.md): A bloq defined by a collection of sub-bloqs and dataflows between them

[`class BloqBuilder`](./qualtran/BloqBuilder.md): A builder class for constructing a `CompositeBloq`.

[`class DidNotFlattenAnythingError`](./qualtran/DidNotFlattenAnythingError.md): An exception raised if `flatten_once()` did not find anything to flatten.

[`class Register`](./qualtran/Register.md): A data type describing a register of qubits.

[`class Signature`](./qualtran/Signature.md): An ordered sequence of `Register`s that follow the rules for a bloq signature.

[`class Side`](./qualtran/Side.md): Denote LEFT, RIGHT, or THRU registers.

[`class BloqInstance`](./qualtran/BloqInstance.md): A unique instance of a Bloq within a `CompositeBloq`.

[`class Connection`](./qualtran/Connection.md): A connection between two `Soquet`s.

[`class DanglingT`](./qualtran/DanglingT.md): The type of the singleton objects `LeftDangle` and `RightDangle`.

[`class Soquet`](./qualtran/Soquet.md): One half of a connection.



<h2 class="add-link">Other Members</h2>

LeftDangle<a id="LeftDangle"></a>
: Instance of <a href="./qualtran/DanglingT.md"><code>qualtran.DanglingT</code></a>

RightDangle<a id="RightDangle"></a>
: Instance of <a href="./qualtran/DanglingT.md"><code>qualtran.DanglingT</code></a>


