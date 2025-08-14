# CirqGateAsBloqBase
`qualtran.cirq_interop.CirqGateAsBloqBase`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L74-L136">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A base class to bootstrap a bloq from a `cirq.Gate`.

Inherits From: [`Bloq`](../../qualtran/Bloq.md)

<!-- Placeholder for "Used in" -->

Bloq authors can inherit from this abstract class and override the `cirq_gate` property
to get a bloq adapted from the cirq gate. Authors can continue to customize the bloq
by overriding methods (like costs, string representations, ...).

Otherwise, this class fulfils the Bloq API by delegating to `cirq.Gate` methods.

This is the base class that provides the functionality for the `CirqGateAsBloq` adapter.
The adapter lets you use any `cirq.Gate` as a bloq immediately (without defining a new class
that inherits from `CirqGateAsBloqBase`), and is used as a fallback in the interoperability
functionality.



<h2 class="add-link">Attributes</h2>

`cirq_gate`<a id="cirq_gate"></a>
: The `cirq.Gate` to use as the source of truth.

`signature`<a id="signature"></a>
: &nbsp;




## Methods

<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L104-L105">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_bloq() -> 'CompositeBloq'
</code></pre>

Decompose this Bloq into its constituent parts contained in a CompositeBloq.

Bloq users can call this function to delve into the definition of a Bloq. If you're
trying to define a bloq's decomposition, consider overriding `build_composite_bloq`
which provides helpful arguments for implementers.

Returns




Raises

`NotImplementedError`
: If there is no decomposition defined; namely: if
  `build_composite_bloq` returns `NotImplemented`.




<h3 id="decompose_from_registers"><code>decompose_from_registers</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L107-L114">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_from_registers(
    *, context: cirq.DecompositionContext, **quregs
) -> cirq.OP_TREE
</code></pre>




<h3 id="my_tensors"><code>my_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L116-L121">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>my_tensors(
    incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
) -> List['qtn.Tensor']
</code></pre>

Override this method to support native quimb simulation of this Bloq.

This method is responsible for returning tensors corresponding to the unitary, state, or
effect of the bloq. Often, this method will return one tensor for a given Bloq, but
some bloqs can be represented in a factorized form using more than one tensor.

By default, calls to <a href="../../qualtran/Bloq.html#tensor_contract"><code>Bloq.tensor_contract()</code></a> will first decompose and flatten the bloq
before initiating the conversion to a tensor network. This has two consequences:
 1) Overriding this method is only necessary if this bloq does not define a decomposition
    or if the fully-decomposed form contains a bloq that does not define its tensors.
 2) Even if you override this method to provide custom tensors, they may not be used
    (by default) because we prefer the flat-decomposed version. This is usually desirable
    for contraction performance; but for finer-grained control see
    <a href="../../qualtran/simulation/tensor/cbloq_to_quimb.html"><code>qualtran.simulation.tensor.cbloq_to_quimb</code></a>.

Quimb defines a connection between two tensors by a shared index. The returned tensors
from this method must use the Qualtran-Quimb index convention:
 - Each tensor index is a tuple `(cxn, j)`
 - The `cxn: qualtran.Connection` entry identifies the connection between bloq instances.
 - The second integer `j` is the bit index within high-bitsize registers,
   which is necessary due to technical restrictions.

Args

`incoming`
: A mapping from register name to Connection (or an array thereof) to use as
  left indices for the tensor network. The shape of the array matches the register's
  shape.

`outgoing`
: A mapping from register name to Connection (or an array thereof) to use as
  right indices for the tensor network.




<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L123-L127">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_cirq_op(
    qubit_manager: 'cirq.QubitManager', **in_quregs
) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]
</code></pre>

Override this method to support conversion to a Cirq operation.

If this method is not overriden, the default implementation will wrap this bloq
in a `BloqAsCirqGate` shim.

Args

`qubit_manager`
: A `cirq.QubitManager` for allocating `cirq.Qid`s.

`**cirq_quregs`
: kwargs mapping from this bloq's left register names to an ndarray of
  `cirq.Qid`. The final dimension of this array corresponds to the registers
  `bitsize` size. Any additional dimensions come first and correspond to the
  register `shape` sizes.




Returns

`op`
: A cirq operation corresponding to this bloq acting on the provided cirq qubits or
  None. This method should return None if and only if the bloq instance truly should
  not be included in the Cirq circuit (e.g. for reshaping bloqs). A bloq with no cirq
  equivalent should raise an exception instead.

`cirq_quregs`
: A mapping from this bloq's right register of the same format as the
  `cirq_quregs` argument. The returned dictionary corresponds to the output qubits.




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L129-L130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    power
)
</code></pre>




<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L132-L133">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjoint() -> 'Bloq'
</code></pre>

The adjoint of this bloq.

Bloq authors can override this method in certain circumstances. Otherwise, the default
fallback wraps this bloq in `Adjoint`.

Please see the documentation for `Adjoint` and the `Adjoint.ipynb` notebook for full
details.



