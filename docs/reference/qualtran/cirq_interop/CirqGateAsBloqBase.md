# CirqGateAsBloqBase
`qualtran.cirq_interop.CirqGateAsBloqBase`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L73-L135">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A Bloq wrapper around a `cirq.Gate`

Inherits From: [`GateWithRegisters`](../../qualtran/GateWithRegisters.md), [`Bloq`](../../qualtran/Bloq.md)

<!-- Placeholder for "Used in" -->




<h2 class="add-link">Attributes</h2>

`cirq_gate`<a id="cirq_gate"></a>
: &nbsp;

`signature`<a id="signature"></a>
: &nbsp;




## Methods

<h3 id="decompose_from_registers"><code>decompose_from_registers</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L91-L102">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_from_registers(
    *, context: cirq.DecompositionContext, **quregs
) -> cirq.OP_TREE
</code></pre>




<h3 id="my_tensors"><code>my_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L104-L109">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L111-L117">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_cirq_op(
    qubit_manager: 'cirq.QubitManager', **in_quregs
) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]
</code></pre>

Allocates/Deallocates qubits for RIGHT/LEFT only registers to construct a Cirq operation


Args

`qubit_manager`
: For allocating/deallocating qubits for RIGHT/LEFT only registers.

`in_quregs`
: Mapping from LEFT register names to corresponding cirq qubits.




Returns




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L131-L132">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    power
)
</code></pre>




<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L134-L135">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjoint() -> 'Bloq'
</code></pre>

The adjoint of this bloq.

Bloq authors can override this method in certain circumstances. Otherwise, the default
fallback wraps this bloq in `Adjoint`.

Please see the documentation for `Adjoint` and the `Adjoint.ipynb` notebook for full
details.

<h3 id="num_qubits"><code>num_qubits</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_qubits() -> int
</code></pre>

The number of qubits this gate acts on.




