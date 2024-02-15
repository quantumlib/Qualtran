# CirqGateAsBloqBase
`qualtran.cirq_interop.CirqGateAsBloqBase`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L62-L141">
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

<h3 id="pretty_name"><code>pretty_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L70-L72">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pretty_name() -> str
</code></pre>




<h3 id="decompose_from_registers"><code>decompose_from_registers</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L85-L96">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_from_registers(
    *, context: cirq.DecompositionContext, **quregs
) -> cirq.OP_TREE
</code></pre>




<h3 id="add_my_tensors"><code>add_my_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L98-L114">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_my_tensors(
    tn: qtn.TensorNetwork,
    tag: Any,
    *,
    incoming: Dict[str, 'SoquetT'],
    outgoing: Dict[str, 'SoquetT']
)
</code></pre>

Override this method to support native quimb simulation of this Bloq.

This method is responsible for adding a tensor corresponding to the unitary, state, or
effect of the bloq to the provided tensor network `tn`. Often, this method will add
one tensor for a given Bloq, but some bloqs can be represented in a factorized form
requiring the addition of more than one tensor.

If this method is not overriden, the default implementation will try to use the bloq's
decomposition to find a dense representation for this bloq.

Args

`tn`
: The tensor network to which we add our tensor(s)

`tag`
: An arbitrary tag that must be forwarded to `qtn.Tensor`'s `tag` attribute.

`incoming`
: A mapping from register name to SoquetT to order left indices for
  the tensor network.

`outgoing`
: A mapping from register name to SoquetT to order right indices for
  the tensor network.




<h3 id="t_complexity"><code>t_complexity</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L116-L117">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>t_complexity() -> 'TComplexity'
</code></pre>

The `TComplexity` for this bloq.

By default, this will recurse into this bloq's decomposition but this
method can be overriden with a known value.

<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L119-L125">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L137-L138">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    power
)
</code></pre>




<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/_cirq_to_bloq.py#L140-L141">View source</a>

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




