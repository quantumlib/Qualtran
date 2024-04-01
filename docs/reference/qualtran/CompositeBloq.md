# CompositeBloq
`qualtran.CompositeBloq`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L65-L411">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A bloq defined by a collection of sub-bloqs and dataflows between them

Inherits From: [`Bloq`](../qualtran/Bloq.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.CompositeBloq(
    connections, signature, bloq_instances=NOTHING
)
</code></pre>



<!-- Placeholder for "Used in" -->


CompositeBloq represents a quantum subroutine as a dataflow compute graph. The
specific native representation is a list of `Connection` objects (i.e. a list of
graph edges). This container should be considered immutable. Additional views
of the graph are provided by methods and properties.

Users should generally use `BloqBuilder` to construct a composite bloq either
directly or by overriding <a href="../qualtran/Bloq.html#build_composite_bloq"><code>Bloq.build_composite_bloq</code></a>.

Throughout this library we will often use the variable name `cbloq` to refer to a
composite bloq.

<h2 class="add-link">Args</h2>

`cxns`<a id="cxns"></a>
: A sequence of `Connection` encoding the quantum compute graph.

`signature`<a id="signature"></a>
: The registers defining the inputs and outputs of this Bloq. This
  should correspond to the dangling `Soquets` in the `cxns`.






<h2 class="add-link">Attributes</h2>

`all_soquets`<a id="all_soquets"></a>
: &nbsp;

`bloq_instances`<a id="bloq_instances"></a>
: &nbsp;

`connections`<a id="connections"></a>
: &nbsp;

`signature`<a id="signature"></a>
: The input and output names and types for this bloq.
  
  This property can be thought of as analogous to the function signature in ordinary
  programming. For example, it is analogous to function declarations in a
  C header (`*.h`) file.
  
  This is the only mandatory method (property) you must implement to inherit from
  `Bloq`. You can optionally implement additional methods to encode more information
  about this bloq.




## Methods

<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L120-L127">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_cirq_op(
    qubit_manager: 'cirq.QubitManager', **cirq_quregs
) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]
</code></pre>

Return a cirq.CircuitOperation containing a cirq-exported version of this cbloq.


<h3 id="to_cirq_circuit"><code>to_cirq_circuit</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L129-L151">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_cirq_circuit(
    qubit_manager: Optional['cirq.QubitManager'] = None, **cirq_quregs
) -> Tuple['cirq.FrozenCircuit', Dict[str, 'CirqQuregT']]
</code></pre>

Convert this CompositeBloq to a `cirq.Circuit`.


Args

`qubit_manager`
: A `cirq.QubitManager` to allocate new qubits.

`**cirq_quregs`
: Mapping from left register names to Cirq qubit arrays.




Returns

`circuit`
: The cirq.FrozenCircuit version of this composite bloq.

`cirq_quregs`
: The output mapping from right register names to Cirq qubit arrays.




<h3 id="from_cirq_circuit"><code>from_cirq_circuit</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L153-L163">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_cirq_circuit(
    circuit: 'cirq.Circuit'
) -> 'CompositeBloq'
</code></pre>

Construct a composite bloq from a Cirq circuit.

Each `cirq.Operation` will be wrapped into a `CirqGate` wrapper bloq. The
resultant composite bloq will represent a unitary with one thru-register
named "qubits" of shape `(n_qubits,)`.

<h3 id="on_classical_vals"><code>on_classical_vals</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L165-L170">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_classical_vals(
    **vals
) -> Dict[str, 'ClassicalValT']
</code></pre>

Support classical data by recursing into the composite bloq.


<h3 id="call_classically"><code>call_classically</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L172-L177">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call_classically(
    **vals
) -> Tuple['ClassicalValT', ...]
</code></pre>

Support classical data by recursing into the composite bloq.


<h3 id="as_composite_bloq"><code>as_composite_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L179-L181">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_composite_bloq() -> 'CompositeBloq'
</code></pre>

This override just returns the present composite bloq.


<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L183-L184">View source</a>

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




<h3 id="build_call_graph"><code>build_call_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L186-L190">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_call_graph(
    ssa: Optional['SympySymbolAllocator']
) -> Set['BloqCountT']
</code></pre>

Return the bloq counts by counting up all the subbloqs.


<h3 id="iter_bloqnections"><code>iter_bloqnections</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L192-L211">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>iter_bloqnections() -> Iterator[Tuple[BloqInstance, List[Connection], List[Connection]]]
</code></pre>

Iterate over Bloqs and their connections in topological order.


Yields




<h3 id="iter_bloqsoqs"><code>iter_bloqsoqs</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L213-L246">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>iter_bloqsoqs() -> Iterator[Tuple[BloqInstance, Dict[str, SoquetT], Tuple[SoquetT, ...]]]
</code></pre>

Iterate over bloq instances and their input soquets.

This method is helpful for "adding from" this existing composite bloq. You must
use `map_soqs` to map this cbloq's soquets to the correct ones for the
new bloq.

```
>>> bb, _ = BloqBuilder.from_signature(self.signature)
>>> soq_map: List[Tuple[SoquetT, SoquetT]] = []
>>> for binst, in_soqs, old_out_soqs in self.iter_bloqsoqs():
>>>    in_soqs = bb.map_soqs(in_soqs, soq_map)
>>>    new_out_soqs = bb.add_t(binst.bloq, **in_soqs)
>>>    soq_map.extend(zip(old_out_soqs, new_out_soqs))
>>> return bb.finalize(**bb.map_soqs(self.final_soqs(), soq_map))
```

Yields

`binst`
: The current bloq instance

`in_soqs`
: A dictionary mapping the binst's register names to predecessor soquets.
  This is suitable for `bb.add(binst.bloq, **in_soqs)`

`out_soqs`
: A tuple of the output soquets of `binst`. This can be used to update
  the mapping from this cbloq's soquets to a modified copy, see the example code.




<h3 id="final_soqs"><code>final_soqs</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L248-L261">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>final_soqs() -> Dict[str, SoquetT]
</code></pre>

Return the final output soquets.

This method is helpful for finalizing an "add from" operation, see `iter_bloqsoqs`.

<h3 id="copy"><code>copy</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L263-L273">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy() -> 'CompositeBloq'
</code></pre>

Create a copy of this composite bloq by re-building it.


<h3 id="flatten_once"><code>flatten_once</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L275-L328">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flatten_once(
    pred: Callable[[<a href="../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>], bool]
) -> 'CompositeBloq'
</code></pre>

Decompose and flatten each subbloq that satisfies `pred`.

This will only flatten "once". That is, we will go through the bloq instances
contained in this composite bloq and (optionally) flatten each one but will not
recursively flatten the results. For a recursive version see `flatten`.

Args

`pred`
: A predicate that takes a bloq instance and returns True if it should
  be decomposed and flattened or False if it should remain undecomposed.
  All bloqs for which this callable returns True must support decomposition.




Returns




Raises

`NotImplementedError`
: If `pred` returns True but the underlying bloq does not
  support `decompose_bloq()`.

`DidNotFlattenAnythingError`
: If none of the bloq instances satisfied `pred`.




<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L330-L338">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjoint() -> 'CompositeBloq'
</code></pre>

Get a composite bloq which is the adjoint of this composite bloq.

The adjoint of a composite bloq is another composite bloq where the order of
operations is reversed and each subbloq is replaced with its adjoint.

<h3 id="flatten"><code>flatten</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L340-L371">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flatten(
    pred: Callable[[<a href="../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>], bool],
    max_depth: int = 1000
) -> 'CompositeBloq'
</code></pre>

Recursively decompose and flatten subbloqs until none satisfy `pred`.

This will continue flattening the results of subbloq.decompose_bloq() until
all bloqs which would satisfy `pred` have been flattened.

Args

`pred`
: A predicate that takes a bloq instance and returns True if it should
  be decomposed and flattened or False if it should remain undecomposed.
  All bloqs for which this callable returns True must support decomposition.

`max_depth`
: To avoid infinite recursion, give up after this many recursive steps.




Returns




Raises

`NotImplementedError`
: If `pred` returns True but the underlying bloq does not
  support `decompose_bloq()`.




<h3 id="debug_text"><code>debug_text</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/composite_bloq.py#L388-L411">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>debug_text() -> str
</code></pre>

Print connection information to assist in debugging.

The output will be a topologically sorted list of BloqInstances with each
topological generation separated by a horizontal line. Each bloq instance is followed
by a list of its incoming and outgoing connections. Note that all non-dangling
connections are represented twice: once as the output of a binst and again as the input
to a subsequent binst.

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class CompositeBloq.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Method generated by attrs for class CompositeBloq.




