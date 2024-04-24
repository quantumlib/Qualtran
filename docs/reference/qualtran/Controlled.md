# Controlled
`qualtran.Controlled`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L281-L457">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A controlled version of `subbloq`.

Inherits From: [`GateWithRegisters`](../qualtran/GateWithRegisters.md), [`Bloq`](../qualtran/Bloq.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.Controlled(
    subbloq, ctrl_spec
)
</code></pre>



<!-- Placeholder for "Used in" -->

This meta-bloq is part of the 'controlled' protocol. As a default fallback,
we wrap any bloq without a custom controlled version in this meta-bloq.

Users should likely not use this class directly. Prefer using `bloq.controlled(ctrl_spec)`,
which may return a tailored Bloq that is controlled in the desired way.

<h2 class="add-link">Args</h2>

`subbloq`<a id="subbloq"></a>
: The bloq we are controlling.

`ctrl_spec`<a id="ctrl_spec"></a>
: The specification for how to control the bloq.






<h2 class="add-link">Attributes</h2>

`ctrl_reg_names`<a id="ctrl_reg_names"></a>
: &nbsp;

`ctrl_regs`<a id="ctrl_regs"></a>
: &nbsp;

`ctrl_spec`<a id="ctrl_spec"></a>
: &nbsp;

`signature`<a id="signature"></a>
: The input and output names and types for this bloq.
  
  This property can be thought of as analogous to the function signature in ordinary
  programming. For example, it is analogous to function declarations in a
  C header (`*.h`) file.
  
  This is the only mandatory method (property) you must implement to inherit from
  `Bloq`. You can optionally implement additional methods to encode more information
  about this bloq.

`subbloq`<a id="subbloq"></a>
: &nbsp;




## Methods

<h3 id="make_ctrl_system"><code>make_ctrl_system</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L299-L319">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>make_ctrl_system(
    bloq: 'Bloq', ctrl_spec: 'CtrlSpec'
) -> Tuple[<a href="../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>, <a href="../qualtran/AddControlledT.html"><code>qualtran.AddControlledT</code></a>]
</code></pre>

A factory method for creating both the Controlled and the adder function.

See <a href="../qualtran/Bloq.html#get_ctrl_system"><code>Bloq.get_ctrl_system</code></a>.

<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L344-L365">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_bloq() -> 'CompositeBloq'
</code></pre>

Decompose this Bloq into its constituent parts contained in a CompositeBloq.

Bloq users can call this function to delve into the definition of a Bloq. The function
returns the decomposition of this Bloq represented as an explicit compute graph wrapped
in a `CompositeBloq` object.

Bloq authors can specify the bloq's decomposition by overriding any of the following two
methods:

- `build_composite_bloq`: Override this method to define a bloq-style decomposition using a
    `BloqBuilder` builder class to construct the `CompositeBloq` directly.
- `decompose_from_registers`: Override this method to define a cirq-style decomposition by
    yielding cirq style operations applied on qubits.

Irrespective of the bloq author's choice of backend to implement the
decomposition, bloq users will be able to access both the bloq-style and Cirq-style
interfaces. For example, users can call:

- `cirq.decompose_once(bloq.on_registers(**cirq_quregs))`: This will yield a `cirq.OPTREE`.
    Bloqs will be wrapped in `BloqAsCirqGate` as needed.
- `bloq.decompose_bloq()`: This will return a `CompositeBloq`.
   Cirq gates will be be wrapped in `CirqGateAsBloq` as needed.

Thus, `GateWithRegisters` class provides a convenient way of defining objects that can be used
interchangeably with both `Cirq` and `Bloq` constructs.

Returns




Raises

`DecomposeNotImplementedError`
: If there is no decomposition defined; namely if both:
  - `build_composite_bloq` raises a `DecomposeNotImplementedError` and
  - `decompose_from_registers` raises a `DecomposeNotImplementedError`.




<h3 id="build_call_graph"><code>build_call_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L367-L371">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_call_graph(
    ssa: 'SympySymbolAllocator'
) -> Set['BloqCountT']
</code></pre>

Override this method to build the bloq call graph.

This method must return a set of `(bloq, n)` tuples where `bloq` is called `n` times in
the decomposition. This method defines one level of the call graph, specifically the
edges from this bloq to its immediate children. To get the full graph,
call <a href="../qualtran/Bloq.html#call_graph"><code>Bloq.call_graph()</code></a>.

By default, this method will use `self.decompose_bloq()` to count the bloqs called
in the decomposition. By overriding this method, you can provide explicit call counts.
This is appropriate if: 1) you can't or won't provide a complete decomposition, 2) you
know symbolic expressions for the counts, or 3) you need to "generalize" the subbloqs
by overwriting bloq attributes that do not affect its cost with generic sympy symbols using
the provided `SympySymbolAllocator`.

<h3 id="on_classical_vals"><code>on_classical_vals</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L373-L383">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_classical_vals(
    **vals
) -> Dict[str, 'ClassicalValT']
</code></pre>

How this bloq operates on classical data.

Override this method if your bloq represents classical, reversible logic. For example:
quantum circuits composed of X and C^nNOT gates are classically simulable.

Bloq definers should override this method. If you already have an instance of a `Bloq`,
consider calling `call_clasically(**vals)` which will do input validation before
calling this function.

Args

`**vals`
: The input classical values for each left (or thru) register. The data
  types are guaranteed to match `self.registers`. Values for registers
  with bitsize `n` will be integers of that bitsize. Values for registers with
  `shape` will be an ndarray of integers of the given bitsize. Note: integers
  can be either Numpy or Python integers. If they are Python integers, they
  are unsigned.




Returns




<h3 id="add_my_tensors"><code>add_my_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L385-L412">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_my_tensors(
    tn: 'qtn.TensorNetwork',
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




<h3 id="wire_symbol"><code>wire_symbol</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L427-L434">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>wire_symbol(
    soq: 'Soquet'
) -> 'WireSymbol'
</code></pre>

On a musical score visualization, use this `WireSymbol` to represent `soq`.

By default, we use a "directional text box", which is a text box that is either
rectangular for thru-registers or facing to the left or right for non-thru-registers.

Override this method to provide a more relevant `WireSymbol` for the provided soquet.
This method can access bloq attributes. For example: you may want to draw either
a filled or empty circle for a control register depending on a control value bloq
attribute.

<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L436-L437">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjoint() -> 'Bloq'
</code></pre>

The adjoint of this bloq.

Bloq authors can override this method in certain circumstances. Otherwise, the default
fallback wraps this bloq in `Adjoint`.

Please see the documentation for `Adjoint` and the `Adjoint.ipynb` notebook for full
details.

<h3 id="pretty_name"><code>pretty_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L439-L440">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pretty_name() -> str
</code></pre>




<h3 id="short_name"><code>short_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L442-L443">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>short_name() -> str
</code></pre>




<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L448-L457">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_cirq_op(
    qubit_manager: 'cirq.QubitManager', **cirq_quregs
) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]
</code></pre>

Allocates/Deallocates qubits for RIGHT/LEFT only registers to construct a Cirq operation


Args

`qubit_manager`
: For allocating/deallocating qubits for RIGHT/LEFT only registers.

`in_quregs`
: Mapping from LEFT register names to corresponding cirq qubits.




Returns




<h3 id="num_qubits"><code>num_qubits</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_qubits() -> int
</code></pre>

The number of qubits this gate acts on.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class Controlled.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Method generated by attrs for class Controlled.




