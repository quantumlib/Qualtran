# Controlled
`qualtran.Controlled`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L582-L674">
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

This bloq represents a "total control" strategy of controlling `subbloq`: the decomposition
of `Controlled(b)` uses the decomposition of `b` and controls each subbloq in that
decomposition.

Users should likely not use this class directly. Prefer using `bloq.controlled(ctrl_spec)`,
which may return a natively-controlled Bloq or a more intelligent construction for
complex control specs.

<h2 class="add-link">Args</h2>

`subbloq`<a id="subbloq"></a>
: The bloq we are controlling.

`ctrl_spec`<a id="ctrl_spec"></a>
: The specification for how to control the bloq.






<h2 class="add-link">Attributes</h2>

`ctrl_reg_names`<a id="ctrl_reg_names"></a>
: The name of the control registers.
  
  This is generated on-the-fly to avoid conflicts with existing register
  names. Users should not rely on the absolute value of this property staying constant.

`ctrl_regs`<a id="ctrl_regs"></a>
: &nbsp;

`ctrl_spec`<a id="ctrl_spec"></a>
: The specification of how the `subbloq` is controlled.

`signature`<a id="signature"></a>
: &nbsp;

`subbloq`<a id="subbloq"></a>
: The bloq being controlled.




## Methods

<h3 id="make_ctrl_system"><code>make_ctrl_system</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L611-L620">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>make_ctrl_system(
    bloq: 'Bloq', ctrl_spec: 'CtrlSpec'
) -> Tuple['_ControlledBase', 'AddControlledT']
</code></pre>

A factory method for creating both the Controlled and the adder function.

See <a href="../qualtran/Bloq.html#get_ctrl_system"><code>Bloq.get_ctrl_system</code></a>.

<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L622-L623">View source</a>

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




<h3 id="build_composite_bloq"><code>build_composite_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L625-L652">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_composite_bloq(
    bb: 'BloqBuilder', **initial_soqs
) -> Dict[str, 'SoquetT']
</code></pre>

Override this method to define a Bloq in terms of its constituent parts.

Bloq authors should override this method. If you already have an instance of a `Bloq`,
consider calling `decompose_bloq()` which will set up the correct context for
calling this function.

Args

`bb`
: A `BloqBuilder` to append sub-Bloq to.

`**soqs`
: The initial soquets corresponding to the inputs to the Bloq.




Returns




<h3 id="build_call_graph"><code>build_call_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L654-L671">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_call_graph(
    ssa: 'SympySymbolAllocator'
) -> 'BloqCountDictT'
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

<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L673-L674">View source</a>

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


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Check equality and either forward a NotImplemented or return the result negated.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class Controlled.




