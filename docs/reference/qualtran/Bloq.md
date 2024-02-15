# Bloq
`qualtran.Bloq`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L70-L505">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Bloq is the primary abstract base class for all operations.

<!-- Placeholder for "Used in" -->

Bloqs let you represent high-level quantum programs and subroutines as a hierarchical
collection of Python objects. The main interface is this abstract base class.

There are two important flavors of implementations of the `Bloq` interface. The first flavor
consists of bloqs implemented by you, the user-developer to express quantum operations of
interest. For example:

```
>>> class ShorsAlgorithm(Bloq):
>>>     ...
```

The other important `Bloq` subclass is `CompositeBloq`, which is a container type for a
collection of sub-bloqs.

There is only one mandatory method you must implement to have a well-formed `Bloq`,
namely `Bloq.registers`. There are many other methods you can optionally implement to
encode more information about the bloq.



<h2 class="add-link">Attributes</h2>

`signature`<a id="signature"></a>
: The input and output names and types for this bloq.
  
  This property can be thought of as analogous to the function signature in ordinary
  programming. For example, it is analogous to function declarations in a
  C header (`*.h`) file.
  
  This is the only mandatory method (property) you must implement to inherit from
  `Bloq`. You can optionally implement additional methods to encode more information
  about this bloq.




## Methods

<h3 id="pretty_name"><code>pretty_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L105-L106">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pretty_name() -> str
</code></pre>




<h3 id="short_name"><code>short_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L108-L113">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>short_name() -> str
</code></pre>




<h3 id="build_composite_bloq"><code>build_composite_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L115-L130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_composite_bloq(
    bb: 'BloqBuilder', **soqs
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




<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L132-L146">View source</a>

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




<h3 id="supports_decompose_bloq"><code>supports_decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L148-L155">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>supports_decompose_bloq() -> bool
</code></pre>

Whether this bloq supports `.decompose_bloq()`.

By default, we check that the method `build_composite_bloq` is overriden. For
extraordinary circumstances, you may need to override this method directly to
return an accurate value.

<h3 id="as_composite_bloq"><code>as_composite_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L157-L166">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_composite_bloq() -> 'CompositeBloq'
</code></pre>

Wrap this Bloq into a size-1 CompositeBloq.

This method is overriden so if this Bloq is already a CompositeBloq, it will
be returned.

<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L168-L180">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjoint() -> 'Bloq'
</code></pre>

The adjoint of this bloq.

Bloq authors can override this method in certain circumstances. Otherwise, the default
fallback wraps this bloq in `Adjoint`.

Please see the documentation for `Adjoint` and the `Adjoint.ipynb` notebook for full
details.

<h3 id="on_classical_vals"><code>on_classical_vals</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L182-L206">View source</a>

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




<h3 id="call_classically"><code>call_classically</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L208-L228">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call_classically(
    **vals
) -> Tuple['ClassicalValT', ...]
</code></pre>

Call this bloq on classical data.

Bloq users can call this function to apply bloqs to classical data. If you're
trying to define a bloq's action on classical values, consider overriding
`on_classical_vals` which promises type checking for arguments.

Args

`**vals`
: The input classical values for each left (or thru) register. The data
  types must match `self.registers`. Values for registers
  with bitsize `n` should be integers of that bitsize or less. Values for registers
  with `shape` should be an ndarray of integers of the given bitsize.
  Note: integers can be either Numpy or Python integers, but should be positive
  and unsigned.




Returns




<h3 id="tensor_contract"><code>tensor_contract</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L230-L240">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tensor_contract() -> 'NDArray'
</code></pre>

Return a contracted, dense ndarray representing this bloq.

This constructs a tensor network and then contracts it according to our registers,
i.e. the dangling indices. The returned array will be 0-, 1- or 2-dimensional. If it is
a 2-dimensional matrix, we follow the quantum computing / matrix multiplication convention
of (right, left) indices.

<h3 id="add_my_tensors"><code>add_my_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L242-L271">View source</a>

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




<h3 id="build_call_graph"><code>build_call_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L273-L288">View source</a>

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

<h3 id="call_graph"><code>call_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L290-L320">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call_graph(
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
    keep: Optional[Sequence['Bloq']] = None,
    max_depth: Optional[int] = None
) -> Tuple['nx.DiGraph', Dict['Bloq', Union[int, 'sympy.Expr']]]
</code></pre>

Get the bloq call graph and call totals.

The call graph has edges from a parent bloq to each of the bloqs that it calls in
its decomposition. The number of times it is called is stored as an edge attribute.
To specify the bloq call counts for a specific node, override <a href="../qualtran/Bloq.html#build_call_graph"><code>Bloq.build_call_graph()</code></a>.

Args

`generalizer`
: If provided, run this function on each (sub)bloq to replace attributes
  that do not affect resource estimates with generic sympy symbols. If the function
  returns `None`, the bloq is omitted from the counts graph. If a sequence of
  generalizers is provided, each generalizer will be run in order.

`keep`
: If this function evaluates to True for the current bloq, keep the bloq as a leaf
  node in the call graph instead of recursing into it.

`max_depth`
: If provided, build a call graph with at most this many layers.




Returns

`g`
: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' reports
  the number of times successor bloq is called via its predecessor.

`sigma`
: Call totals for "leaf" bloqs. We keep a bloq as a leaf in the call graph
  according to `keep` and `max_depth` (if provided) or if a bloq cannot be
  decomposed.




<h3 id="bloq_counts"><code>bloq_counts</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L322-L341">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>bloq_counts(
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None
) -> Dict['Bloq', Union[int, 'sympy.Expr']]
</code></pre>

The number of subbloqs directly called by this bloq.

This corresponds to one level of the call graph, see <a href="../qualtran/Bloq.html#call_graph"><code>Bloq.call_graph()</code></a>.
To specify explicit values for a bloq, override <a href="../qualtran/Bloq.html#build_call_graph"><code>Bloq.build_call_graph(...)</code></a>, not this
method.

Args

`generalizer`
: If provided, run this function on each (sub)bloq to replace attributes
  that do not affect resource estimates with generic sympy symbols. If the function
  returns `None`, the bloq is omitted from the counts graph. If a sequence of
  generalizers is provided, each generalizer will be run in order.




Returns




<h3 id="get_ctrl_system"><code>get_ctrl_system</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L343-L384">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_ctrl_system(
    ctrl_spec: Optional['CtrlSpec'] = None
) -> Tuple['Bloq', 'AddControlledT']
</code></pre>

Get a controlled version of this bloq and a function to wire it up correctly.

Users should likely call <a href="../qualtran/Bloq.html#controlled"><code>Bloq.controlled(...)</code></a> which uses this method behind-the-scenes.
Intrepid bloq authors can override this method to provide a custom controlled version of
this bloq. By default, this will use the <a href="../qualtran/Controlled.html"><code>qualtran.Controlled</code></a> meta-bloq to control any
bloq.

This method must return both a controlled version of this bloq and a callable that
'wires up' soquets correctly.

A controlled version of this bloq has all the registers from the original bloq plus
any additional control registers to support the activation function specified by
the `ctrl_spec`. In the simplest case, this could be one additional 1-qubit register
that activates the bloq if the input is in the |1> state, but additional logic is possible.
See the documentation for `CtrlSpec` for more information.

The second return value ensures we can accurately wire up soquets into the added registers.
It must have the following signature:

    def _my_add_controlled(
        bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
    ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:

Which takes a bloq builder (for adding the controlled bloq), the new control soquets,
input soquets for the existing registers; and returns a sequence of the output control
soquets and a sequence of the output soquets for the existing registers. This complexity
is sadly unavoidable due to the variety of ways of wiring up custom controlled bloqs.

Returns

`controlled_bloq`
: A controlled version of this bloq

`add_controlled`
: A function with the signature documented above that the system
  can use to automatically wire up the new control registers.




<h3 id="controlled"><code>controlled</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L386-L403">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>controlled(
    ctrl_spec: Optional['CtrlSpec'] = None
) -> 'Bloq'
</code></pre>

Return a controlled version of this bloq.

By default, the system will use the <a href="../qualtran/Controlled.html"><code>qualtran.Controlled</code></a> meta-bloq to wrap this
bloq. Bloqs authors can declare their own, custom controlled versions by overriding
<a href="../qualtran/Bloq.html#get_ctrl_system"><code>Bloq.get_ctrl_system</code></a> in the bloq.

Args

`ctrl_spec`
: an optional `CtrlSpec`, which specifies how to control the bloq. The
  default spec means the bloq will be active when one control qubit is in the |1>
  state. See the CtrlSpec documentation for more possibilities including
  negative controls, integer-equality control, and ndarrays of control values.




Returns




<h3 id="t_complexity"><code>t_complexity</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L405-L411">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>t_complexity() -> 'TComplexity'
</code></pre>

The `TComplexity` for this bloq.

By default, this will recurse into this bloq's decomposition but this
method can be overriden with a known value.

<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L413-L440">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_cirq_op(
    qubit_manager: 'cirq.QubitManager', **cirq_quregs
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




<h3 id="on"><code>on</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L442-L468">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on(
    *qubits
) -> 'cirq.Operation'
</code></pre>

A `cirq.Operation` of this bloq operating on the given qubits.

This method supports an alternative decomposition backend that follows a 'Cirq-style'
association of gates with qubits to form operations. Instead of wiring up `Soquet`s,
each gate operates on qubit addresses (`cirq.Qid`s), which are reused by multiple
gates. This method lets you operate this bloq on qubits and returns a `cirq.Operation`.

The primary, bloq-native way of writing decompositions is to override
`build_composite_bloq`. If this is what you're doing, do not use this method.

To provide a Cirq-style decomposition for this bloq, implement a method (typically named
`decompose_from_registers` for historical reasons) that yields a list of `cirq.Operation`s
using `cirq.Gate.on(...)`, <a href="../qualtran/Bloq.html#on"><code>Bloq.on(...)</code></a>, <a href="../qualtran/GateWithRegisters.html#on_registers"><code>GateWithRegisters.on_registers(...)</code></a>, or
<a href="../qualtran/Bloq.html#on_registers"><code>Bloq.on_registers(...)</code></a>.

See Also




<h3 id="on_registers"><code>on_registers</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L470-L490">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_registers(
    **qubit_regs
) -> 'cirq.Operation'
</code></pre>

A `cirq.Operation` of this bloq operating on the given qubit registers.

This method supports an alternative decomposition backend that follows a 'Cirq-style'
association of gates with qubit registers to form operations. See <a href="../qualtran/Bloq.html#on"><code>Bloq.on()</code></a> for
more details.

Args

`**qubit_regs`
: A mapping of register name to the qubits comprising that register.




See Also




<h3 id="wire_symbol"><code>wire_symbol</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L492-L505">View source</a>

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



