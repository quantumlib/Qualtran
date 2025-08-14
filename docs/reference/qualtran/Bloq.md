# Bloq
`qualtran.Bloq`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L97-L638">
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

<h3 id="build_composite_bloq"><code>build_composite_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L132-L147">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L149-L163">View source</a>

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




<h3 id="as_composite_bloq"><code>as_composite_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L165-L174">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_composite_bloq() -> 'CompositeBloq'
</code></pre>

Wrap this Bloq into a size-1 CompositeBloq.

This method is overriden so if this Bloq is already a CompositeBloq, it will
be returned.

<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L176-L188">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjoint() -> 'Bloq'
</code></pre>

The adjoint of this bloq.

Bloq authors can override this method in certain circumstances. Otherwise, the default
fallback wraps this bloq in `Adjoint`.

Please see the documentation for `Adjoint` and the `Adjoint.ipynb` notebook for full
details.

<h3 id="on_classical_vals"><code>on_classical_vals</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L190-L221">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_classical_vals(
    **vals
) -> Mapping[str, 'ClassicalValRetT']
</code></pre>

How this bloq operates on classical data.

Override this method if your bloq represents classical, reversible logic. For example:
quantum circuits composed of X and C^nNOT gates are classically simulable.

Bloq authors should override this method. If you already have an instance of a `Bloq`,
consider calling `call_clasically(**vals)` which will do input validation before
calling this function.

Args

`**vals`
: The input classical values for each left (or thru) register. The data
  types are guaranteed to match `self.signature`. Values for registers
  with a particular dtype will be the corresponding classical data type. Values for
  registers with `shape` will be an ndarray of values of the expected type.




Returns




<h3 id="basis_state_phase"><code>basis_state_phase</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L223-L240">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>basis_state_phase(
    **vals
) -> Union[complex, None]
</code></pre>

How this bloq phases classical basis states.

Override this method if your bloq represents classical logic with basis-state
dependent phase factors. This corresponds to bloqs whose matrix representation
(in the standard basis) is a generalized permutation matrix: a permutation matrix
where each entry can be +1, -1 or any complex number with unit absolute value.
Alternatively, this corresponds to bloqs composed from classical operations
(X, CNOT, Toffoli, ...) and diagonal operations (T, CZ, CCZ, ...).

Bloq authors should override this method. If you are using an instantiated bloq object,
call

If this method is implemented, `on_classical_vals` must also be implemented.
If `on_classical_vals` is implemented but this method is not implemented, it is assumed
that the bloq does not alter the phase.

<h3 id="call_classically"><code>call_classically</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L242-L264">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L266-L300">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tensor_contract(
    superoperator: bool = False
) -> 'NDArray'
</code></pre>

Return a contracted, dense ndarray encoding of this bloq.

This method decomposes and flattens this bloq into a factorized CompositeBloq,
turns that composite bloq into a Quimb tensor network, and contracts it into a dense
ndarray.

The returned array will be 0-, 1-, 2-, or 4-dimensional with indices arranged according to the
bloq's signature and the type of simulation requested via the `superoperator` flag.

If `superoperator` is set to False (the default), a pure-state tensor network will be
constructed.
 - If `bloq` has all thru-registers, the dense tensor will be 2-dimensional with shape `(n, n)`
   where `n` is the number of bits in the signature. We follow the linear algebra convention
   and order the indices as (right, left) so the matrix-vector product can be used to evolve
   a state vector.
 - If `bloq` has all left- or all right-registers, the tensor will be 1-dimensional with
   shape `(n,)`. Note that we do not distinguish between 'row' and 'column' vectors in this
   function.
 - If `bloq` has no external registers, the contracted form is a 0-dimensional complex number.

If `superoperator` is set to True, an open-system tensor network will be constructed.
 - States result in a 2-dimensional density matrix with indices (right_forward, right_backward)
   or (left_forward, left_backward) depending on whether they're input or output states.
 - Operations result in a 4-dimensional tensor with indices (right_forward, right_backward,
   left_forward, left_backward).

Args

`superoperator`
: If toggled to True, do an open-system simulation. This supports
  non-unitary operations like measurement, but is more costly and results in
  higher-dimension resultant tensors.




<h3 id="my_tensors"><code>my_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L302-L334">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>my_tensors(
    incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
) -> List[Union['qtn.Tensor', 'DiscardInd']]
</code></pre>

Override this method to support native quimb simulation of this Bloq.

This method is responsible for returning tensors corresponding to the unitary, state, or
effect of the bloq. Often, this method will return one tensor for a given Bloq, but
some bloqs can be represented in a factorized form using more than one tensor.

By default, calls to <a href="../qualtran/Bloq.html#tensor_contract"><code>Bloq.tensor_contract()</code></a> will first decompose and flatten the bloq
before initiating the conversion to a tensor network. This has two consequences:
 1) Overriding this method is only necessary if this bloq does not define a decomposition
    or if the fully-decomposed form contains a bloq that does not define its tensors.
 2) Even if you override this method to provide custom tensors, they may not be used
    (by default) because we prefer the flat-decomposed version. This is usually desirable
    for contraction performance; but for finer-grained control see
    <a href="../qualtran/simulation/tensor/cbloq_to_quimb.html"><code>qualtran.simulation.tensor.cbloq_to_quimb</code></a>.

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




<h3 id="build_call_graph"><code>build_call_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L336-L353">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_call_graph(
    ssa: 'SympySymbolAllocator'
) -> Union['BloqCountDictT', Set['BloqCountT']]
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

<h3 id="my_static_costs"><code>my_static_costs</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L355-L367">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>my_static_costs(
    cost_key: 'CostKey'
)
</code></pre>

Override this method to provide static costs.

The system will query a particular cost by asking for a `cost_key`. This method
can optionally provide a value, which will be preferred over a computed cost.

Static costs can be provided if the particular cost cannot be easily computed or
as a performance optimization.

This method must return `NotImplemented` if a value cannot be provided for the specified
CostKey.

<h3 id="call_graph"><code>call_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L369-L399">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call_graph(
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
    keep: Optional[Callable[['Bloq'], bool]] = None,
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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L401-L421">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L423-L459">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_ctrl_system(
    ctrl_spec: 'CtrlSpec'
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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L461-L483">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L485-L493">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>t_complexity() -> 'TComplexity'
</code></pre>

The `TComplexity` for this bloq.

By default, this will recurse into this bloq's decomposition but this
method can be overriden with a known value.

<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L495-L522">View source</a>

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




<h3 id="as_pl_op"><code>as_pl_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L524-L541">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_pl_op(
    wires: 'Wires'
) -> 'Operation'
</code></pre>

Override this method to support conversion to a PennyLane operation.

If this method is not overriden, the default implementation will wrap this bloq
in a `FromBloq` shim.

Args

`wires`
: the wires that the op acts on




Returns




<h3 id="on"><code>on</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L543-L569">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L571-L591">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/bloq.py#L593-L624">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>wire_symbol(
    reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
) -> 'WireSymbol'
</code></pre>

On a musical score visualization, use this `WireSymbol` to represent `soq`.

By default, we use a "directional text box", which is a text box that is either
rectangular for thru-registers or facing to the left or right for non-thru-registers.

If reg is specified as `None`, this should return a Text label for the title of
the gate. If no title is needed (as the wire_symbols are self-explanatory),
this should return `Text('')`.

Override this method to provide a more relevant `WireSymbol` for the provided soquet.
This method can access bloq attributes. For example: you may want to draw either
a filled or empty circle for a control register depending on a control value bloq
attribute.



