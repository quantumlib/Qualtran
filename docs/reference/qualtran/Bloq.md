# Bloq
`qualtran.Bloq`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L34-L291">
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
  
  This is the only manditory method (property) you must implement to inherit from
  `Bloq`. You can optionally implement additional methods to encode more information
  about this bloq.




## Methods

<h3 id="pretty_name"><code>pretty_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L69-L70">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pretty_name() -> str
</code></pre>




<h3 id="short_name"><code>short_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L72-L77">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>short_name() -> str
</code></pre>




<h3 id="build_composite_bloq"><code>build_composite_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L79-L94">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_composite_bloq(
    bb: 'BloqBuilder', **soqs
) -> Dict[str, 'SoquetT']
</code></pre>

Override this method to define a Bloq in terms of its constituent parts.

Bloq definers should override this method. If you already have an instance of a `Bloq`,
consider calling `decompose_bloq()` which will set up the correct context for
calling this function.

Args

`bb`
: A `BloqBuilder` to append sub-Bloq to.

`**soqs`
: The initial soquets corresponding to the inputs to the Bloq.




Returns




<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L96-L114">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L116-L123">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>supports_decompose_bloq() -> bool
</code></pre>

Whether this bloq supports `.decompose_bloq()`.

By default, we check that the method `build_composite_bloq` is overriden. For
extraordinary circumstances, you may need to override this method directly to
return an accurate value.

<h3 id="as_composite_bloq"><code>as_composite_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L125-L134">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_composite_bloq() -> 'CompositeBloq'
</code></pre>

Wrap this Bloq into a size-1 CompositeBloq.

This method is overriden so if this Bloq is already a CompositeBloq, it will
be returned.

<h3 id="on_classical_vals"><code>on_classical_vals</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L136-L160">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L162-L182">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L184-L192">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tensor_contract() -> 'NDArray'
</code></pre>

Return a contracted, dense ndarray representing this bloq.

This constructs a tensor network and then contracts it according to our registers,
i.e. the dangling indices. The returned array will be 0-, 1- or 2-dimensional. If it is
a 2-dimensional matrix, we follow the quantum computing / matrix multiplication convention
of (right, left) indices.

<h3 id="add_my_tensors"><code>add_my_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L194-L228">View source</a>

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




<h3 id="bloq_counts"><code>bloq_counts</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L230-L239">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>bloq_counts(
    ssa: 'SympySymbolAllocator'
) -> List['BloqCountT']
</code></pre>

Return a list of `(n, bloq)` tuples where bloq is used `n` times in the decomposition.

By default, this method will use `self.decompose_bloq()` to count up bloqs.
However, you can override this if you don't want to provide a complete decomposition,
if you know symbolic expressions for the counts, or if you need to "generalize"
the subbloqs by overwriting bloq attributes that do not affect its cost with generic
sympy symbols (perhaps with the aid of the provided `SympySymbolAllocator`).

<h3 id="t_complexity"><code>t_complexity</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L241-L247">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>t_complexity() -> 'TComplexity'
</code></pre>

The `TComplexity` for this bloq.

By default, this will recurse into this bloq's decomposition but this
method can be overriden with a known value.

<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L249-L276">View source</a>

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




<h3 id="wire_symbol"><code>wire_symbol</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq.py#L278-L291">View source</a>

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



