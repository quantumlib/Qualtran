# GateWithRegisters
`qualtran.GateWithRegisters`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/gate_with_registers.py#L98-L256">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



`cirq.Gate`s extension with support for composite gates acting on multiple qubit registers.

Inherits From: [`Bloq`](../qualtran/Bloq.md)

<!-- Placeholder for "Used in" -->

Though Cirq was nominally designed for circuit construction for near-term devices the core
concept of the `cirq.Gate`, a programmatic representation of an operation on a state without
a complete qubit address specification, can be leveraged to describe more abstract algorithmic
primitives. To define composite gates, users derive from `cirq.Gate` and implement the
`_decompose_` method that yields the sub-operations provided a flat list of qubits.

This API quickly becomes inconvenient when defining operations that act on multiple qubit
registers of variable sizes. Qualtran extends the `cirq.Gate` idea by introducing a new abstract
base class `GateWithRegisters` containing abstract methods `registers` and optional
method `decompose_from_registers` that provides an overlay to the Cirq flat address API.

As an example, in the following code snippet we use the `GateWithRegisters` to
construct a multi-target controlled swap operation:

```
>>> import attr
>>> import cirq
>>> import qualtran
>>>
>>> @attr.frozen
... class MultiTargetCSwap(qualtran.GateWithRegisters):
...     bitsize: int
...
...     @property
...     def signature(self) -> qualtran.Signature:
...         return qualtran.Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)
...
...     def decompose_from_registers(self, context, ctrl, x, y) -> cirq.OP_TREE:
...         yield [cirq.CSWAP(*ctrl, qx, qy) for qx, qy in zip(x, y)]
...
>>> op = MultiTargetCSwap(2).on_registers(
...     ctrl=[cirq.q('ctrl')],
...     x=cirq.NamedQubit.range(2, prefix='x'),
...     y=cirq.NamedQubit.range(2, prefix='y'),
... )
>>> print(cirq.Circuit(op))
ctrl: ───MultiTargetCSwap───
         │
x0: ─────x──────────────────
         │
x1: ─────x──────────────────
         │
y0: ─────y──────────────────
         │
y1: ─────y──────────────────
```



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

<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/gate_with_registers.py#L148-L188">View source</a>

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




<h3 id="as_cirq_op"><code>as_cirq_op</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/gate_with_registers.py#L190-L199">View source</a>

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




<h3 id="t_complexity"><code>t_complexity</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/gate_with_registers.py#L201-L204">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>t_complexity() -> 'TComplexity'
</code></pre>

The `TComplexity` for this bloq.

By default, this will recurse into this bloq's decomposition but this
method can be overriden with a known value.

<h3 id="wire_symbol"><code>wire_symbol</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/gate_with_registers.py#L206-L209">View source</a>

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

<h3 id="decompose_from_registers"><code>decompose_from_registers</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/gate_with_registers.py#L216-L219">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_from_registers(
    *, context: cirq.DecompositionContext, **quregs
) -> cirq.OP_TREE
</code></pre>




<h3 id="on_registers"><code>on_registers</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/gate_with_registers.py#L241-L244">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_registers(
    **qubit_regs
) -> cirq.Operation
</code></pre>




<h3 id="num_qubits"><code>num_qubits</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_qubits() -> int
</code></pre>

The number of qubits this gate acts on.




