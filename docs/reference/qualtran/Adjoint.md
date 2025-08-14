# Adjoint
`qualtran.Adjoint`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/adjoint.py#L89-L186">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The standard adjoint of `subbloq`.

Inherits From: [`GateWithRegisters`](../qualtran/GateWithRegisters.md), [`Bloq`](../qualtran/Bloq.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.Adjoint(
    subbloq
)
</code></pre>



<!-- Placeholder for "Used in" -->

This metabloq generally delegates all of its protocols (with modifications, read on) to
`subbloq`. This class is used in the default implementation of the adjoint protocol, i.e.,
in the default implementation of <a href="../qualtran/Bloq.html#adjoint"><code>Bloq.adjoint()</code></a>.

This metabloq is appropriate in most cases since there rarely a specialized
(one level) decomposition for a bloq's adjoint. Exceptions can be found for decomposing
some low-level primitives, for example `And`. Even if you use bloqs with specialized
adjoints in your decomposition (i.e. you use `And`), you can still rely on this standard
behavior.

This bloq is defined entirely in terms of how it delegates its protocols. The following
protocols delegate to `subbloq` (with appropriate modification):

 - **Signature**: The signature is the adjoint of `subbloqs`'s signature. Namely, LEFT
   and RIGHT registers are swapped.
 - **Decomposition**: The decomposition is the adjoint of `subbloq`'s decomposition. Namely,
   the order of operations in the resultant `CompositeBloq` is reversed and each bloq is
   replaced with its adjoint.
 - **Adjoint**: The adjoint of an `Adjoint` bloq is the subbloq itself.
 - **Call graph**: The call graph is the subbloq's call graph, but each bloq is replaced
   with its adjoint.
 - **Cirq Interop**: The default `Bloq` implementation is used, which goes via `BloqAsCirqGate`
   as usual.
 - **Wire Symbol**: The wire symbols are the adjoint of `subbloq`'s wire symbols. Namely,
   left- and right-oriented symbols are flipped.
 - **Names**: The string names / labels are that of the `subbloq` with a dagger symbol appended.

Some protocols are impossible to delegate specialized implementations. The `Adjoint` bloq
supports the following protocols with "decompose-only" implementations. This means we always
go via the bloq's decomposition instead of preferring specialized implementations provided by
the bloq author. If a specialized implementation of these protocols are required or you
are trying to represent an adjoint bloq without a decomposition and need to support these
protocols, use a specialized adjoint bloq or attribute instead of this class.

 - Classical simulation is "decompose-only". It is impossible to invert a generic python
   function.
 - Tensor simulation is "decompose-only" due to technical details around the Quimb interop.

<h2 class="add-link">Args</h2>

`subbloq`<a id="subbloq"></a>
: The bloq to wrap.






<h2 class="add-link">Attributes</h2>

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

<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/adjoint.py#L142-L144">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_bloq() -> 'CompositeBloq'
</code></pre>

The decomposition is the adjoint of `subbloq`'s decomposition.


<h3 id="adjoint"><code>adjoint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/adjoint.py#L157-L159">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjoint() -> 'Bloq'
</code></pre>

The 'double adjoint' brings you back to the original bloq.


<h3 id="build_call_graph"><code>build_call_graph</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/adjoint.py#L161-L171">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_call_graph(
    ssa: 'SympySymbolAllocator'
) -> 'BloqCountDictT'
</code></pre>

The call graph takes the adjoint of each of the bloqs in `subbloq`'s call graph.


<h3 id="wire_symbol"><code>wire_symbol</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/adjoint.py#L177-L186">View source</a>

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

Method generated by attrs for class Adjoint.




