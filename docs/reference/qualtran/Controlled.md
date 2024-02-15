# Controlled
`qualtran.Controlled`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L206-L323">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A controlled version of `subbloq`.

Inherits From: [`Bloq`](../qualtran/Bloq.md)

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L224-L244">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>make_ctrl_system(
    bloq: 'Bloq', ctrl_spec: 'CtrlSpec'
) -> Tuple[<a href="../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>, <a href="../qualtran/AddControlledT.html"><code>qualtran.AddControlledT</code></a>]
</code></pre>

A factory method for creating both the Controlled and the adder function.

See <a href="../qualtran/Bloq.html#get_ctrl_system"><code>Bloq.get_ctrl_system</code></a>.

<h3 id="decompose_bloq"><code>decompose_bloq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L266-L287">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L289-L293">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L295-L305">View source</a>

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




<h3 id="wire_symbol"><code>wire_symbol</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L307-L314">View source</a>

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

<h3 id="pretty_name"><code>pretty_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L316-L317">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pretty_name() -> str
</code></pre>




<h3 id="short_name"><code>short_name</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L319-L320">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>short_name() -> str
</code></pre>




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




