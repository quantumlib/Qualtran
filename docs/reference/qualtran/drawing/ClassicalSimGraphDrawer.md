# ClassicalSimGraphDrawer
`qualtran.drawing.ClassicalSimGraphDrawer`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/classical_sim_graph.py#L29-L66">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A graph drawer that labels each edge with a classical value.

Inherits From: [`PrettyGraphDrawer`](../../qualtran/drawing/PrettyGraphDrawer.md), [`GraphDrawer`](../../qualtran/drawing/GraphDrawer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.drawing.classical_sim_graph.ClassicalSimGraphDrawer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.ClassicalSimGraphDrawer(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    vals: Dict[str, 'ClassicalValT']
)
</code></pre>



<!-- Placeholder for "Used in" -->

The (composite) bloq must be composed entirely of classically-simulable bloqs.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The (composite) bloq to draw.

`vals`<a id="vals"></a>
: Input classical values to propogate through the composite bloq.




## Methods

<h3 id="cxn_label"><code>cxn_label</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/classical_sim_graph.py#L48-L54">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cxn_label(
    cxn: <a href="../../qualtran/Connection.html"><code>qualtran.Connection</code></a>
) -> str
</code></pre>

Label the connection with its classical value.


<h3 id="cxn_edge"><code>cxn_edge</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/classical_sim_graph.py#L56-L66">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cxn_edge(
    left_id: str,
    right_id: str,
    cxn: <a href="../../qualtran/Connection.html"><code>qualtran.Connection</code></a>
) -> pydot.Edge
</code></pre>

Overridable method to style a pydot.Edge for connecionts.




