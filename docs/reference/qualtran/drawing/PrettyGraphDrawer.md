# PrettyGraphDrawer
`qualtran.drawing.PrettyGraphDrawer`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/graphviz.py#L368-L405">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A class to encapsulate methods for displaying a CompositeBloq as a graph using graphviz.

Inherits From: [`GraphDrawer`](../../qualtran/drawing/GraphDrawer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.drawing.graphviz.PrettyGraphDrawer`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Graphviz has nodes, edges, and ports. Nodes are HTML tables representing bloq instances.
Each cell in the table has a graphviz port and represents a soquet. Edges connect
node:port tuples representing connections between soquets.

Each node and port has a string identifier. We use the `_IDBuilder` helper class
to assign unique, readable IDs to each object.

Users should call <a href="../../qualtran/drawing/GraphDrawer.html#get_graph"><code>GraphDrawer.get_graph()</code></a> as the primary entry point. Other methods
can be overridden to customize the look of the resulting graph.

To display a graph in a jupyter notebook consider using the SVG utilities:

```
>>> dr = GraphDrawer(cbloq)
>>> dr.get_svg()
```

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq or composite bloq to draw.




## Methods

<h3 id="get_binst_table_attributes"><code>get_binst_table_attributes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/graphviz.py#L369-L370">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_binst_table_attributes() -> str
</code></pre>

Overridable method to configure the desired table attributes for the bloq.


<h3 id="get_binst_header_text"><code>get_binst_header_text</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/graphviz.py#L372-L377">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_binst_header_text(
    binst: <a href="../../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>
)
</code></pre>

Overridable method returning the text used for the header cell of a bloq.


<h3 id="soq_label"><code>soq_label</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/graphviz.py#L379-L388">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>soq_label(
    soq: <a href="../../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
)
</code></pre>

Overridable method for getting label text for a Soquet.


<h3 id="get_default_text"><code>get_default_text</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/graphviz.py#L390-L394">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_default_text(
    reg: <a href="../../qualtran/Register.html"><code>qualtran.Register</code></a>
) -> str
</code></pre>




<h3 id="cxn_edge"><code>cxn_edge</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/drawing/graphviz.py#L396-L405">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cxn_edge(
    left_id: str,
    right_id: str,
    cxn: <a href="../../qualtran/Connection.html"><code>qualtran.Connection</code></a>
) -> pydot.Edge
</code></pre>

Overridable method to style a pydot.Edge for connecionts.




