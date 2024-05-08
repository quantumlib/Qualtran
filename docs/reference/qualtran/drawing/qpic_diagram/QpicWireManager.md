# QpicWireManager
`qualtran.drawing.qpic_diagram.QpicWireManager`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/qpic_diagram.py#L87-L126">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Methods to manage allocation/deallocation of wires for QPIC diagrams.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.qpic_diagram.QpicWireManager()
</code></pre>



<!-- Placeholder for "Used in" -->

QPIC places wires in the order in which they are defined. For each soquet, the wire manager
allocates a new wire with prefix `_wire_name_prefix_for_soq(soq)` and an integer suffix that
corresponds to the smallest integer which does not correspond to an allocated wire.

## Methods

<h3 id="alloc_wire_for_soq"><code>alloc_wire_for_soq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/qpic_diagram.py#L109-L113">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>alloc_wire_for_soq(
    soq: <a href="../../../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
) -> str
</code></pre>




<h3 id="dealloc_wire_for_soq"><code>dealloc_wire_for_soq</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/qpic_diagram.py#L115-L119">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dealloc_wire_for_soq(
    soq: <a href="../../../qualtran/Soquet.html"><code>qualtran.Soquet</code></a>
) -> str
</code></pre>




<h3 id="soq_to_wirename"><code>soq_to_wirename</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/qpic_diagram.py#L121-L123">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>soq_to_wirename(
    soq
) -> str
</code></pre>




<h3 id="soq_to_wirelabel"><code>soq_to_wirelabel</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/qpic_diagram.py#L125-L126">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>soq_to_wirelabel(
    soq
) -> str
</code></pre>






