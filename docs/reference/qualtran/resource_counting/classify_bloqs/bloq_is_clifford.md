# bloq_is_clifford


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/classify_bloqs.py#L163-L208">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Whether the bloq represents a clifford operation.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.classify_bloqs.bloq_is_clifford(
    b: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

This checks against an explicit list of clifford bloqs in the Qualtran standard library,
so it may return `False` for an unknown gate.

This inspects single qubit rotations. If the angles correspond to Clifford angles, this
returns `True`.