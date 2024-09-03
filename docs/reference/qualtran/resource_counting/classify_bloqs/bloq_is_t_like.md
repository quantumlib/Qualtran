# bloq_is_t_like


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/classify_bloqs.py#L134-L160">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Whether a bloq should be counted as a T gate.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.classify_bloqs.bloq_is_t_like(
    b: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

This will return `True` for any instance of `TGate`. It will also consider
single-qubit rotations and return True if the angle corresponds to a T gate
(up to clifford reference frame).