# bloq_is_rotation


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/classify_bloqs.py#L211-L265">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Whether a bloq represents a rotation operation.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.classify_bloqs.bloq_is_rotation(
    b: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

This inspects the single qubit rotation bloqs and returns `True` unless the angle
represents a clifford or T-gate angle.

This function has a shim for counting Controlled[Rotation] gates as a rotation, which
will be remediated when the Qualtran standard library gains a bespoke bloq for each CRot.