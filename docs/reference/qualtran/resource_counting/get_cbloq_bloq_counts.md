# get_cbloq_bloq_counts


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/resource_counting/bloq_counts.py#L56-L80">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Count all the subbloqs in a composite bloq.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.resource_counting.bloq_counts.get_cbloq_bloq_counts`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.get_cbloq_bloq_counts(
    cbloq: <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>,
    generalizer: Callable[[Bloq], Optional[Bloq]] = None
) -> Set[<a href="../../qualtran/resource_counting/BloqCountT.html"><code>qualtran.resource_counting.BloqCountT</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

`CompositeBloq.resource_counting` calls this with no generalizer.

<h2 class="add-link">Args</h2>

`cbloq`<a id="cbloq"></a>
: The composite bloq.

`generalizer`<a id="generalizer"></a>
: A function that replaces bloq attributes that do not affect resource costs
  with sympy placeholders.


