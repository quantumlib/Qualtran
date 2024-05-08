# SympySymbolAllocator
`qualtran.resource_counting.SympySymbolAllocator`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_call_graph.py#L37-L53">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A class that allocates unique sympy symbols for integrating out bloq attributes.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.SympySymbolAllocator()
</code></pre>



<!-- Placeholder for "Used in" -->

When counting, we group bloqs that only differ in attributes that do not affect
resource costs. In practice, we do this by replacing bloqs with a version where
the offending attributes have been set to an arbitrary (but unique) symbol allocated
by this class. We refer to this process as "generalizing".

## Methods

<h3 id="new_symbol"><code>new_symbol</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_call_graph.py#L49-L53">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>new_symbol(
    prefix: str
) -> sympy.Symbol
</code></pre>

Return a unique symbol beginning with _prefix.




