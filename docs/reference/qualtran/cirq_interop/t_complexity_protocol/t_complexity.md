# t_complexity


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/t_complexity_protocol.py#L239-L256">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the TComplexity of a bloq.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.t_complexity_protocol.t_complexity(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>
) -> <a href="../../../qualtran/cirq_interop/t_complexity_protocol/TComplexity.html"><code>qualtran.cirq_interop.t_complexity_protocol.TComplexity</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to compute the T complexity.




<h2 class="add-link">Returns</h2>




<h2 class="add-link">Raises</h2>

`TypeError`<a id="TypeError"></a>
: if none of the strategies can derive the t complexity.


