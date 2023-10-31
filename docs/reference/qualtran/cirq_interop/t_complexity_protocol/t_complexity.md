# t_complexity


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/cirq_interop/t_complexity_protocol.py#L169-L190">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the TComplexity.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.t_complexity_protocol.t_complexity(
    stc: Any, fail_quietly: bool = False
) -> Optional[<a href="../../../qualtran/cirq_interop/t_complexity_protocol/TComplexity.html"><code>qualtran.cirq_interop.t_complexity_protocol.TComplexity</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`stc`<a id="stc"></a>
: an object to compute its TComplexity.

`fail_quietly`<a id="fail_quietly"></a>
: bool whether to return None on failure or raise an error.




<h2 class="add-link">Returns</h2>




<h2 class="add-link">Raises</h2>

`TypeError`<a id="TypeError"></a>
: if fail_quietly=False and the methods fails to compute TComplexity.


