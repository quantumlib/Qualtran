# assert_connections_consistent_qdtypes


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/testing.py#L140-L157">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check that a composite bloq's connections have consistent qdtypes.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.testing.assert_connections_consistent_qdtypes(
    cbloq: <a href="../../qualtran/CompositeBloq.html"><code>qualtran.CompositeBloq</code></a>,
    type_checking_severity: QDTypeCheckingSeverity = QDTypeCheckingSeverity.LOOSE
)
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`cbloq`<a id="cbloq"></a>
: The composite bloq.

`type_checking_severity`<a id="type_checking_severity"></a>
: How strict to be in type checking. See the documentation
  for the QDTypeCheckingSeverity enum for details.


