# show_bloq


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/_show_funcs.py#L33-L50">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Display a visual representation of the bloq in IPython.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.show_bloq(
    bloq: 'Bloq', type: str = &#x27;graph&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to show

`type`<a id="type"></a>
: Either 'graph', 'dtype', or 'musical_score'. By default, display a directed acyclic
  graph of the bloq connectivity. If dtype then the connections are
  labelled with their dtypes rather than bitsizes. Otherwise, draw a
  musical score diagram.


