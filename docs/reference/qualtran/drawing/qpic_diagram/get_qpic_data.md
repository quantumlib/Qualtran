# get_qpic_data


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/qpic_diagram.py#L254-L273">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Get the input data that can be used to draw a latex diagram for `bloq` using `qpic`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.qpic_diagram.get_qpic_data(
    bloq: 'Bloq', file_path: Union[None, pathlib.Path, str] = None
) -> List[str]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: Bloqs to draw the latex diagram for

`file_path`<a id="file_path"></a>
: If specified, the output is stored at the file. Otherwise, the data is returned.




<h2 class="add-link">Returns</h2>


