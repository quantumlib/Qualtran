# qpic_diagram_for_bloq


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/qpic_diagram.py#L361-L393">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generate latex diagram for `bloq` by invoking `qpic`. Assumes qpic is already installed.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.qpic_diagram.qpic_diagram_for_bloq(
    bloq: 'Bloq',
    base_file_path: Union[None, pathlib.Path, str] = None,
    output_type: str = &#x27;pdf&#x27;
) -> str
</code></pre>



<!-- Placeholder for "Used in" -->

Outputs one of the following files, based on `output_type`:
 - base_file_path + '.qpic' - Obtained via get_qpic_data(bloq)
 - base_file_path + '.tex' - Obtained via `qpic -f base_file_path.qpic`
 - base_file_path + '.pdf' - Uses `pdflatex` tool to convert tex to pdf. See
    https://tug.org/applications/pdftex/ for more details on how to install `pdflatex`.
 - base_file_path + '.png' - Uses `convert` tool to convert pdf to png. See
    https://imagemagick.org/ for more details on how to install `convert`.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to generate a qpic diagram for

`base_file_path`<a id="base_file_path"></a>
: Prefix of the path where output file is saved. The output file corresponds
  to f'{base_file_path}.{output_type}'

`output_type`<a id="output_type"></a>
: Format of the diagram generated using qpic. Supported output types are one of
  ['qpic', 'tex', 'pdf', 'png']




<h2 class="add-link">Returns</h2>


