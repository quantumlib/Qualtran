# qpic_input_to_diagram


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/qpic_diagram.py#L284-L358">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Invoke `qpic` script to generate output diagram of type qpic/tex/pdf/png.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.qpic_diagram.qpic_input_to_diagram(
    qpic_file: Union[pathlib.Path, str],
    output_file: Union[None, pathlib.Path, str] = None,
    output_type: str = &#x27;pdf&#x27;
) -> str
</code></pre>



<!-- Placeholder for "Used in" -->

Outputs one of the following files, based on `output_type`:
 - qpic_file.with_suffix('.qpic') - Copies the input qpic_file to output_file.
 - qpic_file.with_suffix('.tex') - Obtained via `qpic -f base_file_path.qpic`
 - qpic_file.with_suffix('.pdf') - Uses `pdflatex` tool to convert tex to pdf. See
    https://tug.org/applications/pdftex/ for more details on how to install `pdflatex`.
 - qpic_file.with_suffix('.png') - Uses `convert` tool to convert pdf to png. See
    https://imagemagick.org/ for more details on how to install `convert`.

<h2 class="add-link">Args</h2>

`qpic_file`<a id="qpic_file"></a>
: Path to file containing input that should be passed to the `qpic` script.

`output_file`<a id="output_file"></a>
: Optional path to the output where generated diagram should be saved. Defaults to
  qpic_file.with_suffix(f".{output_type}")

`output_type`<a id="output_type"></a>
: Format of the diagram generated using qpic. Supported output types are one of
  ['qpic', 'tex', 'pdf', 'png']




<h2 class="add-link">Returns</h2>


