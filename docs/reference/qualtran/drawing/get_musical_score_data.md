# get_musical_score_data


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L557-L645">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Get the musical score data for a (composite) bloq.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.drawing.musical_score.get_musical_score_data`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.get_musical_score_data(
    bloq: <a href="../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    manager: Optional[<a href="../../qualtran/drawing/LineManager.html"><code>qualtran.drawing.LineManager</code></a>] = None
) -> <a href="../../qualtran/drawing/MusicalScoreData.html"><code>qualtran.drawing.MusicalScoreData</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This will first walk through the compute graph to assign each soquet
to a register position. Then we iterate again to finalize drawing-relevant
properties like symbols and the various horizontal and vertical lines.

<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq or composite bloq to get drawing data for

`manager`<a id="manager"></a>
: Optionally provide an override of `LineManager` if you want
  to control the allocation of horizontal (register) lines.


