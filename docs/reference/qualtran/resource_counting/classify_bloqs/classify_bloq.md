# classify_bloq


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/classify_bloqs.py#L51-L68">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Classify a bloq given a bloq_classification.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.classify_bloqs.classify_bloq(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    bloq_classification: Dict[str, str]
) -> str
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: The bloq to classify

`bloq_classification`<a id="bloq_classification"></a>
: A dictionary mapping a classification to a tuple of
  bloqs in that classification.




<h2 class="add-link">Returns</h2>

`classification`<a id="classification"></a>
: The matching key in bloq_classification. Returns other if not classified.


