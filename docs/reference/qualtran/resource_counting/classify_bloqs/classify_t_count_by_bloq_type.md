# classify_t_count_by_bloq_type


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/classify_bloqs.py#L72-L112">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Classify (bin) the T count of a bloq's call graph by type of operation.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.classify_bloqs.classify_t_count_by_bloq_type(
    bloq: <a href="../../../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>,
    bloq_classification: Optional[Dict[str, str]] = None,
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None
) -> Dict[str, Union[int, sympy.Expr]]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`bloq`<a id="bloq"></a>
: the bloq to classify.

`bloq_classification`<a id="bloq_classification"></a>
: An optional dictionary mapping bloq_classifications to bloq types.

`generalizer`<a id="generalizer"></a>
: If provided, run this function on each (sub)bloq to replace attributes
  that do not affect resource estimates with generic sympy symbols. If the function
  returns `None`, the bloq is omitted from the counts graph. If a sequence of
  generalizers is provided, each generalizer will be run in order.



Returns
    classified_bloqs: dictionary containing the T count for different types of bloqs.