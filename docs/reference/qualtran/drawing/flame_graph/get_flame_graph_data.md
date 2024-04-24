# get_flame_graph_data


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/flame_graph.py#L126-L165">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Get the flame graph data for visualizing T-costs distribution of a sequence of bloqs.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.flame_graph.get_flame_graph_data(
    *bloqs,
    file_path: Union[None, pathlib.Path, str] = None,
    keep: Optional[Callable[['Bloq'], bool]] = _keep_if_small,
    **kwargs
) -> List[str]
</code></pre>



<!-- Placeholder for "Used in" -->

For each bloq in the input, this will do a DFS ordering over all edges in the DAG and
add an entry corresponding to each leaf node in the call graph. The string representation
added for a leaf node encodes the entire path taken from the root node to the leaf node
and is repeated a number of times that's equivalent to the weight of that path. Thus, the
length of the output would be roughly equal to the number of T-gates in the Bloq and can be
very high. If you want to limit the output size, consider specifying a `keep` predicate where
the leaf nodes are higher level Bloqs with a larger T-count weight.

<h2 class="add-link">Args</h2>

`bloqs`<a id="bloqs"></a>
: Bloqs to plot the flame graph for

`file_path`<a id="file_path"></a>
: If specified, the output is stored at the file.

`keep`<a id="keep"></a>
: A predicate to determine the leaf nodes in the call graph. The flame graph would use
  these Bloqs as leaf nodes and thus would not contain decompositions for these nodes.

`**kwargs`<a id="**kwargs"></a>
: Additional arguments to be passed to `bloq.call_graph`, like generalizers etc.




<h2 class="add-link">Returns</h2>


