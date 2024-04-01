# iter_ccz2t_factories


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_cost_model.py#L270-L297">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Iterate over CCZ2T (multi)factories in the given range of distillation code distances


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.ccz2t_cost_model.iter_ccz2t_factories(
    l1_start: int = 5, l1_stop: int = 25, l2_stop: int = 41, *, n_factories=1
) -> Iterator[<a href="../../../qualtran/surface_code/MagicStateFactory.html"><code>qualtran.surface_code.MagicStateFactory</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`l1_start`<a id="l1_start"></a>
: `int, optional`
  
  Minimum level 1 distillation distance.

`l1_stop`<a id="l1_stop"></a>
: `int, optional`
  
  Maximum level 1 distillation distance.

`l2_stop`<a id="l2_stop"></a>
: `int, optional`
  
  Maximum level 2 distillation distance. The minimum is
      automatically chosen as 2 + l1_distance, ensuring l2_distance > l1_distance.

`n_factories`<a id="n_factories"></a>
: `int, optional`
  
  Number of factories to be used in parallel.


