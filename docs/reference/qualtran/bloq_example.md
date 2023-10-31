# bloq_example


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/bloq_example.py#L85-L107">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decorator to turn a function into a `BloqExample`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.bloq_example(
    _func: Callable[[], <a href="../qualtran/Bloq.html"><code>qualtran.Bloq</code></a>] = None,
    *,
    generalizer: Callable[[Bloq], Optional[Bloq]] = (lambda x: x)
)
</code></pre>



<!-- Placeholder for "Used in" -->

This will set `name` to the name of the function and `bloq_cls` according to the return-type
annotation. You can also call the decorator with keyword arguments, which will be passed
through to the `BloqExample` constructor.