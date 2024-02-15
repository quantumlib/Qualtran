# MusicalScoreEncoder
`qualtran.drawing.musical_score.MusicalScoreEncoder`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L688-L695">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An encoder that handles `MusicalScoreData` classes and those of its contents.

<!-- Placeholder for "Used in" -->


## Methods

<h3 id="default"><code>default</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L691-L695">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>default(
    o: Any
) -> Any
</code></pre>

Implement this method in a subclass such that it returns a serializable object for ``o``, or calls the base implementation (to raise a ``TypeError``).

For example, to support arbitrary iterators, you could
implement default like this::

    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)





<h2 class="add-link">Class Variables</h2>

item_separator<a id="item_separator"></a>
: `', '`

key_separator<a id="key_separator"></a>
: `': '`


