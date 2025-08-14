# HLineFlavor
`qualtran.drawing.musical_score.HLineFlavor`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L74-L86">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Create a collection of name/value pairs.

<!-- Placeholder for "Used in" -->


#### Example enumeration:



```
>>> class Color(Enum):
...     RED = 1
...     BLUE = 2
...     GREEN = 3
```

#### Access them by:



- attribute access::

```
>>> Color.RED
<Color.RED: 1>
```

- value lookup:

```
>>> Color(1)
<Color.RED: 1>
```

- name lookup:

```
>>> Color['RED']
<Color.RED: 1>
```

Enumerations can be iterated over, and know how many members they have:

```
>>> len(Color)
3
```

```
>>> list(Color)
[<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]
```

Methods can be added to enumerations, and members can have their own
attributes -- see the documentation for details.



<h2 class="add-link">Class Variables</h2>

CLASSICAL<a id="CLASSICAL"></a>
: `<HLineFlavor.CLASSICAL: 2>`

QUANTUM<a id="QUANTUM"></a>
: `<HLineFlavor.QUANTUM: 1>`


