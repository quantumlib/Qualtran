# Module: musical_score


Tools for laying out composite bloq graphs onto a "musical score".



A musical score is one where time proceeds from left to right and each horizontal line
represents a qubit or register of qubits.
## Classes

[`class RegPosition`](../../qualtran/drawing/RegPosition.md): Coordinates for a register when visualizing on a musical score.

[`class HLine`](../../qualtran/drawing/HLine.md): Dataclass representing a horizontal line segment at a given vertical position `x`.

[`class LineManager`](../../qualtran/drawing/LineManager.md): Methods to manage allocation and de-allocation of lines representing a register of qubits.

[`class WireSymbol`](../../qualtran/drawing/WireSymbol.md): Base class for a symbol.

[`class TextBox`](../../qualtran/drawing/TextBox.md): Base class for a symbol.

[`class Text`](../../qualtran/drawing/Text.md): Base class for a symbol.

[`class RarrowTextBox`](../../qualtran/drawing/RarrowTextBox.md): Base class for a symbol.

[`class LarrowTextBox`](../../qualtran/drawing/LarrowTextBox.md): Base class for a symbol.

[`class Circle`](../../qualtran/drawing/Circle.md): Base class for a symbol.

[`class ModPlus`](../../qualtran/drawing/ModPlus.md): Base class for a symbol.

[`class SoqData`](../../qualtran/drawing/musical_score/SoqData.md): Data needed to draw a soquet.

[`class VLine`](../../qualtran/drawing/VLine.md): Data for drawing vertical lines.

[`class MusicalScoreData`](../../qualtran/drawing/MusicalScoreData.md): All the data required to draw a musical score.

[`class MusicalScoreEncoder`](../../qualtran/drawing/musical_score/MusicalScoreEncoder.md): An encoder that handles `MusicalScoreData` classes and those of its contents.

## Functions

[`directional_text_box(...)`](../../qualtran/drawing/directional_text_box.md)

[`draw_musical_score(...)`](../../qualtran/drawing/draw_musical_score.md)

[`dump_musical_score(...)`](../../qualtran/drawing/dump_musical_score.md)

[`frozen(...)`](../../qualtran/drawing/musical_score/frozen.md): partial(func, *args, **keywords) - new function with partial application of the given arguments and keywords.

[`get_musical_score_data(...)`](../../qualtran/drawing/get_musical_score_data.md): Get the musical score data for a (composite) bloq.

## Type Aliases

[`NDArray`](../../qualtran/drawing/musical_score/NDArray.md)



<h2 class="add-link">Other Members</h2>

LeftDangle<a id="LeftDangle"></a>
: Instance of <a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>

RightDangle<a id="RightDangle"></a>
: Instance of <a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>


