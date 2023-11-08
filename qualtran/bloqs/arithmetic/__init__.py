from qualtran.bloqs.arithmetic.addition import Add, AddConstantMod, OutOfPlaceAdder
from qualtran.bloqs.arithmetic.comparison import (
    BiQubitsMixer,
    EqualsAConstant,
    GreaterThan,
    GreaterThanConstant,
    LessThanConstant,
    LessThanEqual,
    SingleQubitCompare,
)
from qualtran.bloqs.arithmetic.conversions import SignedIntegerToTwosComplement, ToContiguousIndex
from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
from qualtran.bloqs.arithmetic.multiplication import (
    MultiplyTwoReals,
    Product,
    ScaleIntByReal,
    Square,
    SquareRealNumber,
    SumOfSquares,
)
