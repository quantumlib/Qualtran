#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from qualtran.bloqs.arithmetic.addition import Add, AddK, OutOfPlaceAdder
from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor, XorK
from qualtran.bloqs.arithmetic.comparison import (
    BiQubitsMixer,
    CLinearDepthGreaterThan,
    EqualsAConstant,
    GreaterThan,
    GreaterThanConstant,
    LessThanConstant,
    LessThanEqual,
    SingleQubitCompare,
)
from qualtran.bloqs.arithmetic.controlled_addition import CAdd
from qualtran.bloqs.arithmetic.conversions import (
    SignedIntegerToTwosComplement,
    SignExtend,
    ToContiguousIndex,
)
from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
from qualtran.bloqs.arithmetic.multiplication import (
    InvertRealNumber,
    MultiplyTwoReals,
    PlusEqualProduct,
    Product,
    ScaleIntByReal,
    Square,
    SquareRealNumber,
    SumOfSquares,
)
from qualtran.bloqs.arithmetic.negate import Negate
from qualtran.bloqs.arithmetic.sorting import BitonicSort, Comparator
from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom
from qualtran.bloqs.arithmetic.trigonometric import ArcSin

from ._shims import CHalf, Lt, MultiCToffoli
