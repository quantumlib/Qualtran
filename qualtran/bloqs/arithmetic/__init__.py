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
    PlusEqualProduct,
    Product,
    ScaleIntByReal,
    Square,
    SquareRealNumber,
    SumOfSquares,
)
