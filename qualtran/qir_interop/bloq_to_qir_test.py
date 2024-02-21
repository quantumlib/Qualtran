import os

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
from pathlib import Path

import attrs
from typing import Dict
from qualtran import Bloq, BloqBuilder, SoquetT, Signature, Register
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.qft.qft_text_book import QFTTextBook
from qualtran.qir_interop.bloq_to_qir import bloq_to_qir


def compare_reference_ir(ir: str, name: str) -> None:
    file = os.path.join(os.path.dirname(__file__), f"resources/{name}.ll")
    expected = Path(file).read_text()
    assert ir == expected


@attrs.frozen
class Swap(Bloq):
    n: int

    @property
    def signature(self):
        return Signature.build(x=self.n, y=self.n)

    def build_composite_bloq(
        self, bb: BloqBuilder, *, x: SoquetT, y: SoquetT
    ) -> Dict[str, SoquetT]:
        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(self.n):
            xs[i], ys[i] = bb.add(CNOT(), ctrl=xs[i], target=ys[i])
        return {'x': bb.join(xs), 'y': bb.join(ys)}


@attrs.frozen
class ExampleBaseBloq(Bloq):
    @property
    def signature(self):
        return Signature.build(x=1, y=1)


@attrs.frozen
class ExampleHighLevelBloq(Bloq):
    n: int

    @property
    def signature(self):
        return Signature.build(x=self.n, y=self.n)

    def build_composite_bloq(
        self, bb: BloqBuilder, *, x: SoquetT, y: SoquetT
    ) -> Dict[str, SoquetT]:
        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(self.n):
            xs[i], ys[i] = bb.add(ExampleBaseBloq(), x=xs[i], y=ys[i])
        return {'x': bb.join(xs), 'y': bb.join(ys)}


@attrs.frozen
class ExampleNonTrivialShapeBloq(Bloq):
    n: int

    @property
    def signature(self):
        return Signature(
            [Register('x', bitsize=1, shape=(self.n,)), Register('y', bitsize=1, shape=(self.n,))]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, *, x: SoquetT, y: SoquetT
    ) -> Dict[str, SoquetT]:
        for i in range(self.n):
            x[i], y[i] = bb.add(ExampleBaseBloq(), x=x[i], y=y[i])
        return {'x': x, 'y': y}


def test_textbook_qft_to_qir():
    qft_bloq = QFTTextBook(10)
    qft_mod = bloq_to_qir(qft_bloq)
    compare_reference_ir(str(qft_mod), "textbook_qft")


def test_multi_qubit_swap_to_qir():
    swap_bloq = Swap(5)
    swap_mod = bloq_to_qir(swap_bloq)
    compare_reference_ir(str(swap_mod), "multi_qubit_swap")


def test_example_high_level_bloq_to_qir():
    high_level_bloq = ExampleHighLevelBloq(5)
    high_level_bloq_mod = bloq_to_qir(high_level_bloq)
    compare_reference_ir(str(high_level_bloq_mod), "example_high_level_bloq")
