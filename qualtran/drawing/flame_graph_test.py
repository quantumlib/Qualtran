#  Copyright 2024 Google LLC
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
from qualtran.bloqs.basic_gates.swap import CSwap
from qualtran.bloqs.mcmt import MultiAnd
from qualtran.bloqs.qft.qft_text_book import QFTTextBook
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.drawing.flame_graph import get_flame_graph_data


def test_get_flame_graph_data_cswap():
    bloq = CSwap(10)
    data = get_flame_graph_data(bloq)
    assert data == [
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
        'CSwap[10](T:70);TwoBitCSwap(T:7)\t7',
    ]


def test_get_flame_graph_data_multi_and():
    bloq = MultiAnd([0, 1] * 6)
    data = get_flame_graph_data(bloq)
    assert data == [
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
        'MultiAnd(n=12)(T:44);And(T:4)\t4',
    ]


def test_get_flame_graph_data_qft_textbook():
    bloq = QFTTextBook(5)
    data = get_flame_graph_data(bloq)
    assert sorted(data) == sorted(
        [
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[4][0.5][True][0](T:200);CZ**0.125(T:50)\t'
            '50',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[4][0.5][True][0](T:200);CZ**0.25(T:50)\t'
            '50',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[4][0.5][True][0](T:200);CZ**0.5(T:50)\t'
            '50',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[4][0.5][True][0](T:200);CZ**0.0625(T:50)\t'
            '50',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[3][0.5][True][0](T:150);CZ**0.25(T:50)\t'
            '50',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[3][0.5][True][0](T:150);CZ**0.5(T:50)\t'
            '50',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[3][0.5][True][0](T:150);CZ**0.125(T:50)\t'
            '50',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[1][0.5][True][0](T:48);CZ**0.5(T:48)\t'
            '48',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[2][0.5][True][0](T:98);CZ**0.25(T:49)\t'
            '49',
            'QFTTextBook[5][True](T:496);PhaseGradientUnitary[2][0.5][True][0](T:98);CZ**0.5(T:49)\t'
            '49',
        ]
    )


def test_get_flame_graph_data_prep_uniform():
    bloq = PrepareUniformSuperposition(12)
    data = sorted(get_flame_graph_data(bloq))
    should_be = sorted(
        [
            'PrepareUniformSuperposition[12][0](T:124);LessThanConstant[2][3](T:8);And(T:4)\t4',
            'PrepareUniformSuperposition[12][0](T:124);LessThanConstant[2][3](T:8);And(T:4)\t4',
            'PrepareUniformSuperposition[12][0](T:124);LessThanConstant[2][3](T:8);And(T:4)\t4',
            'PrepareUniformSuperposition[12][0](T:124);LessThanConstant[2][3](T:8);And(T:4)\t4',
            'PrepareUniformSuperposition[12][0](T:124);Rz(1.2309594173407745)(T:52)\t52',
            'PrepareUniformSuperposition[12][0](T:124);Rz(1.2309594173407745)(T:52)\t52',
            'PrepareUniformSuperposition[12][0](T:124);And(T:4)\t4',
        ]
    )
    assert len(data) == len(should_be)
    for line1, line2 in zip(data, should_be):
        if 'Rz' in line1:
            # Unstable str formatting of Rz float angle on CI vs local
            continue

        assert line1 == line2
