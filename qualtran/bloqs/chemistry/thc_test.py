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

import numpy as np
from cirq_ft.algos.arithmetic_gates import LessThanEqualGate, LessThanGate

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder, Register
from qualtran.bloqs.chemistry.thc import (
    PrepareTHC,
    split_join_cirq_arithmetic_gates,
    UniformSuperpositionTHC,
)
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.testing import execute_notebook


def _make_uniform_superposition():
    from qualtran.bloqs.chemistry.thc import UniformSuperpositionTHC

    num_mu = 10
    num_spin_orb = 4
    return UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)


def _make_prepare():
    from qualtran.bloqs.chemistry.thc import PrepareTHC

    num_mu = 10
    num_spin_orb = 4
    return PrepareTHC(num_mu=num_mu, num_spin_orb=num_spin_orb, keep_bitsize=8)


def test_split_join_arithmetic_gates():
    bb = BloqBuilder()
    bitsize = 9
    val = bb.add_register(Register("val", bitsize=bitsize))
    res = bb.add_register(Register("res", bitsize=1))
    val, res = split_join_cirq_arithmetic_gates(
        bb, CirqGateAsBloq(LessThanGate(bitsize, 7)), val=val, res=res
    )
    cbloq = bb.finalize(val=val, res=res)
    assert cbloq.t_complexity() == CirqGateAsBloq(LessThanGate(bitsize, 7)).t_complexity()
    bb = BloqBuilder()
    x = bb.add_register(Register("x", bitsize=bitsize))
    y = bb.add_register(Register("y", bitsize=bitsize))
    res = bb.add_register(Register("res", bitsize=1))
    x, y, res = split_join_cirq_arithmetic_gates(
        bb, CirqGateAsBloq(LessThanEqualGate(bitsize, bitsize)), x=x, y=y, res=res
    )
    cbloq = bb.finalize(x=x, y=y, res=res)
    assert (
        cbloq.t_complexity() == CirqGateAsBloq(LessThanEqualGate(bitsize, bitsize)).t_complexity()
    )


def test_uniform_superposition():
    num_mu = 10
    num_spin_orb = 4
    usup = UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    qlt_testing.assert_valid_bloq_decomposition(usup)


def test_prepare():
    num_mu = 10
    num_spin_orb = 4
    prep = PrepareTHC(num_mu=num_mu, num_spin_orb=num_spin_orb, keep_bitsize=10)
    qlt_testing.assert_valid_bloq_decomposition(prep)


def test_notebook():
    execute_notebook('thc')
