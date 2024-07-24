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
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary, CCZ2TFactory, LogicalErrorModel, QECScheme


def test_ccz_2t_factory():
    factory = CCZ2TFactory()
    worse_factory = CCZ2TFactory(distillation_l1_d=7, distillation_l2_d=15)

    alg = AlgorithmSummary(
        n_logical_gates=GateCounts(t=10**8, toffoli=10**8), n_algo_qubits=100
    )
    lem = LogicalErrorModel(qec_scheme=QECScheme.make_gidney_fowler(), physical_error=1e-3)

    err1 = factory.factory_error(n_logical_gates=alg.n_logical_gates, logical_error_model=lem)
    err2 = worse_factory.factory_error(n_logical_gates=alg.n_logical_gates, logical_error_model=lem)
    assert err2 > err1

    cyc1 = factory.n_cycles(n_logical_gates=alg.n_logical_gates, logical_error_model=lem)
    cyc2 = worse_factory.n_cycles(n_logical_gates=alg.n_logical_gates, logical_error_model=lem)
    assert cyc2 < cyc1

    foot1 = factory.n_physical_qubits()
    foot2 = worse_factory.n_physical_qubits()
    assert foot2 < foot1
