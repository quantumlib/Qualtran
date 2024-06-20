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
import numpy as np

from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.linalg.polynomial.basic import evaluate_polynomial_of_unitary_matrix


def test_evaluate_polynomial_of_unitary_matrix():
    U = np.array([[0, -1j], [1j, 0]])

    V = evaluate_polynomial_of_unitary_matrix([0, 0, 1], U)
    np.testing.assert_allclose(V, np.eye(2))

    V = evaluate_polynomial_of_unitary_matrix([1, 0, 1], U, offset=-1)
    np.testing.assert_allclose(V, 2 * U)

    U = np.array(MatrixGate.random(2, random_state=42).matrix)
    V = evaluate_polynomial_of_unitary_matrix([1, 2, 3], U)
    np.testing.assert_allclose(V, np.eye(4) + 2 * U + 3 * U @ U)
