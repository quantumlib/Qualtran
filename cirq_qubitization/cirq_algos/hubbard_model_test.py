import pytest

import cirq_qubitization as cq
from cirq_qubitization.cirq_algos.hubbard_model import PrepareHubbard, SelectHubbard
from cirq_qubitization.jupyter_tools import execute_notebook


def test_notebook():
    execute_notebook('hubbard_model')


@pytest.mark.parametrize('dim', [*range(2, 10)])
def test_select_t_complexity(dim):
    select = SelectHubbard(x_dim=dim, y_dim=dim, control_val=1)
    cost = cq.t_complexity(select)
    N = 2 * dim * dim
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t == 10 * N + 14 * logN - 8
    assert cost.rotations == 0


@pytest.mark.parametrize('dim', [*range(2, 10)])
def test_prepare_t_complexity(dim):
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=2, mu=8)
    cost = cq.t_complexity(prepare)
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t <= 32 * logN
    # TODO(#233): The rotation count should reduce to a constant once cost for Controlled-H
    # gates is recognized as $2$ T-gates instead of $2$ rotations.
    assert cost.rotations <= 2 * logN + 9
