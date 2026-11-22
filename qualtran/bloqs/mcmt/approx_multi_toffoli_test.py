import random

import pytest

import qualtran.testing as qlt_testing
from qualtran.bloqs.mcmt.approx_multi_toffoli import (
    _approx_multi_toffoli,
    _multi_and_log_depth,
    _parity_mask,
    ApproxMultiToffoli,
)


@pytest.mark.parametrize(
    "bloq_ex", [_multi_and_log_depth, _parity_mask, _approx_multi_toffoli], ids=lambda b: b.name
)
def test_bloq_examples(bloq_autotester, bloq_ex):
    bloq_autotester(bloq_ex)


def test_approx_multi_toffoli_classical_randomized_notebook_example():
    rng = random.Random(52)
    n = 17
    k = 4
    m = n - 1
    all_ones = (1 << m) - 1

    def random_sample_string(width: int) -> tuple[int, ...]:
        return tuple(rng.randint(0, 1) for _ in range(width))

    def int_to_bits(x: int, width: int) -> tuple[int, ...]:
        return tuple((x >> j) & 1 for j in range(width))

    x_int = rng.randint(0, all_ones)
    sample_strings = tuple(random_sample_string(m) for _ in range(k))
    bloq = ApproxMultiToffoli(n=n, k=k, sample_strings=sample_strings)

    _, _, target = bloq.call_classically(ctrl=int_to_bits(x_int, m))  # type: ignore

    assert target == int(x_int == all_ones)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('approx_multi_toffoli')
