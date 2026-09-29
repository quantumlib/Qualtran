from __future__ import annotations

import random

import kickmix as km
import pytest

from src.kickmix.py.circuit.subs.fuzz_test_util import assert_fuzz_testing_acts_like


def test_init_lookup_diagram():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    output = builder.create_quantum_register(4, name="output")
    builder.init_lookup(
        table=[0b0001, 0b0011, 0b0111, 0b1111],
        address=address,
        target=output,
    )
    assert builder.finish_circuit().text_diagram().strip() == """
q0: -address[0]-------X---@-X------------Z**b0-X---@-X------------Z**b0--------------
                          |              |         |              |
q1: -address[1]-X---@-X---|--------------|---------|--------------|------------Z**b0-
                    |     |              |         |              |
q2:  output[0]  |0>-|-----|-X---X--------|---------|-X---X--------|------------------
                    |     | |   |        |         | |   |        |
q3:  output[1]  |0>-|-----|-|---X--------|---------|-X---X--------|------------------
                    |     | |   |        |         | |   |        |
q4:  output[2]  |0>-|-----|-|---|--------|---------|-X---X--------|------------------
                    |     | |   |        |         | |   |        |
q5:  output[3]  |0>-|-----|-|---|--------|---------|-|---X--------|------------------
                    |     | |   |        |         | |   |        |
q6:             |0>-X-----@-|-@-|--------@-----X---@-|-@-|--------@-----HMR=b0
                          | | | |                  | | | |
q7:                   |0>-X-@-X-@-HMR=b0       |0>-X-@-X-@-HMR=b0
    """.strip()


@pytest.mark.parametrize("n_address", range(4))
@pytest.mark.parametrize("n_output", range(4))
def test_fuzz_init_lookup(n_address: int, n_output: int):
    table = [
        random.randrange(1 << n_output)
        for _ in range(1 << n_address)
    ]

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    output = builder.create_quantum_register(n_output, name="output")
    builder.init_lookup(
        table=table,
        address=address,
        target=output,
    )
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            output ^= table[address]
        """,
        shots=64,
        context={'table': table},
        input_sampler=lambda: {
            'output': 0,
            'address': random.randrange(1 << n_address),
        })


@pytest.mark.parametrize("n_address", range(4))
@pytest.mark.parametrize("n_output", range(4))
def test_fuzz_del_lookup(n_address: int, n_output: int):
    table = [
        random.randrange(1 << n_output)
        for _ in range(1 << n_address)
    ]

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    output = builder.create_quantum_register(n_output, name="output")
    builder.del_lookup(
        table=table,
        address=address,
        target=output,
    )
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            output ^= table[address]
        """,
        shots=64,
        context={'table': table},
        input_sampler=lambda: {
            'address': (a := random.randrange(1 << n_address)),
            'output': table[a],
        })
