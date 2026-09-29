import kickmix as km
import pytest


def test_builder_bit_store_simple():
    builder = km.CircuitBuilder()
    bs = [km.b(k) for k in range(4)]

    builder.bit_store(bs[0], True)
    builder.bit_store(bs[1], False)
    builder.bit_store(bs[2], True, control=bs[3])
    assert builder.finish_circuit() == km.Circuit("""
        BIT_STORE1 b0
        BIT_STORE0 b1
        BIT_STORE1 b2 if b3
    """)


def test_builder_bit_store_broadcasts_over_the_control():
    builder = km.CircuitBuilder()
    flag = builder.create_classical_register(1, name="flag")
    word = builder.create_classical_register(4, name="word")

    # The canonical OR reduction: clear, then set under every bit in turn.
    builder.bit_store(flag, False)
    builder.bit_store(flag[0], True, control=word)
    circuit = builder.finish_circuit()

    sim = km.Simulator(batch_size=16)
    sim.use_same_registers_as(circuit)
    for value in range(16):
        sim.write_within_shot("word", value, value)
    sim.do(circuit)
    for value in range(16):
        assert sim.read_within_shot("flag", value, out=int) == (1 if value else 0)


def test_builder_bit_store_rejects_quantum_controls():
    builder = km.CircuitBuilder()
    with pytest.raises(ValueError, match="must be classical"):
        builder.bit_store(km.b(0), True, control=km.q(0))
    with pytest.raises(ValueError, match="must be classical"):
        builder.bit_store([km.b(0), km.b(1)], True, control=[km.b(2), km.q(0)])
    # The rejected calls must not have left anything behind.
    assert builder.finish_circuit() == km.Circuit()


@pytest.mark.parametrize("value", [False, True])
def test_builder_bit_store_broadcast_modes(value: bool):
    targets = [km.b(k) for k in range(4)]
    for controls in [
        True,
        False,
        [True, False, True, False],
        [km.b(10), km.b(11), km.b(12), km.b(13)],
        [km.b(10), True, False, km.b(13)],
    ]:
        builder = km.CircuitBuilder()
        builder.bit_store(targets, value, control=controls)

        ref = km.CircuitBuilder()
        cs = [controls] * 4 if isinstance(controls, bool) else controls
        for t, c in zip(targets, cs):
            ref.bit_store(t, value, control=c)
        assert builder.finish_circuit() == ref.finish_circuit()

