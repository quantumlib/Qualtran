import cirq_ft

from qualtran.serialization import annotations


def test_t_complexity_to_proto():
    t_complexity = cirq_ft.TComplexity(t=10, clifford=100, rotations=1000)
    proto = annotations.t_complexity_to_proto(t_complexity)
    assert (proto.t, proto.clifford, proto.rotations) == (10, 100, 1000)
    assert annotations.t_complexity_from_proto(proto) == t_complexity
