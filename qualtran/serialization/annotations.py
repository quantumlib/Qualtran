import cirq_ft

from qualtran.protos import annotations_pb2


def t_complexity_to_proto(t_complexity: cirq_ft.TComplexity) -> annotations_pb2.TComplexity:
    return annotations_pb2.TComplexity(
        clifford=t_complexity.clifford, rotations=t_complexity.rotations, t=t_complexity.t
    )


def t_complexity_from_proto(t_complexity: annotations_pb2.TComplexity) -> cirq_ft.TComplexity:
    return cirq_ft.TComplexity(
        clifford=t_complexity.clifford, t=t_complexity.t, rotations=t_complexity.rotations
    )
