import cirq_ft

from qualtran.protos import annotations_pb2


def t_complexity_to_proto(t_complexity: cirq_ft.TComplexity) -> annotations_pb2.TComplexity:
    return annotations_pb2.TComplexity(
        clifford=t_complexity.clifford, rotations=t_complexity.rotations, t=t_complexity.t
    )
