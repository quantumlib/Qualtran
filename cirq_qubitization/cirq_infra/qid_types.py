import cirq


class CleanQubit(cirq.Qid):
    def __init__(self, n: int):
        self.id = n

    def _comparison_key(self) -> int:
        return self.id

    @property
    def dimension(self) -> int:
        return 2

    def __str__(self) -> str:
        return f"_c{self.id}"

    def __repr__(self) -> str:
        return f"CleanQubit({self.id})"


class BorrowableQubit(cirq.Qid):
    def __init__(self, n: int):
        self.id = n

    def _comparison_key(self) -> int:
        return self.id

    @property
    def dimension(self) -> int:
        return 2

    def __str__(self) -> str:
        return f"_b{self.id}"

    def __repr__(self) -> str:
        return f"BorrowableQubit({self.id})"
