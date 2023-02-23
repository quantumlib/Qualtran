import contextlib
from typing import List

from cirq_qubitization.qid_types import BorrowableQubit, CleanQubit


def qalloc_clean(n: int) -> List[CleanQubit]:
    return [CleanQubit() for _ in range(n)]


def qalloc_borrow(n: int) -> List[BorrowableQubit]:
    return [BorrowableQubit() for _ in range(n)]


def qalloc_reset() -> None:
    CleanQubit.reset_count()
    BorrowableQubit.reset_count()
