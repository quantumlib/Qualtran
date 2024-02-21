import abc

from attrs import frozen


class QDType:
    @property
    @abc.abstractmethod
    def num_qubits(self):
        """Number of qubits required to represent a single instance of this data type."""


@frozen
class _Register:

    dtype: QDType

    @property
    def bitsize(self) -> int:
        return self.dtype.bitsize


@frozen
class Register(_Register):

    bitsize: int


if __name__ == "__main__":
    reg = Register(bitsize_or_dtype=7)
