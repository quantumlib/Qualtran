from typing import Union, Tuple, Iterable, Sequence, Optional

import cirq
import numpy as np


def _default_measurement_key(qubits: Iterable[cirq.Qid]) -> str:
    return ','.join(str(q) for q in qubits)


class SimpleMeasurementGate(cirq.Gate):
    """A gate that measures qubits in the computational basis.

    The measurement gate contains a key that is used to identify results
    of measurements.
    """

    def __init__(self, key: Union[str, 'cirq.MeasurementKey'] = '') -> None:
        """Inits MeasurementGate.

        Args:
            key: The string key of the measurement.
        """
        self._mkey = key if isinstance(key, cirq.MeasurementKey) else cirq.MeasurementKey(name=key)

    @property
    def key(self) -> str:
        return str(self.mkey)

    @property
    def mkey(self) -> 'cirq.MeasurementKey':
        return self._mkey

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,)

    def _has_unitary_(self) -> bool:
        return False

    def _is_measurement_(self) -> bool:
        return True

    def _measurement_key_name_(self) -> str:
        return self.key

    def _measurement_key_obj_(self) -> 'cirq.MeasurementKey':
        return self.mkey

    def _kraus_(self):
        size = np.prod(self._qid_shape_, dtype=np.int64)

        def delta(i):
            result = np.zeros((size, size))
            result[i][i] = 1
            return result

        return tuple(delta(i) for i in range(size))

    def _has_kraus_(self):
        return True

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        symbols = ['M']

        # Mention the measurement key.
        label_map = args.label_map or {}
        if not args.known_qubits or self.key != _default_measurement_key(args.known_qubits):
            if self.key not in label_map:
                symbols[0] += f"('{self.key}')"
        if self.key in label_map:
            symbols += '@'

        return cirq.CircuitDiagramInfo(symbols)

    def __repr__(self):
        return f'SimpleMeasurementGate(key={self.mkey})'

    def __eq__(self, other):
        if not isinstance(other, SimpleMeasurementGate):
            return NotImplemented
        return self.key == other.key

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        return True

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']) -> bool:
        from cirq.sim import SimulationState

        if not isinstance(sim_state, SimulationState):
            return NotImplemented
        assert len(qubits) == 1
        sim_state.measure(qubits, self.key, invert_mask=(False,), confusion_map={})
        return True
