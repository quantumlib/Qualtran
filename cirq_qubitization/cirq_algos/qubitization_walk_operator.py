from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import cirq

from cirq_qubitization import cirq_infra, generic_select
from cirq_qubitization.cirq_algos import reflection_using_prepare, state_preparation


@cirq.value_equality()
class QubitizationWalkOperator(cirq_infra.GateWithRegisters):
    r"""Constructs a Szegedy Quantum Walk operator using LCU oracles SELECT and PREPARE.

    Constructs a Szegedy quantum walk operator $W = R_{L} . SELECT$, which is a product of
    two reflections $R_{L} = (2|L><L| - I)$ and $SELECT=\sum_{l}|l><l|H_{l}$.

    The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional irreducible
    vector spaces. For an arbitrary eigenstate $|k>$ of $H$ with eigenvalue $E_k$, $|\ell>|k>$ and
    an orthogonal state $\phi_{k}$ span the irreducible two-dimensional space that $|\ell>|k>$ is
    in under the action of $W$. In this space, $W$ implements a Pauli-Y rotation by an angle of
    $-2arccos(E_{k} / \lambda)$ s.t. $W = e^{i arccos(E_k / \lambda) Y}$.

    Thus, the walk operator $W$ encodes the spectrum of $H$ as a function of eigenphases of $W$
    s.t. $spectrum(H) = \lambda cos(arg(spectrum(W)))$ where $arg(e^{i\phi}) = \phi$.

    Args:
        select: The SELECT lcu gate implementing $SELECT=\sum_{l}|l><l|H_{l}$.
        prepare: Then PREPARE lcu gate implementing
            $PREPARE|00...00> = \sum_{l=0}^{L - 1}\sqrt{\frac{w_{l}}{\lambda}} |l> = |\ell>$
        control_val: If 0/1, a controlled version of the walk operator is constructed. Defaults to
            None, in which case the resulting walk operator is not controlled.
        power: Constructs $W^{power}$ by repeatedly decomposing into `power` copies of $W$.
            Defaults to 1.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    def __init__(
        self,
        select: generic_select.GenericSelect,
        prepare: state_preparation.StatePreparationAliasSampling,
        *,
        control_val: Optional[int] = None,
        power: int = 1,
    ):
        self._select = select
        self._reflect = reflection_using_prepare.ReflectionUsingPrepare(
            prepare, control_val=control_val
        )
        self._control_val = control_val
        assert self._select.control_registers == self._reflect.control_registers
        self.power = power

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return self._select.control_registers

    @cached_property
    def selection_registers(self) -> cirq_infra.Registers:
        return self._reflect.target_registers

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return self._select.target_registers

    @cached_property
    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        select_reg = {reg.name: qubit_regs[reg.name] for reg in self._select.registers}
        reflect_reg = {reg.name: qubit_regs[reg.name] for reg in self._reflect.registers}
        select_op = self._select.on_registers(**select_reg)
        reflect_op = self._reflect.on_registers(**reflect_reg)
        for _ in range(self.power):
            yield select_op
            yield reflect_op

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if self._control_val else '@(0)'] * self.control_registers.bitsize
        wire_symbols += ['W'] * (self.registers.bitsize - self.control_registers.bitsize)
        wire_symbols[-1] = f'W^{self.power}' if self.power != 1 else 'W'
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def controlled(
        self,
        num_controls: int = None,
        control_values: Sequence[Union[int, Collection[int]]] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'QubitizationWalkOperator':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if len(control_values) == 1 and self._control_val is None:
            return QubitizationWalkOperator(
                self._select.controlled(control_values=control_values),
                self._reflect.prepare_gate,
                control_val=control_values[-1],
            )
        raise NotImplementedError(f'Cannot create a controlled version of {self}')

    def with_power(self, new_power: int) -> 'QubitizationWalkOperator':
        return QubitizationWalkOperator(
            self._select, self._reflect.prepare_gate, control_val=self._control_val, power=new_power
        )

    def _value_equality_values_(self):
        return self._select, self._reflect, self._control_val, self.power

    def __pow__(self, power: int):
        return self.with_power(self.power * power)
