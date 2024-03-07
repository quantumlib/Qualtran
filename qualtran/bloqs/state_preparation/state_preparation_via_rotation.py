#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

r"""Controlled State preparation.

This algorithm prepares a state $|\psi\rangle$ in a register initially at $|0\rangle$ by using
rotations $R_y$ for encoding amplitudes and $R_z$ for encoding phases.

Assume one wants to prepare the amplitude of a one qubit state

$$
\sqrt{p_0} |0\rangle + \sqrt{p_1} |1\rangle.
$$

This can be achieved by a rotation $R_y(\theta)$ where $\theta = \cos^{-1}(\sqrt{p_0})$.
For encoding the amplitude of a n-qubit quantum state one could use a similar approach to this, but
chaining conditional probabilities: first rotate qubit 1 by $\theta = \cos^{-1}(\sqrt{p_0})$, then
the second qubit by $\theta_0 = \cos^{-1}(\sqrt{p_{00}/p_{0}})$, conditioned on the first one being
in $|0\rangle$ and $\theta_1 = \cos^{-1}(\sqrt{p_{10}/p_{1}})$ conditioned by the first being in
$|1\rangle$, and so on. Here $p_y$ means the probability that the first len(y) qubits of the
original state are in the state $y$. Refer to equation (8) of [1] for the details.

This general scheme is handled by StatePreparationViaRotations. This class also uses
RotationTree to get the angles of rotation needed (which are converted to the value to be loaded
to the ROM to achieve such a rotation). RotationTree is a tree data structure which holds the
accumulated probability of each substring, i.e., the root holds the probability of measuring the
first qubit at 0, the branch1 node the probability of measuring the second qubit at 0 if the first
was measured at 1 and so on. The $2^i$ rotations needed to prepare the ith qubit are performed by
PRGAViaPhaseGradient. This essentially is a rotation gate array, that is, given a list of
angles it performs the kth rotation when the selection register is on state $|k\rangle$. This
rotation is done in the Z axis, but for encoding amplitude a rotation around Ry is needed, thus the
need of a $R_x(\pm \pi/2)$ gate before and after encoding the amplitudes of each qubit.

In order to perform the rotations as efficiently as possible, the angles are loaded into a register
(rot\_reg) which is added into a phase gradient. Then phase kickback causes an overall offset of
$e^{i2\pi x/2^b}$, where $x$ is the angle value loaded and $b$ the size of the rot\_reg. Below is an
example for rot\_reg\_size=2.

First there is the rot\_reg register with the value to be rotated (3 in this case) and the phase
gradient

$$
|3\rangle(e^{2\pi i 0/4}|0\rangle + e^{2\pi i 1/4}|1\rangle +
          e^{2\pi i 2/4}|2\rangle + e^{2\pi i 3/4}|3\rangle).
$$

Then the rot\_reg $|3\rangle$ register is added to the phase gradient and store the result in the
phase gradient register

$$
|3\rangle(e^{2\pi i 0/4}|3\rangle + e^{2\pi i 1/4}|0\rangle +
          e^{2\pi i 2/4}|1\rangle + e^{2\pi i 3/4}|2\rangle),
$$

but this is equivalent to the original state with a phase offset of $e^{2\pi i 1/4}$.


References:
    [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
    (https://arxiv.org/abs/1812.00954).
        Low, Kliuchnikov, Schaeffer. 2018.

"""

from typing import Dict, List, Tuple

import attrs
import numpy as np
from numpy.typing import ArrayLike

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    GateWithRegisters,
    Signature,
    SoquetT,
    QUInt
)
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.basic_gates.rotation import Rx
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.arithmetic.addition import Add


@attrs.frozen
class StatePreparationViaRotations(GateWithRegisters):
    r"""Controlled state preparation without entangled residual using Ry and Rz rotations from [1].

    Given a quantum state of which the list of coefficients $c_i$ is known
    $$
        |\psi \rangle = \sum_{i=0}^{N-1}c_{i}|i\rangle
    $$
    this gate prepares $|\psi\rangle$ from $|0\rangle$ conditioned by a control qubit
    $$
        U((|0\rangle + |1\rangle)|0\rangle) = |0\rangle |0\rangle + |1\rangle |\psi\rangle.
    $$

    Args:
        phase_bitsize: size of the register that is used to store the rotation angles. Bigger values
            increase the accuracy of the results.
        state_coefficients: tuple of length 2^state_bitsizes that contains the complex coefficients of the state.
        control_bitsize: number of qubits of the control register. Set to zero for an uncontrolled gate.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    phase_bitsize: int
    state_coefficients: Tuple[complex, ...]
    control_bitsize: int = 0
    uncompute: bool = False

    def __attrs_post_init__(self):
        # a valid quantum state has a number of coefficients that is a power of two
        assert len(self.state_coefficients) == 2**self.state_bitsize
        # negative number of control bits is not allowed
        assert self.control_bitsize >= 0
        # the register to which the angle is written must be at least of size two
        assert self.phase_bitsize > 1
        # a valid quantum state must have norm one
        assert np.isclose(np.linalg.norm(self.state_coefficients), 1)

    @property
    def state_bitsize(self) -> int:
        return (len(self.state_coefficients) - 1).bit_length()

    @property
    def signature(self) -> Signature:
        return Signature.build(
            prepare_control=self.control_bitsize,
            target_state=self.state_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Parameters:
        * prepare_control: only if control_bitsize != 0
        * target_state: register where the state is written
        * phase_gradient: phase gradient state (will be left unaffected)
        """
        rotation_tree = RotationTree(self.state_coefficients, self.phase_bitsize, self.uncompute)
        ampl_rv, phase_rv = rotation_tree.get_rom_vals()
        if self.uncompute:
            soqs = self._prepare_phases(phase_rv, bb, **soqs)
            soqs = self._prepare_amplitudes(ampl_rv, bb, **soqs)
        else:
            soqs = self._prepare_amplitudes(ampl_rv, bb, **soqs)
            soqs = self._prepare_phases(phase_rv, bb, **soqs)
        return soqs

    def _prepare_amplitudes(
        self, rom_vals: List[List[int]], bb: BloqBuilder, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        r"""Parameters into soqs:
        * prepare_control: only if control_bitsize != 0
        * target_state: register where the state is written
        * phase_gradient: phase gradient state (will be left unaffected)
        """
        state_qubits = bb.split(soqs.pop("target_state"))
        for i in range(self.state_bitsize):
            # for the normal gate loop from qubit 0 to state_bitsizes-1, if it is the adjoint
            # then the process is run backwards with the opposite turn angles
            if self.uncompute:
                qi = self.state_bitsize - i - 1
            else:
                qi = i
            ctrl_rot_q = PRGAViaPhaseGradient(
                qi, self.phase_bitsize, tuple(rom_vals[qi]), self.control_bitsize + 1
            )
            state_qubits[qi] = bb.add(Rx(angle=np.pi / 2), q=state_qubits[qi])
            if qi:
                # first qubit does not have selection registers, only controls
                soqs["selection"] = bb.join(state_qubits[:qi])
            if self.control_bitsize > 1:
                soqs["control"] = bb.join(
                    np.array([*bb.split(soqs.pop("prepare_control")), state_qubits[qi]])
                )
            elif self.control_bitsize == 1:
                soqs["control"] = bb.join(np.array([soqs.pop("prepare_control"), state_qubits[qi]]))
            else:
                soqs["control"] = state_qubits[qi]
            soqs = bb.add_d(ctrl_rot_q, **soqs)
            separated = bb.split(soqs.pop("control"))
            if self.control_bitsize != 0:
                soqs["prepare_control"] = bb.join(separated[:-1])
            state_qubits[qi] = separated[-1]
            if qi:
                state_qubits[:qi] = bb.split(soqs.pop("selection"))
            state_qubits[qi] = bb.add(Rx(angle=-np.pi / 2), q=state_qubits[qi])

        soqs["target_state"] = bb.join(state_qubits)
        return soqs

    def _prepare_phases(
        self, rom_vals: List[int], bb: BloqBuilder, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        """Encodes the phase of each coefficient.

        Takes into account both the phase of the original coefficient and offsets caused by the
        amplitude preparation.

        It applies a rotation to the target_state register through phase kickback. By using
        target_register as the selection register for rotating an ancilla qubit that is then
        erased, leaving the desired phase in target_register.

        Args:
            rom_vals: list of ROM values that correspond to the angles of the phases to
                be encoded. The ith item contain an integer approximation of the value of the
                ith coefficient, that should be loaded via QROM.
            prepare_control: control qubit for the gate. Gate only takes effect when control is at |1>
            target_state: register that holds the amplitudes of the state to be encoded. It is
                used as the selection register for the ROM value to be loaded, thus for the
                target_state |00>, the angle that corresponds to the phase of |00> is applied.
            phase_gradient: phase gradient state used to apply the rotation, can be obtained from
                the PhaseGradientState class. It is left unaffected and can be reused.
        """
        rot_ancilla = bb.allocate(1)
        rot_ancilla = bb.add(XGate(), q=rot_ancilla)
        if self.control_bitsize > 1:
            soqs["control"] = bb.join(
                np.array([*bb.split(soqs.pop("prepare_control")), rot_ancilla])
            )
        elif self.control_bitsize == 1:
            soqs["control"] = bb.join(np.array([soqs.pop("prepare_control"), rot_ancilla]))
        else:
            soqs["control"] = rot_ancilla
        ctrl_rot = PRGAViaPhaseGradient(
            self.state_bitsize, self.phase_bitsize, tuple(rom_vals), self.control_bitsize + 1
        )
        # rename some registers to make them compatible with PRGAViaPhaseGradient
        soqs["selection"] = soqs.pop("target_state")
        soqs = bb.add_d(ctrl_rot, **soqs)
        soqs["target_state"] = soqs.pop("selection")
        separated = bb.split(soqs.pop("control"))
        if self.control_bitsize != 0:
            soqs["prepare_control"] = bb.join(separated[:-1])
        separated[-1] = bb.add(XGate(), q=separated[-1])
        bb.free(separated[-1])
        return soqs


@bloq_example
def _state_prep_via_rotation() -> StatePreparationViaRotations:
    state_coefs = (
        (-0.42677669529663675 - 0.1767766952966366j),
        (0.17677669529663664 - 0.4267766952966367j),
        (0.17677669529663675 - 0.1767766952966368j),
        (0.07322330470336305 - 0.07322330470336309j),
        (0.4267766952966366 - 0.17677669529663692j),
        (0.42677669529663664 + 0.17677669529663675j),
        (0.0732233047033631 + 0.17677669529663678j),
        (-0.07322330470336308 - 0.17677669529663678j),
    )
    state_prep_via_rotation = StatePreparationViaRotations(
        phase_bitsize=2, state_coefficients=state_coefs
    )
    return state_prep_via_rotation


_STATE_PREP_VIA_ROTATIONS_DOC = BloqDocSpec(
    bloq_cls=StatePreparationViaRotations,
    import_line='from qualtran.bloqs.state_preparation.state_preparation_via_rotation import StatePreparationViaRotations',
    examples=(_state_prep_via_rotation,),
)


@attrs.frozen
class PRGAViaPhaseGradient(Bloq):
    r"""Array of controlled rotations $Z^{\theta_i/2}$ for a list of angles $\theta$.

    It uses phase kickback and thus needs a phase gradient state in order to work. This
    state must be provided externally for efficiency, as it is unaffected and can thus be reused.
    Refer to [1], section on arbitrary quantum state preparation on page 3.

    Args:
        selection_bitsize: number of qubits used for encoding the selection register of the QROM, it
            must be equal to $\lceil \log_2(l_{rv}) \rceil$, where $l_{rv}$ is the number of angles
            provided.
        phase_bitsize: size of the register that is used to store the rotation angles. Bigger values
            increase the accuracy of the results.
        rom_values: the tuple of values to be loaded in the rom, which correspond to the angle of
            each rotation.
        control_bitsize: number of qubits of the control register. Set to zero for an uncontrolled gate.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    selection_bitsize: int
    phase_bitsize: int
    rom_values: Tuple[int, ...]
    control_bitsize: int

    @property
    def signature(self) -> Signature:
        return Signature.build(
            control=self.control_bitsize,
            selection=self.selection_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Parameters:
        * control
        * selection (not necessary if selection_bitsize == 0)
        * phase_gradient
        """
        qrom = QROM(
            [np.array(self.rom_values)],
            selection_bitsizes=(self.selection_bitsize,),
            target_bitsizes=(self.phase_bitsize,),
            num_controls=self.control_bitsize,
        )
        # allocate a register to store the rotation angle
        soqs["target0_"] = bb.allocate(self.phase_bitsize)
        phase_grad = soqs.pop("phase_gradient")
        # load angles in rot_reg (line 1 of eq (8) in [1])
        soqs = bb.add_d(qrom, **soqs)
        # phase kickback via phase_grad += rot_reg (line 2 of eq (8) in [1])
        soqs["target0_"], phase_grad = bb.add(
            # I needed to change this because AddIntoPhaseGrad does not declare bloq decomposition
            Add(QUInt(self.phase_bitsize)),
            a=soqs["target0_"],
            b=phase_grad,
        )
        # uncompute angle load in rot_reg to disentangle it from selection register
        # (line 1 of eq (8) in [1])
        soqs = bb.add_d(qrom, **soqs)
        soqs["phase_gradient"] = phase_grad
        bb.free(soqs.pop("target0_"))
        return soqs


class RotationTree:
    r"""Used by `StatePreparationViaRotations` to get the corresponding rotation angles.

    The rotation angles are used to encode the amplitude of a state using the method described in
    [1], section on arbitrary quantum state preparation, page 3.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    def __init__(self, state: ArrayLike, phase_bitsize: int, uncompute: bool = False):
        self.state_bitsize = (len(state) - 1).bit_length()
        self._calc_amplitude_angles_and_rv(state, phase_bitsize, uncompute)
        self._calc_phase_rom_values(state, phase_bitsize, uncompute)

    def get_rom_vals(self) -> Tuple[List[List[int]], List[int]]:
        return self.amplitude_rom_values, self.phase_rom_values

    def _calc_amplitude_angles_and_rv(
        self, state: ArrayLike, phase_bitsize: int, uncompute: bool
    ) -> None:
        r"""Gives a list of the ROM values to be loaded for preparing the amplitudes of a state.

        The ith element of the returned list is another list with the rom values to be loaded when
        preparing the amplitudes of the ith qubit for the given state.
        """
        slen = len(state)
        self.sum_total = np.zeros(2 * slen)
        for i in range(slen):
            self.sum_total[i + slen] = abs(state[i]) ** 2
        for i in range(slen - 1, 0, -1):
            self.sum_total[i] = self.sum_total[i << 1] + self.sum_total[(i << 1) | 1]
        self.amplitude_rom_values: List[List[int]] = []
        for i in range(self.state_bitsize):
            rom_vals_this_layer: List[int] = []
            for node in range(1 << i, 1 << (i + 1)):
                angle = self._angle_0(node)
                if uncompute:
                    angle = 2 * np.pi - angle
                rom_val = RotationTree._angle_to_rom_value(angle, phase_bitsize)
                rom_vals_this_layer.append(rom_val)
            self.amplitude_rom_values.append(rom_vals_this_layer)

    def _calc_phase_rom_values(self, state: ArrayLike, phase_bitsize: int, uncompute: bool) -> None:
        """Computes the rom value to be loaded to get the phase for each coefficient of the state.

        As we are using the equivalent to controlled Z to do the rotations instead of Rz, there
        is a phase offset for each coefficient that has to be corrected. This offset is half of the
        turn angle applied, and is added to the phase for each coefficient.
        """
        offsets = np.zeros(2**self.state_bitsize)
        for i in range(self.state_bitsize):
            rang = 2 ** (self.state_bitsize - i)
            for j in range(2**i):
                arv = self.amplitude_rom_values[i][j]
                if uncompute:
                    arv = -arv % 2**phase_bitsize
                offsets[j * rang : (j + 1) * rang] += (np.pi * arv / (2**phase_bitsize)) % np.pi
        angles = np.array([np.angle(c) for c in state])
        # flip angle if uncompute
        angles = [(1 - 2 * uncompute) * (a - o) for a, o in zip(angles, offsets)]
        self.phase_rom_values: List[int] = [
            RotationTree._angle_to_rom_value(a, phase_bitsize) for a in angles
        ]

    def _angle_0(self, idx: int) -> float:
        r"""Angle that corresponds to p_0."""
        return 2 * np.arccos(np.sqrt(self._p0(idx)))

    def _p0(self, idx: int) -> float:
        if self.sum_total[idx] == 0:
            return 0
        return self.sum_total[idx << 1] / self.sum_total[idx]

    @staticmethod
    def _angle_to_rom_value(angle: float, phase_bitsize: int) -> int:
        r"""Given an angle, returns the value to be loaded in ROM.

        Returns the value to be loaded to a QROM to encode the given angle with a certain value of
        phase_bitsize.
        """
        rom_value_decimal = 2**phase_bitsize * angle / (2 * np.pi)
        return round(rom_value_decimal) % (2**phase_bitsize)
