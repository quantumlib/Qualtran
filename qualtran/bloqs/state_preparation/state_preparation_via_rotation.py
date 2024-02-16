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

This general scheme is handled by ControlledStatePreparationUsingRotations. This class also uses
RotationTree to get the angles of rotation needed (which are converted to the value to be loaded
to the ROM to achieve such a rotation). RotationTree is a tree data structure which holds the
accumulated probability of each substring, i.e., the root holds the probability of measuring the
first qubit at 0, the branch1 node the probability of measuring the second qubit at 0 if the first
was measured at 1 and so on. The $2^i$ rotations needed to prepare the ith qubit are performed by
ControlledQROMRotateQubit. This essentially is a rotation gate array, that is, given a list of
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

from typing import Dict, Tuple, List

import attrs
import numpy as np
from numpy.typing import ArrayLike

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BoundedQUInt,
    Register,
    Signature,
    SoquetT,
    GateWithRegisters
)
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.basic_gates.rotation import Rx
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad


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
        state_bitsize: number of qubits of the state.
        phase_bitsize: size of the register that is used to store the rotation angles. Bigger values
            increase the accuracy of the results.
        state_coefficients: tuple of length 2^state_bitsizes that contains the complex coefficients of the state.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    state_bitsize: int
    phase_bitsize: int
    state_coefficients: Tuple[complex, ...]
    uncompute: bool = False

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register(
                "target_state",
                BoundedQUInt(bitsize=self.state_bitsize, iteration_length=len(self.state_coefficients)),
            ),
        )

    @property
    def signature(self)  -> Signature:
        return Signature.build(
            control=1, target_state=self.state_bitsize, phase_gradient=self.phase_bitsize
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, *, control: SoquetT, target_state: SoquetT, phase_gradient: SoquetT
    ) -> Dict[str, SoquetT]:
        rom_vals = RotationTree(self.state_coefficients).get_rom_values(self.phase_bitsize)
        # allocate the qubits for the rotation angle register
        rot_reg = bb.allocate(self.phase_bitsize)
        if self.uncompute:
            control, target_state, rot_reg, phase_gradient = self._prepare_phases(
                rom_vals, bb, control, target_state, rot_reg, phase_gradient
            )
            control, target_state, rot_reg, phase_gradient = self._prepare_amplitudes(
                rom_vals, bb, control, target_state, rot_reg, phase_gradient
            )
        else:
            control, target_state, rot_reg, phase_gradient = self._prepare_amplitudes(
                rom_vals, bb, control, target_state, rot_reg, phase_gradient
            )
            control, target_state, rot_reg, phase_gradient = self._prepare_phases(
                rom_vals, bb, control, target_state, rot_reg, phase_gradient
            )
        # deallocate rotation register's qubits
        bb.free(rot_reg)
        return {"control": control, "target_state": target_state, "phase_gradient": phase_gradient}

    def _prepare_amplitudes(
        self,
        rom_vals: ArrayLike,
        bb: BloqBuilder,
        control: SoquetT,
        target_state: SoquetT,
        rot_reg: SoquetT,
        phase_gradient: SoquetT,
    ):
        # if it is the adjoint gate, load the modular negative values to undo the rotations that
        # loaded the amplitudes
        if self.uncompute:
            rom_vals = RotationTree(self.state_coefficients).get_rom_values(
                self.phase_bitsize, uncompute=True
            )
        state_qubits = bb.split(target_state)
        soqs = {"prepare_control": control, "rot_reg": rot_reg, "phase_gradient": phase_gradient}
        for i in range(self.state_bitsize):
            # for the normal gate loop from qubit 0 to state_bitsizes-1, if it is the adjoint
            # then the process is run backwards with the opposite turn angles
            if self.uncompute:
                qi = self.state_bitsize - i - 1
            else:
                qi = i
            ctrl_rot_q = ControlledQROMRotateQubit(qi, self.phase_bitsize, tuple(rom_vals[qi]))
            state_qubits[qi] = bb.add(Rx(angle=np.pi / 2), q=state_qubits[qi])
            if qi:
                # first qubit does not have selection registers, only controls
                soqs["selection"] = bb.join(state_qubits[:qi])
            soqs = bb.add_d(ctrl_rot_q, **soqs, qubit=state_qubits[qi])
            if qi:
                state_qubits[:qi] = bb.split(soqs.pop("selection"))
            state_qubits[qi] = bb.add(Rx(angle=-np.pi / 2), q=soqs.pop("qubit"))

        target_state = bb.join(state_qubits)
        return soqs["prepare_control"], target_state, soqs["rot_reg"], soqs["phase_gradient"]

    def _prepare_phases(
        self,
        amplitude_rom_vals: ArrayLike,
        bb: BloqBuilder,
        control: SoquetT,
        target_state: SoquetT,
        rot_reg: SoquetT,
        phase_gradient: SoquetT,
    ):
        """
        Encodes the phase of each coefficient, taking into account both the phase of the original
        coefficient and offsets caused by the amplitude preparation.

        It applies a rotation to the target_state register through phase kickback. By using
        target_register as the selection register for rotating an ancilla qubit that is then
        erased, leaving the desired phase in target_register.

        Args:
            amplitude_rom_vals: list of ROM values that correspond to the angles of the phases to
                be encoded. The ith item contains the value of the ith coefficient. To get the ROM
                value that corresponds to an angle use RotationTree.angle_to_rom_value.
            control: control qubit for the gate. Gate only takes effect when control is at |1>
            target_state: register that holds the amplitudes of the state to be encoded. It is
                used as the selection register for the ROM value to be loaded, thus for the
                target_state |00>, the angle that corresponds to the phase of |00> is applied.
            rot_reg: register where the rotation angles are written (the target of the ROM gate).
                Must be given in zero state, and is left in the zero state.
            phase_gradient: phase gradient state used to apply the rotation, can be obtained from
                the PhaseGradientState class. It is left unaffected and can be reused.
        """
        rot_ancilla = bb.allocate(1)
        rot_ancilla = bb.add(XGate(), q=rot_ancilla)
        rom_vals = self._get_phase_rom_values(amplitude_rom_vals)
        ctrl_rot = ControlledQROMRotateQubit(
            self.state_bitsize, self.phase_bitsize, tuple(rom_vals)
        )
        control, target_state, rot_ancilla, rot_reg, phase_gradient = bb.add(
            ctrl_rot,
            prepare_control=control,
            selection=target_state,
            qubit=rot_ancilla,
            rot_reg=rot_reg,
            phase_gradient=phase_gradient,
        )
        rot_ancilla = bb.add(XGate(), q=rot_ancilla)
        bb.free(rot_ancilla)
        return control, target_state, rot_reg, phase_gradient

    def _get_phase_rom_values(self, amplitude_rom_vals) -> List[int]:
        """As we are using the equivalent to controlled Z to do the rotations instead of Rz, there
        is a phase offset for each coefficient that has to be corrected. This offset is half of the
        turn angle applied, and is added to the phase for each coefficient.
        """
        offset_angles = [0] * (2**self.state_bitsize)
        for i in range(self.state_bitsize):
            for j in range(2**i):
                item_range = 2 ** (self.state_bitsize - i)
                # if the rom has value 0 the formula gives 180, when it should be 0
                if amplitude_rom_vals[i][j] == 0:
                    offset = 0
                else:
                    offset = np.pi * amplitude_rom_vals[i][j] / (2**self.phase_bitsize)
                for k in range(item_range * j, item_range * (j + 1)):
                    offset_angles[k] += offset
        # if the matrix is the adjoint, the angles have to be undone, thus just load -theta
        if self.uncompute:
            angles = [offset - np.angle(c) for c, offset in zip(self.state_coefficients, offset_angles)]
        else:
            angles = [np.angle(c) - offset for c, offset in zip(self.state_coefficients, offset_angles)]
        rom_values = [RotationTree.angle_to_rom_value(a, self.phase_bitsize) for a in angles]
        return rom_values


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
        state_bitsize=2, phase_bitsize=4, state_coefficients=state_coefs
    )
    return state_prep_via_rotation


_CONTROLLED_STATE_PREP_DOC = BloqDocSpec(
    bloq_cls=StatePreparationViaRotations,
    import_line='from qualtran.bloqs.state_preparation.state_preparation_via_rotation import StatePreparationViaRotations',
    examples=(_state_prep_via_rotation,),
)


@attrs.frozen
class ControlledQROMRotateQubit(Bloq):
    r"""Array of controlled rotations $Z^{\theta_i/2}$ for a list of angles $\theta$.

    It uses phase kickback and thus needs a phase gradient state in order to work. This
    state must be provided externally for efficiency, as it is unaffected and can thus be reused.
    Refer to [1], section on arbitrary quantum state preparation on page 3.

    Args:
        selection_bitsizes: number of qubits used for encoding the selection register of the QROM, it
            must be equal to $\lceil \log_2(l_{rv}) \rceil$, where $l_{rv}$ is the number of angles
            provided.
        rot_reg_bitsizes: size of the register that is used to store the rotation angles. Bigger values
            increase the accuracy of the results.
        rom_values: the tuple of values to be loaded in the rom, which correspond to the angle of
            each rotation. In order to get the rom value that corresponds to a given angle use
            RotationTree.angle_to_rom_value(angle, rot_reg_bitsizes).

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    selection_bitsizes: int
    rot_reg_bitsizes: int
    rom_values: Tuple[int, ...]

    @property
    def signature(self) -> Signature:
        return Signature.build(
            prepare_control=1,
            selection=self.selection_bitsizes,
            qubit=1,
            rot_reg=self.rot_reg_bitsizes,
            phase_gradient=self.rot_reg_bitsizes,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Parameters:
        * prepare_control
        * selection (not necessary if selection_bitsizes == 0)
        * qubit
        * rot_reg
        * phase_gradient
        """
        qrom = QROM(
            [np.array(self.rom_values)],
            selection_bitsizes=(self.selection_bitsizes,),
            target_bitsizes=(self.rot_reg_bitsizes,),
            num_controls=2,
        )
        # both prepare_control and qubit will control the QROM so that this acts as a control Z
        qrom_control = bb.join(np.array([soqs.pop("prepare_control"), soqs.pop("qubit")]))
        phase_grad = soqs.pop("phase_gradient")
        # load angles in rot_reg (line 1 of eq (8) in [1])
        soqs = bb.add_d(qrom, control=qrom_control, target0_=soqs.pop("rot_reg"), **soqs)
        # phase kickback via phase_grad += rot_reg (line 2 of eq (8) in [1])
        soqs["target0_"], phase_grad = bb.add(
            AddIntoPhaseGrad(self.rot_reg_bitsizes, self.rot_reg_bitsizes),
            x=soqs["target0_"],
            phase_grad=phase_grad,
        )
        # uncompute angle load in rot_reg to disentangle it from selection register
        # (line 1 of eq (8) in [1])
        soqs = bb.add_d(qrom, **soqs)
        separated = bb.split(soqs.pop('control'))
        return {
            "prepare_control": separated[0],
            "qubit": separated[1],
            "phase_gradient": phase_grad,
            "rot_reg": soqs.pop('target0_'),
        } | soqs


class RotationTree:
    r"""Used by `StatePreparationUsingRotations` to get the corresponding rotation angles.

    The rotation angles are used to encode the amplitude of a state using the method described in
    [1], section on arbitrary quantum state preparation, page 3.

    The only methods to be used externally are extract_rom_values_from_state, angle_to_rom_value and
    build_rotation_tree_from_state.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    def __init__(self, state: ArrayLike):
        n = len(state)
        self.n_qubits = (n - 1).bit_length()
        self.sum_total = np.zeros(2 * n)
        for i in range(n):
            self.sum_total[i + n] = abs(state[i]) ** 2
        for i in range(n - 1, 0, -1):
            self.sum_total[i] = self.sum_total[i << 1] + self.sum_total[(i << 1) | 1]

    def get_rom_values(self, rot_reg_bitsizes: int, uncompute: bool = False) -> List[int]:
        r"""Gives a list of the ROM values to be loaded for preparing the amplitudes of a state.

        The ith element of the returned list is another list with the rom values to be loaded when
        preparing the amplitudes of the ith qubit for the given state.
        """
        rom_vals = []
        for i in range(self.n_qubits):
            rom_vals_this_layer = []
            for node in range(1 << i, 1 << (i + 1)):
                angle = self.angle_0(node)
                if uncompute:
                    angle = 2 * np.pi - angle
                rom_val = RotationTree.angle_to_rom_value(angle, rot_reg_bitsizes)
                rom_vals_this_layer.append(rom_val)
            rom_vals.append(rom_vals_this_layer)
        return rom_vals

    def angle_0(self, idx: int) -> float:
        r"""Angle that corresponds to p_0."""
        return 2 * np.arccos(np.sqrt(self._p0(idx)))

    def _p0(self, idx: int) -> float:
        if self.sum_total[idx] == 0:
            return 0
        return self.sum_total[idx << 1] / self.sum_total[idx]

    @staticmethod
    def angle_to_rom_value(angle: float, rot_reg_bitsize: int) -> int:
        r"""Given an angle, returns the value to be loaded in ROM.

        Returns the value to be loaded to a QROM to encode the given angle with a certain value of
        rot_reg_bitsizes.
        """
        rom_value_decimal = 2**rot_reg_bitsize * angle / (2 * np.pi)
        return round(rom_value_decimal) % (2**rot_reg_bitsize)
