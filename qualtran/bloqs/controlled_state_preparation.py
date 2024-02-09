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

from typing import Dict, Tuple

import attrs
import numpy as np
from numpy.typing import ArrayLike

from qualtran import Bloq, BloqBuilder, SelectionRegister, Signature, SoquetT
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import OneEffect, OneState, ZeroEffect, ZeroState
from qualtran.bloqs.basic_gates.rotation import Rx
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.select_and_prepare import PrepareOracle


@attrs.frozen
class ControlledStatePreparationUsingRotations(PrepareOracle):
    r""" Controlled state preparation without entangled residual using Ry and Rz rotations from [1].
    
    Given a quantum state of which the list of coefficients $c_i$ is known
    $$
        |\psi \rangle = \sum_{i=0}^{N-1}c_{i}|i\rangle
    $$
    this gate prepares $|\psi\rangle$ from $|0\rangle$ conditioned by a control qubit
    $$
        U((|0\rangle + |1\rangle)|0\rangle) = |0\rangle |0\rangle + |1\rangle |\psi\rangle.
    $$

    Args:
        n_qubits: number of qubits of the state.
        rot_reg_size: size of the register that is used to store the rotation angles. Bigger values
            increase the accuracy of the results.
        state: tuple of length 2^n_qubits that contains the complex coefficients of the state.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    n_qubits: int
    rot_reg_size: int
    state: Tuple
    uncompute: bool = False

    @property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister(
                "target_state", bitsize=self.n_qubits, iteration_length=self.n_qubits
            ),
        )

    @property
    def signature(self):
        return Signature.build(
            control=1, target_state=self.n_qubits, phase_gradient=self.rot_reg_size
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, *, control: SoquetT, target_state: SoquetT, phase_gradient: SoquetT
    ) -> Dict[str, SoquetT]:
        rom_vals = RotationTree.extract_ROM_values_from_state(self.state, self.rot_reg_size)
        # allocate the qubits for the rotation angle register
        rot_reg = bb.join(np.array([bb.add(ZeroState()) for _ in range(self.rot_reg_size)]))
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
        qs = bb.split(rot_reg)
        for q in qs:
            bb.add(ZeroEffect(), q=q)
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
            rom_vals = RotationTree.extract_ROM_values_from_state(
                self.state, self.rot_reg_size, uncompute=True
            )
        state_qubits = bb.split(target_state)
        for i in range(self.n_qubits):
            # for the normal gate loop from qubit 0 to n_qubits-1, if it is the adjoint
            # then the process is run backwards with the opposite turn angles
            if self.uncompute:
                qi = self.n_qubits - i - 1
            else:
                qi = i
            ctrl_rot_q = ControlledQROMRotateQubit(qi, self.rot_reg_size, tuple(rom_vals[qi]))
            state_qubits[qi] = bb.add(Rx(angle=np.pi / 2), q=state_qubits[qi])
            # first qubit does not have selection registers, only controls
            if qi == 0:
                control, state_qubits[qi], rot_reg, phase_gradient = bb.add(
                    ctrl_rot_q,
                    prepare_control=control,
                    qubit=state_qubits[qi],
                    rot_reg=rot_reg,
                    phase_gradient=phase_gradient,
                )
            else:
                sel = bb.join(state_qubits[:qi])
                control, sel, state_qubits[qi], rot_reg, phase_gradient = bb.add(
                    ctrl_rot_q,
                    prepare_control=control,
                    selection=sel,
                    qubit=state_qubits[qi],
                    rot_reg=rot_reg,
                    phase_gradient=phase_gradient,
                )
                state_qubits[:qi] = bb.split(sel)
            state_qubits[qi] = bb.add(Rx(angle=-np.pi / 2), q=state_qubits[qi])

        target_state = bb.join(state_qubits)
        return control, target_state, rot_reg, phase_gradient

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
            - amplitude_rom_vals: list of ROM values that correspond to the angles of the phases to
                be encoded. The ith item contains the value of the ith coefficient. To get the ROM
                value that corresponds to an angle use RotationTree.angle_2_ROM_value.
            - control: control qubit for the gate. Gate only takes effect when control is at |1>
            - target_state: register that holds the amplitudes of the state to be encoded. It is
                used as the selection register for the ROM value to be loaded, thus for the
                target_state |00>, the angle that corresponds to the phase of |00> is applied.
            - rot_reg: register where the rotation angles are written (the target of the ROM gate).
                Must be given in zero state, and is left in the zero state.
            - phase_gradient: phase gradient state used to apply the rotation, can be obtained from
                the PhaseGradientState class. It is left unaffected and can be reused.
        """
        rot_ancilla = bb.add(OneState())
        rom_vals = self._get_phase_ROM_values(amplitude_rom_vals)
        ctrl_rot = ControlledQROMRotateQubit(self.n_qubits, self.rot_reg_size, tuple(rom_vals))
        control, target_state, rot_ancilla, rot_reg, phase_gradient = bb.add(
            ctrl_rot,
            prepare_control=control,
            selection=target_state,
            qubit=rot_ancilla,
            rot_reg=rot_reg,
            phase_gradient=phase_gradient,
        )
        bb.add(OneEffect(), q=rot_ancilla)
        return control, target_state, rot_reg, phase_gradient

    def _get_phase_ROM_values(self, amplitude_rom_vals):
        """As we are using the equivalent to controlled Z to do the rotations instead of Rz, there
        is a phase offset for each coefficient that has to be corrected. This offset is half of the
        turn angle applied, and is added to the phase for each coefficient.
        """
        offset_angles = [0] * (2**self.n_qubits)
        for i in range(self.n_qubits):
            for j in range(2**i):
                item_range = 2 ** (self.n_qubits - i)
                # if the rom has value 0 the formula gives 180, when it should be 0
                if amplitude_rom_vals[i][j] == 0:
                    offset = 0
                else:
                    offset = np.pi * amplitude_rom_vals[i][j] / (2**self.rot_reg_size)
                for k in range(item_range * j, item_range * (j + 1)):
                    offset_angles[k] += offset
        # if the matrix is the adjoint, the angles have to be undone, thus just load -theta
        if self.uncompute:
            angles = [offset - np.angle(c) for c, offset in zip(self.state, offset_angles)]
        else:
            angles = [np.angle(c) - offset for c, offset in zip(self.state, offset_angles)]
        rom_values = [RotationTree.angle_2_ROM_value(a, self.rot_reg_size) for a in angles]
        return rom_values


@attrs.frozen
class ControlledQROMRotateQubit(Bloq):
    r""" Array of controlled rotations $Z^{\theta_i/2}$ for a list of angles $\theta$.
    
    It uses phase kickback and thus needs a phase gradient state in order to work. This
    state must be provided externally for efficiency, as it is unaffected and can thus be reused.
    Refer to [1], section on arbitrary quantum state preparation on page 3.

    Args:
        n_selections: number of qubits used for encoding the selection register of the QROM, it
            must be equal to $\lceil \log_2(l_{rv}) \rceil$, where $l_{rv}$ is the number of angles
            provided.
        rot_reg_size: size of the register that is used to store the rotation angles. Bigger values
            increase the accuracy of the results.
        rom_values: the tuple of values to be loaded in the rom, which correspond to the angle of
            each rotation. In order to get the rom value that corresponds to a given angle use
            RotationTree.angle_2_ROM_value(angle, rot_reg_size).

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    n_selections: int
    rot_reg_size: int
    rom_values: Tuple

    @property
    def signature(self):
        return Signature.build(
            prepare_control=1,
            selection=self.n_selections,
            qubit=1,
            rot_reg=self.rot_reg_size,
            phase_gradient=self.rot_reg_size,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Parameters:
        * prepare_control
        * selection (not necessary if n_selections == 0)
        * qubit
        * rot_reg
        * phase_gradient
        """
        qrom = QROM(
            [np.array(self.rom_values)],
            selection_bitsizes=(self.n_selections,),
            target_bitsizes=(self.rot_reg_size,),
            num_controls=2,
        )
        # both prepare_control and qubit will control the QROM so that this acts as a control Z
        qrom_control = bb.join(np.array([soqs["prepare_control"], soqs["qubit"]]))
        qrom_control, soqs = self._apply_QROM(qrom, bb, qrom_control, soqs)
        soqs["rot_reg"], soqs["phase_gradient"] = bb.add(
            Add(bitsize=self.rot_reg_size), a=soqs["rot_reg"], b=soqs["phase_gradient"]
        )
        qrom_control, soqs = self._apply_QROM(qrom, bb, qrom_control, soqs)
        separated = bb.split(qrom_control)
        soqs["prepare_control"] = separated[0]
        soqs["qubit"] = separated[1]

        return soqs

    def _apply_QROM(
        self, qrom: QROM, bb: BloqBuilder, qrom_control: SoquetT, soqs: Dict[str, SoquetT]
    ):
        if self.n_selections != 0:
            qrom_control, soqs["selection"], soqs["rot_reg"] = bb.add(
                qrom, control=qrom_control, selection=soqs["selection"], target0_=soqs["rot_reg"]
            )
        else:
            qrom_control, soqs["rot_reg"] = bb.add(
                qrom, control=qrom_control, target0_=soqs["rot_reg"]
            )
        return qrom_control, soqs


class RotationTree:
    r""" Used by ControlledStatePreparationUsingRotations to get the corresponding rotation
    angles.
    
    The rotation angles are used to encode the amplitude of a state using the method described in
    [1], section on arbitrary quantum state preparation, page 3.

    The only methods to be used externally are extract_ROM_values_from_state, angle_2_ROM_value,
    rotation_tree_from_state and get_angle_0.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    @staticmethod
    def extract_ROM_values_from_state(state: ArrayLike, rot_reg_size: int, uncompute: bool = False):
        r"""Gives list in which the ith element is a list of the rom values to be loaded when
        preparing the amplitudes of the ith qubit for the given state.
        """
        rotation_tree = RotationTree.rotation_tree_from_state(state)
        next_layer = [rotation_tree]
        rom_vals = []
        while len(next_layer) != 0:
            this_layer = next_layer
            next_layer = []
            rom_vals_this_layer = []
            for tree in this_layer:
                angle = tree.get_angle_0()
                if uncompute:
                    angle = 2 * np.pi - angle
                rom_val = RotationTree.angle_2_ROM_value(angle, rot_reg_size)
                rom_vals_this_layer.append(rom_val)
                if tree.branch0 is not None:
                    next_layer.append(tree.branch0)
                if tree.branch1 is not None:
                    next_layer.append(tree.branch1)
            rom_vals.append(rom_vals_this_layer)
        return rom_vals

    @staticmethod
    def rotation_tree_from_state(state):
        r"""Given a list of coefficients, returns a tree-like object that contains the angles for
        the rotations when preparing the state.
        """
        if len(state) == 2:
            return RotationTree(abs(state[0]) ** 2, abs(state[1]) ** 2, None, None)
        dn_l = RotationTree.rotation_tree_from_state(state[: len(state) // 2])
        dn_r = RotationTree.rotation_tree_from_state(state[len(state) // 2 :])
        return RotationTree(dn_l.sum_total, dn_r.sum_total, dn_l, dn_r)

    def get_angle_0(self):
        r"""Get the angle that corresponds to p_0."""
        return 2 * np.arccos(np.sqrt(self._get_p0()))

    @staticmethod
    def angle_2_ROM_value(angle, rot_reg_size):
        r"""Returns the value to be loaded to a QROM to encode the given angle with a certain value
        of rot_reg_size.
        """
        rom_value_decimal = 2**rot_reg_size * angle / (2 * np.pi)
        return round(rom_value_decimal) % (2**rot_reg_size)

    # do not call, use rotation_tree_from_state
    def __init__(self, sum0, sum1, branch0, branch1):
        self.sum0 = sum0
        self.sum1 = sum1
        self.sum_total = sum0 + sum1
        self.branch0 = branch0
        self.branch1 = branch1

    def _get_p0(self):
        if self.sum_total == 0:
            return 0
        return self.sum0 / self.sum_total
