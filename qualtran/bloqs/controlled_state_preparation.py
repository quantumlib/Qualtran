from typing import Dict, Tuple
import attrs

import numpy as np
from numpy.typing import ArrayLike

from qualtran import Bloq, BloqBuilder, Signature, SoquetT, Side, Register
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import ZeroState, ZeroEffect
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.basic_gates.rotation import Rx


@attrs.frozen
class ControlledRotStatePreparation(Bloq):
    n_qubits: int
    rot_reg_size: int
    state: Tuple

    @property
    def signature(self):
        return Signature.build(
            control=1,
            target_state=self.n_qubits,
            phase_gradient=self.rot_reg_size,
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        *,
        control: SoquetT,
        target_state: SoquetT,
        phase_gradient: SoquetT,
    ) -> Dict[str, SoquetT]:
        rom_vals = RotationTree.extractRomValuesFromState(self.state, self.rot_reg_size)
        rot_reg = bb.join(
            np.array([bb.add(ZeroState()) for _ in range(self.rot_reg_size)])
        )
        state_qubits = bb.split(target_state)
        for i in range(self.n_qubits):
            ctrl_rot_q = ControlledQROMRotateQubit(i, self.rot_reg_size, tuple(rom_vals[i]))
            state_qubits[i] = bb.add(Rx(angle=np.pi / 2), q=state_qubits[i])
            if i == 0:
                control, state_qubits[i], rot_reg, phase_gradient = bb.add(
                    ctrl_rot_q,
                    prepare_control=control,
                    qubit=state_qubits[i],
                    rot_reg=rot_reg,
                    phase_gradient=phase_gradient
                )
            else:
                sel = bb.join(state_qubits[:i])
                control, sel, state_qubits[i], rot_reg, phase_gradient = bb.add(
                    ctrl_rot_q,
                    prepare_control=control,
                    selection=sel,
                    qubit=state_qubits[i],
                    rot_reg=rot_reg,
                    phase_gradient=phase_gradient
                )
                state_qubits[:i] = bb.split(sel)
            state_qubits[i] = bb.add(Rx(angle=-np.pi / 2), q=state_qubits[i])

        target_state = bb.join(state_qubits)
        qs = bb.split(rot_reg)
        for q in qs:
            bb.add(ZeroEffect(), q=q)
        return {
            "control": control,
            "target_state": target_state,
            "phase_gradient": phase_gradient,
        }


@attrs.frozen
class ControlledQROMRotateQubit(Bloq):
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

    def build_composite_bloq(
        self, bb: BloqBuilder, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
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
        qrom_control = bb.join(np.array([soqs["prepare_control"], soqs["qubit"]]))
        if self.n_selections != 0:
            qrom_control, soqs["selection"], soqs["rot_reg"] = bb.add(
                qrom,
                control=qrom_control,
                selection=soqs["selection"],
                target0_=soqs["rot_reg"],
            )
        else:
            qrom_control, soqs["rot_reg"] = bb.add(
                qrom, control=qrom_control, target0_=soqs["rot_reg"]
            )

        soqs["rot_reg"], soqs["phase_gradient"] = bb.add(
            Add(bitsize=self.rot_reg_size), a=soqs["rot_reg"], b=soqs["phase_gradient"]
        )

        if self.n_selections != 0:
            qrom_control, soqs["selection"], soqs["rot_reg"] = bb.add(
                qrom,
                control=qrom_control,
                selection=soqs["selection"],
                target0_=soqs["rot_reg"],
            )
        else:
            qrom_control, soqs["rot_reg"] = bb.add(
                qrom, control=qrom_control, target0_=soqs["rot_reg"]
            )
        separated = bb.split(qrom_control)
        soqs["prepare_control"] = separated[0]
        soqs["qubit"] = separated[1]

        return soqs


class RotationTree:
    def extractRomValuesFromState(state: ArrayLike, rot_reg_size: int):
        rotation_tree = RotationTree.rotationTreeFromState(state)
        next_layer = [rotation_tree]
        rom_vals = []
        while len(next_layer) != 0:
            this_layer = next_layer
            next_layer = []
            rom_vals_this_layer = []
            for tree in this_layer:
                angle = tree.getAngle0()
                rom_val = RotationTree.angle2RomValue(angle, rot_reg_size)
                rom_vals_this_layer.append(rom_val)
                if tree.branch0 is not None:
                    next_layer.append(tree.branch0)
                if tree.branch1 is not None:
                    next_layer.append(tree.branch1)
            rom_vals.append(rom_vals_this_layer)
        return rom_vals

    def rotationTreeFromState(state):
        if len(state) == 2:
            return RotationTree(abs(state[0]) ** 2, abs(state[1]) ** 2, None, None)
        dn_l = RotationTree.rotationTreeFromState(state[: len(state) // 2])
        dn_r = RotationTree.rotationTreeFromState(state[len(state) // 2 :])
        return RotationTree(dn_l.sum_total, dn_r.sum_total, dn_l, dn_r)

    def getAngle0(self):
        return 2*np.arccos(np.sqrt(self.__getP0()))

    def angle2RomValue(angle, rot_reg_size):
        rom_value_decimal = 2**rot_reg_size * (1 - angle / (2 * np.pi))
        return round(rom_value_decimal) % (2**rot_reg_size)

    # do not call, use rotationTreeFromState
    def __init__(self, sum0, sum1, branch0, branch1):
        self.sum0 = sum0
        self.sum1 = sum1
        self.sum_total = sum0 + sum1
        self.branch0 = branch0
        self.branch1 = branch1

    def __getP0(self):
        if self.sum_total == 0:
            return 0
        return self.sum0 / self.sum_total
