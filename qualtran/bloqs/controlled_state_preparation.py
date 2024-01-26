from typing import Dict, Tuple
import attrs

import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, SoquetT, Side, Register
from qualtran.bloqs.basic_gates import ZeroState, PlusState, ZeroEffect
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.basic_gates.rotation import Rx, Rz
from qualtran.bloqs.basic_gates.hadamard import Hadamard


@attrs.frozen
class ControlledRotStatePreparation(Bloq):
    n_qubits: int

    @property
    def signature(self):
        return Signature.build(control=1, state=self.n_qubits)

    def build_composite_bloq(
        self, bb: BloqBuilder, *, control: SoquetT, state: SoquetT
    ) -> Dict[str, SoquetT]:
        state_qb = bb.split(state)
        state = bb.join(state_qb)
        return {"control": control, "state": state}


@attrs.frozen
class ControlledPrepareQubit(Bloq):
    n_selections: int
    rot_reg_size: int
    rom_vals: Tuple

    @property
    def signature(self):
        return Signature.build(
            prepare_control=1,
            selection=self.n_selections,
            qubit=1,
            rot_reg=self.rot_reg_size,
            phase_gradient=self.rot_reg_size
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        *,
        prepare_control: SoquetT,
        selection: SoquetT,
        qubit: SoquetT,
        rot_reg: SoquetT,
        phase_gradient: SoquetT
    ) -> Dict[str, SoquetT]:
        qrom = QROM(
            [np.array(self.rom_vals)],
            selection_bitsizes=(self.n_selections,),
            target_bitsizes=(self.rot_reg_size,),
            num_controls=1,
        )

        prepare_control, selection, rot_reg = bb.add(
            qrom, control=prepare_control, selection=selection, target0_=rot_reg
        )

        qubit = bb.add(Rx(angle=np.pi / 2), q=qubit)
        qubit = bb.add(Rx(angle=-np.pi / 2), q=qubit)

        prepare_control, selection, rot_reg = bb.add(
            qrom, control=prepare_control, selection=selection, target0_=rot_reg
        )
        return {
            "prepare_control": prepare_control,
            "selection": selection,
            "qubit": qubit,
            "rot_reg": rot_reg,
            "phase_gradient": phase_gradient
        }


class RotationTree:
    def extractRomValuesFromState(state, rot_reg_size):
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
        return np.acos(np.sqrt(self.__getP0()))

    def angle2RomValue(angle, rot_reg_size):
        rom_val = 2**rot_reg_size - 1 - 2**rot_reg_size * angle / (2 * np.pi)
        return round(rom_val)

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
