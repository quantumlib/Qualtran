//! Decompiler: translates a [`CompiledModule`] into a human-readable text-based assembly.
//!
//! Intended for debugging purposes only.

use std::collections::HashMap;
use std::fmt::Write;

use crate::fastsim::compiler::{
    CastOp, CompiledModule, CompiledSubroutine, ExternGate, Instruction, OutputMapping,
    RegisterDirection, SlotIdx, SubId, SubroutineBody,
};

/// Decompiles a [`CompiledModule`] into a text-based assembly string.
pub fn decompile(module: &CompiledModule) -> Result<String, String> {
    let mut decompiler = Decompiler {
        module,
        out: String::new(),
    };
    decompiler.decompile()?;
    Ok(decompiler.out)
}

struct Decompiler<'a> {
    module: &'a CompiledModule,
    out: String,
}

impl<'a> Decompiler<'a> {
    fn write_line(&mut self, left: &str, right: &str) -> Result<(), String> {
        writeln!(self.out, "{:<60}# {}", left, right).map_err(|e| e.to_string())
    }

    fn decompile(&mut self) -> Result<(), String> {
        self.write_line(
            &format!("CompiledModule {}", self.module.subroutines.len()),
            &format!("{} subroutines", self.module.subroutines.len()),
        )?;
        writeln!(self.out).map_err(|e| e.to_string())?;

        let mut entrypoints: Vec<(&str, SubId)> = self
            .module
            .intern_to_sub
            .iter()
            .map(|(&id, &sub_id)| (self.module.intern_table.resolve(id), sub_id))
            .collect();
        entrypoints.sort_by_key(|&(name, _)| name);
        for (name, sub_id) in entrypoints {
            self.write_line(&format!("Entrypoint {}", sub_id), name)?;
        }

        for (sub_id, sub) in self.module.subroutines.iter().enumerate() {
            writeln!(self.out).map_err(|e| e.to_string())?;
            let sub_name = self.module.intern_table.resolve(sub.bloq_key);
            self.write_line(&format!("Subroutine {}", sub_id), sub_name)?;

            let slot_map = self.build_slot_map(sub, sub_name)?;

            for reg in &sub.signature {
                let name = self.module.intern_table.resolve(reg.name);
                let dtype = self.module.intern_table.resolve(reg.dtype_name);
                let dir = match reg.direction {
                    RegisterDirection::Thru => "Thru",
                    RegisterDirection::RightOnly => "RightOnly",
                    RegisterDirection::LeftOnly => "LeftOnly",
                    RegisterDirection::Cast => "Cast",
                };
                let slot = sub.slots.intern_to_slot.get(&reg.name).ok_or_else(|| {
                    format!(
                        "Register '{}' not found in SlotInfo for subroutine '{}'",
                        name, sub_name
                    )
                })?;
                self.write_line(
                    &format!("  reg {} {} {}", slot, dir, reg.n_bits),
                    &format!("{}: {} ({} bits)", name, dtype, reg.n_bits),
                )?;
            }

            match &sub.body {
                SubroutineBody::Impl(instructions) => {
                    self.write_line("  body Impl", "")?;
                    for instr in instructions {
                        self.decompile_instruction(instr, &slot_map, sub_name)?;
                    }
                }
                SubroutineBody::Extern(extern_gate) => {
                    self.write_line("  body Extern", "")?;
                    self.decompile_extern_gate(extern_gate, &slot_map, sub_name)?;
                }
                SubroutineBody::Cast(cast_op) => {
                    self.write_line("  body Cast", "")?;
                    self.decompile_cast_op(cast_op, &slot_map, sub_name)?;
                }
            }
        }

        Ok(())
    }

    fn build_slot_map(
        &self,
        sub: &'a CompiledSubroutine,
        sub_name: &str,
    ) -> Result<HashMap<SlotIdx, &'a str>, String> {
        let mut map = HashMap::with_capacity(sub.slots.intern_to_slot.len());
        for (&intern_id, &slot) in &sub.slots.intern_to_slot {
            let name = self.module.intern_table.resolve(intern_id);
            map.insert(slot, name);
        }
        for s in 0..sub.slots.n_slots {
            if !map.contains_key(&s) {
                return Err(format!(
                    "Slot index {} missing in SlotInfo for subroutine '{}'",
                    s, sub_name
                ));
            }
        }
        Ok(map)
    }

    fn get_slot_var(
        &self,
        slot: SlotIdx,
        slot_map: &HashMap<SlotIdx, &'a str>,
        sub_name: &str,
    ) -> Result<&'a str, String> {
        slot_map
            .get(&slot)
            .copied()
            .ok_or_else(|| format!("Slot index {} not found in subroutine '{}'", slot, sub_name))
    }

    fn format_slot_ref(
        &self,
        slot: SlotIdx,
        slot_map: &HashMap<SlotIdx, &'a str>,
        sub_name: &str,
    ) -> Result<(String, String), String> {
        let var_name = self.get_slot_var(slot, slot_map, sub_name)?;
        Ok((format!("{}", slot), format!("{}", var_name)))
    }

    fn format_slot_refs(
        &self,
        slots: &[SlotIdx],
        slot_map: &HashMap<SlotIdx, &'a str>,
        sub_name: &str,
    ) -> Result<(String, String), String> {
        let mut lefts = Vec::with_capacity(slots.len());
        let mut rights = Vec::with_capacity(slots.len());
        for &s in slots {
            let (l, r) = self.format_slot_ref(s, slot_map, sub_name)?;
            lefts.push(l);
            rights.push(r);
        }
        Ok((lefts.join(" "), rights.join(", ")))
    }

    fn decompile_instruction(
        &mut self,
        instr: &Instruction,
        slot_map: &HashMap<SlotIdx, &'a str>,
        sub_name: &str,
    ) -> Result<(), String> {
        match instr {
            Instruction::Call {
                callee,
                arg_mappings,
                output_mapping,
            } => {
                if *callee >= self.module.subroutines.len() {
                    return Err(format!("Subroutine ID {} out of bounds", callee));
                }
                let callee_sub = &self.module.subroutines[*callee];
                let callee_name = self.module.intern_table.resolve(callee_sub.bloq_key);
                let callee_slot_map = self.build_slot_map(callee_sub, callee_name)?;

                self.write_line(&format!("    Call {}", callee), callee_name)?;

                for arg in arg_mappings {
                    let callee_var =
                        self.get_slot_var(arg.callee_slot, &callee_slot_map, callee_name)?;
                    let (left_bits, right_bits) =
                        self.format_slot_refs(&arg.bits, slot_map, sub_name)?;
                    self.write_line(
                        &format!("    arg {} {}", arg.callee_slot, left_bits),
                        &format!("{}<-{}", callee_var, right_bits),
                    )?;
                }

                match output_mapping {
                    OutputMapping::Direct(pairs) => {
                        for pair in pairs {
                            let caller_var = self.get_slot_var(pair.caller, slot_map, sub_name)?;
                            let callee_var =
                                self.get_slot_var(pair.callee, &callee_slot_map, callee_name)?;
                            self.write_line(
                                &format!("    out {} {}", pair.caller, pair.callee),
                                &format!("{}<-{}", caller_var, callee_var),
                            )?;
                        }
                    }
                    OutputMapping::Concat {
                        callee_slots,
                        caller_slot,
                        caller_elements: _,
                    } => {
                        let caller_var = self.get_slot_var(*caller_slot, slot_map, sub_name)?;
                        let mut callee_lefts = Vec::with_capacity(callee_slots.len());
                        let mut callee_rights = Vec::with_capacity(callee_slots.len());
                        for &c_slot in callee_slots {
                            let c_var = self.get_slot_var(c_slot, &callee_slot_map, callee_name)?;
                            callee_lefts.push(format!("{}", c_slot));
                            callee_rights.push(format!("{}", c_var));
                        }
                        self.write_line(
                            &format!("    out {} concat {}", caller_slot, callee_lefts.join(" ")),
                            &format!("{}<-concat({})", caller_var, callee_rights.join(", ")),
                        )?;
                    }
                }
            }
            Instruction::Return { ret_mappings } => {
                self.write_line("    Return", "")?;
                for rem in ret_mappings {
                    let ret_var = self.get_slot_var(rem.callee_slot, slot_map, sub_name)?;
                    let (left_bits, right_bits) =
                        self.format_slot_refs(&rem.bits, slot_map, sub_name)?;
                    self.write_line(
                        &format!("    ret {} {}", rem.callee_slot, left_bits),
                        &format!("{}<-{}", ret_var, right_bits),
                    )?;
                }
            }
            Instruction::InlineX { q, lvalue_slot } => {
                let (q_l, q_r) = self.format_slot_ref(*q, slot_map, sub_name)?;
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineX {} {}", q_l, lvalue_slot),
                    &format!("q<-{}, q->{}", q_r, lv_var),
                )?;
            }
            Instruction::InlineCNOT {
                ctrl,
                target,
                ctrl_lvalue,
                target_lvalue,
            } => {
                let (c_l, c_r) = self.format_slot_ref(*ctrl, slot_map, sub_name)?;
                let (t_l, t_r) = self.format_slot_ref(*target, slot_map, sub_name)?;
                let clv_var = self.get_slot_var(*ctrl_lvalue, slot_map, sub_name)?;
                let tlv_var = self.get_slot_var(*target_lvalue, slot_map, sub_name)?;
                self.write_line(
                    &format!(
                        "    InlineCNOT {} {} {} {}",
                        c_l, t_l, ctrl_lvalue, target_lvalue
                    ),
                    &format!(
                        "ctrl<-{}, target<-{}, ctrl->{}, target->{}",
                        c_r, t_r, clv_var, tlv_var
                    ),
                )?;
            }
            Instruction::InlineToffoli {
                ctrl0,
                ctrl1,
                target,
                ctrl_lvalue,
                ctrl0_lvalue,
                ctrl1_lvalue,
                target_lvalue,
            } => {
                let (c0_l, c0_r) = self.format_slot_ref(*ctrl0, slot_map, sub_name)?;
                let (c1_l, c1_r) = self.format_slot_ref(*ctrl1, slot_map, sub_name)?;
                let (t_l, t_r) = self.format_slot_ref(*target, slot_map, sub_name)?;
                let clv_var = self.get_slot_var(*ctrl_lvalue, slot_map, sub_name)?;
                let tlv_var = self.get_slot_var(*target_lvalue, slot_map, sub_name)?;
                let elem_str = match (ctrl0_lvalue, ctrl1_lvalue) {
                    (Some(c0), Some(c1)) => format!(", elem[{},{}]", c0, c1),
                    _ => String::new(),
                };
                self.write_line(
                    &format!(
                        "    InlineToffoli {} {} {} {} {}",
                        c0_l, c1_l, t_l, ctrl_lvalue, target_lvalue
                    ),
                    &format!(
                        "ctrl0<-{}, ctrl1<-{}, target<-{}, ctrl->{}{}, target->{}",
                        c0_r, c1_r, t_r, clv_var, elem_str, tlv_var
                    ),
                )?;
            }
            Instruction::InlineAnd {
                ctrl0,
                ctrl1,
                target_lvalue,
                ctrl_lvalue,
                ctrl0_lvalue,
                ctrl1_lvalue,
                cv,
            } => {
                let (c0_l, c0_r) = self.format_slot_ref(*ctrl0, slot_map, sub_name)?;
                let (c1_l, c1_r) = self.format_slot_ref(*ctrl1, slot_map, sub_name)?;
                let clv_var = self.get_slot_var(*ctrl_lvalue, slot_map, sub_name)?;
                let tlv_var = self.get_slot_var(*target_lvalue, slot_map, sub_name)?;
                let elem_str = match (ctrl0_lvalue, ctrl1_lvalue) {
                    (Some(c0), Some(c1)) => format!(", elem[{},{}]", c0, c1),
                    _ => String::new(),
                };
                self.write_line(
                    &format!(
                        "    InlineAnd {} {} {} {} cv=[{},{}]",
                        c0_l, c1_l, ctrl_lvalue, target_lvalue, cv[0] as u8, cv[1] as u8
                    ),
                    &format!(
                        "ctrl0<-{}, ctrl1<-{}, ctrl->{}{}, target->{}",
                        c0_r, c1_r, clv_var, elem_str, tlv_var
                    ),
                )?;
            }
            Instruction::InlineAndDag {
                ctrl0,
                ctrl1,
                target,
                ctrl_lvalue,
                ctrl0_lvalue,
                ctrl1_lvalue,
                cv,
            } => {
                let (c0_l, c0_r) = self.format_slot_ref(*ctrl0, slot_map, sub_name)?;
                let (c1_l, c1_r) = self.format_slot_ref(*ctrl1, slot_map, sub_name)?;
                let (t_l, t_r) = self.format_slot_ref(*target, slot_map, sub_name)?;
                let clv_var = self.get_slot_var(*ctrl_lvalue, slot_map, sub_name)?;
                let elem_str = match (ctrl0_lvalue, ctrl1_lvalue) {
                    (Some(c0), Some(c1)) => format!(", elem[{},{}]", c0, c1),
                    _ => String::new(),
                };
                self.write_line(
                    &format!(
                        "    InlineAndDag {} {} {} {} cv=[{},{}]",
                        c0_l, c1_l, t_l, ctrl_lvalue, cv[0] as u8, cv[1] as u8
                    ),
                    &format!(
                        "ctrl0<-{}, ctrl1<-{}, target<-{}, ctrl->{}{}",
                        c0_r, c1_r, t_r, clv_var, elem_str
                    ),
                )?;
            }
            Instruction::InlineTwoBitCSwap {
                ctrl,
                x,
                y,
                ctrl_lvalue,
                x_lvalue,
                y_lvalue,
            } => {
                let (c_l, c_r) = self.format_slot_ref(*ctrl, slot_map, sub_name)?;
                let (x_l, x_r) = self.format_slot_ref(*x, slot_map, sub_name)?;
                let (y_l, y_r) = self.format_slot_ref(*y, slot_map, sub_name)?;
                let clv_var = self.get_slot_var(*ctrl_lvalue, slot_map, sub_name)?;
                let xlv_var = self.get_slot_var(*x_lvalue, slot_map, sub_name)?;
                let ylv_var = self.get_slot_var(*y_lvalue, slot_map, sub_name)?;
                self.write_line(
                    &format!(
                        "    InlineTwoBitCSwap {} {} {} {} {} {}",
                        c_l, x_l, y_l, ctrl_lvalue, x_lvalue, y_lvalue
                    ),
                    &format!(
                        "ctrl<-{}, x<-{}, y<-{}, ctrl->{}, x->{}, y->{}",
                        c_r, x_r, y_r, clv_var, xlv_var, ylv_var
                    ),
                )?;
            }
            Instruction::InlineZ { q, lvalue_slot } => {
                let (q_l, q_r) = self.format_slot_ref(*q, slot_map, sub_name)?;
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineZ {} {}", q_l, lvalue_slot),
                    &format!("q<-{}, q->{}", q_r, lv_var),
                )?;
            }
            Instruction::InlineS { q, lvalue_slot } => {
                let (q_l, q_r) = self.format_slot_ref(*q, slot_map, sub_name)?;
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineS {} {}", q_l, lvalue_slot),
                    &format!("q<-{}, q->{}", q_r, lv_var),
                )?;
            }
            Instruction::InlineT { q, lvalue_slot } => {
                let (q_l, q_r) = self.format_slot_ref(*q, slot_map, sub_name)?;
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineT {} {}", q_l, lvalue_slot),
                    &format!("q<-{}, q->{}", q_r, lv_var),
                )?;
            }
            Instruction::InlineCZ {
                q0,
                q1,
                lvalue_slot,
                q0_lvalue,
                q1_lvalue,
            } => {
                let (q0_l, q0_r) = self.format_slot_ref(*q0, slot_map, sub_name)?;
                let (q1_l, q1_r) = self.format_slot_ref(*q1, slot_map, sub_name)?;
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                let elem_str = match (q0_lvalue, q1_lvalue) {
                    (Some(q0s), Some(q1s)) => format!(", elem[{},{}]", q0s, q1s),
                    _ => String::new(),
                };
                self.write_line(
                    &format!("    InlineCZ {} {} {}", q0_l, q1_l, lvalue_slot),
                    &format!("q0<-{}, q1<-{}, q->{}{}", q0_r, q1_r, lv_var, elem_str),
                )?;
            }
            Instruction::InlineCCZ {
                q0,
                q1,
                q2,
                lvalue_slot,
                q0_lvalue,
                q1_lvalue,
                q2_lvalue,
            } => {
                let (q0_l, q0_r) = self.format_slot_ref(*q0, slot_map, sub_name)?;
                let (q1_l, q1_r) = self.format_slot_ref(*q1, slot_map, sub_name)?;
                let (q2_l, q2_r) = self.format_slot_ref(*q2, slot_map, sub_name)?;
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                let elem_str = match (q0_lvalue, q1_lvalue, q2_lvalue) {
                    (Some(q0s), Some(q1s), Some(q2s)) => format!(", elem[{},{},{}]", q0s, q1s, q2s),
                    _ => String::new(),
                };
                self.write_line(
                    &format!("    InlineCCZ {} {} {} {}", q0_l, q1_l, q2_l, lvalue_slot),
                    &format!(
                        "q0<-{}, q1<-{}, q2<-{}, q->{}{}",
                        q0_r, q1_r, q2_r, lv_var, elem_str
                    ),
                )?;
            }
            Instruction::InlineAllocate {
                n_bits,
                lvalue_slot,
                element_lvalue_slots,
            } => {
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                let elem_info = if element_lvalue_slots.is_empty() {
                    String::new()
                } else {
                    format!(", elem_slots={:?}", element_lvalue_slots)
                };
                self.write_line(
                    &format!("    InlineAllocate {} {}", n_bits, lvalue_slot),
                    &format!("n_bits={}, out->{}{}", n_bits, lv_var, elem_info),
                )?;
            }
            Instruction::InlineZeroState { lvalue_slot } => {
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineZeroState {}", lvalue_slot),
                    &format!("out->{}", lv_var),
                )?;
            }
            Instruction::InlineOneState { lvalue_slot } => {
                let lv_var = self.get_slot_var(*lvalue_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineOneState {}", lvalue_slot),
                    &format!("out->{}", lv_var),
                )?;
            }
            Instruction::InlineFree { n_bits, in_bits } => {
                let (in_l, in_r) = self.format_slot_refs(in_bits, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineFree {} {}", n_bits, in_l),
                    &format!("n_bits={}, in<-[{}]", n_bits, in_r),
                )?;
            }
            Instruction::InlineZeroEffect { q } => {
                let (q_l, q_r) = self.format_slot_ref(*q, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineZeroEffect {}", q_l),
                    &format!("q<-{}", q_r),
                )?;
            }
            Instruction::InlineOneEffect { q } => {
                let (q_l, q_r) = self.format_slot_ref(*q, slot_map, sub_name)?;
                self.write_line(
                    &format!("    InlineOneEffect {}", q_l),
                    &format!("q<-{}", q_r),
                )?;
            }
        }
        Ok(())
    }

    fn decompile_extern_gate(
        &mut self,
        gate: &ExternGate,
        slot_map: &HashMap<SlotIdx, &'a str>,
        sub_name: &str,
    ) -> Result<(), String> {
        match gate {
            ExternGate::XGate { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    XGate {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::CNOT {
                ctrl_slot,
                target_slot,
            } => {
                let c_var = self.get_slot_var(*ctrl_slot, slot_map, sub_name)?;
                let t_var = self.get_slot_var(*target_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    CNOT {} {}", ctrl_slot, target_slot),
                    &format!("ctrl_slot={}, target_slot={}", c_var, t_var),
                )?;
            }
            ExternGate::Toffoli {
                ctrl_slot,
                target_slot,
            } => {
                let c_var = self.get_slot_var(*ctrl_slot, slot_map, sub_name)?;
                let t_var = self.get_slot_var(*target_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    Toffoli {} {}", ctrl_slot, target_slot),
                    &format!("ctrl_slot={}, target_slot={}", c_var, t_var),
                )?;
            }
            ExternGate::And {
                ctrl_slot,
                target_slot,
                cv,
            } => {
                let c_var = self.get_slot_var(*ctrl_slot, slot_map, sub_name)?;
                let t_var = self.get_slot_var(*target_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!(
                        "    And {} {} cv=[{},{}]",
                        ctrl_slot, target_slot, cv[0] as u8, cv[1] as u8
                    ),
                    &format!("ctrl_slot={}, target_slot={}", c_var, t_var),
                )?;
            }
            ExternGate::AndDag {
                ctrl_slot,
                target_slot,
                cv,
            } => {
                let c_var = self.get_slot_var(*ctrl_slot, slot_map, sub_name)?;
                let t_var = self.get_slot_var(*target_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!(
                        "    AndDag {} {} cv=[{},{}]",
                        ctrl_slot, target_slot, cv[0] as u8, cv[1] as u8
                    ),
                    &format!("ctrl_slot={}, target_slot={}", c_var, t_var),
                )?;
            }
            ExternGate::TwoBitCSwap {
                ctrl_slot,
                x_slot,
                y_slot,
            } => {
                let c_var = self.get_slot_var(*ctrl_slot, slot_map, sub_name)?;
                let x_var = self.get_slot_var(*x_slot, slot_map, sub_name)?;
                let y_var = self.get_slot_var(*y_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    TwoBitCSwap {} {} {}", ctrl_slot, x_slot, y_slot),
                    &format!("ctrl_slot={}, x_slot={}, y_slot={}", c_var, x_var, y_var),
                )?;
            }
            ExternGate::ZGate { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    ZGate {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::SGate { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    SGate {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::TGate { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    TGate {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::CZGate { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    CZGate {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::CCZGate { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    CCZGate {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::Allocate { n_bits, out_slot } => {
                let out_var = self.get_slot_var(*out_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    Allocate {} {}", n_bits, out_slot),
                    &format!("n_bits={}, out_slot={}", n_bits, out_var),
                )?;
            }
            ExternGate::ZeroState { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    ZeroState {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::OneState { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    OneState {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::Free { n_bits, in_slot } => {
                let in_var = self.get_slot_var(*in_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    Free {} {}", n_bits, in_slot),
                    &format!("n_bits={}, in_slot={}", n_bits, in_var),
                )?;
            }
            ExternGate::ZeroEffect { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    ZeroEffect {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
            ExternGate::OneEffect { q_slot } => {
                let q_var = self.get_slot_var(*q_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    OneEffect {}", q_slot),
                    &format!("q_slot={}", q_var),
                )?;
            }
        }
        Ok(())
    }

    fn decompile_cast_op(
        &mut self,
        op: &CastOp,
        slot_map: &HashMap<SlotIdx, &'a str>,
        sub_name: &str,
    ) -> Result<(), String> {
        match op {
            CastOp::Split {
                total_bits,
                reg_slot,
            } => {
                let reg_var = self.get_slot_var(*reg_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    Split {} {}", total_bits, reg_slot),
                    &format!("total_bits={}, reg_slot={}", total_bits, reg_var),
                )?;
            }
            CastOp::Join {
                total_bits,
                reg_slot,
            } => {
                let reg_var = self.get_slot_var(*reg_slot, slot_map, sub_name)?;
                self.write_line(
                    &format!("    Join {} {}", total_bits, reg_slot),
                    &format!("total_bits={}, reg_slot={}", total_bits, reg_var),
                )?;
            }
            CastOp::Partition {
                input_slots,
                output_slots,
                total_bits,
            } => {
                let fmt_slots = |slots: &[(SlotIdx, usize)]| -> Result<(String, String), String> {
                    let mut idx_parts = Vec::with_capacity(slots.len());
                    let mut var_parts = Vec::with_capacity(slots.len());
                    for &(slot, n_bits) in slots {
                        let var = self.get_slot_var(slot, slot_map, sub_name)?;
                        idx_parts.push(format!("{}:{}", slot, n_bits));
                        var_parts.push(format!("{}:{}", var, n_bits));
                    }
                    Ok((idx_parts.join(","), var_parts.join(",")))
                };
                let (in_idx, in_var) = fmt_slots(input_slots)?;
                let (out_idx, out_var) = fmt_slots(output_slots)?;
                self.write_line(
                    &format!("    Partition {} [{}] -> [{}]", total_bits, in_idx, out_idx),
                    &format!(
                        "total_bits={}, inputs=[{}], outputs=[{}]",
                        total_bits, in_var, out_var
                    ),
                )?;
            }
        }
        Ok(())
    }
}
