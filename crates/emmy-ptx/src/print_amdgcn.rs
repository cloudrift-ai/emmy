//! The AMD printer: a module as LLVM IR for the `amdgcn` target, which AMD's `llc` assembles as ptxas does PTX. It
//! reads the same instruction list the PTX printer does, one instruction at a time, and decides nothing.
//!
//! The list is not in SSA form: a state is a register defined again. So every register is a stack slot (`alloca`),
//! read with `load` and defined with `store`, and the one pass `mem2reg` turns the slots into values before `llc`
//! runs. A label opens a block, and an instruction under a guard is a branch around it. An instruction only an
//! NVIDIA card has is an `Error::NotImplemented` that names it.

use std::collections::BTreeMap;
use std::fmt::{Display, Write as _};

use emmy_ir::{Error, Result};

use crate::inst::{
    Bank, Base, Cmp, Inst, Item, Label, Module, Opcode, Operand, ParamDecl, Reg, Round, Space,
    Special, Ty,
};
use crate::print::instruction_text;

/// The LLVM type of a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum L {
    I1,
    I8,
    I16,
    I32,
    I64,
    Half,
    Float,
    Double,
}

impl L {
    fn text(self) -> &'static str {
        match self {
            L::I1 => "i1",
            L::I8 => "i8",
            L::I16 => "i16",
            L::I32 => "i32",
            L::I64 => "i64",
            L::Half => "half",
            L::Float => "float",
            L::Double => "double",
        }
    }

    fn bits(self) -> u32 {
        match self {
            L::I1 => 1,
            L::I8 => 8,
            L::I16 | L::Half => 16,
            L::I32 | L::Float => 32,
            L::I64 | L::Double => 64,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, L::Half | L::Float | L::Double)
    }

    /// The integer type of as many bits.
    fn int(self) -> L {
        match self.bits() {
            1 => L::I1,
            8 => L::I8,
            16 => L::I16,
            32 => L::I32,
            _ => L::I64,
        }
    }

    /// The type as an intrinsic's name spells it.
    fn suffix(self) -> &'static str {
        match self {
            L::Half => "f16",
            L::Float => "f32",
            L::Double => "f64",
            int => int.text(),
        }
    }
}

/// The type a register of `bank` holds: PTX registers are bits of a width, so a half is 16 bits and a double 64.
fn held(bank: Bank) -> L {
    match bank {
        Bank::Pred => L::I1,
        Bank::B16 => L::I16,
        Bank::B32 => L::I32,
        Bank::B64 => L::I64,
        Bank::F32 => L::Float,
    }
}

/// The type an instruction of `ty` computes at, where LLVM IR has it.
fn computed(ty: Ty) -> Option<L> {
    Some(match ty {
        Ty::Pred => L::I1,
        Ty::U8 => L::I8,
        Ty::B16 | Ty::U16 | Ty::S16 => L::I16,
        Ty::B32 | Ty::S32 | Ty::U32 => L::I32,
        Ty::B64 | Ty::S64 | Ty::U64 => L::I64,
        Ty::F16 => L::Half,
        Ty::F32 => L::Float,
        Ty::F64 => L::Double,
        _ => return None,
    })
}

fn signed(ty: Ty) -> bool {
    matches!(ty, Ty::S16 | Ty::S32 | Ty::S64)
}

/// The address space of a state space's pointers.
fn pointer(space: Space) -> &'static str {
    match space {
        Space::Shared => "ptr addrspace(3)",
        _ => "ptr addrspace(1)",
    }
}

fn register(reg: Reg) -> String {
    format!("%{}{}", reg.bank.prefix(), reg.number)
}

/// An integer constant of type `to`, wrapped to its width.
fn literal(value: i64, to: L) -> String {
    match to {
        L::I1 => (if value & 1 == 1 { "true" } else { "false" }).into(),
        L::I8 => (value as i8).to_string(),
        L::I16 => (value as i16).to_string(),
        L::I32 => (value as i32).to_string(),
        _ => value.to_string(),
    }
}

/// An f32 constant by its bits: LLVM IR spells a float as the bits of the same value as a double.
fn float(bits: u32) -> String {
    format!("0x{:016X}", f64::from(f32::from_bits(bits)).to_bits())
}

/// The LLVM IR of `module`. An error names the instruction that has no AMD form.
pub fn print(module: &Module) -> Result<String> {
    let mut printer = Printer {
        module,
        body: String::new(),
        temps: 0,
        blocks: 0,
        open: true,
        declared: BTreeMap::new(),
    };
    for item in &module.items {
        match item {
            Item::Inst(inst) => printer.instruction(inst)?,
            Item::Label(label) => printer.bind(&label_text(*label)),
            Item::Comment(text) => {
                let _ = writeln!(printer.body, "\t; {}", text.replace('\n', " "));
            }
            Item::NoUnroll => {}
        }
    }
    if printer.open {
        printer.body.push_str("\tret void\n");
    }
    Ok(printer.frame())
}

fn label_text(label: Label) -> String {
    format!("L{}", label.0)
}

struct Printer<'a> {
    module: &'a Module,
    body: String,
    temps: u32,
    blocks: u32,
    /// Whether the block being written has no branch at its end yet.
    open: bool,
    /// The intrinsics the body calls, each with its declaration.
    declared: BTreeMap<String, String>,
}

impl Printer<'_> {
    fn refuse(&self, inst: &Inst) -> Error {
        let (text, target) = (instruction_text(inst), &self.module.target);
        Error::NotImplemented(format!("no LLVM IR for `{text}` on {target}"))
    }

    /// The module around the body: the triple, the shared buffers, the entry with a stack slot for every register,
    /// the intrinsics' declarations and the kernel's attributes.
    fn frame(self) -> String {
        let module = self.module;
        let entry = &module.entry;
        let mut out = String::new();
        out.push_str("target triple = \"amdgcn-amd-amdhsa\"\n\n");
        for (index, shared) in module.shared.iter().enumerate() {
            let align = shared.align;
            let _ = match shared.bytes {
                Some(bytes) => writeln!(
                    out,
                    "@{entry}_shared_{index} = internal addrspace(3) global [{bytes} x i8] undef, align {align}"
                ),
                None => writeln!(
                    out,
                    "@{entry}_shared_{index} = external addrspace(3) global [0 x i8], align {align}"
                ),
            };
        }
        let params: Vec<String> = module
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| match param {
                ParamDecl::Pointer { align } => {
                    format!("ptr addrspace(1) noundef align {align} %a{index}")
                }
                ParamDecl::U32 => format!("i32 noundef %a{index}"),
            })
            .collect();
        let _ = writeln!(
            out,
            "define protected amdgpu_kernel void @{entry}({}) #0 {{\nentry:",
            params.join(", ")
        );
        for (bank, count) in &module.registers {
            let ty = held(*bank);
            let bytes = ty.bits().max(8) / 8;
            for number in 0..*count {
                let reg = register(Reg {
                    bank: *bank,
                    number,
                });
                let _ = writeln!(
                    out,
                    "\t{reg} = alloca {}, align {bytes}, addrspace(5)",
                    ty.text()
                );
            }
        }
        out.push_str(&self.body);
        out.push_str("}\n\n");
        for declaration in self.declared.values() {
            let _ = writeln!(out, "declare {declaration}");
        }
        let ftz = module.instructions().any(|inst| inst.ftz);
        let _ = writeln!(
            out,
            "\nattributes #0 = {{ nounwind \"target-cpu\"=\"{}\" \"uniform-work-group-size\"=\"true\" \
             \"amdgpu-flat-work-group-size\"=\"1,{}\"{} }}",
            module.target,
            module.max_threads,
            if ftz {
                " \"denormal-fp-math-f32\"=\"preserve-sign,preserve-sign\""
            } else {
                ""
            }
        );
        out
    }

    // --- lines, blocks and values ------------------------------------------------------------------------

    fn fresh(&mut self, prefix: &str) -> String {
        self.blocks += 1;
        format!("{prefix}{}", self.blocks)
    }

    /// One line of the block being written. A line after a branch nothing jumps past opens a block of its own.
    fn line(&mut self, text: impl Display) {
        if !self.open {
            let dead = self.fresh("d");
            let _ = writeln!(self.body, "{dead}:");
            self.open = true;
        }
        let _ = writeln!(self.body, "\t{text}");
    }

    /// A new value: `text` is what computes it.
    fn value(&mut self, text: impl Display) -> String {
        self.temps += 1;
        let name = format!("%t{}", self.temps);
        self.line(format_args!("{name} = {text}"));
        name
    }

    /// Open the block `label`, falling into it from the block before.
    fn bind(&mut self, label: &str) {
        if self.open {
            let _ = writeln!(self.body, "\tbr label %{label}");
        }
        let _ = writeln!(self.body, "{label}:");
        self.open = true;
    }

    fn jump(&mut self, label: &str) {
        self.line(format_args!("br label %{label}"));
        self.open = false;
    }

    /// Go to `then` where `pred` holds (or, negated, where it does not), else to `other`.
    fn fork(&mut self, pred: Reg, negated: bool, then: &str, other: &str) {
        let pred = self.read(pred);
        let (yes, no) = if negated {
            (other, then)
        } else {
            (then, other)
        };
        self.line(format_args!("br i1 {pred}, label %{yes}, label %{no}"));
        self.open = false;
    }

    fn read(&mut self, reg: Reg) -> String {
        let ty = held(reg.bank).text();
        self.value(format_args!(
            "load {ty}, ptr addrspace(5) {}",
            register(reg)
        ))
    }

    /// Define `reg` as `value` of type `from`, widened by its sign where `signed` says.
    fn define(&mut self, reg: Reg, value: String, from: L, signed: bool) {
        let to = held(reg.bank);
        let value = self.fit(value, from, to, signed);
        self.line(format_args!(
            "store {} {value}, ptr addrspace(5) {}",
            to.text(),
            register(reg)
        ));
    }

    /// `value` of type `from` as type `to`: the same bits where the widths agree, else the integer widened or cut.
    fn fit(&mut self, value: String, from: L, to: L, signed: bool) -> String {
        if from == to {
            return value;
        }
        let mut value = value;
        if from.is_float() {
            value = self.value(format_args!(
                "bitcast {} {value} to {}",
                from.text(),
                from.int().text()
            ));
        }
        let (narrow, wide) = (from.int(), to.int());
        if narrow.bits() != wide.bits() {
            let op = match (narrow.bits() > wide.bits(), signed && from != L::I1) {
                (true, _) => "trunc",
                (false, true) => "sext",
                (false, false) => "zext",
            };
            value = self.value(format_args!(
                "{op} {} {value} to {}",
                narrow.text(),
                wide.text()
            ));
        }
        if to.is_float() {
            value = self.value(format_args!(
                "bitcast {} {value} to {}",
                wide.text(),
                to.text()
            ));
        }
        value
    }

    /// The operand `operand` of `inst` as a value of type `to`.
    fn operand(&mut self, inst: &Inst, operand: &Operand, to: L, signed: bool) -> Result<String> {
        match operand {
            Operand::Reg(reg) => {
                let value = self.read(*reg);
                Ok(self.fit(value, held(reg.bank), to, signed))
            }
            Operand::Int(value) if !to.is_float() => Ok(literal(*value, to)),
            // The bits of a float, where the instruction moves bits.
            Operand::Int(value) if !inst.ty.is_float() => {
                let bits = literal(*value, to.int());
                Ok(self.fit(bits, to.int(), to, false))
            }
            Operand::F32(bits) => Ok(match to {
                L::Float => float(*bits),
                L::Half => self.value(format_args!("fptrunc float {} to half", float(*bits))),
                L::Double => self.value(format_args!("fpext float {} to double", float(*bits))),
                int => {
                    let bits = literal(i64::from(*bits), L::I32);
                    self.fit(bits, L::I32, int, false)
                }
            }),
            Operand::Special(special) => {
                let value = self.special(*special);
                Ok(self.fit(value, L::I32, to, false))
            }
            Operand::Shared(index) => {
                let symbol = format!("@{}_shared_{index}", self.module.entry);
                let address = self.value(format_args!("ptrtoint ptr addrspace(3) {symbol} to i32"));
                Ok(self.fit(address, L::I32, to, false))
            }
            _ => Err(self.refuse(inst)),
        }
    }

    /// Call the intrinsic `name`, declared as `ret name(params)`.
    fn call(&mut self, ret: &str, name: &str, params: &[&str], args: &[String]) -> String {
        let declaration = format!("{ret} @{name}({})", params.join(", "));
        self.declared.insert(name.to_string(), declaration);
        let args: Vec<String> = params
            .iter()
            .zip(args)
            .map(|(param, arg)| format!("{param} {arg}"))
            .collect();
        let call = format!("call {ret} @{name}({})", args.join(", "));
        if ret == "void" {
            self.line(call);
            return String::new();
        }
        self.value(call)
    }

    /// A math intrinsic of `ty` over operands of that type.
    fn math(&mut self, name: &str, ty: L, args: &[String]) -> String {
        let name = format!("llvm.{name}.{}", ty.suffix());
        let params = vec![ty.text(); args.len()];
        self.call(ty.text(), &name, &params, args)
    }

    /// A fact of the thread and the launch. A block's extent and the grid's stand in the dispatch packet, the
    /// grid's counted in threads.
    fn special(&mut self, special: Special) -> String {
        use Special::*;
        let axis = match special {
            TidX | CtaidX | NtidX | NctaidX => 0,
            TidY | CtaidY | NtidY | NctaidY => 1,
            TidZ | CtaidZ | NtidZ | NctaidZ => 2,
        };
        let letter = ["x", "y", "z"][axis];
        match special {
            TidX | TidY | TidZ => self.call(
                "i32",
                &format!("llvm.amdgcn.workitem.id.{letter}"),
                &[],
                &[],
            ),
            CtaidX | CtaidY | CtaidZ => self.call(
                "i32",
                &format!("llvm.amdgcn.workgroup.id.{letter}"),
                &[],
                &[],
            ),
            NtidX | NtidY | NtidZ => self.packet("i16", 4 + 2 * axis),
            NctaidX | NctaidY | NctaidZ => {
                let threads = self.packet("i32", 12 + 4 * axis);
                let block = self.packet("i16", 4 + 2 * axis);
                self.value(format_args!("udiv i32 {threads}, {block}"))
            }
        }
    }

    /// The field of `ty` at byte `offset` of the dispatch packet, as 32 bits.
    fn packet(&mut self, ty: &str, offset: usize) -> String {
        let packet = self.call("ptr addrspace(4)", "llvm.amdgcn.dispatch.ptr", &[], &[]);
        let field = self.value(format_args!(
            "getelementptr i8, ptr addrspace(4) {packet}, i64 {offset}"
        ));
        let align = if ty == "i16" { 2 } else { 4 };
        let value = self.value(format_args!(
            "load {ty}, ptr addrspace(4) {field}, align {align}"
        ));
        match ty {
            "i16" => self.value(format_args!("zext i16 {value} to i32")),
            _ => value,
        }
    }

    // --- instructions ------------------------------------------------------------------------------------

    fn instruction(&mut self, inst: &Inst) -> Result<()> {
        match (inst.opcode, inst.guard) {
            (Opcode::Bra, guard) => {
                let target = inst
                    .target()
                    .map(label_text)
                    .ok_or_else(|| self.refuse(inst))?;
                match guard {
                    None => self.jump(&target),
                    Some(guard) => {
                        let next = self.fresh("n");
                        self.fork(guard.pred, guard.negated, &target, &next);
                        self.bind(&next);
                    }
                }
                Ok(())
            }
            (_, None) => self.plain(inst),
            (_, Some(guard)) => {
                let (then, after) = (self.fresh("g"), self.fresh("c"));
                self.fork(guard.pred, guard.negated, &then, &after);
                self.bind(&then);
                self.plain(inst)?;
                self.bind(&after);
                Ok(())
            }
        }
    }

    /// The type `inst` computes at, or the refusal of a type LLVM IR has not.
    fn at(&self, inst: &Inst, ty: Ty) -> Result<L> {
        computed(ty).ok_or_else(|| self.refuse(inst))
    }

    /// The operands of `inst`, each at type `ty`.
    fn operands(&mut self, inst: &Inst, ty: L) -> Result<Vec<String>> {
        let signed = signed(inst.ty);
        inst.operands
            .iter()
            .map(|operand| self.operand(inst, operand, ty, signed))
            .collect()
    }

    fn result(&self, inst: &Inst) -> Result<Reg> {
        inst.results
            .first()
            .copied()
            .ok_or_else(|| self.refuse(inst))
    }

    fn plain(&mut self, inst: &Inst) -> Result<()> {
        use Opcode::*;
        // A float instruction rounds to nearest. The one other mode the rules write is a multiply-add rounded
        // down; the approximate forms are NVIDIA's, the reciprocal square root being the one AMD has too.
        match (inst.opcode, inst.round) {
            (Fma, Some(Round::Rm)) if inst.ty == Ty::F32 && !inst.sat => {
                return self.fma_down(inst);
            }
            (Cvt(_), _) | (Rsqrt, Some(Round::Approx)) | (_, Some(Round::Rn) | None) => {}
            _ => return Err(self.refuse(inst)),
        }
        match inst.opcode {
            Mov => self.mov(inst),
            Add | Sub | Mul | Div | Rem | And | Or | Xor | Shl | Shr | Min | Max => {
                self.binary(inst)
            }
            Neg | Abs | Not | Popc | Sqrt | Rcp | Rsqrt => self.unary(inst),
            Fma => {
                let ty = self.at(inst, inst.ty)?;
                let args = self.operands(inst, ty)?;
                let value = self.math("fma", ty, &args);
                self.float_result(inst, value, ty)
            }
            // The second operand with the sign of the first.
            Copysign => {
                let ty = self.at(inst, inst.ty)?;
                let mut args = self.operands(inst, ty)?;
                args.reverse();
                let value = self.math("copysign", ty, &args);
                self.float_result(inst, value, ty)
            }
            MulHi | MulWide => self.wide_product(inst),
            Setp(cmp) => self.compare(inst, cmp),
            Selp => {
                let ty = self.at(inst, inst.ty)?;
                let [a, b, Operand::Reg(pred)] = &inst.operands[..] else {
                    return Err(self.refuse(inst));
                };
                let a = self.operand(inst, a, ty, signed(inst.ty))?;
                let b = self.operand(inst, b, ty, signed(inst.ty))?;
                let pred = self.read(*pred);
                let text = ty.text();
                let value = self.value(format_args!("select i1 {pred}, {text} {a}, {text} {b}"));
                self.define(self.result(inst)?, value, ty, signed(inst.ty));
                Ok(())
            }
            Cvt(from) => self.convert(inst, from),
            Ld(Space::Param) => self.parameter(inst),
            Ld(space) => self.load(inst, space),
            LdConst => self.load(inst, Space::Global),
            St(space) => self.store(inst, space),
            AtomAdd(space) => {
                let ty = self.at(inst, inst.ty)?;
                let address = self.address(inst, &inst.operands[0], space)?;
                let value = self.operand(inst, &inst.operands[1], ty, signed(inst.ty))?;
                let op = if ty.is_float() { "fadd" } else { "add" };
                let (ptr, text) = (pointer(space), ty.text());
                let old = self.value(format_args!(
                    "atomicrmw {op} {ptr} {address}, {text} {value} monotonic"
                ));
                self.define(self.result(inst)?, old, ty, signed(inst.ty));
                Ok(())
            }
            // A barrier of the whole block. A named one counts the threads that reach it, which AMD's does not.
            BarSync if inst.operands.len() == 1 => {
                self.call("void", "llvm.amdgcn.s.barrier", &[], &[]);
                Ok(())
            }
            Ret => {
                self.line("ret void");
                self.open = false;
                Ok(())
            }
            _ => Err(self.refuse(inst)),
        }
    }

    /// An f32 `a * b + c` rounded towards minus infinity, which LLVM IR has no mode for. The product is exact in a
    /// double and the sum's rounding error is computed beside it, so the sum rounded to nearest in f32 is known
    /// to stand above the exact value or not; where it does, the result is the f32 below it.
    fn fma_down(&mut self, inst: &Inst) -> Result<()> {
        let args = self.operands(inst, L::Float)?;
        let [a, b, c] = &args[..] else {
            return Err(self.refuse(inst));
        };
        let wide = |printer: &mut Self, value: &str| {
            printer.value(format_args!("fpext float {value} to double"))
        };
        let (a, b, c) = (wide(self, a), wide(self, b), wide(self, c));
        let product = self.value(format_args!("fmul double {a}, {b}"));
        let sum = self.value(format_args!("fadd double {product}, {c}"));
        let added = self.value(format_args!("fsub double {sum}, {product}"));
        let kept = self.value(format_args!("fsub double {sum}, {added}"));
        let lost = self.value(format_args!("fsub double {product}, {kept}"));
        let left = self.value(format_args!("fsub double {c}, {added}"));
        let error = self.value(format_args!("fadd double {lost}, {left}"));
        let near = self.value(format_args!("fptrunc double {sum} to float"));
        let back = wide(self, &near);
        let above = self.value(format_args!("fcmp ogt double {back}, {sum}"));
        let equal = self.value(format_args!("fcmp oeq double {back}, {sum}"));
        let short = self.value(format_args!("fcmp olt double {error}, 0.0"));
        let equal = self.value(format_args!("and i1 {equal}, {short}"));
        let above = self.value(format_args!("or i1 {above}, {equal}"));
        let bits = self.value(format_args!("bitcast float {near} to i32"));
        let positive = self.value(format_args!("fcmp ogt float {near}, 0.0"));
        let negative = self.value(format_args!("fcmp olt float {near}, 0.0"));
        let smaller = self.value(format_args!("sub i32 {bits}, 1"));
        let larger = self.value(format_args!("add i32 {bits}, 1"));
        let tiny = literal(0x8000_0001, L::I32);
        let below = self.value(format_args!(
            "select i1 {negative}, i32 {larger}, i32 {tiny}"
        ));
        let below = self.value(format_args!(
            "select i1 {positive}, i32 {smaller}, i32 {below}"
        ));
        let below = self.value(format_args!("bitcast i32 {below} to float"));
        let value = self.value(format_args!(
            "select i1 {above}, float {below}, float {near}"
        ));
        self.define(self.result(inst)?, value, L::Float, false);
        Ok(())
    }

    /// Define the result of a float instruction, clamped to `[0, 1]` with a NaN as zero where it saturates.
    fn float_result(&mut self, inst: &Inst, value: String, ty: L) -> Result<()> {
        let mut value = value;
        if inst.sat {
            if ty != L::Float {
                return Err(self.refuse(inst));
            }
            value = self.math("maxnum", ty, &[value, float(0.0_f32.to_bits())]);
            value = self.math("minnum", ty, &[value, float(1.0_f32.to_bits())]);
        }
        self.define(self.result(inst)?, value, ty, false);
        Ok(())
    }

    /// A move of bits: between registers of one width, of a constant, of two registers into one of twice the
    /// width, or of one register into its two halves.
    fn mov(&mut self, inst: &Inst) -> Result<()> {
        let source = inst.operands.first().ok_or_else(|| self.refuse(inst))?;
        if let [low, high] = inst.results[..] {
            let Operand::Reg(word) = source else {
                return Err(self.refuse(inst));
            };
            let (from, part) = (held(word.bank).int(), held(low.bank).int());
            let word_value = self.read(*word);
            let word_value = self.fit(word_value, held(word.bank), from, false);
            let upper = self.value(format_args!(
                "lshr {} {word_value}, {}",
                from.text(),
                part.bits()
            ));
            self.define(low, word_value, from, false);
            self.define(high, upper, from, false);
            return Ok(());
        }
        let result = self.result(inst)?;
        let to = held(result.bank);
        if let Operand::Tuple(parts) = source {
            let [low, high] = parts[..] else {
                return Err(self.refuse(inst));
            };
            let (wide, part) = (to.int(), held(low.bank));
            let low = self.read(low);
            let low = self.fit(low, part, wide, false);
            let high = self.read(high);
            let high = self.fit(high, part, wide, false);
            let text = wide.text();
            let high = self.value(format_args!("shl {text} {high}, {}", part.bits()));
            let word = self.value(format_args!("or {text} {low}, {high}"));
            self.define(result, word, wide, false);
            return Ok(());
        }
        let value = self.operand(inst, source, to, signed(inst.ty))?;
        self.define(result, value, to, false);
        Ok(())
    }

    fn binary(&mut self, inst: &Inst) -> Result<()> {
        use Opcode::*;
        let ty = self.at(inst, inst.ty)?;
        let signed = signed(inst.ty);
        let args = self.operands(inst, ty)?;
        let [a, b] = &args[..] else {
            return Err(self.refuse(inst));
        };
        let text = ty.text();
        if ty.is_float() {
            let value = match inst.opcode {
                Add => self.value(format_args!("fadd {text} {a}, {b}")),
                Sub => self.value(format_args!("fsub {text} {a}, {b}")),
                Mul => self.value(format_args!("fmul {text} {a}, {b}")),
                Div => self.value(format_args!("fdiv {text} {a}, {b}")),
                // PTX gives the operand that is a number where the other is a NaN.
                Min => self.math("minnum", ty, &args),
                Max => self.math("maxnum", ty, &args),
                _ => return Err(self.refuse(inst)),
            };
            return self.float_result(inst, value, ty);
        }
        let sign = if signed { "s" } else { "u" };
        let value = match inst.opcode {
            Add | Sub | Mul | And | Or | Xor => {
                let op = format!("{:?}", inst.opcode).to_lowercase();
                self.value(format_args!("{op} {text} {a}, {b}"))
            }
            Div => self.value(format_args!("{sign}div {text} {a}, {b}")),
            Rem => self.value(format_args!("{sign}rem {text} {a}, {b}")),
            Min | Max => {
                let op = format!("{sign}{:?}", inst.opcode).to_lowercase();
                self.math(&op, ty, &args)
            }
            Shl | Shr => self.shift(inst, ty, a, b),
            _ => return Err(self.refuse(inst)),
        };
        self.define(self.result(inst)?, value, ty, signed);
        Ok(())
    }

    /// A shift as PTX defines it past the register's width: zero, or the sign for a signed shift right. LLVM IR
    /// leaves it undefined.
    fn shift(&mut self, inst: &Inst, ty: L, value: &str, by: &str) -> String {
        let (text, bits) = (ty.text(), ty.bits());
        let arithmetic = inst.opcode == Opcode::Shr && signed(inst.ty);
        let op = match (inst.opcode, arithmetic) {
            (Opcode::Shl, _) => "shl",
            (_, true) => "ashr",
            (_, false) => "lshr",
        };
        if let Some(Operand::Int(by)) = inst.operands.get(1)
            && (0..i64::from(bits)).contains(by)
        {
            return self.value(format_args!("{op} {text} {value}, {by}"));
        }
        let last = (bits - 1).to_string();
        let within = self.math("umin", ty, &[by.to_string(), last]);
        let shifted = self.value(format_args!("{op} {text} {value}, {within}"));
        if arithmetic {
            return shifted;
        }
        let fits = self.value(format_args!("icmp ult {text} {by}, {bits}"));
        self.value(format_args!("select i1 {fits}, {text} {shifted}, {text} 0"))
    }

    fn unary(&mut self, inst: &Inst) -> Result<()> {
        use Opcode::*;
        let ty = self.at(inst, inst.ty)?;
        let args = self.operands(inst, ty)?;
        let [a] = &args[..] else {
            return Err(self.refuse(inst));
        };
        let text = ty.text();
        if ty.is_float() {
            let value = match inst.opcode {
                Neg => self.value(format_args!("fneg {text} {a}")),
                Abs => self.math("fabs", ty, &args),
                Sqrt => self.math("sqrt", ty, &args),
                Rcp => {
                    let one = self.fit(float(1.0_f32.to_bits()), L::Float, L::Float, false);
                    match ty {
                        L::Float => self.value(format_args!("fdiv float {one}, {a}")),
                        _ => return Err(self.refuse(inst)),
                    }
                }
                Rsqrt if ty == L::Float => {
                    self.call("float", "llvm.amdgcn.rsq.f32", &["float"], &args)
                }
                _ => return Err(self.refuse(inst)),
            };
            return self.float_result(inst, value, ty);
        }
        let value = match inst.opcode {
            Neg => self.value(format_args!("sub {text} 0, {a}")),
            Not => self.value(format_args!("xor {text} {a}, {}", literal(-1, ty))),
            Abs => {
                let name = format!("llvm.abs.{text}");
                self.call(text, &name, &[text, "i1"], &[a.clone(), "false".into()])
            }
            Popc => self.math("ctpop", ty, &args),
            _ => return Err(self.refuse(inst)),
        };
        self.define(self.result(inst)?, value, ty, false);
        Ok(())
    }

    /// The high half of a product (`MulHi`), or the 64-bit product of two 32-bit operands (`MulWide`).
    fn wide_product(&mut self, inst: &Inst) -> Result<()> {
        let ty = self.at(inst, inst.ty)?;
        let signed = signed(inst.ty);
        let args = self.operands(inst, ty)?;
        let [a, b] = &args[..] else {
            return Err(self.refuse(inst));
        };
        let (text, wide) = match ty {
            L::I32 => ("i32", "i64"),
            L::I64 => ("i64", "i128"),
            _ => return Err(self.refuse(inst)),
        };
        let extend = if signed { "sext" } else { "zext" };
        let a = self.value(format_args!("{extend} {text} {a} to {wide}"));
        let b = self.value(format_args!("{extend} {text} {b} to {wide}"));
        let product = self.value(format_args!("mul {wide} {a}, {b}"));
        let result = self.result(inst)?;
        if inst.opcode == Opcode::MulWide {
            if ty != L::I32 {
                return Err(self.refuse(inst));
            }
            self.define(result, product, L::I64, signed);
            return Ok(());
        }
        let high = self.value(format_args!("lshr {wide} {product}, {}", ty.bits()));
        let high = self.value(format_args!("trunc {wide} {high} to {text}"));
        self.define(result, high, ty, signed);
        Ok(())
    }

    fn compare(&mut self, inst: &Inst, cmp: Cmp) -> Result<()> {
        let ty = self.at(inst, inst.ty)?;
        let args = self.operands(inst, ty)?;
        let [a, b] = &args[..] else {
            return Err(self.refuse(inst));
        };
        let sign = if signed(inst.ty) { "s" } else { "u" };
        let op = match (cmp, ty.is_float()) {
            (Cmp::Eq, true) => "fcmp oeq".to_string(),
            (Cmp::Ne, true) => "fcmp one".to_string(),
            (Cmp::Neu, true) => "fcmp une".to_string(),
            (Cmp::Nan, true) => "fcmp uno".to_string(),
            (Cmp::Lt, true) => "fcmp olt".to_string(),
            (Cmp::Le, true) => "fcmp ole".to_string(),
            (Cmp::Gt, true) => "fcmp ogt".to_string(),
            (Cmp::Ge, true) => "fcmp oge".to_string(),
            (Cmp::Eq, false) => "icmp eq".to_string(),
            (Cmp::Ne, false) => "icmp ne".to_string(),
            (Cmp::Lt, false) => format!("icmp {sign}lt"),
            (Cmp::Le, false) => format!("icmp {sign}le"),
            (Cmp::Gt, false) => format!("icmp {sign}gt"),
            (Cmp::Ge, false) => format!("icmp {sign}ge"),
            (Cmp::Neu | Cmp::Nan, false) => return Err(self.refuse(inst)),
        };
        let value = self.value(format_args!("{op} {} {a}, {b}", ty.text()));
        self.define(self.result(inst)?, value, L::I1, false);
        Ok(())
    }

    /// A conversion from `from` to the instruction's type. To an integer it rounds by the instruction's mode and
    /// saturates, a NaN to zero, as PTX does; to a float of its own type it rounds to an integral value.
    fn convert(&mut self, inst: &Inst, from: Ty) -> Result<()> {
        let (source, to) = (self.at(inst, from)?, self.at(inst, inst.ty)?);
        let [operand] = &inst.operands[..] else {
            return Err(self.refuse(inst));
        };
        if inst.sat {
            return Err(self.refuse(inst));
        }
        let value = self.operand(inst, operand, source, signed(from))?;
        let (source_text, to_text) = (source.text(), to.text());
        let integral = |printer: &mut Self, value: String| match inst.round {
            Some(Round::Rni) => Ok(printer.math("roundeven", source, &[value])),
            Some(Round::Rmi) => Ok(printer.math("floor", source, &[value])),
            Some(Round::Rpi) => Ok(printer.math("ceil", source, &[value])),
            Some(Round::Rzi) => Ok(printer.math("trunc", source, &[value])),
            Some(Round::Rn) | None => Ok(value),
            Some(Round::Rm | Round::Rna | Round::Approx) => Err(printer.refuse(inst)),
        };
        let value = match (source.is_float(), to.is_float()) {
            (false, false) => self.fit(value, source, to, signed(from)),
            (false, true) => {
                let op = if signed(from) { "sitofp" } else { "uitofp" };
                self.value(format_args!("{op} {source_text} {value} to {to_text}"))
            }
            (true, false) => {
                let value = integral(self, value)?;
                let sign = if signed(inst.ty) { "s" } else { "u" };
                let name = format!("llvm.fpto{sign}i.sat.{to_text}.{}", source.suffix());
                self.call(to_text, &name, &[source_text], &[value])
            }
            (true, true) if source == to => integral(self, value)?,
            (true, true) => {
                let op = if source.bits() < to.bits() {
                    "fpext"
                } else {
                    "fptrunc"
                };
                self.value(format_args!("{op} {source_text} {value} to {to_text}"))
            }
        };
        self.define(self.result(inst)?, value, to, signed(inst.ty));
        Ok(())
    }

    // --- memory ------------------------------------------------------------------------------------------

    /// A parameter into its register: a buffer's address as 64 bits, a size as it is.
    fn parameter(&mut self, inst: &Inst) -> Result<()> {
        let Some(Operand::Addr(Base::Param(index), 0)) = inst.operands.first() else {
            return Err(self.refuse(inst));
        };
        let result = self.result(inst)?;
        match self.module.params.get(*index as usize) {
            Some(ParamDecl::Pointer { .. }) => {
                let address =
                    self.value(format_args!("ptrtoint ptr addrspace(1) %a{index} to i64"));
                self.define(result, address, L::I64, false);
            }
            Some(ParamDecl::U32) => self.define(result, format!("%a{index}"), L::I32, false),
            None => return Err(self.refuse(inst)),
        }
        Ok(())
    }

    /// The pointer an address operand names in `space`: a register's value or a shared buffer's symbol, plus the
    /// operand's byte offset.
    fn address(&mut self, inst: &Inst, operand: &Operand, space: Space) -> Result<String> {
        let ptr = pointer(space);
        match operand {
            Operand::Addr(Base::Reg(reg), offset) => {
                let width = if space == Space::Shared {
                    L::I32
                } else {
                    L::I64
                };
                let base = self.read(*reg);
                let mut address = self.fit(base, held(reg.bank), width, false);
                let text = width.text();
                if *offset != 0 {
                    address = self.value(format_args!("add {text} {address}, {offset}"));
                }
                Ok(self.value(format_args!("inttoptr {text} {address} to {ptr}")))
            }
            Operand::Addr(Base::Shared(index), offset) if space == Space::Shared => {
                let symbol = format!("@{}_shared_{index}", self.module.entry);
                Ok(self.value(format_args!(
                    "getelementptr i8, {ptr} {symbol}, i32 {offset}"
                )))
            }
            _ => Err(self.refuse(inst)),
        }
    }

    /// The type an element of `ty` has in memory: the instruction's own, or as many bits where LLVM IR has no
    /// such type.
    fn stored(ty: Ty) -> L {
        computed(ty).unwrap_or(match ty.bytes() {
            1 => L::I8,
            2 => L::I16,
            4 => L::I32,
            _ => L::I64,
        })
    }

    /// One load: of one element, or of as many as the registers it defines as one vector.
    fn load(&mut self, inst: &Inst, space: Space) -> Result<()> {
        let address = self.address(inst, &inst.operands[0], space)?;
        let (elem, ptr) = (Self::stored(inst.ty), pointer(space));
        let (text, bytes) = (elem.text(), inst.ty.bytes());
        let signed = signed(inst.ty);
        if let [result] = inst.results[..] {
            let value = self.value(format_args!("load {text}, {ptr} {address}, align {bytes}"));
            self.define(result, value, elem, signed);
            return Ok(());
        }
        let lanes = inst.results.len();
        let align = lanes as i64 * bytes;
        let vector = self.value(format_args!(
            "load <{lanes} x {text}>, {ptr} {address}, align {align}"
        ));
        for (lane, result) in inst.results.iter().enumerate() {
            let value = self.value(format_args!(
                "extractelement <{lanes} x {text}> {vector}, i32 {lane}"
            ));
            self.define(*result, value, elem, signed);
        }
        Ok(())
    }

    /// One store: of one element, or of a tuple's registers as one vector.
    fn store(&mut self, inst: &Inst, space: Space) -> Result<()> {
        let [target, value] = &inst.operands[..] else {
            return Err(self.refuse(inst));
        };
        let address = self.address(inst, target, space)?;
        let (elem, ptr) = (Self::stored(inst.ty), pointer(space));
        let (text, bytes) = (elem.text(), inst.ty.bytes());
        let Operand::Tuple(regs) = value else {
            let value = self.operand(inst, value, elem, false)?;
            self.line(format_args!(
                "store {text} {value}, {ptr} {address}, align {bytes}"
            ));
            return Ok(());
        };
        let lanes = regs.len();
        let mut vector = "poison".to_string();
        for (lane, reg) in regs.iter().enumerate() {
            let value = self.read(*reg);
            let value = self.fit(value, held(reg.bank), elem, false);
            vector = self.value(format_args!(
                "insertelement <{lanes} x {text}> {vector}, {text} {value}, i32 {lane}"
            ));
        }
        let align = lanes as i64 * bytes;
        self.line(format_args!(
            "store <{lanes} x {text}> {vector}, {ptr} {address}, align {align}"
        ));
        Ok(())
    }
}
