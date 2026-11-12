//! The printer: a module as PTX text. Layout only; every decision was made before it runs.

use std::fmt::Write as _;

use crate::inst::{Base, Inst, Item, Label, Module, Opcode, Operand, ParamDecl, Reg};

/// The text of `module`, as ptxas reads it. ASCII only: ptxas refuses any other byte, in a comment too.
pub fn print(module: &Module) -> String {
    let mut out = String::new();
    let entry = &module.entry;
    let _ = writeln!(out, ".version {}.{}", module.version.0, module.version.1);
    let _ = writeln!(out, ".target {}", module.target);
    out.push_str(".address_size 64\n\n");
    for (index, shared) in module.shared.iter().enumerate() {
        let name = symbol(entry, Base::Shared(index as u32));
        let _ = match shared.bytes {
            Some(bytes) => writeln!(out, ".shared .align {} .b8 {name}[{bytes}];", shared.align),
            None => writeln!(out, ".extern .shared .align {} .b8 {name}[];", shared.align),
        };
    }
    let _ = writeln!(out, ".visible .entry {entry}(");
    for (index, param) in module.params.iter().enumerate() {
        let name = symbol(entry, Base::Param(index as u32));
        let last = index + 1 == module.params.len();
        let _ = match param {
            ParamDecl::Pointer { align } => write!(out, "\t.param .u64 .ptr .align {align} {name}"),
            ParamDecl::U32 => write!(out, "\t.param .u32 {name}"),
        };
        out.push_str(if last { "\n" } else { ",\n" });
    }
    out.push_str(")\n");
    let _ = writeln!(out, ".maxntid {}, 1, 1", module.max_threads);
    out.push_str("{\n");
    for (bank, count) in &module.registers {
        if *count > 0 {
            let _ = writeln!(
                out,
                "\t.reg .{} %{}<{count}>;",
                bank.declared(),
                bank.prefix()
            );
        }
    }
    for item in &module.items {
        match item {
            Item::Inst(inst) => {
                let _ = writeln!(out, "\t{};", instruction(entry, inst));
            }
            Item::Label(label) => {
                let _ = writeln!(out, "{}:", label_text(*label));
            }
            Item::Comment(text) => {
                let _ = writeln!(out, "\t// {}", ascii(text));
            }
        }
    }
    out.push_str("}\n");
    out
}

/// `text` with every byte ptxas would refuse replaced.
fn ascii(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii() && !c.is_ascii_control() {
                c
            } else {
                '?'
            }
        })
        .collect()
}

fn label_text(label: Label) -> String {
    format!("L{}", label.0)
}

fn register(reg: Reg) -> String {
    format!("%{}{}", reg.bank.prefix(), reg.number)
}

/// The name of a module symbol: the entry's name and the symbol's position.
fn symbol(entry: &str, base: Base) -> String {
    match base {
        Base::Reg(reg) => register(reg),
        Base::Param(index) => format!("{entry}_param_{index}"),
        Base::Shared(index) => format!("{entry}_shared_{index}"),
    }
}

/// The opcode with its modifiers, in the order PTX spells them: the comparison or the multiply's half, the rounding
/// mode, the flush-to-zero flag, the state space and its cache modifier, the vector width, the type, and for a
/// conversion the type it converts from.
fn mnemonic(inst: &Inst) -> String {
    let mut parts: Vec<&str> = Vec::new();
    let ty = inst.ty.text();
    // What stands between the float modifiers and the type: a state space and, for a vector access, its width.
    let mut memory: Vec<&str> = Vec::new();
    match inst.opcode {
        Opcode::Mov => parts.push("mov"),
        Opcode::Add => parts.push("add"),
        Opcode::Sub => parts.push("sub"),
        Opcode::Mul if inst.ty.is_float() => parts.push("mul"),
        Opcode::Mul => parts.extend(["mul", "lo"]),
        Opcode::MulHi => parts.extend(["mul", "hi"]),
        Opcode::MulWide => parts.extend(["mul", "wide"]),
        Opcode::Div => parts.push("div"),
        Opcode::Rem => parts.push("rem"),
        Opcode::Neg => parts.push("neg"),
        Opcode::Abs => parts.push("abs"),
        Opcode::Min => parts.push("min"),
        Opcode::Max => parts.push("max"),
        Opcode::Shl => parts.push("shl"),
        Opcode::Shr => parts.push("shr"),
        Opcode::And => parts.push("and"),
        Opcode::Or => parts.push("or"),
        Opcode::Xor => parts.push("xor"),
        Opcode::Not => parts.push("not"),
        Opcode::Popc => parts.push("popc"),
        Opcode::Fma => parts.push("fma"),
        Opcode::Rcp => parts.push("rcp"),
        Opcode::Sqrt => parts.push("sqrt"),
        Opcode::Setp(cmp) => parts.extend(["setp", cmp.text()]),
        Opcode::Selp => parts.push("selp"),
        Opcode::Cvt(_) => parts.push("cvt"),
        Opcode::Ld(space) => {
            parts.push("ld");
            memory.push(space.text());
        }
        Opcode::LdConst => {
            parts.push("ld");
            memory.extend(["global", "nc"]);
        }
        Opcode::St(space) => {
            parts.push("st");
            memory.push(space.text());
        }
        Opcode::AtomAdd(space) => parts.extend(["atom", space.text(), "add"]),
        Opcode::Activemask => parts.push("activemask"),
        Opcode::Shfl(shuffle) => parts.extend(["shfl", "sync", shuffle.text()]),
        Opcode::CpAsync(cache) => return format!("cp.async.{}.shared.global", cache.text()),
        Opcode::BarSync => return "bar.sync".into(),
        Opcode::BarWarpSync => return "bar.warp.sync".into(),
        Opcode::CpAsyncCommit => return "cp.async.commit_group".into(),
        Opcode::CpAsyncWait => return "cp.async.wait_group".into(),
        Opcode::GridWait => return "griddepcontrol.wait".into(),
        Opcode::Bra => return "bra".into(),
        Opcode::Ret => return "ret".into(),
    }
    if let Some(round) = inst.round {
        parts.push(round.text());
    }
    if inst.ftz {
        parts.push("ftz");
    }
    parts.extend(memory);
    // A load or a store of several registers is one vector access of that many elements.
    let lanes = match inst.opcode {
        opcode if opcode.is_load() => inst.results.len(),
        Opcode::St(_) => inst.operands.iter().map(Operand::lanes).max().unwrap_or(1),
        _ => 1,
    };
    let vector = format!("v{lanes}");
    if lanes > 1 {
        parts.push(&vector);
    }
    parts.push(ty);
    if let Opcode::Cvt(from) = inst.opcode {
        parts.push(from.text());
    }
    parts.join(".")
}

fn operand(entry: &str, operand: &Operand) -> String {
    match operand {
        Operand::Reg(reg) => register(*reg),
        Operand::Int(value) => value.to_string(),
        Operand::F32(bits) => format!("0f{bits:08X}"),
        Operand::Addr(base, 0) => format!("[{}]", symbol(entry, *base)),
        Operand::Addr(base, offset) => format!("[{}+{offset}]", symbol(entry, *base)),
        Operand::Tuple(regs) => tuple(regs),
        Operand::Shared(index) => symbol(entry, Base::Shared(*index)),
        Operand::Special(special) => special.text().into(),
        Operand::Label(label) => label_text(*label),
    }
}

fn tuple(regs: &[Reg]) -> String {
    let regs: Vec<String> = regs.iter().copied().map(register).collect();
    format!("{{{}}}", regs.join(", "))
}

/// One instruction without its indent and its `;`: the guard, the mnemonic, the results, the operands.
fn instruction(entry: &str, inst: &Inst) -> String {
    let mut out = String::new();
    if let Some(guard) = inst.guard {
        let _ = write!(
            out,
            "@{}{} ",
            if guard.negated { "!" } else { "" },
            register(guard.pred)
        );
    }
    out.push_str(&mnemonic(inst));
    let results = match &inst.results[..] {
        [] => None,
        [one] => Some(register(*one)),
        several => Some(tuple(several)),
    };
    let operands: Vec<String> = results
        .into_iter()
        .chain(inst.operands.iter().map(|o| operand(entry, o)))
        .collect();
    if !operands.is_empty() {
        out.push(' ');
        out.push_str(&operands.join(", "));
    }
    out
}

/// The text of one instruction, for an error message or a test.
pub fn instruction_text(inst: &Inst) -> String {
    instruction("k", inst)
}
