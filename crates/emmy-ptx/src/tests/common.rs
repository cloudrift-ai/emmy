//! What the rule tests share: a kernel from the text form, lowered, assembled by ptxas at `-O0` and at its default
//! level wherever it is installed, and listed one instruction a line.
#![allow(dead_code)]

use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::inst::{Item, Module, Opcode, Space};
use crate::print::{instruction_text, print};
use crate::{Interface, Kernel, Param, ParamKind, Precision, Target, lower};
use emmy_ir::body::Body;
use emmy_ir::dtype::DType;
use emmy_ir::text::parse_body;

pub type Params<'a> = &'a [(&'a str, ParamKind, Option<DType>)];

pub const SM_120: Target = Target { sm: 120 };
pub const EXACT: Precision = Precision {
    fuse: false,
    ftz: false,
};
pub const FUSED: Precision = Precision {
    fuse: true,
    ftz: false,
};

/// One f32 input `a` and one f32 output `o`.
pub const A_O: Params = &[
    ("a", ParamKind::Input, Some(DType::F32)),
    ("o", ParamKind::Output, Some(DType::F32)),
];

/// The alignment a test's launch gives every buffer.
pub const BUFFER_ALIGN: u32 = 16;

pub fn interface(params: Params) -> Interface {
    Interface {
        params: params
            .iter()
            .map(|(name, kind, dtype)| Param {
                name: (*name).into(),
                kind: *kind,
                dtype: *dtype,
                page: None,
                align: BUFFER_ALIGN,
                shape: Vec::new(),
                pair_table: false,
            })
            .collect(),
        ..Interface::default()
    }
}

/// Assemble `ptx` for `target` at `-O0` and at the default level; nothing where ptxas is not installed.
pub fn assemble(ptx: &str, target: &str) {
    static FILES: AtomicU32 = AtomicU32::new(0);
    if Command::new("ptxas").arg("--version").output().is_err() {
        return;
    }
    let stem = format!(
        "{}/k{}_{}",
        std::env::temp_dir().display(),
        std::process::id(),
        FILES.fetch_add(1, Ordering::Relaxed)
    );
    let source = format!("{stem}.ptx");
    std::fs::write(&source, ptx).unwrap();
    for level in ["-O0", "-O3"] {
        let output = Command::new("ptxas")
            .args([level, &format!("-arch={target}"), "-m64", &source, "-o"])
            .arg(format!("{stem}.cubin"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "ptxas {level} refused:\n{}\n{ptx}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(format!("{stem}.cubin"));
}

pub fn parse(body: &str) -> Body {
    parse_body(body).unwrap()
}

/// The module of the kernel `body` is under `interface`, assembled.
pub fn lowered_in(
    body: &str,
    interface: &Interface,
    target: Target,
    precision: Precision,
) -> emmy_ir::Result<Module> {
    let body = parse(body);
    let kernel = Kernel {
        name: "k",
        body: &body,
        interface,
    };
    let module = lower(&kernel, target, precision)?;
    assemble(&print(&module), &module.target);
    Ok(module)
}

/// The module of the kernel `body` is in blocks of 256 threads.
pub fn lowered(
    body: &str,
    params: Params,
    target: Target,
    precision: Precision,
) -> emmy_ir::Result<Module> {
    lowered_in(body, &interface(params), target, precision)
}

/// The code of a module, one line per instruction and label, without the wait on the grid ahead, the parameter
/// loads and the closing `ret`.
pub fn listing(module: &Module) -> Vec<String> {
    module
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Inst(inst)
                if matches!(
                    inst.opcode,
                    Opcode::Ld(Space::Param) | Opcode::GridWait | Opcode::Ret
                ) =>
            {
                None
            }
            Item::Inst(inst) => Some(instruction_text(inst)),
            Item::Label(label) => Some(format!("L{}:", label.0)),
            Item::Comment(_) => None,
            Item::NoUnroll => Some(".pragma \"nounroll\"".into()),
        })
        .collect()
}

pub fn code(body: &str, params: Params, precision: Precision) -> Vec<String> {
    listing(&lowered(body, params, SM_120, precision).unwrap())
}

/// The message a kernel is refused with.
pub fn refusal(body: &str, params: Params) -> String {
    lowered(body, params, SM_120, EXACT)
        .unwrap_err()
        .to_string()
}

/// The statements of `body`, written eight spaces deep, as the body of a statement at the top.
pub fn indented(body: &str) -> String {
    let lines: Vec<String> = body
        .trim_matches('\n')
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| format!("    {}", line.strip_prefix("        ").unwrap_or(line)))
        .collect();
    lines.join("\n")
}
