//! Emmy's execution plan: the JSON contract the compiler writes, read into typed form and
//! validated, plus the memory layout the runtime derives from it.
//!
//! The plan carries a tiny expression grammar — an `int` literal, a `"name"` variable, or
//! `[op, lhs, rhs]` with `op` in `+ - * / // %` — for symbolic shapes, grid factors and runtime
//! constants. Everything else is plain data.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SUPPORTED_PLAN_FORMATS: [u32; 3] = [1, 2, 3];
const PACK_FORMAT: u32 = 1;
const STANDALONE_FORMAT: u32 = 1;
const LAUNCH_DIMENSIONS: usize = 3;
/// Byte alignment of every scratch buffer inside the slab.
pub const SLAB_ALIGN: usize = 256;
/// The region every scratch buffer is packed into.
pub const SCRATCH_REGION: &str = "scratch";

/// Every symbolic axis name bound to its runtime extent.
pub type Env = BTreeMap<String, i64>;

/// An expression of the plan grammar.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Var(String),
    Bin(String, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Num {
    I(i64),
    F(f64),
}

impl Num {
    fn f(self) -> f64 {
        match self {
            Num::I(v) => v as f64,
            Num::F(v) => v,
        }
    }
}

fn floor_div(a: i64, b: i64) -> Result<i64> {
    ensure!(b != 0, "division by zero in a plan expression");
    let q = a / b;
    Ok(if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    })
}

fn floor_mod(a: i64, b: i64) -> Result<i64> {
    ensure!(b != 0, "modulo by zero in a plan expression");
    Ok(((a % b) + b) % b)
}

impl Expr {
    pub fn from_value(value: &Value) -> Result<Expr> {
        match value {
            Value::Number(n) if n.is_i64() => Ok(Expr::Int(n.as_i64().unwrap())),
            Value::Number(n) if n.is_u64() => Ok(Expr::Int(i64::try_from(n.as_u64().unwrap())?)),
            Value::Number(n) => Ok(Expr::Float(n.as_f64().context("invalid literal")?)),
            Value::String(name) => Ok(Expr::Var(name.clone())),
            Value::Array(items) if items.len() == 3 => {
                let op = items[0].as_str().context("malformed expression operator")?;
                ensure!(
                    ["+", "-", "*", "/", "//", "%"].contains(&op),
                    "unknown expression operator {op}"
                );
                Ok(Expr::Bin(
                    op.to_owned(),
                    Box::new(Expr::from_value(&items[1])?),
                    Box::new(Expr::from_value(&items[2])?),
                ))
            }
            other => bail!("malformed plan expression {other}"),
        }
    }

    fn num(&self, env: &Env) -> Result<Num> {
        Ok(match self {
            Expr::Int(v) => Num::I(*v),
            Expr::Float(v) => Num::F(*v),
            Expr::Var(name) => Num::I(
                *env.get(name)
                    .with_context(|| format!("unbound symbolic axis {name:?}"))?,
            ),
            Expr::Bin(op, l, r) => {
                let (l, r) = (l.num(env)?, r.num(env)?);
                match (op.as_str(), l, r) {
                    ("+", Num::I(a), Num::I(b)) => Num::I(a + b),
                    ("-", Num::I(a), Num::I(b)) => Num::I(a - b),
                    ("*", Num::I(a), Num::I(b)) => Num::I(a * b),
                    ("/" | "//", Num::I(a), Num::I(b)) => Num::I(floor_div(a, b)?),
                    ("%", Num::I(a), Num::I(b)) => Num::I(floor_mod(a, b)?),
                    ("+", a, b) => Num::F(a.f() + b.f()),
                    ("-", a, b) => Num::F(a.f() - b.f()),
                    ("*", a, b) => Num::F(a.f() * b.f()),
                    ("/" | "//", a, b) => Num::F((a.f() / b.f()).floor()),
                    ("%", a, b) => Num::F(a.f().rem_euclid(b.f())),
                    (op, _, _) => bail!("unknown expression operator {op}"),
                }
            }
        })
    }

    /// The expression's integer value under `env`.
    pub fn eval(&self, env: &Env) -> Result<i64> {
        match self.num(env)? {
            Num::I(v) => Ok(v),
            Num::F(v) => bail!("expression evaluates to a non-integer {v}"),
        }
    }

    /// The expression's value as a float — how a runtime constant fills its buffer.
    pub fn eval_f64(&self, env: &Env) -> Result<f64> {
        Ok(self.num(env)?.f())
    }

    pub fn is_static(&self) -> bool {
        match self {
            Expr::Int(_) | Expr::Float(_) => true,
            Expr::Var(_) => false,
            Expr::Bin(_, l, r) => l.is_static() && r.is_static(),
        }
    }
}

/// Bytes per stored element of a plan dtype. Shapes count stored elements: a packed fp4 pair
/// is one byte holding two values.
pub fn dtype_bytes(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "f16" | "bf16" | "i16" | "u16" => 2,
        "f32" | "i32" | "u32" | "f16x2" => 4,
        "f64" | "i64" | "u64" => 8,
        "i8" | "u8" | "bool" | "f8e4m3" | "f8e5m2" | "f4e2m1x2" => 1,
        other => bail!("unsupported dtype: {other}"),
    })
}

#[derive(Debug, Clone)]
pub struct Buffer {
    pub name: String,
    pub shape: Vec<Expr>,
    pub dtype: String,
    pub role: String,
}

impl Buffer {
    pub fn is_symbolic(&self) -> bool {
        self.shape.iter().any(|d| !d.is_static())
    }

    pub fn resolve_shape(&self, env: &Env) -> Result<Vec<i64>> {
        self.shape
            .iter()
            .map(|d| {
                let n = d.eval(env)?;
                ensure!(n >= 0, "negative extent in buffer {}", self.name);
                Ok(n)
            })
            .collect()
    }

    pub fn byte_len(&self, env: &Env) -> Result<usize> {
        let mut bytes = dtype_bytes(&self.dtype)?;
        for n in self.resolve_shape(env)? {
            bytes = bytes
                .checked_mul(usize::try_from(n)?)
                .context("buffer size overflow")?;
        }
        Ok(bytes)
    }

    /// The shape when every extent is a literal, `None` for a symbolic buffer.
    pub fn static_shape(&self) -> Option<Vec<i64>> {
        self.resolve_shape(&Env::new()).ok()
    }
}

#[derive(Debug, Clone)]
pub struct Indirect {
    pub arg: String,
    pub table: String,
    pub sel: String,
    pub slot: i64,
}

#[derive(Debug, Clone)]
pub struct Tma {
    pub name: String,
    pub src_buf: String,
    pub box_extents: Vec<u32>,
    pub swizzle: String,
}

#[derive(Debug, Clone)]
pub struct Launch {
    pub node_id: String,
    pub kernel: String,
    pub args: Vec<String>,
    pub grid: Vec<Vec<Expr>>,
    pub block: Vec<Vec<Expr>>,
    pub smem: u32,
    pub zero_outputs: Vec<String>,
    pub zero_prologues: Vec<String>,
    pub writes: Vec<String>,
    pub indirect: Vec<Indirect>,
    pub runtime_args: Vec<String>,
    pub serial: Vec<(String, i64)>,
    pub tma: Vec<Tma>,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub binary_key: Option<String>,
    pub arch_specific: bool,
    pub source: Option<String>,
}

/// One buffer's place in the layout: a byte range inside a named region.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub region: String,
    pub offset: usize,
    pub bytes: usize,
    pub shape: Vec<i64>,
}

/// Where every buffer lives for one symbol environment: named regions the host (or the runtime)
/// allocates, and one placement per buffer. Every non-scratch buffer owns its own region,
/// `"{role}:{name}"`; scratch buffers pack into the one `scratch` region by liveness.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Layout {
    pub regions: BTreeMap<String, usize>,
    pub buffers: BTreeMap<String, Placement>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub format: u32,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub buffers: Vec<Buffer>,
    pub constants: BTreeMap<String, f64>,
    pub runtime_constants: BTreeMap<String, Expr>,
    pub launches: Vec<Launch>,
    pub kernels: BTreeMap<String, Kernel>,
    pub weights: BTreeSet<String>,
    /// Symbolic axis name → the input buffer and dimension it is read from.
    pub bindings: BTreeMap<String, (String, usize)>,
    pub hints: BTreeMap<String, i64>,
    pub caps: BTreeMap<String, i64>,
}

// ---------------------------------------------------------------------------------------------
// Raw JSON form
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBuffer {
    name: String,
    shape: Vec<Value>,
    dtype: String,
    role: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawKernel {
    #[serde(default)]
    binary_key: Option<String>,
    #[serde(default)]
    arch_specific: bool,
    #[serde(default)]
    source: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTma {
    name: String,
    src_buf: String,
    box_extents: Vec<u32>,
    #[serde(default = "default_swizzle")]
    swizzle: String,
}

fn default_swizzle() -> String {
    "NONE".to_owned()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCuda {
    #[serde(default)]
    tma: Vec<RawTma>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLaunch {
    node_id: String,
    kernel: String,
    args: Vec<String>,
    grid: Vec<Vec<Value>>,
    block: Vec<Vec<Value>>,
    smem: u32,
    #[serde(default)]
    zero_outputs: Vec<String>,
    #[serde(default)]
    zero_prologues: Vec<String>,
    #[serde(default)]
    writes: Vec<String>,
    #[serde(default)]
    indirect: Vec<(String, String, String, i64)>,
    #[serde(default)]
    runtime_args: Vec<String>,
    #[serde(default)]
    serial: Vec<(String, i64)>,
    #[serde(default)]
    cuda: Option<RawCuda>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSymbols {
    #[serde(default)]
    bindings: BTreeMap<String, (String, usize)>,
    #[serde(default)]
    hints: BTreeMap<String, i64>,
    #[serde(default)]
    caps: BTreeMap<String, i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPlan {
    format: u32,
    backend: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    buffers: Vec<RawBuffer>,
    #[serde(default)]
    constants: BTreeMap<String, f64>,
    #[serde(default)]
    runtime_constants: BTreeMap<String, Value>,
    launches: Vec<RawLaunch>,
    kernels: BTreeMap<String, RawKernel>,
    #[serde(default)]
    weights: BTreeMap<String, Value>,
    #[serde(default)]
    symbols: Option<RawSymbols>,
}

fn factors(axes: &[Vec<Value>]) -> Result<Vec<Vec<Expr>>> {
    ensure!(
        axes.len() == LAUNCH_DIMENSIONS,
        "launch dimensions must have three axes"
    );
    axes.iter()
        .map(|axis| axis.iter().map(Expr::from_value).collect())
        .collect()
}

/// Multiply one launch axis' factors under `env` into a driver dimension.
pub fn dimensions(axes: &[Vec<Expr>], env: &Env) -> Result<(u32, u32, u32)> {
    let mut dims = [1u32; LAUNCH_DIMENSIONS];
    for (out, axis) in dims.iter_mut().zip(axes) {
        for factor in axis {
            let value = factor.eval(env)?;
            ensure!(value > 0, "launch dimensions must be positive");
            *out = out
                .checked_mul(u32::try_from(value)?)
                .context("launch dimension overflow")?;
        }
    }
    Ok((dims[0], dims[1], dims[2]))
}

impl Program {
    pub fn parse(json: &str) -> Result<Self> {
        let raw: RawPlan = serde_json::from_str(json)?;
        ensure!(
            SUPPORTED_PLAN_FORMATS.contains(&raw.format),
            "unsupported plan format {}",
            raw.format
        );
        ensure!(raw.backend == "cuda", "only CUDA plans are supported");
        let symbols = raw.symbols.unwrap_or(RawSymbols {
            bindings: BTreeMap::new(),
            hints: BTreeMap::new(),
            caps: BTreeMap::new(),
        });
        let program = Program {
            format: raw.format,
            inputs: raw.inputs,
            outputs: raw.outputs,
            buffers: raw
                .buffers
                .into_iter()
                .map(|b| {
                    Ok(Buffer {
                        shape: b
                            .shape
                            .iter()
                            .map(Expr::from_value)
                            .collect::<Result<_>>()?,
                        name: b.name,
                        dtype: b.dtype,
                        role: b.role,
                    })
                })
                .collect::<Result<_>>()?,
            constants: raw.constants,
            runtime_constants: raw
                .runtime_constants
                .iter()
                .map(|(k, v)| Ok((k.clone(), Expr::from_value(v)?)))
                .collect::<Result<_>>()?,
            launches: raw
                .launches
                .into_iter()
                .map(|l| {
                    Ok(Launch {
                        grid: factors(&l.grid)?,
                        block: factors(&l.block)?,
                        node_id: l.node_id,
                        kernel: l.kernel,
                        args: l.args,
                        smem: l.smem,
                        zero_outputs: l.zero_outputs,
                        zero_prologues: l.zero_prologues,
                        writes: l.writes,
                        indirect: l
                            .indirect
                            .into_iter()
                            .map(|(arg, table, sel, slot)| Indirect {
                                arg,
                                table,
                                sel,
                                slot,
                            })
                            .collect(),
                        runtime_args: l.runtime_args,
                        serial: l.serial,
                        tma: l
                            .cuda
                            .map(|c| c.tma)
                            .unwrap_or_default()
                            .into_iter()
                            .map(|t| Tma {
                                name: t.name,
                                src_buf: t.src_buf,
                                box_extents: t.box_extents,
                                swizzle: t.swizzle,
                            })
                            .collect(),
                    })
                })
                .collect::<Result<_>>()?,
            kernels: raw
                .kernels
                .into_iter()
                .map(|(name, k)| {
                    (
                        name,
                        Kernel {
                            binary_key: k.binary_key,
                            arch_specific: k.arch_specific,
                            source: k.source,
                        },
                    )
                })
                .collect(),
            weights: raw.weights.into_keys().collect(),
            bindings: symbols.bindings,
            hints: symbols.hints,
            caps: symbols.caps,
        };
        program.validate()?;
        Ok(program)
    }

    fn validate(&self) -> Result<()> {
        let mut names = BTreeSet::new();
        for buffer in &self.buffers {
            ensure!(
                names.insert(buffer.name.as_str()),
                "duplicate buffer {}",
                buffer.name
            );
            ensure!(
                ["input", "constant", "output", "scratch"].contains(&buffer.role.as_str()),
                "invalid buffer role for {}",
                buffer.name
            );
            dtype_bytes(&buffer.dtype)?;
        }
        for name in self
            .inputs
            .iter()
            .chain(&self.outputs)
            .chain(self.constants.keys())
            .chain(self.runtime_constants.keys())
            .chain(&self.weights)
        {
            ensure!(names.contains(name.as_str()), "unknown buffer {name}");
        }
        for (list, role) in [(&self.inputs, "input"), (&self.outputs, "output")] {
            ensure!(
                list.iter().collect::<BTreeSet<_>>().len() == list.len(),
                "duplicate {role}"
            );
            for name in list {
                ensure!(
                    self.buffers
                        .iter()
                        .any(|b| &b.name == name && b.role == role),
                    "invalid {role} buffer {name}"
                );
            }
        }
        for (axis, (buffer, dim)) in &self.bindings {
            let b = self.buffer(buffer)?;
            ensure!(
                *dim < b.shape.len(),
                "symbolic axis {axis} reads past the rank of {buffer}"
            );
        }
        for launch in &self.launches {
            ensure!(
                self.kernels.contains_key(&launch.kernel),
                "unknown kernel {}",
                launch.kernel
            );
            let descriptors: BTreeSet<&str> = launch.tma.iter().map(|t| t.name.as_str()).collect();
            for name in &launch.args {
                ensure!(
                    names.contains(name.as_str()) || descriptors.contains(name.as_str()),
                    "unknown launch buffer {name}"
                );
            }
            for name in launch
                .zero_outputs
                .iter()
                .chain(&launch.zero_prologues)
                .chain(&launch.writes)
            {
                ensure!(
                    names.contains(name.as_str()),
                    "unknown launch buffer {name}"
                );
            }
            // An indirect operand's table and selector are operands the host binds by address
            // at run time; the plan need not declare them as buffers, and an entry whose arg
            // this launch does not take is simply unused.
            for t in &launch.tma {
                ensure!(
                    names.contains(t.src_buf.as_str()),
                    "unknown TMA source {}",
                    t.src_buf
                );
                ensure!(
                    !t.box_extents.is_empty() && t.box_extents.len() <= 5,
                    "TMA rank must be 1..5"
                );
                ensure!(
                    t.box_extents.iter().all(|&b| (1..=256).contains(&b)),
                    "TMA box extents must be within 1..256"
                );
                ensure!(
                    ["NONE", "B32", "B64", "B128"].contains(&t.swizzle.as_str()),
                    "unknown TMA swizzle {}",
                    t.swizzle
                );
            }
            for (_, extent) in &launch.serial {
                ensure!(*extent >= 0, "negative serial extent");
            }
        }
        Ok(())
    }

    pub fn buffer(&self, name: &str) -> Result<&Buffer> {
        self.buffers
            .iter()
            .find(|b| b.name == name)
            .with_context(|| format!("unknown buffer {name}"))
    }

    /// The environment a program runs at when nothing binds its axes: every hint.
    pub fn default_env(&self) -> Env {
        self.hints.clone()
    }

    /// Every buffer's placement under `env`. Scratch buffers share one slab: a buffer is live
    /// from the launch that first writes it to the last launch that reads it, and buffers whose
    /// intervals do not overlap share bytes (largest first, deterministic).
    pub fn layout(&self, env: &Env) -> Result<Layout> {
        let mut layout = Layout::default();
        let scratch: BTreeSet<&str> = self
            .buffers
            .iter()
            .filter(|b| b.role == "scratch")
            .map(|b| b.name.as_str())
            .collect();
        for buffer in &self.buffers {
            if buffer.role == "scratch" {
                continue;
            }
            let bytes = buffer.byte_len(env)?;
            let region = format!("{}:{}", buffer.role, buffer.name);
            layout.regions.insert(region.clone(), bytes.max(1));
            layout.buffers.insert(
                buffer.name.clone(),
                Placement {
                    region,
                    offset: 0,
                    bytes,
                    shape: buffer.resolve_shape(env)?,
                },
            );
        }
        if scratch.is_empty() {
            return Ok(layout);
        }
        let intervals = self.live_intervals(&scratch)?;
        let mut sizes = BTreeMap::new();
        for name in &scratch {
            sizes.insert(*name, self.buffer(name)?.byte_len(env)?);
        }
        let (offsets, total) = plan_offsets(&intervals, &sizes);
        layout
            .regions
            .insert(SCRATCH_REGION.to_owned(), total.max(1));
        for name in &scratch {
            let buffer = self.buffer(name)?;
            layout.buffers.insert(
                (*name).to_owned(),
                Placement {
                    region: SCRATCH_REGION.to_owned(),
                    offset: offsets[*name],
                    bytes: sizes[*name],
                    shape: buffer.resolve_shape(env)?,
                },
            );
        }
        Ok(layout)
    }

    /// Per scratch buffer, its half-open live interval `[first_write, last_read + 1)` over the
    /// launch order. A launch reads a buffer named in its args or as a TMA source; a serial
    /// launch also reads what its own earlier steps stored. The `+ 1` is load-bearing: a
    /// launch's output overlaps its inputs, so an output never aliases its own input.
    fn live_intervals(&self, scratch: &BTreeSet<&str>) -> Result<BTreeMap<String, (usize, usize)>> {
        let mut first_write: BTreeMap<&str, usize> = BTreeMap::new();
        let mut last_read: BTreeMap<&str, usize> = BTreeMap::new();
        for (i, launch) in self.launches.iter().enumerate() {
            let writes: Vec<&str> = if launch.writes.is_empty() {
                vec![launch.node_id.as_str()]
            } else {
                launch.writes.iter().map(String::as_str).collect()
            };
            for w in writes.iter().chain(
                launch
                    .zero_outputs
                    .iter()
                    .chain(&launch.zero_prologues)
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .iter(),
            ) {
                if scratch.contains(w) {
                    first_write.entry(w).or_insert(i);
                }
            }
            let reads = launch
                .args
                .iter()
                .map(String::as_str)
                .chain(launch.tma.iter().map(|t| t.src_buf.as_str()));
            for name in reads {
                if scratch.contains(name) && (!writes.contains(&name) || !launch.serial.is_empty())
                {
                    last_read.insert(name, i);
                }
            }
        }
        let mut intervals = BTreeMap::new();
        for name in scratch {
            let first = *first_write
                .get(name)
                .with_context(|| format!("scratch buffer {name:?} has no producing launch"))?;
            let last = *last_read.get(name).with_context(|| {
                format!("scratch buffer {name:?} has no consuming launch (dead scratch)")
            })?;
            intervals.insert((*name).to_owned(), (first, last + 1));
        }
        Ok(intervals)
    }
}

/// Greedy-by-size slab packing: place the largest buffers first, each at the lowest aligned
/// offset that collides with no already-placed buffer whose live interval overlaps.
fn plan_offsets(
    intervals: &BTreeMap<String, (usize, usize)>,
    sizes: &BTreeMap<&str, usize>,
) -> (BTreeMap<String, usize>, usize) {
    let mut order: Vec<&String> = intervals.keys().collect();
    order.sort_by_key(|n| (std::cmp::Reverse(sizes[n.as_str()]), (*n).clone()));
    let mut placed: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut offsets = BTreeMap::new();
    let mut total = 0;
    for name in order {
        let size = sizes[name.as_str()];
        let (first, free) = intervals[name];
        let mut occupied: Vec<(usize, usize)> = placed
            .iter()
            .filter(|&&(_, _, f, fr)| first < fr && f < free)
            .map(|&(o, s, _, _)| (o, o + s))
            .collect();
        occupied.sort_unstable();
        let mut off = 0;
        for (lo, hi) in occupied {
            if off + size <= lo {
                break;
            }
            if off < hi {
                off = hi.div_ceil(SLAB_ALIGN) * SLAB_ALIGN;
            }
        }
        offsets.insert(name.clone(), off);
        placed.push((off, size, first, free));
        total = total.max(off + size);
    }
    (offsets, total)
}

pub struct Artifact {
    pub program: Arc<Program>,
    /// The pack's recorded target; `None` for a program the host compiled for the live device.
    pub arch: Option<String>,
    pub bindings: BTreeMap<String, Vec<u8>>,
    pub binaries: BTreeMap<String, PathBuf>,
}

fn member(root: &Path, name: &str) -> Result<PathBuf> {
    let path = root.join(name).canonicalize()?;
    ensure!(
        path.starts_with(root) && path.is_file(),
        "artifact member escapes bundle: {name}"
    );
    Ok(path)
}

impl Artifact {
    /// A program the host compiled itself: the plan's JSON form, the bound input and constant
    /// bytes, and one cubin path per kernel. Nothing is read from disk but the cubins.
    pub fn new(
        plan: &str,
        bindings: BTreeMap<String, Vec<u8>>,
        binaries: BTreeMap<String, PathBuf>,
    ) -> Result<Self> {
        Self::from_program(Arc::new(Program::parse(plan)?), bindings, binaries)
    }

    /// The same, from an already parsed program.
    pub fn from_program(
        program: Arc<Program>,
        bindings: BTreeMap<String, Vec<u8>>,
        binaries: BTreeMap<String, PathBuf>,
    ) -> Result<Self> {
        for (name, path) in &binaries {
            ensure!(
                program.kernels.contains_key(name),
                "cubin for unknown kernel {name}"
            );
            ensure!(
                path.is_file(),
                "missing cubin for kernel {name}: {}",
                path.display()
            );
        }
        for name in program.kernels.keys() {
            ensure!(binaries.contains_key(name), "no cubin for kernel {name}");
        }
        for name in bindings.keys() {
            ensure!(
                program
                    .buffer(name)
                    .map(|b| b.role == "input" || b.role == "constant")?,
                "only inputs and constants may be bound: {name}"
            );
        }
        Ok(Self {
            program,
            arch: None,
            bindings,
            binaries,
        })
    }

    /// A standalone pack: every constant resolved to bytes, cubins bundled, no compiler needed.
    pub fn load(root: &Path, program: &str) -> Result<Self> {
        let root = root.canonicalize()?;
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(member(&root, "manifest.json")?)?)?;
        ensure!(
            manifest["format"] == PACK_FORMAT && manifest["standalone"] == STANDALONE_FORMAT,
            "unsupported standalone pack format"
        );
        let path = manifest["programs"][program]
            .as_str()
            .context("unknown program")?;
        let parsed = Program::parse(&std::fs::read_to_string(member(&root, path)?)?)?;
        let mut bindings = BTreeMap::new();
        for (name, path) in manifest["bindings"][program]
            .as_object()
            .context("missing binding index")?
        {
            ensure!(
                parsed
                    .buffer(name)
                    .map(|b| b.role == "input" || b.role == "constant")?,
                "only inputs and constants may be bound: {name}"
            );
            bindings.insert(
                name.clone(),
                std::fs::read(member(
                    &root,
                    path.as_str().context("invalid binding path")?,
                )?)?,
            );
        }
        for buffer in &parsed.buffers {
            ensure!(
                buffer.role != "constant" || bindings.contains_key(&buffer.name),
                "missing constant bytes: {}",
                buffer.name
            );
        }
        let mut binaries = BTreeMap::new();
        for (name, kernel) in &parsed.kernels {
            let key = kernel.binary_key.as_deref().unwrap_or_default();
            ensure!(
                !key.is_empty() && key.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid cubin key"
            );
            binaries.insert(name.clone(), member(&root, &format!("cubin/{key}.cubin"))?);
        }
        let arch = manifest["environment"]["arch"]
            .as_str()
            .context("missing target architecture")?
            .to_owned();
        Ok(Self {
            program: Arc::new(parsed),
            arch: Some(arch),
            bindings,
            binaries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn example() -> Value {
        json!({
            "format": 1, "backend": "cuda", "inputs": ["x"], "outputs": ["y"],
            "buffers": [
                {"name":"x", "shape":[4], "dtype":"f32", "role":"input"},
                {"name":"y", "shape":[4], "dtype":"f32", "role":"output"},
                {"name":"w", "shape":[1], "dtype":"f32", "role":"constant"}
            ],
            "constants": {"w": 2.0}, "runtime_constants": {}, "weights": {},
            "kernels": {"add": {"binary_key":"ab", "arch_specific":false}},
            "symbols": {"bindings":{},"hints":{},"caps":{}},
            "launches": [{"node_id":"y", "kernel":"add", "args":["x","w","y"],
                "grid":[[1],[1],[1]],"block":[[32],[1],[1]],"smem":0,
                "zero_outputs":[],"runtime_args":[],"cuda":{"tma":[]}}]
        })
    }

    #[test]
    fn plan_parses_and_rejects_bad_references() {
        Program::parse(&example().to_string()).unwrap();
        for (pointer, value) in [
            ("/format", json!(999)),
            ("/buffers/1/name", json!("x")),
            ("/launches/0/args/0", json!("missing")),
            ("/launches/0/kernel", json!("missing")),
            ("/buffers/0/dtype", json!("void")),
            (
                "/launches/0/cuda/tma",
                json!([{"name":"d","src_buf":"nope","box_extents":[8]}]),
            ),
        ] {
            let mut bad = example();
            *bad.pointer_mut(pointer).unwrap() = value;
            assert!(
                Program::parse(&bad.to_string()).is_err(),
                "accepted {pointer}"
            );
        }
        let mut unknown = example();
        unknown["launches"][0]["unrecognized_abi"] = json!(true);
        assert!(Program::parse(&unknown.to_string()).is_err());
    }

    #[test]
    fn expressions_follow_python_integer_semantics() {
        let env: Env = [("n".to_owned(), 70)].into_iter().collect();
        let ceil = Expr::from_value(&json!(["//", ["+", "n", 63], 64])).unwrap();
        assert_eq!(ceil.eval(&env).unwrap(), 2);
        assert_eq!(
            Expr::from_value(&json!(["//", -7, 2]))
                .unwrap()
                .eval(&env)
                .unwrap(),
            -4
        );
        assert_eq!(
            Expr::from_value(&json!(["%", -7, 4]))
                .unwrap()
                .eval(&env)
                .unwrap(),
            1
        );
        assert!(Expr::from_value(&json!("m")).unwrap().eval(&env).is_err());
        assert!(Expr::from_value(&json!(["^", 1, 2])).is_err());
        assert_eq!(
            dimensions(
                &[
                    vec![ceil.clone(), Expr::Int(3)],
                    vec![Expr::Int(4)],
                    vec![Expr::Int(1)]
                ],
                &env
            )
            .unwrap(),
            (6, 4, 1)
        );
        assert!(
            dimensions(
                &[vec![Expr::Int(0)], vec![Expr::Int(1)], vec![Expr::Int(1)]],
                &env
            )
            .is_err()
        );
    }

    #[test]
    fn buffer_size_checks_dtype_shape_and_overflow() {
        let mut buffer = Buffer {
            name: "x".into(),
            shape: vec![Expr::Int(2), Expr::Int(3)],
            dtype: "f16".into(),
            role: "input".into(),
        };
        assert_eq!(buffer.byte_len(&Env::new()).unwrap(), 12);
        buffer.shape = vec![Expr::Int(i64::MAX), Expr::Int(i64::MAX)];
        assert!(buffer.byte_len(&Env::new()).is_err());
        buffer.shape = vec![Expr::Var("n".into())];
        assert!(buffer.is_symbolic() && buffer.static_shape().is_none());
    }

    #[test]
    fn scratch_buffers_share_the_slab_by_liveness() {
        // a -> t1 -> t2 -> t3 -> y. t1 is live through the launch that reads it to produce t2,
        // so t2 cannot alias it (an output never aliases its own input); t3 is born after t1
        // died and reuses t1's bytes.
        let plan = json!({
            "format": 1, "backend": "cuda", "inputs": ["a"], "outputs": ["y"],
            "buffers": [
                {"name":"a", "shape":[1024], "dtype":"f32", "role":"input"},
                {"name":"t1", "shape":[1024], "dtype":"f32", "role":"scratch"},
                {"name":"t2", "shape":[1024], "dtype":"f32", "role":"scratch"},
                {"name":"t3", "shape":[1024], "dtype":"f32", "role":"scratch"},
                {"name":"y", "shape":[1024], "dtype":"f32", "role":"output"}
            ],
            "constants": {}, "runtime_constants": {}, "weights": {},
            "kernels": {"k": {"source": "..."}},
            "launches": [
                {"node_id":"t1","kernel":"k","args":["a","t1"],"grid":[[1],[1],[1]],"block":[[32],[1],[1]],"smem":0},
                {"node_id":"t2","kernel":"k","args":["t1","t2"],"grid":[[1],[1],[1]],"block":[[32],[1],[1]],"smem":0},
                {"node_id":"t3","kernel":"k","args":["t2","t3"],"grid":[[1],[1],[1]],"block":[[32],[1],[1]],"smem":0},
                {"node_id":"y","kernel":"k","args":["t3","y"],"grid":[[1],[1],[1]],"block":[[32],[1],[1]],"smem":0}
            ]
        });
        let program = Program::parse(&plan.to_string()).unwrap();
        let layout = program.layout(&Env::new()).unwrap();
        assert_eq!(layout.regions[SCRATCH_REGION], 8192);
        assert_ne!(layout.buffers["t1"].offset, layout.buffers["t2"].offset);
        assert_eq!(layout.buffers["t3"].offset, layout.buffers["t1"].offset);
        assert_eq!(layout.buffers["y"].region, "output:y");
        // A scratch buffer nobody reads is a lowering-contract violation, not a free slot.
        let mut dead = plan.clone();
        dead["launches"][3]["args"] = json!(["t2", "y"]);
        assert!(
            Program::parse(&dead.to_string())
                .unwrap()
                .layout(&Env::new())
                .is_err()
        );
    }

    #[test]
    fn symbolic_layout_follows_the_environment() {
        let plan = json!({
            "format": 1, "backend": "cuda", "inputs": ["x"], "outputs": ["y"],
            "buffers": [
                {"name":"x", "shape":[1, "n", 8], "dtype":"f16", "role":"input"},
                {"name":"y", "shape":[1, "n", 8], "dtype":"f16", "role":"output"}
            ],
            "constants": {}, "runtime_constants": {}, "weights": {},
            "kernels": {"k": {"source": "..."}},
            "symbols": {"bindings": {"n": ["x", 1]}, "hints": {"n": 32}, "caps": {}},
            "launches": [
                {"node_id":"y","kernel":"k","args":["x","y"],"grid":[[["//", ["+", "n", 15], 16]],[1],[1]],
                 "block":[[16],[1],[1]],"smem":0,"runtime_args":["n"]}
            ]
        });
        let program = Program::parse(&plan.to_string()).unwrap();
        let env: Env = [("n".to_owned(), 5)].into_iter().collect();
        let layout = program.layout(&env).unwrap();
        assert_eq!(layout.buffers["x"].bytes, 80);
        assert_eq!(layout.buffers["x"].shape, vec![1, 5, 8]);
        assert_eq!(
            program.layout(&program.default_env()).unwrap().buffers["x"].bytes,
            512
        );
        assert!(program.layout(&Env::new()).is_err());
    }

    #[test]
    fn artifact_resolves_only_bundled_members() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("emmy-artifact-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(root.join("cubin")).unwrap();
        std::fs::write(
            root.join("plan.json"),
            serde_json::to_vec(&example()).unwrap(),
        )
        .unwrap();
        std::fs::write(root.join("w.bin"), 2.0f32.to_le_bytes()).unwrap();
        std::fs::write(root.join("cubin/ab.cubin"), b"compiler binary").unwrap();
        let mut manifest = json!({"format":1,"standalone":1,"environment":{"arch":"sm_89"},
            "programs":{"p":"plan.json"},"bindings":{"p":{"w":"w.bin"}}});
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(Artifact::load(&root, "p").is_ok());
        assert!(Artifact::load(&root, "missing").is_err());
        manifest["bindings"]["p"] = json!({});
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(
            Artifact::load(&root, "p").is_err(),
            "a pack must bind every constant"
        );
        manifest["bindings"]["p"] = json!({"w": "w.bin"});
        manifest["programs"]["p"] = json!("/etc/passwd");
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(Artifact::load(&root, "p").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
