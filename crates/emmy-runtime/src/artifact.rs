//! Validate the supported subset of Emmy's existing JSON execution plan.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const SUPPORTED_PLAN_FORMATS: [u32; 2] = [1, 3];
const PACK_FORMAT: u32 = 1;
const STANDALONE_FORMAT: u32 = 1;
const LAUNCH_DIMENSIONS: usize = 3;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Buffer {
    pub name: String,
    pub shape: Vec<Value>,
    pub dtype: String,
    pub role: String,
}

impl Buffer {
    pub fn byte_len(&self) -> Result<usize> {
        let size: usize = match self.dtype.as_str() {
            "f16" | "bf16" | "i16" | "u16" => 2,
            "f32" | "i32" | "u32" => 4,
            "f64" | "i64" | "u64" => 8,
            "i8" | "u8" | "bool" => 1,
            other => bail!("unsupported dtype: {other}"),
        };
        self.shape.iter().try_fold(size, |bytes, dim| {
            let n = dim
                .as_u64()
                .context("only static nonnegative shapes are supported")?;
            bytes
                .checked_mul(usize::try_from(n)?)
                .context("buffer size overflow")
        })
    }
}

/// How one buffer is virtualized: it is not one allocation but a table of equal-sized pages, cut
/// along `axis` every `page` elements. `start` names the runtime argument that shifts the
/// buffer's own coordinate to an absolute one, which is how a step writes only its new rows.
/// The runtime owns the pages; the plan only says what shape they have.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Paging {
    pub axis: usize,
    pub page: u64,
    #[serde(default)]
    pub start: Option<String>,
}

impl Paging {
    /// The kernel parameter that carries the page table in place of the buffer's pointer.
    pub fn table(name: &str) -> String {
        format!("{name}__pages")
    }

    /// One page's byte size: the buffer's shape with the paged axis cut to `page`.
    ///
    /// How MANY pages exist is not the plan's business. A step writes a chunk of the cache, so
    /// its buffer spans the chunk while the cache spans a request; only the page's own shape —
    /// the other axes, at their declared extents — is shared between them, and that is what the
    /// pool needs to size a page.
    pub fn page_bytes(&self, buffer: &Buffer) -> Result<usize> {
        let extent = buffer
            .shape
            .get(self.axis)
            .context("paged axis is out of range")?
            .as_u64()
            .context("only static nonnegative shapes are supported")?;
        ensure!(extent > 0 && self.page > 0, "empty paged axis");
        Ok(buffer.byte_len()? / usize::try_from(extent)? * usize::try_from(self.page)?)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    pub binary_key: String,
    #[serde(default)]
    pub arch_specific: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    pub node_id: String,
    pub kernel: String,
    pub args: Vec<String>,
    pub grid: Vec<Vec<Value>>,
    pub block: Vec<Vec<Value>>,
    pub smem: u32,
    pub zero_outputs: Vec<String>,
    #[serde(default)]
    pub zero_prologues: Vec<String>,
    #[serde(default)]
    pub writes: Vec<String>,
    #[serde(default)]
    pub indirect: Vec<Value>,
    pub runtime_args: Vec<String>,
    pub cuda: CudaFeatures,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaFeatures {
    pub tma: Vec<Value>,
}

pub fn dimensions(factors: &[Vec<Value>]) -> Result<(u32, u32, u32)> {
    ensure!(
        factors.len() == LAUNCH_DIMENSIONS,
        "launch dimensions must have three axes"
    );
    let mut dims = [1u32; LAUNCH_DIMENSIONS];
    for (out, axis) in dims.iter_mut().zip(factors) {
        for factor in axis {
            let value = factor
                .as_u64()
                .context("only static launch factors are supported")?;
            *out = out
                .checked_mul(u32::try_from(value)?)
                .context("launch dimension overflow")?;
        }
        ensure!(*out > 0, "launch dimensions must be positive");
    }
    Ok((dims[0], dims[1], dims[2]))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub format: u32,
    pub backend: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub buffers: Vec<Buffer>,
    pub constants: BTreeMap<String, Value>,
    pub runtime_constants: BTreeMap<String, Value>,
    pub launches: Vec<Launch>,
    pub kernels: BTreeMap<String, Kernel>,
    pub weights: BTreeMap<String, Value>,
    #[serde(default)]
    pub paged: BTreeMap<String, Paging>,
    pub symbols: Symbols,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Symbols {
    pub bindings: BTreeMap<String, Value>,
    pub hints: BTreeMap<String, Value>,
    pub caps: BTreeMap<String, Value>,
}

impl Plan {
    pub fn validate(&self, bindings: &BTreeMap<String, Vec<u8>>) -> Result<()> {
        ensure!(
            SUPPORTED_PLAN_FORMATS.contains(&self.format),
            "unsupported plan format {}",
            self.format
        );
        ensure!(self.backend == "cuda", "only CUDA plans are supported");
        ensure!(
            self.runtime_constants.is_empty()
                && self.symbols.bindings.is_empty()
                && self.symbols.hints.is_empty()
                && self.symbols.caps.is_empty(),
            "dynamic plans are not yet supported"
        );
        let tables: BTreeSet<String> = self.paged.keys().map(|n| Paging::table(n)).collect();
        let starts: BTreeSet<&str> = self
            .paged
            .values()
            .filter_map(|p| p.start.as_deref())
            .collect();
        let mut names = BTreeSet::new();
        for buffer in &self.buffers {
            ensure!(
                names.insert(buffer.name.as_str()),
                "duplicate buffer {}",
                buffer.name
            );
            ensure!(
                ["input", "constant", "output", "scratch"].contains(&buffer.role.as_str()),
                "invalid buffer role"
            );
            let bytes = buffer.byte_len()?;
            if let Some(paging) = self.paged.get(&buffer.name) {
                // A paged buffer has no slab to upload into or allocate: the runtime owns its
                // pages and binds their table. Only what the kernel reads and writes is declared.
                ensure!(
                    ["input", "output"].contains(&buffer.role.as_str()),
                    "only inputs and outputs may be paged: {}",
                    buffer.name
                );
                ensure!(
                    !bindings.contains_key(&buffer.name),
                    "paged buffer {} cannot carry bytes",
                    buffer.name
                );
                paging.page_bytes(buffer)?;
                continue;
            }
            if let Some(data) = bindings.get(&buffer.name) {
                ensure!(
                    ["input", "constant"].contains(&buffer.role.as_str()),
                    "only inputs and constants may be bound"
                );
                ensure!(
                    data.len() == bytes,
                    "binding size mismatch for {}",
                    buffer.name
                );
            } else {
                ensure!(
                    buffer.role != "constant",
                    "missing constant bytes: {}",
                    buffer.name
                );
            }
        }
        for name in self
            .inputs
            .iter()
            .chain(&self.outputs)
            .chain(bindings.keys())
            .chain(self.constants.keys())
            .chain(self.weights.keys())
        {
            ensure!(names.contains(name.as_str()), "unknown buffer {name}");
        }
        for name in self.paged.keys() {
            ensure!(names.contains(name.as_str()), "unknown paged buffer {name}");
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
        for launch in &self.launches {
            ensure!(
                self.kernels.contains_key(&launch.kernel),
                "unknown kernel {}",
                launch.kernel
            );
            ensure!(
                launch.indirect.is_empty() && launch.cuda.tma.is_empty(),
                "indirect operands and descriptors are not yet supported"
            );
            // The only runtime argument this runtime supplies is a paged buffer's start; every
            // other one still names a symbolic extent nothing here can resolve.
            for name in &launch.runtime_args {
                ensure!(
                    starts.contains(name.as_str()),
                    "runtime argument {name} is not a paged buffer's start"
                );
            }
            dimensions(&launch.grid)?;
            dimensions(&launch.block)?;
            for name in launch
                .args
                .iter()
                .chain(&launch.zero_outputs)
                .chain(&launch.zero_prologues)
                .chain(&launch.writes)
            {
                ensure!(
                    names.contains(name.as_str()) || tables.contains(name.as_str()),
                    "unknown launch buffer {name}"
                );
            }
            // ``writes`` and the zero lists name buffers; only ``args`` passes pointers, and a
            // paged buffer has none. Zeroing one would have to memset a slab that does not exist.
            for name in &launch.args {
                ensure!(
                    !self.paged.contains_key(name.as_str()),
                    "paged buffer {name} has no pointer to pass; its launch must name {}",
                    Paging::table(name)
                );
            }
            for name in launch.zero_outputs.iter().chain(&launch.zero_prologues) {
                ensure!(
                    !self.paged.contains_key(name.as_str()),
                    "paged buffer {name} cannot be zero-initialized per launch"
                );
            }
        }
        Ok(())
    }
}

pub struct Artifact {
    pub(crate) plan: Plan,
    pub(crate) arch: String,
    pub(crate) bindings: BTreeMap<String, Vec<u8>>,
    pub(crate) binaries: BTreeMap<String, PathBuf>,
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
        let plan: Plan = serde_json::from_slice(&std::fs::read(member(&root, path)?)?)?;
        let mut bindings = BTreeMap::new();
        for (name, path) in manifest["bindings"][program]
            .as_object()
            .context("missing binding index")?
        {
            bindings.insert(
                name.clone(),
                std::fs::read(member(
                    &root,
                    path.as_str().context("invalid binding path")?,
                )?)?,
            );
        }
        plan.validate(&bindings)?;
        let mut binaries = BTreeMap::new();
        for (name, kernel) in &plan.kernels {
            ensure!(
                !kernel.arch_specific,
                "architecture-specific kernels are not yet supported"
            );
            ensure!(
                !kernel.binary_key.is_empty()
                    && kernel.binary_key.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid cubin key"
            );
            binaries.insert(
                name.clone(),
                member(&root, &format!("cubin/{}.cubin", kernel.binary_key))?,
            );
        }
        let arch = manifest["environment"]["arch"]
            .as_str()
            .context("missing target architecture")?
            .to_owned();
        Ok(Self {
            plan,
            arch,
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
    fn plan_requires_bound_constants_and_rejects_unsupported_abi() {
        let data = BTreeMap::from([("w".into(), 2.0f32.to_le_bytes().to_vec())]);
        let plan: Plan = serde_json::from_value(example()).unwrap();
        plan.validate(&data).unwrap();
        assert!(plan.validate(&BTreeMap::new()).is_err());
        assert!(
            plan.validate(&BTreeMap::from([("w".into(), vec![0])]))
                .is_err()
        );
        for (pointer, value) in [
            ("/format", json!(999)),
            ("/buffers/0/shape/0", json!("n")),
            ("/buffers/1/name", json!("x")),
            ("/launches/0/args/0", json!("missing")),
            ("/launches/0/runtime_args", json!(["n"])),
            ("/launches/0/cuda/tma", json!([{}])),
            ("/launches/0/kernel", json!("missing")),
        ] {
            let mut value_plan = example();
            *value_plan.pointer_mut(pointer).unwrap() = value;
            let plan: Plan = serde_json::from_value(value_plan).unwrap();
            assert!(plan.validate(&data).is_err(), "accepted {pointer}");
        }
        let mut unknown = example();
        unknown["launches"][0]["unrecognized_abi"] = json!(true);
        assert!(serde_json::from_value::<Plan>(unknown).is_err());
    }

    /// One step of a cache fill: `k` holds the four keys this step produces, and `past` says which
    /// pages of the cache — pages the runtime owns, not the plan — they land in.
    fn paged_example() -> Value {
        json!({
            "format": 1, "backend": "cuda", "inputs": ["x"], "outputs": ["k"],
            "buffers": [
                {"name":"x", "shape":[1,2,4,8], "dtype":"f32", "role":"input"},
                {"name":"k", "shape":[1,2,4,8], "dtype":"f32", "role":"output"}
            ],
            "constants": {}, "runtime_constants": {}, "weights": {},
            "paged": {"k": {"axis": 2, "page": 8, "start": "past"}},
            "kernels": {"fill": {"binary_key":"ab", "arch_specific":false}},
            "symbols": {"bindings":{},"hints":{},"caps":{}},
            "launches": [{"node_id":"k", "kernel":"fill", "args":["x","k__pages"],
                "grid":[[1],[1],[1]],"block":[[32],[1],[1]],"smem":0,
                "zero_outputs":[],"runtime_args":["past"],"cuda":{"tma":[]}}]
        })
    }

    #[test]
    fn paged_buffer_is_addressed_through_its_table_and_never_allocated() {
        let plan: Plan = serde_json::from_value(paged_example()).unwrap();
        plan.validate(&BTreeMap::new()).unwrap();

        let buffer = plan.buffers.iter().find(|b| b.name == "k").unwrap();
        let paging = &plan.paged["k"];
        assert_eq!(Paging::table("k"), "k__pages");
        // One page is the buffer's shape with the paged axis cut to the page size — so a step
        // whose own buffer spans four keys still sizes the cache's eight-key pages correctly.
        assert_eq!(paging.page_bytes(buffer).unwrap(), 2 * 8 * 8 * 4);

        // A paged buffer carries no bytes: there is no slab to upload into.
        let bound = BTreeMap::from([("k".to_string(), vec![0u8; buffer.byte_len().unwrap()])]);
        assert!(plan.validate(&bound).is_err());
    }

    #[test]
    fn paging_rejects_geometry_and_arguments_it_cannot_honor() {
        for (pointer, value) in [
            // The launch must name the table; the buffer itself has no pointer to pass.
            ("/launches/0/args/1", json!("k")),
            // A runtime argument that is not a paging start still has nothing to resolve it.
            ("/launches/0/runtime_args", json!(["seq_len"])),
            // The axis must exist and a page must hold something.
            ("/paged/k/axis", json!(9)),
            ("/paged/k/page", json!(0)),
            // Only what a kernel reads or writes can be paged.
            ("/paged", json!({"missing": {"axis": 0, "page": 1}})),
        ] {
            let mut value_plan = paged_example();
            *value_plan.pointer_mut(pointer).unwrap() = value;
            let plan: Plan = serde_json::from_value(value_plan).unwrap();
            assert!(
                plan.validate(&BTreeMap::new()).is_err(),
                "accepted {pointer}"
            );
        }
    }

    #[test]
    fn dimensions_multiply_factors_and_reject_unsupported_expressions() {
        assert_eq!(
            dimensions(&[vec![json!(2), json!(3)], vec![json!(4)], vec![json!(1)]]).unwrap(),
            (6, 4, 1)
        );
        for factor in [
            json!("seq_len"),
            json!(-1),
            json!(0),
            json!(["+", 1, 2]),
            json!(true),
            json!(u64::MAX),
        ] {
            assert!(dimensions(&[vec![factor], vec![json!(1)], vec![json!(1)]]).is_err());
        }
    }

    #[test]
    fn buffer_size_checks_dtype_shape_and_overflow() {
        let mut buffer = Buffer {
            name: "x".into(),
            shape: vec![json!(2), json!(3)],
            dtype: "f16".into(),
            role: "input".into(),
        };
        assert_eq!(buffer.byte_len().unwrap(), 12);
        buffer.shape = vec![json!(u64::MAX)];
        assert!(buffer.byte_len().is_err());
        buffer.shape.clear();
        buffer.dtype = "unknown".into();
        assert!(buffer.byte_len().is_err());
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
