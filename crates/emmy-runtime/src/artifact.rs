//! Validate the supported subset of Emmy's existing JSON execution plan.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

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
            let n = dim.as_u64().context("only static nonnegative shapes are supported")?;
            bytes.checked_mul(usize::try_from(n)?).context("buffer size overflow")
        })
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
    ensure!(factors.len() == 3, "launch dimensions must have three axes");
    let mut dims = [1u32; 3];
    for (out, axis) in dims.iter_mut().zip(factors) {
        for factor in axis {
            let value = factor.as_u64().context("only static launch factors are supported")?;
            *out = out.checked_mul(u32::try_from(value)?).context("launch dimension overflow")?;
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
        ensure!([1, 3].contains(&self.format), "unsupported plan format {}", self.format);
        ensure!(self.backend == "cuda", "only CUDA plans are supported");
        ensure!(self.runtime_constants.is_empty() && self.symbols.bindings.is_empty()
            && self.symbols.hints.is_empty() && self.symbols.caps.is_empty(), "dynamic plans are not yet supported");
        let mut names = BTreeSet::new();
        for buffer in &self.buffers {
            ensure!(names.insert(buffer.name.as_str()), "duplicate buffer {}", buffer.name);
            ensure!(["input", "constant", "output", "scratch"].contains(&buffer.role.as_str()), "invalid buffer role");
            let bytes = buffer.byte_len()?;
            if let Some(data) = bindings.get(&buffer.name) {
                ensure!(["input", "constant"].contains(&buffer.role.as_str()), "only inputs and constants may be bound");
                ensure!(data.len() == bytes, "binding size mismatch for {}", buffer.name);
            } else {
                ensure!(buffer.role != "constant", "missing constant bytes: {}", buffer.name);
            }
        }
        for name in self.inputs.iter().chain(&self.outputs).chain(bindings.keys()).chain(self.constants.keys()).chain(self.weights.keys()) {
            ensure!(names.contains(name.as_str()), "unknown buffer {name}");
        }
        for (list, role) in [(&self.inputs, "input"), (&self.outputs, "output")] {
            ensure!(list.iter().collect::<BTreeSet<_>>().len() == list.len(), "duplicate {role}");
            for name in list {
                ensure!(self.buffers.iter().any(|b| &b.name == name && b.role == role), "invalid {role} buffer {name}");
            }
        }
        for launch in &self.launches {
            ensure!(self.kernels.contains_key(&launch.kernel), "unknown kernel {}", launch.kernel);
            ensure!(launch.indirect.is_empty() && launch.cuda.tma.is_empty() && launch.runtime_args.is_empty(),
                "indirect operands, descriptors, and runtime arguments are not yet supported");
            dimensions(&launch.grid)?;
            dimensions(&launch.block)?;
            for name in launch.args.iter().chain(&launch.zero_outputs).chain(&launch.zero_prologues).chain(&launch.writes) {
                ensure!(names.contains(name.as_str()), "unknown launch buffer {name}");
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
    ensure!(path.starts_with(root) && path.is_file(), "artifact member escapes bundle: {name}");
    Ok(path)
}

impl Artifact {
    pub fn load(root: &Path, program: &str) -> Result<Self> {
        let root = root.canonicalize()?;
        let manifest: Value = serde_json::from_slice(&std::fs::read(member(&root, "manifest.json")?)?)?;
        ensure!(manifest["format"] == 1 && manifest["standalone"] == 1, "unsupported standalone pack format");
        let path = manifest["programs"][program].as_str().context("unknown program")?;
        let plan: Plan = serde_json::from_slice(&std::fs::read(member(&root, path)?)?)?;
        let mut bindings = BTreeMap::new();
        for (name, path) in manifest["bindings"][program].as_object().context("missing binding index")? {
            bindings.insert(name.clone(), std::fs::read(member(&root, path.as_str().context("invalid binding path")?)?)?);
        }
        plan.validate(&bindings)?;
        let mut binaries = BTreeMap::new();
        for (name, kernel) in &plan.kernels {
            ensure!(!kernel.arch_specific, "architecture-specific kernels are not yet supported");
            ensure!(!kernel.binary_key.is_empty() && kernel.binary_key.bytes().all(|b| b.is_ascii_hexdigit()), "invalid cubin key");
            binaries.insert(name.clone(), member(&root, &format!("cubin/{}.cubin", kernel.binary_key))?);
        }
        let arch = manifest["environment"]["arch"].as_str().context("missing target architecture")?.to_owned();
        Ok(Self { plan, arch, bindings, binaries })
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
        assert!(plan.validate(&BTreeMap::from([("w".into(), vec![0])])).is_err());
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

    #[test]
    fn dimensions_multiply_factors_and_reject_unsupported_expressions() {
        assert_eq!(dimensions(&[vec![json!(2), json!(3)], vec![json!(4)], vec![json!(1)]]).unwrap(), (6, 4, 1));
        for factor in [json!("seq_len"), json!(-1), json!(0), json!(["+", 1, 2]), json!(true), json!(u64::MAX)] {
            assert!(dimensions(&[vec![factor], vec![json!(1)], vec![json!(1)]]).is_err());
        }
    }

    #[test]
    fn buffer_size_checks_dtype_shape_and_overflow() {
        let mut buffer = Buffer { name: "x".into(), shape: vec![json!(2), json!(3)], dtype: "f16".into(), role: "input".into() };
        assert_eq!(buffer.byte_len().unwrap(), 12);
        buffer.shape = vec![json!(u64::MAX)];
        assert!(buffer.byte_len().is_err());
        buffer.shape.clear();
        buffer.dtype = "unknown".into();
        assert!(buffer.byte_len().is_err());
    }
}
