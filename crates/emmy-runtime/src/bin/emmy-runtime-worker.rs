//! Persistent framed worker. stdout is reserved for control responses; tensors use binary files.

use anyhow::{Context, Result, ensure};
use cudarc::driver::CudaContext;
use emmy_runtime::{artifact::Artifact, cuda::Executor};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Load { root: PathBuf, program: String },
    Bind { inputs: BTreeMap<String, PathBuf> },
    Run { warmup: u32, iterations: u32, capture: bool, outputs: BTreeMap<String, PathBuf> },
    Release,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u32,
    command: Command,
}

fn read_frame(reader: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut header = [0u8; 8];
    if reader.read(&mut header[..1])? == 0 { return Ok(None); }
    reader.read_exact(&mut header[1..])?;
    let size = u64::from_le_bytes(header);
    ensure!(size <= 1024 * 1024, "control frame exceeds 1 MiB");
    let mut bytes = vec![0; size as usize];
    reader.read_exact(&mut bytes)?;
    Ok(Some(bytes))
}

fn main() -> Result<()> {
    ensure!(std::env::args().len() == 1, "worker accepts no arguments; select a GPU with CUDA_VISIBLE_DEVICES");
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let mut context = None;
    let mut executor: Option<Executor> = None;
    while let Some(frame) = read_frame(&mut input)? {
        let response = (|| -> Result<serde_json::Value> {
            let request: Request = serde_json::from_slice(&frame)?;
            ensure!(request.version == 1, "unsupported control protocol version");
            match request.command {
                Command::Load { root, program } => {
                    let artifact = Artifact::load(&root, &program)?;
                    executor = None;
                    if context.is_none() { context = Some(CudaContext::new(0)?); }
                    executor = Some(Executor::load(context.as_ref().unwrap(), artifact)?);
                    Ok(json!({"loaded": true}))
                }
                Command::Bind { inputs } => {
                    let executor = executor.as_mut().context("no loaded program")?;
                    for (name, path) in inputs { executor.bind(&name, &std::fs::read(path)?)?; }
                    Ok(json!({"bound": true}))
                }
                Command::Run { warmup, iterations, capture, outputs } => {
                    let executor = executor.as_mut().context("no loaded program")?;
                    let time_ms = executor.execute(warmup, iterations, capture)?;
                    for (name, path) in outputs { std::fs::write(path, executor.output(&name)?)?; }
                    Ok(json!({"time_ms": time_ms, "captured": capture}))
                }
                Command::Release => { executor = None; Ok(json!({"released": true})) }
            }
        })();
        let failed = response.is_err();
        let value = match response {
            Ok(value) => json!({"version": 1, "result": value}),
            Err(error) => json!({"version": 1, "error": format!("{error:#}"), "retire": true}),
        };
        let bytes = serde_json::to_vec(&value)?;
        output.write_all(&(bytes.len() as u64).to_le_bytes())?;
        output.write_all(&bytes)?;
        output.flush()?;
        if failed {
            // A CUDA fault may poison the context. Do not wait on or reuse its allocations.
            std::process::exit(1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_rejects_truncation_and_oversized_metadata() {
        assert!(read_frame(&mut &[][..]).unwrap().is_none());
        assert!(read_frame(&mut &[1][..]).is_err());
        assert!(read_frame(&mut &(2_000_000u64.to_le_bytes())[..]).is_err());
        let bytes = [3u64.to_le_bytes().as_slice(), b"ab"].concat();
        assert!(read_frame(&mut bytes.as_slice()).is_err());
    }

    #[test]
    fn protocol_reads_operations_and_rejects_unknown_fields() {
        assert!(serde_json::from_value::<Request>(json!({"version":1,"command":{"op":"release"}})).is_ok());
        assert!(serde_json::from_value::<Request>(json!({"version":1,"command":{"op":"release"},"extra":true})).is_err());
    }
}
