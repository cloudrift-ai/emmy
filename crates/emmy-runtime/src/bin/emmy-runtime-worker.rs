//! Persistent framed worker. stdout is reserved for control responses; tensors use binary files.

use anyhow::{Context, Result, ensure};
use emmy_runtime::{
    artifact::Artifact,
    cuda::{Device, Executor},
    generation::Generator,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

const PROTOCOL_VERSION: u32 = 1;
const FRAME_HEADER_BYTES: usize = size_of::<u64>();
const MAX_CONTROL_FRAME_BYTES: u64 = 1024 * 1024;
const MILLISECONDS_PER_SECOND: f64 = 1000.0;
const GPU_ORDINAL: usize = 0;
const FAILURE_EXIT_CODE: i32 = 1;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Load {
        root: PathBuf,
        program: String,
    },
    Bind {
        inputs: BTreeMap<String, PathBuf>,
    },
    Run {
        warmup: u32,
        iterations: u32,
        capture: bool,
        outputs: BTreeMap<String, PathBuf>,
    },
    LoadGeneration {
        root: PathBuf,
    },
    StartGeneration {
        prompt: PathBuf,
    },
    GenerationStep {
        capture: bool,
        logits: Option<PathBuf>,
    },
    Generate {
        prompt: PathBuf,
        max_new_tokens: usize,
        capture: bool,
        output: PathBuf,
    },
    Release,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u32,
    command: Command,
}

fn read_frame(reader: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut header = [0u8; FRAME_HEADER_BYTES];
    if reader.read(&mut header[..1])? == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut header[1..])?;
    let size = u64::from_le_bytes(header);
    ensure!(
        size <= MAX_CONTROL_FRAME_BYTES,
        "control frame exceeds {MAX_CONTROL_FRAME_BYTES} bytes"
    );
    let mut bytes = vec![0; size as usize];
    reader.read_exact(&mut bytes)?;
    Ok(Some(bytes))
}

fn read_tokens(path: &PathBuf) -> Result<Vec<i64>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len() % size_of::<i64>() == 0,
        "invalid token payload size"
    );
    Ok(bytes
        .chunks_exact(size_of::<i64>())
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn main() -> Result<()> {
    ensure!(
        std::env::args().len() == 1,
        "worker accepts no arguments; select a GPU with CUDA_VISIBLE_DEVICES"
    );
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let mut context = None;
    let mut executor: Option<Executor> = None;
    let mut generator: Option<Generator> = None;
    while let Some(frame) = read_frame(&mut input)? {
        let response = (|| -> Result<serde_json::Value> {
            let request: Request = serde_json::from_slice(&frame)?;
            ensure!(
                request.version == PROTOCOL_VERSION,
                "unsupported control protocol version"
            );
            match request.command {
                Command::Load { root, program } => {
                    let started = Instant::now();
                    let artifact = Artifact::load(&root, &program)?;
                    let artifact_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
                    executor = None;
                    generator = None;
                    let started = Instant::now();
                    if context.is_none() {
                        context = Some(Device::new(GPU_ORDINAL)?);
                    }
                    let context_ms = started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND;
                    executor = Some(Executor::load(context.as_ref().unwrap(), artifact)?);
                    Ok(
                        json!({"loaded": true, "artifact_ms": artifact_ms, "context_ms": context_ms,
                        "load_times_ms": executor.as_ref().unwrap().load_times_ms}),
                    )
                }
                Command::Bind { inputs } => {
                    let executor = executor.as_mut().context("no loaded program")?;
                    for (name, path) in inputs {
                        executor.bind(&name, &std::fs::read(path)?)?;
                    }
                    Ok(json!({"bound": true}))
                }
                Command::Run {
                    warmup,
                    iterations,
                    capture,
                    outputs,
                } => {
                    let executor = executor.as_mut().context("no loaded program")?;
                    let metrics = executor.execute(warmup, iterations, capture)?;
                    let started = Instant::now();
                    for (name, path) in outputs {
                        std::fs::write(path, executor.output(&name)?)?;
                    }
                    Ok(
                        json!({"time_ms": metrics.time_ms, "captured": capture, "metrics": metrics,
                        "output_ms": started.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND}),
                    )
                }
                Command::LoadGeneration { root } => {
                    executor = None;
                    generator = None;
                    if context.is_none() {
                        context = Some(Device::new(GPU_ORDINAL)?);
                    }
                    generator = Some(Generator::load(context.as_ref().unwrap(), &root)?);
                    Ok(json!({"loaded": true}))
                }
                Command::StartGeneration { prompt } => {
                    generator
                        .as_mut()
                        .context("no loaded generator")?
                        .start(&read_tokens(&prompt)?)?;
                    Ok(json!({"started": true}))
                }
                Command::GenerationStep { capture, logits } => {
                    let generator = generator.as_mut().context("no loaded generator")?;
                    let token = generator.advance(capture)?;
                    if let Some(path) = logits {
                        std::fs::write(path, generator.logits()?)?;
                    }
                    Ok(json!({"token": token}))
                }
                Command::Generate {
                    prompt,
                    max_new_tokens,
                    capture,
                    output,
                } => {
                    let tokens = generator
                        .as_mut()
                        .context("no loaded generator")?
                        .generate(&read_tokens(&prompt)?, max_new_tokens, capture)?;
                    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
                    std::fs::write(output, bytes)?;
                    Ok(json!({"generated_tokens": tokens.len()}))
                }
                Command::Release => {
                    generator = None;
                    executor = None;
                    Ok(json!({"released": true}))
                }
            }
        })();
        let failed = response.is_err();
        let value = match response {
            Ok(value) => json!({"version": PROTOCOL_VERSION, "result": value}),
            Err(error) => {
                json!({"version": PROTOCOL_VERSION, "error": format!("{error:#}"), "retire": true})
            }
        };
        let bytes = serde_json::to_vec(&value)?;
        output.write_all(&(bytes.len() as u64).to_le_bytes())?;
        output.write_all(&bytes)?;
        output.flush()?;
        if failed {
            // A CUDA fault may poison the context. Do not wait on or reuse its allocations.
            std::process::exit(FAILURE_EXIT_CODE);
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
        assert!(read_frame(&mut &((MAX_CONTROL_FRAME_BYTES + 1).to_le_bytes())[..]).is_err());
        let bytes = [3u64.to_le_bytes().as_slice(), b"ab"].concat();
        assert!(read_frame(&mut bytes.as_slice()).is_err());
    }

    #[test]
    fn protocol_reads_operations_and_rejects_unknown_fields() {
        assert!(
            serde_json::from_value::<Request>(
                json!({"version":PROTOCOL_VERSION,"command":{"op":"release"}})
            )
            .is_ok()
        );
        assert!(
            serde_json::from_value::<Request>(
                json!({"version":PROTOCOL_VERSION,"command":{"op":"release"},"extra":true})
            )
            .is_err()
        );
    }
}
