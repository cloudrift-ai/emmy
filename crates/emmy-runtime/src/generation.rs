//! Single-request cached generation over a compiler-prepared token-step program.

use crate::{
    artifact::Artifact,
    cuda::{Device, Executor},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;

const GENERATION_FORMAT: u32 = 2;
const DEFAULT_TEMPERATURE: f64 = 0.0;
const DEFAULT_TOP_P: f64 = 1.0;
const DEFAULT_SEED: u64 = 0;
const TOKEN_BYTES: usize = size_of::<i64>();
const MAX_CONTEXT: usize = 4096;
const PROGRAM: &str = "decode";

/// Request-local sampling controls. Temperature zero selects greedy decoding.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            seed: DEFAULT_SEED,
        }
    }
}

impl Sampling {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "temperature must be finite and nonnegative"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub context_length: usize,
    pub vocab_size: usize,
    pub eos_ids: Vec<i64>,
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == GENERATION_FORMAT,
            "unsupported generation format"
        );
        ensure!(
            (1..=MAX_CONTEXT).contains(&self.context_length),
            "unsupported context length"
        );
        ensure!(
            self.vocab_size > 0 && self.vocab_size <= i64::MAX as usize,
            "invalid vocabulary size"
        );
        ensure!(
            self.eos_ids
                .iter()
                .all(|&id| id >= 0 && (id as usize) < self.vocab_size),
            "invalid EOS token"
        );
        Ok(())
    }

    fn validate_prompt(&self, prompt: &[i64]) -> Result<()> {
        ensure!(
            !prompt.is_empty() && prompt.len() <= self.context_length,
            "prompt outside context capacity"
        );
        ensure!(
            prompt
                .iter()
                .all(|&id| id >= 0 && (id as usize) < self.vocab_size),
            "token outside vocabulary"
        );
        Ok(())
    }
}

pub struct Generator {
    executor: Executor,
    config: Config,
    position: usize,
    prompt_length: usize,
    stopped: bool,
}

impl Generator {
    pub fn load(device: &Device, root: &Path) -> Result<Self> {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let config: Config = serde_json::from_value(manifest["key"]["generation"].clone())?;
        config.validate()?;
        let artifact = Artifact::load(root, PROGRAM)?;
        // Fixed interface names and geometry are part of the generation contract. Check before GPU allocation.
        for (name, dtype, role, shape) in [
            ("prompt", "i64", "input", vec![config.context_length as i64]),
            ("prompt_length", "i64", "input", vec![1]),
            ("position", "i64", "input", vec![1]),
            ("sampling", "f64", "input", vec![2]),
            ("seed", "u64", "input", vec![1]),
            ("next_token", "i64", "output", vec![1]),
            ("logits", "f16", "output", vec![1, config.vocab_size as i64]),
        ] {
            let buffer = artifact
                .program
                .buffer(name)
                .context("missing generation buffer")?;
            ensure!(
                buffer.dtype == dtype
                    && buffer.role == role
                    && buffer.static_shape().as_deref() == Some(shape.as_slice()),
                "invalid generation buffer {name}"
            );
        }
        ensure!(
            artifact.program.inputs == ["prompt", "prompt_length", "position", "sampling", "seed"],
            "invalid generation inputs"
        );
        ensure!(
            artifact.program.outputs == ["logits", "next_token"],
            "invalid generation outputs"
        );
        Ok(Self {
            executor: Executor::load(device, artifact)?,
            config,
            position: 0,
            prompt_length: 0,
            stopped: true,
        })
    }

    /// Reset request state. Old cache entries are invisible until overwritten at their absolute positions.
    pub fn start(&mut self, prompt: &[i64], sampling: Sampling) -> Result<()> {
        self.config.validate_prompt(prompt)?;
        sampling.validate()?;
        self.stopped = true;
        self.executor.bind(
            "sampling",
            &[
                sampling.temperature.to_le_bytes(),
                sampling.top_p.to_le_bytes(),
            ]
            .concat(),
        )?;
        self.executor.bind("seed", &sampling.seed.to_le_bytes())?;
        let mut bytes = vec![0; self.config.context_length * TOKEN_BYTES];
        for (slot, token) in bytes
            .as_chunks_mut::<TOKEN_BYTES>()
            .0
            .iter_mut()
            .zip(prompt)
        {
            slot.copy_from_slice(&token.to_le_bytes());
        }
        self.executor.bind("prompt", &bytes)?;
        self.executor
            .bind("prompt_length", &(prompt.len() as i64).to_le_bytes())?;
        self.position = 0;
        self.prompt_length = prompt.len();
        self.stopped = false;
        Ok(())
    }

    /// Process one prompt or decode token; only generated tokens are observed by the CPU.
    pub fn advance(&mut self, capture: bool, ignore_eos: bool) -> Result<Option<i64>> {
        ensure!(
            !self.stopped && self.position < self.config.context_length,
            "generation is stopped or context is full"
        );
        self.stopped = true;
        self.executor
            .bind("position", &(self.position as i64).to_le_bytes())?;
        self.executor.advance(capture)?;
        self.position += 1;
        if self.position < self.prompt_length {
            self.stopped = false;
            return Ok(None);
        }
        let bytes = self.executor.output("next_token")?;
        let token = i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid sampled token size"))?,
        );
        ensure!(
            token >= 0 && (token as usize) < self.config.vocab_size,
            "invalid sampled token"
        );
        self.stopped = !ignore_eos && self.config.eos_ids.contains(&token);
        Ok(Some(token))
    }

    /// Diagnostic transfer only. Normal generation downloads one selected token per decode step.
    pub fn logits(&self) -> Result<Vec<u8>> {
        self.executor.output("logits")
    }

    pub fn generate(
        &mut self,
        prompt: &[i64],
        max_new_tokens: usize,
        capture: bool,
        sampling: Sampling,
    ) -> Result<Vec<i64>> {
        self.config.validate_prompt(prompt)?;
        ensure!(
            max_new_tokens <= self.config.context_length - prompt.len(),
            "generation exceeds context capacity"
        );
        self.start(prompt, sampling)?;
        let mut tokens = Vec::new();
        while tokens.len() < max_new_tokens && !self.stopped {
            if let Some(token) = self.advance(capture, false)? {
                tokens.push(token);
            }
        }
        self.stopped = true;
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validate_sampling_controls() {
        Sampling::default().validate().unwrap();
        Sampling {
            temperature: f64::MIN_POSITIVE,
            top_p: f64::MIN_POSITIVE,
            seed: u64::MAX,
        }
        .validate()
        .unwrap();
        for temperature in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                Sampling {
                    temperature,
                    ..Sampling::default()
                }
                .validate()
                .is_err()
            );
        }
        for top_p in [0.0, -1.0, 1.01, f64::NAN, f64::INFINITY] {
            assert!(
                Sampling {
                    top_p,
                    ..Sampling::default()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn reject_invalid_geometry_and_prompt_before_submission() {
        let mut config = Config {
            version: GENERATION_FORMAT,
            context_length: 8,
            vocab_size: 32,
            eos_ids: vec![31],
        };
        config.validate().unwrap();
        config.validate_prompt(&[0, 31]).unwrap();
        for prompt in [vec![], vec![-1], vec![32], vec![1; 9]] {
            assert!(config.validate_prompt(&prompt).is_err());
        }
        config.context_length = MAX_CONTEXT + 1;
        assert!(config.validate().is_err());
        config.context_length = 8;
        config.eos_ids = vec![32];
        assert!(config.validate().is_err());
    }
}
