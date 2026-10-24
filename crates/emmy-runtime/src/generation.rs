//! Single-request cached generation over a compiler-prepared token-step program. The step
//! computes the logits and the greedy token on the device; sampling at positive temperature
//! happens here, on the host, so nothing the device runs is hand-written.

use crate::{
    artifact::{Artifact, Paging, member},
    cuda::{Device, DeviceMemory, Executor, HostMemory},
    pager::Pager,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, Instant};

const GENERATION_FORMAT: u32 = 9;
const DEFAULT_TEMPERATURE: f64 = 0.0;
const DEFAULT_TOP_P: f64 = 1.0;
const DEFAULT_SEED: u64 = 0;
const TOKEN_BYTES: usize = size_of::<i64>();
const MAX_CONTEXT: usize = 4096;
const PROGRAM: &str = "decode";
const LOGIT_BYTES: usize = size_of::<f32>();

/// The token one decode step selects from its FP32 logits at positive temperature: the tokens in
/// descending logit order, ties by ascending id, the smallest prefix of that order reaching top-p,
/// and a draw over it in id order with f64 weights against a SplitMix64 counter of the request
/// seed and the generated-token `index`, so captured and uncaptured steps select the same token
/// for the same logits. Greedy selection is the decode program's own output. A nonfinite logit
/// fails the request.
fn sample(logits: &[f32], sampling: &Sampling, index: u64) -> Result<i64> {
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "nonfinite logit"
    );
    let maximum = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|&value| ((f64::from(value) - f64::from(maximum)) / sampling.temperature).exp())
        .collect();
    let target_mass = sampling.top_p * weights.iter().sum::<f64>();
    // Signed zeros compare equal, and every logit is finite, so the order is total.
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .expect("finite logits")
            .then(a.cmp(&b))
    });
    let mut kept = vec![false; logits.len()];
    let mut mass = 0.0;
    for &token in &order {
        kept[token] = true;
        mass += weights[token];
        if mass >= target_mass {
            break;
        }
    }
    let mut random = sampling
        .seed
        .wrapping_add(0x9e37_79b9_7f4a_7c15u64.wrapping_mul(index));
    random = (random ^ (random >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    random = (random ^ (random >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    random ^= random >> 31;
    let target = (random >> 11) as f64 * 2f64.powi(-53) * mass;
    let (mut cumulative, mut last) = (0.0, None);
    for token in 0..logits.len() {
        if !kept[token] || weights[token] == 0.0 {
            continue;
        }
        last = Some(token as i64);
        cumulative += weights[token];
        if target < cumulative {
            return Ok(token as i64);
        }
    }
    // Rounding at the final cumulative boundary.
    last.context("no token carries weight")
}

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
    pub prefill_size: usize,
    pub eos_ids: Vec<i64>,
    pub state: Vec<String>,
}

/// Whether the prefill program may take the chunk at `position`. A chunk's rows past the
/// prompt embed the previous selection and write cache rows the decode step overwrites, so it
/// is taken wherever the whole chunk fits the context and the prompt's last token stays for
/// decode. Carried state has no such repair: a row past the prompt would be folded into the
/// request, so then every row of the chunk must be a prompt token before the last.
fn chunk_fits(
    position: usize,
    prompt_length: usize,
    prefill_size: usize,
    context_length: usize,
    carried: bool,
) -> bool {
    position + 1 < prompt_length
        && position + prefill_size <= context_length
        && (!carried || position + prefill_size < prompt_length)
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
            (1..=self.context_length).contains(&self.prefill_size),
            "invalid prefill size"
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
        let mut names = std::collections::BTreeSet::new();
        ensure!(
            self.state.iter().all(|name| names.insert(name)),
            "a state buffer is named twice"
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
    // The borrower drops before the decode executor that owns the shared allocations, and both
    // before the pager's tables and slots and the pages they point at.
    prefill: Option<Executor>,
    executor: Executor,
    _pager: Pager,
    _homes: Vec<(DeviceMemory, HostMemory)>,
    config: Config,
    sampling: Sampling,
    position: usize,
    prompt_length: usize,
    stopped: bool,
}

impl Generator {
    pub fn load(device: &Device, root: &Path) -> Result<Self> {
        let root = &root.canonicalize()?;
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
            ("next_token", "i64", "input", vec![1]),
            ("logits", "f32", "output", vec![1, config.vocab_size as i64]),
            ("token", "i64", "output", vec![1]),
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
            artifact.program.inputs == ["prompt", "prompt_length", "position", "next_token"],
            "invalid generation inputs"
        );
        ensure!(
            artifact.program.outputs == ["logits", "token"],
            "invalid generation outputs"
        );
        let mut prefill_artifact = if config.prefill_size > 1 {
            Some(Artifact::load(root, "prefill")?)
        } else {
            None
        };
        let mut shared = Vec::new();
        let mut shared_tables = Vec::new();
        if let Some(prefill) = &prefill_artifact {
            ensure!(
                prefill.program.inputs == ["prompt", "prompt_length", "position", "next_token"]
                    && prefill.program.outputs.is_empty(),
                "invalid prefill interface"
            );
            for buffer in &prefill.program.buffers {
                if buffer.role == "scratch"
                    || (buffer.role == "output"
                        && !artifact
                            .program
                            .buffer(&buffer.name)
                            .is_ok_and(|b| b.role == "output"))
                {
                    continue;
                }
                if buffer.role == "constant"
                    && artifact.bindings.get(&buffer.name) != prefill.bindings.get(&buffer.name)
                {
                    continue;
                }
                let original = artifact
                    .program
                    .buffer(&buffer.name)
                    .context("missing shared prefill buffer")?;
                ensure!(
                    buffer.dtype == original.dtype
                        && buffer.role == original.role
                        && buffer.static_shape() == original.static_shape()
                        && prefill.program.paged.get(&buffer.name)
                            == artifact.program.paged.get(&buffer.name),
                    "invalid shared prefill buffer {}",
                    buffer.name
                );
                // A paged buffer lends its page table where a flat one lends its region.
                if artifact.program.paged.contains_key(&buffer.name) {
                    shared_tables.push(buffer.name.clone());
                } else {
                    shared.push(buffer.name.clone());
                }
            }
        }
        // Prefill uploads none of what it shares, so its copy of those bytes goes before the pages take the host.
        if let Some(prefill) = &mut prefill_artifact {
            prefill.bindings.retain(|name, _| !shared.contains(name));
        }
        for name in &config.state {
            let buffer = artifact
                .program
                .buffer(name)
                .context("missing state buffer")?;
            ensure!(
                buffer.role == "output" && !artifact.program.paged.contains_key(name),
                "state buffer {name} is not a flat output the step writes in place"
            );
        }
        // The buffers the pager holds keep every page at its home, read straight from the pack: the
        // resident pages in device memory, the rest in pinned host memory, and the device slots
        // their paging declares. One pager serves both programs: decode loads with its tables
        // lent, and prefill shares them with decode's other buffers.
        let (mut pager, mut homes, mut lent) = (Pager::default(), Vec::new(), BTreeMap::new());
        let env = artifact.program.default_env();
        let mut held = BTreeSet::new();
        for (name, entry) in manifest["pages"].as_object().into_iter().flatten() {
            let buffer = artifact.program.buffer(name)?;
            let paging = artifact
                .program
                .paged
                .get(name)
                .with_context(|| format!("{name} has pages but is not paged"))?;
            ensure!(
                paging.geometry(buffer, &env)?.0 == 1,
                "{name} is not paged along its leading axis, so its bytes are not page after page"
            );
            let (pages, page_bytes) = (
                paging.page_count(buffer, &env)?,
                paging.page_bytes(buffer, &env)?,
            );
            let slots = paging
                .slots
                .with_context(|| format!("{name} has pages but no device slots"))?;
            let resident = entry["resident"]
                .as_u64()
                .with_context(|| format!("{name} has no resident page count"))?
                as usize;
            ensure!(
                resident <= pages,
                "{name} has {resident} resident pages of {pages}"
            );
            let path = member(root, entry["path"].as_str().context("invalid page path")?)?;
            let mut file = std::fs::File::open(&path)?;
            ensure!(
                file.metadata()?.len() == (pages * page_bytes) as u64,
                "{} does not hold {pages} pages of {page_bytes} bytes",
                path.display()
            );
            let on_device = DeviceMemory::read(device, &mut file, resident * page_bytes)?;
            let home = HostMemory::read(device, &mut file, (pages - resident) * page_bytes)?;
            pager.declare(
                device,
                name,
                (on_device.ptr(), home.device_ptr()?),
                (pages, page_bytes, slots, resident),
            )?;
            lent.extend(pager.regions(name)?);
            homes.push((on_device, home));
            held.insert(name.clone());
        }
        for program in std::iter::once(&artifact).chain(&prefill_artifact) {
            for (name, paging) in &program.program.paged {
                let owner = name.strip_suffix("__view").unwrap_or(name);
                ensure!(
                    paging.slots.is_none() || held.contains(owner),
                    "paged buffer {name} has no pages in the pack"
                );
            }
        }
        let mut executor = Executor::load_with(device, artifact, None, Some(lent))?;
        for name in &held {
            pager.bind(&mut executor, name)?;
        }
        let prefill = if let Some(artifact) = prefill_artifact {
            // Prefill loads onto decode's memory: each buffer it shares is lent as it loads, so
            // nothing shared is allocated or uploaded twice.
            let layout = artifact.program.layout(&artifact.program.default_env())?;
            let mut lent = BTreeMap::new();
            for name in &shared {
                let view = executor.buffer(name)?;
                lent.insert(layout.buffers[name].region.clone(), (view.ptr, view.bytes));
            }
            let mut prefill = Executor::load_with(device, artifact, None, Some(lent))?;
            for name in shared_tables {
                let (ptr, len) = executor.page_table(&name)?;
                prefill.set_external(&Paging::table(&name), ptr, len)?;
            }
            Some(prefill)
        } else {
            None
        };
        Ok(Self {
            prefill,
            executor,
            _pager: pager,
            _homes: homes,
            config,
            sampling: Sampling::default(),
            position: 0,
            prompt_length: 0,
            stopped: true,
        })
    }

    /// Reset request state. Old cache entries are invisible until overwritten at their absolute
    /// positions, and the state a step carries starts at zero.
    pub fn start(&mut self, prompt: &[i64], sampling: Sampling) -> Result<()> {
        self.config.validate_prompt(prompt)?;
        sampling.validate()?;
        self.stopped = true;
        self.sampling = sampling;
        for name in &self.config.state {
            self.executor.zero(name)?;
        }
        self.executor.bind("next_token", &0i64.to_le_bytes())?;
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
        if let Some(prefill) = &mut self.prefill {
            // A chunk's rows past the prompt embed the selection too; nothing reads their cache
            // rows before the decode step rewrites them.
            for name in ["prompt", "prompt_length", "next_token"] {
                let view = self.executor.buffer(name)?;
                prefill.bind_device(name, view.ptr, view.bytes)?;
            }
        }
        self.position = 0;
        self.prompt_length = prompt.len();
        self.stopped = false;
        Ok(())
    }

    /// Consume a prefill chunk, or one token on the unchanged decode program. The chunk rule
    /// (`chunk_fits`) says which; the decode step covers the rest.
    pub fn advance(&mut self, capture: bool, ignore_eos: bool) -> Result<Option<i64>> {
        ensure!(!self.stopped, "generation is stopped");
        if chunk_fits(
            self.position,
            self.prompt_length,
            self.config.prefill_size,
            self.config.context_length,
            !self.config.state.is_empty(),
        ) && let Some(prefill) = &mut self.prefill
        {
            self.stopped = true;
            prefill.bind("position", &(self.position as i64).to_le_bytes())?;
            prefill.advance(capture)?;
            self.position += self
                .config
                .prefill_size
                .min(self.prompt_length - self.position - 1);
            self.stopped = false;
            return Ok(None);
        }
        self.step(capture, ignore_eos)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Diagnostic single-token execution, including during prefill. A decode step takes the
    /// token the program selected, or samples one here from its logits, and hands it back for
    /// the next step.
    pub fn step(&mut self, capture: bool, ignore_eos: bool) -> Result<Option<i64>> {
        ensure!(
            !self.stopped && self.position < self.config.context_length,
            "generation is stopped or context is full"
        );
        self.stopped = true;
        self.executor
            .bind("position", &(self.position as i64).to_le_bytes())?;
        self.executor.advance(capture)?;
        let position = self.position;
        self.position += 1;
        if self.position < self.prompt_length {
            self.stopped = false;
            return Ok(None);
        }
        let token = if self.sampling.temperature == 0.0 {
            // The program's own greedy pick; a nonfinite logit leaves it out of range.
            let bytes = self.executor.output("token")?;
            i64::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid token size"))?,
            )
        } else {
            let bytes = self.executor.output("logits")?;
            let logits: Vec<f32> = bytes
                .as_chunks::<LOGIT_BYTES>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            ensure!(
                logits.len() == self.config.vocab_size,
                "invalid logits size"
            );
            // The generated-token index: one for the first token after the prompt.
            sample(
                &logits,
                &self.sampling,
                (position + 2 - self.prompt_length) as u64,
            )?
        };
        ensure!(
            token >= 0 && (token as usize) < self.config.vocab_size,
            "invalid selected token"
        );
        self.executor.bind("next_token", &token.to_le_bytes())?;
        self.stopped = !ignore_eos && self.config.eos_ids.contains(&token);
        Ok(Some(token))
    }

    /// Diagnostic transfer only; a decode step at positive temperature downloads them anyway.
    pub fn logits(&self) -> Result<Vec<u8>> {
        self.executor.output("logits")
    }

    /// Each generated token, with the time since the request started at which the host had it.
    pub fn generate(
        &mut self,
        prompt: &[i64],
        max_new_tokens: usize,
        capture: bool,
        sampling: Sampling,
    ) -> Result<Vec<(i64, Duration)>> {
        self.config.validate_prompt(prompt)?;
        ensure!(
            max_new_tokens <= self.config.context_length - prompt.len(),
            "generation exceeds context capacity"
        );
        let started = Instant::now();
        self.start(prompt, sampling)?;
        let mut tokens = Vec::new();
        while tokens.len() < max_new_tokens && !self.stopped {
            if let Some(token) = self.advance(capture, false)? {
                tokens.push((token, started.elapsed()));
            }
        }
        self.stopped = true;
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The independent nucleus distribution of `values`: a stable sort by value, the smallest
    /// prefix reaching top-p, renormalized.
    fn nucleus(values: &[f32], temperature: f64, top_p: f64) -> Vec<f64> {
        let n = values.len();
        let mut probabilities = vec![0.0; n];
        let maximum = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = values
            .iter()
            .map(|&v| ((f64::from(v) - f64::from(maximum)) / temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());
        let mut cumulative = 0.0;
        for &token in &order {
            cumulative += weights[token] / total;
            probabilities[token] = weights[token];
            if cumulative >= top_p {
                break;
            }
        }
        let mass: f64 = probabilities.iter().sum();
        probabilities.iter().map(|p| p / mass).collect()
    }

    #[test]
    fn sampling_matches_an_independent_nucleus_distribution() {
        // The reference sorts the tokens directly.
        let n = 257;
        let mut state = 928u64;
        let mut normal = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 * 2f64.powi(-53);
            (12.0 * (u - 0.5)) as f32 * 0.5
        };
        let random: Vec<f32> = (0..n).map(|_| normal()).collect();
        let zeros = vec![0.0f32; n];
        let edges: Vec<f32> = (0..n)
            .map(|i| [-65504.0f32, 65504.0, -0.0, 0.0][i % 4])
            .collect();
        // Distinct scores inside one FP16 bin stay apart in FP32.
        let close: Vec<f32> = (0..n)
            .map(|i| 20.078 + 0.004 * i as f32 / (n - 1) as f32)
            .collect();
        for logits in [random, zeros, edges, close] {
            let values = logits.clone();
            for (temperature, top_p) in [(0.7, 0.8), (2.0, 1.0), (1e-300, 0.01), (1e300, 0.5)] {
                let expected = nucleus(&values, temperature, top_p);
                let draws = 1024;
                let mut counts = vec![0usize; n];
                for seed in 0..draws {
                    let sampling = Sampling {
                        temperature,
                        top_p,
                        seed,
                    };
                    let token = sample(&logits, &sampling, 1).unwrap() as usize;
                    assert!(expected[token] > 0.0, "{temperature} {top_p} drew {token}");
                    counts[token] += 1;
                }
                for token in 0..n {
                    let frequency = counts[token] as f64 / draws as f64;
                    let p = expected[token];
                    // Six binomial standard deviations plus one draw for rounding.
                    let bound = 6.0 * (p * (1.0 - p) / draws as f64).sqrt() + 1.0 / draws as f64;
                    assert!(
                        (frequency - p).abs() <= bound,
                        "{temperature} {top_p} token {token}: {frequency} vs {p}"
                    );
                }
            }
        }
        // The same seed and index select the same token; a nonfinite logit fails.
        let sampling = Sampling {
            temperature: 1.0,
            top_p: 0.9,
            seed: 77,
        };
        let logits: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.25).collect();
        assert_eq!(
            sample(&logits, &sampling, 3).unwrap(),
            sample(&logits, &sampling, 3).unwrap()
        );
        assert_ne!(
            sample(&logits, &sampling, 3).unwrap(),
            sample(
                &logits,
                &Sampling {
                    seed: 78,
                    ..sampling
                },
                3
            )
            .unwrap()
        );
        for invalid in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut bad = logits.clone();
            bad[3] = invalid;
            assert!(sample(&bad, &sampling, 1).is_err());
        }
    }

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
    fn carried_state_takes_a_chunk_only_inside_the_prompt() {
        // Dense: a chunk is taken while the prompt's last token stays for decode and the chunk
        // fits the context, rows past the prompt included.
        assert!(chunk_fits(0, 2, 4, 8, false));
        assert!(chunk_fits(4, 8, 4, 8, false));
        assert!(!chunk_fits(5, 8, 4, 8, false));
        assert!(!chunk_fits(7, 8, 4, 8, false));
        // Carried: every row must be a prompt token before the last.
        assert!(!chunk_fits(0, 2, 4, 8, true));
        assert!(chunk_fits(0, 5, 4, 8, true));
        assert!(!chunk_fits(0, 4, 4, 8, true));
        assert!(chunk_fits(4, 9, 4, 16, true));
        assert!(!chunk_fits(4, 8, 4, 16, true));
    }

    #[test]
    fn reject_invalid_geometry_and_prompt_before_submission() {
        let mut config = Config {
            version: GENERATION_FORMAT,
            context_length: 8,
            vocab_size: 32,
            prefill_size: 1,
            eos_ids: vec![31],
            state: vec!["state".into(), "history".into()],
        };
        config.validate().unwrap();
        config.state.push("state".into());
        assert!(config.validate().is_err());
        config.state.clear();
        config.validate_prompt(&[0, 31]).unwrap();
        for prompt in [vec![], vec![-1], vec![32], vec![1; 9]] {
            assert!(config.validate_prompt(&prompt).is_err());
        }
        for size in [0, 9] {
            config.prefill_size = size;
            assert!(config.validate().is_err());
        }
        config.prefill_size = 1;
        config.context_length = MAX_CONTEXT + 1;
        assert!(config.validate().is_err());
        config.context_length = 8;
        config.eos_ids = vec![32];
        assert!(config.validate().is_err());
    }
}
