//! Single-request cached generation over a compiler-prepared token-step program. The step
//! computes the logits and the greedy token on the device; sampling at positive temperature
//! happens here, on the host, so nothing the device runs is hand-written. The prompt stays on the
//! host: each step uploads the tokens it embeds. A pack with a drafter also holds a verify
//! program, which at temperature zero checks the last step's drafts and drafts the next ones,
//! deciding on the device which drafts stand.

use crate::{
    artifact::{Artifact, Env, Paging, member},
    cuda::{Device, DeviceMemory, Executor, HostMemory},
    pager::Pager,
    vision::{Image, Rows, STEP_INPUTS, Tower, Vision},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, Instant};

const GENERATION_FORMAT: u32 = 13;
const DEFAULT_TEMPERATURE: f64 = 0.0;
const DEFAULT_TOP_P: f64 = 1.0;
const DEFAULT_SEED: u64 = 0;
const TOKEN_BYTES: usize = size_of::<i64>();
/// The longest context a pack may declare: the window of the checkpoints native serves.
const MAX_CONTEXT: usize = 262_144;
/// The symbolic axis attention reads the cache to: the leading positions a step attends over.
const WINDOW: &str = "cache_tokens";
/// The window grows in steps of this many positions, so a request captures a graph per step.
const WINDOW_STEP: usize = 4096;
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
    /// A pack that answers images: its vision tower (`vision.rs`).
    #[serde(default)]
    pub vision: Option<Vision>,
    /// The tokens the verify program drafts a step; zero for a pack without one.
    pub drafts: usize,
}

/// Whether the prefill program may take the chunk at `position`. A chunk's rows past the
/// prompt embed a padding token and write cache rows the decode step overwrites, so it
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

/// The positions a verify step at `position` reaches: the target's rows, the token it continues
/// from and the `drafts` drafts, and the drafter's later steps, each one past the last accepted
/// row, up to one row short of twice the drafts past `position`.
fn verify_reach(drafts: usize) -> usize {
    2 * drafts
}

/// The static shape of step input `name` in a program of `rows` rows: the position is one
/// word, an M-RoPE position three a row, every other input one a row.
fn input_shape(name: &str, rows: i64) -> Vec<i64> {
    match name {
        "position" => vec![1],
        "mrope" => vec![rows, 3],
        _ => vec![rows],
    }
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
        ensure!(
            verify_reach(self.drafts) < self.context_length,
            "invalid draft count"
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

    /// The decode and prefill programs' step inputs: the position, the tokens they embed, in a
    /// pack with a drafter the token after each of them, which the drafter embeds as it writes its
    /// cache, and in a pack with a tower each row's M-RoPE position and image row.
    fn inputs(&self) -> Vec<&'static str> {
        let next = (self.drafts > 0).then_some("next_tokens");
        ["position", "tokens"]
            .into_iter()
            .chain(next)
            .chain(self.vision.iter().flat_map(|_| STEP_INPUTS))
            .collect()
    }
}

pub struct Generator {
    // The borrowers drop before the decode executor that owns the shared allocations, and all
    // before the pager's tables and slots and the pages they point at.
    tower: Option<Tower>,
    prefill: Option<Executor>,
    verify: Option<Executor>,
    executor: Executor,
    _pager: Pager,
    _homes: Vec<(DeviceMemory, HostMemory)>,
    config: Config,
    sampling: Sampling,
    position: usize,
    prompt: Vec<i64>,
    // The token the last decode step selected, which the next one embeds.
    selected: i64,
    // The drafts the last verify step made for the next one; zeros before the first, which may
    // match by chance, and a draft the target confirms is its greedy token whatever drafted it.
    drafts: Vec<i64>,
    // The rows of logits the last step's tokens came from: a verify step's, or decode's one.
    verified: Option<usize>,
    // A pack with a tower: each row's M-RoPE position and image row, which every step uploads.
    rows: Option<Rows>,
    stopped: bool,
}

/// The cache positions a step of `rows` rows at `position` attends over: every position it reads,
/// in steps of [`WINDOW_STEP`] up to the context.
fn window(position: usize, rows: usize, context: usize) -> i64 {
    (position + rows).next_multiple_of(WINDOW_STEP).min(context) as i64
}

/// Point `executor`'s attention at its [`window`]; a program with no attention has none. Each
/// window replays its own graph, so the env moves only when the window grows.
fn set_window(executor: &mut Executor, position: usize, rows: usize, context: usize) -> Result<()> {
    let window = window(position, rows, context);
    match executor.env().get(WINDOW) {
        Some(&current) if current != window => {
            executor.set_env(Env::from([(WINDOW.to_owned(), window)]))
        }
        _ => Ok(()),
    }
}

/// Check that `artifact` has the step buffer `name` with this dtype, role and static shape.
fn check_buffer(
    artifact: &Artifact,
    name: &str,
    dtype: &str,
    role: &str,
    shape: &[i64],
) -> Result<()> {
    let buffer = artifact
        .program
        .buffer(name)
        .with_context(|| format!("missing generation buffer {name}"))?;
    ensure!(
        buffer.dtype == dtype
            && buffer.role == role
            && buffer.static_shape().as_deref() == Some(shape),
        "invalid generation buffer {name}"
    );
    Ok(())
}

/// The buffers `borrower` takes from decode's `owner` as it loads: every output decode carries
/// (the cache, the state), constant whose binding is decode's file, and pager table, split into
/// the flat ones it lends as regions and the paged ones whose page tables it lends. A program's
/// step inputs and outputs are its own: the host binds and reads them every step.
fn shared(owner: &Artifact, borrower: &Artifact) -> Result<(Vec<String>, Vec<String>)> {
    let (mut flat, mut paged) = (Vec::new(), Vec::new());
    for buffer in &borrower.program.buffers {
        let own = &borrower.program;
        if buffer.role == "scratch"
            || own.inputs.contains(&buffer.name)
            || own.outputs.contains(&buffer.name)
            || (buffer.role == "output"
                && !owner
                    .program
                    .buffer(&buffer.name)
                    .is_ok_and(|b| b.role == "output"))
        {
            continue;
        }
        // A constant decode and the borrower share is one file of the pack.
        if buffer.role == "constant"
            && owner.files.get(&buffer.name) != borrower.files.get(&buffer.name)
        {
            continue;
        }
        let original = owner
            .program
            .buffer(&buffer.name)
            .context("missing shared buffer")?;
        ensure!(
            buffer.dtype == original.dtype
                && buffer.role == original.role
                && buffer.static_shape() == original.static_shape()
                && own.paged.get(&buffer.name) == owner.program.paged.get(&buffer.name),
            "invalid shared buffer {}",
            buffer.name
        );
        // A paged buffer lends its page table where a flat one lends its region.
        if owner.program.paged.contains_key(&buffer.name) {
            paged.push(buffer.name.clone());
        } else {
            flat.push(buffer.name.clone());
        }
    }
    Ok((flat, paged))
}

/// Load `artifact` onto decode's memory: each buffer it shares (`shared`) is lent as it loads,
/// so nothing shared is allocated or uploaded twice.
fn borrow(
    device: &Device,
    owner: &Executor,
    artifact: Artifact,
    (flat, paged): (Vec<String>, Vec<String>),
) -> Result<Executor> {
    let layout = artifact.program.layout(&artifact.program.default_env())?;
    let mut lent = BTreeMap::new();
    for name in &flat {
        let view = owner.buffer(name)?;
        lent.insert(layout.buffers[name].region.clone(), (view.ptr, view.bytes));
    }
    let mut borrower = Executor::load_with(device, artifact, None, Some(lent))?;
    for name in paged {
        let (ptr, len) = owner.page_table(&name)?;
        borrower.set_external(&Paging::table(&name), ptr, len)?;
    }
    Ok(borrower)
}

/// The token after each of `rows` rows at `position`, which the drafter embeds as it writes its
/// cache: the prompt's, else token zero, whose cache row a later step rewrites.
fn next_tokens(prompt: &[i64], position: usize, rows: usize) -> Vec<u8> {
    (1..=rows)
        .flat_map(|row| {
            prompt
                .get(position + row)
                .copied()
                .unwrap_or(0)
                .to_le_bytes()
        })
        .collect()
}

/// Little-endian i64 words.
fn words(bytes: &[u8]) -> Vec<i64> {
    bytes
        .as_chunks::<TOKEN_BYTES>()
        .0
        .iter()
        .map(|word| i64::from_le_bytes(*word))
        .collect()
}

impl Generator {
    /// Load the pack at `root`; with `draft` off a pack's verify program stays unloaded and every
    /// step decodes one token.
    pub fn load(device: &Device, root: &Path, draft: bool) -> Result<Self> {
        let root = &root.canonicalize()?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let config: Config = serde_json::from_value(manifest["key"]["generation"].clone())?;
        config.validate()?;
        let artifact = Artifact::load(root, PROGRAM)?;
        let (vocab, inputs) = (config.vocab_size as i64, config.inputs());
        // Fixed interface names and geometry are part of the generation contract. Check before GPU allocation.
        for name in &inputs {
            check_buffer(&artifact, name, "i64", "input", &input_shape(name, 1))?;
        }
        check_buffer(&artifact, "logits", "f32", "output", &[1, vocab])?;
        check_buffer(&artifact, "token", "i64", "output", &[1])?;
        ensure!(
            artifact.program.inputs == inputs,
            "invalid generation inputs"
        );
        ensure!(
            artifact.program.outputs == ["logits", "token"],
            "invalid generation outputs"
        );
        let mut prefill_artifact = if config.prefill_size > 1 {
            let prefill = Artifact::load(root, "prefill")?;
            for name in &inputs {
                let shape = input_shape(name, config.prefill_size as i64);
                check_buffer(&prefill, name, "i64", "input", &shape)?;
            }
            ensure!(
                prefill.program.inputs == inputs && prefill.program.outputs.is_empty(),
                "invalid prefill interface"
            );
            Some(prefill)
        } else {
            None
        };
        // A verify step's rows: the token it continues from and the drafts.
        let rows = config.drafts as i64 + 1;
        let mut verify_artifact = if draft && config.drafts > 0 {
            let verify = Artifact::load(root, "verify")?;
            let drafted: Vec<String> = (1..=config.drafts).map(|j| format!("draft{j}")).collect();
            // Its rows are past the prompt: with a tower it takes their M-RoPE positions alone.
            let inputs: Vec<&str> = ["position", "tokens"]
                .into_iter()
                .chain(config.vision.iter().map(|_| STEP_INPUTS[0]))
                .collect();
            for name in &inputs {
                check_buffer(&verify, name, "i64", "input", &input_shape(name, rows))?;
            }
            check_buffer(&verify, "logits", "f32", "output", &[rows, vocab])?;
            check_buffer(&verify, "token", "i64", "output", &[rows])?;
            for name in std::iter::once("accepted").chain(drafted.iter().map(String::as_str)) {
                check_buffer(&verify, name, "i64", "output", &[1])?;
            }
            let outputs = ["logits", "token", "accepted"]
                .into_iter()
                .map(str::to_owned)
                .chain(drafted);
            ensure!(
                verify.program.inputs == inputs
                    && verify.program.outputs.iter().cloned().eq(outputs),
                "invalid verify interface"
            );
            Some(verify)
        } else {
            None
        };
        let prefill_shared = prefill_artifact
            .as_ref()
            .map(|p| shared(&artifact, p))
            .transpose()?;
        let verify_shared = verify_artifact
            .as_ref()
            .map(|v| shared(&artifact, v))
            .transpose()?;
        // A borrower reads none of the files it shares with decode: decode uploads them and lends them.
        for (borrower, shared) in [
            (&mut prefill_artifact, &prefill_shared),
            (&mut verify_artifact, &verify_shared),
        ] {
            if let (Some(borrower), Some((flat, _))) = (borrower, shared) {
                borrower.files.retain(|name, _| !flat.contains(name));
            }
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
        // their paging declares. One pager serves every program: decode loads with its tables
        // lent, and the others share them with decode's other buffers.
        let (mut pager, mut homes, mut lent) = (Pager::default(), Vec::new(), BTreeMap::new());
        let towers = match &config.vision {
            Some(vision) => Tower::artifacts(root, vision, &artifact)?,
            None => Vec::new(),
        };
        let mut held = BTreeSet::new();
        for (name, entry) in manifest["pages"].as_object().into_iter().flatten() {
            // The tower's pools are buffers of its programs; every other paged buffer is decode's.
            let owner = std::iter::once(&artifact)
                .chain(towers.iter().map(|(_, tower)| tower))
                .find(|a| a.program.paged.contains_key(name))
                .with_context(|| format!("{name} has pages but is not paged"))?;
            let (buffer, paging) = (owner.program.buffer(name)?, &owner.program.paged[name]);
            let env = owner.program.default_env();
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
            if std::ptr::eq(owner, &artifact) {
                lent.extend(pager.regions(name)?);
            }
            homes.push((on_device, home));
            held.insert(name.clone());
        }
        for program in std::iter::once(&artifact)
            .chain(&prefill_artifact)
            .chain(&verify_artifact)
        {
            for (name, paging) in &program.program.paged {
                let owner = name.strip_suffix("__view").unwrap_or(name);
                ensure!(
                    paging.slots.is_none() || held.contains(owner),
                    "paged buffer {name} has no pages in the pack"
                );
            }
        }
        let decoded: BTreeSet<String> = held
            .iter()
            .filter(|name| artifact.program.paged.contains_key(*name))
            .cloned()
            .collect();
        let mut executor = Executor::load_with(device, artifact, None, Some(lent))?;
        for name in &decoded {
            pager.bind(&mut executor, name)?;
        }
        let tower = match &config.vision {
            Some(vision) => Some(Tower::load(device, vision, towers, &executor, &pager)?),
            None => None,
        };
        let prefill = prefill_artifact
            .zip(prefill_shared)
            .map(|(artifact, shared)| borrow(device, &executor, artifact, shared))
            .transpose()?;
        let verify = verify_artifact
            .zip(verify_shared)
            .map(|(artifact, shared)| borrow(device, &executor, artifact, shared))
            .transpose()?;
        Ok(Self {
            tower,
            prefill,
            verify,
            executor,
            _pager: pager,
            _homes: homes,
            drafts: vec![0; config.drafts],
            config,
            sampling: Sampling::default(),
            position: 0,
            prompt: Vec::new(),
            selected: 0,
            verified: None,
            rows: None,
            stopped: true,
        })
    }

    /// Reset request state. Old cache entries are invisible until overwritten at their absolute
    /// positions, and the state a step carries starts at zero. The prompt stays on the host.
    pub fn start(&mut self, prompt: &[i64], sampling: Sampling) -> Result<()> {
        self.config.validate_prompt(prompt)?;
        sampling.validate()?;
        self.stopped = true;
        self.sampling = sampling;
        for name in &self.config.state {
            self.executor.zero(name)?;
        }
        self.prompt = prompt.to_vec();
        self.rows = self.tower.as_ref().map(|_| Rows::text(prompt.len()));
        self.position = 0;
        self.drafts.fill(0);
        self.stopped = false;
        Ok(())
    }

    /// [`start`](Self::start) a request whose prompt carries `image`: the tower writes its rows,
    /// and every prompt token takes its M-RoPE position and, for an image token, its image row.
    pub fn start_image(
        &mut self,
        prompt: &[i64],
        image: &Image,
        capture: bool,
        sampling: Sampling,
    ) -> Result<()> {
        let tower = self
            .tower
            .as_ref()
            .context("the pack holds no vision tower")?;
        let rows = Rows::image(prompt, image, tower.vision())?;
        self.start(prompt, sampling)?;
        self.stopped = true;
        if let Some(tower) = &mut self.tower {
            tower.encode(image, capture)?;
        }
        self.rows = Some(rows);
        self.stopped = false;
        Ok(())
    }

    /// Consume a prefill chunk, a verify step or one token on the decode program, and return the
    /// tokens it generated. The chunk rule (`chunk_fits`) says where a chunk is taken; at temperature
    /// zero a pack's verify program takes every step from the prompt's last token on, where its rows
    /// fit the context; the decode step covers the rest.
    pub fn advance(&mut self, capture: bool, ignore_eos: bool) -> Result<Vec<i64>> {
        ensure!(!self.stopped, "generation is stopped");
        let (rows, length) = (self.config.prefill_size, self.prompt.len());
        if chunk_fits(
            self.position,
            length,
            rows,
            self.config.context_length,
            !self.config.state.is_empty(),
        ) && let Some(prefill) = &mut self.prefill
        {
            self.stopped = true;
            // The chunk's prompt tokens; a row past the prompt embeds token zero, and nothing reads
            // its cache row before the decode step rewrites it.
            let mut tokens = vec![0; rows * TOKEN_BYTES];
            let chunk = &self.prompt[self.position..length.min(self.position + rows)];
            for (slot, token) in tokens
                .as_chunks_mut::<TOKEN_BYTES>()
                .0
                .iter_mut()
                .zip(chunk)
            {
                slot.copy_from_slice(&token.to_le_bytes());
            }
            set_window(prefill, self.position, rows, self.config.context_length)?;
            prefill.bind("position", &(self.position as i64).to_le_bytes())?;
            prefill.bind("tokens", &tokens)?;
            if self.config.drafts > 0 {
                prefill.bind(
                    "next_tokens",
                    &next_tokens(&self.prompt, self.position, rows),
                )?;
            }
            if let Some(image) = &self.rows {
                image.bind(prefill, self.position, rows)?;
            }
            prefill.advance(capture)?;
            self.position += rows.min(length - self.position - 1);
            self.stopped = false;
            return Ok(Vec::new());
        }
        if self.sampling.temperature == 0.0
            && self.position + 1 >= length
            && self.position + verify_reach(self.config.drafts) <= self.config.context_length
            && self.verify.is_some()
        {
            return self.verify(capture, ignore_eos);
        }
        Ok(self.step(capture, ignore_eos)?.into_iter().collect())
    }

    /// One verify step: the target over the token it continues from and the last step's drafts,
    /// which takes the drafts its greedy choice confirms and carries the state after the last
    /// accepted row, all decided on the device, and the drafter's drafts for the next step. The
    /// tokens are the target's picks through the last accepted row, up to the first EOS.
    fn verify(&mut self, capture: bool, ignore_eos: bool) -> Result<Vec<i64>> {
        self.stopped = true;
        let drafts = self.config.drafts;
        let first = self
            .prompt
            .get(self.position)
            .copied()
            .unwrap_or(self.selected);
        let tokens: Vec<u8> = std::iter::once(first)
            .chain(self.drafts.iter().copied())
            .flat_map(i64::to_le_bytes)
            .collect();
        let verify = self.verify.as_mut().context("no verify program")?;
        set_window(
            verify,
            self.position,
            verify_reach(drafts),
            self.config.context_length,
        )?;
        verify.bind("position", &(self.position as i64).to_le_bytes())?;
        verify.bind("tokens", &tokens)?;
        if let Some(rows) = &self.rows {
            rows.bind(verify, self.position, drafts + 1)?;
        }
        verify.advance(capture)?;
        let picks = words(&verify.output("token")?);
        let accepted = words(&verify.output("accepted")?)[0];
        ensure!(
            (0..=drafts as i64).contains(&accepted),
            "invalid accepted draft count"
        );
        for (j, draft) in self.drafts.iter_mut().enumerate() {
            *draft = words(&verify.output(&format!("draft{}", j + 1))?)[0];
        }
        let vocab = self.config.vocab_size as i64;
        // A nonfinite logit leaves a pick, or a draft, out of range.
        ensure!(
            picks
                .iter()
                .chain(&self.drafts)
                .all(|t| (0..vocab).contains(t)),
            "invalid selected token"
        );
        let picked = &picks[..=accepted as usize];
        self.position += picked.len();
        self.selected = picked[picked.len() - 1];
        let stop = picked
            .iter()
            .position(|token| !ignore_eos && self.config.eos_ids.contains(token));
        let generated = picked[..stop.map_or(picked.len(), |at| at + 1)].to_vec();
        self.verified = Some(generated.len());
        self.stopped = stop.is_some();
        Ok(generated)
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
        // The prompt's token at this position while the prompt lasts, else the last selection.
        let token = self
            .prompt
            .get(self.position)
            .copied()
            .unwrap_or(self.selected);
        self.verified = None;
        set_window(
            &mut self.executor,
            self.position,
            1,
            self.config.context_length,
        )?;
        self.executor
            .bind("position", &(self.position as i64).to_le_bytes())?;
        self.executor.bind("tokens", &token.to_le_bytes())?;
        if self.config.drafts > 0 {
            self.executor
                .bind("next_tokens", &next_tokens(&self.prompt, self.position, 1))?;
        }
        if let Some(rows) = &self.rows {
            rows.bind(&mut self.executor, self.position, 1)?;
        }
        self.executor.advance(capture)?;
        let position = self.position;
        self.position += 1;
        if self.position < self.prompt.len() {
            self.stopped = false;
            return Ok(None);
        }
        let token = if self.sampling.temperature == 0.0 {
            // The program's own greedy pick; a nonfinite logit leaves it out of range.
            words(&self.executor.output("token")?)[0]
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
                (position + 2 - self.prompt.len()) as u64,
            )?
        };
        ensure!(
            token >= 0 && (token as usize) < self.config.vocab_size,
            "invalid selected token"
        );
        self.selected = token;
        self.stopped = !ignore_eos && self.config.eos_ids.contains(&token);
        Ok(Some(token))
    }

    /// Diagnostic transfer only; a decode step at positive temperature downloads them anyway. A
    /// verify step's are the rows of the tokens it generated.
    pub fn logits(&self) -> Result<Vec<u8>> {
        match (self.verified, &self.verify) {
            (Some(rows), Some(verify)) => {
                let mut bytes = verify.output("logits")?;
                bytes.truncate(rows * self.config.vocab_size * LOGIT_BYTES);
                Ok(bytes)
            }
            _ => self.executor.output("logits"),
        }
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
            let generated = self.advance(capture, false)?;
            let at = started.elapsed();
            tokens.extend(generated.into_iter().map(|token| (token, at)));
        }
        tokens.truncate(max_new_tokens);
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
    fn the_drafter_embeds_the_prompt_token_after_each_row() {
        // Past the prompt a row's next token is unknown: token zero, whose cache row a later step rewrites.
        assert_eq!(words(&next_tokens(&[5, 6, 7, 8], 1, 3)), [7, 8, 0]);
        assert_eq!(words(&next_tokens(&[5], 0, 1)), [0]);
    }

    #[test]
    fn the_window_covers_every_position_a_step_reads() {
        assert_eq!(window(0, 1, MAX_CONTEXT), WINDOW_STEP as i64);
        assert_eq!(window(WINDOW_STEP - 1, 1, MAX_CONTEXT), WINDOW_STEP as i64);
        assert_eq!(window(WINDOW_STEP, 1, MAX_CONTEXT), 2 * WINDOW_STEP as i64);
        assert_eq!(
            window(WINDOW_STEP - 64, 64, MAX_CONTEXT),
            WINDOW_STEP as i64
        );
        assert_eq!(
            window(WINDOW_STEP - 63, 64, MAX_CONTEXT),
            2 * WINDOW_STEP as i64
        );
        // A context short of a step, or past the last whole step, caps it.
        assert_eq!(window(3, 1, 8), 8);
        assert_eq!(window(MAX_CONTEXT - 1, 1, MAX_CONTEXT), MAX_CONTEXT as i64);
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
            vision: None,
            drafts: 0,
        };
        config.validate().unwrap();
        // A verify step's reach must stay inside the context.
        config.drafts = 3;
        config.validate().unwrap();
        config.drafts = 4;
        assert!(config.validate().is_err());
        config.drafts = 0;
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
