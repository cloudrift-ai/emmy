//! A generation pack's vision tower: one compiled program per image width, which writes an
//! image's rows into the `images` buffer decode owns and both token programs merge in place of
//! the image tokens' embeddings, and each step's M-RoPE positions and image rows. The host
//! writes every input the tower reads (`serving/native/ARCHITECTURE.md`, "Images"); nothing
//! here knows a model.

use crate::{
    artifact::{Artifact, Paging},
    cuda::{Device, Executor},
    pager::Pager,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

const INTEGER_BYTES: usize = size_of::<i64>();
const TABLE_ENTRY_BYTES: u64 = size_of::<u64>() as u64;
/// The tower program's inputs, in order; its one output is the image rows.
pub const INPUTS: [&str; 5] = [
    "patches",
    "pos_index",
    "pos_weight",
    "positions",
    "patches_valid",
];
pub const IMAGES: &str = "images";
/// The token programs' inputs a pack with a tower adds: each row's M-RoPE position and the
/// image row it takes in place of its embedding.
pub const STEP_INPUTS: [&str; 2] = ["mrope", "image_rows"];

/// The tower's contract in the pack key: its widths in image tokens, the image token the
/// prompt carries for each of an image's rows, the geometry the host writes the inputs at, and
/// the pools of paged block weights with each block's buffer that reads its page.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vision {
    pub widths: Vec<usize>,
    pub image_token_id: i64,
    pub vision_start_token_id: i64,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub position_grid: usize,
    pub pools: BTreeMap<String, Vec<String>>,
}

/// One image of a request, as the host prepared it: how many image tokens it merges into, the
/// tower program's inputs at the narrowest width that holds them, and the M-RoPE position of
/// every prompt token.
pub struct Image {
    pub tokens: usize,
    pub inputs: BTreeMap<String, Vec<u8>>,
    pub positions: Vec<[i64; 3]>,
}

pub(crate) struct Tower {
    vision: Vision,
    /// By width, narrowest first.
    programs: Vec<(usize, Executor)>,
}

/// A request's per-row inputs: the prompt's M-RoPE positions, how far a generated token's
/// position sits from its place in the context, and which image row each prompt token takes.
pub(crate) struct Rows {
    positions: Vec<[i64; 3]>,
    delta: i64,
    images: Vec<i64>,
}

impl Vision {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.widths.is_empty() && self.widths.windows(2).all(|w| w[0] < w[1]),
            "the tower's widths must ascend"
        );
        ensure!(
            self.patch_size > 0 && self.merge_size > 0 && self.position_grid > 0,
            "invalid tower geometry"
        );
        Ok(())
    }

    /// The narrowest width holding an image of `tokens` image tokens.
    pub fn width(&self, tokens: usize) -> Result<usize> {
        self.widths
            .iter()
            .copied()
            .find(|&width| width >= tokens)
            .with_context(|| {
                format!(
                    "an image of {tokens} image tokens is wider than the tower ({})",
                    self.widths.last().copied().unwrap_or(0)
                )
            })
    }
}

impl Tower {
    /// The tower's programs, checked against the contract and decode's `images` buffer before
    /// any allocation.
    pub(crate) fn artifacts(
        root: &Path,
        vision: &Vision,
        decode: &Artifact,
    ) -> Result<Vec<(usize, Artifact)>> {
        vision.validate()?;
        let images = decode.program.buffer(IMAGES)?;
        ensure!(images.role == "output", "decode's images are not its own");
        let mut artifacts = Vec::new();
        for &width in &vision.widths {
            let artifact = Artifact::load(root, &format!("tower{width}"))?;
            ensure!(
                artifact.program.inputs == INPUTS && artifact.program.outputs == [IMAGES],
                "invalid tower interface"
            );
            let output = artifact.program.buffer(IMAGES)?;
            let (shape, full) = (output.static_shape(), images.static_shape());
            ensure!(
                output.dtype == images.dtype
                    && shape.as_ref().zip(full.as_ref()).is_some_and(|(s, f)| {
                        s.len() == 2 && s[0] == width as i64 && s[1..] == f[1..] && s[0] <= f[0]
                    }),
                "tower{width} writes no leading rows of decode's images"
            );
            artifacts.push((width, artifact));
        }
        Ok(artifacts)
    }

    /// Load the tower's programs onto decode's `images` and the pager's pools: each lends its
    /// tables, and each block's buffer reads its page through the block's entry of its pool's
    /// page table. A constant the widest program shares with a narrower one is one file of the
    /// pack, uploaded once and lent.
    pub(crate) fn load(
        device: &Device,
        vision: &Vision,
        artifacts: Vec<(usize, Artifact)>,
        decode: &Executor,
        pager: &Pager,
    ) -> Result<Self> {
        let images = decode.buffer(IMAGES)?;
        let mut programs: Vec<(usize, Executor)> = Vec::new();
        // The widest program's binding files, which a narrower one's constants may share.
        let mut widest_files = BTreeMap::new();
        for (width, mut artifact) in artifacts.into_iter().rev() {
            let layout = artifact.program.layout(&artifact.program.default_env())?;
            let mut lent = BTreeMap::new();
            lent.insert(
                layout.buffers[IMAGES].region.clone(),
                (images.ptr, images.bytes),
            );
            for pool in vision.pools.keys() {
                lent.extend(pager.regions(pool)?);
            }
            if let Some((_, widest)) = programs.last() {
                let shared: Vec<String> = artifact
                    .files
                    .iter()
                    .filter(|(name, path)| widest_files.get(*name) == Some(*path))
                    .map(|(name, _)| name.clone())
                    .collect();
                for name in &shared {
                    let view = widest.buffer(name)?;
                    lent.insert(layout.buffers[name].region.clone(), (view.ptr, view.bytes));
                }
                artifact.files.retain(|name, _| !shared.contains(name));
            } else {
                widest_files = artifact.files.clone();
            }
            let mut executor = Executor::load_with(device, artifact, None, Some(lent))?;
            for (pool, buffers) in &vision.pools {
                pager.bind(&mut executor, pool)?;
                let (table, _) = pager.table(pool, "table")?;
                for (block, buffer) in buffers.iter().enumerate() {
                    executor.set_external(
                        &Paging::table(buffer),
                        table + TABLE_ENTRY_BYTES * block as u64,
                        TABLE_ENTRY_BYTES as usize,
                    )?;
                }
            }
            for (name, paging) in &executor.program().paged {
                let owner = name.strip_suffix("__view").unwrap_or(name);
                let pooled = vision.pools.contains_key(owner)
                    || vision.pools.values().flatten().any(|b| b == name);
                ensure!(
                    paging.slots.is_none() || pooled,
                    "paged buffer {name} of tower{width} has no pool"
                );
            }
            programs.insert(0, (width, executor));
        }
        Ok(Self {
            vision: vision.clone(),
            programs,
        })
    }

    /// Run the tower on `image`, writing its rows into decode's `images`.
    pub(crate) fn encode(&mut self, image: &Image, capture: bool) -> Result<()> {
        let width = self.vision.width(image.tokens)?;
        let (_, executor) = self
            .programs
            .iter_mut()
            .find(|(w, _)| *w == width)
            .context("no tower program at that width")?;
        ensure!(
            image.inputs.len() == INPUTS.len()
                && INPUTS.iter().all(|name| image.inputs.contains_key(*name)),
            "an image names the tower's inputs {INPUTS:?}"
        );
        for (name, bytes) in &image.inputs {
            let view = executor.buffer(name)?;
            ensure!(
                bytes.len() == view.bytes,
                "tower{width} input {name} takes {} bytes, not {}",
                view.bytes,
                bytes.len()
            );
            executor.bind(name, bytes)?;
        }
        executor.advance(capture)
    }

    pub(crate) fn vision(&self) -> &Vision {
        &self.vision
    }
}

impl Rows {
    /// A text prompt's rows: every M-RoPE position the token's own, no image row.
    pub(crate) fn text(length: usize) -> Self {
        Self {
            positions: (0..length as i64).map(|p| [p, p, p]).collect(),
            delta: 0,
            images: vec![-1; length],
        }
    }

    /// A prompt with one image: its tokens' M-RoPE positions, and the image rows in order over
    /// the prompt's image tokens, as many as the image merges into. A generated token's
    /// position is its place in the context plus the prompt's last position's lead over it, as
    /// vLLM continues M-RoPE past the prompt.
    pub(crate) fn image(prompt: &[i64], image: &Image, vision: &Vision) -> Result<Self> {
        ensure!(
            image.positions.len() == prompt.len(),
            "an image request gives every prompt token an M-RoPE position"
        );
        let mut next = 0;
        let images: Vec<i64> = prompt
            .iter()
            .map(|&token| {
                if token == vision.image_token_id {
                    next += 1;
                    next - 1
                } else {
                    -1
                }
            })
            .collect();
        ensure!(
            next == image.tokens as i64,
            "the prompt holds {next} image tokens for an image of {}",
            image.tokens
        );
        let last = image
            .positions
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(-1);
        Ok(Self {
            positions: image.positions.clone(),
            delta: last + 1 - prompt.len() as i64,
            images,
        })
    }

    /// Bind the `rows` rows at `position` into `executor`: past the prompt a row takes the
    /// generated position and no image row.
    pub(crate) fn bind(&self, executor: &mut Executor, position: usize, rows: usize) -> Result<()> {
        let mut mrope = Vec::with_capacity(rows * 3 * INTEGER_BYTES);
        let mut images = Vec::with_capacity(rows * INTEGER_BYTES);
        for row in position..position + rows {
            let at = row as i64 + self.delta;
            let (three, image) = self
                .positions
                .get(row)
                .map_or(([at, at, at], -1), |&p| (p, self.images[row]));
            three.iter().for_each(|p| mrope.extend(p.to_le_bytes()));
            images.extend(image.to_le_bytes());
        }
        executor.bind(STEP_INPUTS[0], &mrope)?;
        executor.bind(STEP_INPUTS[1], &images)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vision() -> Vision {
        Vision {
            widths: vec![4, 16],
            image_token_id: 9,
            vision_start_token_id: 8,
            patch_size: 2,
            temporal_patch_size: 2,
            merge_size: 2,
            position_grid: 4,
            pools: BTreeMap::new(),
        }
    }

    #[test]
    fn an_image_takes_the_narrowest_width_that_holds_it() {
        let vision = vision();
        assert_eq!(vision.width(1).unwrap(), 4);
        assert_eq!(vision.width(4).unwrap(), 4);
        assert_eq!(vision.width(5).unwrap(), 16);
        assert!(vision.width(17).is_err());
    }

    #[test]
    fn image_rows_follow_the_image_tokens_and_positions_continue_past_the_prompt() {
        let prompt = [1, 8, 9, 9, 9, 9, 2, 3];
        let positions = vec![
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 3],
            [2, 3, 2],
            [2, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ];
        let image = Image {
            tokens: 4,
            inputs: BTreeMap::new(),
            positions,
        };
        let rows = Rows::image(&prompt, &image, &vision()).unwrap();
        assert_eq!(rows.images, [-1, -1, 0, 1, 2, 3, -1, -1]);
        // The last prompt position is 5 at index 7: a generated token at 8 sits at 6.
        assert_eq!(rows.delta, -2);
        let short = Image { tokens: 3, ..image };
        assert!(Rows::image(&prompt, &short, &vision()).is_err());
        let text = Rows::text(3);
        assert_eq!(text.positions[2], [2, 2, 2]);
        assert_eq!((text.delta, text.images), (0, vec![-1, -1, -1]));
    }
}
