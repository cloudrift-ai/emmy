//! The image processor of a pack that answers images: an image of a chat message (a base64
//! `data:` URL or an http(s) URL) becomes the vision tower's inputs as the checkpoint's own
//! processor writes them — resized into the processor's pixel bounds on whole merge blocks,
//! normalized, cut into patch rows in merge-block order — beside the grid-derived inputs the
//! tower reads (its position table's interpolation corners and weights, each patch's rotary
//! row and column), and the prompt's image placeholder becomes one image token per merged
//! patch, each prompt token taking its M-RoPE position. vLLM is the reference for every rule.
use anyhow::{Context, Result, anyhow, bail, ensure};
use base64::Engine;
use emmy_runtime::vision::{Image, Vision};
use image::{RgbImage, imageops::FilterType};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// The most bytes an image URL may hand back.
const MAX_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
/// The processor's widest aspect ratio, as transformers' `smart_resize` refuses past it.
const MAX_ASPECT: f64 = 200.0;

/// The checkpoint processor's fields the tower's inputs follow (`preprocessor_config.json`).
#[derive(Debug, Deserialize)]
struct Preprocessor {
    image_mean: [f32; 3],
    image_std: [f32; 3],
    #[serde(default = "rescale")]
    rescale_factor: f32,
    size: Size,
}
fn rescale() -> f32 {
    1.0 / 255.0
}
#[derive(Debug, Deserialize)]
struct Size {
    shortest_edge: u64,
    longest_edge: u64,
}

pub struct Processor {
    vision: Vision,
    /// The fused normalization a pixel value takes: `(value - mean) / std` in FP32 over raw
    /// 0-255 values, as transformers fuses its rescale into the mean and the deviation.
    mean: [f32; 3],
    std: [f32; 3],
    pixels: (u64, u64),
}

/// One image ready for the tower.
pub struct Prepared {
    /// The image's patch grid: height and width in patches.
    pub grid: (usize, usize),
    pub tokens: usize,
    pub inputs: BTreeMap<String, Vec<u8>>,
}

impl Processor {
    /// The processor of the pack at `root` when its key holds a tower, else `None`.
    pub fn load(root: &Path, vision: Option<Vision>) -> Result<Option<Self>> {
        let Some(vision) = vision else {
            return Ok(None);
        };
        let config: Preprocessor =
            serde_json::from_slice(&std::fs::read(root.join("preprocessor_config.json"))?)
                .context("preprocessor_config.json")?;
        ensure!(
            config.size.shortest_edge <= config.size.longest_edge,
            "invalid processor pixel bounds"
        );
        let mean = std::array::from_fn(|c| config.image_mean[c] / config.rescale_factor);
        let std = std::array::from_fn(|c| config.image_std[c] / config.rescale_factor);
        Ok(Some(Self {
            vision,
            mean,
            std,
            pixels: (config.size.shortest_edge, config.size.longest_edge),
        }))
    }

    pub fn image_token(&self) -> i64 {
        self.vision.image_token_id
    }

    /// Read the image at `url` and prepare it.
    pub fn prepare(&self, url: &str) -> Result<Prepared> {
        self.prepare_rgb(rgb(&fetch(url)?)?)
    }

    fn prepare_rgb(&self, image: RgbImage) -> Result<Prepared> {
        let (patch, merge) = (self.vision.patch_size, self.vision.merge_size);
        let (height, width) = smart_resize(
            image.height() as u64,
            image.width() as u64,
            (patch * merge) as u64,
            self.pixels,
        )?;
        let image = if (height, width) == (image.height() as u64, image.width() as u64) {
            image
        } else {
            image::imageops::resize(&image, width as u32, height as u32, FilterType::CatmullRom)
        };
        let grid = (height as usize / patch, width as usize / patch);
        let patches = grid.0 * grid.1;
        let tokens = patches / (merge * merge);
        let rows = self.vision.width(tokens)? * merge * merge;
        let temporal = self.vision.temporal_patch_size;
        let columns = 3 * temporal * patch * patch;
        let mut values = vec![0u16; rows * columns];
        for (row, (y0, x0)) in merged_order(grid, merge).enumerate() {
            let line = &mut values[row * columns..(row + 1) * columns];
            for c in 0..3 {
                for py in 0..patch {
                    for px in 0..patch {
                        let pixel =
                            image.get_pixel((x0 * patch + px) as u32, (y0 * patch + py) as u32);
                        let value = bf16((f32::from(pixel[c]) - self.mean[c]) / self.std[c]);
                        // An image is one frame: every temporal slot of a patch holds it.
                        for t in 0..temporal {
                            line[((c * temporal + t) * patch + py) * patch + px] = value;
                        }
                    }
                }
            }
        }
        let (index, weight) = interpolation(grid, merge, self.vision.position_grid, rows);
        let mut positions = vec![0i64; rows * 2];
        for (row, (y, x)) in merged_order(grid, merge).enumerate() {
            positions[2 * row] = y as i64;
            positions[2 * row + 1] = x as i64;
        }
        let inputs = BTreeMap::from([
            ("patches".to_owned(), words(&values)),
            ("pos_index".to_owned(), integers(&index)),
            ("pos_weight".to_owned(), words(&weight)),
            ("positions".to_owned(), integers(&positions)),
            ("patches_valid".to_owned(), integers(&[patches as i64])),
        ]);
        Ok(Prepared {
            grid,
            tokens,
            inputs,
        })
    }

    /// The prompt with its one image placeholder grown to the image's tokens, and the image
    /// the runtime starts the request with: every prompt token's M-RoPE position, as vLLM
    /// numbers them — text tokens one after another, an image's tokens at its start position
    /// plus their merged row and column, the text after it from past the image's widest side.
    pub fn expand(&self, tokens: &[i64], prepared: Prepared) -> Result<(Vec<i64>, Image)> {
        let image = self.vision.image_token_id;
        ensure!(
            tokens.iter().filter(|&&t| t == image).count() == 1,
            "the prompt holds one image placeholder per image"
        );
        let merge = self.vision.merge_size;
        let (rows, columns) = (prepared.grid.0 / merge, prepared.grid.1 / merge);
        let mut expanded = Vec::with_capacity(tokens.len() + prepared.tokens);
        let mut positions = Vec::with_capacity(tokens.len() + prepared.tokens);
        let mut next = 0i64;
        for &token in tokens {
            if token != image {
                expanded.push(token);
                positions.push([next; 3]);
                next += 1;
                continue;
            }
            for r in 0..rows {
                for c in 0..columns {
                    expanded.push(image);
                    positions.push([next, next + r as i64, next + c as i64]);
                }
            }
            next += rows.max(columns) as i64;
        }
        Ok((
            expanded,
            Image {
                tokens: prepared.tokens,
                inputs: prepared.inputs,
                positions,
            },
        ))
    }
}

/// An image's bytes: a base64 `data:` URL's payload, or the body an http(s) URL answers with.
fn fetch(url: &str) -> Result<Vec<u8>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, data) = rest.split_once(',').context("a data URL holds a comma")?;
        ensure!(header.ends_with(";base64"), "a data URL's image is base64");
        return base64::engine::general_purpose::STANDARD
            .decode(data.trim())
            .map_err(|e| anyhow!("base64: {e}"));
    }
    ensure!(
        url.starts_with("http://") || url.starts_with("https://"),
        "an image URL is a data, http or https URL"
    );
    let mut bytes = Vec::new();
    ureq::get(url)
        .call()
        .map_err(|e| anyhow!("{url}: {e}"))?
        .into_body()
        .into_reader()
        .take(MAX_IMAGE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_IMAGE_BYTES,
        "the image exceeds 64 MiB"
    );
    Ok(bytes)
}

/// An image's RGB pixels; a transparent pixel lands on white, as vLLM converts RGBA to RGB.
fn rgb(bytes: &[u8]) -> Result<RgbImage> {
    let image = image::load_from_memory(bytes).map_err(|e| anyhow!("image: {e}"))?;
    if !image.color().has_alpha() {
        return Ok(image.to_rgb8());
    }
    let rgba = image.to_rgba8();
    Ok(RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let [r, g, b, a] = rgba.get_pixel(x, y).0;
        let over =
            |c: u8| ((u32::from(c) * u32::from(a) + 255 * (255 - u32::from(a)) + 127) / 255) as u8;
        image::Rgb([over(r), over(g), over(b)])
    }))
}

/// transformers' `smart_resize`: the height and width nearest the image's on whole `factor`
/// blocks (Python's round, half to even), scaled into `[min, max]` pixels when outside them.
fn smart_resize(
    height: u64,
    width: u64,
    factor: u64,
    (min, max): (u64, u64),
) -> Result<(u64, u64)> {
    ensure!(height > 0 && width > 0, "an empty image");
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > MAX_ASPECT {
        bail!("an image's aspect ratio is at most {MAX_ASPECT}");
    }
    let mut bar = ((h / f).round_ties_even() * f, (w / f).round_ties_even() * f);
    if bar.0 * bar.1 > max as f64 {
        let beta = (h * w / max as f64).sqrt();
        bar = (
            f.max((h / beta / f).floor() * f),
            f.max((w / beta / f).floor() * f),
        );
    } else if bar.0 * bar.1 < min as f64 {
        let beta = (min as f64 / (h * w)).sqrt();
        bar = ((h * beta / f).ceil() * f, (w * beta / f).ceil() * f);
    }
    Ok((bar.0 as u64, bar.1 as u64))
}

/// Each patch's (row, column) in the order the tower reads its rows: merge block after merge
/// block, row-major within a block.
fn merged_order(
    (height, width): (usize, usize),
    merge: usize,
) -> impl Iterator<Item = (usize, usize)> {
    (0..height * width).map(move |index| {
        let (block, within) = (index / (merge * merge), index % (merge * merge));
        let (row, column) = (block / (width / merge), block % (width / merge));
        (
            row * merge + within / merge,
            column * merge + within % merge,
        )
    })
}

/// The position table's bilinear interpolation onto the patch grid, in the tower's row order,
/// as vLLM's tower computes it: each row's four corners (`[4, rows]`, flattened table indices)
/// and their weights in BF16, from FP32 fractions over a grid scaled to span the table. The
/// rows past the image read corner 0 at weight zero.
fn interpolation(
    grid: (usize, usize),
    merge: usize,
    side: usize,
    rows: usize,
) -> (Vec<i64>, Vec<u16>) {
    let scale = |n: usize| {
        if n > 1 {
            ((side - 1) as f64 / (n - 1) as f64) as f32
        } else {
            0.0
        }
    };
    let (h_scale, w_scale) = (scale(grid.0), scale(grid.1));
    let (mut index, mut weight) = (vec![0i64; 4 * rows], vec![0u16; 4 * rows]);
    for (row, (y, x)) in merged_order(grid, merge).enumerate() {
        let (h, w) = (y as f32 * h_scale, x as f32 * w_scale);
        let (hf, wf) = (h.floor() as usize, w.floor() as usize);
        let (hc, wc) = ((hf + 1).min(side - 1), (wf + 1).min(side - 1));
        let (dh, dw) = (h - hf as f32, w - wf as f32);
        let w11 = dh * dw;
        let (w10, w01) = (dh - w11, dw - w11);
        let w00 = 1.0 - dh - w01;
        for (corner, (at, value)) in [
            (hf * side + wf, w00),
            (hf * side + wc, w01),
            (hc * side + wf, w10),
            (hc * side + wc, w11),
        ]
        .into_iter()
        .enumerate()
        {
            index[corner * rows + row] = at as i64;
            weight[corner * rows + row] = bf16(value);
        }
    }
    (index, weight)
}

/// An FP32 value's BF16 bits, rounded to nearest even.
fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn words(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn integers(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> Processor {
        Processor {
            vision: Vision {
                widths: vec![4, 16],
                image_token_id: 9,
                vision_start_token_id: 8,
                patch_size: 2,
                temporal_patch_size: 2,
                merge_size: 2,
                position_grid: 3,
                pools: BTreeMap::new(),
            },
            mean: [127.5; 3],
            std: [127.5; 3],
            pixels: (16, 256),
        }
    }

    #[test]
    fn resize_rounds_to_whole_blocks_inside_the_pixel_bounds() {
        // Already on whole 32-pixel blocks inside the bounds: unchanged (the probe images).
        assert_eq!(
            smart_resize(448, 448, 32, (65536, 1 << 20)).unwrap(),
            (448, 448)
        );
        assert_eq!(
            smart_resize(1024, 1024, 32, (65536, 1 << 20)).unwrap(),
            (1024, 1024)
        );
        // Python rounds half to even: 1.5 blocks round to 2, 2.5 to 2.
        assert_eq!(smart_resize(48, 80, 32, (0, 1 << 20)).unwrap(), (64, 64));
        // Past the maximum the image shrinks keeping its aspect, below the minimum it grows.
        assert_eq!(
            smart_resize(2048, 1024, 32, (65536, 1 << 20)).unwrap(),
            (1440, 704)
        );
        assert_eq!(
            smart_resize(100, 100, 32, (65536, 1 << 20)).unwrap(),
            (256, 256)
        );
        // transformers' own answers for two more.
        assert_eq!(
            smart_resize(300, 500, 32, (65536, 1 << 20)).unwrap(),
            (288, 512)
        );
        assert_eq!(
            smart_resize(1200, 1600, 32, (65536, 1 << 20)).unwrap(),
            (864, 1152)
        );
        assert!(smart_resize(1, 300, 32, (0, 1 << 20)).is_err());
    }

    #[test]
    fn patches_come_in_merge_block_order_and_tokens_take_mrope_positions() {
        let processor = processor();
        // A 4 x 8 image is a 2 x 4 patch grid of 2-pixel patches: two image tokens.
        let image = RgbImage::from_fn(8, 4, |x, y| image::Rgb([(x * 10 + y) as u8, 255, 0]));
        let prepared = processor.prepare_rgb(image).unwrap();
        assert_eq!((prepared.grid, prepared.tokens), ((2, 4), 2));
        let patches = &prepared.inputs["patches"];
        // Padded to the narrow width: 4 tokens of 4 patches, 3 channels x 2 frames x 2 x 2.
        assert_eq!(patches.len(), 16 * 24 * 2);
        let value = |row: usize, column: usize| {
            u16::from_le_bytes([
                patches[2 * (row * 24 + column)],
                patches[2 * (row * 24 + column) + 1],
            ])
        };
        // Row 2 is the first block's bottom-left patch: pixels (0..2, 2..4); its red top-left pixel is 2.
        assert_eq!(value(2, 0), bf16((2.0 - 127.5) / 127.5));
        // Both frames of a patch hold the same pixels; green 255 normalizes to one.
        assert_eq!(value(2, 4), value(2, 0));
        assert_eq!(value(2, 8), bf16(1.0));
        // Row 4 starts the second block: its top-left patch is at column 2, pixels (4..6, 0..2).
        assert_eq!(value(4, 0), bf16((40.0 - 127.5) / 127.5));
        let positions = &prepared.inputs["positions"];
        let at =
            |row: usize| i64::from_le_bytes(positions[16 * row..16 * row + 8].try_into().unwrap());
        assert_eq!((at(2), at(4)), (1, 0));
        let (prompt, image) = processor.expand(&[1, 8, 9, 7, 6], prepared).unwrap();
        assert_eq!(prompt, [1, 8, 9, 9, 7, 6]);
        // The tokens sit at a 1 x 2 merged grid starting at 2; the text after it from 4.
        assert_eq!(
            image.positions,
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [4, 4, 4],
                [5, 5, 5]
            ]
        );
        assert!(
            processor
                .expand(&[1, 2], processor.prepare_rgb(RgbImage::new(8, 4)).unwrap())
                .is_err()
        );
    }

    #[test]
    fn interpolation_spans_the_table_with_corner_weights_summing_to_one() {
        // A 4 x 4 grid over a 3 x 3 table: the last patch lands on the table's last corner.
        let (index, weight) = interpolation((4, 4), 2, 3, 20);
        let corners = |row: usize| {
            (0..4)
                .map(|c| (index[c * 20 + row], weight[c * 20 + row]))
                .collect::<Vec<_>>()
        };
        assert_eq!(corners(0), [(0, bf16(1.0)), (1, 0), (3, 0), (4, 0)]);
        // Row 15 is patch (3, 3), at table position (2, 2): every corner the last entry.
        assert_eq!(corners(15)[0], (8, bf16(1.0)));
        // Row 1 is patch (0, 1), at 2/3 of the way to column 1.
        let fraction = 2.0f32 / 3.0;
        assert_eq!(corners(1)[0], (0, bf16(1.0 - fraction)));
        assert_eq!(corners(1)[1], (1, bf16(fraction)));
        assert_eq!(corners(19), [(0, 0), (0, 0), (0, 0), (0, 0)]);
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16(1.0), 0x3f80);
        assert_eq!(bf16(-1.0), 0xbf80);
        assert_eq!(bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn a_data_url_carries_base64_and_alpha_lands_on_white() {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            2,
            2,
            image::Rgba([0, 0, 0, 0]),
        ))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png)
        );
        assert_eq!(
            rgb(&fetch(&url).unwrap()).unwrap().get_pixel(1, 1).0,
            [255, 255, 255]
        );
        assert!(fetch("file:///etc/passwd").is_err());
        assert!(fetch("data:image/png,abc").is_err());
    }
}
