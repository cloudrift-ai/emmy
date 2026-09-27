//! Execute a paged program on a real device.
//!
//! The pack is produced by `scripts/export_paged_pack.py`, which compiles a step that writes a
//! chunk of a KV cache at a runtime position. Point `EMMY_PAGED_PACK` at it to run this; without
//! it the test skips, because the pack needs a compiler and this crate has none.

use emmy_runtime::artifact::{Artifact, Env};
use emmy_runtime::cuda::{Device, Executor};
use std::path::PathBuf;

/// Mirrors the exporter: a (1, 2, 32, 8) f32 cache of four 8-key pages, filled four keys at a
/// time by a step whose own output is (1, 2, 4, 8).
const CHUNK: usize = 4;
const PAGE: usize = 8;
const KEYS: usize = 32;
const ROW: usize = 8;
const KV_HEADS: usize = 2;

#[test]
fn a_step_writes_its_chunk_into_the_pages_it_is_given() {
    let Some(root) = std::env::var_os("EMMY_PAGED_PACK").map(PathBuf::from) else {
        eprintln!("skipped: set EMMY_PAGED_PACK to a pack from scripts/export_paged_pack.py");
        return;
    };
    let device = Device::new(0).expect("no CUDA device");
    let artifact = Artifact::load(&root, "step").expect("load pack");
    let mut executor = Executor::load(&device, artifact).expect("load program");

    // The runtime owns the cache: four pages, sized by what the plan says a page holds.
    assert_eq!(
        executor.page_bytes("cache").expect("cache is paged"),
        KV_HEADS * PAGE * ROW * size_of::<f32>()
    );
    executor
        .alloc_pages("cache", KEYS / PAGE)
        .expect("allocate pages");

    // Every step reads the same input buffer and lands its rows at a different position.
    let mut expected = vec![0f32; KV_HEADS * KEYS * ROW];
    for (step, past) in (0..KEYS).step_by(CHUNK).enumerate() {
        let chunk: Vec<f32> = (0..KV_HEADS * CHUNK * ROW)
            .map(|i| (step * 1000 + i) as f32)
            .collect();
        for head in 0..KV_HEADS {
            for key in 0..CHUNK {
                for lane in 0..ROW {
                    let from = (head * CHUNK + key) * ROW + lane;
                    // The step applies the pack's own math; only where it lands is under test.
                    expected[(head * KEYS + past + key) * ROW + lane] = chunk[from].tanh();
                }
            }
        }
        // The position a step writes at is an ordinary runtime argument of the environment.
        executor
            .set_env(Env::from([("past".to_owned(), past as i64)]))
            .expect("set past");
        executor
            .bind("chunk", bytemuck_cast(&chunk))
            .expect("bind chunk");
        executor.run_once().expect("run step");
    }

    // Read the cache back page by page and compare against the same fill done on the host.
    let mut cache = Vec::with_capacity(expected.len());
    for page in 0..KEYS / PAGE {
        let bytes = executor.read_page("cache", page).expect("read page");
        cache.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        );
    }
    // Pages are laid out head-major inside a page, so compare per page rather than flat. The
    // tolerance covers the device's tanh against the host's; a wrong page or offset is off by
    // whole rows, not by a fifth digit.
    for (page_index, page) in cache
        .as_chunks::<{ KV_HEADS * PAGE * ROW }>()
        .0
        .iter()
        .enumerate()
    {
        for head in 0..KV_HEADS {
            for key in 0..PAGE {
                let absolute = page_index * PAGE + key;
                let got = &page[(head * PAGE + key) * ROW..][..ROW];
                let want = &expected[(head * KEYS + absolute) * ROW..][..ROW];
                assert!(
                    got.iter()
                        .zip(want)
                        .all(|(g, w)| (g - w).abs() <= 1e-4 * w.abs().max(1.0)),
                    "page {page_index} head {head} key {key}: {got:?} != {want:?}"
                );
            }
        }
    }
}

fn bytemuck_cast(values: &[f32]) -> &[u8] {
    // Safety: f32 has no padding and no invalid bit patterns, so its bytes are readable as u8.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
