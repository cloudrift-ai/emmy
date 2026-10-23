//! The pager: the owner of a paged buffer's pages and tables.
//!
//! A paged buffer declared with fewer slots than pages keeps its pages where the host put them —
//! pinned, mapped host memory — and a pool of device slots the pages move into as launches touch
//! them. The move is the paging program's (`emmy/compiler/backend/paging.py`), spliced into the
//! plan ahead of every launch that touches the buffer; the pager allocates what that program
//! reads and writes, fills the tables for a buffer nothing has touched, and binds them to an
//! executor's plan: the tables as the inputs named after the buffer, the page table under the
//! name the kernels take it by. Nothing here runs on the device; the tables change only through
//! the program, on the stream, so a captured graph records every move.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};

use crate::artifact::Paging;
use crate::cuda::{Device, DeviceMemory, Executor};

/// The tables, by the suffix their plan buffer takes after the paged buffer's name.
pub const TABLES: [&str; 8] = [
    "table",
    "slot_page",
    "page_slot",
    "age",
    "host",
    "slot_address",
    "clock",
    "slots",
];

const POINTER_BYTES: usize = std::mem::size_of::<u64>();

struct Buffer {
    pages: usize,
    slots: usize,
    page_bytes: usize,
    tables: BTreeMap<&'static str, DeviceMemory>,
}

impl Buffer {
    /// A table's byte length: the entries the paging program takes it with.
    fn len(&self, table: &str) -> usize {
        match table {
            "table" | "page_slot" => (self.pages + 1) * POINTER_BYTES,
            "slot_page" | "age" => (self.slots + 1) * POINTER_BYTES,
            "host" => (self.pages + 1) * POINTER_BYTES,
            "slot_address" => self.slots * POINTER_BYTES,
            "clock" => POINTER_BYTES,
            _ => (self.slots + 1) * self.page_bytes,
        }
    }
}

#[derive(Default)]
pub struct Pager {
    buffers: BTreeMap<String, Buffer>,
}

fn i64s(values: impl IntoIterator<Item = i64>) -> Vec<u8> {
    values.into_iter().flat_map(i64::to_le_bytes).collect()
}

impl Pager {
    /// Declare `name`: the geometry `(pages, page_bytes, slots)` — that many pages of that many
    /// bytes, living at `host` (pinned, mapped host memory, page after page), with that many
    /// device slots. Every table starts as a buffer nothing has touched: each page at its host
    /// address, every slot empty, the clock at one. One spare slot past the last takes the stores
    /// the paging program points past the end.
    pub fn declare(
        &mut self,
        device: &Device,
        name: &str,
        host: u64,
        geometry: (usize, usize, usize),
    ) -> Result<()> {
        let (pages, page_bytes, slots) = geometry;
        ensure!(
            pages > 0 && slots > 0 && page_bytes > 0,
            "{name}: a paged buffer has pages, slots and bytes"
        );
        ensure!(
            !self.buffers.contains_key(name),
            "{name} is already declared"
        );
        let slot_memory = DeviceMemory::zeroed(device, (slots + 1) * page_bytes)?;
        let slot_address: Vec<i64> = (0..slots)
            .map(|slot| (slot_memory.ptr() + (slot * page_bytes) as u64) as i64)
            .collect();
        let spare = (slot_memory.ptr() + (slots * page_bytes) as u64) as i64;
        let host_address: Vec<i64> = (0..pages)
            .map(|page| (host + (page * page_bytes) as u64) as i64)
            .collect();
        let contents: [(&'static str, Vec<u8>); 7] = [
            ("table", i64s(host_address.iter().copied().chain([spare]))),
            (
                "slot_page",
                i64s(std::iter::repeat_n(pages as i64, slots + 1)),
            ),
            (
                "page_slot",
                i64s(std::iter::repeat_n(slots as i64, pages + 1)),
            ),
            ("age", i64s(std::iter::repeat_n(0, slots + 1))),
            ("host", i64s(host_address.iter().copied().chain([spare]))),
            ("slot_address", i64s(slot_address.iter().copied())),
            ("clock", i64s([1])),
        ];
        let mut tables = BTreeMap::new();
        for (table, bytes) in contents {
            let memory = DeviceMemory::zeroed(device, bytes.len())?;
            memory.write(&bytes)?;
            tables.insert(table, memory);
        }
        tables.insert("slots", slot_memory);
        self.buffers.insert(
            name.to_owned(),
            Buffer {
                pages,
                slots,
                page_bytes,
                tables,
            },
        );
        Ok(())
    }

    /// A table's device address and byte length.
    pub fn table(&self, name: &str, table: &str) -> Result<(u64, usize)> {
        let buffer = self.buffer(name)?;
        let memory = buffer
            .tables
            .get(table)
            .with_context(|| format!("{name} has no table {table}"))?;
        Ok((memory.ptr(), buffer.len(table)))
    }

    /// `name`'s tables as the regions of the inputs a plan names after the buffer, for a host that
    /// lends them to an executor as it loads.
    pub fn regions(&self, name: &str) -> Result<BTreeMap<String, (u64, usize)>> {
        TABLES
            .iter()
            .map(|table| Ok((format!("input:{name}__{table}"), self.table(name, table)?)))
            .collect()
    }

    /// Bind `name`'s tables into `executor`: each as the input the plan names after the buffer,
    /// the page table as what the kernels read the buffer through, and the host table as the page
    /// table of the pages' view the copy reads each page at its home through.
    pub fn bind(&self, executor: &mut Executor, name: &str) -> Result<()> {
        let buffer = self.buffer(name)?;
        for (region, (ptr, len)) in self.regions(name)? {
            executor.set_region(&region, ptr, len, true)?;
        }
        executor.set_external(
            &Paging::table(name),
            buffer.tables["table"].ptr(),
            (buffer.pages + 1) * POINTER_BYTES,
        )?;
        executor.set_external(
            &Paging::table(&format!("{name}__view")),
            buffer.tables["host"].ptr(),
            (buffer.pages + 1) * POINTER_BYTES,
        )
    }

    fn buffer(&self, name: &str) -> Result<&Buffer> {
        self.buffers
            .get(name)
            .with_context(|| format!("{name} is not a paged buffer the pager declared"))
    }
}
