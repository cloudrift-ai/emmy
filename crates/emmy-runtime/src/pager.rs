//! The pager: the owner of a paged buffer's pages and tables.
//!
//! A paged buffer declared with fewer slots than pages keeps its pages where the host put them —
//! pinned, mapped host memory — and a pool of device slots the pages move into as launches touch
//! them. The move is the paging program's (`emmy/compiler/backend/paging.py`), spliced into the
//! plan ahead of every launch that touches the buffer; the pager allocates what that program
//! reads and writes, fills the tables for a buffer nothing has touched, and binds them to an
//! executor's plan: the tables as the inputs named after the buffer, the page table under the
//! name the kernels take it by. Nothing here runs on the device; the tables change only through
//! the program, on the stream, so a captured graph records every move. A buffer with no slots
//! has no paging program, since nothing can move: its page table points every page at its home,
//! and the kernels read each page there.

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

/// A paged buffer's tables, each with its byte length: the entries the paging program takes it
/// with. A buffer with no slots holds only its page table.
struct Buffer {
    tables: BTreeMap<&'static str, (DeviceMemory, usize)>,
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
    /// the paging program points past the end. With no slots there is only the page table.
    pub fn declare(
        &mut self,
        device: &Device,
        name: &str,
        host: u64,
        geometry: (usize, usize, usize),
    ) -> Result<()> {
        let (pages, page_bytes, slots) = geometry;
        ensure!(
            pages > 0 && page_bytes > 0,
            "{name}: a paged buffer has pages and bytes"
        );
        ensure!(
            !self.buffers.contains_key(name),
            "{name} is already declared"
        );
        let host_address: Vec<i64> = (0..pages)
            .map(|page| (host + (page * page_bytes) as u64) as i64)
            .collect();
        let mut tables = BTreeMap::new();
        let contents: Vec<(&'static str, Vec<u8>)> = if slots == 0 {
            vec![("table", i64s(host_address))]
        } else {
            let slot_memory = DeviceMemory::zeroed(device, (slots + 1) * page_bytes)?;
            let slot_address: Vec<i64> = (0..slots)
                .map(|slot| (slot_memory.ptr() + (slot * page_bytes) as u64) as i64)
                .collect();
            let spare = (slot_memory.ptr() + (slots * page_bytes) as u64) as i64;
            tables.insert("slots", (slot_memory, (slots + 1) * page_bytes));
            vec![
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
                ("slot_address", i64s(slot_address)),
                ("clock", i64s([1])),
            ]
        };
        for (table, bytes) in contents {
            let memory = DeviceMemory::zeroed(device, bytes.len())?;
            memory.write(&bytes)?;
            tables.insert(table, (memory, bytes.len()));
        }
        self.buffers.insert(name.to_owned(), Buffer { tables });
        Ok(())
    }

    /// A table's device address and byte length.
    pub fn table(&self, name: &str, table: &str) -> Result<(u64, usize)> {
        let (memory, len) = self
            .buffer(name)?
            .tables
            .get(table)
            .with_context(|| format!("{name} has no table {table}"))?;
        Ok((memory.ptr(), *len))
    }

    /// `name`'s tables as the regions of the inputs a plan names after the buffer, for a host that
    /// lends them to an executor as it loads. A buffer with no slots has none: no paging program
    /// takes its tables.
    pub fn regions(&self, name: &str) -> Result<BTreeMap<String, (u64, usize)>> {
        if !self.buffer(name)?.tables.contains_key("slots") {
            return Ok(BTreeMap::new());
        }
        TABLES
            .iter()
            .map(|table| Ok((format!("input:{name}__{table}"), self.table(name, table)?)))
            .collect()
    }

    /// Bind `name`'s tables into `executor`: each as the input the plan names after the buffer,
    /// the page table as what the kernels read the buffer through, and the host table as the page
    /// table of the pages' view the copy reads each page at its home through.
    pub fn bind(&self, executor: &mut Executor, name: &str) -> Result<()> {
        for (region, (ptr, len)) in self.regions(name)? {
            executor.set_region(&region, ptr, len, true)?;
        }
        let (table, len) = self.table(name, "table")?;
        executor.set_external(&Paging::table(name), table, len)?;
        if let Some((host, len)) = self.buffer(name)?.tables.get("host") {
            executor.set_external(&Paging::table(&format!("{name}__view")), host.ptr(), *len)?;
        }
        Ok(())
    }

    fn buffer(&self, name: &str) -> Result<&Buffer> {
        self.buffers
            .get(name)
            .with_context(|| format!("{name} is not a paged buffer the pager declared"))
    }
}
