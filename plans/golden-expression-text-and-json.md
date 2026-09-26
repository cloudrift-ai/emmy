# Golden files: expressions as text, then JSON

Measured 2026-09-25 on the RTX 5090 dev box, after #909 put every stored golden and corpus case on the one wire.

| golden | YAML | YAML parse | JSON | JSON parse | gzip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B FP8, V100 | 13.2 MB | 8.0 s | 6.4 MB | 0.09 s | 0.5 MB |
| Gemma 4 12B, RTX 5090 | 1.4 MB | 0.9 s | 1.3 MB | 0.007 s | 0.06 MB |

- Bytes on disk are not the cost: git stores a blob compressed, so the repository pays about the gzip column. The
  cost is the parse, paid at every load: test collection, and every compile that scopes the file as evidence.
- Three quarters of the FP8 file is Loop IR kernel bodies, and 48% of the pool bytes are index expressions written
  as nested maps: 27 361 expression trees, 3.06 MB, which as text (`((a5 / 128) * 128) + a6`) are 0.56 MB. In the
  Gemma file expressions are 7% of the pools; the FP8 file is the outlier because its whole-layer kernels carry
  unrolled sibling bodies, 3.5 MB of wire for the largest.
- Decoding that largest kernel takes 9 s. The wire walker and its type reflection are 7% of it; the rest builds and
  normalizes the Loop IR (the body ordering pass alone is a quarter). Text expressions shrink what the YAML parser
  and the walker see; they do not touch normalization, which is a separate finding.

## Phase 1 — expressions as text

**Spelling.** The C-like form `pretty()` and the CUDA renderer already print, with parentheses only where the
`_PRECEDENCE` table needs them:

- a var is its name; a builtin (`threadIdx.x`, `blockDim.y`, …) is its name — the reader tells the two apart by the
  fixed builtin name set, and the writer refuses a var that spells like one;
- a literal is written by its dtype: int `3`, float `0.5` / `1.0` (always a point or an exponent), bool `true` /
  `false`. `Literal(2, "float")` is written `2.0` and reads back equal. No stored literal today carries a dtype
  other than its value's kind (checked over every golden and case);
- binary `a5 / 128 * 128 + a6`, with the reader accepting a negative literal on the right (`a1 + -1` is stored today);
  call `min(a, b)`; ternary `c ? a : b`; cast `(f32)a`;
- `FlatIndex` never reaches a stored file (kernel-body only, bound by a `Let`), so it has no spelling.

**Where it plugs in.**

- `Expr` becomes the base class the eight classes already share (`_ExprOps`), not a union of them. The base owns
  `to_wire` (the text) and `from_wire` (the parser). Every field typed `Expr`, `Expr | None` or `tuple[Expr, ...]`
  then writes text through the walker with no walker change: a concrete wire class with its own codec holds its
  payload bare. The seven per-class tags and the Var / Literal / Builtin codecs go.
- The parser is precedence climbing over the `_PRECEDENCE` table, about 80 lines in `expr.py`; a parse error names
  the position in the text.
- `Dim.from_wire` reads `{expr: "num_tokens * 2", hint: 512}` through `Expr.from_wire`, not `decode`, and
  `symbolic_vars` takes the free names of that expression the same way.
- Specialization stops walking the wire for tagged dicts: an index expression is now a string it cannot tell from
  a name. It becomes a typed walk over the decoded objects — every `Expr`-typed field (the walker's type hints say
  which) mapped through substitute and simplify, every `Dim` through the binding — and its two callers, the record's
  program specialization and the freeze's rehint, hand it objects, not wires. This is the one real design change of
  the phase.
- The tune DB's perf rows store the kernel wire as JSON. Bump the DB schema version so a local DB holding the old
  spelling is rebuilt, as the rekey flow does.

**Conversion.** Re-encode every stored file — the fifteen repository goldens, the serving golden, the 215 corpus
cases, any freeze — by `Graph.from_wire(old).to_wire()` per pool entry, the idempotent converter #909 used. Corpus
names are labels since #818, so no regen. The FP8 golden takes about a minute.

**Verification.**

- `tests/compiler/ir/test_wire.py`: a round trip per expression class; the precedence cases `a - (b - c)`,
  `(a + b) * c`, `a ^ b + c`, `a & b * c`; a negative literal; one literal per dtype; a parse error that names its
  position.
- The fixed-point test over every stored kernel and the strict decode of every golden row (`make test`).
- `scripts/digest_kernels.py` at the base commit and after: every rendered kernel source byte-identical.
- The table above re-measured, plus serial collection time. Expected: the FP8 pools fall from 6.3 MB to 3.9 MB in
  JSON terms and further in YAML, since most of the nesting goes; the parse should roughly halve, 62 213 leaves
  becoming 27 361 scalars.

## Phase 2 — JSON

**Decision gate.** After phase 1, time `GoldenFile.load` on the FP8 golden. If the parse is still seconds, do this
phase. If it is under about a second, stop here: the rows block of a YAML file is easier to read and edit, and that is
worth more than the remaining parse time.

**Layout.** A JSON object written by a small writer (about 30 lines) so a diff still lands on one entry: header keys
one per line; under `configs`, one realization per line beneath its config header; `programs` and `loops` one entry
per line, compact. Read with `json.load`, no custom loader. `gpu_name` stays the first line, so the head read in
`repository.py` keeps skipping foreign multi-megabyte files by parsing that one line.

- Extension `.json`. Every stored file is renamed with `git mv`, and every glob and name follows: the records package
  data in `pyproject.toml`, `_repository_golden_paths`, `case_files`, the freeze file names and the LFS pattern in
  `.gitattributes`, the `emmy dataset` globs, the `emmy trace -o` defaults, the `-o fresh.yaml` convention of
  `emmy compile`, README, the ARCHITECTURE files, GLOSSARY, tutorial 07, the `refresh-golden` and `onboard-model`
  skills, and the CI workflows that name golden paths.
- A corpus case carries a leading `#` comment, the evidence note, which JSON cannot hold. It becomes an optional
  `note` field of `GoldenFile`, written right after the header; `leading_comment` and its re-prepend in `write_case`
  go. A win on its own: the note survives every dump instead of being restored by hand.
- `program_text` and `kernel_pool_text`, what `emmy golden kernels` and `--ir loop -o` print, write the same
  one-entry-per-line JSON.
- Removed: the YAML dumper and its flow styling (`_flow`, `_short_flow`, `_style_wire_value`, `_style_program`,
  `_style_config`, `_style_block`, `_dump_block`, about 80 lines), replaced by the writer.
- Lost: a row's knobs one per line (a row is one line); hand edits must mind commas and quotes.

**Verification.** Phase 1's gates, plus the head read on a JSON file, the corpus note round trip, and the table.

## Not in this plan

- A textual Loop IR — statements, bodies and axes as text with a parser, the way LLVM and MLIR store IR. It is the
  right long-term shape for the pools and would make a stored kernel readable and editable, but it needs a statement
  parser and should wait for what the expression parser looks like.
- Decode time. The body ordering pass is a quarter of a large kernel's decode and normalization most of the rest;
  that is a normalization finding, not a format one.
- The FP8 kernels' size. Unrolled sibling bodies are a fusion question.
