# Scope reads on `Body`

The compiler asks its name questions of a scope (a body), but the statement interface answers them one statement at a
time. Callers rebuild the scope answer in Python, and the per-statement answers re-walk the whole subtree on every
call. A profile of one placement gate node (68 s) puts about 16 million calls into Rust, roughly 8 s of the run, most
of it repeated subtree walks and the Python loops around them.

## Goal

Four reads per body, computed once in the crate and cached with the body:

| Read | Meaning | Replaces |
| --- | --- | --- |
| `Body.defined` | every name a statement of the body defines, at any depth | `{n for s in body for n in exposed_defines(s)}`, walks over `defines()`, `definitions.keys()` |
| `Body.free` | every name the body reads from outside, values and coordinates, less what it defines or binds | `set().union(*(free_names(s) for s in body)) - defined` |
| `Body.buffers` | the external buffers read and written, first use order, less kernel-local declarations | `BodyOp._derive_io_names`'s walk |
| `Body.pruned(roots)` | the body cut to what `roots` need, nested bodies cut too, to a fixpoint over carried states | the cut pass's `_pruned` |

`free_names(stmt)` and `exposed_defines(stmt)` keep their meaning and are computed from the nested bodies' cached
`free` / `defined`, so a call costs the statement's own fields, not its subtree:

- `defined(stmt) = stmt.defines() ∪ ⋃ defined(b)` over its nested bodies `b`
- `free(stmt) = (own reads ∪ ⋃ (free(b) − binds_axes(stmt))) − defined(stmt)`; a `deps_deep` statement reads `deps`
  only and its subtree's definitions stay out
- `free(body) = ⋃ free(s) − defined(body)`

The prototype keeps every answer identical to today's: the order-blind subtraction stays (a read ahead of its
definition in the same body is not free), so no identity, golden or corpus case moves.

## Steps

1. Crate: `defined` / `free` / `buffers` caches on `BodyData`, the statement reads over them, `Body::pruned`.
   Verify: the crate's tests, and a Python check that the new reads equal the old ones over every corpus kernel and
   every golden target body.
2. Extension: getters `defined`, `free`, `buffers`, method `pruned`.
3. Callers: the lambda (`__post_init__`, `closing`, `defined`, `cone`), the cut's `_pruned`, `_derive_io_names`, the
   tile normalization's `_names`, `_accums_sum_names`, and the other `exposed_defines` unions.
   Verify: `tests/compiler/ir`, the cut and lambda tests, golden freshness and corpus derived-half nodes.
4. Measure: the same placement node before and after, unprofiled and profiled.

## Later, if the prototype pays

- Drop the members nothing outside the package uses (`rename_free`, `fields_of`, `structural_form`,
  `has_side_effects`, `bound_axes`, `deps_deep`), and fold the lambda's two normalization steps into one cached call.
- Decide whether `free` should respect read order inside a body. That is a behaviour change: it moves lambda params
  and identities, so it needs its own restamp.
- Return the same Python set on a repeated read, if conversion still shows in the profile.
