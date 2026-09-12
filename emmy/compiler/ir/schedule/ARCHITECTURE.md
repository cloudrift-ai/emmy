# Schedule model

`Schedule` is the generic immutable kernel × node × edge assignment. Node sites are non-negative integers and edge
sites are `(consumer node id, operand position)` tuples; the assignment contains no problem, target, path spelling, or
lowering facts. A concrete schedule family may carry derived lowering facts in a separate materialization type.

A schedule enumeration has two terms, and the interface names both. A `ScheduleProblem` is the problem and the
target factored into `Site`s — one per node in composition order, the kernel site last — beside the knob row it was
built with. A site answers `options`: the values it may take on its own, with no other site in view. That is the
SOURCE of every candidate. A site the row names offers the row's value alone, parsed and checked with the same
per-choice rules a catalog value passes through; a site the row leaves free offers its catalog. Nothing downstream
generates a candidate, so nothing has to filter one away.

`ScheduleContext` is the immutable compatibility prefix `c`: what earlier sites decided. It owns the compatibility
between sites and nothing else. Its defining operations are a lazy frontier and composition:

    for pick in context.extensions():
        next = context.extend(pick)

Every context assignment and extension is a `Schedule[KernelT, NodeT, EdgeT]`; a non-`None` kernel marks completion.
`extensions` yields the next site's options that compose with the prefix; `extend` composes one and returns a context
containing the composed facts, leaving the original unchanged, or raises `ScheduleRefused`. `extend` is also the
validation boundary for a complete classic assignment supplied directly by a pinned golden, even when that assignment
was not emitted by `extensions`. The generic `schedule(context)` recursively composes those lazy frontiers and yields
only complete assignments. Recursion is the generic Algorithm 1 traversal; consumers do not write a family-specific
visitor or feed contexts back themselves. The driver knows no concrete family, pipeline fork type, site order, or
enumeration slice. `narrowed(row)` is the one way a row enters after construction: an EMPTY prefix over the same
problem with the row installed, which is what a descent that already holds a row asks for before expanding anything.
The pipeline's generic schedule-fork adapter preserves the same contexts as deferred search branches without adding
compatibility logic.

Classic sites additionally expose their independent factors — a node site's `nodes` and `edges`, the kernel site's
`kernels` — and `ClassicProblem.bounds` reports the size of their product without building it. There is no product
OBJECT: the sites are the factors, so a type holding a copy of them would be a second answer to one question. Tests
build the literal product themselves, and a bounded test hands the sites hand-written factors by subclassing them
(`tests/compiler/helpers.literal_classic_context`), which is the only way to offer a site values it did not project:

    D(p, t, row) = K × ∏ N(node) × ∏ E(edge)      # what the sites offer, under the row
    Algorithm 1(p, t, row) = {a ∈ D(p, t, row) | extend(c + p + t, a) succeeds}

`ClassicScheduleContext` evaluates the compatibility term; the problem's sites ARE the domain term. Its frontier may
omit a pick when `c + p + t` proves that no completion exists, but repeated generic expansion must enumerate exactly
the accepted set in every node traversal order. A row never changes the compatibility relation and is not inspected
by the generic traversal: it changes what a site offers, and only there.

Reusable leaf choices such as `Work`, `Tile`, `Reduce`, `Stage`, and `Raster` contain neither sites nor target facts.
The `TileOp` **is** the site index — there is no second object over the same term. Whether a bilinear site takes
`TILE` and `STAGE` at all (`contracts`) is the tile's question, since it needs the kernel's extents: a pair whose
role-less side shares a coordinate with the other side qualifies only while that coordinate partitions the reduction
(it composes with the reduction index) or the role-bearing side's reads are value-dead in it (a merged weight's
reshape residue); a B that changes with the row it is contracted against is no slab per tile. It derives stable
node ids,
operand-edge sites, each site's projection or reduction view, and each contraction's schedule-independent
`ContractionFacts` — its effective K axis, computed-A cone seam, nested producer, and fragment need. The seam
(`cone_seam`) splits the cone's edges at the K axis into a row-invariant prologue, a per-chunk statistic (a reduce
that reads K only through one block guard, such as a grouped activation scale's maximum) and a per-cell body, and
keeps one lowering of a fold two cell edges read (attention's output and its own row sum): the tree forms that fold
twice as equal nodes, and a fill that replicates the cell per output cell would otherwise declare its states twice.
`ir/schedule/views` supplies the vocabulary (`node_view`, `Projection`, `Reduction`, `Contraction`,
`ContractionFacts`) and the one derivation that is not a projection of the site table, `contraction_facts`; the tile
layer reads through them. The composition context publishes the schedule-facing API (`node`, `site`, `operand`,
`producer`, `incident_edges`, the key spellings) and does not re-export the kernel's structural members under second
names. A contraction view belongs to a node site and expresses its operand roles as edge positions, and is not
another Fold node. A concrete codec alone translates integer and tuple sites to wire spellings, and the route it spells is the site
record's own.

The node list is the one walk, `ir/tile/path.sites`, deduplicated by object identity. That walk yields a `Site` per
node — the term, the axes in scope, the segment path — so the schedule's integer ids, the tree-path codec's segments,
and the cut pass's scopes are all readings of one traversal and cannot drift. Operands are visited in stored order,
which formation orients (a contraction's A first), and a route spells the stored position taken at each departure;
an ambiguous node family spells its site by that route (`TILE@map.1/twist.1/inner`), the one grammar `PLACE` uses.

**Every derivation memoizes on the ROOT, not on the wrapper**: several `TileOp`s exist over one term across a
lowering, and a cache on the wrapper silently re-derives per wrapper. `schedule_nodes`, `schedule_views` and
`contraction_facts` all key their memo on the Fold root, and the `TileOp` properties are accessors over it.

## Classic schedule

The classic family is the `classic/` package, one role per module: `schedule` (the choice types, the sites' wire
spellings and keys), `refusals` (every per-choice legality rule), `sites` (the source),
`context` (the join), `codec` (the wire boundary) and `materialize` (the lowering boundary). Imports flow in that
order and nothing in the package imports the tile package at module level, which is what lets `ir/tile/ops` read
the assignment names through the package.

`ClassicProblem` (`classic/sites`) is `p + t` and the row: the unscheduled TileOp, its target, and the knob row
whose values its sites offer where it names them. `ClassicScheduleContext` is the immutable `c + p + t` prefix over
that problem. Everything a schedule choice cannot change is derived from the tile and the target and memoized on the
term it derives from — a contraction's `ContractionFacts` on the Fold root (`TileOp.contractions`), the packed operand
readings and the placement on the TileOp, and the per-target support tables on the TileOp beside their target. The
context owns all classic compatibility: worker inventory, physical-axis agreement, fragment seams, raster eligibility,
resource limits, producer-band/TMA agreement, target availability. A site's tuples hold choices only; an expensive
local support record is derived lazily, once per site object, after the context has selected one node and its
incident edge values. This node-plus-incident-edges frontier is granular enough to reject mixed transport and
fragment-seam combinations before they create subtrees, without materializing the full node × edge product.
`extensions` emits partial schedules at that granularity; `extend` derives and composes their support. Kernel picks
form the final frontier: the kernel site's catalog is what the node sites' choices imply, so it is the last site. The
fragment-seam relation has no pipeline-side copy.

Classic domain projection, move catalogs, packed-operand readings, staging resolution, materialization, and
compatibility all live in `ir/schedule`. The sites are the only source of choices; pipeline search neither defines
nor filters them. `ir/schedule` may import other IR modules but never the pipeline layer. The pipeline retains only
knob/pin reads (folded into the row), pool identity, sampling, and the generic lazy-Fork adapter.

A reduction domain is projected from node and kernel facts alone, so the shapes the kernel factorizer cannot bind are
decided once, at the offer, and never dropped from a priced row later. The partition catalog is offered only on the
reduce nodes the binder builds the kernel around — the roots it peels from the root projection (`ops.kernel_roots`: a
tiled contraction's root, every one of them for a multi-output kernel, else the first operand); a reduce nested under
a root or beside it lowers serially inside its reader, so it carries the serial fold only, as does an observed node
and one whose reduce reads a boundary store's output sweep. The contraction per-cell tier reads that same
projection, so a contraction inherits those readings rather than restating them.

One binder fact is a relation between root sites rather than a node domain, so it composes in `extend` beside the
worker and physical-axis agreements: the binder builds a kernel around several output-tiled roots only where the
projection partitions its outputs by root (`ops.projection_regions` — each store reads exactly one root's region);
where it does not, one tiled root is the kernel's root and every other reduce lowers serially, so the context refuses
a second output-tiled root among those roots. The row that tiled both — a gate/up projection whose one output reads
both channels — used to be offered, ranked first, and refused at materialize.

A CHUNKED carrier's seam is stricter than an ordinary consumer's. The ordinary need tolerates an untiled producer;
this one is built on the fragment, since the chunk's score IS the producer's tile — so the producer must be
warp-tiled at the same atom, one warp column wide, with the chunk as its N tile and the same register rows
(`_fragment_agreements`). That equation is FlashAttention's own shape, and stating it here is what keeps the tier
from being offered a row its emission would have to ignore. The chunk is the consumer `TILE` atom's K width, not a
staging slab, and must contain a complete logical C fragment so its score can repack into the paired contraction's A
operand. Coordinate masks remain eligible only when their complementary branches form an additive cell mask.

The tier also demands the score's own contraction extent be STATIC: the chunk covers it in one pass and holds a query
fragment per atom-K step, so a symbolic extent there has no step count to hold them at. And the ATOM it names is the
EXPECTATION's — the score keeps that atom's f32 sibling (`wide_accumulate`), so the reduced-accumulate cell can run
the expectation's mma chain at the consumer-die full rate without moving the softmax's running max and denominator off
f32; the chunk partial promotes into the f32 carrier once per chunk, which is the promote cadence.

The tier's other refusals (`_chunk_refusal`) are the same kind of statement, and two of them are about the score's own
PREFIX — the carrier's lift cut to its score role, which is where an SDPA mask lands. Only what reaches a fragment
loader needs a gmem address: whatever supplies the pivot, and the streamed value. The prefix's remaining leaves are
read once ahead of the chunk loop, so a mask's fill / zero constants may be a computed pair with no slab at all, and
its statements may include a `Select` on the score fragment's OWN coordinates, which the emitter evaluates per element.
Both were blanket refusals, and either one sent a masked attention target to the scalar tier whole.

A `wgmma` atom is a 16×8 warp sub-cell like `mma_m16n8k16`, but its instruction is issued by four M-adjacent warps
over 64 rows and N columns, so `_wgmma_refusal` narrows the row it can hold: a `w<4k>x1` warp grid (the unit decode
is N-fastest, so contiguous warp ids stack along M), one fragment row per warp (`f1x<C>`, a second row would sit
64 rows down) with `C` a multiple of N/8 (whole instructions along N), a `k4` chunk (one 128-byte swizzle row per
descriptor) and a shared-memory stage on every operand (the instruction reads descriptors, never fragments). The
unpinned catalog drops such rows; a pin raises with the rule's message, and the tile check runs before the stage
check so that message wins.

`TileOp.stage_edges` offers a transport at every operand of every contracting site, a chunked carrier's included —
which tier then puts which operand on a slab is the tier's own business. The chunked site used to be excluded on the
reading that it "takes its chunk off the `TILE`, so a transport spelling there would decide nothing"; a `Stage` never
spelled that chunk (the resolver derives `bk_elems` from `Tile.bk`), so what the exclusion decided was that
attention's value channel reads gmem-direct. That one transport now covers BOTH operands the carrier streams —
`chunk_key_stage` says whether the score's key joins the value on the ring, and the resolver sizes the ring at both.

A row — a hand pin from the environment, a golden row a descent follows — reaches a site as the value it names,
never as a filter over a catalog. A site the row names by its exact codec key parses the value and checks it with
the same per-choice rules its catalog passes through (the atom, chunk and plan refusals, the catalog's own grid and
budgets, membership in the reduction and transport catalogs), so a row can select a value the catalog would have
offered and never manufacture one. A named value the site cannot take empties the site and the kernel enumerates no
row — the loud direction; under `validate_pins=False`, the reading a row published across the peer kernels of a
multi-kernel target takes, the site keeps its catalog instead. A warp-group tile its grid cannot feed, a transport
the card cannot run, and a hand-pinned transport no support resolves raise with the rule's own message
(`loud_pins`), which the descent's `with_row` turns off: a stale row is answered by an empty site and the caller
re-decides.

A bare pin on a kernel that spells its family per site is a DISJUNCTION over those sites: one carries the value and
every other is OFF. That is the reading `unreproducible_pin_flag` and `evidence_row_vouches` already give a bare key,
so a row measured under a bare pin reads back the same way it was pinned. Each such site therefore offers the pin's
value beside OFF, and the completed schedule is asked which site carried it (`unrealized_bare_pin`) — the one place
in the enumeration where a pin is decided across sites rather than at one. Reading it as a conjunction instead makes
it unsatisfiable on exactly the kernels that need it most: attention spells `TILE` at its score contraction and at
its chunked value channel, and no schedule carries one mma tile at both.

The precision policy (`allow_f16_accumulate`, `allow_fp8`) filters the CATALOG: an f16-accumulate or FP8 atom is
offered unpinned only where the compile allowed it. A row naming such a tile is an authored, legal choice and
bypasses the policy. Likewise a transposed raster (`gn4`, `gn8`) is never the catalog's own offer and is taken only
where a row names it.

`ClassicScheduleCodec` is the concrete strict wire boundary. Its public encode and decode operations validate through
one `ClassicScheduleContext`; private syntax-only parsing and encoding let graph reconstruction attach materialization
before the `TileOp` constructor performs that same validation once. It owns canonical complete-row and prefix-delta
encoding for `WORK`, `TILE`, `REDUCE`, `STAGE`, and `RASTER`. There is no codec base class: a second schedule family
should demonstrate any shared codec contract before one is extracted.

The structural cut phase runs before assignment composition. The single `030_cut` pass reaches a fixpoint over two
ordered domains: stored-Fold-edge placement first, then cross-CTA reduction splitting. Every successful choice and
fresh piece re-enters the same rule. `030_cut` presents its restricted structural frontier through a schedule context;
`040_schedule` supplies a `ClassicScheduleContext`. Both passes use the same generic `schedule` traversal.
