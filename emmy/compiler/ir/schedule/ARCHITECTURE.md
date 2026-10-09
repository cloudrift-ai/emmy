# Schedule model

`Schedule` is the generic immutable kernel × node × edge schedule. Node sites are non-negative integers and edge
sites are `(consumer node id, operand position)` tuples; a schedule contains no problem, target, path spelling, or
lowering facts. A concrete schedule family may carry derived lowering facts in a separate materialization type.

A schedule enumeration has two terms, and the interface names both. A `ScheduleProblem` is the problem and the
target factored into `Site`s — one per node in composition order, the kernel site last — beside the knob row it was
built with. A site offers the values it may take on its own, with no other site in view, as its own factors; the
interface names no option product, because a product object is the one thing every eager table is built from. That
is the SOURCE of every candidate. A site the row names offers the row's value alone, parsed and checked with the
same per-choice rules a catalog value passes through; a site the row leaves free offers its catalog. Nothing
downstream generates a candidate, so nothing has to filter one away.

`ScheduleContext` is the immutable compatibility prefix `c`: what earlier sites decided. It owns the compatibility
between sites and nothing else. Its defining operations are a lazy frontier and composition:

    for pick in context.extensions():
        next = context.extend(pick)

Every context prefix and extension is a `Schedule[KernelT, NodeT, EdgeT]`; a non-`None` kernel marks completion.
`extensions` yields the next site's options that compose with the prefix; `extend` composes one and returns a context
containing the composed facts, leaving the original unchanged, or raises `ScheduleRefused`. `random_step(rng)` is
the third operation, one option `extensions` would yield, already composed, or `None`: the step of a random descent.
It costs what it touches: the default materializes the frontier and extends one pick, which is right only where the
frontier is small by construction (the cut pass's structural choices, the register tier's one kernel choice), and a
family whose frontier is a product of factors draws factor by factor, deriving only what the draw reaches and never a
site's product.
`extend` is also the validation boundary for a complete classic schedule supplied directly by a pinned golden, even
when that assignment was not emitted by `extensions`. The generic `schedule(context)` recursively composes those lazy
frontiers and yields only complete schedules. Recursion is the generic Algorithm 1 traversal; consumers do not write a
family-specific visitor or feed contexts back themselves. The driver knows no concrete family, pipeline fork type,
site order, or enumeration slice. `narrowed(row)` is the one way a row enters after construction: an EMPTY prefix over
the same problem with the row installed, which is what a descent that already holds a row asks for before expanding
anything. Strict narrowing marks only the supplied codec keys as exact; inherited peer-kernel pins keep their ordinary
tolerant reading. A row narrows WITHIN the live hand pins and never lifts one: where an evidence row and an
environment pin name the same site with different values the site keeps the pin's, no leaf equals the row, and the
descent that followed the row re-decides. A recorded receipt of the same kernel used to win that disagreement, so an
`EMMY_KNOBS` fast-math pin silently deployed the standard receipt. The pipeline's generic schedule-fork adapter
preserves the same contexts as deferred search branches without adding compatibility logic.

Classic sites additionally expose their independent factors — a node site's `nodes` and `edges`, the kernel site's
`kernels` — and `ClassicProblem.bound` reports the size of their product without building it. There is no product
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
that reads K only through one block guard, such as a grouped activation scale's maximum) and a per-cell body. Only
external reads cross these parts; a value defined inside the consuming body is not bridged. Equal folds shared by
cell edges lower once through `Body.coalesce`, so attention's output and row sum do not redeclare the same states in
a replicated fill.
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
the choice types and key spellings through the package.

`ClassicProblem` (`classic/sites`) is `p + t` and the row: the unscheduled TileOp, its target, and the knob row
whose values its sites offer where it names them. `ClassicScheduleContext` is the immutable `c + p + t` prefix over
that problem. Everything a schedule choice cannot change is derived from the tile and the target and memoized on the
term it derives from — a contraction's `ContractionFacts` on the Fold root (`TileOp.contractions`), the packed operand
readings, the placement and the two site relations the binder dictates (`shared_roots`, `chain_pairs`) on the
TileOp, and a node choice's supports on the site that offers it. The compatibility rules — worker inventory,
physical-axis agreement, fragment seams, the shared-root and chain rules — are stated in `refusals` beside the
per-choice ones, as one function of a pick and a RELATION: what a prefix has decided that those rules read (the
inventory it claimed, its axis and fragment agreements, and the decided nodes only where a rule reads them — every
one while no inventory is claimed, all of them at a shared root or a chain member). The context carries that
relation and composes it; the kernel-level rules (raster eligibility, resource limits, the producer band) stay with
it. A node site holds one record per node choice with the facts that are the tile's alone — the inventory it
claims, its placed geometry and axis agreements, the seam claims that read no transport — and, derived only when
asked, the choice's supports: the choice paired with each transport its own catalog offers (`stage_candidates`)
that resolves (the stage resolver, the plan and budget refusals). The site's edge catalog is the union over its
choices, so a row can name any of them, but a choice never takes a transport another choice brought: the 8-deep
ring is wgmma's alone, and an mma tile that drew it made a leaf the row-narrowed descent could not rebuild. A
transport that takes an operand's base address never feeds a paged one (`ClassicProblem.paged`, the graph's
`cuda.paged_buffers` hint): TMA encodes the address on the host, and cp.async's per-thread addresses cannot
resolve a page per element, so a buffer of several pages loses both. A prefix filters the site's choices by those
tile-level facts, one
filter per relation kept on the site, so prefixes that decided different nodes but agree on the facts read one
answer; on the kernels measured that filter alone finds every dead prefix. The supports of the choices it admits
are then filtered by the one claim a support completes, its transport's K slab at an ordinary seam. `extensions`
reads that whole frontier; `random_step` never does — it draws an admitted choice, keeps it as often as it has
admitted supports and takes one of those, so the draw is uniform over the frontier's (choice, transport) pairs while a
descent derives supports only for the choices it touched; a choice with none leaves the draw. It composes the drawn
support without `extend`'s re-check, which would repeat the admission the draw just made. A hand-pinned transport no choice resolves raises with the rule's message the first time
a prefix reads the site. Kernel picks form the final frontier: the kernel site's catalog is what the node sites'
choices imply, so it is the last site. The fragment-seam relation has no pipeline-side copy.

A pointwise map's site reads a catalog of its own (`map_tile_moves`): the per-cell form and the register strips that
hand one thread 2, 3, 4 or 8 contiguous inner-axis elements, each offered when it divides a static inner extent. It
is not the scalar-contraction ladder, which stops at 4 because a contraction's strip also carries accumulators; a map
holds none, and sharing the ladder once dropped the 8-wide strip a recorded QK-norm row deploys.

A cooperative reduce's inventory is `t<coop>`: one CTA of `coop` threads per output cell. Where `coop` is at most a
warp, the combine is a lane butterfly that stays inside the cell's lanes, so the kernel site also offers the packed
inventory `t<coop>x<cells>` (`packed_works`), a 128-thread CTA holding several cells; the flat thread decode already
hands consecutive cells to consecutive lane groups. `REDUCE=coop` reads its width off the inventory's first unit.
Packing needs every operand read straight from gmem: a staged row is one CTA-wide slab per cell. A node prefix spells
`t<coop>` and the packed leaf grows it at an `x` boundary, which `Fork.admits` accepts for `WORK`.

The native block-scaled FP4 stage copies codes and scales in complete K-contiguous rows. A runtime activation
row extent is legal for cp.async: the shared-memory fill clamps both copies to the last valid row, and output
stores mask padded rows. The row count does not change a copy's K-inner stride or alignment. TMA requires a static
row extent because its box transport does not use that clamp. Runtime row counts must be positive. With a static
unit N, the shared-memory fill clamps every padded row to the sole input row, and the output stores mask the padded
columns. This masked N is legal for cp.async: padding never changes a K-contiguous copy's address or alignment.
Other masked N extents and masked TMA boxes remain unsupported. K and the code/scale spans must still satisfy the
stage's divisibility and 16-byte copy alignment rules; the ring may finish after any whole K tile, including a partial
cycle of its depth.

Classic domain projection, move catalogs, packed-operand readings, staging resolution, materialization, and
compatibility all live in `ir/schedule`. The sites are the only source of choices; pipeline search neither defines
nor filters them. `ir/schedule` may import other IR modules but never the pipeline layer. The pipeline retains only
knob/pin reads (folded into the row), pool identity, sampling, and the generic lazy-Fork adapter.

Fragment epilogue legality checks lowered work outside roots the binder can compute together, including boundary stores.
A grid's free axes are already bound during that check, including a unit row that only an output store reads.
A sibling reduction or a contraction whose output cannot be partitioned remains work the epilogue must execute. Its
loop excludes tensor-core atoms before ranking, avoiding repeated materialization refusals.

A reduction domain is projected from node and kernel facts alone, so the shapes the kernel factorizer cannot bind are
decided once, at the offer, and never dropped from a priced row later. The partition catalog is offered on the reduce
nodes the binder builds the kernel around — the roots it peels from the root projection (`ops.kernel_roots`: a tiled
contraction's root, every one of them for a multi-output kernel, else the first operand) and the folds those roots'
cones close over (`ops.chain_members`, which the binder's chain arm strides around one shared lane axis). A reduce
NESTED under one of those lowers serially inside its reader, so it carries the serial fold only, as does a node whose
reduce reads a boundary store's output sweep. An observed root without a provider chain also offers cooperative
widths up to one warp, with no register partials or transposition: the materializer retains every inclusive prefix.
Whether a fill takes a root's cone over is the
SCHEDULE's answer and not the term's, so a contraction a tier could fold whole still offers its own row statistic the
member catalog: the untiled tiers bind it as a fold beside the root, and reading the tier off the term left a
cooperative reduce evaluating a 16384-wide statistic once per thread. The contraction per-cell tier reads that same
projection, so a contraction inherits those readings rather than restating them.

Two binder facts are relations between sites rather than node domains, so they compose in `extend` beside the worker
and physical-axis agreements. The binder builds a kernel around several scheduled roots only where the projection
partitions its outputs by root (`ops.projection_regions` — each store reads exactly one root's region); where it does
not, one root is the kernel's root and every other reduce lowers serially, so the context refuses a second scheduled
root among those roots — an output tile, or a cooperative or ILP reduce, since `TILE` and `REDUCE` both select the
root the binder builds around. The row that tiled both — a gate/up projection whose one output reads both channels —
used to be offered, ranked first, and refused at materialize; the row that cooperated on both — DeepSeek V4's
`k_div_35_reduce` — was accepted while the binder honoured neither, so its measurement belonged to the serial kernel.
And a chain binds only in the binder's
untiled arm, so the context refuses a partitioned chain member beside an output-tiled root: there the fill evaluates
the cone per cell, statistic included, and the member's partition would realize as nothing — an unreproducible pin
rather than a refusal.

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
f32; the chunk partial promotes into the f32 carrier once per chunk, which is the promote cadence. The streamed
value selects the multiplicand dtype; the score's wider accumulator does not select a different atom family. How many
register columns of those partials are live at once is one rule the offer and the emitter both ask
(`chunk_partial_columns`): the whole row while the thread's register envelope holds it, else one column pair. At head
width 256 the row is 32 fragments, and with every partial live beside the carrier and the hoisted query the thread
needs 260 registers against 255; the register budget counts the pair there, so the reduced-accumulate expectation is
offered.

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
check so that message wins. A `wgmma` also holds its whole accumulator in registers at once, so it cannot spill:
`_wgmma_register_refusal` refuses a row whose accumulators, plus the registers a lane keeps beside them
(descriptors, addresses, ring counters), exceed the per-thread register envelope its CTA size leaves — ptxas would
refuse that kernel.

`stage_moves` offers the `STAGE` product of transport, ring depth and register depth. A node's stage filter keeps the
8-deep ring to `wgmma` tiles, the only ones it has paid on; elsewhere it would only multiply the candidates every
compile prices.

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
re-decides. Strict complete-row decode does not take that tolerant catalog fallback: once parsing or an intrinsic
check fails, the empty site is returned without walking the catalog.

A kernel-scoped hand pin remains partial: every named value must realize, while unnamed families retain their
catalogs. Exact replay applies to the supplied keys of a complete row followed from evidence, not to an ordinary
hand pin merely because it names one kernel. Omitting `WORK` from a reduction pin therefore leaves worker selection
open; an impossible named worker or reduction still refuses.

A bare pin on a kernel that spells its family per site is a DISJUNCTION over those sites: one carries the value and
every other is OFF. That is the reading `unreproducible_pin_flag` and `evidence_row_vouches` already give a bare key,
so a row measured under a bare pin reads back the same way it was pinned. Each such site therefore offers the pin's
value beside OFF, and the completed schedule is asked which site carried it (`unrealized_bare_pin`) — the one place
in the enumeration where a pin is decided across sites rather than at one. Reading it as a conjunction instead makes
it unsatisfiable on exactly the kernels that need it most: attention spells `TILE` at its score contraction and at
its chunked value channel, and no schedule carries one mma tile at both.

The precision policy (`allow_f16_accumulate`, `allow_fp8`) filters the CATALOG: an f16-accumulate or FP8 atom is
offered unpinned only where the compile allowed it. A hand pin naming such a tile is an authored, legal choice and
bypasses the policy; a row a descent follows (`with_row` — measured evidence) does not, so an FP16-accumulate row
recorded or measured in the standard lane cannot deploy there. Likewise a transposed raster (`gn4`, `gn8`) is never the catalog's own offer and is taken only
where a row names it.

`ClassicScheduleCodec` is the concrete strict wire boundary. Its public encode and decode operations validate through
one `ClassicScheduleContext`; private syntax-only parsing and encoding let graph reconstruction attach materialization
before the `TileOp` constructor performs that same validation once. It owns canonical complete-row and prefix-delta
encoding for `WORK`, `TILE`, `REDUCE`, `STAGE`, and `RASTER`. There is no codec base class: a second schedule family
should demonstrate any shared codec contract before one is extracted.

A complete classic hand pin is decoded through that same codec before the compatibility frontier is traversed.
Pins addressing a peer kernel's sites do not prevent direct decoding. A refused row keeps the ordinary peer fallback.

Transposed cooperative reductions use 32 output lanes by default. Volta's catalog also offers `coop-t/n8`: eight
output lanes, with the remaining threads partitioning the reduction. `/v<n>` still names adjacent columns per lane.
On an ordinary `coop` band, `/v<n>` names the adjacent reduce elements a lane reads per step, so a contiguous operand
reads as one vector; the lane strides by `coop · n`. A prefix scan does not take it: the scan keeps one inclusive
state per lane. The catalog does not offer `coop/v<n>` itself, since the prior cannot tell it from the plain band; a
row or a pin names it.
Both layouts use the same reduction choice, codec and materializer. The worker count is divisible by the output lane
count.

The structural cut phase runs before any schedule is composed. The single `030_cut` pass reaches a fixpoint over two
ordered domains: stored-Fold-edge placement first, then cross-CTA reduction splitting. Every successful choice and
fresh piece re-enters the same rule. `030_cut` presents its restricted structural frontier through a schedule context;
`040_schedule` supplies a `ClassicScheduleContext`. Both passes use the same generic `schedule` traversal.

## Register storage across ordered steps

The `reg` transport names stored register intermediates. Reuse can span consumers or loop iterations; recurrence
is not part of the transport's meaning. `d1/reg` provides one slot without prefetch. The current implementation
supports this transport through the ordered matrix-loop schedule below; other schedules do not yet offer it.

`RegisterContext` uses the same problem, site, codec, and lazy enumeration interfaces as the classic schedule.
It offers one kernel choice for a static ordered loop with one matrix state, pointwise operations, and additive
matrix contractions, read off the fold that carries the state (`Fold.carries`, its `cells` ending in the column
and the warp-owned row). The structural reading proves that each warp's rows are independent through every
contraction, that every carrier read takes the previous step at the warp's own rows, and that output matrices
share the same batch coordinates. Other recurrences retain the classic schedule, which realizes the carrying loop
as one launch per step over a state buffer. A zero-axis root derived from a step keeps the time coordinate as a
trailing lambda parameter whenever its body still reads it. Unit batch coordinates are omitted only from ownership
matching, after checking their external extents; the stored output indices remain unchanged. Both schedule families
restore omitted unit coordinates against the seed tensor's bound shape before computing its address.

`WORK=w<M>x1` assigns independent groups of sixteen value rows to warps. `TILE` names an FP16 atom with both
C→A and C→B repacking support and `f1x<N>`, where `N` covers all state columns. The atom registry supplies
the fragment geometry: eight columns for m16n8k16, sixteen for Volta m8n8k4. `STAGE=d1/reg` gives the carried state
one register slot across steps; there is no shared-memory ring or operand prefetch. Equal loads, pointwise
operations, and products share FP32 register results within a step. Operand conversions stay beside each MMA
to shorten their live ranges. All reads finish before the carried slot is updated.

The catalog offers FP32 accumulation and FP16 partial accumulation under `F16_MMA_F32_ACC` (also enabled by
`FAST_MATH`). Both convert matrix operands to FP16 and keep the carried state in FP32. For FP16 accumulation,
`TILE`'s K chunk is the promotion interval in atom steps: `k4` promotes and clears the partial accumulator every
64 products on m16n8k16 or 16 on Volta m8n8k4, including a shorter final chunk. An explicit tile row can select
the arithmetic directly. Register pressure and numerical error depend on the shape and inputs.
