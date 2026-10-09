> Historical engineering note. It records a review or probe, not the current product overview. Start at the [repository README](../../README.md) and [docs index](../README.md).

# M1 Interfaces

Interface source of truth for the M1 engine (`libs/usdGen/usdGen`) and the M1
imaging lane (`libs/usdGenImaging/usdGenImaging`). Every signature below is
copied from the installed headers; when a section says "see header" the header
is the source of truth and this document is the index.

Plan sources: `plan/02-schema.md` (schema, dirty classes), `plan/03-execution-engine.md`
(engine), `plan/06-imaging.md` (imaging lane). Frozen contracts:
`docs/freezes/C1.md` (C1 schema registry), `docs/freezes/C2.md` (tile
publication contract), `docs/freezes/C5.md`.

Namespaces:

- `usdGen` (global) — engine core: types, graph, ops, buffers, session.
- `usdGenImaging` (global) — imaging lane: session, bridge, dirty router,
  data sources, tokens.
- `pxr` — plugin-registry-visible classes (prim adapters, schema adapters,
  scene index plugins, dirty router) so `TfType`/registry machinery can
  reference them. The router forwards to the global `usdGen::` engine types.

---

## 1. Build surface

- `libs/usdGen` — static/shared engine target `usdGen`; public headers in
  `libs/usdGen/usdGen/`. Built hidden-visibility + `--exclude-libs,ALL`; only
  symbols marked `USDGEN_CORE_API` (`usdGen/export.h`) are exported.
- `libs/usdGenImaging` — imaging lane target; public headers in
  `libs/usdGenImaging/usdGenImaging/`.
- `libs/usdGenSchema` — generated schema plugin (`plugin/usdGenSchema/resources/
  generatedSchema.usda`): five prim types + `UsdGenRestAPI` API schema.
- `libs/usdGenShaders` — shader plugin (`plugin/usdGenShaders/resources/shaders`).

Engine canonical version API (sol S-8): `usdGen::GetVersionString()` in
`usdGen/usdGen.h`. The M1 imaging identity/version surface lives in
`UsdGenImagingRegistry` (see §5).

---

## 2. C1/C2/C5 frozen contracts

- **C1** (`docs/freezes/C1.md`): the operator registry — C1 type names
  (`UsdGenScatter`, `UsdGenGrow`, `UsdGenNoise`, `UsdGenLength`, `UsdGenWidth`,
  ...), their `usdGen:space` defaults, `TopoFx`, the topology/value parameter
  partition (02-schema.md §6 rows), and ramp encodings (S11). The engine's
  `UsdGenOp::TopologyParameters()/ValueParameters()` implement this partition.
- **C2** (`docs/freezes/C2.md`): the tile publication contract — one
  `UsdGenTilePublication` per `<description>/__usdGenRender/tile_NNNN` prim
  (4-digit zero pad); exact-size primvar arrays (`points.size() ==
  sum(curveVertexCounts)`, SI-1); the contract constants `type="cubic"`,
  `wrap="pinned"`, `refineLevel=2` (no M1 tumble tier); chunk/tile arithmetic
  (worked example: 100 000 curves, chunkSize 512, tileTarget 64 ⇒ nChunks
  196, chunksPerTile 4, nTiles 49).
- **C5** (`docs/freezes/C5.md`): imaging-side M1 additions (RestAPI sourcing,
  dirty routing wiring) per the frozen review.

---

## 3. Plugin registration surface (usdGenImaging plugin)

One shared adapter body behind the five registered prim adapters, plus the
API-schema adapter. All classes sit in the `pxr` namespace.

### 3.1 Prim adapters — `primAdapter.h`

`UsdGenPrimAdapterBase : UsdImagingSceneIndexPrimAdapter` publishes **every**
declared `usdGen:*` property of the prim as a data source under the `usdGen`
container (S14 pull-all) and maps each property edit to its precise C1 locator
so `UsdGenDirtyRouter` gets exact dirties (06 §2.7; 02 §6 is the emitted
property→locator table).

- `GetImagingSubprims(prim)` → `{TfToken()}` — ONE Hydra prim per USD prim;
  operator and map prims are data-only (must not become rprims);
  Groom/Description/GuideSet carry Imageable state through
  `UsdImagingDataSourcePrim`.
- `GetImagingSubprimType(...)` → empty token for all five types.
- `GetImagingSubprimData(prim, subprim, stageGlobals)` → the `usdGen`
  container data source.
- `InvalidateImagingSubprim(prim, subprim, properties, type)` → the C1
  locators for the edited properties.
- `static Mappings(schemaTypeName)` → `UsdImagingDataSourceMapped::PropertyMappings`,
  built once per schema type from `UsdPrimDefinition` (R6 skip-list, never a
  name-prefix test), cached under an internal lock (released before use).
- `static LocatorForProperty(property)` → `usdGen:clump:size` ⇒ `clump/size`
  (relative; `Mappings` makes it absolute under the `usdGen` prefix).
- `static IsSingleTarget(property)` → true for the two relationships 02 §2
  declares "exactly one target"; every other usdGen relationship is
  array-mapped.

Five one-line `final` subclasses (one per concrete prim type, ADR §2.3):
`UsdGenGroomAdapter`, `UsdGenDescriptionAdapter`, `UsdGenOperatorAdapter`,
`UsdGenMapAdapter`, `UsdGenGuideSetAdapter`.

### 3.2 RestAPI schema adapter — `restApiSchemaAdapter.h/.cpp`

`UsdGenRestAPIAdapter` registers the `usdGen/rest` API schema (M0:
registration only). **M1 adds the wiring below (§6)**; the M0 file is
extended, not replaced (scaffold-issues.md). Known open defect: review M-8
(`docs/review/sol-m1-engine-notes.md`) — the common prim adapter body and the
Rest factory were inert at M0 (null/empty returns); M1 implements one shared
adapter body, surfaces every declared property, maps every property to
dirtiness, and implements Default-time RestAPI sourcing.

### 3.3 Scene index plugins

- `groomSceneIndexPlugin.h` — `UsdGenGroomSceneIndexPlugin`: the scene index
  that owns a `UsdGenImagingSession` per key and publishes tiles (see §5).
- `metadataSceneIndexPlugin.h` — metadata-only scene index (pass-through M0).

### 3.4 Shader discovery plugin

`usdGenShadersDiscoveryPlugin.h` — discovers the `usdGenShaders` shader
resources for Storm preview. See header.

---

## 4. Engine API (`usdGen::`)

Headers in `libs/usdGen/usdGen/`. The engine is pure-value in, publication
out: it never sees a `UsdStage`, a scene index or an `SdfLayer` (S8).

### 4.1 `types.h` — ids, enums, constants, tile arithmetic

- Ids: `UsdGenNodeId` (u32, dense topo order), `UsdGenChunkId` (u32),
  `UsdGenTileId` (u16), `UsdGenSurfaceId` (u16);
  `kUsdGenInvalidNode == ~0u`.
- `UsdGenEpoch = std::array<uint64_t, 2>` — Merkle structural digest and
  capture epoch (explicit `operator==/!=`; a bare `==` would recurse).
- `enum class UsdGenSpace { Inherit, Rest, Deformed }` — `Inherit` ⇔ authored
  `usdGen:space = "auto"` (R9), resolves to the op type's `UsdGenOp::Space()`.
- `enum class UsdGenTopoFx { None, CurveCount, CvCount, Both }` (R14).
- `enum class UsdGenRole { Curves, Reference }` (I3 reference lane).
- `enum class UsdGenReadPhase { Base, Preceding, Final, Explicit }`
  (`Preceding` is a v1 alias for `Final`, R9).
- Dirty bits (unscoped `uint32_t` constants — never written
  `UsdGenDirtyBits::X`): `UsdGenDirtyNone 0`, `UsdGenDirtyParameter (1<<0)`,
  `UsdGenDirtyCapture (1<<1)`, `UsdGenDirtyTopology (1<<2)`,
  `UsdGenDirtySurfacePoints (1<<3)`, `UsdGenDirtySurfaceXform (1<<4)`,
  `UsdGenDirtySurfaceTopo (1<<5)`, `UsdGenDirtyMap (1<<6)`,
  `UsdGenDirtyStructural (1<<7)`, `UsdGenDirtyLiveOverride (1<<8)`.
- `enum class UsdGenCommitReason { SetTime, LiveOverride, SceneFrameDirty,
  NoticeBatchEnd }` (ADR §4.3 as amended by R32; 06 §3.9).
- `enum class UsdGenContext { Interactive, Render }` (R13 exclusive).
- Chunk/tile: `kUsdGenMinChunkSize 128`, `kUsdGenDefaultChunkSize 512`,
  `kUsdGenMaxChunkSize 1024`, `kUsdGenTileTargetDefault 64`,
  `kUsdGenTileMin 32`, `kUsdGenTileMax 256`; `ClampChunkSize(int)`;
  `ComputeNumChunks(maxCurves, chunkSize=512)`,
  `ComputeChunksPerTile(nChunks, tileTarget=64)`, `ComputeNumTiles(...)` —
  R21 verbatim, frozen worked example in C2.
- `bool KeepCurve(uint64_t curveId, float keepFraction)` — density decimation
  predicate (R12/R13; `kSaltDensity != 0` so the surviving set is never
  `{hairId < keepFraction}`).

### 4.2 `graphDesc.h` — the pure-value input

- `UsdGenParamValue { TfToken name; VtValue value; bool animated; }` — one
  authored `usdGen:*` property, prefix stripped (C1 registry).
- `UsdGenRampDesc { VtVec2fArray knots; VtFloatArray positions;
  VtVec3fArray colors; TfToken interpolation; }` — S11 encoding, resolved by
  the adapter; interpolation default `catmullRom` (R11).
- `UsdGenNodeDesc` — `path`, `type`, `mode`, `algorithmVersion` (0 == track
  newest, R17), `enabled`, `seed`, `blend`, `space`, `readPhase`, `inputs`,
  `references`, `curves` (→ `curveSets` indices), `surfaces`, `maps`,
  `params` (EVERY mapped locator, S14 pull-all), `ramps`.
- `UsdGenCurveSetDesc` — one C3 `BasisCurves` input: `path`, `role`,
  `curveRole` (C3 marker), `curveVertexCounts`, `points`/`rest`, `widths`,
  `skinPrim`, `curveId` (u64[], R12), `skinPrimUv` (texCoord2f, not "st"),
  `rootFrame`, `frozenEpoch` ("usdgen1:sha1:...", S42), `guideBlend`,
  `curveGeneration`.
- `UsdGenSurfaceDesc` — `path`, `id`, `faceVertexCounts/Indices`,
  `restPoints` (S12, at `UsdTimeCode::Default()`), `points` (deformed, at
  `desc.time`), `samples` (sorted; `samples[0].time == desc.time`, R23),
  `velocities` (P1 only), `uv`, `subsetFaces` (R15 GeomSubset),
  `worldMatrix` (post-flattening, S4), `surfaceGeneration`.
- `UsdGenMapDesc { path, type, resolvedAssetPath (S13 stage-free),
  textureGeneration, params }`.
- `UsdGenLookDesc` — M1 bakes the CPU half of the displayColor contract
  (02 §2.12 defaults; 07 §1.2): root/tip colours, ramp colours/positions/
  interpolation/exponent, `bakeMode perCurve|perCV`, `bakeTarget
  displayColor|primvar|none`, hue/value jitter, seed.
- `UsdGenGraphDesc` — `description`, `terminal`, `nodes` (namespace order,
  Kahn tie-break), `curveSets`, `surfaces`, `maps`, `look`, `xformMatrix`,
  `purpose`, `visibility`, `materialPath`, `pickTarget`,
  `densityScale`/`renderDensityScale`, `tileTarget 64`, `curveBasis
  "bspline"`, `motionMode`, `motionSampleCount` (clamped [2,16]),
  `forwardSurfaceSamples`, `schemaVersion 1`, `time`.

### 4.3 `curveBuffer.h` — buffers, views, publication boundary (C2)

- `UsdGenPlane { name, interpolation, type (float|int), arity, VtFloatArray
  f, VtIntArray i }` — one named extra plane (R24 pins type/arity:
  `clumpId_<level>` uniform int arity 1; `guideIndex`/`guideWeight`
  uniform, elementSize 3).
- `UsdGenChunkDesc { firstCurve, curveCount, liveCount, firstCv, cvCount
  (0 == ragged), tile, surface, boundsRest }` — surface-major, then Morton
  order over rest roots (§1.7).
- `UsdGenCurveBuffer` — per-CV planar SoA (`px/py/pz`, `width` (empty ==
  inherit), `hairT`, `extraCv`), per-curve AoS (`curveId` u64[], `rootPrim`,
  `rootUV` (== "st"), `rootT/rootN/rootB`, `cvOffsets` (present iff any
  chunk ragged), `extraCurve`), `chunks`, `totalCurves`,
  `totalCvs`, `topologyVersion`, `valueVersion`. Handoff to imaging is by
  value (VtArray COW refcount bump, R20).
- `UsdGenChunkView` — what a kernel may touch: chunk-local bases, writable
  `px/py/pz`, conditional `width`/`hairT` (bound by `PlanesTouched()`),
  read-only `inPx/...`, `outF/outI/inF/inI` slot planes, `Cv(c,i)` /
  `CvRagged(c,i)` indexing. Never `const_cast`.
- `UsdGenTileView { tile, firstChunk, chunkCount, totalLiveCurves,
  totalLiveCvs, extent, pointsDirty, widthsDirty }` — read-only window for
  the interleaver (03 §6.3); dirty flags are engine-internal.
- `UsdGenReferenceSet { buffer, localX/Y/Z, guideBlend, generation }` —
  reference lane (I3), evaluated in full, never chunk-dirty.
- **C2 publication boundary** (`docs/freezes/C2.md`):
  - `UsdGenTilePublication` — per `<description>/__usdGenRender/tile_NNNN`:
    topology (`curveVertexCounts`, `basis`), constants `refineLevel 2`;
    exact-size primvars `points` (vertex, size == sum(curveVertexCounts)),
    `widths` (vertex or constant, never varying), `hairT` (vertex), `hairId`
    (uniform float [0,1)), `st` (uniform root UV), `displayColor` (uniform
    perCurve bake or vertex perCV), `bakeColor`; `extraUniform` op planes;
    `velocities` (P1 only — **blocked** `HdBlockDataSource` in P0/P2);
    per-tile `extentMin/Max`; hand-authored `xformMatrix`,
    `xform/resetXformStack` (constant, never payload), `purpose`,
    `visibility`, `materialPath` + `materialPurpose "allPurpose"`,
    `primOrigin` (pickTarget-dependent), `dependencySurface`
    (`__dependencies: dependedOnPrimPath`).
  - `UsdGenInstancerPublication` — M6-only; empty in M1.
  - `UsdGenPrimSetSignature { tileCount, instancerCount, primPaths (sorted),
    primvarNames, primTypes }` — structural signature for the generation
    diff (03 §6.1).
  - `UsdGenTileDirty { tile, primPath, removed, added, pointsDirty,
    widthsDirty, xformDirty, newPrimvars, dirtyPrimvars }` — consumed by
    the imaging notice emitter (06 §5.1 rules).
  - `UsdGenDirtyReport { tiles, surfaceXformDirty }`.

### 4.4 `op.h` — operator kernel interface (03 §8.1)

- `UsdGenDiagnostics { errors, warnings, HasErrors(), Error(), Warn() }` —
  collected by a node; the imaging layer logs (engine stays TF_WARN-free on
  hot paths).
- `UsdGenParamView` — `FindParam(name)` (nullptr when unauthored and no C1
  default), typed getters with fallback (`GetToken`, `GetDouble` (float
  tolerated), `GetInt` (uint32 tolerated), `GetBool`, `GetVtValue`). 26.08
  note: no `GetAsSafe`; getters use `IsHolding<T>()` + `UncheckedGet<T>()`.
- `UsdGenCapture` — opaque per-node payload, epoch-keyed;
  `ValidForTopology(buffer)`, `Buffer()`, `OwnsBuffer()`, `MutableBuffer()`,
  `Clone()` (type-erased copy for the E-6 incremental recompile path).
- `UsdGenCapturePayload : UsdGenCapture` — M1 shared base: `perCurve`,
  `perCv`, `rampLut` (257 entries); **every concrete payload must override
  `Clone()` with a full-type copy** (base-level clone would slice — the E-6
  defect).
- `UsdGenCaptureContext { desc, params, references, surface, readPhase,
  seed, upstreamGeneration, dispatcher, diag }`.
- `UsdGenEvalContext { time, shutterOffset (0 in P0/P1), desc, params,
  references, seed }`.
- `class UsdGenOp`:
  - Static description: `Type()` (C1 name), `Space()`, `ReadPhase()`,
    `TopologyEffect()` (R14), `Role()`, `TopologyParameters()` ∪
    `ValueParameters()` == mapped property set (S14, asserted with
    `USDGEN_OP_CHECKS=1`), `ReferenceInputs()`, `OutputPrimvars()`,
    `InputPrimvars()` (slot order, R24).
  - `Bind(params, diag)`.
  - `CaptureDigest(ctx)` → `UsdGenEpoch`; `Capture(ctx, upstream, *out,
    diag)` (a generator must populate its output topology into
    `out->buffer`).
  - `Evaluate(ctx, capture, view)` — const, allocates nothing, touches no
    stage (S8); parallel per chunk in the arena (I8/R22).
  - `IsGenerator()`, `CreateCapture()`, `PlanesTouched()` (bitmask:
    `kPlanePoints 1<<0`, `kPlaneWidths 1<<1`, `kPlaneHairT 1<<2`).
- Five M1 types: `UsdGenScatterOp`, `UsdGenGrowOp`, `UsdGenNoiseOp`,
  `UsdGenLengthOp`, `UsdGenWidthOp` (02 §2.6/§2.7.1).
- `UsdGenBlendEnvelope` / `UsdGenBlendEnvelopeVec3` — the blend envelope,
  applied by the **framework**, not kernels (03 §8.5; exact endpoints:
  `w <= 0` aliases the input, `w >= 1` runs no blend pass — usdRig
  `RigExecBlendEnvelope` contract).

### 4.5 `graph.h` — the compiled graph

- `UsdGenCompiledNode` — one compiled node: `desc`/`descIdx`, `id`, `type`,
  `algorithmVersion`, `op` (unique_ptr), `input` (primary, dense-range
  tested) + `inputs`, resolved `space`/`readPhase`/`topoFx`/`role`/`enabled`;
  `structuralDigest` (Merkle d(n), 03 §3.3), `captureEpoch` (§3.4),
  `valueVersion`, `topologySeq`; `buffer` (S24), `capture`, `chunks`,
  `chunkDirty` (one byte per chunk, UsdGenDirtyBits), `chunkCaptured`;
  compile-time routing (`paramView`, `paramRouting` (name → bits),
  `descendants`, `surface`/`hasSurface`, `curveRefs`, `mapRefs`,
  `captureNeeded`); evaluation-skip state (`paramValueDigest`,
  `lastParamDigest` — keeps SI-2's "nothing else" notice set exact).
- `class UsdGenGraph` (move-only):
  - `NodeCount()`, `Node(id)`, `Desc()`, `NodeDesc(id)`.
  - Dirty routing targets (03 §5.2 hops 2–3; the imaging router feeds
    these): `DirtyParameter(id, param)`, `DirtyCapture(id)`,
    `DirtyTopology(id)`, `DirtySurface(surface, bits)`,
    `DirtyChunks(id, chunks)`, `DirtyMap(mapPrim)` (asset path edit,
    `ReloadMaps()`, textureGeneration bump), `DirtyCurves(curvePrim)` (C3
    BasisCurves changed: guide edit, re-freeze, re-import).
  - `Output()` (terminal node's output buffer), `Chunks(id)`, `Tiles()`,
    `AnyDirty()`, `NodeIdForPath(path)`, `MarkNode(id, bits)` (ORs bits and
    propagates value bits to every strict descendant), `Repartition(
    totalCurves, cvCount)` (re-partitions every node + tiles after a
    topology change; unchanged nodes keep their chunk/dirty bytes).
  - Partition state: `ChunkSize()`, `TileTarget()`, `ChunksPerTile()`,
    `NumTiles()`, `TerminalNodeId()`.

### 4.6 `scheduler.h` — private task arena + commit driver

- I8: every parallel region runs in a **private** `tbb::task_arena` sized at
  max(8, fast-core count) on heterogeneous Linux (the measured 8-thread
  knee of EV-001/EV-008 is the floor), never the process-default
  arena; `USDGEN_THREAD_LIMIT` pins it, else one-shot `CalibrateThreads()`
  (sweeps {2,4,8,16,min(20,physical)}, smallest within 5 % of best). All
  loops are `tbb::parallel_for` inside `_arena.execute` — never `pxr
  work::`.
- `UsdGenNodeRunStats { id, captureMs, evalMs, chunksEvaluated }`;
  `UsdGenRunResult { terminalOutput, tiles, superseded, topologyChanged,
  diagnostics, nodeStats }`.
- `UsdGenWorkDispatcher(arena)` — `ParallelFor(count, body, payload)` for
  intra-node capture parallelism.
- `UsdGenScheduler::Run(graph, evalCtx, generationRequested)` — the 8-step
  commit (03 §5.4; steps 1–6 engine-side, step 7 diff + step 8 notices on
  the session/imaging side): topological evaluation of dirty chunks,
  reference lane first, dirty-tile interleave, supersession checks between
  nodes.

### 4.7 `session.h` — the engine front door

- `UsdGenPendingDirty { structural, surfaceTopology, nodeBits (map<NodeId,
  bits>), surfaceBits }` — pure-value payload from the imaging router (03
  §5.1); a structural change resets the pending set to
  `UsdGenDirtyStructural` on every node.
- `class UsdGenSession(int threadLimit = 0)` (non-copyable):
  - Input (imaging/notice thread): `SetGraphDesc(desc)` (session **copies**
    the desc, 03 §2.2; marks structural-dirty), `SetContext(ctx)`,
    `AccumulateDirty(pending)` (thread-safe, lock-free fast path when
    clean), `NeedsCommit()`.
  - Commit (commit thread only, E-2): `Commit(frame, reason)` → the
    published generation, or the **previous** generation unchanged when
    superseded mid-run (leaves `_dirty` set). 8-step commit under one
    non-recursive `_commitMutex`.
  - `Graph()` (M1Imaging contract: the router rebuilds its table after each
    commit), `Generation()`, `LastReport()` (consumed by the imaging notice
    emitter, 06 §5.1), `InvalidateAllValues()` (bench hook E-1/E-2).
  - Density drag (S28): `BeginDensityDrag()` / `EndDensityDrag()`.
  - Diagnostics: `Stats()`, `NodeStats(id)`.

### 4.8 Remaining engine headers

- `compiler.h` — `UsdGenCompiler`: fills the graph at compile time (03 §3);
  Merkle structural digests (§3.3), capture epochs (§3.4), incremental
  recompile reuses captures via `UsdGenCapture::Clone()` (E-6). See header.
- `opRegistry.h` — C1 registry lookup: C1 type name → `UsdGenOp` factory
  (docs/freezes/C1.md). See header.
- `generationStore.h` — `UsdGenGeneration` / `UsdGenGenerationConstPtr`:
  immutable published generation payload; the imaging side `atomic_load`s
  it (I7). See header.
- `opParams.h` — `UsdGenBaseTopologyParams()` / `UsdGenBaseValueParams()`,
  the shared halves of every operator's parameter partition. The mask is no
  longer a block: `usdGen:mask` is one value-class parameter declared by each
  styler and deformer, and a connection to a `UsdGenExpression` drives it.
  See header.
- `stats.h` — `UsdGenStats` + per-commit timing ring (03 §7) drained by the
  imaging bridge; `UsdGenNodeStats`. See header.
- `usdGen.h` — umbrella header: `USDGEN_CORE_API std::string
  GetVersionString()`, `kDefaultChunkSize 512`, `kDefaultTileTarget 64`.
- `export.h` — `USDGEN_CORE_API` visibility macro.

---

## 5. Imaging lane API (`usdGenImaging::`)

Headers in `libs/usdGenImaging/usdGenImaging/`.

### 5.1 `usdGenImagingSession.h` — session key, session, global store

- `UsdGenSessionKey { UsdStageWeakPtr stage (PRIMARY — the stage object,
  never its root layer), std::string sessionId (usdGen:sessionId, FALLBACK
  only), SdfPath groomRoot }`. A stage wins over `sessionId` (two stages
  authoring the same `sessionId` must NOT share a session); with neither,
  the key degrades to the groom root path plus one `USDGEN_COMMIT` debug
  line (06 §3.7/§9). Hash/equality defined in-header (`UsdGenSessionKeyHash`).
- `class UsdGenImagingSession : TfRefBase, TfWeakBase` — one usdGen session:
  owns the engine session, the current frame/context, the publication
  generation counter. Several scene indices may attach (renderer switch,
  Storm + hdPrman preflight); the session owns ONE current frame, ONE
  context, ONE generation — each attached index republishes the same
  generation; the prim set never differs per index.
  - `Key()`, `Engine()` (→ `usdGen::UsdGenSession *`).
  - `SetTime(frame)` — trigger (a)/(b) time source; **frame-only, never
    commits**.
  - `SetContext(ctx)` — ADR §2.3 explicit context (session property).
  - `Commit(reason)` — trigger (a) explicit (live-override) and trigger (c)
    end-of-batch (06 §3.9, R32 unconditional). Publishes a generation only
    when dirty or a desc was staged; publication generation advances in
    lockstep with the engine's.
  - `MarkAppDriver()` / `HasAppDriver()` — once an external `SetTime` has
    driven the session, frame dirties never trigger commits (rule b).
  - `MarkNeedsDesc()` / `NeedsDesc()` — the staged `UsdGenGraphDesc` is
    consumed by the next commit (S14: one desc pull per topology
    generation, not per frame).
  - `RegisterRepublishCallback(cb)` / `UnregisterRepublishCallback(token)` —
    invoked after a publishing commit on the commit thread; callbacks must
    not block on `GetPrim`; invoked with the lock released (re-entry safe).
  - `Generation()` (monotonic publication generation),
    `LatestGeneration()` (lock-free atomic load, I7),
    `AttachedIndices()`, `NoteAttach()`, `NoteDetach()`.
  - Refs: `UsdGenImagingSessionRefPtr = TfRefPtr<UsdGenImagingSession>`;
    `UsdGenSessionHandle` alias.
- `class UsdGenSessionStore` (process-global, `GetInstance()`):
  - `Attach(key)` (find-or-create, increments attach count; engine session
    created with the engine-resolved thread limit), `Detach(key)` (drops the
    store's strong ref at zero), `Find(key)`.
  - Registry-level forwarding to every live session: `SetTime(frame)`
    (additionally marks every session app-driven and commits at the new
    frame — the future `UsdGenImaging_SetTime` C ABI forwards here),
    `SetContext(ctx)`, `Commit(reason)`, `Generation()` (max over live
    sessions), `ReloadMaps()` (S13: `ArNotice::ResolverChanged` + bump every
    map's `textureGeneration`; marks every session desc-dirty, warns once).
  - Weak-stage registry: `static SetStage(groomRoot, stage)` /
    `static FindGroomStage(groomRoot)` — the usdprimvar adapter for
    `UsdGenGroom` calls `SetStage` while the stage index builds its data
    sources; a dead stage falls back to `sessionId`.
  - `LiveSessions()` — snapshot of strong refs for iteration without
    holding the store lock across engine calls.

### 5.2 `usdGenEngineBridge.h` — commit-thread gate + stats drain

- `UsdGenEngineBridge` (per-session member of the scene index):
  - E-2 commit-thread gate: `RegisterCommitThread(tid)` records the first
    commit thread; `HasCommitThread()`, `IsCommitThread()`;
    `GateCommit(descriptionPath, where)` must be called before every engine
    `Commit` — `TF_CODING_ERROR`s off-thread commits.
  - `SetTimingHook(hook)` (empty function removes) + `DrainStats(stats)` —
    forwards every ring entry of `usdGen::UsdGenStats` not seen yet,
    oldest first, to the optional hook.

### 5.3 `usdGenDirtyRouter.h` — hop-1 routing (03 §5.1/§5.2)

- `UsdGenDirtyRouter` (pxr namespace; forwards to `usdGen::` types):
  - `Entry { node, bits, surface, surfaceScoped }` — a locator prefix under
    `usdGen/` (or a known surface locator) → (engine node, dirty bits,
    optional surface).
  - `Rebuild(graph)` — one entry list per prim path, generated from each
    node's `TopologyParameters()`/`ValueParameters()` partition plus the 02
    §6 rows, longest prefix first (`usdGen/clump` matches every
    `usdGen:clump:*` leaf; `usdGen/mode` does NOT match `usdGen/length/
    mode`). The bare `usdGen` container locator (an adapter resync) routes
    as graph-structural for that prim (02 §6.6 rule 2).
  - Hop-1 from `_PrimsDirtied`: `Route(entries, *out)`, `RouteAdded(...)`,
    `RouteRemoved(...)` — O(entries): one hash lookup each + longest-prefix
    walk; **NEVER cooks** (I7, S17).
  - `EntryCount()`; `_touchesGraphPath(path)` filters structural notices
    (true when `path` contains/is/contained-by a graph-referenced prim).
  - `SdfPathHash` — std::hash adapter for `SdfPath` (26.08 ships no
    `SdfPath::Hash`).

### 5.4 `usdGenTokens.h`

Single definition of the `usdGen` container token, shared by
`primAdapter.cpp` (schema mappings container) and the RestAPI data source /
scene index (published data root). Plan: 06 §3.5. Canonical home: the global
`usdGenImaging` namespace; the router and prim adapter sit in `pxr` (TfType/
registry visibility) and see the names via using-declarations.

### 5.5 Remaining imaging headers

- `usdGenGraphDescBuilder.h` — builds the `UsdGenGraphDesc` from Hydra data
  sources (S14 pull-all, S11 ramp resolution, S13 asset resolution). See
  header.
- `usdGenTilePublisher.h` — publishes a `UsdGenDirtyReport` as scene-index
  additions/removals/primvar dirties per the 06 §5.1 notice rules (C2).
  See header.
- `registry.h` — `UsdGenImagingRegistry`: the M1 imaging identity/version
  surface (replaces the removed M0 placeholder `GetImagingVersionString`,
  sol S-8). See header.
- `usdGenEnable.h` — feature/enable toggles. See header.
- `api.h` — public umbrella; deliberately carries **no** free version
  function (sol S-8 note in header).

---

## 6. RestAPI data sourcing (`usdGenRestApiDataSource.h`)

The `usdGen/rest` container publishes the REST surface of a bound Mesh:
`usdGen/rest/points` (VtVec3fArray at `UsdTimeCode::Default()`),
`usdGen/rest/faceVertexCounts` (VtIntArray),
`usdGen/rest/faceVertexIndices` (VtIntArray), `usdGen/rest/st` (VtVec2fArray,
primary uv set). Rest points are sampled at `UsdTimeCode::Default()` by a
custom `UsdImagingDataSourceMapped::AttributeMapping::factory` whose data
source calls `Get<T>(&r, UsdTimeCode::Default())` — the rest channel costs
no per-frame dirty even when scene time moves (MEASURED, plan/research/
G-stage-free-parameter-and-time-transport.md §3 route R2). Capture-on-first-
cook is rejected (S12).

```cpp
/// AttributeMapping::factory for the usdGen/rest container: builds the four
/// rest leaves on the given prim. Returns nullptr when the prim is not a
/// Mesh (the caller warns once per prim, 02 §2.15).
HdContainerDataSourceHandle UsdGenRestApiContainerFactory(
    UsdPrim const &prim,
    UsdImagingDataSourceStageGlobals const &globals);

/// Invalidation mapping required for the container to be dirtiable (02 §2.15):
///   points / faceVertexCounts / faceVertexIndices -> usdGen/rest/<leaf>
///   primvars:rest                                  -> usdGen/rest/points
///   usdGen:rest:source|primvar|file                -> their own usdGen/rest/* leaves
///   usdGen:rest:file assetPath                     -> usdGen/rest/points (asset)
/// Returns the set of container locators to dirty for the given authoring
/// properties (empty set == no invalidation).
HdDataSourceLocatorSet UsdGenRestApiInvalidateMapping(
    UsdPrim const &prim,
    TfTokenVector const &properties,
    UsdImagingPropertyInvalidationType invalidationType);
```

Wiring (M1, C5): the imaging lane registers the factory with
`UsdImagingDataSourceMapped::RegisterPropertyMappings` and routes invalidations
through the mapping above; M0 `restApiSchemaAdapter.{h,cpp}` is **extended,
not replaced**. Open defect M-8 (docs/review/sol-m1-engine-notes.md): at M0
the factory returned null and the invalidation map was empty — both must be
implemented and both SI-7 tests met.

---

## 7. Test contracts

- `tests/testUsdGenGraph.cpp` — T0 engine tests: synthetic `UsdGenGraphDesc`
  → compile → commit → assert on the published `UsdGenDirtyReport` /
  `UsdGenTilePublication`. No stage, no plugin load (the engine never sees
  a stage, S8). This is the primary behavioral proof for §4.
- `tests/testUsdGenPluginDiscovery.cpp` — plugin load: iterates
  `PXR_PLUGINPATH_NAME` (skipping `plugin/usd` and `plugin/usd/image`),
  loads plug assets, registers `usdGenSchema` / `usdGenImaging` plugins via
  `TfRegistry<PlugPlugin>`; then asserts the five prim types + `UsdGenRestAPI`
  are resolvable and the prim adapters are claimed.
- `tests/testUsdGenContracts.cpp` — the C1/C2/C5 contract test (SI-1 tile
  sizes, SI-2 notice set, SI-7 RestAPI). **Current state: a 77-exit stub —
  the contract assertions are not yet implemented**; building it out is the
  remaining test work for M1 (see `docs/freezes/C1.md` / `C2.md` / `C5.md`).

Build/verify (M1): `set -a; . bin/_env.sh; set +a` then
`cmake --build build --target testUsdGenContracts testUsdGenGraph
testUsdGenPluginDiscovery` (EngineCore owns `libs/usdGen/**`; imaging lane
owns `libs/usdGenImaging/**`).
