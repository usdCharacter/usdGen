# usdGen restructure: remaining work

Continue from draft PR [#21](https://github.com/usdCharacter/usdGen/pull/21), branch `claude/usdgen-restructure`. Branch from its head, not from main.
Design, contracts and evidence: [architecture review](https://claude.ai/artifact/DtQ2RD4mYhFZaTghW6up8e). Step numbers below are its "Migration order".

## Fixed decisions

- Option B: nothing under `libs/core/` includes or links pxr. Only `libs/hosts/` (usdGenUsd, usdGenHydra) touch OpenUSD.
- Layout: `libs/base/usdGenMath`, `libs/core/usdGen/{api,registries,compile,exec}`, with backends and primitives nested under `libs/core/` (`backends/usdGenCuda`, `backends/usdGenVulkan`, `primitives/usdGen{Surface,Curves,Expr}`). Hydra adapters go in `usdGenHydra/primitives/<name>/`.
- The engine never names an op, backend or primitive. Registration through one `usdGenRegister(Registrar&)` per library is the only way to extend it. Registries hand out handles, not strings, and freeze before the first Session.
- No locks, mutexes, condition variables, once_flags, TBB or atomic `shared_ptr` in core or hosts. Use atomic publication, refcounts and epoch reclamation.
- No heap allocation after warm-up. Only usdGen.dll allocates engine memory, and nothing that owns memory crosses a DLL boundary.
- Time follows Hydra 2.0: `SetTime(frame)` plus shutter offsets, `InputSource::SampleTimes/Sample`, `ElementSet::SampleTimes/Column`, and dirtiness as locators.
- nanobind over the C++ API replaces cApi.h and ctypes.
- No backward compatibility.
- Out of scope: rigExec, cross-primitive work, CPU/GPU parity work.
- Style: compile/exec subdirectories, public and private headers, terse comments.

## Where it stands (PR #21, head after the PLAN commit)

| Commit | What |
|---|---|
| bench | `tests/perf/benchUsdGenRestructure.cpp` and `compareBench.py` |
| step 1 | the seven independent bug fixes |
| gates | `cmake/gates/`: RatchetScan, CheckLinkRatchet, Gates.cmake, baselines; `testUsdGenWarmAllocations` |
| publish path | `UsdGenFurOcclusionCache` (thickness and shell mask reused across bakes), parallel scalp cap; not a numbered step |
| step 2, partial | `registrar.h/.cpp`, frozen op registry, scheduler scratch kept across runs, static trace scopes |
| step 3, partial | `parkingWord.h/.cpp` (futex / WaitOnAddress); workerPool has no mutex |
| PLAN | this file; MSVC fix for the counting allocators |

Main against the branch: 4-core Xeon @ 2.1 GHz, Linux, GCC 13.3 Release, 4 threads, medians of 5 alternating rounds.

| Metric | main | branch |
|---|---|---|
| Cold cook, 100k curves | 143.6 ms | 133.9 ms |
| Warm recook | 128.2 ms | 122.4 ms |
| Param-edit recook | 116.0 ms | 110.7 ms |
| Hydra publish (54 tiles) | 954.5 ms | 350.7 ms |
| Hydra pull | 0.36 ms | 0.48 ms (noise) |
| Allocations, warm cook | 110 | 25 |
| Allocations, param edit | 140 | 80 |
| Allocations, Hydra publish | 42,344 | 14,902 |

Gates (ctest label `gate`). The baselines only go down; lower them in the same commit that removes hits.

| Gate | Baseline now | Target |
|---|---|---|
| `testUsdGenNoLocks` (locks, atomic shared_ptr, TBB) | empty | empty |
| `testUsdGenNoGuardedStatics` | 4 (usdGenMath ramp.cpp) | empty |
| `testUsdGenNoPxrIncludes` | see `noPxrIncludes.txt` | empty under libs/core |
| `testUsdGenNoPxrLink_usdGen` | tf gf vt arch js plug ts ar work trace | none |
| `testUsdGenNoSdfBelowHosts` (no `Sdf*`, no `pxr/usd/sdf` under libs/base, libs/core) | empty | empty |
| `testUsdGenNoPxrLink_usdGenMath` | arch tf gf vt work | none |
| `testUsdGenWarmAllocations` | 0 per warm cook, 0 per param edit (10k curves, all modules counted); 0 interns each | 0 / 0 |
| `testUsdGenNoOpNameLiterals` | empty | empty |
| `testUsdGenNoTokenIntern` | token.cpp, curveLoader.cpp, furOcclusion.cpp | token.cpp only |
| `testUsdGenEngineNoOpCoupling` | empty | empty |
| `testUsdGenTokenChecks` (+`Off`) | an exec-phase intern fails `Run` | stays |
| `testUsdGenNoShimIncludes` (old include spellings) | see `noShimIncludes.txt` | empty at step 12 |
| `testUsdGenCookGoldens`, `testUsdGenOutputGoldens`, `testUsdGenOpContracts` | byte-identical | never regenerated without a reviewed output change |
| `testUsdGenEngineNoCurveBuffer` | empty | empty (subsumed by the 6.5 gate) |
| `testUsdGenNoPrimitiveInEngine` (op names, primitive words, basisCurves, render namespace, CUDA/Vulkan/primitive includes in libs/core/usdGen; comments count) | empty | empty |
| `testUsdGenNoBackendInEngine` (8a: CUDA, Vulkan, Metal, `USDGEN_ENABLE_` in libs/core/usdGen; comments count) | empty | empty |
| `testUsdGenNoBlockingGpuWait` (8b) | 132 hits in 35 files | the keep list at 8.7/8.8 |
| `testUsdGenTimelineSubmitOnly` (8c) | 60 hits in 24 files | `queueTimeline.cpp` only at 8.8e |
| `testUsdGenNoDisallowedLink_{usdGen,usdGenCuda,usdGenVulkan,usdGenImaging}` (8d) | green | stays |
| `testUsdGenNoPxrLink_{usdGenCuda,usdGenVulkan}` | empty | empty |
| `testUsdGenNoPxrLink_{usdGenSurface,usdGenExpr,usdGenCurves}` | arch work trace | none (11) |
| `testUsdGenNoPxrLink_{usdGenCurvesCuda,usdGenCurvesVulkan}` | arch | none (11) |
| `testUsdGenNoPxrUsdImagingBelowHosts` (10a: no `pxr/usd`, `pxr/imaging` include under libs/base, libs/core) | empty | empty |

To refresh a baseline, use the `cmake -DWRITE=ON ... -P cmake/gates/RatchetScan.cmake` line in the header of `Gates.cmake`.

These tests fail on main too, so they aren't regressions:
- `testUsdGenScenePublication` ("4098 superseded CPU generations ...")
- `testUsdGenPomadeIndex` (G6 move budget, timing)
- `testUsdGenGroomExamples` (flaky)
- `testUsdGenStorm{Look,Tangent,HgiResource,Refine,Material}` need `make_scene.py`, so they need a Python-enabled OpenUSD. CI gets past these.

CI on #21: `ctest -L '^T[01]$'` fails only on ScenePublication and PomadeIndex, the same as main's last 5 runs. Build, install, offline configure and software Vulkan all pass.

## Windows (verified 2026-11-24)

RTX 4090, MSVC 14.36, CUDA 12.6, Vulkan on, serial `ctest -L '^(T0|T1)$'`: 420 of 422 pass.
- `testUsdGenScenePublication` fails as on main.
- `testUsdGenPomadeCommit` TN-4 (8 ms idle slot) and `testUsdGenWorkerPool` "at least two workers ran tasks" are timing checks that also fail intermittently on main.
- The 36 Vulkan tests that failed on main on Windows now pass. Causes: a per-module copy of an inline function-local static registry, a text-mode SPIR-V read, a DLL-local `atexit`, and two submissions that ran past the 2 s TDR limit.
- Allocation counting covers every module (`allocCounter.h` wraps the CRT heap imports), so the Windows numbers are real.

Bench, main (b741e6c) against the branch, same machine, 4 threads, medians of 5 alternating rounds:

| Metric | main | branch |
|---|---|---|
| Cold cook, 100k curves | 128.6 ms | 125.0 ms |
| Warm recook | 121.0 ms | 118.4 ms |
| Param-edit recook | 103.3 ms | 3.7 ms |
| Hydra publish | 663.1 ms | 303.6 ms |
| Hydra pull | 0.41 ms | 0.44 ms |

main's allocation rows read 0 on Windows because its counter only sees the test exe.

## Not verified

- **Linux build of everything after the PLAN commit.** Steps 2 and 3 were built and tested on Windows only. CI covers Linux with CUDA off.
- **TSan.** The stress tests ran natively; MSVC has no TSan. Run `testUsdGenLockFreeStress`, `testUsdGenEpoch`, `testUsdGenTaskScheduler`, `testUsdGenPomadeLanes` under TSan on Linux.
- **ARM64.** Store-load handshakes use seq_cst fences, but nothing ran on ARM64.

## Run locally

```
cmake --build build --config Release
ctest --test-dir build -C Release -L gate --output-on-failure
ctest --test-dir build -C Release -j8 --timeout 300
```

The benchmark needs `PXR_PLUGINPATH_NAME` set to the build's `usd/*/resources` dirs, without fakeHdPrman and *Tools, separated by `;` on Windows. compareBench.py does that for you:

```
build\Release\benchUsdGenRestructure.exe --threads 4 --runs 5 [--no-hydra]
python tests/perf/compareBench.py 5 <mainBuild> <branchBuild>
```

To get a main build, use a worktree at b741e6c: main's code plus the bench. Alternate the rounds, because single runs on a busy machine drift by about 10%. Record the rounds, machine and config with every table.

## Remaining work

Each step is one commit (or a short series). Build, tests and gates stay green after every commit, and each step ends with a compareBench table in its PR.

### 2 and 3. Done

- **2:** size-class pools (`pool.h`), per-thread scratch (`scratch.h`), captures and written planes recycled through retirement, compiled node kept on a value-only edit. Budgets are 0 / 0.
- **3:** `epoch.h` (epoch reclamation, intrusive refcounts, atomic publication), `taskScheduler.h` (lanes, pending counts, process-wide `UsdGenParallelFor`), `sharedCell.h`, `parkingWord.h` wake words. No lock, TBB or atomic shared_ptr is left under `libs/`; usdGen no longer links TBB.

Left over, none of it blocking 4:

| What | Where it goes |
|---|---|
| CUDA executor still returns synchronously (parks on a callback-woken fence); the completion lane exists but no production path uses it | step 8 |
| Vulkan: only deform and RBF binding use the queue timeline; about 35 submit sites still make a fence each; `sourceWidthJob` calls the blocking `DeformPipeline::Begin` | step 8 |
| Retirement lane is a dedicated thread in usdGenExecutionResources (below usdGen, cannot reach the scheduler) | step 8, when that DLL folds into the backend runtime |
| `UsdGenWorkerPool` and `UsdGenTaskScheduler` are two pools | merge during step 5 |
| Ops outside the bench graph still make a fresh capture on re-capture (no `Recyclable()`) | step 5, ElementSet arenas |
| No cook-lane epoch drain; idle scheduler workers drain | step 5 |
| Pomade lanes keep their own threads | step 13 |
| Cache domains, Ptex cache and op registry are made on first use, not in `Engine::Create` | step 5 (step 4 freezes the registries at the first compile) |

### 4 and 4a. Done

- **4:** `registries/`: op, type and backend-kernel registries with dense handles, frozen by the first compile. Every op registers an `UsdGenOpDef` (param decls with dirty class, routing and expression admission; planes; traits; hooks); `UsdGenOp` has no static virtuals and `UsdGenParamView` reads by slot only. CUDA and Vulkan kernels are registered rows (`UsdGenKernelDef`). `UsdGenInternToken` is the only interner; `UsdGenScheduler::Run`, the cook after `_Prepare` and CUDA/Vulkan completion fail on an exec-phase intern (`USDGEN_TOKEN_CHECKS=0` warns). Gates: no op-name literal, no `TfToken` from text outside the interner and three host-side files, warm cook and param edit 0 allocations and 0 interns (`testUsdGenTokenChecks`, `testUsdGenWarmAllocations`). Goldens and contracts byte-identical to S4-0.
- **4a:** Clump, Length and Collide name their planes through `emittedPlanes` with a generic indexed-plane allocator; RBF binding is Deform's `makeAnimationBinding` hook. No engine file outside ops/ includes an op header or casts to an op class.

Bench, main (b741e6c) against step 4, same Windows machine, 4 threads, medians of 5 alternating rounds, no other build running:

| Metric | main | step 4 |
|---|---|---|
| Cold cook, 100k curves | 131.3 ms | 125.7 ms |
| Warm recook | 121.2 ms | 108.0 ms |
| Param-edit recook | 102.9 ms | 3.2 ms |
| Hydra publish | 662.6 ms | 295.6 ms |
| Hydra pull | 0.41 ms | 0.44 ms |

Left over:

| What | Where it goes |
|---|---|
| `TfToken` from text in curveLoader.cpp (44, host loader), furOcclusion.cpp (15, publish look), limitSurface.cpp (6, pxOsd) | 7, 9, 10 |
| Device channels by `std::string` (cudaGeneration.cpp); capture-time plane lookups by token (wholeStrand, clump, trim, length); `UsdGenPlane::interpolation/type` are tokens | 5 |
| ops/rbfField.h included by rbfAnimation.h, vulkan/rbfVkPlan.h, vulkan/executionPlan.cpp, vulkan/surfaceVk.cpp (engineNoOpCoupling baseline) | 6 and 8 |
| Hydra vocabulary in sessionCooker (engine tokens, no interning) | 9 |
| Op registry is built on first use and frozen by the first compile; Ptex cache and the cache-domain table are still made on first use. There is no `Engine::Create` to own them | 5 |
| ~~CUDA plan task types spelled in two pieces for gate 1~~ | done in 5.6: `CudaScatterGrow`/`CudaPublication` |
| `testUsdGenScenePublication` also aborts at exit intermittently (0xc0000409), on restructure/s4 too | not a step; owner of the imaging exit path |

### M0 and 5. Done

- **M0:** `libs/base/usdGenMath`, `libs/core/usdGen/{api,registries,compile,exec}`, `libs/core/backends/usdGen{Cuda,Vulkan}`, `libs/core/primitives/usdGen{Surface,Curves,Expr}`. Old include spellings forward through `cmake/IncludeShim.cmake`; `testUsdGenNoShimIncludes` ratchets them down.
- **5.0:** `testUsdGenCookGoldens`: published geometry of record (examples plus 32 single-op scenes, frames 0/1/12, a value and a topology edit).
- **5.1, 5.2:** `api/{span,token,ids,channel,arena,elementSet,writeSet}.h`, `exec/chunkView.h`; the CurveSet layout with its partition and groups hooks, views and converters in `usdGenCurves/types/curveSet.*`.
- **5.3:** every op on `Capture(ctx, UsdGenOpInputs)` / `Evaluate(ctx, cap, UsdGenChunkView)`; the old virtuals and the bridge are gone.
- **5.4:** node outputs are element sets in recycled arenas; chunk ranges and publication groups come from the layout hooks; reference values hold sets; owning and whole-strand captures recycle; the epoch drains at the end of every cook.
- **5.5:** generic names (`countChanges`, `UsdGenLane`, `DirtyInput/inputRefs/UpdateInput`, device element and layout names).
- **5.6:** `curveBuffer.h` is gone from graph, scheduler, cache, op and the registries (`UsdGenPlane` is in `compile/types.h`, `UsdGenReferenceSet` in `exec/chunkView.h`, `UsdGenResolvedReferenceValue` in `op.h`). Gate `testUsdGenEngineNoCurveBuffer`. CUDA plan task types are `CudaScatterGrow`/`CudaPublication` (no op-name dodge).
- Goldens, contracts and cook digests byte-identical throughout; warm cook and param edit 0 / 0.

Bench, main (b741e6c) against step 5, same Windows machine, 4 threads, medians of 5 alternating rounds. The machine was NOT idle: four other worktrees were running ctest suites, so absolute times run about 30% above the step-4 table; compare the columns, not the tables.

| Metric | main | step 5 |
|---|---|---|
| Cold cook, 100k curves | 169.0 ms | 168.1 ms |
| Warm recook | 156.8 ms | 156.4 ms |
| Param-edit recook | 133.6 ms | 5.4 ms |
| Hydra publish | 881.7 ms | 367.8 ms |
| Hydra pull | 0.48 ms | 0.42 ms |
| Allocations, warm cook / param edit | 0 / 0 | 0 / 0 |

Left over:

| What | Where it goes |
|---|---|
| `compiler.cpp` `BuildReferenceValue` builds a curve buffer (gate baseline) | 7.3a |
| `generationStore.h` includes `curveBuffer.h` for the publication types, so `session.*`, `sessionCooker.cpp` and `executionCache.cpp` still see it (gate baseline) | 6.3 |
| ~~Scheduler, compiler, graph and sessionCooker name `UsdGenCurveSetView` and include `usdGenCurves/types/curveSet.h`~~ | done in 6.5 |
| `UsdGenPlane` (tokens for interpolation/type, VtArray payload) is still the registry's plane layout; the brief's `UsdGenChannelDesc` replaces it | 11 (not done in 6) |
| Device channels by `std::string` (`cudaGeneration.cpp` `NamedChannel`) | 8 |
| `UsdGenWorkerPool` is not merged into `UsdGenTaskScheduler`. Perf item; needs its own bench first (task latency, `ParallelFor` throughput, warm-cook time at 100k) | 8 |
| Re-capture allocations outside `testUsdGenWarmAllocations` (per warm re-capture at 10k curves): Clump about 920 (`PickCentres` hash, membership vectors), GuideInterpolate about 1300 (`Delaunay2D`, `BuildGuides`, clumpRef kd-tree), Collide about 20000 (`ValidateColliderMesh` edge map, BVH), Twist and Frizz `Compute` vectors | 6 (usdGenCurves owns the ops) |
| Non-owning captures (Clump, Collide, Smooth, Straighten, ...) are not `Recyclable` | 6, with the item above |
| `UsdGenBuildClumpMotion(UsdGenCurveSetView)` copies the planes, so previous-capture reuse never hits for set inputs | 6 |
| `Recompile` rebuilds every reference value (about 8 allocations per edit with one reference set) | 7 (reuse by NodeKey and generation) |
| Op registry, Ptex cache and cache-domain table are made on first use; there is no `Engine::Create` | 12 |

### 6. Split the primitives. Done (ran after 7, see D1)

- **6.1:** `registries/paramEvaluator.h`: connected parameters behind `UsdGenParamEvaluatorState` (usdGenExpr's `UsdGenCpuParameters`); registrar `RegisterParamEvaluator`, `RegisterPublisher`, `UsdGenEngineEnsureInitialized`.
- **6.2:** `usdGenSurface/types/rootSet.*`: Scatter outputs a RootSet; typed geometry ports and the compiler edge check; CUDA and Vulkan scatter inputs read `UsdGenRootSetView`.
- **6.3:** publication behind `UsdGenPublisherDef` (`usdGenCurves/publish/curvePublisher.*`, `curvePublication.h`); `UsdGenGeneration::publication` is type-erased; hosts read `UsdGenCurvePublicationOf`.
- **6.4:** `usdGenSurface`, `usdGenExpr`, `usdGenCurves` are SHARED targets with `usdGenRegister`; the registry build loads them from usdGen's directory (`USDGEN_EXTENSIONS` overrides); hosts and tests link `usdGenWithPrimitives`.
- **6.5:** the scheduler finds static channels by role (Position, Width, Parameter, Rest, per element), not by curve slot; undeclared emitted channels take the layout's `namedChannel` hook; evaluation failures the op's `failureMessage` hook; an unrun node's output the registered default primitive (`kUsdGenTypeDefaultPrimitive`). Input kinds register `UsdGenInputDesc` roles (Surface, Map) and hooks (validate, digest, lane, authoredChannels, surface ids and Default-space variants, resolveMap, paintSource); the evaluator def gains validateInputs, validateBindings, bindingDependencies (`usdGenExpr/exprValidate.cpp`). `expr::Domain/ScalarType/ValueShape` live in `api/valueShape.h`. Spec names: `UsdGenOpNode::inputSets`, `UsdGenGraphSpec::basis`; the engine tokens drop curves, perCurve and the render namespace (hosts own them). Gate `testUsdGenNoPrimitiveInEngine`; `testUsdGenEngineBare` links usdGen only, loads no extension, and compiles and commits a primitive-free op.
- Goldens, contracts and cook digests byte-identical; warm cook and param edit 0 / 0.

Milestone verification (steps 7 and 6, Windows): clean configure and full build, serial `ctest -L '^(T0|T1)$'`: 441 run, 423 pass, 14 skipped, 4 failed. `testUsdGenScenePublication` fails as on main; `testUsdGenPomadeCommit` TN-4 and `testPomadeSoakSmoke` (latency, device-wide memory) are load noise (Soak passed alone; TN-4 got 10.2 ms with CPU at 100%). `testUsdGenRbfDeform` was a real failure: since 7.2 the fixtures lowered without a `NodeKeyMap`, so every `Recompile` copied the rest column, its address moved, and Deform's identity-keyed driver selection never hit; the fixtures now lower through a host key map, assertions unchanged. A CUDA-off, Vulkan-off tree configures; the CUDA/Vulkan-plumbed engine TUs (compiler, scheduler, session, sessionCooker, sessionBackendExecutor, deviceGeneration, registrar, cudaExecution, cudaParameters, cudaSurfaceInput, registerCurves{Cuda,Vulkan}) compile in it.

`benchUsdGenRestructure` lowers its two GraphDescs once, before the measured regions, through one key map: since 7.2 `compile.cold` and `cook.paramEdit` had timed the host's SdfPath lowering too (mapless, so every column copied; 0.94 ms compile, 58 allocations per edit), which the main column never paid.

Bench, main (b741e6c) against steps 7+6, same Windows machine, 4 threads, medians of 5 alternating rounds. The machine was NOT idle (CPU at 93-100%: other worktrees building and running usdRig ctest suites), so absolute times run above the step-5 table; compare ratios to main.

| Metric | main | steps 7+6 |
|---|---|---|
| Cold cook, 100k curves | 192.0 ms | 180.0 ms |
| Warm recook | 172.8 ms | 163.9 ms |
| Param-edit recook | 144.9 ms | 6.4 ms |
| Hydra publish | 927.5 ms | 427.9 ms |
| Hydra pull | 0.56 ms | 0.51 ms |
| Allocations, warm cook / param edit | 0 / 0 | 0 / 0 |

Against main, warm recook is 0.95 (step 5: 1.00), param edit 0.044 (0.040), Hydra publish 0.46 (0.42; step 4 idle 0.45). No clear regression at this load; rerun on an idle machine before reading the last two.

Left over:

| What | Where it goes |
|---|---|
| `noPrimitiveInEngine.txt`: CUDA plan include in compiler.cpp, session.h, sessionCooker.cpp; `cuda_runtime_api.h` in session.cpp; the Vulkan default provider in sessionBackendExecutor.cpp; the curve device layout check in exec/device/deviceGeneration.cpp (9 hits) | 8 |
| Primitive sources compiled into usdGen for the backends (`USDGEN_ENGINE_RESIDENT_PRIMITIVE_SOURCES`); usdGen keeps the primitive include roots PUBLIC | 8 |
| Hosts link usdGenCurves for publication (`UsdGenCurvePublicationOf`) | 9 |
| The look block (`UsdGenLookDesc`, `basis`) is still engine spec data; `look.bakeMode` is empty unless a host fills the schema fallback (`UsdGenSchemaLook`) | 9 |
| The engine still knows surfaces and maps as roles (`UsdGenOpNode::surfaces/maps`, surface ids, resolved map values) | 10/11 |
| An empty GraphSpec still fails to compile ("no operator nodes"); the bare-engine test compiles a primitive-free op instead | design call |
| Execution cache keys come from per-kind digest hooks in record order: equality-only values changed, never persisted | none |
| `UsdGenPlane` (tokens, VtArray payload) is still the op registry's plane layout | 11 |
| Re-capture allocations, non-`Recyclable` non-owning captures, `UsdGenBuildClumpMotion` plane copies (step-5 leftovers) | not done in 6; usdGenCurves |

### 7. GraphSpec and NodeKey replace SdfPath. Done

- **7.1:** `api/nodeKey.h`; `libs/hosts/usdGenUsd` with `NodeKeyMap` (SdfPath to node keys, collision re-probe, `KeysUnder`).
- **7.2:** the engine runs on `UsdGenGraphSpec` and node keys; `graphDesc.h` is host-only; `api/inputDesc.h`.
- **7.3a/b:** input records replace typed spec inputs; the builders, image maps and schema live under `libs/hosts/usdGenUsd`, the imaging library under `libs/hosts/usdGenHydra`.
- **7.4:** `usdGenExpr/expressions/valueShape.h` holds a closed value-type table (`FindValueType`, `ValueShapeFromValueType`, `ValueTypeDefault`): Sdf's type names plus the legacy names `SdfSchema` still resolves, with tuple dimensions and Sdf's defaults. `testUsdGenValueTypeTable` (usdGenUsd) checks every registered name and alias against `SdfSchema`. The `SdfValueTypeName` helpers (`ValueShapeFromSdfType`, `FixedArrayElementCount`) moved to `usdGenUsd/resolve/sdfValueShape.h`. sdf is off usdGen's link line and out of `noPxrLink_usdGen.txt`; gate `testUsdGenNoSdfBelowHosts` (no `Sdf*` and no `pxr/usd/sdf` under libs/base and libs/core, comment lines excepted) has an empty baseline.

Left over:

| What | Where it goes |
|---|---|
| `asset`, `opaque`, `group` and `pathExpression` rows have no default in the table (their C++ types live in sdf); a value preview of an expression output declared with one of them now gets an empty `$value` literal instead of Sdf's default. Such outputs have no evaluable shape either way | step 11 (Value) |
| A value type a plugin registers with `SdfSchema` at run time is unknown to the table (`ValueShapeFromNativeType` returns an invalid shape); the stage builder still decodes `SdfValueTypeName` directly | when a plugin needs it |
| `VtValue`, `TfToken` and Gf types remain in core headers (`graphSpec.h` and others; `noPxrIncludes.txt`) | 11 |
| `usdGenUsd::Lower`/`Compile`/`Recompile` without a `NodeKeyMap` use a fresh map, so no column cache: every lowering copies every input column and identity-keyed caches (Deform's driver selection, rest digests) miss. Hosts pass their map; many tests and benches do not | when it matters (or make the mapless overloads take a map) |

### 8. Generic GPU backends

- **Goal:**
  - `backends/usdGenCuda` and `backends/usdGenVulkan` become runtimes only: plan runner, fusion engine, streams, pools, block cache.
  - The kernels move to `usdGenCurvesCuda` and `usdGenCurvesVulkan`, with SPIR-V paths registered.
- **Where:** `cudaExecution.cpp` (about 575 curve references), `gpu/curve*`, `vulkan/curveGrowPipeline`, `deformPipeline`, `exprVk*`.
- **Done:** the link gate holds (a backend depends on usdGen only), one block cache, and the Vulkan software-device CI job passes.
- **Needs a GPU:** CUDA and Vulkan timings before and after.
- **Depends on:** 3 (GPU sync rows) and 6.

M8a (8.0-8.6f, with 9.0, 9.1 and 10.1). Done:

- **P0:** the root CMakeLists.txt is split into `cmake/targets/*.cmake` and `cmake/targets/tests/*.cmake`.
- **8.0:** `benchUsdGenRestructure --backend cpu|cuda|vulkan` (Session section); gates 8a-8c.
- **8.1:** `UsdGenSerialLane`; usdGenExecutionResources folds into usdGen.
- **8.2-8.4:** backend registry by handle (`registries/backend.h`); usdGenVulkan and usdGenCuda are SHARED extensions loaded after the primitives (`USDGEN_DEFAULT_EXTENSIONS`); CUDA cooks through the Session device provider; the engine names no backend (gates 8a and `noPrimitiveInEngine` empty).
- **8.5:** primitive sources leave usdGen; `rbfField` is `usdGenMath/rbf.h` with an injected parallel-for; usdGen drops the primitive include roots, ptex, seexpr, pxOsd and hio.
- **8.6a-f:** device program interfaces (`usdGenCuda/compile/program.h`, `usdGenVulkan/compile/program.h`), shader library by registered path, `UsdGenDeviceOwner::ChannelAddress` (CUDA-GL interop without usdGenCuda); kernels, executors and shaders live in SHARED usdGenCurvesCuda and usdGenCurvesVulkan plus OBJECT usdGen{Surface,Expr}{Cuda,Vulkan}; the runtimes link usdGen and the SDK only (gates 8d); `noise_gpu.cu` reaches Noise through a function pointer.

Milestone verification M8a (Windows, RTX 4090, MSVC 14.36, CUDA 12.6, Vulkan on): clean configure and full build; serial `ctest -L '^(T0|T1)$'`: 459 run, 441 pass, 14 skipped, 4 failed. `testUsdGenScenePublication` fails as on main; `testUsdGenPomadeCommit` TN-4, `testPomadeSoakSmoke` and `testUsdGenPomadeHierarchy` TN-1 (4 ms move, also failed in the step-3 and step-4 full runs) are load timing and pass alone (other worktrees were building, CPU at 50-100%). `testUsdGenInstallTree` and `testUsdGenConsumer` are Unix-only; `cmake --install` was checked by hand (every extension DLL in lib/, 53 SPIR-V files in share/usdGen/vulkan, usdGenShaders resources and plugInfo from the moved folder).

One real failure: a CUDA-off, Vulkan-on tree (the CI Vulkan job's configuration) did not link on Windows. usdGenCurvesCuda's program.cpp called the inline `cuda::Workspace::DeviceIndex()` of a `USDGEN_CUDA_API` class; without CUDA no usdGenCuda source includes workspace.h, so MSVC imported a symbol nobody exported. `CreateState` now reports "built without CUDA" when `USDGEN_ENABLE_CUDA` is off (no provider calls it then). GCC emits the inline locally, so Linux was not affected. With the fix the CUDA-off/Vulkan-on tree builds and passes `-L '^vulkan$'` (129 run, 9 CUDA-parity skips), and the CUDA-off/Vulkan-off tree builds and runs `-L '^(T0|T1)$'` (226 run) failing only ScenePublication and the Pomade timing checks below. `noPxrLink_usdGenSurface.txt` loses the stale pxOsd row (10.1 links OpenSubdiv directly).

Linux audit (no Linux machine): the Clang-for-Linux syntax pass over all 530 translation units of both CUDA-off trees finds only the known environment misses (Windows Python headers in usdNoodles bindings, `isatty` in the generated SeExpr lexer). L1: `CMAKE_INSTALL_RPATH $ORIGIN` for every target, `INSTALL_RPATH $ORIGIN` on usdGenCurves{Cuda,Vulkan}, build-tree RUNPATH by default; each extension's DT_NEEDED names what it uses (the Windows import tables match the link lines: usdGenCuda imports usdGen + CUDA, usdGenVulkan usdGen + vulkan-1). L2: cross-DSO interfaces with virtual methods (providers, programs, hook state) have out-of-line destructors or only same-DSO `dynamic_cast` (op captures are file-local). L3: no function-local statics in libs/core headers. L5: the fp-contract and `-fno-math-errno` lists name existing paths (`deformPipeline.cpp` and the exprVk files at their new homes; `rbf.cpp` keeps the toolchain default). L6: `usdgen_opensubdiv_target()` prefers `OpenSubdiv::osdCPU`. W1: largest export table is usdGenExpr (20,665).

Bench, main (b741e6c) against M8a, CPU, 4 threads, medians of 5 alternating rounds, machine not idle (CPU 45-50%):

| Metric | main | M8a |
|---|---|---|
| Cold cook, 100k curves | 200.0 ms | 168.8 ms |
| Warm recook | 175.8 ms | 161.7 ms |
| Param-edit recook | 148.5 ms | 5.6 ms |
| Hydra publish | 940.8 ms | 364.1 ms |
| Hydra pull | 0.51 ms | 0.44 ms |
| Allocations, warm cook / param edit | 0 / 0 | 0 / 0 |

GPU Session section, 8.0 (683a8c7) against M8a, same conditions (CPU 20-40%):

| Metric | CUDA 8.0 | CUDA M8a | Vulkan 8.0 | Vulkan M8a |
|---|---|---|---|---|
| session.cold | 29.9 ms | 28.5 ms | 265.1 ms | 259.2 ms |
| session.warm | 9.0 ms | 8.9 ms | 77.0 ms | 84.1 ms |
| session.paramEdit | 27.8 ms | 27.3 ms | 171.9 ms | 165.2 ms |

Vulkan warm read +9% and +11% in two 5-round runs; 8 alternating rounds of 9 runs each gave 61.1 ms (8.0) and 60.9 ms (M8a), so it is noise.

Left over:

| What | Where it goes |
|---|---|
| usdGenCuda and usdGenCurvesCuda exist in CUDA-off builds (CPU-side admission, "cuda" registered unavailable); the brief's L8 wanted no target | design call |
| Gates 8b and 8c at their 8.0 ceilings | 8.7, 8.8 |
| CUDA-off trees: `testUsdGenPomadeHierarchy` TN-1 (CPU tessellation, ~4-5 ms against 4 ms) and `testUsdGenPomadeIndex` G6 (no pinned memory, 6-9 ms against 8 ms) fail on this machine, the same at 2baa04c | Pomade budgets per lane |
| Linux CI run (Vulkan on lavapipe, install check) | when the branch is pushed |
| 8.7-8.11 (sync executor deletion, async CUDA/Vulkan cooks, channel tokens, worker pool into the scheduler) | M8b |


### 9. PublishDesc and the Hydra adapters

- **Done (M8a):** 9.0 `api/publishDesc.h` and `UsdGenTypeDef::publish`; usdGenCurves registers the CurveSet publish desc and its publisher calls the `previewValues` and `occluder` hooks. 9.1 the hair shaders live in `libs/hosts/usdGenHydra/primitives/curves/shaders` (plugin name, resource layout and install dir unchanged).

- **Goal:** tiles, publication types, hair shaders, fur occlusion (including the new `UsdGenFurOcclusionCache`) and the Storm index layouts move into `usdGenHydra`: generic publishers plus `primitives/curves/`. Locator routing goes through the API.
- **Where:** `sessionCooker.cpp:674` (tile paths), `:2403` (basisCurves), `furOcclusion.*`.
- **Done:** usdGen has no Hydra vocabulary.
- **Proves it:** `hydra.publish` and `hydra.pull` must not regress; scene publication and pomade index behave as on main.
- **Depends on:** 6 and 7.

### 10. OpenSubdiv Far replaces pxOsd. Done

- **10.1 (M8a):** `limitSurface.cpp` builds its `Far::TopologyRefiner` through its own `TopologyRefinerFactory` specialization instead of `PxOsdRefinerFactory`; usdGenSurface links `usdgen_opensubdiv_target()`; gate 10a `testUsdGenNoPxrUsdImagingBelowHosts` is on with an empty baseline; no pxOsd in any link baseline.

- **Goal:** usdGenSurface uses OpenSubdiv Far instead of pxOsd (`limitSurface.cpp`).
- **Gate:** turn on the include gate for `pxr/usd` and `pxr/imaging` under libs/core.
- **Done:** pxOsd is gone from the link baseline. This is all option A would have given.

### 11. Base types

- **Goal:**
  - Add Token, the usdGenMath types, CowArray, Value and Status, first as aliases of Tf, Gf and Vt.
  - Move the code onto them.
  - Swap in the arena-backed implementations and add host conversions.
- **Done:** `noPxrIncludes.txt` is empty for libs/core, both link baselines are empty, and usdGenMath (today it links arch tf gf vt work) moves to `libs/base`.
- **Depends on:** 7 and 10.

### 12. Install only api/

- **Goal:** install only `api/`, re-home the tests that reach internal headers, and use hidden visibility.
- **Gate:** add the install gate, which fails if anything outside `core/usdGen/api/` is installed.

### 13. nanobind

- **Goal:** the app layer moves to nanobind over the C++ API. Delete `usdGenImaging/cApi.h`, `usdGenToolsApi.h` and the ctypes layer. Pomade moves onto `Cow<>` state over usdGenCurves.
- **Note:** Pomade's lock rows (step 3) are easiest to do here.

### 14. Second primitive

- **Goal:** add feathers or scales.
- **Done:** no file in usdGen changes, and at most one adapter folder appears under `usdGenHydra/primitives/`. That is the test that the split worked.

## Order and parallelism

```
4 -> 4a -> 5 -> 7 -> 6 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14
               └─> 8 (after 6)
```
