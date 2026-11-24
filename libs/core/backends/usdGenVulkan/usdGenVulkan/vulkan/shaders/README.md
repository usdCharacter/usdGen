<!--
Copyright (c) 2026 Nick Burkard
SPDX-License-Identifier: MIT
-->

# Vulkan compute shader ABI

`width.comp` is the scalar Width kernel. `widthPipeline` dispatches this
restricted kernel; Session planning also uses width blending and expression
pipelines for additional controls. Native execution tests cover the kernel
and adapter.

Descriptor set 0 uses tightly packed storage buffers:

| Binding | Payload | Access |
| --- | --- | --- |
| 0 | `pointCount` float32 input widths | Read only |
| 1 | `pointCount` float32 output widths | Write only |
| 2 | One uint32 error bitmask | Atomic read/write |

Push constants are 12 bytes: uint32 point count at byte 0, float32 width at
byte 4, and uint32 replace (0 multiply, 1 replace) at byte 8. Workgroup size
is 256. The host validates controls even for zero points and skips zero-size
dispatch. It binds distinct input/output storage, initializes status to zero,
and establishes transfer/compute/read visibility explicitly. Neither partial
output nor output with nonzero status may be published. Owners survive native
completion, including failed or cancelled publication.

This kernel requires unit masks/blend, a flat profile, unit root/tip scales,
no taper, and no maps/expressions. Do not silently route other controls through
this kernel; use the matching pipeline for those controls.

Compile and validate without checking generated SPIR-V into source:

```sh
mkdir -p build-vulkan-shaders
.vulkan-toolchain/root/usr/bin/glslang -V --target-env vulkan1.2 \
  libs/usdGen/usdGen/vulkan/shaders/width.comp \
  -o build-vulkan-shaders/width.spv
.vulkan-toolchain/root/usr/bin/spirv-val --target-env vulkan1.2 \
  build-vulkan-shaders/width.spv
```

Successful compilation/validation does not prove device execution or parity.

## Width overlap witness

`widthOverlapWitness.comp` is the test-only overlap probe, ported from
`gpu/widthOverlapWitness.cu`. Descriptor set 0 binding 0 is three uint32
counters (live, max-live, arrivals), all zeroed before the first probe.
Push constants are 8 bytes: uint32 expected arrivals at byte 0 and uint32
spin-iteration dwell budget at byte 4. One thread records the probe and
spins until arrivals reach expected or the budget exhausts; the budget
replaces the CUDA kernel's clock64() deadline so the probe stays portable.
The host admits only expected >= 2 with a nonzero budget (see
`vulkan/widthOverlapWitness.h`). A serial queue observes maxActive 1;
concurrent queues observe 2.

## Length cut/extend

`lengthCutExtend.comp` uses descriptor set 0 bindings 0 and 1 for the packed
XYZ point buffer and curve offsets, binding 2 for the packed XYZ output, and
binding 3 for the uint32 error status. Its push-constant ABI is exactly 16
bytes: `curves` (uint32), `points` (uint32), `phase` (uint32), and `value`
(float32). Phase 0 validates, phase 1 applies a multiplicative scale, and
phase 2 applies an absolute set operation. The operation is the literal
`keepParam` behavior: cutting samples the existing parameter positions and
extension moves only the tip along the final usable tangent. There is no
reparameterization.

Exact no-op copies preserve the input values bit-for-bit. Numeric edits use
the same geometric arc-length and finite-value rules as CUDA, including its
degenerate-strand and tangent behavior, so the shader is intended to retain
CUDA geometric parity rather than introduce a new interpolation policy.

## Length reparameterization

`lengthReparam.comp` adds descriptor-set 0 binding 4, a read-only float32
`hairT` buffer, after the point, offset, output, and error bindings. Its push
constant ABI is 20 bytes: the preceding `curves`, `points`, `phase`, and
`value` fields plus `hasHairT` (uint32). When `hasHairT` is zero, the shader
uses ordinal coordinates; when nonzero, it uses supplied coordinates. Every
coordinate must be finite and in `[0,1]`, but coordinates need not be
monotonic. Endpoints are not required to be 0 and 1, and roots may move.

The final tip is always forced to the requested extended target when the
target exceeds the current length. Even when the target equals the current
length, the shader resamples rather than taking a copy fast path. `radialScale`
and `Cull` do not rebuild the `hairT` buffer; copy-on-write retains the
`hairT` payload for downstream consumers.
# `lengthMinimum.comp` ABI

`lengthMinimum.comp` uses five storage buffers: binding 0 input points, 1 curve
offsets, 2 fresh output points, 3 atomic status, and 4 optional read-only
`hairT`.  Its 32-byte push constant is `{ uint curves, points, phase; float
value; uint hasHairT; float minimum; uint method, rebuild; }`. `method` is 0
for radial scale or 1 for cut/extend; `rebuild` is 0 for keep-param or 1 for
reparameterization. Phase 1 applies a multiplicative value, phase 2 an
absolute value; both clamp the result to `minimum`.

## Length literal randomization V1

`lengthLiteralV1.comp` preserves the minimum module's first five bindings and
adds binding 5: optional stable curve IDs as `uvec2` low/high words, stride 8.
Its 48-byte push block preserves the minimum module's first 32 bytes and appends
`float randomLo, randomHi; uint seedBits, hasStableIds;`. Missing IDs use the
current curve ordinal; supplied IDs are consumed unchanged, including after Cull.
Native publication must prove both stable-ID and hairT owners against the exact
predecessor, including absence. Output points remain a fresh COW allocation.

Random endpoints must be finite and nonnegative; reversed endpoints are sorted.
Paired 32-bit arithmetic implements the pinned CUDA SplitMix64 hash exactly,
without requiring shaderInt64. The signed authored node seed retains its uint32
bit pattern. The high 24 hash bits produce the float draw in [0,1).
Scale evaluates `(current * value) * multiplier` in that order; Set evaluates
`value * multiplier`. The floor follows CUDA `fmaxf`: raw NaN selects the
finite minimum, while an infinite final target is rejected. This explicitly
differs from the older minimum module's raw-nonfinite rejection; old module
ABIs and behavior are unchanged. Nonneutral random ranges select V1; [1,1]
continues through the existing modules regardless of seed.

V1 composes literal randomization with the already-supported method, rebuild,
minimum and downstream actual-length Cull controls. Masks and mapped/field
controls are not implemented by this module.

## Length scalar envelope V1

`lengthEnvelopeV1.comp` retains the six LiteralV1 bindings. Its 56-byte
push block appends `float resolvedEnvelope; uint zeroEnvelope;` to the
48-byte LiteralV1 block. Native admission validates authored node blend and
literal mask amount independently in [0,1], then resolves their float32 product
once for both dispatch and threshold handling. The explicit zero flag prevents
independent host/device multiplication from changing the topology decision.

Phase 1 is Scale, phase 2 is Set, and phase 3 is validation-only copy for
nonneutral-envelope pure Cull. Every phase validates source points, offsets,
hairT, current arc and random controls before a zero-envelope copy. A zero
envelope bypasses target/factor computation, including the positive-floor
zero-current rejection, and preserves every point bit-for-bit. Nonzero
envelopes apply `input + (changed - input) * resolvedEnvelope` with staged
arithmetic, then validate the actual resulting arc.

Nonneutral-envelope pure Cull requires the validation-copy stage before
compaction; neutral Cull retains its old route. Authored positive thresholds
still define topology stages even when a zero envelope makes the effective
threshold zero. Source ID/hairT provenance and COW retention are unchanged.
This module does not implement expression fields, mask profiles, or disabled
nodes. Its numeric and end-to-end validation is still in progress.

The final envelope subtraction, multiplication and addition use the shared
`float32Envelope.glsl` integer binary32 round-to-nearest/even implementation.
This preserves subnormal values even on devices that do not expose
`shaderDenormPreserveFloat32`, without requiring shaderInt64 or CPU geometry
readback. The preceding geometric edit still follows its existing float32
contract. The helper is being validated independently against host binary32
operations; the exact reversed-unit-line subnormal regression already passes.

## Phase-2 gap ports

The phase-2 Vulkan gap children added the following compute shaders. Each is
compiled by the `_usdgen_compile_vulkan_shader` CMake helper (glslangValidator
`-V --target-env vulkan1.2` plus `spirv-val`), and each is proven by its
dedicated parity test rather than by compilation alone.

- `exprVkEvaluate.comp`, `exprVkContext.comp`, `exprVkWidth.comp`,
  `exprVkExprOp.comp` (expression gap): SeExpr IR graph evaluation, the 33
  context fields plus owners map, width application, and UsdGenExprOp
  application. Require `shaderFloat64` and `shaderInt64`; dispatch is 256
  threads per group. Host-split transcendentals run on the CPU lane through
  field-table slots >= 33. Proven by `testUsdGenVulkanExprVkParity`.
- `resampleVk.comp` (resample gap): indexed-CV resampling with the exact CUDA
  double-precision formulas. Requires `shaderFloat64`. Proven by
  `testUsdGenVulkanResampleVk`.
- `mapVkSample.comp` (maps gap): compute twin of the CUDA image sampler
  (wrap/index/texel/bilinear plus epilogue, grid-stride dispatch) with manual
  filtering and explicit `fma()` at the five sites the host oracle fuses.
  Proven by `testUsdGenVulkanMapVkSample`.
- `styleVkWidthRamp.comp`, `styleVkLength.comp`, `styleVkGrow.comp` (styleops
  gap): Vulkan twin of `gpu/styleOps.cu` (WidthRamp, Length, Grow) with
  `precise` FP order matching CUDA `--fmad=false`. Proven by
  `testUsdGenVulkanStyleVkParity`.
- `rbfVkExtent.comp`, `rbfVkGram.comp`, `rbfVkBuildMatrix.comp`, `rbfVkLu.comp`,
  `rbfVkRhs.comp`, `rbfVkTriSolve.comp`, `rbfVkEvaluate.comp`, `rbfVkApply.comp`
  (rbf gap): device-side RBF Bind (extent/gram/matrix-build/LU), Solve
  (RHS/triangular-solve), Evaluate, and Apply for gap Deform configs.
  Proven by `testUsdGenVulkanRbfVkParity`.
- `picktileVkPick.comp`, `picktileVkFootprint.comp`, `picktileVkTiles.comp`,
  `picktileVkBounds.comp`, `picktileVkIndices.comp` (picktiles gap): picking,
  footprint, tile spans, tile bounds, and index streams. Proven by
  `testUsdGenVulkanPicktileVkParity`.

The curvesource (csourceVk) and surface gaps added no shaders: loader
resampling matches the CUDA device resample within its contract, and the
surface fit reuses the existing `deformEvaluate`/`deformApply` dispatches.
Their parity tests are `testUsdGenVulkanCsourceVkParity` and
`testUsdGenVulkanSurfaceVk`.
