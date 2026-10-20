# OpenSubdiv limit scattering

Set `int usdGen:subdivisionLevel = 3` on `UsdGenScatter`. The default `0`
retains existing polygon-cage sampling. The limit paths are CPU-only and need
`subdivisionScheme = "catmullClark"`. The default random per-face distribution
also needs a quad mesh; the generation modes work on any polygon mesh (see
[Generation modes](#generation-modes)).

The mesh's boundary interpolation, face-varying interpolation, triangle rule,
holes, corner sharpnesses and crease chains are captured through both the
USD-stage and Hydra builders. Subdivision opinions enter the capture/cache
identity; changes to mesh topology, scheme or tags invalidate surface users.
The renderer and scatterer must receive the same cage and tags.

Level 3 means OpenSubdiv Far adaptive isolation depth 3, using regular bicubic
and Gregory end-cap patches. An 8×8 grid per coarse face estimates its area
and weights sampling. A root's final position and two derivatives come from
patch-basis evaluation, not interpolation of those grid triangles. Area
integration remains an approximation; extraordinary patches use OpenSubdiv's
Gregory approximation at the requested isolation depth.

Root IDs still derive from `(seed, coarse face, root index)` and roots remain
Morton sorted. The original face index and normalized face `(u,v)` survive
through Grow/stylers. CPU Ptex and face-varying paint samplers use those
coordinates instead of projecting the limit point onto the cage. Normal and
tangent frames come from patch derivatives, with USD orientation respected.

## Generation modes

The generation-order distributions (`even`, `uvGrid`, `vertex`, `perGuide`,
or `countMode = "surface"`) draw their roots on the cage exactly as at level 0:
the same reference draws, density-map thinning and strand identities. Each
surviving root then moves to the limit point under it, and its normal becomes
the limit normal, so hair sits on the smooth surface the renderer draws
rather than on the cage inside or outside it. Triangles and n-gons work: a
cage point on a quad maps affinely onto the quad's patch; on any other polygon
it goes through the regular-polygon layout to the Ptex sub-patch around its
nearest corner, so a shared edge or corner is the same limit point from every
face. Roots on hole faces are dropped. `rootPrim`/`rootUV` keep their cage
meaning here (the face and the texture `st`), which is what density maps and
other UV samplers read.

Supported levels are 1–6. Non-Catmull-Clark schemes fail with diagnostics
instead of silently sampling a different surface, and so does non-quad input
on the random per-face path. Convert
non-quads before creating the topology-dependent Ptex maps. CUDA admission
accepts level 0 and rejects nonzero levels until that execution path supports
the same patch/coordinate semantics. The source is the rest surface, as with
the original Scatter contract; this change does not add an animated binding
deformer.

For the puppet still pass, all emitter meshes and the visible body carry the
same 7,276-face quad cage, `edgeAndCorner` boundaries, right-handed orientation,
explicit empty sharpness/hole arrays, and vertex rest points. The visible
mesh has linear face-varying atlas coordinates and no authored faceted
normals. Storm's `veryhigh` complexity maps to refinement level 3 in the
installed OpenUSD engine.

Validation includes an independent analytic bicubic paraboloid for root
positions and normals, thread determinism, preserved UVs, hole exclusion,
orientation, sharpness capture identity, rejection of bad inputs, a varying
Ptex fixture proving that retained UVs are used, and stage/Hydra parity. For
the generation modes, `testUsdGenLimitScatter` also checks that a mixed
triangle/quad mesh and a pentagon evaluate seamlessly across shared edges and
sub-patches, and that even-scattered roots land on the limit point under their
cage point (on the analytic paraboloid where the patches are regular).

Reference implementation APIs: installed `pxOsd/refinerFactory.h`,
`opensubdiv/far/patchTable.h`, `patchTableFactory.h`, `patchMap.h`, and
`primvarRefiner.h`; upstream [OpenSubdiv](https://github.com/PixarAnimationStudios/OpenSubdiv).
