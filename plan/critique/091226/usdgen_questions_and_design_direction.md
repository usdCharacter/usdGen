# Questions and Design Direction for a Portable USD Groom System

## Why usdGen is useful

USD can store renderable hair as `BasisCurves`, but it does not provide a standard way to describe how those curves were generated. A baked curve file preserves the result, not the guides, density rules, interpolation, clumps, masks, noise, widths, and other operations that created it.

Groom construction usually remains inside a DCC, a host groomer, Yeti, or another grooming package. Moving a groom often means exporting millions of final curves. The receiving application can render them but cannot easily change density, edit the construction graph, or regenerate the hair from its guides.

usdGen could provide an open representation of groom construction. A compatible application could evaluate the graph, and a renderer could generate final hair only when needed. Applications without usdGen could use an optional baked curve result.

This would provide:

- A smaller asset made from a skin mesh, guides, follicles, parameters, and relationships.
- A groom that can be inspected, edited, and regenerated outside its original application.
- Separate viewport and render densities from the same source.
- Stable results through defined seeds, IDs, and operator versions.
- A common path from USD groom data to standard renderable curves.

The value is not a new curve format. USD already has one. The value is a portable description of how render curves are constructed.

## Influence from current grooming workflows

The usdGen design appears strongly influenced by a host groomer. A host groomer provides useful ideas such as descriptions, generators, guides, modifiers, masks, expressions, clumping, noise, and width controls.

The design should also reflect current host-application workflows. A DCC represents grooming as geometry moving through a node graph. Skin geometry, guides, generated strands, masks, and animation data are visible inputs and outputs. Operators work on curves and attributes and can be connected in different orders.^1

Other systems use similar ideas:

- Yeti uses a graph containing meshes, guides, grooms, generated fibers, effects, samplers, merges, and outputs.^2
- One published groom graph treats guides and final hairs as curves that pass through creation, styling, utility, import, and export nodes.^3
- a character asset Groom Tools combines strand generation, guide interpolation, masks, clumps, parting lines, width, collision handling, and attribute transfer.^4

These systems share a simple model: geometry and attributes move through a construction graph. usdGen should standardize that model instead of copying one product interface.

## The groom bundle

The four basic ideas remain the skin, guides, follicles, and construction graph, but the bundle must also describe how those inputs are used for deformation and rendering. The usdGen reference already separates a groom, one or more descriptions, guide sets, operators, maps, source curves, deformation, and generated Hydra output. The bundle should keep that useful structure while using direct names.

| Bundle element | What it contains | Purpose |
|---|---|---|
| Groom and descriptions | Groom ID, descriptions for regions such as scalp hair or eyebrows, units, versions, and material relationships | Organizes several related hair systems in one asset |
| Rest skin | Rest mesh, topology, UV sets, named regions, and surface attributes | Defines the fixed surface used for grooming and binding |
| Animated skin source | Relationship to the skinned or deformed version of the rest mesh | Drives follicles and skin-based deformation at each frame |
| Follicles | Stable IDs, exact rest-skin locations, rest frames, regions, and seeds | Defines where guide and render hairs grow |
| Rest guide sets | Guide curves, IDs, groups, widths, follicles, orientations, and guide attributes | Defines the authored shape and structure of the groom |
| Animated guide sources | Relationships to rigged, deformed, or simulated versions of guide sets | Drives guide motion without changing the rest groom |
| Construction graph | Generator and styling operators, connections, parameters, ramps, masks, and seeds | Builds and styles the rest render hair |
| Maps and masks | Density, length, region, parting, clump, and operator masks stored on geometry or as referenced data | Controls where and how operators affect the groom |
| Rest render hair | Optional generated or imported dense curves with stable strand and follicle IDs | Allows deformation without regenerating hair every frame |
| Bindings and weights | Follicle-to-skin binding, hair-to-guide weights, interpolation-mesh data, shape capture, point capture, or RBF sample data | Stores expensive relationships once and reuses them during animation |
| Render settings | Preview and final density, curve basis, widths, required attributes, materials, purpose, and tiling | Defines the standard `BasisCurves` output presented to Hydra |
| Cached or baked curves | Optional evaluated rest or animated `BasisCurves` | Speeds evaluation and supports applications without usdGen |

Every geometry type can store data at the point and primitive level. The custom usdGen library reads this data while evaluating the graph.

Examples include:

- Skin point data such as rest position, normals, UVs, density, length, and region masks.
- Guide point data such as position, curve parameter, orientation, width, and simulation velocity.
- Guide primitive data such as guide ID, group, clump ID, and follicle ID.
- Follicle data such as skin face, barycentric position, UV, rest frame, strand ID, and random seed.
- Render-hair point data such as position, width, color, and curve parameter.
- Render-hair primitive data such as strand ID, guide IDs, guide weights, clump ID, and material group.

### Follicles

A follicle is the growth point of a guide or render hair. It should be more than a 3D point. A useful follicle record contains:

- A stable follicle ID.
- The skin mesh it belongs to.
- A face or triangle ID and barycentric coordinates.
- An optional UV coordinate for texture lookup.
- A rest position and orientation frame.
- Region and group membership.
- A deterministic generation seed when the follicle is procedural.

Guide follicles are normally stored because artists edit guides. Render-hair follicles may be stored or regenerated from the rest skin, density data, distribution operator, seed, and operator version. Once generated, they still need stable IDs and exact skin bindings.

UV coordinates alone should not define attachment. UV seams and overlapping UVs can point to more than one place. A face and barycentric position provide an exact location. UVs remain useful for textures, masks, and groom transfer.

### Two paths inside the bundle

The bundle should describe a rest-construction path and a deformation path. Both paths refer to the same skin, guides, follicles, IDs, and coordinate spaces.

```text
REST CONSTRUCTION
rest skin
  + follicles
  + rest guide sets
  + maps and masks
  + construction graph
  -> rest render hair

DEFORMATION
rest skin + animated skin
  + rest guides + animated or simulated guides
  + follicles
  + saved bindings and weights
  + optional rest render hair
  -> animated render hair
```

This split does not place deformation outside the groom bundle. It keeps the rest and animated inputs together while preserving their different roles. Rest skin and rest guides define the stable reference. Animated skin and animated guides supply time-varying motion. Follicles connect the curves to the skin. Bindings connect dense render hair to the drivers.

Generated data should record which skin, guide set, graph, parameters, and operator versions produced it. A consumer should be able to detect when weights or cached curves are out of date.

## Questions about the bundle

1. Which parts are authored, externally referenced, cached, or regenerated?
2. What data must be stored for every follicle?
3. How are follicle and strand IDs kept stable when density changes?
4. How are guide groups, parts, and clump levels represented?
5. How do masks and attributes move from skin and guides to generated hair?
6. How are rest space, object space, and world space identified?
7. What changes require guide weights or deformation bindings to be rebuilt?
8. Can an imported dense groom enter the graph without being regenerated?
9. How are rest guides paired with their deformed, rigged, or simulated guide sources?
10. Can the bundle include baked curves when the evaluator is unavailable?

## Interpolation methods

Interpolation should be divided by purpose. Generating hair from guides is different from moving existing hair with animation.

### Generating render hair from guides

| Method | Use | Limitation |
|---|---|---|
| Surface based guide weights | General fur and hair where parts and skin regions matter | Requires a stored surface weight map or interpolation mesh |
| Distance based guide blend | Fast generation from ordinary sparse guides | Can choose guides across folds, parts, or nearby surfaces |
| Closest guide | Preview or very dense guide sets | Can create visible boundaries between guide regions |

A DCC Hair Generate uses guide distance or skin coordinates, influence radius, influence decay, guide count, guide angle, clump crossover, and skin-space orientation.^5 Its Guide Interpolation Mesh stores guide indices and smooth biharmonic weights on a low-resolution skin mesh.^6

usdGen proposes up to three guide indices and weights per generated hair. Three guides may be a useful fast profile, but the bundle should allow other counts and methods.

### Moving an existing groom with guides

| Method | Use | Limitation |
|---|---|---|
| Whole curve shape matching | Long hair and imported dense grooms | Capture is more expensive |
| Stored guide weights | Hair generated from known guides | Quality depends on the original assignment |
| Surface interpolation mesh | Fur and styles organized along the skin | Less useful for tangled hair volumes |
| One closest guide | Fast preview | Motion can change sharply between guides |

the host application’s Guide Shape Interpolation compares the full shape of each groom curve with nearby guides using samples along the curve. Guides can animate dense curves with different lengths and point counts.^7 This is a useful reference for imported grooms.

### Moving hair with animated skin

| Method | Use | Limitation |
|---|---|---|
| Follicle frame plus animated guides | General character hair | Requires stable guide bindings |
| Surface frame | Short fur | A single root transform cannot bend long hair |
| Point or tetrahedral deform | Braids and tangled volumes | Requires a suitable deformation mesh and capture |
| RBF surface deformation | Smooth skin deformation or groom transfer | Can be expensive and still needs exact follicle attachment |

The a host renderer reference note proposes an RBF field built from stable rest and animated skin samples, followed by a correction that keeps each follicle attached. a character asset Groom Transfer RBF uses RBF to move guides between related source and target skins.^4

RBF should be one deformation method. It does not replace guide interpolation, follicle binding, simulation, collision handling, or shape preservation.

## Generation and deformation should be separate

Generation creates the rest groom. It may create follicles, strands, CVs, widths, IDs, and attributes. Changing density or segment count can change topology.

Deformation moves an existing groom. It should keep the same follicles, strands, IDs, and CV layout. It reads animated skin or guides and writes new point positions.

```text
REST GROOM
skin mesh + guide curves
  -> create follicles
  -> generate render hairs
  -> apply clump, noise, length, width, and other operators
  -> rest groom

ANIMATION
rest groom + animated skin
  -> move follicles and establish the skin motion baseline
  -> add rigged or simulated guide motion
  -> transfer guide motion to render hair
  -> optionally restore length, bend, roots, and clumps
  -> animated render curves
```

This supports render-time generation, saved rest grooms, imported dense grooms, external guide simulation, and final baked curves.

Skin motion should not be applied twice. If the skin deforms both render hair and the guide baseline, guide transfer should add only the difference between the baseline guide and the final guide.

## A custom library instead of SeExpr

SeExpr provides expressions and procedural noise, but it also adds a parser, evaluator, function system, build dependency, and execution model. The current usdGen implementation mainly uses its noise code. That may be too large a dependency for the actual requirement.

VEX is tied to SideFX. MaterialX and Open Shading Language are designed mainly for shading. A general scripting language is difficult to restrict and optimize for millions of hair points.

A small custom library could define only what usdGen needs:

- Vector and matrix math.
- Curve frames and interpolation.
- Ramps.
- Stable hashes and random values.
- Matching CPU and GPU noise functions.
- Access to named skin, follicle, guide, strand, and CV attributes.
- Basic arithmetic, conditionals, and a fixed function list.

The language should remain small. Complex grooming behavior should remain visible as graph operators. Owning the library would let usdGen define portable behavior, deterministic random values, CPU and GPU matching, version rules, and clear errors. The cost is maintaining the parser and functions.

## Rendering value

A DCC demonstrates why render-time hair generation is useful. Its Solaris Hair Procedural generates hair from guides or deforms existing curves. The host application also uses `HoudiniHairDeformAPI` and an `HD_HairDeform` Hydra scene-index plug-in to replace curve points during rendering.^7,8

This reduces stored geometry and evaluates deformation and motion-blur samples close to the renderer. The limitation is that the contract is defined by SideFX and depends on SideFX components.

usdGen could provide the same benefit through an open USD schema and evaluator:

- The USD asset stores the skin, guides, follicles, attributes, and graph.
- The evaluator generates or deforms hair when Hydra requests it.
- Hydra receives ordinary `BasisCurves` with points, widths, IDs, primvars, bounds, and materials.
- Storm, RenderMan, Karma, or another curve renderer receives standard geometry.
- Optional baked `BasisCurves` support applications without usdGen.

This allows preview and final density to come from one groom and avoids storing millions of curves until needed. Tiled generation can update only affected regions.

The case for usdGen is not that it replaces DCC grooming. The case is that host applications, Yeti, standalone groomers, studio tools, and renderers could exchange one open groom construction bundle.

## Main concerns

- Define the bundle before expanding the operator list.
- Do not limit the design to a host groomer terms.
- Treat follicles as first class data.
- Separate guide generation from guide deformation.
- Give RBF a clear and limited role.
- Provide optional baked curves when the evaluator is unavailable.

## Proposed next decisions

1. Define the bundle around groom descriptions, rest and animated skin, rest and animated guide sets, follicles, the construction graph, bindings, and render output.
2. Define point and primitive attributes for each geometry type.
3. Define stable IDs for follicles, guides, render hairs, clumps, and outputs.
4. Define how procedural follicles regenerate deterministically.
5. Separate topology-changing generation from topology-preserving deformation.
6. Support several interpolation methods instead of one fixed guide model.
7. Define exact follicle attachment using face and barycentric data.
8. Define how animated skin and guides combine without applying motion twice.
9. Study a small custom math, noise, and expression library as a SeExpr replacement.
10. Define render-time generation, caching, invalidation, motion samples, and baked output.

## References

1. SideFX, [Hair and Fur](https://www.sidefx.com/docs/houdini/fur/index.html), the host application documentation.
2. Peregrine Labs, [Yeti Documentation](https://docs.peregrinelabs.com/).
3. Daniela Hasenbring and Henrik Karlsson, [Hair Grooming with Imageworks Fyber](https://history.siggraph.org/wp-content/uploads/2022/06/2021-Talks-Hasenbring_Hair-Grooming-with-Imageworks-Fyber.pdf), SIGGRAPH 2021 Talks.
4. the host vendor, [a character asset Groom Tools](https://dev.epicgames.com/documentation/character-asset/mh-groom-tools).
5. SideFX, [Hair Generate](https://www.sidefx.com/docs/houdini/nodes/obj/hairgen.html).
6. SideFX, [Guide Interpolation Mesh](https://www.sidefx.com/docs/houdini/nodes/sop/guideinterpolationmesh.html).
7. SideFX, [Configure Guide Deform](https://www.sidefx.com/docs/houdini/nodes/lop/configureguidedeform.html), the host application documentation.
8. SideFX, [a DCC Procedural Hair](https://www.sidefx.com/docs/houdini/nodes/lop/houdinihairprocedural.html).
9. Local reference note, `_interp.md`.
10. Local report, `houdini-22-hair-fur-research.md`.
11. n-burk, [usdGen repository](https://github.com/n-burk/usdGen).
