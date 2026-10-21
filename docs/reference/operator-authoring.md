<!-- Copyright (c) 2026 Nick Burkard -->
<!-- SPDX-License-Identifier: MIT -->

# Authoring an operator stack

An `UsdGenDescription` binds a surface and owns one `Ops` scope. Place operator
prims beneath that scope. The composed hierarchy supplies the execution order:
children run before parents, and lower siblings run before upper siblings. The
first sibling is the terminal result that usdGen publishes. The small
[`Scatter → Grow` scene](../../examples/scatter-grow-plane.usda) authors
`Grow` above `Scatter` for this reason. There is no authored
`usdGen:input` wire for an ordinary stack.

Most stacks begin with `Scatter` followed by `Grow` or `GuideInterpolate`.
`CurveSource` and `ReferenceSource` can begin with authored curves instead.
Modifiers then refine shape, width, and movement. A source that is disabled
produces an empty curve set; a disabled modifier passes its upstream curves
through. On schema-authored operators, `usdGen:enabled` defaults to `true`.

On schema-authored operators, `usdGen:seed` defaults to `0`. Random draws use stable curve
IDs, so keeping the seed and IDs fixed makes random variation repeatable.
Stylers and deformers inherit `usdGen:mask`, a weight from 0 to 1 that
defaults to 1. A mask of zero preserves the upstream result; a value between
zero and one blends the effect. Put an authored number on the attribute for
uniform strength, or connect a typed `UsdGenExpression` result to vary it
across roots or CVs. Evaluation metadata determines whether an expression
runs once for the groom, each strand, or each CV. The
[`expression-width-plane` scene](../../examples/expression-width-plane.usda)
shows a connected width and frizz mask. `UsdGenExprOp` is different: it
evaluates a program as a geometry or width operation in the stack.

## Default space and pose space

Operators evaluate in Default space until a `UsdGenDeform`. Every operator
upstream of the first deform on its path reads each bound surface (and each
`Collide` collider) at its Default-time rest — the `UsdGenRestAPI` rest that
Scatter already roots on — so its result does not depend on how the surface is
posed or animated. The deform carries that rest-space groom onto the current,
posed surface, and the deform and every operator after it see the posed
geometry. Animating or rigging the surface therefore re-runs only the deform and
what follows it: the rest of the stack keeps its cached result, the strands
are not regrown, and when nothing but points changed the published tiles keep
every other channel and refresh just their points. A stack with no deform stays
at rest. A surface without `UsdGenRestAPI` has no separate Default-time data
and is read as it is, as before. The rule holds on the CPU lane (the
host scheduler); the CUDA and Vulkan executors keep their own surface inputs.

Schema defaults describe attributes, not guaranteed visible results. For
example, `Collide` has `usdGen:pushAmount = 0`, so its default correction is
zero, and `Wind` starts with zero strengths. A relationship needs a valid
target before its related effect can appear. The reference pages call out
declared options that the current kernel refuses, such as vector Displace.
`UsdGenInstance` is declared in the schema but has no registered kernel.
`UsdGenWidthBlend` has a runtime kernel but no generated USD schema type.

The CPU example scenes render through stock Storm. Some operators are on the
CPU reference lane only; CUDA admission rejects those types. A groom whose
`UsdGenGroom` prim selects CUDA may cook device curves, but the stock Storm
device-resident handoff is not available. For an interactive display example,
start with the plain `Xform` groom parent in
[`scatter-grow-plane.usda`](../../examples/scatter-grow-plane.usda) and the
[examples guide](../../examples/README.md).

Pomade's viewport Scale gizmo calls its helper `scaleFactor`; that is an
authoring gesture for scaling selected tube parts. It is not a `UsdGenScale`
operator attribute. The operator's authored scalar is `usdGen:scale`. See
[Pomade](../pomade-tool.md) for the tool and its commit workflow.
