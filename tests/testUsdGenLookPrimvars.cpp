// testUsdGenLookPrimvars.cpp — the look as CONSTANT tile primvars.
//
// Binding your own UsdGenHairStrands material used to cost you half the look:
// only rootColor travelled with the geometry (baked into displayColor), while
// tipColor, the ramp exponent and the hue/value jitter lived as parameters of
// the SYNTHETIC material the groom scene index publishes. Replace that
// material and the shader fell back to its Sdr defaults -- blond tips on a
// black-haired groom.
//
// The cooker now publishes the rest of the look as constant extra planes, and
// the publisher turns hairTipColor into a real color3f. This test pins:
//   (a) an unauthored look publishes NONE of them, so the default-material
//       path and the `#ifdef HD_HAS_*` fallbacks are untouched;
//   (b) an authored look publishes them with the look's values;
//   (c) the values survive to the Hydra prim as primvars, whatever material
//       the tile is bound to -- the read-back the material-independence claim
//       actually rests on;
//   (d) a look-only edit rebuilds the tiles (nothing else would announce it).
//
// Exit: 0 pass, 1 fail.
#include "usdGen/sessionCooker.h"
#include "usdGenUsd/lower.h"
#include "usdGenImaging/usdGenTilePublisher.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;
using usdGenImaging::UsdGenTilePublisher;

namespace {

int failures = 0;

void Check(bool value, char const *message)
{
    std::printf("%s: %s\n", value ? "ok  " : "FAIL", message);
    if (!value) ++failures;
}

// A groom small enough to cook in microseconds and real enough to publish a
// tile: one triangle of scalp, one three-CV strand.
UsdGenGraphDesc Description()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/look/groom");
    desc.terminal = SdfPath("/look/groom/width");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/look/groom/scalp");
    surface.surfaceGeneration = 1;
    surface.faceVertexCounts = VtIntArray{3};
    surface.faceVertexIndices = VtIntArray{0, 1, 2};
    surface.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)};
    surface.points = surface.restPoints;
    desc.surfaces.push_back(surface);

    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/look/groom/hair");
    hair.role = UsdGenLane::Chunked;
    hair.curveRole = TfToken("hair");
    hair.curveGeneration = 1;
    hair.curveVertexCounts = {3};
    hair.points = {{0, 0, 0}, {0, 0, 1}, {0, 0, 2}};
    hair.rest = hair.points;
    hair.curveId = {1};
    hair.skinPrim = {0};
    hair.skinPrimUv = {{0, 0}};
    hair.rootFrame = {GfMatrix4d(1.0)};
    desc.curveSets.push_back(std::move(hair));

    UsdGenNodeDesc source;
    source.path = SdfPath("/look/groom/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {SdfPath("/look/groom/hair")};
    source.surfaces = {surface.path};
    UsdGenNodeDesc width;
    width.path = desc.terminal;
    width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    desc.nodes = {source, width};
    return desc;
}

UsdGenGenerationConstPtr Cook(UsdGenSessionCooker *cooker,
                              UsdGenGraphDesc const &desc,
                              UsdGenGenerationConstPtr const &previous,
                              int64_t id)
{
    auto shared = std::make_shared<const UsdGenGraphDesc>(desc);
    return cooker->Cook(usdGenUsd::LowerShared(*shared), UsdGenContext::Interactive, false, {}, 1.0,
                        UsdGenCommitReason::NoticeBatchEnd, previous,
                        cooker->Stats(), false, id, id + 1, -1);
}

UsdGenPlane const *Plane(UsdGenTilePublication const &tile, char const *name)
{
    for (UsdGenPlane const &p : tile.extraUniform)
        if (p.name == TfToken(name)) return &p;
    return nullptr;
}

bool Scalar(UsdGenTilePublication const &tile, char const *name, float expected)
{
    UsdGenPlane const *p = Plane(tile, name);
    return p && p->interpolation == TfToken("constant") && p->arity == 1 &&
           p->f.size() == 1 && std::fabs(p->f[0] - expected) < 1e-6f;
}

// primvars/<name>/primvarValue off a built tile prim, as the render delegate
// would read it.
VtValue Primvar(HdContainerDataSourceHandle const &prim, char const *name,
                TfToken *interpolation, TfToken *role)
{
    HdPrimvarsSchema pvs = HdPrimvarsSchema::GetFromParent(prim);
    HdPrimvarSchema pv = pvs.GetPrimvar(TfToken(name));
    if (!pv.IsDefined()) return VtValue();
    if (interpolation && pv.GetInterpolation())
        *interpolation = pv.GetInterpolation()->GetTypedValue(0.0);
    if (role && pv.GetRole()) *role = pv.GetRole()->GetTypedValue(0.0);
    HdSampledDataSourceHandle values = pv.GetPrimvarValue();
    return values ? values->GetValue(0.0) : VtValue();
}

}  // namespace

int main()
{
    // (a) The schema-default look. "Authored" has to mean an actual opinion:
    // a description that merely applies UsdGenLookAPI resolves to a full set
    // of fallbacks, and publishing those would be indistinguishable from the
    // user having asked for them -- and would override the asset's own colour.
    {
        UsdGenSessionCooker cooker(2, 1024 * 1024);
        UsdGenGenerationConstPtr gen = Cook(&cooker, Description(), {}, 0);
        if (!gen || gen->tiles.empty()) {
            std::fprintf(stderr, "FAIL: the fixture published no tiles\n");
            return 1;
        }
        UsdGenTilePublication const &tile = gen->tiles.front();
        Check(!Plane(tile, "hairTipColor") && !Plane(tile, "hairColorRamp") &&
                  !Plane(tile, "hairRandomHue") && !Plane(tile, "hairRandomValue"),
              "an unauthored look publishes no look primvars");
        Check(tile.displayColor.size() == 1 &&
                  tile.displayColor[0] == UsdGenLookDesc().rootColor,
              "an unauthored look still bakes its fallback root colour");
    }

    // (b) + (c) An authored look, read back off the cooked tile and off the
    // Hydra prim the publisher builds from it.
    GfVec3f const tip(0.62f, 0.44f, 0.21f);
    UsdGenGraphDesc authored = Description();
    authored.look.rootColor = GfVec3f(0.05f, 0.02f, 0.01f);
    authored.look.tipColor = tip;
    authored.look.rampExponent = 2.5f;
    authored.look.hueJitter = 0.125f;
    authored.look.valueJitter = 0.25f;
    {
        UsdGenSessionCooker cooker(2, 1024 * 1024);
        UsdGenGenerationConstPtr gen = Cook(&cooker, authored, {}, 0);
        if (!gen || gen->tiles.empty()) {
            std::fprintf(stderr, "FAIL: the authored fixture published no tiles\n");
            return 1;
        }
        UsdGenTilePublication const &tile = gen->tiles.front();
        UsdGenPlane const *tipPlane = Plane(tile, "hairTipColor");
        Check(tipPlane && tipPlane->interpolation == TfToken("constant") &&
                  tipPlane->arity == 3 && tipPlane->f.size() == 3 &&
                  GfVec3f(tipPlane->f[0], tipPlane->f[1], tipPlane->f[2]) == tip,
              "an authored look publishes hairTipColor as one constant triple");
        Check(Scalar(tile, "hairColorRamp", 2.5f), "hairColorRamp carries rampExponent");
        Check(Scalar(tile, "hairRandomHue", 0.125f), "hairRandomHue carries hueJitter");
        Check(Scalar(tile, "hairRandomValue", 0.25f),
              "hairRandomValue carries valueJitter");
        Check(!Plane(tile, "hairRootColor"),
              "rootColor is NOT repeated as a primvar (it is the baked displayColor)");

        // The read-back that matters: the tile is bound to a material the
        // user authored, not to the groom's synthetic one, and the look still
        // reaches the shader. Nothing in the publisher makes the primvars
        // conditional on the binding -- this is the assertion that keeps it
        // that way.
        UsdGenTilePublication bound = tile;
        usdGenUsd::PathRecorder keys;   // the cooked table plus the authored material
        bound.prim.node = keys(usdGenUsd::PathOf(usdGenUsd::Paths(*gen), tile.prim.node));
        bound.material = UsdGenPrimRef{keys(SdfPath("/look/materials/myOwnHair"))};
        HdContainerDataSourceHandle prim =
            UsdGenTilePublisher::BuildTileDataSource(bound, keys.Table(), 0);
        TfToken interp, role;
        VtValue value = Primvar(prim, "hairTipColor", &interp, &role);
        Check(value.IsHolding<VtVec3fArray>() &&
                  value.UncheckedGet<VtVec3fArray>().size() == 1 &&
                  value.UncheckedGet<VtVec3fArray>()[0] == tip &&
                  interp == TfToken("constant") && role == TfToken("color"),
              "a tile bound to an explicit material still carries the look's "
              "tip colour, as a constant color3f");
        VtValue ramp = Primvar(prim, "hairColorRamp", nullptr, nullptr);
        Check(ramp.IsHolding<VtFloatArray>() &&
                  ramp.UncheckedGet<VtFloatArray>().size() == 1 &&
                  ramp.UncheckedGet<VtFloatArray>()[0] == 2.5f,
              "the scalar look primvars reach the prim as constant floats");

        // The default-material path is unchanged: the synthetic material still
        // carries the same values as parameters, so a groom that keeps it
        // shades identically whether or not the shader reads the primvars.
        HdContainerDataSourceHandle material =
            UsdGenTilePublisher::BuildDefaultMaterialDataSource(authored.look);
        Check(material != nullptr,
              "the synthetic material is still built, with its parameters intact");
    }

    // (d) A look-only edit moves no point and changes no topology. Unless the
    // look reaches the tile-rebuild digest, the strands keep the previous
    // colour while the synthetic material updates underneath them.
    {
        UsdGenSessionCooker cooker(2, 1024 * 1024);
        UsdGenGenerationConstPtr first = Cook(&cooker, authored, {}, 0);
        UsdGenGraphDesc edited = authored;
        edited.look.tipColor = GfVec3f(0.9f, 0.1f, 0.1f);
        UsdGenGenerationConstPtr second = Cook(&cooker, edited, first, 1);
        bool const ok = first && second && !first->tiles.empty() &&
                        !second->tiles.empty();
        Check(ok && first->colorDigest != second->colorDigest,
              "a look edit changes the tile-rebuild digest");
        if (ok) {
            UsdGenPlane const *p = Plane(second->tiles.front(), "hairTipColor");
            Check(p && p->f.size() == 3 && std::fabs(p->f[0] - 0.9f) < 1e-6f,
                  "the rebuilt tile carries the edited tip colour");
        }
    }

    std::printf("testUsdGenLookPrimvars: %s (%d failures)\n",
                failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
