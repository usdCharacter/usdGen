#include "usdGen/ops/curveSource.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenSurface/maps/mapInput.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGenCurves/types/curveCapture.h"

#include "usdGen/curveLoader.h"
#include "usdGen/surfaceCageSource.h"
#include "usdGen/op.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/curveBuffer.h"
#include "usdGenMath/usdGenMath/kernels.h"


#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include "usdGen/registries/registrar.h"

namespace usdGen {

namespace {

struct UsdGenCurveSourceCapture final : public UsdGenCurveCapture
{
   std::unique_ptr<UsdGenCapture> Clone() const override
   {
      auto c = std::make_unique<UsdGenCurveSourceCapture>();
      *c = *this;
      return c;
   }
   bool OwnsBuffer() const override { return true; }
   bool Recyclable() const override { return true; }
   bool Recycle() override
   {
      UsdGenRecycleBuffer(_buffer);
      return true;
   }
};

bool BuildAuthoredPlanes(UsdGenCaptureContext const &ctx,
                         std::vector<UsdGenPlane> *extraCv,
                         std::vector<UsdGenPlane> *extraCurve,
                         size_t *authoredCurves, size_t *authoredCvs,
                         std::vector<uint32_t> *authoredOffsets,
                         bool *hasSource,
                         UsdGenDiagnostics *diag)
{
   if (!extraCv || !extraCurve || !authoredCurves || !authoredCvs ||
       !authoredOffsets || !hasSource)
      return false;
   *authoredCurves = *authoredCvs = 0;
   authoredOffsets->clear();
   *hasSource = false;
   if (!ctx.desc || !ctx.params || !ctx.params->node ||
       ctx.params->node->inputSets.empty()) return true;
   UsdGenNodeKey const path = ctx.params->node->inputSets.front();
   UsdGenInputRecord const *curveSetRecord = UsdGenFindCurveSetInput(*ctx.desc, path);
   if (!curveSetRecord) return true;
   UsdGenCurveSetInputView const curveSetView = UsdGenCurveSetInputView::Of(*curveSetRecord);
   UsdGenCurveSetInputView const *curveSet = &curveSetView;
   // A plain CurveSource relationship may carry no named planes (many
   // existing descriptors omit the optional CurveSet topology). Only an
   // authored-plane payload requires source-topology reconciliation here.
   *hasSource = !curveSet->authoredPlanes.empty();
   if (!*hasSource) return true;
   *authoredCurves = curveSet->curveVertexCounts.size();
   for (int count : curveSet->curveVertexCounts) {
      if (count < 0 || *authoredCvs > std::numeric_limits<size_t>::max() -
                                      static_cast<size_t>(count) ||
          *authoredCvs > std::numeric_limits<uint32_t>::max() -
                                      static_cast<uint32_t>(count)) {
         if (diag) diag->Error("UsdGenCurveSource::Capture: authored source topology is invalid");
         return false;
      }
      *authoredCvs += static_cast<size_t>(count);
   }
   bool uniform = true;
   int firstCount = curveSet->curveVertexCounts.empty()
       ? 0 : curveSet->curveVertexCounts.front();
   for (int count : curveSet->curveVertexCounts) uniform = uniform && count == firstCount;
   if (!uniform) {
      authoredOffsets->assign(*authoredCurves + 1, 0);
      for (size_t i = 0; i != *authoredCurves; ++i)
         (*authoredOffsets)[i + 1] = (*authoredOffsets)[i] +
             curveSet->curveVertexCounts[i];
   }
   try {
      for (UsdGenAuthoredPlaneView const &authored : curveSet->authoredPlanes) {
         size_t elements = 0;
         UsdGenToken interpolation;
         std::vector<UsdGenPlane> *destination = extraCv;
         switch (authored.domain) {
         case UsdGenAuthoredPlaneDomain::Point:
            elements = *authoredCvs; interpolation = UsdGenTokens().vertex; break;
         case UsdGenAuthoredPlaneDomain::Primitive:
            elements = *authoredCurves; interpolation = UsdGenTokens().uniform;
            destination = extraCurve; break;
         case UsdGenAuthoredPlaneDomain::Groom:
            elements = 1; interpolation = UsdGenTokens().constant;
            destination = extraCurve; break;
         default:
            if (diag) diag->Error("UsdGenCurveSource::Capture: authored plane has invalid domain");
            return false;
         }
         if (elements > std::numeric_limits<size_t>::max() / authored.arity ||
             elements * authored.arity !=
                 (authored.type == UsdGenAuthoredPlaneType::Float32
                      ? authored.floatValues.size() : authored.intValues.size())) {
            if (diag) diag->Error("UsdGenCurveSource::Capture: authored plane '" +
                                  authored.name.GetString() +
                                  "' cardinality does not match source topology");
            return false;
         }
         UsdGenPlane plane;
         plane.name = authored.name;
         plane.interpolation = interpolation;
         plane.arity = authored.arity;
         if (authored.type == UsdGenAuthoredPlaneType::Float32) {
            plane.type = UsdGenTokens().float_;
            plane.f.assign(authored.floatValues.begin(), authored.floatValues.end());
         } else if (authored.type == UsdGenAuthoredPlaneType::Int32) {
            plane.type = UsdGenTokens().int_;
            plane.i.assign(authored.intValues.begin(), authored.intValues.end());
         } else {
            if (diag) diag->Error("UsdGenCurveSource::Capture: authored plane '" +
                                  authored.name.GetString() + "' has invalid type");
            return false;
         }
         destination->push_back(std::move(plane));
      }
      auto byName = [](UsdGenPlane const &a, UsdGenPlane const &b) {
         return a.name < b.name;
      };
      std::sort(extraCv->begin(), extraCv->end(), byName);
      std::sort(extraCurve->begin(), extraCurve->end(), byName);
      return true;
   } catch (...) {
      if (diag) diag->Error("UsdGenCurveSource::Capture: authored plane materialization failed");
      return false;
   }
}

}  // namespace

namespace {
using namespace curveSourceP;
using D = expr::Domain;

// Only native groom resampleTo/useRest accept an expression (the "usdGen:"
// spellings canonicalize to them).
constexpr UsdGenExprAdmission kIntGroom{expr::ScalarType::Int32, 1, "int", D::Groom};
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};

// A generator: no upstream curves to leave untouched, no mask. Order is the
// old TopologyParameters() list.
UsdGenParamDecl const kParams[kCount] = {
   {"input", UsdGenParamClass::Capture},
   {"surface", UsdGenParamClass::Capture},
   {"seed", UsdGenParamClass::Capture},
   {"enabled", UsdGenParamClass::Capture},
   {"mode", UsdGenParamClass::Capture},
   {"useRest", UsdGenParamClass::Capture, true, kBoolGroom},
   {"usdGen:useRest", UsdGenParamClass::Capture, true, kBoolGroom},
   {"interpolationMode", UsdGenParamClass::Capture},
   {"densityMultiplier", UsdGenParamClass::Capture},
   {"regionMapChannel", UsdGenParamClass::Capture},
   {"expectMapGeneration", UsdGenParamClass::Capture},
   {"resampleTo", UsdGenParamClass::Capture, true, kIntGroom},
   {"idSource", UsdGenParamClass::Capture},
   {"staleAction", UsdGenParamClass::Capture},
   {"expectEpoch", UsdGenParamClass::Capture},
   {"rebind", UsdGenParamClass::Capture},
   {"usdGen:resampleTo", UsdGenParamClass::Capture, false, kIntGroom},
};
constexpr char const *kTokens[] = {"load", "none", "surfaceCage"};
constexpr UsdGenMapSlotDecl kMaps[] = {{"usdGen:regionMap"}};

std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &record, void *)
{
   return std::make_unique<UsdGenCurveSourceOp>(record);
}
}  // namespace

UsdGenEpoch UsdGenCurveSourceOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
   UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
   uint64_t h = 1469598103934665603ULL;
   auto bytes = [&](void const *data, size_t size) {
      auto const *p = static_cast<unsigned char const *>(data);
      for (size_t i = 0; i != size; ++i) {
         h ^= p[i]; h *= 0x100000001b3ULL;
      }
   };
   auto feed = [&](std::string_view k, uint64_t v) {
      bytes(k.data(), k.size());
      bytes(&v, sizeof(v));
   };
   feed("mode", p ? static_cast<uint64_t>(p->GetToken(kMode, Record().Token(kTokLoad)).Hash()) : 0);
   feed("seed", ctx.seed);
   // The surface-source fallback width is descriptor-wide rather than a node
   // parameter.  It still changes the captured buffer, so it must participate
   // in capture identity or a retained source would keep the old COW owner.
   uint32_t defaultWidthBits = 0;
   if (ctx.desc)
      std::memcpy(&defaultWidthBits, &ctx.desc->defaultWidth,
                  sizeof(defaultWidthBits));
   feed("defaultWidth", defaultWidthBits);
   // All resolved node parameters participate, including unsupported values:
   // their edit must not retain a capture that was produced while a different
   // admission decision applied.
   if (p && p->node) {
      feed("nodeParams", p->node->params.size());
      for (UsdGenParamValue const &param : p->node->params) {
         std::string const name = param.name.GetString();
         bytes(name.data(), name.size());
         feed("value", param.value.GetHash());
      }
      feed("curveRelationships", p->node->inputSets.size());
      for (UsdGenNodeKey const path : p->node->inputSets) {
         bytes(&path.value, sizeof(path.value));
      }
      feed("surfaceRelationships", p->node->surfaces.size());
      for (UsdGenNodeKey const path : p->node->surfaces) {
         bytes(&path.value, sizeof(path.value));
      }
      feed("mapBindings", p->node->mapBindings.size());
      for (UsdGenMapBinding const &binding : p->node->mapBindings) {
         std::string const relationship = binding.relationship.GetString();
         bytes(&binding.map.value, sizeof(binding.map.value));
         bytes(relationship.data(), relationship.size());
      }
      // Legacy direct descriptors may still use maps without typed bindings.
      feed("maps", p->node->maps.size());
      for (UsdGenNodeKey const path : p->node->maps) {
         bytes(&path.value, sizeof(path.value));
      }
      if (ctx.desc) {
         auto digestMap = [&](UsdGenNodeKey path) {
            UsdGenInputRecord const *mapRecord = UsdGenFindMapInput(*ctx.desc, path);
            if (!mapRecord) { feed("missingMap", 1); return; }
            UsdGenMapInputView const mapView = UsdGenMapInputView::Of(*mapRecord);
            UsdGenMapInputView const *map = &mapView;
            std::string const type = map->type.GetString();
            bytes(type.data(), type.size());
            bytes(map->resolvedAssetPath.data(), map->resolvedAssetPath.size());
            feed("textureGeneration", map->textureGeneration);
            feed("mapParams", map->params.size());
            for (UsdGenParamValue const &param : map->params) {
               std::string const name = param.name.GetString();
               bytes(name.data(), name.size());
               feed("value", param.value.GetHash());
               feed("animated", param.animated);
            }
         };
         if (!p->node->mapBindings.empty()) {
            for (UsdGenMapBinding const &binding : p->node->mapBindings)
               digestMap(binding.map);
         } else for (UsdGenNodeKey const path : p->node->maps) {
            digestMap(path);
         }
      }
   }
   // C3 data is a captured external owner, not just optional named planes.
   // Hash its complete payload so a descriptor-only recompile recaptures into
   // fresh COW storage without changing an older published generation.
   if (ctx.desc && p && p->node && p->node->inputSets.size() == 1) {
      UsdGenNodeKey const path = p->node->inputSets.front();
      UsdGenInputRecord const *foundRecord = UsdGenFindCurveSetInput(*ctx.desc, path);
      if (!foundRecord) {
         feed("missingCurveSet", 1);
      } else {
         UsdGenCurveSetInputView const foundView = UsdGenCurveSetInputView::Of(*foundRecord);
         UsdGenCurveSetInputView const *found = &foundView;
         auto token = [&](char const *key, UsdGenToken const &value) {
            std::string const text = value.GetString();
            bytes(key, std::strlen(key));
            bytes(text.data(), text.size());
         };
         auto vec3 = [&](char const *key, UsdGenSpan<const UsdGenVec3f> values) {
            bytes(key, std::strlen(key)); feed("size", values.size());
            for (UsdGenVec3f const &v : values) bytes(&v[0], 3 * sizeof(float));
         };
         auto vec2 = [&](char const *key, UsdGenSpan<const UsdGenVec2f> values) {
            bytes(key, std::strlen(key)); feed("size", values.size());
            for (UsdGenVec2f const &v : values) bytes(&v[0], 2 * sizeof(float));
         };
         token("curveRole", found->curveRole);
         token("type", found->type); token("basis", found->basis); token("wrap", found->wrap);
         token("widthsInterpolation", found->widthsInterpolation);
         feed("role", static_cast<uint64_t>(found->role));
         feed("restFromCurrent", found->restFromCurrentPoints);
         feed("curveGeneration", found->curveGeneration);
         bytes(found->frozenEpoch.data(), found->frozenEpoch.size());
         // Ordinary C3 points stay source-local. Root binding consumes only
         // the surface-to-source transform, not shared ancestor motion.
         UsdGenToken const &none = Record().Token(kTokNone);
         bool const cageMode = p->GetToken(kInterpolationMode, none) != none;
         UsdGenInputRecord const *bound = p->node->surfaces.size() == 1
             ? UsdGenFindSurfaceInput(*ctx.desc, p->node->surfaces.front())
             : nullptr;
         if (!cageMode && bound) {
            UsdGenMatrix4d const relative =
                UsdGenSurfaceInputView::Of(*bound).worldMatrix * found->worldMatrix.GetInverse();
            bytes(relative.GetArray(), 16 * sizeof(double));
            bool validWorld = std::isfinite(found->worldMatrix.GetDeterminant()) &&
                              found->worldMatrix.GetDeterminant() != 0.0;
            for (int r=0;r<4;++r) for (int c=0;c<4;++c)
               validWorld = validWorld && std::isfinite(found->worldMatrix[r][c]);
            feed("validSourceWorld", validWorld);
         } else bytes(found->worldMatrix.GetArray(), 16 * sizeof(double));
         feed("curveCounts", found->curveVertexCounts.size());
         if (!found->curveVertexCounts.empty())
            bytes(found->curveVertexCounts.cdata(), found->curveVertexCounts.size() * sizeof(int));
         vec3("points", found->points); vec3("rest", found->rest);
         feed("widths", found->widths.size());
         if (!found->widths.empty()) bytes(found->widths.cdata(), found->widths.size() * sizeof(float));
         feed("skinPrim", found->skinPrim.size());
         if (!found->skinPrim.empty()) bytes(found->skinPrim.cdata(), found->skinPrim.size() * sizeof(int));
         feed("curveId", found->curveId.size());
         if (!found->curveId.empty()) bytes(found->curveId.cdata(), found->curveId.size() * sizeof(uint64_t));
         vec2("skinPrimUv", found->skinPrimUv);
         feed("rootFrame", found->rootFrame.size());
         for (UsdGenMatrix4d const &frame : found->rootFrame)
            bytes(&frame[0][0], 16 * sizeof(double));
         feed("authoredPlanes", found->authoredPlanes.size());
         for (UsdGenAuthoredPlaneView const &plane : found->authoredPlanes) {
            std::string const name = plane.name.GetString();
            bytes(name.data(), name.size());
            feed("type", static_cast<uint64_t>(plane.type));
            feed("domain", static_cast<uint64_t>(plane.domain));
            feed("arity", plane.arity);
            feed("floatValues", plane.floatValues.size());
            if (!plane.floatValues.empty()) bytes(plane.floatValues.cdata(), plane.floatValues.size() * sizeof(float));
            feed("intValues", plane.intValues.size());
            if (!plane.intValues.empty()) bytes(plane.intValues.cdata(), plane.intValues.size() * sizeof(int));
         }
         // surfaceCage is immutable source data, not a generic authored
         // plane.  Hash every captured array so editing sparse rails, owner
         // controls, profile/chart data, or triangle ownership always
         // recaptures transient dense C3 without relying on a generation
         // bump from a scene adapter.
         feed("surfaceCage", found->surfaceCage ? 1 : 0);
         if (found->surfaceCage) {
            UsdGenSurfaceCageView const &cage = *found->surfaceCage;
            auto ints = [&](char const *key, UsdGenSpan<const int> values) {
               feed(key, values.size());
               if (!values.empty()) bytes(values.cdata(), values.size() * sizeof(int));
            };
            auto floats = [&](char const *key, UsdGenSpan<const float> values) {
               feed(key, values.size());
               if (!values.empty()) bytes(values.cdata(), values.size() * sizeof(float));
            };
            auto vec3i = [&](char const *key, UsdGenSpan<const UsdGenVec3i> values) {
               feed(key, values.size());
               for (UsdGenVec3i const &value : values)
                  bytes(&value[0], 3 * sizeof(int));
            };
            ints("cageOwnerIds", cage.ownerIds);
            floats("cageOwnerDensities", cage.ownerDensities);
            ints("cageOwnerSeeds", cage.ownerSeeds);
            ints("cageOwnerCvCounts", cage.ownerCvCounts);
            floats("cageOwnerEdgeBias", cage.ownerEdgeBias);
            ints("cageProfileOffsets", cage.ownerLengthProfileOffsets);
            vec2("cageProfile", cage.ownerLengthProfile);
            floats("cageNormalizedT", cage.normalizedT);
            vec3i("cageTriangles", cage.triangles);
            ints("cageTriangleOwners", cage.triangleOwnerIndices);
            vec2("cageTriangleCharts", cage.triangleRootCharts);
            vec2("cageChartCentroids", cage.ownerChartCentroids);
            floats("cageChartMeanRadii", cage.ownerChartMeanRadii);
         }
      }
   }
   // Root-frame construction consumes the complete rest-surface snapshot,
   // not only its face count.  Every consumed channel therefore participates
   // in capture identity even when an adapter forgot to bump its generation.
   if (ctx.desc && p && p->node && p->node->surfaces.size() == 1) {
      UsdGenNodeKey const path = p->node->surfaces.front();
      UsdGenInputRecord const *foundRecord = UsdGenFindSurfaceInput(*ctx.desc, path);
      if (!foundRecord) feed("missingSurface", 1);
      else {
         UsdGenSurfaceInputView const foundView = UsdGenSurfaceInputView::Of(*foundRecord);
         UsdGenSurfaceInputView const *found = &foundView;
         // The general generation includes posed points and sample times,
         // even on a static rest snapshot. Root capture reads the rest data
         // below, so playback must not regenerate a rest-bound source.
         // Surface-cage interpolation has additional surface dependencies.
         UsdGenToken const &none = Record().Token(kTokNone);
         if (p->GetToken(kInterpolationMode, none) != none)
            feed("surfaceGeneration", found->surfaceGeneration);
         feed("restFromCurrentPoints", found->restFromCurrentPoints);
         feed("restNormalDomain", static_cast<uint64_t>(found->restNormalDomain));
         feed("surfaceFaceCounts", found->faceVertexCounts.size());
         if (!found->faceVertexCounts.empty())
            bytes(found->faceVertexCounts.cdata(), found->faceVertexCounts.size() * sizeof(int));
         feed("surfaceFaceIndices", found->faceVertexIndices.size());
         if (!found->faceVertexIndices.empty())
            bytes(found->faceVertexIndices.cdata(), found->faceVertexIndices.size() * sizeof(int));
         feed("surfaceRestPoints", found->restPoints.size());
         for (UsdGenVec3f const &point : found->restPoints)
            bytes(&point[0], 3 * sizeof(float));
         feed("surfaceRestNormals", found->restNormals.size());
         for (UsdGenVec3f const &normal : found->restNormals)
            bytes(&normal[0], 3 * sizeof(float));
         // Ordinary C3 already hashes the relative binding transform above.
         if (p->GetToken(kInterpolationMode, none) != none ||
             p->node->inputSets.size() != 1)
            bytes(found->worldMatrix.GetArray(), 16 * sizeof(double));
      }
   }
   return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenCurveSourceOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
   (void)params; (void)diag;
   return true;
}

std::unique_ptr<UsdGenCapture> UsdGenCurveSourceOp::CreateCapture() const
{
   return std::make_unique<UsdGenCurveSourceCapture>();
}

uint32_t UsdGenCurveSourceOp::PlanesTouched() const
{
   return kPlanePoints | kPlaneWidths | kPlaneHairT;
}

bool UsdGenCurveSourceOp::Capture(UsdGenCaptureContext const &ctx,
                                  UsdGenOpInputs const &in,
                                  UsdGenCapture *out,
                                  UsdGenDiagnostics *diag)
{
    (void)in;
   if (!ctx.desc) {
      if (diag) diag->Error("UsdGenCurveSource::Capture: no graph description");
      return false;
   }
   if (!std::isfinite(ctx.desc->defaultWidth) || ctx.desc->defaultWidth < 0.0f) {
      if (diag) diag->Error(
          "UsdGenCurveSource::Capture: default width must be finite and non-negative");
      return false;
   }
   // A CurveSource with an explicit C3 relationship is never allowed to
   // silently fall back to a surface surrogate.  Build a fresh candidate
   // first, then exchange the capture owner only after every validation and
   // allocation succeeded; retained generations continue to own the old
   // buffer on failure.
   if (ctx.params && ctx.params->node && !ctx.params->node->inputSets.empty()) {
      if (ctx.params->node->inputSets.size() != 1) {
         if (diag) diag->Error(
             "UsdGenCurveSource::Capture: node '" +
             ctx.desc->Label(ctx.params->node->key) +
             "' must bind exactly one authored C3 curve target");
         return false;
      }
      UsdGenNodeKey const path = ctx.params->node->inputSets.front();
      UsdGenInputRecord const *foundRecord = UsdGenFindCurveSetInput(*ctx.desc, path);
      if (!foundRecord) {
         if (diag) diag->Error("UsdGenCurveSource::Capture: node '" +
                               ctx.desc->Label(ctx.params->node->key) +
                               "' bound C3 target '" + ctx.desc->Label(path) +
                               "' was not found");
         return false;
      }
      UsdGenToken const &none = Record().Token(kTokNone);
      UsdGenToken const &interpolationMode = ctx.params
          ? ctx.params->GetToken(kInterpolationMode, none)
          : none;
      if (interpolationMode != none &&
          interpolationMode != Record().Token(kTokSurfaceCage)) {
         if (diag) diag->Error("UsdGenCurveSource::Capture: interpolationMode must be none or surfaceCage");
         return false;
      }
      UsdGenCurveSetInputView const found = UsdGenCurveSetInputView::Of(*foundRecord);
      UsdGenCurveSetData transient;
      UsdGenCurveSetInputView transientView;
      UsdGenCurveSetInputView const *source = &found;
      if (interpolationMode == Record().Token(kTokSurfaceCage)) {
         std::string error;
         if (!UsdGenBuildSurfaceCageCurveSet(*ctx.desc, *ctx.params->node,
                                             found, &transient, &error)) {
            if (diag) diag->Error("UsdGenCurveSource::Capture: surfaceCage: " + error);
            return false;
         }
         transientView = UsdGenCurveSetInputView::Of(transient, found.key);
         source = &transientView;
      }
      UsdGenCurveBuffer candidate;
      if (!UsdGenCurveLoader::Load(ctx, *source, &candidate, diag, &bindingCache_)) return false;
      UsdGenCurveSourceCapture &cap = *static_cast<UsdGenCurveSourceCapture *>(out);
      candidate.topologyVersion = cap.MutableBuffer().topologyVersion + 1;
      cap.MutableBuffer() = std::move(candidate);
      return true;
   }
   // No C3 relationship keeps the established surface-backed compatibility
   // path below.  It intentionally cannot stand in for malformed C3 input.
   if (ctx.surface >= UsdGenSurfaceInputs(*ctx.desc).size()) {
      if (diag) diag->Error("UsdGenCurveSource::Capture: no bound usdGen:surface");
      return false;
   }
   UsdGenSurfaceInputView const &surf = UsdGenSurfaceAt(*ctx.desc, ctx.surface);
   if (surf.faceVertexCounts.empty() || surf.restPoints.empty()) {
      if (diag) diag->Warn(std::string("UsdGenCurveSource::Capture: surface '") +
                           ctx.desc->Label(surf.key) + "' has no topology; 0 roots");
      UsdGenCurveSourceCapture &cap = *static_cast<UsdGenCurveSourceCapture *>(out);
      UsdGenCurveBuffer &buf = cap.MutableBuffer();
      buf.topologyVersion += 1;
      buf.totalCurves = 0;
      buf.totalCvs = 0;
      buf.px.clear(); buf.py.clear(); buf.pz.clear();
      buf.rest.clear(); buf.width.clear(); buf.hairT.clear();
      std::vector<UsdGenPlane> extraCv, extraCurve;
      size_t authoredCurves = 0, authoredCvs = 0;
      std::vector<uint32_t> authoredOffsets;
      bool hasSource = false;
      if (!BuildAuthoredPlanes(ctx, &extraCv, &extraCurve, &authoredCurves,
                               &authoredCvs, &authoredOffsets, &hasSource,
                               diag)) return false;
      if (hasSource && (authoredCurves != 0 || authoredCvs != 0)) {
         if (diag) diag->Error("UsdGenCurveSource::Capture: authored planes have no source topology");
         return false;
      }
      buf.extraCv = std::move(extraCv);
      buf.extraCurve = std::move(extraCurve);
      return true;
   }
   int const resampleTo = ctx.params
       ? std::max(ctx.params->GetInt(kResampleTo,
                                    ctx.params->GetInt(kUsdGenResampleTo, 0)), 0)
       : 0;
   int totalCurves = static_cast<int>(surf.faceVertexCounts.size());
   int totalCvs = 0;
   if (resampleTo > 0) {
      size_t const uni = size_t(totalCurves) * size_t(resampleTo);
      if (uni > size_t(std::numeric_limits<int>::max())) {
         if (diag) diag->Error("UsdGenCurveSource::Capture: totalCurves * resampleTo exceeds INT_MAX");
         return false;  // reject; *buf left unwritten
      }
      totalCvs = int(uni);  // uniform: resampleTo CVs per curve
   } else {
      for (int c = 0; c < totalCurves; ++c) totalCvs += surf.faceVertexCounts[c];
   }
   UsdGenCurveSourceCapture &cap = *static_cast<UsdGenCurveSourceCapture *>(out);
   UsdGenCurveBuffer &buf = cap.MutableBuffer();
   buf.topologyVersion += 1;
   buf.totalCurves = totalCurves;
   buf.totalCvs = totalCvs;
   // 04 §2.4 `usdGen:resampleTo`: 0 = keep the source CV counts -> ragged
   // (cvOffsets = {0, cumsum(counts)}, size totalCurves+1, [totalCurves] ==
   // totalCvs; chunks stay cvCount == 0). > 0 = uniform resample at capture;
   // cvOffsets stays empty (uniform fast path).
   if (resampleTo == 0) {
      buf.cvOffsets.assign(totalCurves + 1, 0);
      for (int c = 0; c < totalCurves; ++c)
         buf.cvOffsets[c + 1] = buf.cvOffsets[c] + surf.faceVertexCounts[c];
   } else {
      buf.cvOffsets.clear();
   }
   buf.px.resize(totalCvs);
   buf.py.resize(totalCvs);
   buf.pz.resize(totalCvs);
   buf.rest.resize(totalCvs);
   buf.width.resize(totalCvs);
   buf.hairT.resize(totalCvs);
   std::vector<UsdGenPlane> authoredExtraCv, authoredExtraCurve;
   size_t authoredCurves = 0, authoredCvs = 0;
   std::vector<uint32_t> authoredOffsets;
   bool hasSource = false;
   if (!BuildAuthoredPlanes(ctx, &authoredExtraCv, &authoredExtraCurve,
                            &authoredCurves, &authoredCvs, &authoredOffsets,
                            &hasSource, diag))
      return false;
   if (hasSource) {
      if (authoredCurves != static_cast<size_t>(totalCurves)) {
         if (diag) diag->Error("UsdGenCurveSource::Capture: authored source curve count " +
                               std::to_string(authoredCurves) +
                               " does not match produced curve count " +
                               std::to_string(totalCurves));
         return false;
      }
      UsdGenCurveBuffer authored;
      authored.totalCurves = static_cast<uint32_t>(authoredCurves);
      authored.totalCvs = static_cast<uint32_t>(authoredCvs);
      if (!authoredOffsets.empty()) {
         authored.cvOffsets.resize(authoredOffsets.size());
         for (size_t i = 0; i != authoredOffsets.size(); ++i) {
            if (authoredOffsets[i] > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
               if (diag) diag->Error("UsdGenCurveSource::Capture: authored topology offset exceeds INT_MAX");
               return false;
            }
            authored.cvOffsets[i] = static_cast<int>(authoredOffsets[i]);
         }
      }
      authored.extraCv = std::move(authoredExtraCv);
      authored.extraCurve = std::move(authoredExtraCurve);
      buf.extraCv.clear();
      buf.extraCurve.clear();
      std::string planeError;
      bool const topologyDiff = authored.totalCvs != buf.totalCvs ||
          authored.cvOffsets != buf.cvOffsets;
      if (topologyDiff && (!authored.extraCv.empty() || !authored.extraCurve.empty()) &&
          !UsdGenResampleExtraPlanes(authored, &buf, &planeError)) {
         if (diag) diag->Error("UsdGenCurveSource::Capture: " + planeError);
         return false;
      }
      if (!topologyDiff) {
         buf.extraCv = std::move(authored.extraCv);
         buf.extraCurve = std::move(authored.extraCurve);
      }
   } else {
      // A recapture with no authored source relationship must not retain
      // planes from an older capture or previous generation.
      buf.extraCv.clear();
      buf.extraCurve.clear();
   }
   size_t srcBase = 0, dst = 0;
   for (int c = 0; c < totalCurves; ++c) {
      int const srcCount = surf.faceVertexCounts[c];
      int const n = resampleTo > 0 ? resampleTo : srcCount;
      for (int i = 0; i < n; ++i) {
         UsdGenVec3f p(0.0f);
         if (srcCount > 0) {
            // ragged: source CV i verbatim (t == i, f == 0); uniform: walk
            // the source polyline at an even index step (linear resample).
            double t = (resampleTo > 0 && srcCount > 1 && n > 1)
                ? double(i) * double(srcCount - 1) / double(n - 1)
                : double(resampleTo == 0 ? i : 0);
            int j = static_cast<int>(t);
            if (j > srcCount - 1) j = srcCount - 1;
            int const j2 = std::min(j + 1, srcCount - 1);
            float const f = float(t - double(j));
            UsdGenVec3f const &a = surf.restPoints[srcBase + size_t(j)];
            UsdGenVec3f const &b = surf.restPoints[srcBase + size_t(j2)];
            p = a + f * (b - a);
         }
         buf.px[dst] = p[0];
         buf.py[dst] = p[1];
         buf.pz[dst] = p[2];
         buf.rest[dst] = p;
         if (!buf.width.empty()) buf.width[dst] = ctx.desc->defaultWidth;
         if (!buf.hairT.empty())
            buf.hairT[dst] = n > 1 ? float(i) / float(n - 1) : 0.0f;
         ++dst;
      }
      srcBase += size_t(std::max(srcCount, 0));
   }
   return true;
}

void UsdGenCurveSourceOp::Evaluate(UsdGenEvalContext const &ctx,
                                   UsdGenCapture const &capture,
                                   UsdGenChunkView const &chunk) const
{
   (void)ctx; (void)capture; (void)chunk;
}

UsdGenOpId UsdGenRegisterCurveSourceOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenCurveSource";
    d.displayName = "CurveSource";
    d.create = &Create;
    d.countChanges = UsdGenCountsAtLevel(1);
    d.lane = UsdGenLane::Chunked;
    d.geometryInputArity = 0;
    d.traits = kUsdGenOpGenerator;
    d.params = UsdGenSpan(kParams);
    d.mapSlots = UsdGenSpan(kMaps);
    d.tokens = UsdGenSpan(kTokens);
    return r.RegisterOp(d);
}

}  // namespace usdGen
