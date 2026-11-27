// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenCookGoldens: the published geometry of record (restructure 5.0).
//
// Complements testUsdGenOutputGoldens (frame 0, terminal planes, tile
// points/widths/counts). Every subject is cooked through one UsdGenSession
// in five steps: frames 0, 1 and 12, then a value edit and a topology edit
// at frame 12. Subjects: the in-code graphs of golden/goldenGraphs.cpp (all
// registered ops) and every UsdGenDescription in examples/**/*.usda the
// stage builder accepts on the CPU lane, once per variant selection.
//
// Per step, 128-bit FNV-1a over raw bits, in published order: every tile's
// prim path text, curveVertexCounts, points, widths, hairT, hairId, st,
// displayColor, bakeColor, velocities, extraUniform planes (name,
// interpolation, type, arity, bytes), extent bits and presentation fields;
// the guide tiles, instancers, scalp cap and prim-set signature; and every
// column of the terminal set (compile + scheduler run at the same time).
//
// Must equal tests/golden/cookDigests.<os>-<cc>.txt byte for byte. A
// toolchain without its own file writes cookDigests.<os>-<cc>.actual.txt
// (CI uploads it) and skips (exit 77).
// --write regenerates (an intended output change; never to make a refactor
// pass). USDGEN_COOK_GOLDEN_THREADS overrides the thread count (default 4;
// the digests must not depend on it). USDGEN_GOLDEN_VERBOSE=1 prints
// diagnostics.

#include "utils/pxrInterop.h"
#include "golden/goldenGraphs.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "nodeSets.h"

#include "usdGen/compile/compiler.h"
#include "usdGen/compile/graph.h"
#include "usdGenUsd/lower.h"
#include "usdGenUsd/publishPaths.h"
#include "usdGen/exec/generationStore.h"
#include "usdGen/exec/scheduler.h"
#include "usdGen/exec/session.h"
#include "usdGen/registries/op.h"
#include "usdGenHydraCurves/publication.h"
#include "usdGenCurves/types/curveBuffer.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/primDefinition.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/variantSets.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

#if defined(_WIN32)
#define USDGEN_COOK_OS "windows"
#elif defined(__APPLE__)
#define USDGEN_COOK_OS "macos"
#else
#define USDGEN_COOK_OS "linux"
#endif
#if defined(_MSC_VER)
#define USDGEN_COOK_CC "msvc"
#elif defined(__clang__)
#define USDGEN_COOK_CC "clang"
#elif defined(__GNUC__)
#define USDGEN_COOK_CC "gcc"
#else
#define USDGEN_COOK_CC "unknown"
#endif
constexpr char const *kGoldenName = "cookDigests." USDGEN_COOK_OS "-" USDGEN_COOK_CC ".txt";
constexpr char const *kActualName = "cookDigests." USDGEN_COOK_OS "-" USDGEN_COOK_CC ".actual.txt";

bool g_verbose = false;
int g_threads = 4;
int g_failures = 0;

void Fail(std::string const &what)
{
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
}

/// 128-bit FNV-1a: prime 2^88 + 0x13b, multiplied in 64-bit halves.
struct Fnv128
{
    uint64_t hi = 0x6c62272e07bb0142ull;
    uint64_t lo = 0x62b821756295c58dull;

    void Byte(unsigned char b)
    {
        lo ^= b;
        uint64_t const k = 0x13bull;
        uint64_t const m0 = (lo & 0xffffffffull) * k;
        uint64_t const m1 = (lo >> 32) * k;
        uint64_t const low = m0 + (m1 << 32);
        uint64_t const carry = (m1 >> 32) + (low < m0 ? 1u : 0u);
        hi = hi * k + carry + (lo << 24);
        lo = low;
    }
    void Bytes(void const *data, size_t n)
    {
        auto const *p = static_cast<unsigned char const *>(data);
        for (size_t i = 0; i < n; ++i) Byte(p[i]);
    }
    void U64(uint64_t v) { Bytes(&v, sizeof(v)); }
    void Str(std::string const &s)
    {
        U64(s.size());
        Bytes(s.data(), s.size());
    }
    template <class T>
    void Arr(VtArray<T> const &a)
    {
        U64(a.size());
        if (!a.empty()) Bytes(a.cdata(), a.size() * sizeof(T));
    }
    std::string Hex() const
    {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%016" PRIx64 "%016" PRIx64, hi, lo);
        return buf;
    }
};

template <class T>
std::string ArrayValue(VtArray<T> const &a)
{
    Fnv128 h;
    h.Arr(a);
    return "n=" + std::to_string(a.size()) + " " + h.Hex();
}

/// One subject's ordered (key, value) rows; key = "<step> <field>".
struct Rows
{
    std::vector<std::pair<std::string, std::string>> rows;
    void Add(std::string key, std::string value) { rows.emplace_back(std::move(key), std::move(value)); }
};

// ---- publication digests -----------------------------------------------------

void HashPlanes(Fnv128 *h, std::vector<UsdGenPlane> const &planes)
{
    h->U64(planes.size());
    for (UsdGenPlane const &p : planes) {
        h->Str(p.name.GetString());
        h->Str(p.interpolation.GetString());
        h->Str(p.type.GetString());
        h->U64(p.arity);
        h->Arr(p.f);
        h->Arr(p.i);
    }
}

void HashMatrix(Fnv128 *h, GfMatrix4d const &m) { h->Bytes(m.data(), 16 * sizeof(double)); }

void HashExtent(Fnv128 *h, GfVec3d const &lo, GfVec3d const &hi)
{
    h->Bytes(lo.data(), 3 * sizeof(double));
    h->Bytes(hi.data(), 3 * sizeof(double));
}

/// Field-major digests of a tile list: each field hashes every tile in
/// published order (tile id and index first), so a row names the field.
void AddTiles(Rows *out, std::string const &step, char const *what,
              UsdGenGeneration const &gen, std::vector<UsdGenTilePublication> const &tiles)
{
    // Paths as the host names them (the engine publishes keys).
    auto path = [&gen](UsdGenPrimRef const &prim) { return usdGenUsd::PrimPath(gen, prim).GetString(); };
    size_t curves = 0, cvs = 0;
    for (UsdGenTilePublication const &t : tiles) {
        curves += t.curveVertexCounts.size();
        cvs += t.points.size();
    }
    std::string const p = step + " " + what;
    out->Add(p, "n=" + std::to_string(tiles.size()) + " curves=" + std::to_string(curves) +
                    " cvs=" + std::to_string(cvs));
    if (tiles.empty()) return;
    using Field = std::function<void(Fnv128 *, UsdGenTilePublication const &)>;
    std::pair<char const *, Field> const fields[] = {
        {"path", [&](Fnv128 *h, UsdGenTilePublication const &t) { h->Str(path(t.prim)); }},
        {"curveVertexCounts", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.curveVertexCounts); }},
        {"points", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.points); }},
        {"widths", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.widths); }},
        {"hairT", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.hairT); }},
        {"hairId", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.hairId); }},
        {"st", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.st); }},
        {"displayColor", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.displayColor); }},
        {"bakeColor", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.bakeColor); }},
        {"velocities", [](Fnv128 *h, UsdGenTilePublication const &t) { h->Arr(t.velocities); }},
        {"extraUniform", [](Fnv128 *h, UsdGenTilePublication const &t) { HashPlanes(h, t.extraUniform); }},
        {"extent", [](Fnv128 *h, UsdGenTilePublication const &t) { HashExtent(h, t.extentMin, t.extentMax); }},
        {"presentation", [&](Fnv128 *h, UsdGenTilePublication const &t) {
             h->Str(t.basis);
             h->U64(uint64_t(int64_t(t.refineLevel)));
             HashMatrix(h, t.xformMatrix);
             h->Str(t.purpose.GetString());
             h->Str(t.visibility.GetString());
             h->Str(path(t.material));
             h->Str(t.materialPurpose.GetString());
             h->Str(path(t.origin));
             h->Str(path(t.dependency));
         }},
    };
    for (auto const &f : fields) {
        Fnv128 h;
        for (size_t i = 0; i < tiles.size(); ++i) {
            h.U64(i);
            h.U64(tiles[i].tile);
            f.second(&h, tiles[i]);
        }
        out->Add(p + "." + f.first, h.Hex());
    }
    // Extra plane names, so a diff shows which plane set moved.
    std::string names;
    for (UsdGenPlane const &pl : tiles.front().extraUniform)
        names += (names.empty() ? "" : ",") + pl.name.GetString() + "/" +
            pl.interpolation.GetString() + "/" + pl.type.GetString() + "/" +
            std::to_string(int(pl.arity));
    out->Add(p + ".extraNames", names.empty() ? "-" : names);
}

void AddScalp(Rows *out, std::string const &step, UsdGenGeneration const &gen,
              UsdGenScalpShadowPublication const &s)
{
    if (s.IsEmpty()) {
        out->Add(step + " scalp", "empty");
        return;
    }
    Fnv128 geo, planes, meta;
    geo.Arr(s.points);
    geo.Arr(s.normals);
    geo.Arr(s.faceVertexCounts);
    geo.Arr(s.faceVertexIndices);
    geo.Arr(s.latticeIds);
    HashPlanes(&planes, s.extraUniform);
    meta.Str(usdGenUsd::PrimPath(gen, s.prim).GetString());
    HashExtent(&meta, s.extentMin, s.extentMax);
    meta.Str(s.purpose.GetString());
    meta.Str(s.visibility.GetString());
    meta.Str(usdGenUsd::PrimPath(gen, s.material).GetString());
    meta.U64(s.digest);
    meta.U64(s.latticeKey);
    meta.Bytes(&s.lift, sizeof(s.lift));
    out->Add(step + " scalp", "points=" + std::to_string(s.points.size()) + " faces=" +
                                  std::to_string(s.faceVertexCounts.size()));
    out->Add(step + " scalp.geometry", geo.Hex());
    out->Add(step + " scalp.planes", planes.Hex());
    out->Add(step + " scalp.meta", meta.Hex());
}

void AddGeneration(Rows *out, std::string const &step, UsdGenGeneration const &gen)
{
    UsdGenCurvePublication const &pub = UsdGenCurvePublicationOf(gen);
    AddTiles(out, step, "tiles", gen, pub.tiles);
    AddTiles(out, step, "guides", gen, pub.guides);
    AddScalp(out, step, gen, pub.scalpShadow);
    Fnv128 inst;
    inst.U64(pub.instancers.size());
    for (UsdGenInstancerPublication const &i : pub.instancers) {
        inst.Str(usdGenUsd::PrimPath(gen, i.prim).GetString());
        inst.U64(i.prototypes.size());
        for (UsdGenPrimRef const &p : i.prototypes) inst.Str(usdGenUsd::PrimPath(gen, p).GetString());
        inst.U64(i.instanceIndices.size());
        for (VtIntArray const &a : i.instanceIndices) inst.Arr(a);
        inst.U64(i.mask.size());
        for (bool b : i.mask) inst.Byte(b ? 1 : 0);
    }
    out->Add(step + " instancers", "n=" + std::to_string(pub.instancers.size()) + " " + inst.Hex());
    Fnv128 sig;
    sig.U64(uint64_t(int64_t(pub.signature.tileCount)));
    sig.U64(uint64_t(int64_t(pub.signature.instancerCount)));
    for (UsdGenPrimRef const &s : pub.signature.prims) sig.Str(usdGenUsd::PrimPath(gen, s).GetString());
    for (auto const &names : pub.signature.primvarNames) {
        sig.U64(names.size());
        for (std::string const &s : names) sig.Str(s);
    }
    for (std::string const &s : pub.signature.primTypes) sig.Str(s);
    out->Add(step + " signature", sig.Hex());
    // colorDigest is left out: it mixes TfToken::Hash (pointer-derived).
}

/// Every column of the terminal set, extra planes in their own order.
void AddTerminal(Rows *out, std::string const &step, UsdGenCurveBuffer const &b)
{
    std::string const p = step + " terminal.";
    out->Add(p + "counts", "curves=" + std::to_string(b.totalCurves) + " cvs=" +
                               std::to_string(b.totalCvs));
    out->Add(p + "px", ArrayValue(b.px));
    out->Add(p + "py", ArrayValue(b.py));
    out->Add(p + "pz", ArrayValue(b.pz));
    out->Add(p + "width", ArrayValue(b.width));
    out->Add(p + "hairT", ArrayValue(b.hairT));
    out->Add(p + "rest", ArrayValue(b.rest));
    out->Add(p + "curveId", ArrayValue(b.curveId));
    out->Add(p + "cvOffsets", ArrayValue(b.cvOffsets));
    out->Add(p + "rootPrim", ArrayValue(b.rootPrim));
    out->Add(p + "rootUV", ArrayValue(b.rootUV));
    out->Add(p + "rootT", ArrayValue(b.rootT));
    out->Add(p + "rootN", ArrayValue(b.rootN));
    out->Add(p + "rootB", ArrayValue(b.rootB));
    Fnv128 cv, curve;
    HashPlanes(&cv, b.extraCv);
    HashPlanes(&curve, b.extraCurve);
    out->Add(p + "extraCv", "n=" + std::to_string(b.extraCv.size()) + " " + cv.Hex());
    out->Add(p + "extraCurve", "n=" + std::to_string(b.extraCurve.size()) + " " + curve.Hex());
}

void PrintDiagnostics(std::string const &label, std::vector<std::string> const &errors)
{
    if (!g_verbose) return;
    for (std::string const &e : errors) std::printf("  [%s] %s\n", label.c_str(), e.c_str());
}

// ---- edits ---------------------------------------------------------------------

UsdPrimDefinition const *Definition(TfToken const &type)
{
    return UsdSchemaRegistry::GetInstance().FindConcretePrimDefinition(type);
}

std::string Num(double v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

float Nudge(float v) { return v == 0.0f ? 0.25f : (std::fabs(v) <= 1.0f ? v * 0.5f : v * 1.5f); }

UsdGenParamValue *FindParam(UsdGenNodeDesc *node, TfToken const &name)
{
    for (UsdGenParamValue &p : node->params)
        if (p.name == name) return &p;
    return nullptr;
}

/// The node's current value of a param: authored, else the schema fallback.
VtValue Current(UsdGenNodeDesc *node, TfToken const &name)
{
    if (UsdGenParamValue const *p = FindParam(node, name)) return p->value;
    VtValue fallback;
    if (UsdPrimDefinition const *def = Definition(node->type))
        if (UsdPrimDefinition::Attribute const attr =
                def->GetAttributeDefinition(TfToken("usdGen:" + name.GetString())))
            attr.GetFallbackValue(&fallback);
    return fallback;
}

void SetParam(UsdGenNodeDesc *node, TfToken const &name, VtValue const &value)
{
    if (UsdGenParamValue *p = FindParam(node, name)) p->value = value;
    else node->params.push_back({name, value, false});
}

std::string Leaf(SdfPath const &p) { return p.GetName(); }

/// Value edit: the terminal op's first float (else int) schema attribute,
/// nudged. Returns the label, empty when the terminal has none.
std::string ValueEdit(UsdGenGraphDesc *desc)
{
    for (UsdGenNodeDesc &n : desc->nodes) {
        if (n.path != desc->terminal) continue;
        UsdPrimDefinition const *def = Definition(n.type);
        if (!def) {
            // No schema class (UsdGenWidthBlend): the first authored float.
            for (UsdGenParamValue &pv : n.params)
                if (pv.value.IsHolding<float>()) {
                    float const v = Nudge(pv.value.UncheckedGet<float>());
                    pv.value = VtValue(v);
                    return Leaf(n.path) + "." + pv.name.GetString() + "=" + Num(v);
                }
            return {};
        }
        std::vector<TfToken> names = def->GetPropertyNames();
        std::sort(names.begin(), names.end(),
                  [](TfToken const &a, TfToken const &b) { return a.GetString() < b.GetString(); });
        for (int pass = 0; pass < 2; ++pass)
            for (TfToken const &prop : names) {
                std::string const full = prop.GetString();
                if (full.compare(0, 7, "usdGen:") != 0) continue;
                UsdPrimDefinition::Attribute const attr = def->GetAttributeDefinition(prop);
                if (!attr) continue;
                VtValue fallback;
                attr.GetFallbackValue(&fallback);
                TfToken const name(full.substr(7));
                VtValue const cur = Current(&n, name);
                if (pass == 0 && fallback.IsHolding<float>() && cur.IsHolding<float>()) {
                    float const v = Nudge(cur.UncheckedGet<float>());
                    SetParam(&n, name, VtValue(v));
                    return Leaf(n.path) + "." + name.GetString() + "=" + Num(v);
                }
                if (pass == 1 && fallback.IsHolding<int>() && cur.IsHolding<int>()) {
                    int const v = cur.UncheckedGet<int>() + 1;
                    SetParam(&n, name, VtValue(v));
                    return Leaf(n.path) + "." + name.GetString() + "=" + std::to_string(v);
                }
            }
        return {};
    }
    return {};
}

/// Drop the last curve of a curve set: every per-curve and per-CV array.
bool DropLastCurve(UsdGenCurveSetDesc *set)
{
    size_t const curves = set->curveVertexCounts.size();
    if (curves < 2 || set->surfaceCage) return false;
    size_t const cvs = set->points.size();
    size_t const lastCvs = size_t(set->curveVertexCounts[curves - 1]);
    if (lastCvs > cvs) return false;
    auto trim = [&](auto &a) {
        if (a.size() == curves) a.resize(curves - 1);
        else if (a.size() == cvs) a.resize(cvs - lastCvs);
    };
    trim(set->points);
    trim(set->rest);
    trim(set->widths);
    trim(set->skinPrim);
    trim(set->curveId);
    trim(set->skinPrimUv);
    trim(set->rootFrame);
    for (UsdGenAuthoredPlaneDesc &p : set->authoredPlanes) {
        bool const point = p.domain == UsdGenAuthoredPlaneDomain::Point;
        if (!point && p.domain != UsdGenAuthoredPlaneDomain::Primitive) continue;
        size_t const elems = point ? cvs : curves;
        size_t const keep = point ? cvs - lastCvs : curves - 1;
        if (p.floatValues.size() == elems * p.arity) p.floatValues.resize(keep * p.arity);
        if (p.intValues.size() == elems * p.arity) p.intValues.resize(keep * p.arity);
    }
    set->curveVertexCounts.resize(curves - 1);
    ++set->curveGeneration;
    return true;
}

/// Topology edit: Scatter density x1.5, else the first curve set an op reads
/// loses its last curve, else Grow segments +2, else GuideInterpolate
/// cvCount +2. Returns the label, empty when none applies.
std::string TopologyEdit(UsdGenGraphDesc *desc)
{
    TfToken const scatter("UsdGenScatter"), grow("UsdGenGrow"),
        interpolate("UsdGenGuideInterpolate");
    for (UsdGenNodeDesc &n : desc->nodes)
        if (n.type == scatter && n.enabled) {
            VtValue const cur = Current(&n, TfToken("density"));
            if (!cur.IsHolding<float>()) continue;
            float const v = cur.UncheckedGet<float>() * 1.5f;
            SetParam(&n, TfToken("density"), VtValue(v));
            return Leaf(n.path) + ".density=" + Num(v);
        }
    for (UsdGenNodeDesc const &n : desc->nodes) {
        if (!n.enabled) continue;
        for (SdfPathVector const *paths : {&n.curves, &n.references})
            for (SdfPath const &c : *paths)
                for (UsdGenCurveSetDesc &set : desc->curveSets)
                    if (set.path == c && DropLastCurve(&set))
                        return Leaf(set.path) + ".dropLastCurve";
    }
    for (TfToken const &type : {grow, interpolate})
        for (UsdGenNodeDesc &n : desc->nodes) {
            if (n.type != type || !n.enabled) continue;
            TfToken const name(type == grow ? "segments" : "cvCount");
            VtValue const cur = Current(&n, name);
            if (!cur.IsHolding<int>()) continue;
            int const v = cur.UncheckedGet<int>() + 2;
            SetParam(&n, name, VtValue(v));
            return Leaf(n.path) + "." + name.GetString() + "=" + std::to_string(v);
        }
    return {};
}

// ---- cooking -------------------------------------------------------------------

/// Compile + one scheduler run: the terminal set's columns.
void CookTerminal(UsdGenGraphDesc const &desc, std::string const &step, std::string const &label,
                  Rows *rows)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) {
        PrintDiagnostics(label + " compile", compiled.errors);
        rows->Add(step + " terminal", "compile-failed");
        return;
    }
    UsdGenScheduler scheduler(g_threads);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Spec();
    ctx.time = desc.time;
    UsdGenRunResult const run = scheduler.Run(graph, ctx, 1);
    if (run.diagnostics.HasErrors()) {
        PrintDiagnostics(label + " run", run.diagnostics.errors);
        rows->Add(step + " terminal", "run-failed");
        return;
    }
    rows->Add(step + " terminal", "ok");
    AddTerminal(rows, step, OutputBuffer(graph));
}

/// One step of a subject's session: set the descriptor, commit, digest.
void Step(UsdGenSession *session, UsdGenGraphDesc const &desc, std::string const &step,
          UsdGenCommitReason reason, std::string const &label, Rows *rows)
{
    usdGenUsd::SetGraphDesc(*session, desc);
    UsdGenSession::SnapshotPtr const snap = session->CommitSnapshot(desc.time, reason);
    if (snap) PrintDiagnostics(label + " " + step, snap->diagnostics.errors);
    bool const failed = !snap || snap->diagnostics.HasErrors();
    if (!snap || !snap->generation) rows->Add(step + " status", "none");
    else {
        rows->Add(step + " status", failed ? "failed" : "ok");
        // A failed attempt retains the last good generation: digest only a
        // fresh one.
        if (!failed) AddGeneration(rows, step, *snap->generation);
    }
    CookTerminal(desc, step, label, rows);
}

using DescAt = std::function<UsdGenGraphDesc(double)>;

/// The five steps over one session; the callback builds the descriptor at a frame.
Rows CookSubject(std::string const &label, DescAt const &at, int *topologyEdits,
                 std::vector<std::string> *unchangedTopology)
{
    Rows rows;
    UsdGenSession session(g_threads);
    double const frames[] = {0.0, 1.0, 12.0};
    for (double f : frames) {
        UsdGenGraphDesc desc = at(f);
        desc.time = f;
        desc.executionBackend = kUsdGenBackendCpu;
        Step(&session, desc, "f" + Num(f), f == 0.0 ? UsdGenCommitReason::LiveOverride
                                                     : UsdGenCommitReason::SetTime,
             label, &rows);
    }
    UsdGenGraphDesc desc = at(12.0);
    desc.time = 12.0;
    desc.executionBackend = kUsdGenBackendCpu;
    std::string const value = ValueEdit(&desc);
    rows.Add("value edit", value.empty() ? "n/a" : value);
    if (!value.empty())
        Step(&session, desc, "value", UsdGenCommitReason::LiveOverride, label, &rows);
    std::string const topo = TopologyEdit(&desc);
    rows.Add("topology edit", topo.empty() ? "n/a" : topo);
    if (!topo.empty()) {
        std::string before;
        for (auto const &r : rows.rows)
            if (r.first == "value terminal.counts" || r.first == "f12 terminal.counts")
                before = r.second;  // the last one written wins: value, else f12
        Step(&session, desc, "topology", UsdGenCommitReason::LiveOverride, label, &rows);
        ++*topologyEdits;
        for (auto const &r : rows.rows)
            if (r.first == "topology terminal.counts" && r.second == before)
                unchangedTopology->push_back(label + " (" + topo + ")");
    }
    session.Drain();
    return rows;
}

// ---- output table ----------------------------------------------------------------

struct Table
{
    std::vector<std::string> order;
    std::map<std::string, Rows> subjects;
    void Put(std::string const &s, Rows rows)
    {
        if (!subjects.count(s)) order.push_back(s);
        subjects[s] = std::move(rows);
    }
    std::string Text() const
    {
        std::ostringstream os;
        os << "# testUsdGenCookGoldens: published geometry of record (FNV-1a-128 per field).\n"
              "# '== <subject>', then '<step> <field> <value>' rows. Never edit by hand;\n"
              "# --write regenerates.\n"
              "# toolchain " USDGEN_COOK_OS "-" USDGEN_COOK_CC "\n";
        for (std::string const &s : order) {
            os << "== " << s << '\n';
            for (auto const &r : subjects.at(s).rows) os << r.first << ' ' << r.second << '\n';
        }
        return os.str();
    }
};

struct Counters
{
    int topologyEdits = 0;
    std::vector<std::string> unchangedTopology;
};

void RunGraphs(Table *table, Counters *c)
{
    for (usdGenGolden::GoldenGraph const &g : usdGenGolden::MakeGoldenGraphs()) {
        UsdGenGraphDesc const base = g.desc;
        table->Put("graph/" + g.name,
                   CookSubject(g.name, [&](double) { return base; }, &c->topologyEdits,
                               &c->unchangedTopology));
    }
}

void CookExampleStage(UsdStageRefPtr const &stage, std::string const &name, Table *table,
                      Counters *c)
{
    std::vector<SdfPath> descriptions;
    for (UsdPrim const &prim : stage->Traverse())
        if (prim.GetTypeName() == TfToken("UsdGenDescription")) descriptions.push_back(prim.GetPath());
    for (SdfPath const &path : descriptions) {
        std::string const subject = name + ":" + path.GetString();
        UsdGenGraphDesc const probe = usdGenImaging::BuildGraphDescFromStage(stage, path);
        if (!probe.validationErrors.empty()) {
            PrintDiagnostics(subject + " builder", probe.validationErrors);
            Rows rows;
            rows.Add("all status", "rejected");
            table->Put(subject, std::move(rows));
            continue;
        }
        auto at = [&](double f) {
            usdGenImaging::UsdGenGraphDescBuildOptions options;
            options.time = f;
            return usdGenImaging::BuildGraphDescFromStage(stage, path, options);
        };
        table->Put(subject, CookSubject(subject, at, &c->topologyEdits, &c->unchangedTopology));
    }
}

void RunExamples(Table *table, Counters *c)
{
    namespace fs = std::filesystem;
    fs::path const root = fs::path(USDGEN_TEST_SOURCE_DIR) / "examples";
    std::vector<std::string> files;
    for (auto const &entry : fs::recursive_directory_iterator(root))
        if (entry.is_regular_file() && entry.path().extension() == ".usda")
            files.push_back(fs::relative(entry.path(), root).generic_string());
    std::sort(files.begin(), files.end());
    if (files.empty()) Fail("no examples found under " + root.generic_string());
    auto folded = [](fs::path const &p) {
        std::string s = fs::weakly_canonical(p).generic_string();
        for (char &ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    };
    std::string const rootText = folded(root);
    for (std::string const &rel : files) {
        UsdStageRefPtr const stage = UsdStage::Open((root / rel).string(), UsdStage::LoadAll);
        std::string const name = "example/" + rel;
        if (!stage) {
            Rows rows;
            rows.Add("all status", "unopenable");
            table->Put(name, std::move(rows));
            continue;
        }
        // Self-contained examples only (same rule as testUsdGenOutputGoldens).
        bool selfContained = stage->GetCompositionErrors().empty();
        for (SdfLayerHandle const &layer : stage->GetUsedLayers()) {
            std::string const real = layer->GetRealPath();
            if (!real.empty() && folded(fs::path(real)).rfind(rootText, 0) != 0)
                selfContained = false;
        }
        if (!selfContained) continue;
        CookExampleStage(stage, name, table, c);
        struct Choice { SdfPath prim; std::string set, current; std::vector<std::string> names; };
        std::vector<Choice> choices;
        for (UsdPrim const &prim : stage->Traverse()) {
            UsdVariantSets sets = prim.GetVariantSets();
            std::vector<std::string> setNames = sets.GetNames();
            std::sort(setNames.begin(), setNames.end());
            for (std::string const &s : setNames) {
                UsdVariantSet vs = sets.GetVariantSet(s);
                std::vector<std::string> names = vs.GetVariantNames();
                std::sort(names.begin(), names.end());
                choices.push_back({prim.GetPath(), s, vs.GetVariantSelection(), names});
            }
        }
        stage->SetEditTarget(stage->GetSessionLayer());
        for (Choice const &ch : choices)
            for (std::string const &v : ch.names) {
                if (v == ch.current) continue;
                UsdVariantSet vs = stage->GetPrimAtPath(ch.prim).GetVariantSet(ch.set);
                vs.SetVariantSelection(v);
                CookExampleStage(stage, name + "{" + ch.prim.GetString() + ":" + ch.set + "=" + v + "}",
                                 table, c);
                vs.ClearVariantSelection();
            }
    }
}

// ---- compare ---------------------------------------------------------------------

std::string ReadFile(std::string const &path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream os;
    os << in.rdbuf();
    std::string text = os.str();
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
}

/// Prints the first differing lines and a count.
void Compare(std::string const &expected, std::string const &actual)
{
    std::istringstream a(expected), b(actual);
    std::string la, lb, subject;
    size_t line = 0, diffs = 0;
    while (true) {
        bool const moreA = bool(std::getline(a, la));
        bool const moreB = bool(std::getline(b, lb));
        if (!moreA && !moreB) break;
        ++line;
        if (moreA && la.rfind("== ", 0) == 0) subject = la.substr(3);
        if (moreA && moreB && la == lb) continue;
        if (++diffs <= 24)
            std::printf("  line %zu (%s)\n    expected %s\n    actual   %s\n", line, subject.c_str(),
                        moreA ? la.c_str() : "<eof>", moreB ? lb.c_str() : "<eof>");
    }
    std::printf("%zu lines differ\n", diffs);
}

}  // namespace

int main(int argc, char **argv)
{
    usdGenTest::EnsureHydraPublishers();
    g_verbose = std::getenv("USDGEN_GOLDEN_VERBOSE") != nullptr;
    if (char const *t = std::getenv("USDGEN_COOK_GOLDEN_THREADS")) g_threads = std::max(1, std::atoi(t));
    bool rewrite = false;
    for (int i = 1; i < argc; ++i) rewrite = rewrite || std::strcmp(argv[i], "--write") == 0;
    if (!Definition(TfToken("UsdGenLength"))) {
        std::printf("FAIL: the usdGen schema is not loaded (PXR_PLUGINPATH_NAME)\n");
        return 1;
    }
    std::string const goldenPath = std::string(USDGEN_GOLDEN_DIR) + "/" + kGoldenName;
    std::string const expected = ReadFile(goldenPath);

    Table table;
    Counters counters;
    RunGraphs(&table, &counters);
    RunExamples(&table, &counters);
    std::string const text = table.Text();
    std::printf("%zu subjects, %d topology edits, %zu bytes, %d threads\n", table.order.size(),
                counters.topologyEdits, text.size(), g_threads);
    for (std::string const &s : counters.unchangedTopology)
        Fail("topology edit left the terminal counts unchanged: " + s);
    if (counters.topologyEdits == 0) Fail("no subject took a topology edit");

    if (rewrite) {
        std::ofstream(goldenPath, std::ios::binary) << text;
        std::printf("wrote %s\n", goldenPath.c_str());
        return g_failures ? 1 : 0;
    }
    if (expected.empty()) {
        // Float bits belong to one toolchain: record this one's for review.
        std::ofstream(kActualName, std::ios::binary) << text;
        std::printf("SKIP: no tests/golden/%s for this toolchain (wrote %s)\n", kGoldenName, kActualName);
        return g_failures ? 1 : 77;
    }
    if (expected != text) {
        std::ofstream(kActualName, std::ios::binary) << text;
        Compare(expected, text);
        Fail(std::string("output differs from tests/golden/") + kGoldenName + " (actual: " +
             kActualName + ")");
    }
    if (g_failures) std::printf("testUsdGenCookGoldens: FAILED (%d)\n", g_failures);
    else std::printf("testUsdGenCookGoldens: PASS\n");
    return g_failures ? 1 : 0;
}
