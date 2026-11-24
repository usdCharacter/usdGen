// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenOutputGoldens: the CPU lane's output of record.
//
// Cooks (1) the in-code graphs of golden/goldenGraphs.cpp, which together
// cover every registered op, (2) one probe per op per schema attribute with
// a non-fallback value (each enum token, a nudged number, a flipped bool), and
// (3) every UsdGenDescription in examples/**/*.usda the stage builder
// accepts, once per variant selection. Each cook is hashed per plane
// (FNV-1a-64 of the raw bytes): the terminal buffer's px py pz width hairT
// rest curveId cvOffsets and root planes, every extra plane sorted by name,
// and the session's tile publications (points, widths, vertex counts).
// The result must equal tests/golden/cpuOutput.txt byte for byte.
//
// Fixed thread count, no pointer-derived hashes, no timestamps, no absolute
// paths in the output. USDGEN_GOLDEN_WRITE=1 rewrites the golden file (a
// design change; never to make a refactor pass). USDGEN_GOLDEN_VERBOSE=1
// prints each cook's diagnostics.

#include "golden/goldenGraphs.h"

#include "usdGen/compiler.h"
#include "usdGen/generationStore.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/session.h"
#include "usdGenImaging/usdGenGraphDescBuilderStage.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/schema.h"
#include "pxr/usd/usd/primDefinition.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/variantSets.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

constexpr int kThreads = 4;
// Float bits depend on the compiler and its libm: the golden belongs to one
// toolchain, and another skips (exit 77) rather than compare.
#if defined(_MSC_VER)
constexpr char const *kToolchain = "msvc";
#elif defined(__clang__)
constexpr char const *kToolchain = "clang";
#elif defined(__GNUC__)
constexpr char const *kToolchain = "gcc";
#else
constexpr char const *kToolchain = "unknown";
#endif
constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

bool g_verbose = false;
int g_failures = 0;

void Fail(std::string const &what)
{
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
}

uint64_t Fnv(void const *data, size_t bytes, uint64_t h = kFnvOffset)
{
    auto const *p = static_cast<unsigned char const *>(data);
    for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * kFnvPrime;
    return h;
}

std::string Hex(uint64_t h)
{
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016" PRIx64, h);
    return buf;
}

template <class T>
std::string ArrayValue(VtArray<T> const &a)
{
    return "n=" + std::to_string(a.size()) + " " +
        Hex(a.empty() ? kFnvOffset : Fnv(a.cdata(), a.size() * sizeof(T)));
}

/// One cook's output: ordered (plane, value) rows.
struct Rows
{
    std::vector<std::pair<std::string, std::string>> rows;
    void Add(std::string plane, std::string value) { rows.emplace_back(std::move(plane), std::move(value)); }
    uint64_t Combined() const
    {
        uint64_t h = kFnvOffset;
        for (auto const &r : rows) {
            h = Fnv(r.first.data(), r.first.size(), h);
            h = Fnv(r.second.data(), r.second.size(), h);
        }
        return h;
    }
};

void AddPlanes(Rows *out, char const *domain, std::vector<UsdGenPlane> const &planes)
{
    std::vector<UsdGenPlane const *> sorted;
    for (UsdGenPlane const &p : planes) sorted.push_back(&p);
    std::sort(sorted.begin(), sorted.end(), [](UsdGenPlane const *a, UsdGenPlane const *b) {
        return a->name.GetString() < b->name.GetString();
    });
    for (UsdGenPlane const *p : sorted) {
        std::string const shape = p->type.GetString() + "/" + p->interpolation.GetString() +
            "/" + std::to_string(int(p->arity)) + " ";
        out->Add(std::string(domain) + ":" + p->name.GetString(),
                 shape + (p->type == TfToken("int") ? ArrayValue(p->i) : ArrayValue(p->f)));
    }
}

void AddBuffer(Rows *out, UsdGenCurveBuffer const &b)
{
    out->Add("counts", "curves=" + std::to_string(b.totalCurves) +
                           " cvs=" + std::to_string(b.totalCvs));
    out->Add("px", ArrayValue(b.px));
    out->Add("py", ArrayValue(b.py));
    out->Add("pz", ArrayValue(b.pz));
    out->Add("width", ArrayValue(b.width));
    out->Add("hairT", ArrayValue(b.hairT));
    out->Add("rest", ArrayValue(b.rest));
    out->Add("curveId", ArrayValue(b.curveId));
    out->Add("cvOffsets", ArrayValue(b.cvOffsets));
    out->Add("rootPrim", ArrayValue(b.rootPrim));
    out->Add("rootUV", ArrayValue(b.rootUV));
    out->Add("rootT", ArrayValue(b.rootT));
    out->Add("rootN", ArrayValue(b.rootN));
    out->Add("rootB", ArrayValue(b.rootB));
    AddPlanes(out, "cv", b.extraCv);
    AddPlanes(out, "curve", b.extraCurve);
}

void PrintDiagnostics(std::string const &label, std::vector<std::string> const &errors)
{
    if (!g_verbose) return;
    for (std::string const &e : errors) std::printf("  [%s] %s\n", label.c_str(), e.c_str());
}

/// Compile + one scheduler run. Status "ok", "compile-failed" or "run-failed".
std::string CookScheduler(UsdGenGraphDesc const &desc, std::string const &label, Rows *rows)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        PrintDiagnostics(label + " compile", compiled.errors);
        return "compile-failed";
    }
    UsdGenScheduler scheduler(kThreads);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    ctx.time = desc.time;
    UsdGenRunResult const run = scheduler.Run(graph, ctx, 1);
    if (run.diagnostics.HasErrors()) {
        PrintDiagnostics(label + " run", run.diagnostics.errors);
        return "run-failed";
    }
    AddBuffer(rows, graph.Output());
    return "ok";
}

/// One session commit; the tile publications' points, widths and counts.
void CookSession(UsdGenGraphDesc const &desc, std::string const &label, Rows *rows)
{
    UsdGenSession session(kThreads);
    session.SetGraphDesc(desc);
    UsdGenGenerationConstPtr const gen = session.Commit(desc.time, UsdGenCommitReason::LiveOverride);
    PrintDiagnostics(label + " session", session.LastDiagnostics().errors);
    if (!gen) {
        rows->Add("tiles", "none");
        return;
    }
    std::vector<UsdGenTilePublication const *> tiles;
    for (UsdGenTilePublication const &t : gen->tiles) tiles.push_back(&t);
    std::sort(tiles.begin(), tiles.end(), [](auto const *a, auto const *b) { return a->tile < b->tile; });
    uint64_t counts = kFnvOffset, points = kFnvOffset, widths = kFnvOffset;
    size_t curves = 0, cvs = 0;
    for (UsdGenTilePublication const *t : tiles) {
        uint32_t const id = t->tile;
        counts = Fnv(&id, sizeof(id), counts);
        counts = Fnv(t->curveVertexCounts.cdata(), t->curveVertexCounts.size() * sizeof(int), counts);
        uint64_t const np = t->points.size(), nw = t->widths.size();
        points = Fnv(&np, sizeof(np), points);
        points = Fnv(t->points.cdata(), np * sizeof(GfVec3f), points);
        widths = Fnv(&nw, sizeof(nw), widths);
        widths = Fnv(t->widths.cdata(), nw * sizeof(float), widths);
        curves += t->curveVertexCounts.size();
        cvs += np;
    }
    rows->Add("tiles", "n=" + std::to_string(tiles.size()) + " curves=" +
                           std::to_string(curves) + " cvs=" + std::to_string(cvs));
    rows->Add("tiles.counts", Hex(counts));
    rows->Add("tiles.points", Hex(points));
    rows->Add("tiles.widths", Hex(widths));
}

// ---- output table --------------------------------------------------------------

struct Table
{
    std::vector<std::string> order;                 // graph names, first-seen order
    std::map<std::string, Rows> graphs;
    void Put(std::string const &graph, Rows rows)
    {
        if (!graphs.count(graph)) order.push_back(graph);
        graphs[graph] = std::move(rows);
    }
    std::string Text() const
    {
        std::ostringstream os;
        os << "# testUsdGenOutputGoldens: CPU-lane output of record (FNV-1a-64 per plane).\n"
              "# <graph> <plane> <value>. Never edit by hand; USDGEN_GOLDEN_WRITE=1 rewrites.\n"
              "# toolchain " << kToolchain << "\n";
        for (std::string const &g : order)
            for (auto const &r : graphs.at(g).rows) os << g << ' ' << r.first << ' ' << r.second << '\n';
        return os.str();
    }
};

// ---- schema probes -------------------------------------------------------------

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

/// Non-fallback values for one attribute: every other allowed token, or one
/// nudged number / flipped bool. Empty when the type is not probed.
std::vector<std::pair<std::string, VtValue>> ProbeValues(UsdPrimDefinition::Attribute const &attr)
{
    std::vector<std::pair<std::string, VtValue>> out;
    VtValue fallback;
    attr.GetFallbackValue(&fallback);
    if (fallback.IsHolding<TfToken>()) {
        VtTokenArray allowed;
        attr.GetMetadata(SdfFieldKeys->AllowedTokens, &allowed);
        for (TfToken const &t : allowed)
            if (t != fallback.UncheckedGet<TfToken>()) out.push_back({t.GetString(), VtValue(t)});
    } else if (fallback.IsHolding<float>()) {
        float const v = Nudge(fallback.UncheckedGet<float>());
        out.push_back({Num(v), VtValue(v)});
    } else if (fallback.IsHolding<double>()) {
        double const v = Nudge(float(fallback.UncheckedGet<double>()));
        out.push_back({Num(v), VtValue(v)});
    } else if (fallback.IsHolding<int>()) {
        int const v = fallback.UncheckedGet<int>() + 1;
        out.push_back({std::to_string(v), VtValue(v)});
    } else if (fallback.IsHolding<bool>()) {
        bool const v = !fallback.UncheckedGet<bool>();
        out.push_back({v ? "true" : "false", VtValue(v)});
    } else if (fallback.IsHolding<GfVec2f>()) {
        GfVec2f v = fallback.UncheckedGet<GfVec2f>();
        for (int i = 0; i < 2; ++i) v[i] = Nudge(v[i]);
        out.push_back({"(" + Num(v[0]) + "," + Num(v[1]) + ")", VtValue(v)});
    } else if (fallback.IsHolding<VtVec2fArray>()) {
        // ramp knots: a three-knot dip in place of the flat default
        out.push_back({"ramp3", VtValue(VtVec2fArray{GfVec2f(0.0f, 1.0f), GfVec2f(0.5f, 0.4f),
                                                     GfVec2f(1.0f, 0.8f)})});
    } else if (fallback.IsHolding<GfVec3f>()) {
        GfVec3f v = fallback.UncheckedGet<GfVec3f>();
        for (int i = 0; i < 3; ++i) v[i] = Nudge(v[i]);
        out.push_back({"(" + Num(v[0]) + "," + Num(v[1]) + "," + Num(v[2]) + ")", VtValue(v)});
    }
    return out;
}

void SetParam(UsdGenNodeDesc *node, TfToken const &name, VtValue const &value)
{
    for (UsdGenParamValue &p : node->params)
        if (p.name == name) { p.value = value; return; }
    node->params.push_back({name, value, false});
}

/// The tuned fixture of `type` with its op edited by `fn`: status plus a
/// combined hash.
template <class Fn>
void ProbeNode(Table *table, TfToken const &type, std::string const &what, Fn const &fn)
{
    UsdGenGraphDesc desc = usdGenGolden::MakeOpGraph(type, true);
    for (UsdGenNodeDesc &n : desc.nodes)
        if (n.path == usdGenGolden::OpPath()) fn(&n);
    std::string const graph = "probe/" + type.GetString() + "/" + what;
    Rows cooked, rows;
    std::string const status = CookScheduler(desc, graph, &cooked);
    rows.Add("status", status);
    if (status == "ok") rows.Add("all", cooked.rows.front().second + " " + Hex(cooked.Combined()));
    table->Put(graph, std::move(rows));
}

void Probe(Table *table, TfToken const &type, TfToken const &name, std::string const &text,
           VtValue const &value)
{
    ProbeNode(table, type, name.GetString() + "=" + text,
              [&](UsdGenNodeDesc *n) { SetParam(n, name, value); });
}

void RunProbes(Table *table)
{
    for (TfToken const &type : usdGenGolden::FixtureTypes()) {
        UsdPrimDefinition const *def = Definition(type);
        if (!def) {
            // No schema class: probed by hand below.
            if (type != TfToken("UsdGenWidthBlend")) Fail("no schema definition for " + type.GetString());
            continue;
        }
        // The descriptor's own enabled/seed fields (the params of the same
        // name are probed below like any attribute).
        ProbeNode(table, type, "node.enabled=false", [](UsdGenNodeDesc *n) { n->enabled = false; });
        ProbeNode(table, type, "node.seed=8", [](UsdGenNodeDesc *n) { n->seed = 8; });
        std::vector<TfToken> names = def->GetPropertyNames();
        std::sort(names.begin(), names.end(),
                  [](TfToken const &a, TfToken const &b) { return a.GetString() < b.GetString(); });
        for (TfToken const &prop : names) {
            std::string const full = prop.GetString();
            if (full.compare(0, 7, "usdGen:") != 0) continue;
            UsdPrimDefinition::Attribute const attr = def->GetAttributeDefinition(prop);
            if (!attr) continue;
            for (auto const &probe : ProbeValues(attr))
                Probe(table, type, TfToken(full.substr(7)), probe.first, probe.second);
        }
    }
    TfToken const widthBlend("UsdGenWidthBlend");
    Probe(table, widthBlend, TfToken("widthBlend:weight"), "0.8", VtValue(0.8f));
    Probe(table, widthBlend, TfToken("enabled"), "false", VtValue(false));
}

// ---- in-code graphs --------------------------------------------------------------

void RunGraphs(Table *table)
{
    std::set<std::string> covered;
    for (usdGenGolden::GoldenGraph const &g : usdGenGolden::MakeGoldenGraphs()) {
        // Every authored param must be a schema attribute or a declared
        // parameter of its op: a typo would silently cook the fallback.
        for (UsdGenNodeDesc const &n : g.desc.nodes) {
            UsdPrimDefinition const *def = Definition(n.type);
            UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
            UsdGenOpId const id = registry.Find(n.type);
            for (UsdGenParamValue const &p : n.params) {
                bool known = def && def->GetAttributeDefinition(TfToken("usdGen:" + p.name.GetString()));
                // A routed decl of the op, exactly as spelled.
                if (id != UsdGenOpId::Invalid) {
                    UsdGenOpRecord const &rec = registry.Record(id);
                    for (UsdGenParamSlot s = 0; s < rec.ParamCount(); ++s)
                        known = known || (rec.Def().params[s].routed && rec.ParamName(s) == p.name);
                }
                if (!known)
                    Fail(g.name + ": " + n.type.GetString() + " has no parameter " +
                         p.name.GetString());
            }
        }
        Rows rows;
        std::string const status = CookScheduler(g.desc, g.name, &rows);
        rows.rows.insert(rows.rows.begin(), {"status", status});
        if (status != "ok") Fail(g.name + ": " + status);
        else if (rows.rows[1].second.rfind("curves=0 ", 0) == 0) Fail(g.name + ": cooks no curves");
        if (status == "ok") {
            for (UsdGenNodeDesc const &n : g.desc.nodes) covered.insert(n.type.GetString());
            CookSession(g.desc, g.name, &rows);
        }
        table->Put(g.name, std::move(rows));
    }
    for (uint32_t i = 0; i < UsdGenOpRegistry::Get().Count(); ++i)
        if (TfToken const &type = UsdGenOpRegistry::Get().Record(UsdGenOpId(i)).Name();
            !covered.count(type.GetString()))
            Fail("registered op " + type.GetString() + " is not covered by a cooking in-code graph");
}

// ---- examples ---------------------------------------------------------------------

void CookExampleStage(UsdStageRefPtr const &stage, std::string const &name, Table *table)
{
    std::vector<SdfPath> descriptions;
    for (UsdPrim const &prim : stage->Traverse())
        if (prim.GetTypeName() == TfToken("UsdGenDescription")) descriptions.push_back(prim.GetPath());
    for (SdfPath const &path : descriptions) {
        std::string const graph = name + ":" + path.GetString();
        UsdGenGraphDesc desc = usdGenImaging::BuildGraphDescFromStage(stage, path);
        Rows rows;
        if (!desc.validationErrors.empty()) {
            PrintDiagnostics(graph + " builder", desc.validationErrors);
            rows.Add("status", "rejected");
            table->Put(graph, std::move(rows));
            continue;
        }
        desc.executionBackend = UsdGenExecutionBackend::CpuReference;
        std::string const status = CookScheduler(desc, graph, &rows);
        rows.rows.insert(rows.rows.begin(), {"status", status});
        if (status == "ok") CookSession(desc, graph, &rows);
        table->Put(graph, std::move(rows));
    }
}

void RunExamples(Table *table)
{
    namespace fs = std::filesystem;
    fs::path const root = fs::path(USDGEN_TEST_SOURCE_DIR) / "examples";
    std::vector<std::string> files;
    for (auto const &entry : fs::recursive_directory_iterator(root))
        if (entry.is_regular_file() && entry.path().extension() == ".usda")
            files.push_back(fs::relative(entry.path(), root).generic_string());
    std::sort(files.begin(), files.end());
    if (files.empty()) Fail("no examples found under " + root.generic_string());
    for (std::string const &rel : files) {
        UsdStageRefPtr const stage = UsdStage::Open((root / rel).string(), UsdStage::LoadAll);
        std::string const name = "example/" + rel;
        if (!stage) {
            Rows rows;
            rows.Add("status", "unopenable");
            table->Put(name, std::move(rows));
            continue;
        }
        // Only self-contained examples: a reference outside examples/ (or a
        // missing one) would make the result depend on local files.
        bool selfContained = stage->GetCompositionErrors().empty();
        auto folded = [](fs::path const &p) {
            std::string s = fs::weakly_canonical(p).generic_string();
            for (char &c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        std::string const rootText = folded(root);
        for (SdfLayerHandle const &layer : stage->GetUsedLayers()) {
            std::string const real = layer->GetRealPath();
            if (!real.empty() && folded(fs::path(real)).rfind(rootText, 0) != 0)
                selfContained = false;
        }
        if (!selfContained) continue;
        CookExampleStage(stage, name, table);
        // Each non-default variant selection, one set at a time.
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
        for (Choice const &c : choices)
            for (std::string const &v : c.names) {
                if (v == c.current) continue;
                UsdVariantSet vs = stage->GetPrimAtPath(c.prim).GetVariantSet(c.set);
                vs.SetVariantSelection(v);
                CookExampleStage(stage, name + "{" + c.prim.GetString() + ":" + c.set + "=" + v + "}",
                                 table);
                vs.ClearVariantSelection();
            }
    }
}

// ---- compare -------------------------------------------------------------------------

std::map<std::string, std::string> Parse(std::string const &text,
                                         std::vector<std::string> *order)
{
    std::map<std::string, std::string> rows;
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        size_t const a = line.find(' ');
        size_t const b = a == std::string::npos ? a : line.find(' ', a + 1);
        if (b == std::string::npos) continue;
        std::string const key = line.substr(0, b);
        if (!rows.count(key)) order->push_back(key);
        rows[key] = line.substr(b + 1);
    }
    return rows;
}

/// The file with CRLF folded to LF (git may convert on checkout).
std::string ReadFile(std::string const &path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream os;
    os << in.rdbuf();
    std::string text = os.str();
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
}

bool Compare(std::string const &expectedText, std::string const &actualText)
{
    std::vector<std::string> expectedOrder, actualOrder;
    auto const expected = Parse(expectedText, &expectedOrder);
    auto const actual = Parse(actualText, &actualOrder);
    // Per graph: planes that changed, vanished or appeared.
    std::map<std::string, std::vector<std::string>> changed, missing, added;
    std::vector<std::string> graphs;
    auto graphOf = [](std::string const &key) { return key.substr(0, key.find(' ')); };
    auto planeOf = [](std::string const &key) { return key.substr(key.find(' ') + 1); };
    auto note = [&](std::map<std::string, std::vector<std::string>> &m, std::string const &key) {
        std::string const g = graphOf(key);
        if (!changed.count(g) && !missing.count(g) && !added.count(g)) graphs.push_back(g);
        m[g].push_back(planeOf(key));
    };
    size_t diffs = 0, shown = 0;
    for (std::string const &key : expectedOrder) {
        auto const it = actual.find(key);
        if (it == actual.end()) { note(missing, key); ++diffs; }
        else if (it->second != expected.at(key)) {
            note(changed, key);
            ++diffs;
            if (changed[graphOf(key)].size() <= 3 && ++shown <= 24)
                std::printf("  %s\n    expected %s\n    actual   %s\n", key.c_str(),
                            expected.at(key).c_str(), it->second.c_str());
        }
    }
    for (std::string const &key : actualOrder)
        if (!expected.count(key)) { note(added, key); ++diffs; }
    if (diffs == 0) return true;
    auto join = [](std::vector<std::string> const &v) {
        std::string s;
        for (size_t i = 0; i < v.size() && i < 12; ++i) s += (i ? ", " : "") + v[i];
        if (v.size() > 12) s += ", ... (" + std::to_string(v.size()) + ")";
        return s;
    };
    std::printf("--- per-graph diff summary ---\n");
    for (std::string const &g : graphs) {
        std::printf("%s\n", g.c_str());
        if (changed.count(g)) std::printf("    changed: %s\n", join(changed[g]).c_str());
        if (missing.count(g)) std::printf("    missing: %s\n", join(missing[g]).c_str());
        if (added.count(g)) std::printf("    new:     %s\n", join(added[g]).c_str());
    }
    std::printf("%zu of %zu golden rows differ in %zu graphs\n", diffs, expected.size(),
                graphs.size());
    return false;
}

}  // namespace

int main()
{
    g_verbose = std::getenv("USDGEN_GOLDEN_VERBOSE") != nullptr;
    char const *write = std::getenv("USDGEN_GOLDEN_WRITE");
    if (!Definition(TfToken("UsdGenLength"))) {
        std::printf("FAIL: the usdGen schema is not loaded (PXR_PLUGINPATH_NAME)\n");
        return 1;
    }

    std::string const goldenPath = std::string(USDGEN_GOLDEN_DIR) + "/cpuOutput.txt";
    bool const rewrite = write && std::strcmp(write, "1") == 0;
    std::string const expected = ReadFile(goldenPath);
    std::string const toolchain = std::string("\n# toolchain ") + kToolchain + "\n";
    if (!rewrite && !expected.empty() && expected.find(toolchain) == std::string::npos) {
        std::printf("SKIP: tests/golden/cpuOutput.txt was recorded with another toolchain\n");
        return 77;
    }

    Table table;
    RunGraphs(&table);
    RunProbes(&table);
    RunExamples(&table);
    std::string const text = table.Text();
    std::printf("%zu graphs, %zu bytes of golden text\n", table.order.size(), text.size());

    if (rewrite) {
        std::ofstream(goldenPath, std::ios::binary) << text;
        std::printf("wrote %s\n", goldenPath.c_str());
        return g_failures ? 1 : 0;
    }
    if (expected.empty()) Fail("missing golden file tests/golden/cpuOutput.txt");
    else if (expected != text) {
        std::ofstream("cpuOutput.actual.txt", std::ios::binary) << text;
        Compare(expected, text);
        Fail("output differs from tests/golden/cpuOutput.txt (actual: cpuOutput.actual.txt)");
    }
    if (g_failures) std::printf("testUsdGenOutputGoldens: FAILED (%d)\n", g_failures);
    else std::printf("testUsdGenOutputGoldens: PASS\n");
    return g_failures ? 1 : 0;
}
