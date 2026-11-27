// Paintable per-surface attribute maps and stroke accumulation:
// spec validation, texel access, sampling, dab falloff/modes, move
// interpolation, and the press/move/release/abort stroke contract
// (plan/08-tools.md §2.1: moves recompute from the press-time base).
#include "usdGenSurface/maps/attributeMap.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what, int line)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL (line %d): %s\n", line, what.c_str());
    }
}
// Explicit check: the build may define NDEBUG, so assert() proves nothing.
#define CHECK(cond, what) Check(static_cast<bool>(cond), what, __LINE__)

bool Near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

UsdGenAttributeMapSpec Spec(int faces = 1, int res = 8, int channels = 1)
{
    UsdGenAttributeMapSpec spec;
    spec.numFaces = faces;
    spec.resolution = res;
    spec.channels = channels;
    return spec;
}

void CheckSpecValidation()
{
    std::string error;
    CHECK(!UsdGenAttributeMap::Create(Spec(0), &error) && !error.empty(),
          "numFaces 0 is rejected");
    CHECK(!UsdGenAttributeMap::Create(Spec(1, 3), &error), "resolution 3 (non-pow2) rejected");
    CHECK(!UsdGenAttributeMap::Create(Spec(1, 512), &error), "resolution 512 rejected");
    CHECK(!UsdGenAttributeMap::Create(Spec(1, 0), &error), "resolution 0 rejected");
    CHECK(!UsdGenAttributeMap::Create(Spec(1, 8, 2), &error), "channels 2 rejected");
    CHECK(!UsdGenAttributeMap::Create(Spec(1, 8, 4), &error), "channels 4 rejected");
    UsdGenAttributeMapSpec bad = Spec();
    bad.defaultValue = std::nanf("");
    CHECK(!UsdGenAttributeMap::Create(bad, &error), "NaN default rejected");
    // Null error sink still fails closed.
    CHECK(!UsdGenAttributeMap::Create(Spec(0), nullptr), "null error sink still fails");

    error.clear(); // Create only writes *error on failure, like UsdGenPtexTexture::Open
    auto map = UsdGenAttributeMap::Create(Spec(4, 16, 3), &error);
    CHECK(map && error.empty(), "valid spec opens");
    CHECK(map->NumFaces() == 4, "NumFaces");
    CHECK(map->Resolution() == 16, "Resolution");
    CHECK(map->Channels() == 3, "Channels");
    CHECK(map->FloatCount() == 4 * 16 * 16 * 3, "FloatCount");
}

void CheckTexels()
{
    std::string error;
    auto map = UsdGenAttributeMap::Create(Spec(2, 8), &error);
    float value = -1.0f;
    CHECK(map->GetTexel(0, 0, 0, 0, &value) && value == 0.0f, "fresh texel is default");
    CHECK(map->SetTexel(1, 7, 7, 0, 2.5f), "SetTexel in range");
    CHECK(map->GetTexel(1, 7, 7, 0, &value) && value == 2.5f, "GetTexel reads back");
    CHECK(!map->GetTexel(2, 0, 0, 0, &value), "face past the end rejected");
    CHECK(!map->GetTexel(0, 8, 0, 0, &value), "texel s past the end rejected");
    CHECK(!map->GetTexel(0, 0, -1, 0, &value), "negative texel t rejected");
    CHECK(!map->GetTexel(0, 0, 0, 1, &value), "channel past the end rejected");
    CHECK(!map->GetTexel(0, 0, 0, 0, nullptr), "null out-param rejected");
    CHECK(!map->SetTexel(0, 0, 0, 0, std::nanf("")), "NaN write rejected");
    map->Fill(0.25f);
    CHECK(map->GetTexel(1, 7, 7, 0, &value) && value == 0.25f, "Fill overwrites");
    map->Fill(std::nanf(""));  // ignored, never a NaN fill
    CHECK(map->GetTexel(0, 0, 0, 0, &value) && value == 0.25f, "NaN Fill ignored");

    UsdGenAttributeMapSpec clamped = Spec();
    clamped.defaultValue = 2.0f;
    clamped.clamp01 = true;
    auto cmap = UsdGenAttributeMap::Create(clamped, &error);
    CHECK(cmap->GetTexel(0, 0, 0, 0, &value) && value == 1.0f, "default clamped at create");
    CHECK(cmap->SetTexel(0, 1, 1, 0, -3.0f), "clamped write accepted");
    CHECK(cmap->GetTexel(0, 1, 1, 0, &value) && value == 0.0f, "write clamped to [0, 1]");
}

void CheckSampling()
{
    std::string error;
    auto map = UsdGenAttributeMap::Create(Spec(1, 2), &error);
    map->SetTexel(0, 0, 0, 0, 0.0f);
    map->SetTexel(0, 1, 0, 0, 1.0f);
    map->SetTexel(0, 0, 1, 0, 2.0f);
    map->SetTexel(0, 1, 1, 0, 3.0f);
    float value = -1.0f;
    CHECK(map->Sample(0, 0.1f, 0.1f, 0, UsdGenAttributeMapInterp::Nearest, &value) &&
              value == 0.0f,
          "nearest corner");
    CHECK(map->Sample(0, 0.9f, 0.1f, 0, UsdGenAttributeMapInterp::Nearest, &value) &&
              value == 1.0f,
          "nearest picks the texel under (u, v)");
    CHECK(map->Sample(0, 0.5f, 0.5f, 0, UsdGenAttributeMapInterp::Bilinear, &value) &&
              Near(value, 1.5, 1e-6),
          "bilinear centre is the four-corner mean");
    CHECK(map->Sample(0, 0.0f, 0.0f, 0, UsdGenAttributeMapInterp::Bilinear, &value) &&
              value == 0.0f,
          "bilinear corner is exact");
    CHECK(map->Sample(0, 1.0f, 1.0f, 0, UsdGenAttributeMapInterp::Bilinear, &value) &&
              value == 3.0f,
          "bilinear far corner is exact");
    CHECK(map->Sample(0, -2.0f, 99.0f, 0, UsdGenAttributeMapInterp::Bilinear, &value) &&
              value == 2.0f,
          "(u, v) clamps to [0, 1]");
    auto nodes = UsdGenAttributeMap::Create(Spec(1, 4), &error);
    nodes->SetTexel(0, 1, 1, 0, 1.0f);
    CHECK(nodes->Sample(0, 1.0f / 3.0f, 1.0f / 3.0f, 0,
                        UsdGenAttributeMapInterp::Bilinear, &value) &&
              Near(value, 1.0, 1e-6),
          "bilinear is exact on interior grid nodes");
    CHECK(nodes->Sample(0, 0.5f, 0.5f, 0, UsdGenAttributeMapInterp::Bilinear, &value) &&
              Near(value, 0.25, 1e-6),
          "and blends evenly between them");
    CHECK(!map->Sample(1, 0.5f, 0.5f, 0, UsdGenAttributeMapInterp::Bilinear, &value),
          "face past the end rejected");
    CHECK(!map->Sample(0, 0.5f, 0.5f, 1, UsdGenAttributeMapInterp::Nearest, &value),
          "channel past the end rejected");
    CHECK(!map->Sample(0, std::nanf(""), 0.5f, 0, UsdGenAttributeMapInterp::Nearest, &value),
          "NaN u rejected");
    CHECK(!map->Sample(0, 0.5f, 0.5f, 0, UsdGenAttributeMapInterp::Nearest, nullptr),
          "null out-param rejected");
}

void CheckDigest()
{
    std::string error;
    auto a = UsdGenAttributeMap::Create(Spec(2, 8), &error);
    auto b = a->Clone();
    CHECK(a->Digest() == b->Digest(), "clone digests identically");
    CHECK(std::memcmp(a->Data(), b->Data(), a->FloatCount() * sizeof(float)) == 0,
          "clone is bitwise identical");
    b->SetTexel(1, 0, 0, 0, 0.5f);
    CHECK(a->Digest() != b->Digest(), "one texel flips the digest");
    auto c = UsdGenAttributeMap::Create(Spec(2, 16), &error);
    CHECK(a->Digest() != c->Digest(), "the spec feeds the digest");
}

UsdGenBrushDab Dab(int face = 0, float u = 0.5f, float v = 0.5f)
{
    UsdGenBrushDab dab;
    dab.face = face;
    dab.u = u;
    dab.v = v;
    return dab;
}

void CheckDabValidation()
{
    std::string error;
    auto base = UsdGenAttributeMap::Create(Spec(2, 8), &error);
    UsdGenBrushStroke stroke(base);
    CHECK(!stroke.AddDab(Dab(2), &error) && !error.empty(), "face past the end rejected");
    UsdGenBrushDab nan = Dab();
    nan.u = std::nanf("");
    CHECK(!stroke.AddDab(nan, &error), "NaN u rejected");
    UsdGenBrushDab neg = Dab();
    neg.radius = -1.0f;
    CHECK(!stroke.AddDab(neg, &error), "negative radius rejected");
    UsdGenBrushDab strong = Dab();
    strong.strength = 1.5f;
    CHECK(!stroke.AddDab(strong, &error), "strength above 1 rejected");
    UsdGenBrushDab chan = Dab();
    chan.channel = 1;
    CHECK(!stroke.AddDab(chan, &error), "channel past the end rejected");
    CHECK(stroke.DabCount() == 0, "rejected dabs record nothing");
    UsdGenBrushDab zero = Dab();
    zero.strength = 0.0f;
    CHECK(stroke.AddDab(zero, &error), "strength-0 dab is valid");
    CHECK(stroke.DabCount() == 1, "valid dab recorded");
    CHECK(stroke.Preview()->Digest() == base->Digest(), "strength-0 dab is a no-op");
    UsdGenBrushStroke off(base);
    UsdGenBrushDab far = Dab(0, 5.0f, 5.0f);
    far.radius = 0.1f;
    far.strength = 1.0f;
    far.value = 1.0f;
    far.mode = UsdGenBrushMode::Smooth;
    CHECK(off.AddDab(far, &error), "off-face dab is valid");
    CHECK(off.Preview()->Digest() == base->Digest(), "off-face dab touches nothing");
}

void CheckSetDab()
{
    std::string error;
    auto base = UsdGenAttributeMap::Create(Spec(4, 8), &error);
    UsdGenBrushStroke stroke(base);
    UsdGenBrushDab dab = Dab(2);
    dab.radius = 1.0f;  // covers the addressed face from its centre
    dab.strength = 1.0f;
    dab.value = 1.0f;
    dab.falloff = UsdGenBrushFalloff::Constant;
    CHECK(stroke.AddDab(dab, &error), "full-face dab accepted");
    auto done = stroke.Preview();
    float value = 0.0f;
    for (int t = 0; t < 8; ++t)
        for (int s = 0; s < 8; ++s) {
            done->GetTexel(2, s, t, 0, &value);
            if (value != 1.0f) break;
        }
    CHECK(value == 1.0f, "strength-1 constant Set paints the face");
    done->GetTexel(0, 4, 4, 0, &value);
    bool othersClean = value == 0.0f;
    done->GetTexel(1, 4, 4, 0, &value);
    othersClean = othersClean && value == 0.0f;
    done->GetTexel(3, 4, 4, 0, &value);
    CHECK(othersClean && value == 0.0f, "v1 dab clips at the face border");

    // Partial strength mixes toward the value.
    UsdGenBrushStroke half(base);
    UsdGenBrushDab mid = Dab();
    mid.radius = 1.0f;
    mid.strength = 0.25f;
    mid.value = 1.0f;
    mid.falloff = UsdGenBrushFalloff::Constant;
    half.AddDab(mid, &error);
    half.Preview()->GetTexel(0, 0, 0, 0, &value);
    CHECK(Near(value, 0.25, 1e-6), "Set mixes by strength * weight");
}

void CheckFalloffShape()
{
    std::string error;
    // Dab centred exactly on a grid node, radius two node spacings.
    auto base = UsdGenAttributeMap::Create(Spec(1, 8), &error);
    auto paint = [&](UsdGenBrushFalloff falloff) {
        UsdGenBrushStroke stroke(base);
        UsdGenBrushDab dab = Dab(0, 3.0f / 7.0f, 3.0f / 7.0f);  // texel (3, 3) node
        dab.radius = 2.0f / 7.0f;
        dab.strength = 1.0f;
        dab.value = 1.0f;
        dab.falloff = falloff;
        stroke.AddDab(dab, &error);
        return stroke.Preview();
    };
    float centre = 0.0f, side = 0.0f, diag = 0.0f, far = 0.0f;
    auto linear = paint(UsdGenBrushFalloff::Linear);
    linear->GetTexel(0, 3, 3, 0, &centre);
    linear->GetTexel(0, 4, 3, 0, &side);
    linear->GetTexel(0, 4, 4, 0, &diag);
    linear->GetTexel(0, 7, 7, 0, &far);
    CHECK(centre == 1.0f, "linear: dab centre paints fully");
    CHECK(Near(side, 0.5, 1e-5), "linear: one-texel neighbour is half weight");
    CHECK(Near(diag, 1.0 - std::sqrt(2.0) / 2.0, 1e-4), "linear: diagonal follows 1 - d/r");
    CHECK(far == 0.0f, "linear: outside the radius is untouched");

    auto smooth = paint(UsdGenBrushFalloff::Smooth);
    smooth->GetTexel(0, 3, 3, 0, &centre);
    smooth->GetTexel(0, 4, 4, 0, &diag);
    double const t = 1.0 - std::sqrt(2.0) / 2.0;
    double const s = t * t * (3.0 - 2.0 * t);
    CHECK(centre == 1.0f, "smooth: dab centre paints fully");
    CHECK(Near(diag, s, 1e-4), "smooth: diagonal follows smoothstep");
    float linearDiag = 0.0f;
    linear->GetTexel(0, 4, 4, 0, &linearDiag);
    CHECK(diag < linearDiag, "smooth falls off faster than linear below t = 0.5");

    auto constant = paint(UsdGenBrushFalloff::Constant);
    constant->GetTexel(0, 4, 4, 0, &diag);
    CHECK(diag == 1.0f, "constant: every texel inside paints fully");
}

void CheckAddAndChannels()
{
    std::string error;
    auto base = UsdGenAttributeMap::Create(Spec(1, 8, 3), &error);
    UsdGenBrushStroke stroke(base);
    UsdGenBrushDab dab = Dab();
    dab.radius = 1.0f;
    dab.strength = 1.0f;
    dab.value = 0.25f;
    dab.mode = UsdGenBrushMode::Add;
    dab.falloff = UsdGenBrushFalloff::Constant;
    stroke.AddDab(dab, &error);
    stroke.AddDab(dab, &error);  // dabs apply in order
    float r = 0.0f, g = 0.0f, b = 0.0f;
    auto done = stroke.Preview();
    done->GetTexel(0, 0, 0, 0, &r);
    done->GetTexel(0, 0, 0, 1, &g);
    done->GetTexel(0, 0, 0, 2, &b);
    CHECK(Near(r, 0.5, 1e-6) && Near(g, 0.5, 1e-6) && Near(b, 0.5, 1e-6),
          "Add accumulates in dab order over all channels");

    UsdGenBrushStroke one(base);
    UsdGenBrushDab single = Dab();
    single.radius = 1.0f;
    single.strength = 1.0f;
    single.value = 1.0f;
    single.channel = 1;
    single.falloff = UsdGenBrushFalloff::Constant;
    one.AddDab(single, &error);
    auto painted = one.Preview();
    painted->GetTexel(0, 0, 0, 0, &r);
    painted->GetTexel(0, 0, 0, 1, &g);
    painted->GetTexel(0, 0, 0, 2, &b);
    CHECK(r == 0.0f && g == 1.0f && b == 0.0f, "explicit channel paints one channel");
}

void CheckZeroRadiusDab()
{
    std::string error;
    auto base = UsdGenAttributeMap::Create(Spec(1, 8), &error);
    UsdGenBrushStroke stroke(base);
    UsdGenBrushDab dab = Dab(0, 0.5f, 0.5f);
    dab.radius = 0.0f;
    dab.strength = 1.0f;
    dab.value = 1.0f;
    stroke.AddDab(dab, &error);
    auto done = stroke.Preview();
    double total = 0.0;
    for (size_t i = 0; i < done->FloatCount(); ++i) total += done->Data()[i];
    CHECK(Near(total, 1.0, 1e-6), "zero-radius dab paints exactly one texel");
}

void CheckSmoothDab()
{
    std::string error;
    auto base = UsdGenAttributeMap::Create(Spec(1, 4), &error);
    for (int t = 0; t < 4; ++t)
        for (int s = 0; s < 4; ++s)
            base->SetTexel(0, s, t, 0, float((s + t) % 2));  // checkerboard
    UsdGenBrushStroke stroke(base);
    UsdGenBrushDab dab = Dab();
    dab.radius = 1.0f;
    dab.strength = 1.0f;
    dab.mode = UsdGenBrushMode::Smooth;
    dab.falloff = UsdGenBrushFalloff::Constant;
    stroke.AddDab(dab, &error);
    auto done = stroke.Preview();
    float corner = 0.0f;
    done->GetTexel(0, 0, 0, 0, &corner);
    // Clamped 3x3 mean of the corner: (0*4 + 1*2 + 1*2 + 0) / 9.
    CHECK(Near(corner, 4.0 / 9.0, 1e-5), "Smooth relaxes toward the neighbourhood mean");
    float lo = 1.0f, hi = 0.0f;
    for (size_t i = 0; i < done->FloatCount(); ++i) {
        lo = std::min(lo, done->Data()[i]);
        hi = std::max(hi, done->Data()[i]);
    }
    CHECK(hi - lo < 1.0f, "Smooth reduces the range");
}

void CheckMoveInterpolation()
{
    std::string error;
    auto base = UsdGenAttributeMap::Create(Spec(2, 8), &error);
    UsdGenBrushStroke stroke(base);
    CHECK(stroke.AddMove(0, 0.1f, 0.5f, 0.1f, 1.0f, 1.0f, -1, UsdGenBrushMode::Set,
                         UsdGenBrushFalloff::Constant, &error),
          "first move stamps one dab");
    CHECK(stroke.DabCount() == 1, "first move dab count");
    // 0.8 of travel at spacing 0.5 * radius 0.1: 16 stamps plus no double
    // paint of the start point.
    CHECK(stroke.AddMove(0, 0.9f, 0.5f, 0.1f, 1.0f, 1.0f, -1, UsdGenBrushMode::Set,
                         UsdGenBrushFalloff::Constant, 0.5f, &error),
          "second move interpolates");
    CHECK(stroke.DabCount() == 17, "interpolated dab count covers the segment");
    UsdGenBrushDab const &last = stroke.Dab(stroke.DabCount() - 1);
    CHECK(Near(last.u, 0.9, 1e-6) && Near(last.v, 0.5, 1e-6), "interpolation ends at the move");
    CHECK(stroke.AddMove(1, 0.5f, 0.5f, 0.1f, 1.0f, 1.0f, -1, UsdGenBrushMode::Set,
                         UsdGenBrushFalloff::Constant, &error),
          "face change stamps without interpolating");
    CHECK(stroke.DabCount() == 18, "face change adds one dab");
    CHECK(!stroke.AddMove(1, 0.6f, 0.5f, 0.1f, 1.0f, 1.0f, -1, UsdGenBrushMode::Set,
                          UsdGenBrushFalloff::Constant, 0.0f, &error),
          "spacing 0 rejected");
    CHECK(stroke.DabCount() == 18, "rejected move records nothing");
}

void CheckStrokeContract()
{
    std::string error;
    auto base = UsdGenAttributeMap::Create(Spec(1, 8), &error);
    base->Fill(0.25f);
    uint64_t const baseDigest = base->Digest();
    UsdGenBrushStroke stroke(base);
    UsdGenBrushDab dab = Dab();
    dab.radius = 0.5f;
    dab.strength = 0.75f;
    dab.value = 1.0f;
    stroke.AddDab(dab, &error);
    auto first = stroke.Preview();
    auto second = stroke.Preview();
    CHECK(first->Digest() == second->Digest(), "Preview is idempotent");
    CHECK(std::memcmp(first->Data(), second->Data(), first->FloatCount() * sizeof(float)) == 0,
          "Preview is bitwise repeatable");
    CHECK(first->Digest() != baseDigest, "Preview differs from the base");
    CHECK(base->Digest() == baseDigest, "Preview never mutates the base");
    auto committed = stroke.Commit();
    CHECK(committed->Digest() == first->Digest(), "Commit is the recomputed map");
    CHECK(stroke.DabCount() == 1, "Commit/Preview keep the stroke");
    stroke.Abort();
    CHECK(stroke.DabCount() == 0, "Abort forgets the dabs");
    CHECK(stroke.Preview()->Digest() == baseDigest, "Abort restores the base exactly");
}

void CheckClampedStroke()
{
    std::string error;
    UsdGenAttributeMapSpec spec = Spec();
    spec.clamp01 = true;
    auto base = UsdGenAttributeMap::Create(spec, &error);
    base->Fill(0.9f);
    UsdGenBrushStroke stroke(base);
    UsdGenBrushDab dab = Dab();
    dab.radius = 1.0f;
    dab.strength = 1.0f;
    dab.value = 0.5f;
    dab.mode = UsdGenBrushMode::Add;
    dab.falloff = UsdGenBrushFalloff::Constant;
    stroke.AddDab(dab, &error);
    float value = 0.0f;
    stroke.Preview()->GetTexel(0, 0, 0, 0, &value);
    CHECK(value == 1.0f, "clamped map clips dab results to [0, 1]");
}

}  // namespace

int main()
{
    CheckSpecValidation();
    CheckTexels();
    CheckSampling();
    CheckDigest();
    CheckDabValidation();
    CheckSetDab();
    CheckFalloffShape();
    CheckAddAndChannels();
    CheckZeroRadiusDab();
    CheckSmoothDab();
    CheckMoveInterpolation();
    CheckStrokeContract();
    CheckClampedStroke();

    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenAttributeMap: OK\n");
    return 0;
}
