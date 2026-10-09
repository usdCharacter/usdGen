// Digest contract for capture-class hashing (usdGen/digest.h): values are
// internal cache keys, so the contract is determinism + input sensitivity,
// never golden values.
//
//   * UsdGenDigestBytes is deterministic, sensitive to every byte
//     (all tail lengths), and order-sensitive across lanes.
//   * UsdGenDigestCombine4 is order-sensitive.
//   * UsdGenScatterOp::CaptureDigest is stable across repeat calls and moves
//     when density or surface content moves (in-process).
//   * Attribute-cook digest stability/sensitivity lives in
//     testUsdGenAttributeInstance (it exercises the same lane helper).
#include "usdGen/digest.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/ops/scatter.h"

#include <cstdio>
#include <string>
#include <vector>

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
#define CHECK(cond, what) Check(static_cast<bool>(cond), what, __LINE__)

void CheckBytes()
{
    // Determinism + empty input.
    CHECK(UsdGenDigestBytes(nullptr, 0, UsdGenDigestOffset) ==
              UsdGenDigestBytes(nullptr, 0, UsdGenDigestOffset),
          "empty digest is deterministic");
    std::vector<unsigned char> buf(71);
    for (size_t i = 0; i < buf.size(); ++i)
        buf[i] = static_cast<unsigned char>(i * 37 + 11);
    for (size_t n = 0; n <= 70; ++n) {
        uint64_t const a = UsdGenDigestBytes(buf.data(), n, UsdGenDigestOffset);
        uint64_t const b = UsdGenDigestBytes(buf.data(), n, UsdGenDigestOffset);
        if (a != b) {
            CHECK(false, "digest is deterministic at every length");
            break;
        }
    }
    // Every byte position feeds the digest, at every tail length.
    for (size_t n = 1; n <= 70; ++n) {
        uint64_t const base =
            UsdGenDigestBytes(buf.data(), n, UsdGenDigestOffset);
        for (size_t p = 0; p < n; ++p) {
            buf[p] ^= 0xff;
            uint64_t const moved =
                UsdGenDigestBytes(buf.data(), n, UsdGenDigestOffset);
            buf[p] ^= 0xff;
            if (moved == base) {
                CHECK(false, "every byte flips the digest");
                break;
            }
        }
    }
    // Lane order matters: a rotation is not a fixed point.
    {
        unsigned char ab[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        unsigned char ba[8] = {5, 6, 7, 8, 1, 2, 3, 4};
        CHECK(UsdGenDigestBytes(ab, 8, UsdGenDigestOffset) !=
                  UsdGenDigestBytes(ba, 8, UsdGenDigestOffset),
              "byte order feeds the digest");
    }
    // Distinct seeds separate.
    CHECK(UsdGenDigestBytes(buf.data(), 64, 1) !=
              UsdGenDigestBytes(buf.data(), 64, 2),
          "the seed feeds the digest");
}

void CheckCombine()
{
    uint64_t const base = UsdGenDigestCombine4(1, 2, 3, 4);
    CHECK(base == UsdGenDigestCombine4(1, 2, 3, 4), "combine is deterministic");
    CHECK(base != UsdGenDigestCombine4(2, 1, 3, 4),
          "combine is order-sensitive");
    CHECK(base != UsdGenDigestCombine4(1, 2, 3, 5),
          "every lane feeds the combine");
}

UsdGenGraphDesc SmallDesc(double density)
{
    UsdGenGraphDesc d;
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/surface");
    s.restPoints = {GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 1, 0),
                    GfVec3f(0, 1, 0)};
    s.uv = {GfVec2f(0, 0), GfVec2f(1, 0), GfVec2f(1, 1), GfVec2f(0, 1)};
    s.faceVertexCounts = {4};
    s.faceVertexIndices = {0, 1, 2, 3};
    d.surfaces.push_back(s);
    UsdGenNodeDesc n;
    n.path = SdfPath("/scatter");
    n.type = TfToken("UsdGenScatter");
    n.seed = 41;
    n.surfaces = {s.path};
    n.params = {{TfToken("density"), VtValue(density), false}};
    d.nodes.push_back(n);
    return d;
}

UsdGenEpoch ScatterDigest(UsdGenGraphDesc const &desc)
{
    UsdGenScatterOp op;
    UsdGenParamView params{&desc, &desc.nodes[0]};
    UsdGenCaptureContext ctx;
    ctx.desc = &desc;
    ctx.params = &params;
    ctx.surface = 0;
    ctx.seed = uint32_t(desc.nodes[0].seed);
    return op.CaptureDigest(ctx);
}

void CheckScatterDigest()
{
    UsdGenGraphDesc const desc = SmallDesc(8.0);
    UsdGenEpoch const a = ScatterDigest(desc);
    UsdGenEpoch const b = ScatterDigest(desc);
    CHECK(a == b, "scatter digest is stable across repeat calls");
    UsdGenGraphDesc dense = SmallDesc(9.0);
    CHECK(ScatterDigest(dense) != a, "density moves the scatter digest");
    UsdGenGraphDesc moved = SmallDesc(8.0);
    moved.surfaces[0].restPoints[2] = GfVec3f(1, 2, 0);
    CHECK(ScatterDigest(moved) != a, "surface content moves the digest");
    UsdGenGraphDesc flipped = SmallDesc(8.0);
    flipped.nodes[0].params.push_back(
        UsdGenParamValue{TfToken("flip"), VtValue(true), false});
    CHECK(ScatterDigest(flipped) != a, "flip moves the scatter digest");
}

}  // namespace

int main()
{
    CheckBytes();
    CheckCombine();
    CheckScatterDigest();
    if (g_failures == 0)
        std::printf("testUsdGenDigest: PASS\n");
    else
        std::printf("testUsdGenDigest: %d FAILURES\n", g_failures);
    return g_failures ? 1 : 0;
}
