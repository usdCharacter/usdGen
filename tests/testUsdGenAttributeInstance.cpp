// Authored maps into groom cooking/instancing: the capture-time cook that
// samples one map channel at every strand root and decides keep/drop plus
// one prototype index per kept strand. Engine-only (T0), like the map.
//
//   * Validation fails closed and leaves the result untouched.
//   * Sampled values match UsdGenAttributeMap::Sample (bilinear), the same
//     call the groom preview makes.
//   * keep = sampled && value >= threshold; unreadable roots report the
//     default and drop explicitly, never landing on prototype zero.
//   * Prototypes partition the kept strands in strand order; value 1.0
//     lands on the last prototype.
//   * The cook is deterministic and its digest folds map, params and
//     assignments for the capture epoch.
#include "usdGen/maps/attributeInstance.h"
#include "usdGen/maps/attributeMap.h"

#include "tbb/task_arena.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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

bool Near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

std::shared_ptr<UsdGenAttributeMap> TwoFaceMap(int res = 4)
{
    UsdGenAttributeMapSpec spec;
    spec.numFaces = 2;
    spec.resolution = res;
    spec.channels = 1;
    std::string error;
    return UsdGenAttributeMap::Create(spec, &error);
}

UsdGenAttributeInstanceRoot Root(int face, float u = 0.5f, float v = 0.5f)
{
    UsdGenAttributeInstanceRoot root;
    root.face = face;
    root.u = u;
    root.v = v;
    return root;
}

UsdGenAttributeInstanceInput CookInput(std::shared_ptr<const UsdGenAttributeMap> map)
{
    UsdGenAttributeInstanceInput input;
    input.map = std::move(map);
    input.roots = {Root(0), Root(1)};
    return input;
}

void CheckValidation()
{
    std::string error;
    UsdGenAttributeInstanceResult result;
    result.kept = 999;  // a failed cook leaves the result untouched

    UsdGenAttributeInstanceInput nullMap = CookInput(TwoFaceMap());
    nullMap.map.reset();
    CHECK(!UsdGenAttributeCookInstances(nullMap, &result, &error) && !error.empty(),
          "null map rejected");
    CHECK(result.kept == 999, "a rejected cook touches nothing");

    UsdGenAttributeInstanceInput badChannel = CookInput(TwoFaceMap());
    badChannel.channel = 1;
    CHECK(!UsdGenAttributeCookInstances(badChannel, &result, &error),
          "channel past the end rejected");

    UsdGenAttributeInstanceInput negativeChannel = CookInput(TwoFaceMap());
    negativeChannel.channel = -1;
    CHECK(!UsdGenAttributeCookInstances(negativeChannel, &result, &error),
          "negative channel rejected");

    UsdGenAttributeInstanceInput noRoots = CookInput(TwoFaceMap());
    noRoots.roots.clear();
    CHECK(!UsdGenAttributeCookInstances(noRoots, &result, &error),
          "empty roots rejected");

    UsdGenAttributeInstanceInput noProtos = CookInput(TwoFaceMap());
    noProtos.numPrototypes = 0;
    CHECK(!UsdGenAttributeCookInstances(noProtos, &result, &error),
          "numPrototypes 0 rejected");

    UsdGenAttributeInstanceInput nanThreshold = CookInput(TwoFaceMap());
    nanThreshold.threshold = std::nanf("");
    CHECK(!UsdGenAttributeCookInstances(nanThreshold, &result, &error),
          "NaN threshold rejected");

    UsdGenAttributeInstanceInput nanDefault = CookInput(TwoFaceMap());
    nanDefault.defaultValue = std::nanf("");
    CHECK(!UsdGenAttributeCookInstances(nanDefault, &result, &error),
          "NaN default rejected");

    CHECK(!UsdGenAttributeCookInstances(CookInput(TwoFaceMap()), nullptr, &error),
          "null result rejected");
    CHECK(!UsdGenAttributeCookInstances(nullMap, &result, nullptr),
          "null error sink still fails");
    CHECK(result.kept == 999, "every rejected cook touches nothing");
}

void CheckSamplingParity()
{
    auto map = TwoFaceMap();
    for (int t = 0; t < 4; ++t)
        for (int s = 0; s < 4; ++s) {
            map->SetTexel(0, s, t, 0, float(s + t * 4) / 16.0f);
            map->SetTexel(1, s, t, 0, 1.0f - float(s) / 8.0f);
        }
    UsdGenAttributeInstanceInput input = CookInput(map);
    input.roots = {Root(0, 0.13f, 0.71f), Root(1, 0.87f, 0.29f), Root(0, 1.0f, 1.0f)};
    input.threshold = -1.0f;  // keep everything: this check is about values
    std::string error;
    UsdGenAttributeInstanceResult result;
    CHECK(UsdGenAttributeCookInstances(input, &result, &error), "cook accepts");
    CHECK(result.kept == 3, "negative threshold keeps all readable roots");
    bool parity = result.values.size() == 3;
    for (size_t i = 0; i < input.roots.size() && parity; ++i) {
        float want = 0.0f;
        map->Sample(input.roots[i].face, input.roots[i].u, input.roots[i].v, 0,
                    UsdGenAttributeMapInterp::Bilinear, &want);
        parity = result.values[i] == want;  // same call, bitwise
    }
    CHECK(parity, "cooked values match bilinear Sample bitwise");
    // res == 1 takes Sample's Nearest path; out-of-range uv clamps.
    auto one = TwoFaceMap(1);
    one->SetTexel(0, 0, 0, 0, 0.25f);
    one->SetTexel(1, 0, 0, 0, 0.75f);
    UsdGenAttributeInstanceInput input1 = CookInput(one);
    input1.roots = {Root(0, -0.5f, 2.0f), Root(1, 0.5f, 0.5f)};
    input1.threshold = -1.0f;
    UsdGenAttributeInstanceResult result1;
    CHECK(UsdGenAttributeCookInstances(input1, &result1, &error),
          "res-1 cook accepts");
    bool parity1 = result1.values.size() == 2;
    for (size_t i = 0; i < input1.roots.size() && parity1; ++i) {
        float want = 0.0f;
        one->Sample(input1.roots[i].face, input1.roots[i].u,
                    input1.roots[i].v, 0, UsdGenAttributeMapInterp::Bilinear,
                    &want);
        parity1 = result1.values[i] == want;
    }
    CHECK(parity1, "res-1 cooked values match Sample bitwise");
}

void CheckThreshold()
{
    auto map = TwoFaceMap();
    map->Fill(0.0f);
    for (int t = 0; t < 4; ++t)
        for (int s = 0; s < 4; ++s) map->SetTexel(1, s, t, 0, 1.0f);
    UsdGenAttributeInstanceInput input = CookInput(map);
    input.threshold = 0.5f;
    std::string error;
    UsdGenAttributeInstanceResult result;
    CHECK(UsdGenAttributeCookInstances(input, &result, &error), "threshold cook accepts");
    CHECK(result.kept == 1, "one strand survives the threshold");
    CHECK(result.keep.size() == 2 && result.keep[0] == 0 && result.keep[1] == 1,
          "the dense face keeps, the empty face drops");
    CHECK(result.prototype[0] == -1 && result.prototype[1] == 0,
          "the dropped strand assigns no prototype");
    CHECK((result.instanceIndices.size() == 1 &&
              result.instanceIndices[0] == std::vector<int>{1}),
          "instanceIndices holds the kept strand");
}

void CheckPrototypes()
{
    auto map = TwoFaceMap(1);  // one texel per face: exact values
    map->SetTexel(0, 0, 0, 0, 0.0f);
    map->SetTexel(1, 0, 0, 0, 1.0f);
    UsdGenAttributeInstanceInput input;
    input.map = map;
    input.roots = {Root(0), Root(1), Root(1, 0.0f, 0.0f)};
    input.threshold = 0.0f;
    input.numPrototypes = 3;
    std::string error;
    UsdGenAttributeInstanceResult result;
    CHECK(UsdGenAttributeCookInstances(input, &result, &error), "prototype cook accepts");
    CHECK((result.prototype == std::vector<int>{0, 2, 2}),
          "0.0 -> first prototype, 1.0 -> last");
    CHECK((result.instanceIndices.size() == 3 &&
              result.instanceIndices[0] == std::vector<int>{0} &&
              result.instanceIndices[1].empty() &&
              result.instanceIndices[2] == std::vector<int>{1, 2}),
          "instanceIndices partition kept strands in strand order");
}

void CheckUnreadableRoots()
{
    auto map = TwoFaceMap();
    map->Fill(1.0f);
    UsdGenAttributeInstanceInput input;
    input.map = map;
    UsdGenAttributeInstanceRoot nan = Root(0);
    nan.u = std::nanf("");
    input.roots = {Root(0), Root(7), nan};  // ok, past the end, NaN uv
    input.threshold = 0.5f;
    input.defaultValue = 0.75f;  // above threshold: still dropped
    std::string error;
    UsdGenAttributeInstanceResult result;
    CHECK(UsdGenAttributeCookInstances(input, &result, &error),
          "unreadable roots do not fail the cook");
    CHECK((result.kept == 1 && result.keep == std::vector<uint8_t>{1, 0, 0}),
          "unreadable roots drop explicitly");
    CHECK(Near(result.values[1], 0.75, 0.0) && Near(result.values[2], 0.75, 0.0),
          "unreadable roots report the default value");
    CHECK(result.prototype[1] == -1 && result.prototype[2] == -1,
          "no unresolved root is assigned prototype zero");
}

void CheckDeterminismAndDigest()
{
    auto map = TwoFaceMap();
    map->Fill(0.6f);
    UsdGenAttributeInstanceInput input = CookInput(map);
    input.numPrototypes = 2;
    std::string error;
    UsdGenAttributeInstanceResult first, second;
    CHECK(UsdGenAttributeCookInstances(input, &first, &error), "first cook accepts");
    CHECK(UsdGenAttributeCookInstances(input, &second, &error), "second cook accepts");
    CHECK(first.digest == second.digest, "the digest is stable");
    CHECK(first.values == second.values && first.keep == second.keep &&
              first.prototype == second.prototype &&
              first.instanceIndices == second.instanceIndices && first.kept == second.kept,
          "repeated cooks are identical");

    map->SetTexel(1, 0, 0, 0, 0.0f);  // one texel moves the bilinear value
    UsdGenAttributeInstanceResult moved;
    CHECK(UsdGenAttributeCookInstances(input, &moved, &error), "moved cook accepts");
    CHECK(moved.digest != first.digest, "one texel flips the digest");

    UsdGenAttributeInstanceInput retuned = input;
    retuned.threshold = 0.9f;
    UsdGenAttributeInstanceResult tuned;
    CHECK(UsdGenAttributeCookInstances(retuned, &tuned, &error), "retuned cook accepts");
    CHECK(tuned.digest != first.digest, "the parameters feed the digest");
    CHECK(tuned.kept == 0 && first.kept == 2, "the threshold culls through the digest");
}

void CheckThreadedEquivalence()
{
    // 40K roots sit above the threading threshold: a 1-worker arena
    // cooks serially while the default arena spreads over workers, and
    // both must agree bitwise on every field -- planes, slot lists in
    // strand order, kept total, and digest. (On a single-core machine
    // both take the serial path; the equivalence still holds.)
    UsdGenAttributeMapSpec spec;
    spec.numFaces = 64;
    spec.resolution = 4;
    spec.channels = 1;
    std::string error;
    auto map = UsdGenAttributeMap::Create(spec, &error);
    CHECK(static_cast<bool>(map), "threaded-equivalence map builds");
    for (int f = 0; f < 64; ++f)
        for (int t = 0; t < 4; ++t)
            for (int s = 0; s < 4; ++s)
                map->SetTexel(f, s, t, 0, float((f + s + t) % 100) / 99.0f);
    UsdGenAttributeInstanceInput input;
    input.map = map;
    input.numPrototypes = 4;
    input.threshold = 0.5f;
    input.defaultValue = 0.25f;
    for (int i = 0; i < 40000; ++i) {
        UsdGenAttributeInstanceRoot root;
        root.face = i % 64;
        root.u = float(i % 101) / 100.0f;
        root.v = float((i * 7) % 101) / 100.0f;
        if (i % 997 == 0) root.face = -1;         // unreadable: drops
        if (i % 991 == 0) root.u = std::nanf("");  // unreadable: drops
        input.roots.push_back(root);
    }
    UsdGenAttributeInstanceResult serial, threaded;
    tbb::task_arena serialArena(1);
    serialArena.execute([&] {
        CHECK(UsdGenAttributeCookInstances(input, &serial, &error),
              "serial cook accepts");
    });
    CHECK(UsdGenAttributeCookInstances(input, &threaded, &error),
          "threaded cook accepts");
    CHECK(serial.kept > 10000 && serial.kept < 40000,
          "the equivalence input keeps a strict subset");
    CHECK(serial.values == threaded.values && serial.keep == threaded.keep &&
              serial.prototype == threaded.prototype &&
              serial.instanceIndices == threaded.instanceIndices &&
              serial.kept == threaded.kept && serial.digest == threaded.digest,
          "serial and threaded cooks agree bitwise");
    // Huge prototype counts stay on the serial path (no chunk-local fan
    // out): the cook still accepts and assigns every kept root in range.
    {
        UsdGenAttributeInstanceInput wide = input;
        wide.numPrototypes = 5000;
        UsdGenAttributeInstanceResult got;
        CHECK(UsdGenAttributeCookInstances(wide, &got, &error),
              "wide-prototype cook accepts");
        CHECK(got.kept == serial.kept, "wide kept total matches");
        CHECK(got.values == serial.values && got.keep == serial.keep,
              "wide planes match");
        for (size_t i = 0; i < got.prototype.size(); ++i) {
            if (got.keep[i])
                CHECK(got.prototype[i] >= 0 && got.prototype[i] < 5000,
                      "wide slots stay in range");
            else
                CHECK(got.prototype[i] == -1, "wide drops mark -1");
        }
    }
}

}  // namespace

int main()
{
    CheckValidation();
    CheckSamplingParity();
    CheckThreshold();
    CheckPrototypes();
    CheckUnreadableRoots();
    CheckDeterminismAndDigest();
    CheckThreadedEquivalence();

    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("testUsdGenAttributeInstance: OK\n");
    return 0;
}
