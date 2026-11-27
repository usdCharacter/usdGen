// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenValue (step 11.7): UsdGenValue against VtValue.
//   - every row of the value-type table: Sdf's default converts to the row's
//     default (ToEngineValue) and back to Sdf's type and value (ToHostValue),
//     and the engine type name names a type with the same C++ type
//   - the held type is kept: no numeric casting, IsHolding is exact
//   - hash and equality laws for every kind: copies equal and hash equal,
//     +0 == -0, kinds never compare equal across types
//   - host values with no row are boxed and come back unchanged; Boxed
//     payloads (expression outputs) compare by content

#include "usdGen/api/value.h"
#include "usdGenExpr/exprInput.h"
#include "usdGenExpr/expressions/valueShape.h"
#include "usdGenUsd/convert/value.h"

#include "pxr/base/gf/range3f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/sdf/schema.h"
#include "pxr/usd/sdf/valueTypeName.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

void CheckLaws(UsdGenValue const &v, std::string const &what)
{
    UsdGenValue const copy = v;
    UsdGenValue moved = UsdGenValue(copy);
    UsdGenValue const back = std::move(moved);
    Check(copy == v && back == v && !(copy != v), what + ": copies compare equal");
    Check(copy.GetHash() == v.GetHash() && back.GetHash() == v.GetHash(), what + ": copies hash equal");
    Check(copy.Kind() == v.Kind() && copy.ElementKind() == v.ElementKind(), what + ": copies keep the kind");
    Check(moved.IsEmpty(), what + ": moved-from is empty");
    UsdGenValue swapped;
    swapped.Swap(moved);
    swapped = back;
    Check(swapped == v, what + ": assignment");
}

void CheckRows()
{
    size_t rows = 0;
    for (expr::UsdGenValueTypeInfo const &row : expr::AllValueTypes()) {
        std::string const name = row.name;
        PXR_NS::SdfValueTypeName const type = PXR_NS::SdfSchema::GetInstance().FindType(row.name);
        Check(static_cast<bool>(type), name + ": Sdf type");
        Check(row.makeDefault != nullptr, name + ": row has a default");
        if (!type || !row.makeDefault) continue;
        ++rows;
        PXR_NS::VtValue const sdf = type.GetDefaultValue();
        UsdGenValue const ours = expr::ValueTypeDefault(row);
        UsdGenValue const in = usdGenUsd::ToEngineValue(sdf);
        PXR_NS::VtValue const out = usdGenUsd::ToHostValue(in);
        bool const group = ours.Kind() == UsdGenValueKind::Group;
        Check(in == ours || (group && in.Kind() == UsdGenValueKind::Opaque),
              name + ": ToEngineValue(Sdf default) is the row default (" + in.GetTypeName() + " vs " +
                  ours.GetTypeName() + ")");
        Check(out.GetType() == sdf.GetType(), name + ": round trip type " + out.GetTypeName() + " vs " +
                                                  sdf.GetTypeName());
        Check(out == sdf, name + ": round trip value");
        PXR_NS::VtValue const host = usdGenUsd::ToHostValue(ours);
        Check(host.GetType() == sdf.GetType() && host == sdf, name + ": ToHostValue(row default)");
        PXR_NS::SdfValueTypeName const named =
            PXR_NS::SdfSchema::GetInstance().FindType(PXR_NS::TfToken(ours.GetTypeName()));
        Check(static_cast<bool>(named) && named.GetType() == type.GetType() && named.IsArray() == type.IsArray(),
              name + ": GetTypeName " + ours.GetTypeName() + " names the same type");
        Check(ours.IsArrayValued() == type.IsArray(), name + ": array kind");
        Check(!ours.IsArrayValued() || ours.GetArraySize() == 0, name + ": default array is empty");
        CheckLaws(ours, name);
        CheckLaws(in, name + " (converted)");
    }
    Check(rows == expr::AllValueTypes().size(), "every row was checked");
    std::printf("%zu rows round trip\n", rows);
}

void CheckExactTypes()
{
    UsdGenValue const i(1), f(1.0f), d(1.0), u(1u), i64(int64_t(1)), u64(uint64_t(1)), b(true);
    Check(i.IsHolding<int>() && !i.IsHolding<float>() && !i.IsHolding<double>() && !i.IsHolding<unsigned int>(),
          "int holds int only");
    Check(f.IsHolding<float>() && !f.IsHolding<double>(), "float holds float only");
    Check(d.IsHolding<double>() && !d.IsHolding<float>(), "double holds double only");
    Check(u.IsHolding<unsigned int>() && i64.IsHolding<int64_t>() && u64.IsHolding<uint64_t>() && b.IsHolding<bool>(),
          "integer kinds");
    Check(i != f && f != d && i != u && i64 != u64 && i != b, "kinds never compare equal across types");
    Check(i.GetHash() != f.GetHash() && f.GetHash() != d.GetHash(), "the hash covers the kind");
    Check(i.GetWithDefault<float>(7.0f) == 7.0f && i.Get<float>() == 0.0f && i.Get<int>() == 1, "Get falls back");

    // The host conversion keeps the held C++ type.
    Check(usdGenUsd::ToEngineValue(PXR_NS::VtValue(2.5)).IsHolding<double>(), "double stays double");
    Check(usdGenUsd::ToEngineValue(PXR_NS::VtValue(2.5f)).IsHolding<float>(), "float stays float");
    Check(usdGenUsd::ToEngineValue(PXR_NS::VtValue(3)).IsHolding<int>(), "int stays int");
    Check(usdGenUsd::ToEngineValue(PXR_NS::VtValue(std::string("s"))).IsHolding<std::string>(), "string stays string");
    UsdGenValue const tok = usdGenUsd::ToEngineValue(PXR_NS::VtValue(PXR_NS::TfToken("tokenValue")));
    Check(tok.IsHolding<UsdGenToken>() && tok.UncheckedGet<UsdGenToken>() == "tokenValue", "TfToken becomes a token");
    Check(!tok.IsHolding<std::string>(), "a token is not a string");
    PXR_NS::SdfAssetPath const asset("a.tex", "/r/a.tex");
    UsdGenValue const a = usdGenUsd::ToEngineValue(PXR_NS::VtValue(asset));
    Check(a.IsHolding<UsdGenAssetPath>() && !a.IsHolding<std::string>() &&
              a.UncheckedGet<UsdGenAssetPath>().resolved == "/r/a.tex",
          "SdfAssetPath becomes an asset path");
    Check(usdGenUsd::ToHostValue(a) == PXR_NS::VtValue(asset), "asset path round trip");
    PXR_NS::VtArray<PXR_NS::GfVec3f> const points{PXR_NS::GfVec3f(1, 2, 3)};
    UsdGenValue const arr = usdGenUsd::ToEngineValue(PXR_NS::VtValue(points));
    Check(arr.IsHolding<UsdGenVec3fArray>() && arr.GetArraySize() == 1 &&
              static_cast<void const *>(arr.UncheckedGet<UsdGenVec3fArray>().cdata()) ==
                  static_cast<void const *>(points.cdata()),
          "arrays convert zero copy");
    Check(usdGenUsd::ToHostValue(arr) == PXR_NS::VtValue(points), "array round trip");
}

void CheckZeroAndHash()
{
    Check(UsdGenValue(0.0f) == UsdGenValue(-0.0f) && UsdGenValue(0.0f).GetHash() == UsdGenValue(-0.0f).GetHash(),
          "float +0 == -0");
    Check(UsdGenValue(0.0) == UsdGenValue(-0.0) && UsdGenValue(0.0).GetHash() == UsdGenValue(-0.0).GetHash(),
          "double +0 == -0");
    Check(UsdGenValue(UsdGenHalf(0.0f)).GetHash() == UsdGenValue(UsdGenHalf(-0.0f)).GetHash(), "half +0 == -0");
    Check(UsdGenValue(UsdGenVec3f(0.0f, -0.0f, 1.0f)) == UsdGenValue(UsdGenVec3f(-0.0f, 0.0f, 1.0f)) &&
              UsdGenValue(UsdGenVec3f(0.0f, -0.0f, 1.0f)).GetHash() ==
                  UsdGenValue(UsdGenVec3f(-0.0f, 0.0f, 1.0f)).GetHash(),
          "vector +0 == -0");
    UsdGenFloatArray const x{1.0f, -0.0f}, y{1.0f, 0.0f};
    Check(UsdGenValue(x) == UsdGenValue(y) && UsdGenValue(x).GetHash() == UsdGenValue(y).GetHash(),
          "arrays compare by element");
    Check(UsdGenValue(1.0f) != UsdGenValue(2.0f), "different values differ");
    Check(UsdGenValue(std::string("a")) == UsdGenValue("a") && UsdGenValue("a").GetHash() == UsdGenValue(std::string("a")).GetHash(),
          "strings compare by text");
    Check(UsdGenValue(UsdGenMatrix4d(1.0)) == UsdGenValue(UsdGenMatrix4d(1.0)) &&
              UsdGenValue(UsdGenMatrix4d(1.0)) != UsdGenValue(UsdGenMatrix4d(2.0)),
          "heap kinds compare by value");
    Check(UsdGenValue().IsEmpty() && UsdGenValue() == UsdGenValue() && UsdGenValue() != UsdGenValue(0),
          "empty");
    std::vector<std::string> const strings{"a", "b"};
    UsdGenValue const s(strings);
    Check(s.IsHolding<std::vector<std::string>>() && s.GetArraySize() == 2 && s.IsArrayValued(), "string arrays");
    CheckLaws(s, "string[]");
    CheckLaws(UsdGenValue(UsdGenMatrix4d(1.0)), "matrix4d");
    CheckLaws(UsdGenValue(x), "float[]");
}

void CheckBoxed()
{
    PXR_NS::VtValue const range(PXR_NS::GfRange3f(PXR_NS::GfVec3f(0, 0, 0), PXR_NS::GfVec3f(1, 1, 1)));
    UsdGenValue const boxed = usdGenUsd::ToEngineValue(range);
    Check(boxed.Kind() == UsdGenValueKind::Boxed && !boxed.IsEmpty(), "a host value with no row is boxed");
    Check(usdGenUsd::ToHostValue(boxed) == range, "a boxed host value comes back unchanged");
    Check(usdGenUsd::ToEngineValue(range) == boxed &&
              usdGenUsd::ToEngineValue(range).GetHash() == boxed.GetHash(),
          "boxed host values compare and hash as the value");
    CheckLaws(boxed, "boxed host value");

    std::vector<UsdGenExpressionOutputDesc> outputs(1);
    UsdGenValue const o1 = UsdGenExpressionOutputsValue(outputs);
    UsdGenValue const o2 = UsdGenExpressionOutputsValue(outputs);
    Check(o1 == o2 && o1.GetHash() == o2.GetHash(), "expression outputs compare by content");
    Check(UsdGenExpressionOutputsOf(o1) && UsdGenExpressionOutputsOf(o1)->size() == 1 &&
              !UsdGenExpressionSlotsOf(o1),
          "boxed payloads by tag");
    Check(!UsdGenExpressionOutputsOf(UsdGenValue(1)), "a non-boxed value has no payload");
    CheckLaws(o1, "expression outputs");
}

}  // namespace

int main()
{
    CheckRows();
    CheckExactTypes();
    CheckZeroAndHash();
    CheckBoxed();
    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
