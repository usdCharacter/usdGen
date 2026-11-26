// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenValueTypeTable: the engine's closed value-type table (step 7.4)
// against SdfSchema. For every registered type name and alias:
//   - the table has a row with Sdf's array flag and tuple dimensions
//   - the table's shape equals the host's Sdf decoding, for each count case
//   - the row's default equals Sdf's default; a row without one is a type
//     whose C++ value lives in sdf
// and every row names a registered type; unknown names resolve to nothing.

#include "usdGenUsd/resolve/sdfValueShape.h"
#include "usdGenExpr/expressions/valueShape.h"

#include "pxr/base/tf/token.h"
#include "pxr/base/tf/type.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/sdf/opaqueValue.h"
#include "pxr/usd/sdf/pathExpression.h"
#include "pxr/usd/sdf/schema.h"
#include "pxr/usd/sdf/valueTypeName.h"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen::expr;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

bool SameShape(ValueShape const &a, ValueShape const &b)
{
    return a.scalar == b.scalar && a.elementCount == b.elementCount &&
        a.components == b.components && a.rows == b.rows &&
        a.columns == b.columns && a.isArray == b.isArray;
}

bool SdfOnlyType(SdfValueTypeName const &type)
{
    TfType const t = type.GetScalarType().GetType();
    return t == TfType::Find<SdfAssetPath>() || t == TfType::Find<SdfOpaqueValue>() ||
        t == TfType::Find<SdfPathExpression>();
}

// `name` resolves through SdfSchema (an alias to the type FindType picks),
// so the reference is that type, not the one whose alias list named it.
void CheckName(TfToken const &name, std::set<std::string> *covered)
{
    std::string const n = name.GetString();
    SdfValueTypeName const type = SdfSchema::GetInstance().FindType(name);
    Check(static_cast<bool>(type), n + ": SdfSchema does not resolve it");
    if (!type) return;
    UsdGenValueTypeInfo const info = FindValueType(name);
    if (!info) {
        SdfTupleDimensions const dims = type.GetScalarType().GetDimensions();
        Check(false, n + ": no table row (resolves to " + type.GetAsToken().GetString() +
              ", " + type.GetScalarType().GetType().GetTypeName() + ", dims " +
              std::to_string(dims.size) + ")");
        return;
    }
    covered->insert(info.name);
    Check(FindValueType(n.c_str()).name == info.name, n + ": text and token lookups differ");
    Check(info.isArray == type.IsArray(), n + ": array flag");
    SdfTupleDimensions const dims = type.GetScalarType().GetDimensions();
    Check(info.dimensionCount == dims.size, n + ": dimension count");
    Check(dims.size < 1 || info.dimension0 == dims.d[0], n + ": dimension 0");
    Check(dims.size < 2 || info.dimension1 == dims.d[1], n + ": dimension 1");

    struct Case { uint32_t count; bool known; };
    for (Case const c : {Case{0, false}, Case{0, true}, Case{5, true}, Case{7, false}}) {
        ValueShape const want = ValueShapeFromSdfType(type, c.count, c.known);
        Check(SameShape(ValueShapeFromValueType(info, c.count, c.known), want),
              n + ": shape (count " + std::to_string(c.count) + ")");
        Check(SameShape(ValueShapeFromNativeType(name, c.count, c.known), want),
              n + ": native-type shape (count " + std::to_string(c.count) + ")");
    }

    VtValue const sdfDefault = type.GetDefaultValue();
    if (info.makeDefault) {
        VtValue const ours = ValueTypeDefault(info);
        Check(ours.GetType() == sdfDefault.GetType(), n + ": default type " +
              ours.GetTypeName() + " vs " + sdfDefault.GetTypeName());
        Check(ours == sdfDefault, n + ": default value");
    } else {
        Check(SdfOnlyType(type), n + ": row has no default but the type is not sdf-only");
        Check(ValueTypeDefault(info).IsEmpty(), n + ": sdf-only default is empty");
    }
}

} // namespace

int main()
{
    std::set<std::string> covered;
    size_t names = 0;
    for (SdfValueTypeName const &type : SdfSchema::GetInstance().GetAllTypes()) {
        CheckName(type.GetAsToken(), &covered);
        ++names;
        for (TfToken const &alias : type.GetAliasesAsTokens()) {
            if (alias == type.GetAsToken()) continue;
            CheckName(alias, &covered);
            ++names;
        }
    }

    usdGen::UsdGenSpan<const UsdGenValueTypeInfo> const rows = AllValueTypes();
    for (UsdGenValueTypeInfo const &row : rows) {
        Check(static_cast<bool>(SdfSchema::GetInstance().FindType(row.name)),
              std::string(row.name) + ": row names no registered type");
        Check(covered.count(row.name) == 1,
              std::string(row.name) + ": row not reached from GetAllTypes");
    }

    Check(!FindValueType("notAType"), "unknown name has no row");
    Check(!FindValueType(TfToken()), "empty token has no row");
    Check(!FindValueType(static_cast<char const *>(nullptr)), "null name has no row");
    Check(SameShape(ValueShapeFromNativeType(TfToken("notAType")),
                    ValueShapeFromSdfType(SdfValueTypeName())), "unknown shape");
    Check(ValueTypeDefault(UsdGenValueTypeInfo{}).IsEmpty(), "unknown default is empty");

    std::printf("%zu type names, %zu table rows\n", names, rows.size());
    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
