#!/usr/bin/env python3
"""Checks for the expression/GPU authoring additions to the codeless schema."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "libs", "hosts", "usdGenUsd", "schema", "resources")
entries = [e for e in os.environ.get("PXR_PLUGINPATH_NAME", "").split(os.pathsep)
           if e and "usdGenSchema" not in e]
os.environ["PXR_PLUGINPATH_NAME"] = os.pathsep.join([RES] + entries)

try:
    from pxr import Sdf, Usd
except ImportError as exc:
    print("SKIP [setup] cannot import pxr (OpenUSD python bindings unavailable): %s" % exc)
    raise SystemExit(77)

registry = Usd.SchemaRegistry()
expr = registry.FindConcretePrimDefinition("UsdGenExpression")
if expr is None:
    print("FAIL [expression] UsdGenExpression is not registered")
    raise SystemExit(1)

source = expr.GetPropertyDefinition("usdGen:expr:source")
if source is None:
    print("FAIL [expression] source property is missing")
    raise SystemExit(1)

layer = Sdf.Layer.FindOrOpen(os.path.join(RES, "generatedSchema.usda"))
if layer is None:
    print("FAIL [metadata] generated schema cannot be opened")
    raise SystemExit(1)

def metadata(prim_name, prop_name):
    spec = layer.GetPrimAtPath("/" + prim_name).properties.get(prop_name)
    return spec.customData.get("usdGen", {}).get("evaluation") if spec else None

checks = {
    "UsdGenWidth.usdGen:width": metadata("UsdGenWidth", "usdGen:width"),
    "UsdGenNoise.usdGen:noise:magnitude": metadata("UsdGenNoise", "usdGen:noise:magnitude"),
    "UsdGenOperator.usdGen:enabled": metadata("UsdGenOperator", "usdGen:enabled"),
}
bad = [name for name, value in checks.items() if value not in ("groom", "primitive", "point")]
if bad:
    print("FAIL [metadata] missing evaluation granularity: " + ", ".join(bad))
    raise SystemExit(1)

# Audit every eligible authored property, not only three representative ops.
# Underlying TfType equality includes role aliases (point/normal/color) while
# scalarType also covers numeric array attributes such as ramp knots.
numeric_names = (
    "Bool", "UChar", "Int", "UInt", "Int64", "UInt64", "Half", "Float", "Double",
    "Int2", "Int3", "Int4", "Half2", "Half3", "Half4", "Float2", "Float3", "Float4",
    "Double2", "Double3", "Double4", "Matrix2d", "Matrix3d", "Matrix4d",
    "Quath", "Quatf", "Quatd",
)
numeric_types = [getattr(Sdf.ValueTypeNames, name).type for name in numeric_names]
source_layer = Sdf.Layer.FindOrOpen(os.path.join(ROOT, "libs", "hosts", "usdGenUsd", "schema", "schema.usda"))
if source_layer is None:
    raise SystemExit("FAIL [metadata] source schema cannot be opened")
checked = 0
missing = []
for prim in source_layer.rootPrims:
    for prop in prim.properties:
        if not isinstance(prop, Sdf.AttributeSpec) or prop.typeName.scalarType.type not in numeric_types:
            continue
        expected = prop.customData.get("usdGen", {}).get("evaluation")
        actual = metadata(prim.name, prop.name)
        if expected not in ("groom", "primitive", "point") or actual != expected:
            missing.append(str(prop.path))
        checked += 1
if missing:
    raise SystemExit("FAIL [metadata] numeric attribute granularity missing or changed:\n" + "\n".join(missing))

# Native output attributes remain authored and typed per expression prim; the
# schema must not force a single output type or duplicate it with a token attr.
example = Usd.Stage.Open(os.path.join(ROOT, "plan", "examples", "operator-network.usda"))
if example is None:
    print("FAIL [example] operator-network.usda cannot be opened")
    raise SystemExit(1)
for path, expected in (
    ("/Character/Groom/hair/Expressions/cvCount", Sdf.ValueTypeNames.Int),
    ("/Character/Groom/hair/Expressions/enableFrizz", Sdf.ValueTypeNames.Bool),
    ("/Character/Groom/hair/Expressions/strandWidth", Sdf.ValueTypeNames.Float),
):
    attr = example.GetPrimAtPath(path).GetAttribute("outputs:result")
    if not attr or attr.GetTypeName() != expected:
        print("FAIL [outputs] typed outputs:result missing or mismatched at " + path)
        raise SystemExit(1)
if example.GetPrimAtPath("/Character/Groom/hair/Ops/width").GetAttribute(
        "usdGen:width").GetConnections() != [Sdf.Path(
            "/Character/Groom/hair/Expressions/strandWidth.outputs:result")]:
    print("FAIL [connection] width expression connection is missing")
    raise SystemExit(1)

groom = layer.GetPrimAtPath("/UsdGenGroom")
backend = groom.properties.get("usdGen:execution:backend") if groom else None
if backend is None or backend.typeName != Sdf.ValueTypeNames.Token or backend.default != "cuda":
    print("FAIL [backend] expected uniform token cuda execution backend")
    raise SystemExit(1)

print("expression schema OK; %d numeric metadata defaults and cuda backend present" % checked)
