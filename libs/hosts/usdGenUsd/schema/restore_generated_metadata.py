#!/usr/bin/env python3
"""Copy evaluation metadata from source schema specs into generated specs.

usdGenSchema drops arbitrary property customData, and it also FLATTENS every
inherited property onto each concrete type. A property declared once on an
abstract class (usdGen:enabled on UsdGenOperator, usdGen:mask on
UsdGenStyler) therefore appears on dozens of generated prims with no
customData at all, and any consumer reading the evaluation granularity off a concrete
UsdPrimDefinition silently falls back to "groom".

Walk the source `inherits` chain so a flattened copy is restored from the
class that declares it.
"""
import sys
from pxr import Sdf

source = Sdf.Layer.FindOrOpen(sys.argv[1])
generated = Sdf.Layer.FindOrOpen(sys.argv[2])
if source is None or generated is None:
    raise SystemExit("cannot open schema layers")


def _inherit_paths(prim):
    """Direct base classes of a source prim spec, as Sdf paths."""
    out = []
    items = prim.inheritPathList
    for listop in (items.explicitItems, items.addedItems, items.prependedItems,
                   items.appendedItems, items.orderedItems):
        for path in listop:
            if path not in out:
                out.append(path)
    return out


# evaluation and full doc for every (source prim path, property name).
declared = {}
documented = {}
for prim in source.rootPrims:
    for prop in prim.properties:
        evaluation = prop.customData.get("usdGen", {}).get("evaluation")
        if evaluation:
            declared[(prim.path, prop.name)] = evaluation
        if prop.documentation:
            documented[(prim.path, prop.name)] = prop.documentation

byPath = {prim.path: prim for prim in source.rootPrims}


def _resolve(table, primPath, name, seen):
    """`table` entry for `name` on `primPath`, following inherits; None if absent."""
    if primPath in seen:
        return None
    seen.add(primPath)
    direct = table.get((primPath, name))
    if direct:
        return direct
    prim = byPath.get(primPath)
    if prim is None:
        return None
    for base in _inherit_paths(prim):
        found = _resolve(table, base, name, seen)
        if found:
            return found
    return None


# API schemas are applied, not inherited, so their declarations are not on any
# inherits chain. usdGen auto-applies nothing today, but an applied API schema's
# properties are namespaced, so resolving an unmatched name against the API
# declarations is unambiguous.
apiFallback = {}
docFallback = {}
for (primPath, name), evaluation in declared.items():
    if primPath.name.endswith("API"):
        apiFallback.setdefault(name, evaluation)
for (primPath, name), doc in documented.items():
    if primPath.name.endswith("API"):
        docFallback.setdefault(name, doc)

restored = 0
docs = 0
for prim in generated.rootPrims:
    sourcePrim = byPath.get(prim.path)
    for target in prim.properties:
        evaluation = None
        doc = None
        if sourcePrim is not None:
            evaluation = _resolve(declared, prim.path, target.name, set())
            doc = _resolve(documented, prim.path, target.name, set())
        if not evaluation:
            evaluation = apiFallback.get(target.name)
        if not doc:
            doc = docFallback.get(target.name)
        # usdGenSchema keeps only the first sentence of an authored doc as
        # customData.userDocBrief; restore the whole statement, which carries
        # the units/range and the expression-evaluation level.
        if doc and target.documentation != doc:
            target.documentation = doc
            docs += 1
        if not evaluation:
            continue
        data = dict(target.customData)
        usdgen = dict(data.get("usdGen", {}))
        if usdgen.get("evaluation") == evaluation:
            continue
        usdgen["evaluation"] = evaluation
        data["usdGen"] = usdgen
        target.customData = data
        restored += 1
generated.Save()
print("restored evaluation metadata on %d generated properties" % restored)
print("restored full doc on %d generated properties" % docs)
