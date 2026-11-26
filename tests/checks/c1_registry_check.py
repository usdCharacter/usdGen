#!/usr/bin/env python3
"""C1 registry check (M1 schema-contract loop, plan/02-schema.md normative).

Loads the CURRENT on-disk schema from libs/hosts/usdGenUsd/schema/resources and
asserts, for every prim type in docs/freezes/C1.md section 1, that
UsdSchemaRegistry resolves it (concrete / abstract / API), and for every
property row in a C1 section-2 table, that the property exists on the right
type with matching base type, uniformity, default and token allow-list.

Exit 0 only if every check passes. One FAIL line per divergence, each
carrying its C1 section reference.

Run:  set -a && . bin/_env.sh && set +a && python3 tests/checks/c1_registry_check.py
The checker itself pins PXR_PLUGINPATH_NAME at the on-disk source plugin
and drops any other usdGenSchema entry (e.g. the build tree) so a stale
generated copy can never shadow the sources under test.
"""

import os
import re
import sys

# --- Pin plugin discovery BEFORE importing pxr -------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_GEN = os.path.dirname(os.path.dirname(_HERE))  # <repo>/tests/checks -> <repo>
_SOURCE_RES = os.path.join(_GEN, "libs", "hosts", "usdGenUsd", "schema", "resources")
_C1 = os.path.join(_GEN, "docs", "freezes", "C1.md")


def _pin_plugin_path():
    entries = []
    for e in os.environ.get("PXR_PLUGINPATH_NAME", "").split(os.pathsep):
        if not e:
            continue
        # Drop any other usdGenSchema provider (build tree) so the on-disk
        # source plugin is the single registration.
        if "usdGenSchema" in e and os.path.abspath(e) != os.path.abspath(_SOURCE_RES):
            continue
        entries.append(e)
    os.environ["PXR_PLUGINPATH_NAME"] = os.pathsep.join([_SOURCE_RES] + entries)


_pin_plugin_path()

try:
    from pxr import Sdf, Usd
except ImportError as exc:
    print("FAIL [setup] cannot import pxr: %s" % exc)
    sys.exit(2)

FAILURES = []


def fail(ref, msg):
    FAILURES.append("FAIL [%s] %s" % (ref, msg))


# --- Parse C1.md ---------------------------------------------------------------
def parse_c1(path):
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    types = {"concrete": [], "abstract": [], "api": []}
    prop_tables = {}  # type name -> (section ref, [row dict])
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^##\s+2\.\s+(.+)$", line)
        if m:
            tname = re.search(r"(UsdGen\w+|v2 stylers)", m.group(1)).group(1)
            rows = []
            i += 1
            # Skip to the header separator row (|---|---|...; the frozen
            # tables use a leading-colon variant `|:---|---|...`).
            while i < len(lines) and not re.match(r"^\s*\|:?[\s:\-|`]+\|\s*$", lines[i]):
                i += 1
            ncols = len([c for c in lines[i].strip().strip("|").split("|") if c.strip()])
            i += 1  # first data row
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                # Split on unescaped pipes only: `\|` inside Default cells
                # (token allow-lists) is content, not a column separator.
                cells = [c.strip() for c in re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))]
                if ncols == 6 and len(cells) == 6:
                    # "On type" table (v2 §2.7.2): retarget the row onto the
                    # named prim type instead of the section heading.
                    rows.append(
                        {
                            "property": cells[0].strip("`"),
                            "type": cells[1].strip("`"),
                            "uniform": cells[2],
                            "default": cells[3],
                            "dirty": cells[5],
                            "line": i + 1,
                            "ontype": cells[4].strip("`"),
                        }
                    )
                elif len(cells) >= 5:
                    rows.append(
                        {
                            "property": cells[0].strip("`"),
                            "type": cells[1].strip("`"),
                            "uniform": cells[2],
                            "default": cells[3],
                            "dirty": cells[4],
                            "line": i + 1,
                            "ontype": None,
                        }
                    )
                i += 1
            prop_tables[tname] = ("C1.md §2 `%s`" % tname, rows)
            continue
        i += 1

    text = "\n".join(lines)
    m = re.search(r"Concrete:\s*((?:`UsdGen\w+`,?\s*)+)", text)
    if m:
        types["concrete"] = re.findall(r"`(UsdGen\w+)`", m.group(1))
    m = re.search(r"Abstract:\s*((?:`UsdGen\w+`,?\s*)+)", text)
    if m:
        types["abstract"] = re.findall(r"`(UsdGen\w+)`", m.group(1))
    # API paragraph spans two lines and embeds `UsdGenOperator` (the
    # auto-apply target, not a schema) — keep only *API names.
    m = re.search(r"API schemas[^`]*((?:`UsdGen\w+`[^\n`]*\n?)+)", text)
    if m:
        types["api"] = [t for t in re.findall(r"`(UsdGen\w+)`", m.group(1)) if t.endswith("API")]
    return types, prop_tables


# --- Registry helpers (concrete -> abstract -> applied-API fallbacks) ---------
def prim_def_for(registry, type_name):
    d = registry.FindConcretePrimDefinition(type_name)
    if d is not None:
        return d, "concrete"
    d = registry.FindAbstractPrimDefinition(type_name)
    if d is not None:
        return d, "abstract"
    d = registry.FindAppliedAPIPrimDefinition(type_name)
    if d is not None:
        return d, "api"
    return None, None


def norm_default(cell):
    """Parse a C1 Default cell -> ('skip',) | ('rel',) | ('value', py)."""
    c = cell.strip()
    if c in ("—", "-", "&mdash;", "") or c.startswith("(inherits"):
        return ("skip",)
    if c == "[]":
        return ("value", [])
    m = re.match(r'^"(.*)"\s*(\(.*\))?\s*$', c)
    if m:
        return ("value", m.group(1))
    if c in ("true", "false"):
        return ("value", c == "true")
    try:
        return ("value", int(c.split()[0]))
    except ValueError:
        pass
    try:
        return ("value", float(c.split()[0]))
    except ValueError:
        pass
    return ("skip",)


def allowed_from_cell(cell):
    m = re.search(r"\(([^()]*)\)", cell)
    if not m:
        return None
    body = m.group(1)
    if "|" not in body and "\\|" not in body:
        return None
    toks = [t.strip().strip("`").strip('"').strip("'") for t in re.split(r"\\?\|\s*", body)]
    toks = [t for t in toks if re.fullmatch(r"[A-Za-z0-9_]+", t or "")]
    return toks or None


def defaults_equal(expected, actual):
    if isinstance(expected, list):
        try:
            return len(actual) == 0
        except TypeError:
            return False
    if actual is None:
        return False
    if isinstance(expected, bool):
        return actual is expected or actual == expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        return actual == expected and isinstance(actual, int)
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return str(actual) == expected
    return False


# --- Main ----------------------------------------------------------------------
def main():
    if not os.path.isfile(_C1):
        print("FAIL [setup] C1.md not found at %s" % _C1)
        return 2
    if not os.path.isfile(os.path.join(_SOURCE_RES, "generatedSchema.usda")):
        print("FAIL [setup] on-disk schema not found under %s" % _SOURCE_RES)
        return 2

    types, prop_tables = parse_c1(_C1)
    registry = Usd.SchemaRegistry()

    n_types_ok = 0
    n_types_total = 0
    for kind in ("concrete", "abstract", "api"):
        for t in types[kind]:
            n_types_total += 1
            ref = "C1.md §1"
            d, found = prim_def_for(registry, t)
            if d is None:
                fail(ref, "%s: not resolved by UsdSchemaRegistry" % t)
                continue
            if kind == "concrete" and not registry.IsConcrete(t):
                fail(ref, "%s: listed concrete but registry says otherwise" % t)
                continue
            if kind == "abstract" and not registry.IsAbstract(t):
                fail(ref, "%s: listed abstract but registry says otherwise" % t)
                continue
            if kind == "api" and "API" not in str(registry.GetSchemaKind(t)):
                fail(ref, "%s: listed API schema but kind=%s"
                    % (t, registry.GetSchemaKind(t)))
                continue
            n_types_ok += 1

    n_props_ok = 0
    n_props_total = 0
    for type_name, (ref, rows) in sorted(prop_tables.items()):
        d, found = prim_def_for(registry, type_name)
        plain = [r for r in rows if not r.get("ontype")]
        if d is None and plain:
            for r in plain:
                n_props_total += 1
                fail(ref, "%s: type not resolved; cannot check %s"
                     % (type_name, r["property"]))
        for r in rows:
            if d is None and not r.get("ontype"):
                continue  # already failed above
            n_props_total += 1
            pname = r["property"]
            rref = "%s (line %d)" % (ref, r["line"])
            check_type = r.get("ontype") or type_name
            dd = prim_def_for(registry, check_type)[0] if r.get("ontype") else d
            if dd is None:
                fail(rref, "%s: type %s not resolved; cannot check %s"
                     % (type_name, check_type, pname))
                continue
            spec = dd.GetSpecType(pname)
            is_rel = r["type"] == "rel"
            if is_rel:
                if spec != Sdf.SpecTypeRelationship:
                    fail(rref, "%s %s: expected relationship, spec=%s"
                         % (type_name, pname, spec))
                    continue
                if dd.GetRelationshipDefinition(pname) is None:
                    fail(rref, "%s %s: relationship missing" % (type_name, pname))
                    continue
                n_props_ok += 1
                continue
            if spec != Sdf.SpecTypeAttribute:
                fail(rref, "%s %s: expected attribute, spec=%s"
                     % (type_name, pname, spec))
                continue
            ad = dd.GetAttributeDefinition(pname)
            if ad is None:
                fail(rref, "%s %s: attribute missing" % (type_name, pname))
                continue
            if str(ad.GetTypeName()) != r["type"]:
                fail(rref, "%s %s: type %s != C1 %s"
                     % (type_name, pname, ad.GetTypeName(), r["type"]))
                continue
            if r["uniform"] == "yes":
                want = Sdf.VariabilityUniform
            elif r["uniform"] == "no":
                want = Sdf.VariabilityVarying
            else:
                want = None
            if want is not None and ad.GetVariability() != want:
                fail(rref, "%s %s: variability %s != C1 uniform=%s"
                     % (type_name, pname, ad.GetVariability(), r["uniform"]))
                continue
            parsed = norm_default(r["default"])
            if len(parsed) == 2:
                if not defaults_equal(parsed[1], ad.GetFallbackValue()):
                    fail(rref, "%s %s: default %r != C1 %s"
                         % (type_name, pname, ad.GetFallbackValue(), r["default"]))
                    continue
            want_toks = allowed_from_cell(r["default"])
            if want_toks is not None:
                got = dd.GetPropertyMetadata(pname, "allowedTokens")
                got_list = [str(t) for t in got] if got else []
                if sorted(got_list) != sorted(want_toks):
                    fail(rref, "%s %s: allowedTokens %s != C1 %s"
                         % (type_name, pname, got_list, want_toks))
                    continue
            n_props_ok += 1

    n_concrete = len(types["concrete"])
    n_abstract = len(types["abstract"])
    print("c1_registry_check: plugin=%s" % _SOURCE_RES)
    print("types OK %d/%d (concrete %d/%d, abstract %d/%d, api %d/%d)"
          % (n_types_ok, n_types_total, sum(1 for t in types["concrete"] if registry.IsConcrete(t)),
             n_concrete, sum(1 for t in types["abstract"] if registry.IsAbstract(t)),
             n_abstract, len(types["api"]), len(types["api"])))
    print("properties OK %d/%d across %d type table(s)"
          % (n_props_ok, n_props_total, len(prop_tables)))
    for f in FAILURES:
        print(f)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
