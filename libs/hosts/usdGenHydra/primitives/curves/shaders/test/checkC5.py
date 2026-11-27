#!/usr/bin/env python3
"""M1/C5 check (gate L-2, Sdr half). Two parts:

1. Extract the glslfx `parameters`+`textures` (the C5 `inputs:` block,
   plan/07 §2.2) from the three shipped shaders, prove they are
   byte-identical and match docs/freezes/C5.md verbatim. Also pin each
   file's single materialTag to its plan/07 §2.1 value (gate S-5 static half).
2. Load each .glslfx through Sdr (Sdr.Registry().GetShaderNodeFromAsset,
   sourceType "glslfx") and assert it parses, reports exactly the 21
   C5-frozen input names with the C5 defaults, and declares the primvars
   metadata of its variant; then assert the three shader DEFS
   (usdGenShaders/resources/shaders/shaderDefs.usda, discovered via the
   UsdGenShadersDiscoveryPlugin) are registered in the Sdr Registry with
   the same 21 inputs, the `surface` terminal output, and the variant's
   primvar metadata (M-12).

Part 2 needs the plugin search path: run from a shell with
`source bin/_env.sh`. Exits 77 (ctest SKIP) only when OpenUSD's Python
bindings cannot be imported at all; any other failure exits 1.

Usage: python3 checkC5.py [shaderDir] [freezeMd]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHADERS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "resources", "shaders")
FREEZE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "docs", "freezes", "C5.md")

FILES = [
    "usdGenHairPreview.glslfx",
    "usdGenHairPreviewTranslucent.glslfx",
    "usdGenHairPreviewPrimvar.glslfx",
]

# Sdr identifier per file == Shader prim name in shaderDefs.usda (the
# identifier IS the prim name, UsdShadeShaderDefUtils::GetDiscoveryResults);
# file names are lowerCamel, identifiers UpperCamel.
IDENTS = {
    "usdGenHairPreview.glslfx": "UsdGenHairPreview",
    "usdGenHairPreviewTranslucent.glslfx": "UsdGenHairPreviewTranslucent",
    "usdGenHairPreviewPrimvar.glslfx": "UsdGenHairPreviewPrimvar",
}

# Variant A (both usdGenHairPreview files) declares hairT, hairId, st
# attributes; variant B (Primvar) adds hairTangent. C5.md table pins this.
PRIMVARS = {
    "usdGenHairPreview.glslfx": {"hairId", "hairT", "st"},
    "usdGenHairPreviewTranslucent.glslfx": {"hairId", "hairT", "st"},
    "usdGenHairPreviewPrimvar.glslfx": {"hairId", "hairT", "hairTangent", "st"},
}
for _primvars in PRIMVARS.values():
    _primvars.update({"furTauP", "furTauN", "furTipTauP", "furTipTauN"})

# plan/07 §2.1 + C5.md table pin one materialTag per file; §2.7 rule 1 (one
# tag per description) is what keeps gate S-5 (drawBatches == 1) intact — a
# zero-, double-, or wrong-tagged file splits the draw batch as surely as
# mixing materials does. The glslfx parser does not surface the metadata
# block to Sdr node metadata, so this is checked on the raw configuration
# JSON in Part 1.
EXPECTED_TAG = {
    "usdGenHairPreview.glslfx": "defaultMaterialTag",
    "usdGenHairPreviewTranslucent.glslfx": "translucent",
    "usdGenHairPreviewPrimvar.glslfx": "defaultMaterialTag",
}


def tag_check():
    """Gate S-5 static half: each shipped glslfx declares exactly one
    materialTag in its metadata block, with the value §2.1 pins for that file."""
    ok = True
    for f in FILES:
        path = os.path.join(SHADERS, f)
        with open(path) as fh:
            s = fh.read()
        m = re.search(r"^-- configuration\n(\{.*?\n\})\n", s, re.M | re.S)
        assert m, "no configuration block in %s" % path
        cfg_text = m.group(1)
        n = len(re.findall(r'"materialTag"\s*:', cfg_text))
        if n != 1:
            print("FAIL: %s declares %d materialTag entries (want exactly 1)"
                  % (f, n))
            ok = False
            continue
        md = json.loads(cfg_text).get("metadata")
        tag = md.get("materialTag") if isinstance(md, dict) else None
        want = EXPECTED_TAG[f]
        if tag != want:
            print("FAIL: %s materialTag %r (want %r)" % (f, tag, want))
            ok = False
        else:
            print("OK: %s single materialTag %r" % (f, tag))
    return ok


def c5_block(path):
    """Return the verbatim text of the parameters+textures entries.

    The glslfx configuration is a single JSON object between the
    '-- configuration' header line and the '---' body separator. We parse it
    (valid JSON, enforced) and re-emit just the two C5 keys in their original
    order with the original indentation, so the extraction is stable and
    comparable byte-for-byte.
    """
    with open(path) as f:
        s = f.read()
    m = re.search(r"^-- configuration\n(\{.*?\n\})\n", s, re.M | re.S)
    assert m, "no configuration block in %s" % path
    cfg = json.loads(m.group(1))
    assert set(k for k in ("parameters", "textures")) <= set(cfg), \
        "missing C5 keys in %s" % path

    # Re-emit verbatim: slice the original source between the start of the
    # "parameters" entry and the end of the "textures" entry (the next
    # configuration key is "attributes") so whitespace matches the shipped
    # file exactly.
    pstart = s.index('    "parameters": {')
    end = s.index('\n    "attributes": {', pstart)
    return s[pstart:end].rstrip()


def c5_entries(block):
    """(names, defaults) in source order from a parameters+textures block.

    `block` is the C5 text with a trailing comma; wrap it in braces (minus
    the comma) and let json decide the entry lists — the contract is read
    from the FROZEN text, not from the shaders, so a renamed shader input
    cannot make the check vacuous.
    """
    obj = json.loads("{" + block.rstrip().rstrip(",") + "}")
    names = list(obj["parameters"]) + list(obj["textures"])
    defaults = dict(obj["parameters"])
    defaults.update(obj["textures"])
    return names, {k: v.get("default") for k, v in defaults.items()}


def _same_default(val, want):
    """Compare one Sdr default (GfVec*/float from the bindings) against the
    JSON default from the freeze (list or number). Sdr round-trips defaults
    through float32 (float / GfVec3f), so compare at float32 resolution."""
    def close(a, b):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    if isinstance(want, list):
        try:
            got = tuple(val)
        except TypeError:
            return False
        return len(got) == len(want) and all(close(a, b) for a, b in zip(got, want))
    try:
        return close(val, want)
    except (TypeError, ValueError):
        return False


def sdr_check(names, defaults):
    """The L-2 Sdr half. True/False, or None to signal 'cannot run here'."""
    try:
        from pxr import Sdr
    except ImportError as e:
        print("SKIP: OpenUSD python bindings not importable (%s)." % e)
        print("SKIP: run from a shell with 'source bin/_env.sh' (plan/10 §3.6).")
        return None

    reg = Sdr.Registry()
    ok = True

    # (a) Each shipped .glslfx parses directly through SdrGlslfxParserPlugin
    # (sourceType "glslfx"): the 21 C5 inputs with the C5 defaults, plus the
    # primvars of its variant. This path reports no outputs — the parser
    # surfaces parameters+textures as inputs and attributes as primvars
    # metadata only (parserPlugin.cpp ParseShaderNode); the terminal output
    # is checked via the registered defs in (b).
    want = sorted(names)
    for f in FILES:
        path = os.path.abspath(os.path.join(SHADERS, f))
        node = reg.GetShaderNodeFromAsset(path, sourceType="glslfx")
        if not node:
            print("FAIL: Sdr could not parse %s "
                  "(sdrGlslfx parser plugin missing from PXR_PLUGINPATH?)" % f)
            ok = False
            continue
        ins = node.GetShaderInputNames()
        pv = set(str(t) for t in node.GetPrimvars())
        bad = [n for n in ins
               if not _same_default(node.GetShaderInput(n).GetDefaultValue(),
                                    defaults.get(n))]
        if sorted(ins) != want or pv != PRIMVARS[f] or bad:
            print("FAIL: %s: inputs!=C5 %s | primvars %s (want %s) | defaults off %s"
                  % (f, sorted(set(ins) ^ set(want)) or "ok",
                     sorted(pv), sorted(PRIMVARS[f]), bad or "ok"))
            ok = False
            continue
        print("OK: Sdr parsed %s: %d C5 inputs, primvars %s"
              % (f, len(ins), "|".join(sorted(pv))))

    # (b) The three shader DEFS registered via the UsdGenShadersDiscoveryPlugin
    # (plugin/usdGenShaders plugInfo + resources/shaders/shaderDefs.usda):
    # each present in the Registry with the same 21 inputs, the same
    # defaults authored in USD notation, the 1 `surface` terminal output, and
    # the variant's primvar metadata (M-12: without it HdSt never requests
    # hairT/hairId/st from the mesh — materialNetwork.cpp:1136).
    listed = set(str(n.GetName()) for n in reg.GetAllShaderNodes())
    for f in FILES:
        ident = IDENTS[f]
        node = reg.GetShaderNodeByIdentifier(ident)
        if not node:
            print("FAIL: def '%s' not registered in Sdr (plugin discovery "
                  "broken? listed usdGen nodes: %s)"
                  % (ident, sorted(x for x in listed if "Hair" in x) or "none"))
            ok = False
            continue
        ins = node.GetShaderInputNames()
        outs = list(node.GetShaderOutputNames())
        pv = set(str(t) for t in node.GetPrimvars())
        bad = [n for n in ins
               if not _same_default(node.GetShaderInput(n).GetDefaultValue(),
                                    defaults.get(n))]
        if sorted(ins) != want or outs != ["surface"] or pv != PRIMVARS[f] or bad:
            print("FAIL: def %s: inputs!=C5 %s | outputs %s (want ['surface']) "
                  "| primvars %s (want %s) | defaults off %s"
                  % (ident, sorted(set(ins) ^ set(want)) or "ok", outs,
                     sorted(pv), sorted(PRIMVARS[f]), bad or "ok"))
            ok = False
            continue
        print("OK: def %s registered: %d C5 inputs, output %s, primvars %s"
              % (ident, len(ins), outs[0], "|".join(sorted(pv))))
    return ok


def main():
    blocks = {}
    for f in FILES:
        p = os.path.join(SHADERS, f)
        blocks[f] = c5_block(p)

    ref = blocks[FILES[0]]
    ok = True
    for f in FILES[1:]:
        if blocks[f] != ref:
            print("FAIL: C5 block of %s differs from %s" % (f, FILES[0]))
            import difflib
            for line in difflib.unified_diff(
                    ref.splitlines(), blocks[f].splitlines(),
                    FILES[0], f, lineterm=""):
                print(line)
            ok = False
        else:
            print("OK: %s C5 block identical to %s" % (f, FILES[0]))

    # Gate S-5 static half (plan/07 §2.1 + §2.7 rule 1): one materialTag per
    # file, with the value the table pins for that file.
    ok = ok and tag_check()

    n_params = ref.count('"documentation"')  # one per entry: 20 params + 1 texture
    print("C5 entries (doc'd):", n_params, "(expect 21: 20 parameters + 1 texture)")
    ok = ok and n_params == 21

    # Cross-check against the frozen block in docs/freezes/C5.md.
    with open(FREEZE) as f:
        freeze = f.read()
    fm = re.search(r"^```\n(    \"parameters\": \{.*?^    \},\n)```\n", freeze, re.M | re.S)
    if not fm:
        print("FAIL: could not locate the fenced C5 block in %s" % FREEZE)
        return 1
    frozen = fm.group(1).rstrip("\n")
    live = ref.rstrip("\n") + "\n"
    if frozen.rstrip("\n") == live.rstrip("\n"):
        print("OK: docs/freezes/C5.md block identical to shipped files")
    else:
        print("FAIL: docs/freezes/C5.md block differs from shipped files")
        import difflib
        for line in difflib.unified_diff(
                frozen.splitlines(), live.splitlines(),
                "docs/freezes/C5.md", "shipped", lineterm=""):
            print(line)
        ok = False

    # Part 2: the Sdr half (L-2). Only meaningful if the byte contract held.
    if ok:
        names, defaults = c5_entries(frozen)
        print("C5 frozen names (%d):" % len(names), " ".join(names))
        if len(names) != 21:
            print("FAIL: freeze block names %d != 21" % len(names))
            ok = False
        else:
            res = sdr_check(names, defaults)
            if res is None:
                print("C5 CHECK: PARTIAL — byte contract PASS, Sdr half SKIPPED (no pxr)")
                return 77
            ok = ok and res

    print("C5 CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
