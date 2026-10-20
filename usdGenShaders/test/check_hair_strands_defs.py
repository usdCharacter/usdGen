#!/usr/bin/env python3
"""Sdr validation of the two the host renderer-parity strand-hair shader defs.

The counterpart of checkC5.py for `UsdGenHairStrands` /
`UsdGenHairStrandsTranslucent` (plan/16-ue-hair-parity.md WS1). Those two are
NOT part of the C5 freeze -- C5 covers the `inputs:` block of the three
`UsdGenHairPreview*` files and nothing else -- so they get their own check
rather than being folded into checkC5.py, which must keep asserting exactly
three files with exactly 21 inputs.

Three parts:

1. The two shipped glslfx files are byte-identical outside their header
   comment and their `materialTag`. The parameters+textures block in
   particular must not drift, because the two defs share one input contract.
2. Each file declares exactly one materialTag, with the value the plan pins
   (gate S-5 static half: a zero-, double- or wrong-tagged file splits the
   draw batch).
3. Sdr parses each file and registers each def (via the
   UsdGenShadersDiscoveryPlugin) with the expected input names, the expected
   defaults, the `surface` terminal output, and the primvars metadata HdSt
   uses to request hairT/hairId/st/furTau* from the mesh
   (pxr/imaging/hdSt/materialNetwork.cpp:1136 -- without it the shader reads
   nothing but displayColor).

Part 3 needs the plugin search path (bin/record_usd.ps1 / bin/_env.sh set it).
Exits 77 (ctest SKIP) only when OpenUSD's Python bindings cannot be imported;
any other failure exits 1.

With --model/--built (ctest passes both) it also checks the build's copy of
the shader directory: that its generated usdGenHairModel.glslfx selects the
configured strand model (USDGEN_HAIR_SHADING_MODEL) and that every other
strand file is identical to the source. CMake copies shaders/ at configure
time, so a source edit without a re-configure fails here instead of rendering
the stale shader.

Usage: python3 check_hair_strands_defs.py [shaderDir] [--model karis|marschner
                                          --built BUILD_SHADER_DIR]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("shaders", nargs="?",
                 default=os.path.join(HERE, "..", "resources", "shaders"))
_ap.add_argument("--model", choices=("karis", "marschner"))
_ap.add_argument("--built")
ARGS = _ap.parse_args()
SHADERS = ARGS.shaders

FILES = [
    "usdGenHairStrands.glslfx",
    "usdGenHairStrandsTranslucent.glslfx",
]

IDENTS = {
    "usdGenHairStrands.glslfx": "UsdGenHairStrands",
    "usdGenHairStrandsTranslucent.glslfx": "UsdGenHairStrandsTranslucent",
}

EXPECTED_TAG = {
    "usdGenHairStrands.glslfx": "defaultMaterialTag",
    "usdGenHairStrandsTranslucent.glslfx": "translucent",
}

# The `attributes` block of both files, i.e. what HdSt is told to bind.
PRIMVARS = {"hairId", "hairT", "st",
            "furTauP", "furTauN", "furTipTauP", "furTipTauN",
            # Local density for the Marschner model's hair occlusion.
            "furDensity",
            # The look, carried as constant primvars so it reaches the shader
            # whatever material is bound. HdSt only asks the prim for a primvar
            # the def names here, so dropping one of these silently reverts
            # that half of the look to the material's Sdr default.
            "hairTipColor", "hairColorRamp", "hairRandomHue", "hairRandomValue"}

# Shader input names and defaults. baseColor
# and tipColor mirror the C1 schema defaults usdGen:look:rootColor/:tipColor,
# because on a usdGen tile the description's look supplies them
# (UsdGenTilePublisher::BuildDefaultMaterialDataSource).
EXPECTED = [
    ("baseColor", [0.035, 0.018, 0.008]),
    ("tipColor", [0.210, 0.115, 0.045]),
    ("colorRamp", 1.0),
    ("useMelanin", 0.0),
    ("melanin", 0.5),
    ("redness", 0.0),
    ("dyeColor", [1.0, 1.0, 1.0]),
    ("roughness", 0.35),
    ("specular", 0.5),
    ("scatter", 0.0),
    ("backlit", 1.0),
    ("opacity", 1.0),
    ("randomHue", 0.0),
    ("randomValue", 0.0),
    ("randomRoughness", 0.0),
    ("selfShadow", 1.0),
    ("hairCoverageScale", 1.0),
    ("minPixelWidth", 1.0),
    # Matching controls (docs/storm-fur.md "Matching an imported look"). Each
    # default selects the pre-existing expression, so these values ARE the
    # "unchanged look" contract: -1 = follow roughness / built-in constant.
    ("primaryRoughness", -1.0),
    ("longitudinalRoughness", -1.0),
    ("azimuthalRoughness", -1.0),
    ("cuticleTilt", -1.0),
    ("backScatter", 1.0),
    ("usePresetColors", 0.0),
    ("presetSrgb", 0.0),
    ("presetColorA1", [0.035, 0.018, 0.008]),
    ("presetColorA2", [0.12, 0.065, 0.026]),
    ("presetColorA3", [0.21, 0.115, 0.045]),
    ("presetPointsA", [0.0, 0.33, 0.66, 1.0]),
    ("presetColorB1", [0.25, 0.16, 0.08]),
    ("presetColorB2", [0.45, 0.32, 0.17]),
    ("presetColorB3", [0.6, 0.45, 0.26]),
    ("presetPointsB", [0.0, 0.33, 0.66, 1.0]),
    ("presetColorC1", [0.01, 0.006, 0.004]),
    ("presetColorC2", [0.01, 0.006, 0.004]),
    ("presetColorC3", [0.01, 0.006, 0.004]),
    ("presetPointsC", [0.0, 0.33, 0.66, 1.0]),
    ("presetWeights", [1.0, 0.0, 0.0, 0.0]),
    ("jitterMode", 0.0),
    ("randomSaturation", 0.0),
    ("hueOffset", 0.0),
    ("saturationOffset", 0.0),
    ("valueOffset", 0.0),
    ("rootBlendRate", 0.0),
    ("indirectMultiplier", 1.0),
    ("ambientFill", 0.0),
    ("rootFade", 0.0),
    # Marschner-model inputs (unused by the Karis lobes).
    ("ior", 1.55),
    ("aoIntensity", 0.0),
    ("baseColorMap", [1.0, 1.0, 1.0]),
]
NAMES = [n for n, _ in EXPECTED]
DEFAULTS = dict(EXPECTED)


def _configuration(path):
    with open(path) as f:
        s = f.read()
    m = re.search(r"^-- configuration\n(\{.*?\n\})\n", s, re.M | re.S)
    assert m, "no configuration block in %s" % path
    return s, m.group(1)


def _inputs_block(path):
    """The verbatim parameters+textures text, as checkC5.py slices it."""
    s, _ = _configuration(path)
    start = s.index('    "parameters": {')
    end = s.index('\n    "attributes": {', start)
    return s[start:end].rstrip()


def block_check():
    ref = _inputs_block(os.path.join(SHADERS, FILES[0]))
    ok = True
    for f in FILES[1:]:
        live = _inputs_block(os.path.join(SHADERS, f))
        if live != ref:
            print("FAIL: inputs block of %s differs from %s" % (f, FILES[0]))
            import difflib
            for line in difflib.unified_diff(ref.splitlines(),
                                             live.splitlines(),
                                             FILES[0], f, lineterm=""):
                print(line)
            ok = False
        else:
            print("OK: %s inputs block identical to %s" % (f, FILES[0]))
    n = ref.count('"documentation"')
    print("inputs (doc'd): %d (expect %d)" % (n, len(NAMES)))
    if n != len(NAMES):
        print("FAIL: %d documented entries, expected %d" % (n, len(NAMES)))
        ok = False
    return ok


def tag_check():
    ok = True
    for f in FILES:
        path = os.path.join(SHADERS, f)
        _, cfg_text = _configuration(path)
        n = len(re.findall(r'"materialTag"\s*:', cfg_text))
        if n != 1:
            print("FAIL: %s declares %d materialTag entries (want 1)" % (f, n))
            ok = False
            continue
        tag = json.loads(cfg_text).get("metadata", {}).get("materialTag")
        want = EXPECTED_TAG[f]
        if tag != want:
            print("FAIL: %s materialTag %r (want %r)" % (f, tag, want))
            ok = False
        else:
            print("OK: %s single materialTag %r" % (f, tag))
    return ok


def _same_default(val, want):
    def close(a, b):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    if isinstance(want, list):
        try:
            got = tuple(val)
        except TypeError:
            return False
        return len(got) == len(want) and all(close(a, b)
                                             for a, b in zip(got, want))
    try:
        return close(val, want)
    except (TypeError, ValueError):
        return False


def sdr_check():
    """True/False, or None when Sdr cannot be reached at all."""
    try:
        from pxr import Sdr
    except ImportError as e:
        print("SKIP: OpenUSD python bindings not importable (%s)." % e)
        return None

    reg = Sdr.Registry()
    ok = True
    want = sorted(NAMES)

    for f in FILES:
        path = os.path.abspath(os.path.join(SHADERS, f))
        node = reg.GetShaderNodeFromAsset(path, sourceType="glslfx")
        if not node:
            print("FAIL: Sdr could not parse %s (sdrGlslfx parser plugin "
                  "missing from PXR_PLUGINPATH?)" % f)
            ok = False
            continue
        ins = node.GetShaderInputNames()
        pv = set(str(t) for t in node.GetPrimvars())
        bad = [n for n in ins
               if not _same_default(node.GetShaderInput(n).GetDefaultValue(),
                                    DEFAULTS.get(n))]
        if sorted(ins) != want or pv != PRIMVARS or bad:
            print("FAIL: %s: input mismatch %s | primvars %s (want %s) | "
                  "defaults off %s"
                  % (f, sorted(set(ins) ^ set(want)) or "ok", sorted(pv),
                     sorted(PRIMVARS), bad or "ok"))
            ok = False
            continue
        print("OK: Sdr parsed %s: %d inputs, primvars %s"
              % (f, len(ins), "|".join(sorted(pv))))

    listed = set(str(n.GetName()) for n in reg.GetAllShaderNodes())
    for f in FILES:
        ident = IDENTS[f]
        node = reg.GetShaderNodeByIdentifier(ident)
        if not node:
            print("FAIL: def '%s' not registered in Sdr (discovery broken? "
                  "usdGen nodes seen: %s)"
                  % (ident, sorted(x for x in listed if "UsdGen" in x) or "none"))
            ok = False
            continue
        ins = node.GetShaderInputNames()
        outs = list(node.GetShaderOutputNames())
        pv = set(str(t) for t in node.GetPrimvars())
        bad = [n for n in ins
               if not _same_default(node.GetShaderInput(n).GetDefaultValue(),
                                    DEFAULTS.get(n))]
        if sorted(ins) != want or outs != ["surface"] or pv != PRIMVARS or bad:
            print("FAIL: def %s: input mismatch %s | outputs %s (want "
                  "['surface']) | primvars %s | defaults off %s"
                  % (ident, sorted(set(ins) ^ set(want)) or "ok", outs,
                     sorted(pv), bad or "ok"))
            ok = False
            continue
        print("OK: def %s registered: %d inputs, output %s, primvars %s"
              % (ident, len(ins), outs[0], "|".join(sorted(pv))))

    # The three C5 defs must still be there: adding these two must not have
    # displaced them from discovery.
    for ident in ("UsdGenHairPreview", "UsdGenHairPreviewTranslucent",
                  "UsdGenHairPreviewPrimvar", "UsdGenValuePreview"):
        if not reg.GetShaderNodeByIdentifier(ident):
            print("FAIL: pre-existing def '%s' is no longer registered" % ident)
            ok = False
    print("OK: the three C5 defs and UsdGenValuePreview are still registered")
    return ok


# The strand files whose build copy must match the source exactly.
STRAND_FILES = FILES + ["usdGenHairStrandsBsdf.glslfx",
                        "usdGenHairStrandsMarschner.glslfx", "shaderDefs.usda"]


def built_check(model, built):
    ok = True
    path = os.path.join(built, "usdGenHairModel.glslfx")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        print("FAIL: no generated model selector (%s)" % e)
        return False
    found = re.findall(r"^#define USDGEN_STRANDS_MODEL_([A-Z]+) 1$", text, re.M)
    if found != [model.upper()]:
        print("FAIL: %s selects %s, the build is configured for %s"
              % (path, found, model))
        ok = False
    else:
        print("OK: build selects the %s strand model" % model)
    for f in STRAND_FILES:
        a = open(os.path.join(SHADERS, f), "rb").read()
        try:
            b = open(os.path.join(built, f), "rb").read()
        except OSError:
            b = None
        if a != b:
            print("FAIL: build copy of %s is stale or missing -- re-run CMake "
                  "(shaders/ is copied at configure time)" % f)
            ok = False
    if ok:
        print("OK: build copies of %d strand files match the source"
              % len(STRAND_FILES))
    return ok


def main():
    ok = block_check()
    ok = tag_check() and ok
    if ARGS.model and ARGS.built:
        ok = built_check(ARGS.model, ARGS.built) and ok
    if not ok:
        print("HAIR STRANDS DEF CHECK: FAIL")
        return 1
    res = sdr_check()
    if res is None:
        print("HAIR STRANDS DEF CHECK: PARTIAL -- byte contract PASS, "
              "Sdr half SKIPPED (no pxr)")
        return 77
    print("HAIR STRANDS DEF CHECK:", "PASS" if res else "FAIL")
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
