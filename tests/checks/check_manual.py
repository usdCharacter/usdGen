#!/usr/bin/env python3
"""Check the visual manual against the schema, runtime registry, and its files.

Run from any directory with ``python tests/checks/check_manual.py``. This is a
stdlib-only source check; it does not import pxr or require a render device.

Copyright (c) 2026 Nick Burkard
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import struct
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs" / "site"
OPERATORS = ROOT / "docs" / "reference" / "operators.json"
MEDIA = ROOT / "docs" / "reference" / "media.json"
CHARACTER = ROOT / "docs" / "reference" / "character.json"
UI = ROOT / "docs" / "reference" / "ui.json"
SCHEMA = ROOT / "libs" / "hosts" / "usdGenUsd" / "schema" / "schema.usda"
REGISTRY = ROOT / "libs" / "usdGen" / "usdGen" / "opRegistry.cpp"
ISSUES: list[str] = []
WARNINGS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        ISSUES.append(message)


def schema_operators() -> tuple[set[str], dict[str, set[str]]]:
    source = SCHEMA.read_text(encoding="utf-8")
    # The codeless schema declares abstract bases without a Python class name.
    # Concrete chain nodes inherit directly from one of these bases. Instance
    # is the one concrete node that inherits Operator rather than a role base.
    pattern = re.compile(
        r'^class\s+(?:(UsdGen\w+)\s+)?"(UsdGen\w+)"\s*\((.*?)^\)\s*^\{',
        re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(source))
    operators: set[str] = set()
    parameters: dict[str, set[str]] = {}
    for index, match in enumerate(matches):
        if not match.group(1):
            continue  # abstract schema class
        if not re.search(r'inherits\s*=\s*</UsdGen(?:Operator|Generator|Styler|Deformer)>',
                         match.group(3)):
            continue
        name = match.group(2)
        operators.add(name)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[match.end():end]
        parameters[name] = set(re.findall(
            r'^\s*(?:rel|(?:custom|uniform|varying)\s+)?[A-Za-z][A-Za-z0-9\[\]]*'
            r'\s+(usdGen:[A-Za-z0-9:]+)\b', body, re.MULTILINE))
    check(bool(operators), "could not discover concrete operators in schema.usda")
    return operators, parameters


def registered_operators() -> set[str]:
    source = REGISTRY.read_text(encoding="utf-8")
    operators = set(re.findall(r'Register\(TfToken\("(UsdGen\w+)"\)', source))
    check(bool(operators), "could not discover registered kernels in opRegistry.cpp")
    return operators


def local_path(value: str, base: Path) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or value.startswith("//"):
        return None
    if not parsed.path:
        return None
    path = unquote(parsed.path)
    if path.startswith("/"):
        return SITE / path.lstrip("/")
    return base / path


def check_media_path(value: str, owner: str) -> None:
    check(isinstance(value, str) and bool(value.strip()),
          f"{owner}: missing media path")
    if not isinstance(value, str) or not value.strip():
        return
    path = local_path(value, ROOT)
    check(path is not None, f"{owner}: media must be a local file: {value}")
    if path is None:
        return
    resolved = path.resolve()
    media_dir = (SITE / "media").resolve()
    check(resolved.is_relative_to(media_dir),
          f"{owner}: media path must be in docs/site/media: {value}")
    if not resolved.is_relative_to(media_dir):
        return
    check(resolved.is_file(), f"{owner}: missing media file: {value}")
    if resolved.is_file():
        check(resolved.stat().st_size > 1024,
              f"{owner}: media file is too small to be useful: {value}")


def check_result_png(value: str, owner: str) -> None:
    """Reject a diagram or tiny dummy file as a registered kernel's result."""
    check(value.lower().endswith(".png"),
          f"{owner}: registered kernel needs a primary PNG result")
    path = ROOT / value
    if not path.is_file():
        return
    with path.open("rb") as stream:
        header = stream.read(24)
    check(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR",
          f"{owner}: primary result is not a PNG capture")
    if len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", header[16:24])
        check(width >= 500 and height >= 300,
              f"{owner}: primary result resolution is too small ({width}x{height})")


def check_operators(schema: set[str], runtime: set[str],
                    schema_params: dict[str, set[str]]) -> None:
    if not OPERATORS.is_file():
        ISSUES.append("missing docs/reference/operators.json")
        return
    try:
        document = json.loads(OPERATORS.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        ISSUES.append(f"invalid operators.json: {exc}")
        return
    check(isinstance(document, dict) and
          document.get("copyright") == "Copyright (c) 2026 Nick Burkard" and
          document.get("license") == "SPDX-License-Identifier: MIT",
          "operators.json lacks project copyright/license metadata")
    entries = document.get("operators") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        ISSUES.append("operators.json must contain an operators array")
        return
    ids = [entry.get("id") for entry in entries if isinstance(entry, dict)
           and isinstance(entry.get("id"), str)]
    check(len(ids) == len(entries), "operators array contains a non-object or invalid ID")
    check(len(ids) == len(set(ids)), "operators array has duplicate IDs")
    documented = set(ids)
    check(documented == schema | runtime,
          "operator coverage differs from schema/registry: missing %s; extra %s" %
          (sorted((schema | runtime) - documented), sorted(documented - (schema | runtime))))

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        operator = entry.get("id")
        if not isinstance(operator, str):
            continue
        for field in ("title", "category", "summary", "status"):
            value = entry.get(field)
            check(isinstance(value, str) and bool(value.strip()),
                  f"{operator}: missing {field}")
        summary = entry.get("summary", "")
        workflow = entry.get("workflow", [])
        if isinstance(summary, str):
            check(len(summary.split()) >= 5, f"{operator}: summary is too thin")
        check(isinstance(workflow, list) and bool(workflow),
              f"{operator}: workflow must contain a step")
        if isinstance(workflow, list):
            check(sum(len(str(step).split()) for step in workflow) >= 8,
                  f"{operator}: workflow is too thin")
        if operator in runtime:
            check(entry.get("status") == "implemented",
                  f"{operator}: registered kernel must have implemented status")
        elif operator in schema:
            check(entry.get("status") == "reserved",
                  f"{operator}: schema-only type must have reserved status")
        if operator in runtime:
            prose = " ".join(str(entry.get(field, "")) for field in
                             ("summary", "workflow", "tips", "pitfalls"))
            check(not re.search(r'\b(?:TODO|TBD|lorem ipsum|coming soon|placeholder)\b',
                                prose, re.IGNORECASE),
                  f"{operator}: implemented operator contains placeholder prose")
        params = entry.get("parameters")
        check(isinstance(params, list), f"{operator}: parameters must be an array")
        if isinstance(params, list):
            names = [param.get("name") for param in params if isinstance(param, dict)]
            check(len(names) == len(params),
                  f"{operator}: parameter array contains a non-object")
            check(len(names) == len(set(names)),
                  f"{operator}: duplicate parameter name")
            missing = schema_params.get(operator, set()) - set(names)
            check(not missing, f"{operator}: undocumented schema parameters {sorted(missing)}")
            for param in params:
                if isinstance(param, dict):
                    for field in ("name", "type", "description"):
                        value = param.get(field)
                        check(isinstance(value, str) and bool(value.strip()),
                              f"{operator}: parameter lacks {field}")
                    allowed = param.get("allowed")
                    if allowed is not None:
                        name = param.get("name", "(unnamed)")
                        check(isinstance(allowed, list) and bool(allowed),
                              f"{operator}/{name}: allowed tokens must be a list")
                        if isinstance(allowed, list):
                            check(all(isinstance(token, str) and
                                      (not token or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', token))
                                      for token in allowed),
                                  f"{operator}/{name}: malformed allowed token")
                            default = param.get("default")
                            if isinstance(default, str):
                                try:
                                    unquoted = ast.literal_eval(default)
                                except (SyntaxError, ValueError):
                                    unquoted = default
                                check(unquoted in allowed,
                                      f"{operator}/{name}: default is not an allowed token")
        source = entry.get("source")
        check(isinstance(source, list) and bool(source),
              f"{operator}: source must list source files")
        if isinstance(source, list):
            for file in source:
                check(isinstance(file, str) and (ROOT / file).is_file(),
                      f"{operator}: missing source file {file}")
        examples = entry.get("examples")
        check(isinstance(examples, list), f"{operator}: examples must be an array")
        if isinstance(examples, list):
            for file in examples:
                check(isinstance(file, str) and (ROOT / file).is_file(),
                      f"{operator}: missing example scene {file}")
        check(isinstance(entry.get("inputs"), list),
              f"{operator}: inputs must be an array")
        check(isinstance(entry.get("output"), str) and bool(entry["output"].strip()),
              f"{operator}: missing output description")


def check_media(expected: set[str], runtime: set[str]) -> None:
    if not MEDIA.is_file():
        ISSUES.append("missing docs/reference/media.json")
        return
    try:
        document = json.loads(MEDIA.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        ISSUES.append(f"invalid media.json: {exc}")
        return
    check(isinstance(document, dict) and
          document.get("copyright") == "Copyright (c) 2026 Nick Burkard" and
          document.get("license") == "SPDX-License-Identifier: MIT",
          "media.json lacks project copyright/license metadata")
    operators = document.get("operators") if isinstance(document, dict) else None
    if not isinstance(operators, dict):
        ISSUES.append("media.json must contain an operators object")
        return
    check(set(operators) == expected,
          "media coverage differs from operator inventory: missing %s; extra %s" %
          (sorted(expected - set(operators)), sorted(set(operators) - expected)))
    check(expected - runtime == {"UsdGenInstance"},
          f"unexpected schema-only operator types: {sorted(expected - runtime)}")
    native_fixtures = {"UsdGenPart", "UsdGenWidthBlend", "UsdGenFreeze"}
    hashes: dict[str, list[str]] = {}
    for operator, media in operators.items():
        check(isinstance(media, dict), f"{operator}: media record must be an object")
        if not isinstance(media, dict):
            continue
        if operator in runtime:
            check(media.get("status") not in ("diagram", "unavailable"),
                  f"{operator}: registered kernel cannot use a primary diagram")
            check(media.get("status") == ("native fixture" if operator in native_fixtures else None),
                  f"{operator}: unexpected primary media status")
            check("MoonRay" in str(media.get("renderer", "")),
                  f"{operator}: primary result is not credited as a MoonRay capture")
            image = media.get("image")
            if isinstance(image, str):
                check_result_png(image, operator)
        elif operator == "UsdGenInstance":
            check(media.get("status") == "diagram",
                  "UsdGenInstance: reserved type needs an explicit source diagram")
        if operator == "UsdGenDeform":
            check(media.get("frame") == 24 and
                  media.get("scene") == "examples/rbf-guides-plane.usda" and
                  media.get("generator") == "tools/docs/render_pathtraced.py" and
                  "108,000 of 120,000 CVs" in str(media.get("provenance", "")),
                  "UsdGenDeform: missing verified time-24 native bake provenance")
        if media.get("status") == "unavailable":
            check(operator == "UsdGenInstance",
                  f"{operator}: registered operators require an image")
            check(isinstance(media.get("provenance"), str) and
                  bool(media["provenance"].strip()),
                  f"{operator}: unavailable media needs an explanation")
            continue
        check_media_path(media.get("image"), operator)
        image = media.get("image")
        if isinstance(image, str) and (ROOT / image).is_file():
            digest = hashlib.sha256((ROOT / image).read_bytes()).hexdigest()
            hashes.setdefault(digest, []).append(operator)
            check(media.get("imageSha256") == digest,
                  f"{operator}: imageSha256 does not match the published image")
        animation = media.get("animation")
        for variant in media.get("variants", []):
            label = f'{operator}: {variant.get("label", "variant")}'
            for field in ("image", "animation", "scene"):
                path = variant.get(field)
                if field == "scene":
                    check(isinstance(path, str) and (ROOT / path).is_file(),
                          f"{label}: missing practice scene")
                else:
                    check_media_path(path, label)
                if isinstance(path, str) and (ROOT / path).is_file():
                    digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                    check(variant.get(field + "Sha256") == digest,
                          f"{label}: {field}Sha256 does not match the published file")
            for field in ("label", "caption", "renderer", "provenance", "animationProvenance"):
                check(isinstance(variant.get(field), str) and bool(variant[field].strip()),
                      f"{label}: media variant lacks {field}")
            if isinstance(variant.get("image"), str):
                check_result_png(variant["image"], label)
        if animation:
            check_media_path(animation, operator)
            provenance = media.get("animationProvenance")
            check(isinstance(provenance, str) and bool(provenance.strip()),
                  f"{operator}: animation lacks capture provenance")
        for field in ("caption", "renderer", "provenance"):
            value = media.get(field)
            check(isinstance(value, str) and bool(value.strip()),
                  f"{operator}: media record lacks {field}")
        diagram = media.get("status") == "diagram"
        if diagram:
            check(str(media.get("renderer", "")).lower().find("diagram") >= 0,
                  f"{operator}: diagram must identify itself as a diagram")
            check(str(media.get("image", "")).lower().endswith(".svg"),
                  f"{operator}: source diagram should use an SVG image")
            source = media.get("source")
            source_files = [source] if isinstance(source, str) else source
            check(isinstance(source_files, list) and bool(source_files),
                  f"{operator}: diagram needs source code citation")
            if isinstance(source_files, list):
                for path in source_files:
                    check(isinstance(path, str) and (ROOT / path).is_file(),
                          f"{operator}: missing diagram source file {path}")
            generator = media.get("generator")
            check(isinstance(generator, str) and (ROOT / generator).is_file(),
                  f"{operator}: missing diagram generator file {generator}")
        else:
            check(str(media.get("image", "")).lower().endswith((".png", ".jpg", ".jpeg", ".webp")),
                  f"{operator}: live capture must be a raster still")
        for field in (() if diagram else ("scene", "generator")):
            value = media.get(field)
            check(isinstance(value, str) and (ROOT / value).is_file(),
                  f"{operator}: missing {field} file {value}")
        scene = media.get("scene")
        if not diagram and isinstance(scene, str) and (ROOT / scene).is_file():
            scene_digest = hashlib.sha256((ROOT / scene).read_bytes()).hexdigest()
            check(media.get("sceneSha256") == scene_digest,
                  f"{operator}: sceneSha256 does not match the authored scene")
        if (not diagram and media.get("status") != "native fixture" and
                isinstance(scene, str) and (ROOT / scene).is_file()):
            check(operator in (ROOT / scene).read_text(encoding="utf-8"),
                  f"{operator}: linked scene does not author this operator")
        if media.get("status") == "native fixture":
            check(operator in native_fixtures,
                  f"{operator}: unexpected native fixture claim")
            check(media.get("generator") == "tools/render_utility_fixtures.py",
                  f"{operator}: native fixture requires the fixture exporter")
            if isinstance(scene, str) and (ROOT / scene).is_file():
                short = re.sub(r'(?<!^)(?=[A-Z])', '-',
                               operator.removeprefix("UsdGen")).lower()
                fixture = (ROOT / scene).parent / f"{short}-fixture.usda"
                check(fixture.is_file(), f"{operator}: cooked fixture layer is missing")
                check(f"@./{short}-fixture.usda@" in
                      (ROOT / scene).read_text(encoding="utf-8"),
                      f"{operator}: studio scene does not include cooked fixture")
                if fixture.is_file():
                    content = fixture.read_text(encoding="utf-8")
                    check('def BasisCurves' in content,
                          f"{operator}: cooked fixture has no curve result")
                    if operator == "UsdGenPart":
                        ids = re.search(r'primvars:partId\s*=\s*\[([^]]+)\]', content)
                        check(ids is not None and
                              {-1, 0, 1} <= set(map(int, re.findall(r'-?\d+', ids.group(1)))),
                              "UsdGenPart: cooked fixture lacks all three partId states")
                    elif operator == "UsdGenWidthBlend":
                        widths = {}
                        for name in ("Width_A", "Width_Blend", "Width_B"):
                            section = re.search(
                                rf'def BasisCurves "{name}"\s*\{{(.*?)(?=\n    def BasisCurves|\n}})',
                                content, re.DOTALL)
                            value = re.search(r'float\[\] widths\s*=\s*\[([0-9.eE+-]+)',
                                              section.group(1)) if section else None
                            if value:
                                widths[name] = float(value.group(1))
                        check(len(widths) == 3 and
                              abs(widths.get("Width_Blend", 0) -
                                  (widths.get("Width_A", 0) + widths.get("Width_B", 0)) / 2)
                              < 1e-6,
                              "UsdGenWidthBlend: cooked widths are not the midpoint result")
                    elif operator == "UsdGenFreeze":
                        check('def BasisCurves "Freeze_Updated_Live_Input"' in content and
                              'def BasisCurves "Freeze_Held_Output"' in content,
                              "UsdGenFreeze: fixture lacks live and held curve sets")
    for names in hashes.values():
        if len(names) > 1:
            WARNINGS.append("identical stills need editorial review: " + ", ".join(names))


class _LocalReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if value and key in ("href", "src", "poster"):
                self.links.append(value)
            if tag == "img" and key == "src" and value:
                self.images.append(value)


def check_site() -> None:
    index = SITE / "index.html"
    check(index.is_file(), "missing docs/site/index.html; build the manual first")
    if not index.is_file():
        return
    for html in SITE.rglob("*.html"):
        parser = _LocalReferences()
        parser.feed(html.read_text(encoding="utf-8"))
        for ref in parser.links:
            target = local_path(ref, html.parent)
            if target is not None:
                check(target.exists(), f"{html.relative_to(ROOT)}: broken local link {ref}")
    try:
        entries = json.loads(OPERATORS.read_text(encoding="utf-8"))["operators"]
    except (OSError, ValueError, KeyError, TypeError):
        return
    try:
        media_records = json.loads(MEDIA.read_text(encoding="utf-8"))["operators"]
    except (OSError, ValueError, KeyError, TypeError):
        media_records = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        short = entry["id"].removeprefix("UsdGen")
        slug = re.sub(r'(?<!^)(?=[A-Z])', '-', short).lower()
        page = SITE / "operators" / f"{slug}.html"
        check(page.is_file(),
              f"missing operator page for {entry['id']}: operators/{slug}.html")
        media = media_records.get(entry["id"], {})
        has_image = isinstance(media, dict) and bool(media.get("image"))
        if page.is_file() and (entry.get("status") == "implemented" or has_image):
            image_name = Path(str(media.get("image", ""))).name
            check(bool(image_name) and image_name in page.read_text(encoding="utf-8"),
                  f"{entry['id']}: generated page does not display its primary result image")
        if entry["id"] == "UsdGenCollide" and page.is_file():
            content = page.read_text(encoding="utf-8")
            visual = content.split('<section id="visual"', 1)[-1].split('</section>', 1)[0]
            primary = re.search(r'<img src="([^"]+)"', visual)
            check(media.get("inlineAnimation") is True and primary is not None and
                  Path(urlsplit(primary.group(1)).path).name == "collide.gif",
                  "UsdGenCollide: animated sphere must be the primary inline visual")
        if entry["id"] == "UsdGenWind" and media.get("variants") and page.is_file():
            content = page.read_text(encoding="utf-8")
            visual = content.split('<section id="visual"', 1)[-1].split('</section>', 1)[0]
            primary_images = [Path(urlsplit(src).path).name for src in
                              re.findall(r'<img src="([^"]+)"', visual)]
            check(primary_images == ["wind-clumped.gif", "wind-unclumped.gif"],
                  "UsdGenWind: both matched GIFs must be primary inline visuals")
            check(media.get("scene") == "examples/docs/operators/wind-clumped.usda" and
                  [variant.get("scene") for variant in media["variants"]] == [
                      "examples/docs/operators/wind-clumped.usda",
                      "examples/docs/operators/wind-unclumped.usda"],
                  "UsdGenWind: primary and variant practice scenes must be the 6,500 pair")
    gallery = SITE / "gallery.html"
    if gallery.is_file():
        unavailable = sum(1 for media in media_records.values()
                          if isinstance(media, dict) and media.get("status") == "unavailable")
        check(gallery.read_text(encoding="utf-8").count('visual--waiting') <= unavailable,
              "gallery still contains placeholder images for implemented operators")
    character = SITE / "character-groom.html"
    if character.is_file():
        content = character.read_text(encoding="utf-8")
        check('visual--waiting' not in content,
              "character groom page still shows placeholder images")
        check("Netflix Animation Studios ALab Copyright 2025 Netflix, Inc. All rights reserved."
              in content, "character groom page lacks the ALab image credit")
    for filename in ("stoat-bare.png", "stoat-guides.png", "stoat-groom.png", "stoat-detail.png"):
        check_media_path(f"docs/site/media/{filename}", "character groom")
    check((SITE / "media" / "ALab-LICENSE.md").is_file(),
          "missing redistributed ALab license beside derived images")


def check_character() -> None:
    if not CHARACTER.is_file():
        ISSUES.append("missing docs/reference/character.json")
        return
    try:
        data = json.loads(CHARACTER.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        ISSUES.append(f"invalid character.json: {exc}")
        return
    if not isinstance(data, dict):
        ISSUES.append("character.json must be an object")
        return
    check(data.get("copyright_notice") ==
          "Netflix Animation Studios ALab Copyright 2025 Netflix, Inc. All rights reserved.",
          "character.json lacks the required ALab image notice")
    check(data.get("license") == "ASWF Digital Assets License v1.1",
          "character.json misstates the ALab asset license")
    for field in ("scene", "bare_scene", "guide_scene"):
        value = data.get(field)
        check(isinstance(value, str) and (ROOT / value).is_file(),
              f"character.json: missing {field} file {value}")
    images = data.get("images")
    check(isinstance(images, dict), "character.json lacks an images object")
    if not isinstance(images, dict):
        return
    for name in ("bare", "guides", "groom", "detail"):
        record = images.get(name)
        check(isinstance(record, dict), f"character.json lacks {name} image metadata")
        if not isinstance(record, dict):
            continue
        for field in ("path", "caption", "renderer", "provenance"):
            value = record.get(field)
            check(isinstance(value, str) and bool(value.strip()),
                  f"character.json {name} image lacks {field}")
        path = record.get("path")
        if isinstance(path, str):
            check_media_path(f"docs/site/{path}", f"character {name}")


def check_ui(authorable: set[str]) -> None:
    if not UI.is_file():
        ISSUES.append("missing docs/reference/ui.json for verified usdview walkthroughs")
        return
    try:
        document = json.loads(UI.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        ISSUES.append(f"invalid ui.json: {exc}")
        return
    check(isinstance(document, dict) and document.get("version") == 1 and
          document.get("copyright") == "Copyright (c) 2026 Nick Burkard" and
          document.get("license") == "SPDX-License-Identifier: MIT",
          "ui.json lacks version or project copyright/license metadata")
    records = document.get("screenshots") if isinstance(document, dict) else None
    if not isinstance(records, list):
        ISSUES.append("ui.json must contain a screenshots array")
        return
    ids = [record["id"] for record in records if isinstance(record, dict)
           and isinstance(record.get("id"), str)]
    check(len(ids) == len(records) and len(ids) == len(set(ids)),
          "ui.json screenshot IDs must be unique strings")
    try:
        operator_entries = json.loads(OPERATORS.read_text(encoding="utf-8"))["operators"]
        parameters = {entry["id"]: {param["name"]: param["type"]
                                    for param in entry["parameters"]}
                      for entry in operator_entries}
    except (OSError, ValueError, KeyError, TypeError):
        parameters = {}

    operator_records: dict[str, dict] = {}
    operator_images: set[str] = set()
    operator_image_hashes: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        screenshot_id = record.get("id")
        check(isinstance(screenshot_id, str) and bool(screenshot_id.strip()),
              "ui.json contains an invalid screenshot ID")
        label = str(screenshot_id)
        for field in ("image", "title", "caption", "scene", "capture", "ui"):
            value = record.get(field)
            check(isinstance(value, str) and bool(value.strip()),
                  f"{label}: UI screenshot lacks {field}")
        image = record.get("image")
        if isinstance(image, str):
            check(image.startswith("docs/site/media/ui/"),
                  f"{label}: UI screenshot must live under docs/site/media/ui")
            check_media_path(image, label)
            check_result_png(image, label)
        scene = record.get("scene")
        check(isinstance(scene, str) and (ROOT / scene).is_file(),
              f"{label}: UI source scene is missing: {scene}")
        prose = " ".join(str(record.get(key, "")) for key in
                         ("title", "caption", "capture", "ui"))
        check(not re.search(r'\b(?:TODO|TBD|coming soon|placeholder|pending)\b',
                            prose, re.IGNORECASE),
              f"{label}: UI record contains placeholder text")
        operator = record.get("operator")
        if not operator:
            continue
        check(isinstance(operator, str) and operator in authorable,
              f"{label}: UI capture is not an authorable registered operator: {operator}")
        if not isinstance(operator, str):
            continue
        check(operator not in operator_records,
              f"{operator}: multiple UI records; expected one verified capture")
        operator_records[operator] = record
        if isinstance(image, str):
            check(image not in operator_images,
                  f"{operator}: operator UI screenshot reuses another operator's image")
            operator_images.add(image)
            if (ROOT / image).is_file():
                digest = hashlib.sha256((ROOT / image).read_bytes()).hexdigest()
                check(digest not in operator_image_hashes,
                      f"{operator}: UI screenshot pixels duplicate another operator")
                operator_image_hashes.add(digest)
        selected = record.get("selectedPrim")
        check(isinstance(selected, str) and selected.startswith("/"),
              f"{operator}: missing selectedPrim in the capture")
        actions = record.get("actions")
        check(isinstance(actions, list) and bool(actions) and
              all(isinstance(action, str) and len(action.split()) >= 3
                  for action in actions),
              f"{operator}: UI actions need concrete verified steps")
        check("Noodles" in str(record.get("ui", "")),
              f"{operator}: UI capture does not identify Noodles")
        edited = record.get("editedField")
        if edited:
            dtype = parameters.get(operator, {}).get(edited)
            supported = {"float", "bool", "int", "token", "uint64", "string",
                         "float2", "vector3f"}
            base = dtype.removeprefix("uniform ").split(" (")[0] if dtype else None
            check(base in supported,
                  f"{operator}: editedField {edited} is missing or not inline-editable")
            if isinstance(actions, list) and all(isinstance(a, str) for a in actions):
                target_steps = [i for i, action in enumerate(actions)
                                if "Layer Editor" in action]
                label = str(record.get("visibleField", edited.removeprefix("usdGen:")))
                edit_steps = [i for i, action in enumerate(actions)
                              if label.lower() in action.lower() and
                              ("click" in action.lower() or "toggle" in action.lower())]
                check(bool(target_steps) and bool(edit_steps) and
                      min(target_steps) < min(edit_steps),
                      f"{operator}: choose a persistent edit layer before changing a value")
                check(any("Ctrl+S" in action for action in actions),
                      f"{operator}: UI steps omit saving the edited layer")
                if base == "bool":
                    edit_text = " ".join(actions[i].lower() for i in edit_steps)
                    check("checkbox" in edit_text or "toggle" in edit_text,
                          f"{operator}: boolean field instructions must use the checkbox")
        else:
            check(record.get("readOnly") is True and
                  isinstance(record.get("reason"), str) and bool(record["reason"].strip()),
                  f"{operator}: no editedField or explicit read-only reason")
    check(set(operator_records) == authorable,
          "Noodles UI captures differ from authorable operators: missing %s; extra %s" %
          (sorted(authorable - set(operator_records)),
           sorted(set(operator_records) - authorable)))
    check(sum(str(value).startswith("pomade-") for value in ids) >= 12,
          "Pomade visual tour needs at least 12 verified usdview states")
    check(sum(str(value).startswith("alab-") for value in ids) >= 7,
          "ALab walkthrough needs inspection and Noodles edit/save screenshots")

    embedded: set[Path] = set()
    for html in SITE.rglob("*.html"):
        parser = _LocalReferences()
        parser.feed(html.read_text(encoding="utf-8"))
        for src in parser.images:
            target = local_path(src, html.parent)
            if target is not None:
                embedded.add(target.resolve())
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("image"), str):
            check((ROOT / record["image"]).resolve() in embedded,
                  f"{record.get('id')}: UI screenshot is not embedded in a manual page")

    for page_name, minimum in (("getting-started.html", 7),
                               ("pomade-workspace.html", 12),
                               ("character-groom.html", 7)):
        page = SITE / page_name
        check(page.is_file(), f"missing artist walkthrough page {page_name}")
        if page.is_file():
            source = page.read_text(encoding="utf-8")
            check(source.count("ui-step") >= minimum and
                  source.count("ui-visual") >= minimum,
                  f"{page_name}: too few illustrated numbered UI steps")
            check("ui-pending" not in source,
                  f"{page_name}: UI walkthrough still has pending screenshots")
    check((SITE / "technical-setup.html").is_file(),
          "manual is missing the technical setup appendix")
    for operator in authorable:
        record = operator_records.get(operator)
        if not record:
            continue
        short = operator.removeprefix("UsdGen")
        slug = re.sub(r'(?<!^)(?=[A-Z])', '-', short).lower()
        page = SITE / "operators" / f"{slug}.html"
        if page.is_file() and isinstance(record.get("image"), str):
            source = page.read_text(encoding="utf-8")
            check('id="in-usdview"' in source and "ui-step" in source and
                  "ui-visual" in source and Path(record["image"]).name in source,
                  f"{operator}: operator page omits its verified Noodles UI step")
            check("ui-pending" not in source,
                  f"{operator}: operator UI step is still pending")
    for operator, state in (("UsdGenWidthBlend", "runtime-only"),
                            ("UsdGenInstance", "unavailable")):
        short = operator.removeprefix("UsdGen")
        slug = re.sub(r'(?<!^)(?=[A-Z])', '-', short).lower()
        page = SITE / "operators" / f"{slug}.html"
        if page.is_file():
            source = page.read_text(encoding="utf-8")
            check(f'data-ui-state="{state}"' in source and "ui-exception" in source,
                  f"{operator}: page must explain its UI authoring exception")


def main() -> int:
    schema, schema_params = schema_operators()
    runtime = registered_operators()
    check_operators(schema, runtime, schema_params)
    check_media(schema | runtime, runtime)
    check_site()
    check_character()
    check_ui(schema & runtime)
    for warning in WARNINGS:
        print(f"WARN: {warning}")
    if ISSUES:
        for issue in ISSUES:
            print(f"FAIL: {issue}")
        print(f"{len(ISSUES)} manual check(s) failed")
        return 1
    print(f"ok: {len(schema)} schema operators, {len(runtime)} registered operators, "
          f"{len(schema | runtime)} documented names; local media and links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
