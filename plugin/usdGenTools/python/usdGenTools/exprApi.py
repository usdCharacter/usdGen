# The editor's link to the engine: a ctypes binding to the usdGenImaging C ABI
# (libs/hosts/usdGenHydra/usdGenImaging/usdGenToolsApi.h), plus the pure-text
# helpers the editor needs for literal controls and error marking.
#
# No Qt is imported here on purpose: everything in this module is testable in a
# plain python process, and the dock keeps its Qt dependencies to itself.
#
# Nothing about the expression language is hard-coded here. The function list,
# the variable table and the compiler all come from the shared library, so the
# editor tracks whatever the engine actually accepts. When the library cannot
# be loaded the editor still works as a plain text editor and says why.

import ctypes
import os
import re
import sys
from collections import namedtuple

# usdGen::expr::Domain, as the C ABI spells it.
DOMAIN_GROOM = 1
DOMAIN_PRIMITIVE = 2
DOMAIN_POINT = 4

DOMAIN_NAMES = ("groom", "primitive", "point")
DOMAIN_CODES = {"groom": DOMAIN_GROOM,
                "primitive": DOMAIN_PRIMITIVE,
                "point": DOMAIN_POINT}

# The C ABI record layouts this module knows how to parse.
EXPECTED_API_VERSION = 2

# Large enough for every diagnostic the frontend produces; the ABI truncates
# rather than failing, so this only bounds how much of a pathological error
# message survives.
_ERROR_BUFFER_BYTES = 4096

Diagnostic = namedtuple("Diagnostic", "line column message")
Variable = namedtuple("Variable", "name scalarType components domains valid doc")
Function = namedtuple("Function", "name signature doc insert category")
Literal = namedtuple("Literal", "start end text value label")

# One control parsed out of a top-level `$name = <value>; # <annotation>`
# statement. Offsets are into the whole source, so a control rewrites exactly
# its own value and leaves every other character of the text alone.
Control = namedtuple(
    "Control",
    "name kind valueStart valueEnd valueText comment commentStart commentEnd "
    "data")


def _libraryNames():
    if sys.platform == "win32":
        return ["usdGenImaging.dll"]
    if sys.platform == "darwin":
        return ["libusdGenImaging.dylib"]
    return ["libusdGenImaging.so"]


def _searchDirectories():
    """Where usdGenImaging may sit, most specific first.

    The package is staged into <build>/python/usdGenTools by CMake and the
    shared library is written to <build> itself, so walking up from this file
    finds the library of the very build that staged this plugin -- which is
    what the launchers put on PATH anyway, but this way an interpreter that
    merely has the package importable also works."""
    dirs = []
    explicit = os.environ.get("USDGEN_IMAGING_LIBRARY")
    if explicit:
        dirs.append(explicit if os.path.isdir(explicit)
                    else os.path.dirname(explicit))
    here = os.path.dirname(os.path.abspath(__file__))
    build = os.path.dirname(os.path.dirname(here))  # <build>/python/usdGenTools
    for candidate in (build, os.path.join(build, "lib"),
                      os.path.join(build, "bin")):
        dirs.append(candidate)
    return [d for d in dirs if d and os.path.isdir(d)]


def _load(path, directory):
    """CDLL(path), with `directory` searchable for its dependencies.

    usdGenImaging imports usdGen and usdGenSchema from beside it, and since
    Python 3.8 a Windows CDLL no longer searches the loaded library's own
    directory for those."""
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        with os.add_dll_directory(directory):
            return ctypes.CDLL(path, winmode=0)
    return ctypes.CDLL(path)


def _loadLibrary():
    """(handle, describingText). handle is None when nothing could be loaded."""
    attempts = []
    for directory in _searchDirectories():
        for name in _libraryNames():
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                continue
            try:
                return _load(path, directory), path
            except OSError as exc:
                attempts.append("%s: %s" % (path, exc))
    # Last resort: whatever the platform loader finds on PATH / LD_LIBRARY_PATH.
    for name in _libraryNames():
        try:
            return ctypes.CDLL(name), name
        except OSError as exc:
            attempts.append("%s: %s" % (name, exc))
    return None, "; ".join(attempts) or "usdGenImaging was not found"


class ExpressionApi(object):
    """The three entry points the editor needs, or a clear reason it has none."""

    def __init__(self):
        self._lib = None
        self._reason = ""
        handle, description = _loadLibrary()
        if handle is None:
            self._reason = "usdGenImaging could not be loaded (%s)" % description
            return
        try:
            self._bind(handle)
        except AttributeError as exc:
            self._reason = ("%s does not export the usdGenTools ABI (%s)"
                            % (description, exc))
            return
        version = handle.UsdGenTools_ApiVersion()
        if version != EXPECTED_API_VERSION:
            self._reason = ("%s speaks usdGenTools ABI %d, this plugin speaks %d"
                            % (description, version, EXPECTED_API_VERSION))
            return
        self._lib = handle
        self._path = description

    def _bind(self, lib):
        lib.UsdGenTools_ApiVersion.restype = ctypes.c_int
        lib.UsdGenTools_ApiVersion.argtypes = []
        lib.UsdGenTools_CompileExpression.restype = ctypes.c_int
        lib.UsdGenTools_CompileExpression.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_int]
        lib.UsdGenTools_ListVariables.restype = ctypes.c_int
        lib.UsdGenTools_ListVariables.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.UsdGenTools_ListFunctions.restype = ctypes.c_int
        lib.UsdGenTools_ListFunctions.argtypes = [ctypes.c_char_p, ctypes.c_int]

    @property
    def available(self):
        return self._lib is not None

    @property
    def unavailableReason(self):
        return self._reason

    @property
    def libraryPath(self):
        return getattr(self, "_path", "")

    def _readList(self, call):
        """Run the ABI's size-then-read protocol for a delimited-text list."""
        needed = call(None, 0)
        if needed <= 1:
            return ""
        buffer = ctypes.create_string_buffer(needed)
        written = call(buffer, needed)
        if written <= 0 or written > needed:
            return ""
        return buffer.value.decode("utf-8", "replace")

    def compile(self, source, domain, components=1):
        """[] when the expression compiles, else the Diagnostics that stopped it.

        Raises RuntimeError when there is no library: the caller decides
        whether that is fatal (it is not, for the editor)."""
        if not self.available:
            raise RuntimeError(self._reason)
        buffer = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        status = self._lib.UsdGenTools_CompileExpression(
            source.encode("utf-8"), int(domain), int(components), buffer,
            _ERROR_BUFFER_BYTES)
        if status == 0:
            return []
        if status == 2:
            return [Diagnostic(0, 0, "the editor asked for an evaluation domain "
                                     "or output width the engine does not accept")]
        return ParseDiagnostics(buffer.value.decode("utf-8", "replace"))

    def variables(self, domain=0):
        if not self.available:
            return []
        text = self._readList(
            lambda buf, n: self._lib.UsdGenTools_ListVariables(int(domain), buf, n))
        found = []
        for line in text.splitlines():
            fields = line.split("\t")
            if len(fields) < 6:
                continue
            name, scalar, components, domains, valid, doc = fields[:6]
            found.append(Variable(name, scalar, int(components or 0),
                                  tuple(d for d in domains.split(",") if d),
                                  valid == "1", doc))
        return found

    def functions(self):
        if not self.available:
            return []
        text = self._readList(
            lambda buf, n: self._lib.UsdGenTools_ListFunctions(buf, n))
        found = []
        for line in text.splitlines():
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            # A library that predates the category field still parses: the
            # browser groups by a name-based classification instead.
            category = fields[4] if len(fields) > 4 and fields[4] else \
                FunctionCategory(fields[0])
            found.append(Function(fields[0], fields[1], fields[2], fields[3],
                                  category))
        return found


_API = None


def GetApi():
    """The process-wide binding. Loading is attempted once."""
    global _API
    if _API is None:
        _API = ExpressionApi()
    return _API


def ParseDiagnostics(text):
    """The ABI's `line\tcolumn\tmessage` records, as Diagnostics."""
    found = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            if line.strip():
                found.append(Diagnostic(0, 0, line.strip()))
            continue
        try:
            found.append(Diagnostic(int(fields[0]), int(fields[1]),
                                    "\t".join(fields[2:])))
        except ValueError:
            found.append(Diagnostic(0, 0, line.strip()))
    return found


# ---- text helpers ------------------------------------------------------

# One pass over the source, matching whole tokens so that digits inside a
# comment, a string, a `$variable` or an identifier are never mistaken for a
# numeric literal of their own.
TOKEN_RE = re.compile(r"""
      (?P<comment>\#[^\n]*)
    | (?P<string>"(?:[^"\\\n]|\\.)*"?)
    | (?P<variable>\$[A-Za-z_][A-Za-z_0-9]*)
    | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)
""", re.VERBOSE)


def OffsetToLineColumn(text, offset):
    """1-based (line, column) of a character offset, clamped to the end."""
    offset = max(0, min(offset, len(text)))
    line = text.count("\n", 0, offset) + 1
    start = text.rfind("\n", 0, offset) + 1
    return line, offset - start + 1


def LineColumnToOffset(text, line, column):
    """Inverse of OffsetToLineColumn, clamped into the text."""
    lines = text.split("\n")
    line = max(1, min(line, len(lines)))
    offset = sum(len(l) + 1 for l in lines[:line - 1])
    return min(len(text), offset + max(0, column - 1))


def _EnclosingCall(text, position):
    """(function name, argument index, has more arguments) for the call that
    encloses `position`, or None when the literal is not inside one."""
    depth = 0
    commas = 0
    index = position - 1
    while index >= 0:
        ch = text[index]
        if ch in ")]":
            depth += 1
        elif ch == "(":
            if depth == 0:
                break
            depth -= 1
        elif ch == "[":
            if depth == 0:
                return None
            depth -= 1
        elif ch == "," and depth == 0:
            commas += 1
        index -= 1
    if index < 0:
        return None
    end = index
    start = end
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        start -= 1
    name = text[start:end]
    if not name or not (name[0].isalpha() or name[0] == "_"):
        return None
    depth = 0
    more = False
    for ch in text[position:]:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            if depth == 0:
                break
            depth -= 1
        elif ch == "," and depth == 0:
            more = True
            break
    return name, commas, more


def _Snippet(text, start, end, before=10, after=6):
    """A short slice of the line around [start, end), ellipsised, with the
    literal itself bracketed so the row says which number it drives.

    More context is kept before the literal than after: what a number means is
    almost always written to its left."""
    lineStart = text.rfind("\n", 0, start) + 1
    lineEnd = text.find("\n", end)
    if lineEnd < 0:
        lineEnd = len(text)
    left = max(lineStart, start - before)
    right = min(lineEnd, end + after)
    snippet = (text[left:start] + "[" + text[start:end] + "]"
               + text[end:right]).strip()
    if left > lineStart:
        snippet = "…" + snippet
    if right < lineEnd:
        snippet = snippet + "…"
    return snippet


def LiteralLabel(text, start, end):
    """A short human label for the literal at [start, end).

    Inside a call this reads like `fit(..., 0.15)`, which is how a user thinks
    about the number; otherwise it falls back to the surrounding text."""
    call = _EnclosingCall(text, start)
    literal = text[start:end]
    if call is None:
        return _Snippet(text, start, end)
    name, index, more = call
    parts = ["..."] * index + [literal]
    if more:
        parts.append("...")
    return "%s(%s)" % (name, ", ".join(parts))


def ScanLiterals(text):
    """Every numeric literal in `text`, in source order.

    Comments, string literals and the digits inside identifiers are skipped."""
    found = []
    for match in TOKEN_RE.finditer(text):
        if match.lastgroup != "number":
            continue
        start, end = match.span()
        body = match.group()
        try:
            value = float(body)
        except ValueError:
            continue
        found.append(Literal(start, end, body, value,
                             LiteralLabel(text, start, end)))
    return found


def FormatLiteral(value, template=""):
    """`value` as source text, keeping the shape of the literal it replaces.

    An integer literal stays an integer only while the value is whole, so
    dragging 2 to 2.5 widens it rather than silently rounding."""
    if float(value).is_integer() and "." not in template and "e" not in template.lower():
        return str(int(round(value)))
    text = "%.6g" % value
    if "." not in text and "e" not in text and "inf" not in text:
        text += ".0"
    return text


def DefaultRange(value):
    """(minimum, maximum) for a literal's slider.

    A number already in [0, 1] is almost always a normalised weight, so it
    keeps that range; anything else gets [0, 2x] around itself so the current
    value sits in the middle and can be doubled or taken to zero."""
    if 0.0 <= value <= 1.0:
        return 0.0, 1.0
    if value > 0.0:
        return 0.0, 2.0 * value
    if value < 0.0:
        return 2.0 * value, 0.0
    return 0.0, 1.0


# ---- function categories (fallback) ------------------------------------

# The C ABI reports a category per function. This mirror exists only so a
# library built before the field existed still groups sensibly; the ABI is the
# authority whenever it speaks.
_CATEGORY_BY_NAME = {}
for _names, _category in (
        ("abs ceil floor round trunc sign exp log log10 pow sqrt cbrt fmod "
         "hypot invert", "Math"),
        ("min max clamp fit mix lerp smoothstep linearstep boxstep gaussstep "
         "remap bias contrast gamma compress expand", "Interpolation"),
        ("sin cos tan asin acos atan atan2 sinh cosh tanh deg rad",
         "Trigonometry"),
        ("noise snoise vnoise cnoise snoise4 vnoise4 cnoise4 turbulence "
         "vturbulence cturbulence fbm vfbm cfbm fbm4 vfbm4 cfbm4 cellnoise "
         "ccellnoise pnoise voronoi svoronoi cvoronoi hash rand", "Noise"),
        ("curve ccurve spline", "Curves"),
        ("dot cross norm length dist up ortho rotate angle comp []", "Vectors"),
        ("hsi midhsi rgbtohsl hsltorgb ctransform", "Color"),
        ("printf", "Strings"),
        ("+ - * / < <= > >= == != ?:", "Operators")):
    for _name in _names.split():
        _CATEGORY_BY_NAME[_name] = _category


def FunctionCategory(name):
    """The group a function belongs in, by name. `Other` when unrecognised."""
    return _CATEGORY_BY_NAME.get(name, "Other")


# ---- the control grammar -----------------------------------------------
#
# SeExpr2's editor drives its controls from the expression text, not from a
# scan for numbers: every TOP-LEVEL statement of the form
#
#     $name = <value>;            # <annotation>
#
# becomes one control, and editing the control rewrites exactly that <value>.
# What follows implements that grammar. It is pure text work -- no Qt, no
# engine -- so the widgets and the tests share one parser.

_ASSIGN_RE = re.compile(r"^(?:[ \t]*\#[^\n]*\n|\s)*(\$[A-Za-z_][A-Za-z_0-9]*)"
                        r"[ \t]*=[ \t]*")
_NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$")
_STRING_RE = re.compile(r'^"(?:[^"\\]|\\.)*"$', re.S)
_CURVE_RE = re.compile(r"^(ccurve|curve)[ \t]*\((.*)\)$", re.S)

# Interpolation codes, as SeExpr's Curve spells them. The code stored with a
# knot governs the segment that ENDS at it, which is what Curve::getValue does.
INTERP_NONE = 0
INTERP_LINEAR = 1
INTERP_SMOOTH = 2
INTERP_SPLINE = 3
INTERP_MONOTONE = 4
INTERP_NAMES = ("none", "linear", "smooth", "spline", "monotone")


def _EndOfString(text, index):
    """Index just past the string literal that opens at `index`."""
    limit = len(text)
    index += 1
    while index < limit:
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == '"':
            return index + 1
        index += 1
    return limit


def _EndOfComment(text, index):
    """Index of the newline ending the comment at `index`, or the text end."""
    stop = text.find("\n", index)
    return len(text) if stop < 0 else stop


def ScanStatements(text):
    """Every top-level statement of `text`, as
    (codeStart, codeEnd, commentStart, commentEnd).

    A statement runs to the next ';' that is not inside brackets, a string or a
    comment. codeEnd excludes any trailing comment; the comment reported with a
    statement is the first one on the same line as its terminator, which is
    where SeExpr's `# 0, 1` annotation lives."""
    found = []
    limit = len(text)
    depth = 0
    start = 0
    codeEnd = -1
    commentStart = commentEnd = -1
    sawCode = False
    index = 0

    def close(end, comment):
        if text[start:end].strip():
            found.append((start, end, comment[0], comment[1]))

    while index < limit:
        char = text[index]
        if char == "#":
            stop = _EndOfComment(text, index)
            # A comment that FOLLOWS code ends that code, and the first such
            # comment is the statement's annotation. Comments BEFORE any code
            # are a header: they belong to the statement but end nothing, so
            # what matters is whether real code has been seen, not whether
            # anything at all has.
            if codeEnd < 0 and sawCode:
                codeEnd = index
                commentStart, commentEnd = index, stop
            index = stop
            continue
        if char == '"':
            sawCode = True
            index = _EndOfString(text, index)
            continue
        if not char.isspace():
            sawCode = True
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ";" and depth == 0:
            end = index if codeEnd < 0 else codeEnd
            comment = (commentStart, commentEnd)
            if comment[0] < 0:
                # The usual spelling: `$x = 0.5; # 0, 1`. Take the comment that
                # follows the terminator on the same line.
                probe = index + 1
                while probe < limit and text[probe] in " \t":
                    probe += 1
                if probe < limit and text[probe] == "#":
                    stop = _EndOfComment(text, probe)
                    comment = (probe, stop)
                    index = stop - 1
            close(end, comment)
            start = index + 1
            codeEnd = -1
            commentStart = commentEnd = -1
            sawCode = False
        index += 1

    close(limit if codeEnd < 0 else codeEnd, (commentStart, commentEnd))
    return found


def SplitArguments(text):
    """`text` split on top-level commas, brackets and strings respected."""
    parts = []
    depth = 0
    start = 0
    index = 0
    limit = len(text)
    while index < limit:
        char = text[index]
        if char == '"':
            index = _EndOfString(text, index)
            continue
        if char == "#":
            index = _EndOfComment(text, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return parts


def _Number(text):
    """float(text) when `text` is a plain numeric literal, else None."""
    text = text.strip()
    if not _NUMBER_RE.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _VectorLiteral(text):
    """The components of a `[a, b, c]` literal, or None."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    values = [_Number(part) for part in SplitArguments(text[1:-1])]
    if len(values) < 2 or any(v is None for v in values):
        return None
    return tuple(values)


def ParseAnnotation(comment):
    """(hints, range) from a control's `# ...` comment.

    hints is the set of bare words in it, lowercased -- `color`, `string`,
    `curve` and so on -- and range is the first two numbers when it carries a
    pair like `# 0, 10`, else None."""
    body = comment.lstrip("#").strip() if comment else ""
    hints = set(re.findall(r"[A-Za-z][A-Za-z_0-9]*", body.lower()))
    numbers = [float(n) for n in
               re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", body)]
    span = (numbers[0], numbers[1]) if len(numbers) >= 2 else None
    return hints, span


def _NumberRange(value, span, isInt):
    """The slider range for a number control.

    The annotation wins. Without one the range is SeExpr's [0, 1], widened
    just far enough to contain the value -- a slider that cannot reach the
    number it is showing would be worse than a wider one."""
    if span is not None:
        low, high = span
    else:
        low, high = min(0.0, value), max(1.0, value)
    if high <= low:
        high = low + 1.0
    return low, high


def _ParseCurve(valueText, isColor):
    """{lookup, knots} for a `curve($t, p, v, i, ...)` value, or None.

    Anything that is not a literal knot list -- a computed position, a missing
    interpolation code -- means there is no control to build, and the
    statement keeps whatever text it has."""
    match = _CURVE_RE.match(valueText.strip())
    if not match:
        return None
    arguments = SplitArguments(match.group(2))
    if len(arguments) < 4 or (len(arguments) - 1) % 3:
        return None
    lookup = arguments[0].strip()
    if not lookup:
        return None
    knots = []
    for index in range(1, len(arguments), 3):
        position = _Number(arguments[index])
        interp = _Number(arguments[index + 2])
        if isColor:
            value = _VectorLiteral(arguments[index + 1])
            if value is not None and len(value) != 3:
                return None
        else:
            value = _Number(arguments[index + 1])
        if position is None or value is None or interp is None:
            return None
        knots.append((position, value, int(interp)))
    return {"lookup": lookup, "knots": sorted(knots, key=lambda k: k[0])}


def ScanControls(text):
    """Every control the text declares, in source order.

    A top-level `$name = <value>` whose value is a literal the editor can drive
    -- a number, a vector, a curve or a string -- yields a Control. Anything
    else (a computed right-hand side, an unparsable curve) yields none, and the
    text keeps it untouched."""
    found = []
    for codeStart, codeEnd, commentStart, commentEnd in ScanStatements(text):
        body = text[codeStart:codeEnd]
        match = _ASSIGN_RE.match(body)
        if not match:
            continue
        valueText = body[match.end():]
        trimmed = valueText.rstrip()
        valueStart = codeStart + match.end()
        valueEnd = valueStart + len(trimmed)
        comment = text[commentStart:commentEnd] if commentStart >= 0 else ""
        hints, span = ParseAnnotation(comment)

        kind, data = None, None
        isColorCurve = trimmed.lstrip().startswith("cc")
        curve = _ParseCurve(trimmed, isColor=isColorCurve)
        vector = None if curve else _VectorLiteral(trimmed)
        number = None if (curve or vector) else _Number(trimmed)
        if curve is not None:
            kind = "ccurve" if isColorCurve else "curve"
            data = curve
        elif vector is not None:
            kind = "color" if "color" in hints else "vector"
            low, high = span if span else (0.0, 1.0)
            data = {"values": vector, "minimum": low, "maximum": high}
        elif number is not None:
            isInt = "." not in trimmed and "e" not in trimmed.lower()
            kind = "int" if isInt else "float"
            low, high = _NumberRange(number, span, isInt)
            data = {"value": number, "minimum": low, "maximum": high}
        elif _STRING_RE.match(trimmed):
            kind = "string"
            data = {"value": DecodeStringLiteral(trimmed)}
        if kind is None:
            continue
        found.append(Control(match.group(1), kind, valueStart, valueEnd,
                             trimmed, comment, commentStart, commentEnd, data))
    return found


def ControlSpans(controls):
    """The (start, end) value spans of `controls`, for excluding the numbers
    a control already drives from the plain-literal fallback."""
    return [(c.valueStart, c.valueEnd) for c in controls]


# ---- writing values back ------------------------------------------------

def DecodeStringLiteral(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.replace('\\"', '"').replace("\\\\", "\\")


def EncodeStringLiteral(value):
    return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')


def FormatNumber(value, isInt=False):
    """A number as expression source, keeping integers integral."""
    if isInt:
        return str(int(round(value)))
    return FormatLiteral(value, "0.0")


def FormatVector(values):
    return "[%s]" % ", ".join(FormatNumber(v) for v in values)


def FormatCurve(function, lookup, knots):
    """`curve($t, p, v, i, ...)` source for a knot list.

    Knots are written in position order, which is the order the engine expects
    and the order the widget keeps them in."""
    parts = [lookup]
    for position, value, interp in sorted(knots, key=lambda k: k[0]):
        parts.append(FormatNumber(position))
        parts.append(FormatVector(value) if isinstance(value, (tuple, list))
                     else FormatNumber(value))
        parts.append(str(int(interp)))
    return "%s(%s)" % (function, ", ".join(parts))


def FormatControlValue(control, value):
    """`value` as the source text that replaces `control`'s current value.

    What `value` is depends on the control's kind: a number, a component
    tuple, a knot list, or a string."""
    if control.kind in ("int", "float"):
        return FormatNumber(value, control.kind == "int")
    if control.kind in ("vector", "color"):
        return FormatVector(value)
    if control.kind in ("curve", "ccurve"):
        return FormatCurve(control.kind, control.data["lookup"], value)
    if control.kind == "string":
        return EncodeStringLiteral(value)
    return control.valueText


def DeclarationFor(name, kind, value, minimum=0.0, maximum=1.0, lookup="$t"):
    """The `$name = <value>; # <annotation>` line the Add Widget dialog
    inserts. One line, no trailing newline."""
    if kind in ("int", "float"):
        isInt = kind == "int"
        text = FormatNumber(value, isInt)
        annotation = "%s, %s" % (FormatNumber(minimum, isInt),
                                 FormatNumber(maximum, isInt))
    elif kind in ("vector", "color"):
        text = FormatVector(value)
        annotation = "color" if kind == "color" else "vector"
    elif kind in ("curve", "ccurve"):
        text = FormatCurve(kind, lookup, value)
        annotation = "ccurve" if kind == "ccurve" else "curve"
    elif kind == "string":
        text = EncodeStringLiteral(value)
        annotation = "string"
    else:
        raise ValueError("unknown control kind %r" % kind)
    return "%s = %s; # %s" % (name, text, annotation)


# ---- curve evaluation (for drawing) -------------------------------------

def _CatmullRom(p0, p1, p2, p3, t):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t +
                  (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                  (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)


def _MonotoneTangents(positions, values):
    """Fritsch-Carlson tangents: a cubic through these never overshoots."""
    count = len(values)
    slopes = []
    for i in range(count - 1):
        span = positions[i + 1] - positions[i]
        slopes.append(0.0 if span <= 0 else (values[i + 1] - values[i]) / span)
    tangents = [0.0] * count
    for i in range(count):
        if i == 0:
            tangents[i] = slopes[0] if slopes else 0.0
        elif i == count - 1:
            tangents[i] = slopes[-1]
        elif slopes[i - 1] * slopes[i] <= 0:
            tangents[i] = 0.0
        else:
            tangents[i] = (slopes[i - 1] + slopes[i]) / 2.0
    for i, slope in enumerate(slopes):
        if slope == 0:
            tangents[i] = tangents[i + 1] = 0.0
            continue
        a, b = tangents[i] / slope, tangents[i + 1] / slope
        scale = a * a + b * b
        if scale > 9.0:
            factor = 3.0 / (scale ** 0.5)
            tangents[i] = factor * a * slope
            tangents[i + 1] = factor * b * slope
    return tangents


def EvaluateCurve(knots, x):
    """The curve's value at `x`, the way SeExpr's Curve reads a knot list.

    The interpolation code stored WITH a knot governs the segment that ends at
    it, and the curve is constant outside the knot range -- both are
    Curve::getValue's behaviour, not choices made here. `knots` is
    [(position, value, interp)] with scalar or 3-component values."""
    if not knots:
        return 0.0
    knots = sorted(knots, key=lambda k: k[0])
    if isinstance(knots[0][1], (tuple, list)):
        return tuple(
            EvaluateCurve([(p, v[c], i) for p, v, i in knots], x)
            for c in range(len(knots[0][1])))
    if x <= knots[0][0]:
        return float(knots[0][1])
    if x >= knots[-1][0]:
        return float(knots[-1][1])
    index = 0
    while index + 1 < len(knots) and knots[index + 1][0] <= x:
        index += 1
    left, right = knots[index], knots[min(index + 1, len(knots) - 1)]
    span = right[0] - left[0]
    t = 0.0 if span <= 0 else (x - left[0]) / span
    interp = right[2]
    if interp <= INTERP_NONE:
        return float(left[1])
    if interp == INTERP_LINEAR:
        return float(left[1] + (right[1] - left[1]) * t)
    if interp == INTERP_SMOOTH:
        smooth = t * t * (3.0 - 2.0 * t)
        return float(left[1] + (right[1] - left[1]) * smooth)
    if interp == INTERP_SPLINE:
        before = knots[max(0, index - 1)][1]
        after = knots[min(index + 2, len(knots) - 1)][1]
        return float(_CatmullRom(before, left[1], right[1], after, t))
    positions = [k[0] for k in knots]
    values = [k[1] for k in knots]
    tangents = _MonotoneTangents(positions, values)
    t2, t3 = t * t, t * t * t
    return float((2 * t3 - 3 * t2 + 1) * values[index] +
                 (t3 - 2 * t2 + t) * span * tangents[index] +
                 (-2 * t3 + 3 * t2) * values[index + 1] +
                 (t3 - t2) * span * tangents[index + 1])
