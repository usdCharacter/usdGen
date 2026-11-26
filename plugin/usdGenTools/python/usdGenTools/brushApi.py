# The brush tool's link to the engine: a ctypes binding to the usdGenImaging
# attribute-brush C ABI (libs/hosts/usdGenHydra/usdGenImaging/usdGenBrushApi.h).
#
# The mesh pick, the dab kernel with its footprint spill, the cross-face
# smooth, corner extraction, preview colours and the viewport overlay all run
# in C++ behind opaque handles; brushMap / brushPick / brushPreview /
# brushAuthor call through here when the library loads and fall back to their
# pure-Python paths when it does not (the Qt-free T1 suite runs either way).
# A snapshot's faceMask (a GeomSubset binding's parent-mesh faces) goes into
# the mesh handle (UsdGenBrush_MeshCreateMasked), so pick, spill and smooth
# honour it natively exactly as the Python twins do.
#
# The loader mirrors exprApi.py: USDGEN_IMAGING_LIBRARY, then the build tree
# the package was staged from (<build>, <build>/lib, <build>/bin), then the
# platform loader. The library must report UsdGenBrush_ApiVersion() == 1.
#
# Setting USDGEN_BRUSH_NATIVE=0 forces the pure-Python fallback (tests use it
# to exercise both paths in one process via reset()).
#
# Qt-free and pxr-free; numpy is required only by the array helpers.

import ctypes
import os

try:
    from . import exprApi as _exprApi
except ImportError:  # file-path test load
    import exprApi as _exprApi

EXPECTED_API_VERSION = 1

MODE_CODES = {"set": 0, "add": 1, "smooth": 2, "erase": 3}
FALLOFF_CODES = {"constant": 0, "linear": 1, "smooth": 2}
COLORMAP_CODES = {"heat": 0, "gray": 1}

_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_double_p = ctypes.POINTER(ctypes.c_double)

_state = {"loaded": False, "lib": None, "reason": "", "path": ""}


def _bind(lib):
    def sig(name, restype, argtypes):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes

    vp = ctypes.c_void_p
    sig("UsdGenBrush_ApiVersion", ctypes.c_int, [])
    sig("UsdGenBrush_MeshCreate", vp,
        [_c_double_p, ctypes.c_int, _c_int_p, ctypes.c_int])
    sig("UsdGenBrush_MeshCreateMasked", vp,
        [_c_double_p, ctypes.c_int, _c_int_p, ctypes.c_int, _c_int_p,
         ctypes.c_int])
    sig("UsdGenBrush_MeshDestroy", None, [vp])
    sig("UsdGenBrush_MeshFaceCount", ctypes.c_int, [vp])
    sig("UsdGenBrush_MeshFaceInMask", ctypes.c_int, [vp, ctypes.c_int])
    sig("UsdGenBrush_MeshPick", ctypes.c_int,
        [vp, _c_double_p, _c_double_p, _c_int_p, _c_float_p, _c_float_p,
         _c_double_p])
    sig("UsdGenBrush_MeshFootprint", ctypes.c_int,
        [vp, ctypes.c_int, ctypes.c_float, ctypes.c_float, _c_double_p,
         ctypes.c_float, _c_int_p, _c_float_p, _c_float_p, _c_float_p,
         ctypes.c_int])
    sig("UsdGenBrush_MeshFaceEdgeLen", ctypes.c_float, [vp, ctypes.c_int])
    sig("UsdGenBrush_MeshSuggestResolution", ctypes.c_int,
        [vp, ctypes.c_int, ctypes.c_char_p, ctypes.c_int])
    sig("UsdGenBrush_StrokeCreate", vp,
        [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, _c_float_p,
         ctypes.c_float])
    sig("UsdGenBrush_StrokeDestroy", None, [vp])
    sig("UsdGenBrush_StrokeDab", ctypes.c_int,
        [vp, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
         ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float])
    sig("UsdGenBrush_StrokeTakeTouched", ctypes.c_int,
        [vp, _c_int_p, ctypes.c_int])
    sig("UsdGenBrush_StrokeDabCount", ctypes.c_int, [vp])
    sig("UsdGenBrush_StrokeFaceCount", ctypes.c_int, [vp])
    sig("UsdGenBrush_StrokeChannels", ctypes.c_int, [vp])
    sig("UsdGenBrush_StrokeWorkingCorners", ctypes.c_int, [vp, _c_float_p])
    sig("UsdGenBrush_StrokeCommitCorners", ctypes.c_int, [vp, _c_float_p])
    sig("UsdGenBrush_StrokeAbort", None, [vp])
    sig("UsdGenBrush_StrokePreviewColors", ctypes.c_int,
        [vp, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
         _c_float_p])
    sig("UsdGenBrush_PreviewSet", ctypes.c_int,
        [ctypes.c_char_p, _c_float_p, ctypes.c_int])
    sig("UsdGenBrush_PreviewClear", ctypes.c_int, [ctypes.c_char_p])
    sig("UsdGenBrush_PreviewIndexCount", ctypes.c_int, [])
    sig("UsdGenBrush_PreviewClearAll", ctypes.c_int, [])
    sig("UsdGenBrush_GroomCookCount", ctypes.c_longlong, [])
    sig("UsdGenBrush_GroomPublishCount", ctypes.c_longlong, [])
    sig("UsdGenBrush_GroomCurveStats", ctypes.c_int,
        [ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong),
         ctypes.POINTER(ctypes.c_double)])


def load():
    """The bound library, or None (see reason()). Loads once per process."""
    if _state["loaded"]:
        return _state["lib"]
    _state["loaded"] = True
    if os.environ.get("USDGEN_BRUSH_NATIVE", "1") in ("0", "false", "off"):
        _state["reason"] = "USDGEN_BRUSH_NATIVE=0 forces the Python path"
        return None
    try:
        import numpy  # noqa: F401  (every array helper needs it)
    except ImportError:
        _state["reason"] = "numpy is unavailable"
        return None
    handle, description = _exprApi._loadLibrary()
    if handle is None:
        _state["reason"] = "usdGenImaging could not be loaded (%s)" % description
        return None
    try:
        _bind(handle)
    except AttributeError as exc:
        _state["reason"] = ("%s does not export the brush ABI (%s)"
                            % (description, exc))
        return None
    version = handle.UsdGenBrush_ApiVersion()
    if version != EXPECTED_API_VERSION:
        _state["reason"] = ("%s speaks brush ABI %d, this plugin speaks %d"
                            % (description, version, EXPECTED_API_VERSION))
        return None
    _state["lib"] = handle
    _state["path"] = description
    return handle


def available():
    return load() is not None


def reason():
    load()
    return _state["reason"]


def reset(native=None):
    """Forget the load (tests): native False forces the Python path."""
    _state.update({"loaded": False, "lib": None, "reason": "", "path": ""})
    if native is False:
        os.environ["USDGEN_BRUSH_NATIVE"] = "0"
    elif native is True:
        os.environ.pop("USDGEN_BRUSH_NATIVE", None)


# -- array helpers (numpy) ------------------------------------------------

def _ptr(array, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def floatPtr(array):
    return _ptr(array, ctypes.c_float)


def doublePtr(array):
    return _ptr(array, ctypes.c_double)


def intPtr(array):
    return _ptr(array, ctypes.c_int)


# -- handle wrappers ---------------------------------------------------------

class NativeMesh(object):
    """A C++ mesh handle over a brushPick.MeshSnapshot (points + quads)."""

    def __init__(self, lib, handle, faceCount):
        self._lib = lib
        self.handle = handle
        self.faceCount = faceCount

    @staticmethod
    def create(points, faces, faceMask=None):
        """NativeMesh or None. points: [(x,y,z)] or (N,3); faces: [(4 ids)].

        faceMask: the paintable parent-mesh face ids (a GeomSubset's), or
        None for every face (UsdGenBrush_MeshCreateMasked)."""
        lib = load()
        if lib is None:
            return None
        import numpy as np
        try:
            pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64)
                                       .reshape((-1, 3)))
            idx = np.ascontiguousarray(np.asarray(faces, dtype=np.int32)
                                       .reshape((-1, 4)))
            mask = None
            if faceMask is not None:
                # One spare slot: an empty mask must still pass a non-NULL
                # pointer (NULL means "no mask").
                ids = sorted(int(f) for f in faceMask)
                mask = np.ascontiguousarray(
                    np.asarray(ids + [0], dtype=np.int32))
        except (TypeError, ValueError):
            return None
        if len(idx) == 0 or len(pts) == 0:
            return None
        if mask is None:
            handle = lib.UsdGenBrush_MeshCreate(doublePtr(pts), len(pts),
                                                intPtr(idx), len(idx))
        else:
            handle = lib.UsdGenBrush_MeshCreateMasked(
                doublePtr(pts), len(pts), intPtr(idx), len(idx),
                intPtr(mask), len(mask) - 1)
        if not handle:
            return None
        return NativeMesh(lib, handle, len(idx))

    def close(self):
        if self.handle:
            self._lib.UsdGenBrush_MeshDestroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def pick(self, origin, direction):
        """(face, u, v, point) or None."""
        o = (ctypes.c_double * 3)(*[float(x) for x in origin])
        d = (ctypes.c_double * 3)(*[float(x) for x in direction])
        face = ctypes.c_int(-1)
        u = ctypes.c_float(0.0)
        v = ctypes.c_float(0.0)
        p = (ctypes.c_double * 3)()
        got = self._lib.UsdGenBrush_MeshPick(
            self.handle, o, d, ctypes.byref(face), ctypes.byref(u),
            ctypes.byref(v), p)
        if got != 1:
            return None
        return (face.value, float(u.value), float(v.value),
                (p[0], p[1], p[2]))

    def footprint(self, face, u, v, point, worldRadius, cap=4096):
        """[(face, u, v, radiusUV)], primary first; [] on error.

        The C call returns the TOTAL footprint size; a result larger than
        `cap` is re-queried once with a buffer that fits it."""
        pt = None
        if point is not None:
            pt = (ctypes.c_double * 3)(*[float(x) for x in point])
        for _attempt in range(2):
            faces = (ctypes.c_int * cap)()
            us = (ctypes.c_float * cap)()
            vs = (ctypes.c_float * cap)()
            rs = (ctypes.c_float * cap)()
            n = self._lib.UsdGenBrush_MeshFootprint(
                self.handle, int(face), float(u), float(v), pt,
                float(worldRadius), faces, us, vs, rs, cap)
            if n <= cap:
                break
            cap = int(n)
        if n <= 0:
            return []
        n = min(n, cap)
        return [(faces[i], float(us[i]), float(vs[i]), float(rs[i]))
                for i in range(n)]

    def faceEdgeLen(self, face):
        return float(self._lib.UsdGenBrush_MeshFaceEdgeLen(self.handle,
                                                           int(face)))

    def faceInMask(self, face):
        """True when the mask lets a stroke paint `face`."""
        return self._lib.UsdGenBrush_MeshFaceInMask(self.handle,
                                                    int(face)) == 1

    def suggestResolution(self, budgetTexels):
        buf = ctypes.create_string_buffer(256)
        res = self._lib.UsdGenBrush_MeshSuggestResolution(
            self.handle, int(budgetTexels), buf, 256)
        return (res, buf.value.decode("utf-8", "replace"))


def meshFor(snapshot):
    """The snapshot's cached NativeMesh (built lazily), or None."""
    if snapshot is None:
        return None
    try:
        cached = getattr(snapshot, "nativeMesh", None)
    except Exception:
        return None
    if cached is False:  # tried before and failed: do not retry
        return None
    if cached is not None:
        return cached if cached.handle else None
    if load() is None:
        return None
    try:
        mesh = NativeMesh.create(snapshot.points, snapshot.faces,
                                 getattr(snapshot, "faceMask", None))
    except Exception:
        mesh = None
    try:
        snapshot.nativeMesh = mesh if mesh is not None else False
    except Exception:
        pass
    return mesh


def previewSet(primPath, rgb):
    """Push faceVarying colours ((N,3) float32 array) for primPath."""
    lib = load()
    if lib is None:
        return False
    import numpy as np
    arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32)
                               .reshape((-1, 3)))
    return lib.UsdGenBrush_PreviewSet(str(primPath).encode("utf-8"),
                                      floatPtr(arr), len(arr)) == 1


def previewClear(primPath):
    lib = load()
    if lib is None:
        return False
    return lib.UsdGenBrush_PreviewClear(str(primPath).encode("utf-8")) == 1


def previewClearAll():
    """Drop every overlay in the process. The count removed (0 unloaded)."""
    lib = load()
    if lib is None:
        return 0
    return max(0, int(lib.UsdGenBrush_PreviewClearAll()))


def previewIndexCount():
    lib = load()
    if lib is None:
        return 0
    return int(lib.UsdGenBrush_PreviewIndexCount())


def groomCookCount():
    """Groom scene-index cooks issued in this process (-1 unloaded)."""
    lib = load()
    if lib is None:
        return -1
    return int(lib.UsdGenBrush_GroomCookCount())


def groomPublishCount():
    """Groom generations published into a scene index (-1 unloaded)."""
    lib = load()
    if lib is None:
        return -1
    return int(lib.UsdGenBrush_GroomPublishCount())


def groomCurveStats(groomPath):
    """(curves, totalLength) the groom scene indices publish for the
    description or groom root at `groomPath`, or None when nothing is
    published (or the library is unloaded). totalLength sums every
    strand's control-polygon length, so a length repaint moves it."""
    lib = load()
    if lib is None:
        return None
    curves = ctypes.c_longlong(0)
    length = ctypes.c_double(0.0)
    found = lib.UsdGenBrush_GroomCurveStats(
        str(groomPath).encode("utf-8"), ctypes.byref(curves),
        ctypes.byref(length))
    if found != 1:
        return None
    return (int(curves.value), float(length.value))
