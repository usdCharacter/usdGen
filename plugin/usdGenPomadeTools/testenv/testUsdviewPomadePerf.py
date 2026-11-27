# testUsdviewPomadePerf -- T3 STUB: the TN-1/TN-2 probes on the fixture,
# with the reference-scene gates (TN-4/TN-7) as skips (plan/17 S7).
# TN-3 lives in testUsdviewPomadePick.py (P6 exit).
#
#   testusdview --testScript plugin/usdGenPomadeTools/testenv/testUsdviewPomadePerf.py \
#               examples/pomade-graph-scalp.usda
#
# with the same environment testUsdviewPomadeGraph.py uses. The probes time
# real C ABI work (Tube-mode moves, K11 picks) on the 4x4 fixture and gate
# it at the S7 budgets with ~1000x headroom, so they are green by
# construction here; the gates themselves run on the S7 reference scene
# (examples/pomade-reference.usda via examples/tools/make_pomade_reference.py
# --wish) and stay TODO until that harness exists.
#
# Proven here:
#   * a Tube-mode move (center CV edit + version poll) completes, timed;
#   * a K11 pick completes, timed;
#   * TN-4 (idle-swap budget) and TN-7 (bake isolation) skip with their
#     harness reasons;
#   * SS-06: a ~100k-face scalp binds (PomadeSession.activate) within
#     FAST_BIND_S, and a usdview that bound a scalp through the plugin
#     exits within QUIT_BUDGET_S of its script ending with its per-process
#     bake directory removed (a child testusdview run of this same file).
import ctypes
import os
import sys
import time

failures = 0
skips = 0

PICK_GUIDE = 16

# S7 budgets. The fixture gates below reuse the numbers with enormous
# headroom; the reference-scene gates keep them exactly.
TN1_MOVE_MS = 8.0
TN2_PICK_MS = 0.5

# SS-06 budgets. The bind covers the whole activate() -- stage read,
# Pomade_BindScalp, committer + bake start, first publish -- of a 317x317
# quad grid; the per-element Python conversion it replaced spent most of
# its time in list comprehensions before the model saw a single face.
FAST_BIND_FACES_SIDE = 317
FAST_BIND_S = 3.0
QUIT_BUDGET_S = 20.0
# Set in the child testusdview's environment to the path its result goes
# to; the same file then runs as the quit probe's child.
QUIT_CHILD_ENV = "USDGENPOMADE_PERF_QUIT_CHILD"


def check(ok, what):
    global failures
    if ok:
        print("ok:   %s" % what)
    else:
        failures += 1
        print("FAIL: %s" % what)


def skip(what):
    global skips
    skips += 1
    print("SKIP: %s" % what)


def locate(x, z):
    """(face, u, v) for a scalp position (x, z)."""
    import math
    ix = min(max(int(math.floor(x)), 0), 3)
    iz = min(max(int(math.floor(z)), 0), 3)
    return (ix * 4 + iz, float(z - iz), float(x - ix))


def strokeRect(dll, model, corners, snap):
    """Stroke a closed rectangle; returns (rc, chain, closed)."""
    samples = []
    per = 5
    for k in range(len(corners)):
        x0, z0 = corners[k]
        x1, z1 = corners[(k + 1) % len(corners)]
        for i in range(per):
            t = float(i) / float(per)
            samples.append((x0 + (x1 - x0) * t, z0 + (z1 - z0) * t))
    samples.append(corners[0])
    faces = (ctypes.c_int * len(samples))()
    uvs = (ctypes.c_float * (2 * len(samples)))()
    for i, (x, z) in enumerate(samples):
        face, u, v = locate(x, z)
        faces[i] = face
        uvs[2 * i] = u
        uvs[2 * i + 1] = v
    out = (ctypes.c_int * 4096)()
    count = ctypes.c_int(0)
    closed = ctypes.c_int(0)
    weldStart = ctypes.c_int(0)
    weldEnd = ctypes.c_int(0)
    rc = dll.Pomade_GraphStroke(model, faces, uvs, len(samples),
                               ctypes.c_float(snap), ctypes.c_float(0.05),
                               out, 4096, ctypes.byref(count),
                               ctypes.byref(closed), ctypes.byref(weldStart),
                               ctypes.byref(weldEnd))
    return (rc, [out[i] for i in range(count.value)], bool(closed.value))


def run(stage):
    global failures
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = ""
    if here:
        sys.path.insert(0, os.path.normpath(os.path.join(
            here, "..", "python")))
    try:
        from usdGenPomadeTools import pomadeLib
    except ImportError as exc:
        print("FAIL: cannot import usdGenPomadeTools: %s" % exc)
        return 1
    try:
        lib = pomadeLib.Library()
    except OSError as exc:
        print("FAIL: %s" % exc)
        return 1
    dll = lib.dll

    scalp = stage.GetPrimAtPath("/Scalp")
    check(scalp, "the stage carries /Scalp")
    if not scalp:
        return 1
    points = scalp.GetAttribute("points").Get()
    counts = scalp.GetAttribute("faceVertexCounts").Get()
    indices = scalp.GetAttribute("faceVertexIndices").Get()
    flat = [float(c) for p in points for c in (p[0], p[1], p[2])]
    pts = (ctypes.c_float * len(flat))(*flat)
    cnt = (ctypes.c_int * len(counts))(*[int(c) for c in counts])
    idx = (ctypes.c_int * len(indices))(*[int(i) for i in indices])

    model = ctypes.c_void_p(None)
    check(dll.Pomade_Create(ctypes.byref(model)) == 0, "model creates")
    check(dll.Pomade_BindScalp(model, pts, len(flat), cnt, len(counts),
                              idx, len(indices)) == 0, "scalp binds")
    rc, _, closed = strokeRect(
        dll, model, [(0.0, 1.0), (4.0, 1.0), (4.0, 3.0), (0.0, 3.0)], 0.1)
    check(rc == 0 and closed, "the region stroke closes")
    check(dll.Pomade_Rasterise(model) == 0, "K3 rasterises")
    check(dll.Pomade_BuildTubeFromRegion(model, 0, 5, 8, ctypes.c_float(2.0))
          == 0, "the tube builds")
    check(dll.Pomade_SetFillParams(model, ctypes.c_float(50.0), 8, 11,
                                  ctypes.c_float(0.0), None, 0) == 0,
          "fill params set")
    check(dll.Pomade_RefillGuides(model, ctypes.c_float(1.0)) == 0,
          "the refill runs")

    # -- TN-1 probe: one Tube-mode move -----------------------------------
    # A move is a center edit plus the version poll the shelf does after
    # every gesture step; the K6/K7 passes it will carry land with P4.
    moves = 200
    t0 = time.perf_counter()
    for i in range(moves):
        dll.Pomade_MoveCenterCV(model, 2, ctypes.c_float(0.01),
                               ctypes.c_float(0.0), ctypes.c_float(0.0))
        dll.Pomade_GetVersion(model)
    perMove = (time.perf_counter() - t0) / moves * 1000.0
    print("info: TN-1 probe (fixture): %.4f ms/move, budget %.1f ms "
          "(gate runs on the S7 reference scene)" % (perMove, TN1_MOVE_MS))
    check(perMove < TN1_MOVE_MS, "TN-1 probe under budget on the fixture")

    # -- TN-2 probe: one K11 pick ------------------------------------------
    guides = ctypes.c_int(0)
    cv = ctypes.c_int(0)
    dll.Pomade_GetGuideCounts(model, ctypes.byref(guides), ctypes.byref(cv))
    if guides.value > 0:
        xyz = (ctypes.c_float * (3 * guides.value * cv.value))()
        gcounts = (ctypes.c_int * guides.value)()
        got = ctypes.c_int(0)
        dll.Pomade_ReadGuidePreview(model, xyz, len(xyz), gcounts,
                                   len(gcounts), ctypes.byref(got))
        viewProj = (ctypes.c_float * 16)(1, 0, 0, 0, 0, 1, 0, 0,
                                         0, 0, 1, 0, 0, 0, 0, 1)
        px = (xyz[0] * 0.5 + 0.5) * 400.0
        py = (1.0 - (xyz[1] * 0.5 + 0.5)) * 400.0
        hit = ctypes.c_int(0)
        kind = ctypes.c_uint(0)
        index = ctypes.c_int(0)
        sub = ctypes.c_int(0)
        dist = ctypes.c_float(0)
        depth = ctypes.c_float(0)
        picks = 500
        t0 = time.perf_counter()
        for _ in range(picks):
            dll.Pomade_Pick(model, viewProj, 400, 400, ctypes.c_float(px),
                           ctypes.c_float(py), ctypes.c_float(5.0), PICK_GUIDE,
                           ctypes.byref(hit), ctypes.byref(kind),
                           ctypes.byref(index), ctypes.byref(sub),
                           ctypes.byref(dist), ctypes.byref(depth))
        perPick = (time.perf_counter() - t0) / picks * 1000.0
        print("info: TN-2 probe (fixture): %.4f ms/pick, budget %.1f ms "
              "(gate runs at 770 K tube verts)" % (perPick, TN2_PICK_MS))
        check(perPick < TN2_PICK_MS, "TN-2 probe under budget on the fixture")
        check(hit.value == 1, "the probe pick hits")
    else:
        skip("TN-2 probe: the refill yielded no guides")

    # -- reference-scene gates (stubs) --------------------------------------
    # TN-3 moved out: testUsdviewPomadePick.py gates K11 vs view.pick() at
    # 100 pixels (P6 exit); this stub keeps the fixture probes only.
    skip("TN-3 pick accuracy: see testUsdviewPomadePick")
    # TODO(TN-4): main-thread TransferContent of the reference groom in
    # <= 5 ms per idle slot; needs pomade-reference.usda at --wish scale
    # plus the idle-swap harness (the T1 TN-4 section in
    # tests/hosts/usdGenPomade/testUsdGenPomadeCommit.cpp proves the slot budget meanwhile).
    skip("TN-4 swap: needs the S7 reference scene + idle-swap harness")
    # TODO(TN-7): a Tube-mode drag while a full bake is in flight; needs
    # the bake worker under the reference scalp.
    skip("TN-7 bake isolation: needs the reference bake in flight")

    dll.Pomade_Destroy(model)
    fastBind()
    print("testUsdviewPomadePerf: %d failure(s), %d skip(s)"
          % (failures, skips))
    return 1 if failures else 0


def gridScalpStage(side):
    """An in-memory stage whose /Scalp is a side x side quad grid."""
    import numpy
    from pxr import Usd, UsdGeom, Vt
    verts = side + 1
    axis = numpy.linspace(0.0, 4.0, verts, dtype=numpy.float32)
    xs, zs = numpy.meshgrid(axis, axis, indexing="ij")
    points = numpy.stack([xs.ravel(), numpy.zeros(verts * verts,
                                                  numpy.float32),
                          zs.ravel()], axis=1).astype(numpy.float32)
    rows, cols = numpy.meshgrid(numpy.arange(side), numpy.arange(side),
                                indexing="ij")
    first = (rows * verts + cols).ravel()
    indices = numpy.stack([first, first + verts, first + verts + 1,
                           first + 1], axis=1).ravel().astype(numpy.int32)
    stage = Usd.Stage.CreateInMemory("pomadeFastBind")
    mesh = UsdGeom.Mesh.Define(stage, "/Scalp")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        numpy.full(side * side, 4, numpy.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    return stage


def fastBind():
    """SS-06: a ~100k-face scalp binds within FAST_BIND_S."""
    try:
        import numpy  # noqa: F401 -- the fixture is built with it
    except ImportError:
        skip("fast bind: numpy is not importable")
        return
    from usdGenPomadeTools import pomadeSession, pomadeToolState
    side = FAST_BIND_FACES_SIDE
    stage = gridScalpStage(side)
    session = pomadeSession.PomadeSession(pomadeToolState.PomadeToolState())
    heard = []
    session.setStatusSink(lambda *args: heard.append(args))
    t0 = time.perf_counter()
    mesh = session._scalpMesh("/Scalp", stage)
    convert = time.perf_counter() - t0
    t0 = time.perf_counter()
    ok = session.activate("/Scalp", stage=stage)
    elapsed = time.perf_counter() - t0
    print("info: fast bind: %d faces, conversion %.3f s, activate %.3f s "
          "(budget %.1f s)" % (side * side, convert, elapsed, FAST_BIND_S))
    check(mesh is not None and len(mesh[1]) == side * side,
          "fast bind: the grid reads as %d faces" % (side * side))
    check(ok and session.model is not None,
          "fast bind: the ~100k-face scalp binds (%r)" % heard)
    check(elapsed < FAST_BIND_S,
          "fast bind: activate() within %.1f s (%.3f s)"
          % (FAST_BIND_S, elapsed))
    session.deactivate()


def quitChild(appController, resultPath):
    """The quit probe's child: bind through the plugin, then just return.

    Nothing here tears the session down: that is the container's quit
    hook's job, and whether it did it is what the parent measures.
    """
    import json
    result = {"pid": os.getpid(), "bound": False, "bakeDir": "",
              "baked": False, "end": 0.0}
    try:
        import usdGenPomadeTools
        dataModel = appController._dataModel
        registry = appController._plugRegistry
        dataModel.selection.setPrimPath("/Scalp")
        registry.getCommandPlugin("usdGenPomadeTools.openWorkspace").run()
        registry.getCommandPlugin("usdGenPomadeTools.bindScalp").run()
        session = usdGenPomadeTools.container().session
        result["bound"] = session is not None and session.model is not None
        if result["bound"]:
            result["bakeDir"] = session.bakeDir
            # A real region map in the directory when the bake obliges;
            # a marker file otherwise, so "the directory is gone" is never
            # vacuously true.
            dll = session.dll
            strokeRect(dll, session.model,
                       [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
                       0.1)
            session.rasterise()
            session.ensureRegionTubes()
            if session.rebake():
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline and \
                        session.hasPendingWork():
                    session.pump()
                    time.sleep(0.02)
                result["baked"] = bool(session._state.bakedVersion)
            if session.bakeDir:
                os.makedirs(session.bakeDir, exist_ok=True)
                with open(os.path.join(session.bakeDir, "quitProbe.txt"),
                          "w") as marker:
                    marker.write("SS-06 quit probe\n")
    finally:
        result["end"] = time.time()
        with open(resultPath, "w") as handle:
            json.dump(result, handle)
    return 0


def quitProbe():
    """SS-06: a bound usdview exits promptly and takes its bake dir along.

    Runs this file again in a child testusdview (the same command line
    CTest gave this one) with QUIT_CHILD_ENV set. The child binds a scalp
    and returns without any teardown of its own; testusdview then closes
    its windows and the interpreter exits, so only the container's quit
    hook (aboutToQuit, or atexit where the event loop never ends) can join
    the workers and remove the directory.
    """
    import json
    import subprocess
    import tempfile
    argv = list(sys.argv)
    if "--testScript" not in argv or len(argv) < 4:
        skip("quit probe: not running under testusdview (%r)" % argv)
        return
    script = argv[argv.index("--testScript") + 1]
    scene = argv[-1]
    handle, resultPath = tempfile.mkstemp(prefix="pomadeQuitProbe",
                                          suffix=".json")
    os.close(handle)
    env = dict(os.environ)
    env.pop("USDGENPOMADE_BAKE_DIR", None)   # the per-process default
    env[QUIT_CHILD_ENV] = resultPath
    try:
        proc = subprocess.Popen(
            [sys.executable, argv[0], "--testScript", script, scene],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            output, _ = proc.communicate(timeout=200)
            exited = time.time()
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate()
            exited = None
        text = output.decode("utf-8", "replace") if output else ""
        try:
            with open(resultPath) as stream:
                result = json.load(stream)
        except (OSError, ValueError):
            result = {}
    finally:
        try:
            os.remove(resultPath)
        except OSError:
            pass
    clean = ("Traceback" not in text and "FAIL:" not in text)
    if exited is None or not result or not clean:
        # The child's own words, defanged so CTest's FAIL:/Traceback
        # expression fires on this test's verdict alone.
        for line in text.splitlines()[-25:]:
            print("info: child| %s" % line.replace("FAIL:", "F-AIL:")
                  .replace("Traceback", "T-raceback"))
    check(exited is not None, "quit probe: the child usdview exits")
    check(bool(result.get("bound")),
          "quit probe: the child bound /Scalp through the plugin (%r)"
          % result)
    if exited is not None and result.get("end"):
        lag = exited - float(result["end"])
        check(lag <= QUIT_BUDGET_S,
              "quit probe: it exits %.1f s after its script (budget %.0f s)"
              % (lag, QUIT_BUDGET_S))
    bakeDir = result.get("bakeDir", "")
    print("info: quit probe: child pid %s, bake dir %s, baked %s"
          % (result.get("pid"), bakeDir, result.get("baked")))
    check(bool(bakeDir) and not os.path.exists(bakeDir),
          "quit probe: its per-process bake directory is gone (%r)"
          % bakeDir)
    check(clean, "quit probe: the child printed no traceback or failure")


def _testenvDir():
    """This file's directory; testusdview execs it with no __file__."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    for index, argument in enumerate(sys.argv):
        if argument == "--testScript" and index + 1 < len(sys.argv):
            return os.path.dirname(os.path.abspath(sys.argv[index + 1]))
    return ""


def ladderChip(appController):
    """FB-02: a drag that steps the fallback ladder says so in the HUD.

    The budget is forced to 0 ms so every real move is over it: the chip
    must name the live rung while the button is held, linger about a
    second after the release restored full detail, then go.  With
    `ladderEnabled` off the same drag never steps at all.
    """
    here = _testenvDir()
    if here and here not in sys.path:
        sys.path.insert(0, here)
    try:
        import pomadeT3
        from usdGenPomadeTools import pomadeCamera, pomadeLadder
    except ImportError as exc:
        check(False, "ladder chip: the T3 helpers import: %s" % exc)
        return
    session, viewport, state, _workspace, container = pomadeT3.openAndBind(
        appController, "/Scalp")
    try:
        check(session is not None and session.model is not None and
              viewport is not None and viewport.installed,
              "ladder chip: /Scalp binds with a live viewport")
        if session is None or session.model is None or viewport is None:
            return
        view = viewport.view
        stage = appController._dataModel.stage
        check(pomadeT3.frameScalp(stage, view), "ladder chip: top camera")
        view.setFocus()
        pomadeT3.wait(50)
        viewport.setPointerInside(True)
        mouse = pomadeT3.Mouse(view)
        mouse.direct = True
        viewport.setMode("graph")
        viewport.setSubMode("draw")
        camera = pomadeCamera.resolve(view)
        check(camera is not None, "ladder chip: the camera resolves")
        if camera is None:
            return
        hud = viewport._hudOverlay

        def stroke(z):
            points = []
            for i in range(25):
                p = camera.worldToPixels((0.6 + 2.8 * i / 24.0, 0.0, z))
                points.append((p[0], p[1]))
            return points

        state.moveBudgetMs = 0.0
        path = stroke(0.6)
        mouse.press(path[0])
        for point in path[1:]:
            mouse.move(point)
        pomadeT3.wait(10)
        step = int(state.ladderStep)
        chip = hud.chip if hud is not None else ""
        check(step >= 2 and chip == pomadeLadder.chipLabel(step) and
              "(auto)" in chip and hud.isVisible(),
              "ladder chip: a 0 ms budget steps the ladder and the HUD chip "
              "names the rung (step %d, %r)" % (step, chip))
        mouse.release(path[-1])
        pomadeT3.wait(10)
        check(int(state.ladderStep) == 0 and hud.chip.startswith(chip),
              "ladder chip: the release restores full detail and the chip "
              "lingers (%r)" % hud.chip)
        pomadeT3.wait(_chipLingerMs() + 400)
        check(hud.chip == "",
              "ladder chip: about a second later the chip is gone (%r)"
              % hud.chip)

        state.ladderEnabled = False
        path = stroke(3.4)
        mouse.press(path[0])
        for point in path[1:]:
            mouse.move(point)
        pomadeT3.wait(10)
        check(int(state.ladderStep) == 0 and hud.chip == "",
              "ladder chip: with the ladder disabled the same drag never "
              "steps (step %d, %r)" % (state.ladderStep, hud.chip))
        mouse.release(path[-1])
    finally:
        state = getattr(container, "pomadeState", None)
        if state is not None:
            state.ladderEnabled = True
            state.moveBudgetMs = pomadeLadder.MOVE_BUDGET_MS
        viewport = getattr(container, "viewport", None)
        session = getattr(container, "session", None)
        try:
            if viewport is not None:
                viewport.uninstall()
        finally:
            if session is not None:
                session.deactivate()


def _chipLingerMs():
    from usdGenPomadeTools import pomadeViewport
    return int(pomadeViewport.LADDER_CHIP_LINGER_MS)


def testUsdviewInputFunction(appController):
    resultPath = os.environ.get(QUIT_CHILD_ENV)
    if resultPath:
        return quitChild(appController, resultPath)
    model = getattr(appController, "_dataModel", appController)
    code = run(model.stage)
    before = failures
    ladderChip(appController)
    print("testUsdviewPomadePerf: ladder chip %d failure(s)"
          % (failures - before))
    before = failures
    quitProbe()
    print("testUsdviewPomadePerf: quit probe %d failure(s)"
          % (failures - before))
    return 1 if failures else code


if __name__ == "__main__":
    # Direct stage run (no usdview): python testUsdviewPomadePerf.py stage.usda
    from pxr import Usd
    if len(sys.argv) != 2:
        print("usage: testUsdviewPomadePerf.py <stage.usda>")
        sys.exit(2)
    sys.exit(run(Usd.Stage.Open(sys.argv[1])))
