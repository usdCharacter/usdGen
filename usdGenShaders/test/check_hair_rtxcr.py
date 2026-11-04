#!/usr/bin/env python3
# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""Numerical checks of the shipped GLSL, executed on OpenGL compute.

Checks axial/grazing cases, roughness/IOR limits, absorption monotonicity,
rotation invariance, the analytic diffuse cylinder response and normalized
longitudinal scattering. Uses the same context setup as the Storm render test.
No second implementation of the far-field evaluator is kept here.
"""
import argparse
import math
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usdrecord", required=True)
    args = parser.parse_args()
    try:
        import numpy as np
        from OpenGL import GL
        from OpenGL.GL.shaders import compileProgram, compileShader
        ctx = runpy.run_path(args.usdrecord)["_SetupOpenGLContext"]()
    except (ImportError, OSError, RuntimeError) as exc:
        print("SKIP: no GL test context:", exc)
        return 77
    version = (int(GL.glGetIntegerv(GL.GL_MAJOR_VERSION)),
               int(GL.glGetIntegerv(GL.GL_MINOR_VERSION)))
    if version < (4, 3) or not GL.glDispatchCompute:
        print("SKIP: OpenGL 4.3 compute unavailable")
        return 77

    path = Path(__file__).resolve().parents[1] / "resources/shaders/usdGenHairStrandsRtxcr.glslfx"
    source = path.read_text(encoding="utf-8")
    core = source.split("-- glsl UsdGenStrands.Rtxcr\n", 1)[1].split("---", 1)[0]
    shader = "#version 430\n#define USDGEN_STRANDS_MODEL_RTXCR 1\n" + core + r"""
layout(local_size_x = 64) in;
struct Case { vec4 lightWidth; vec4 viewIor; vec4 colorAzimuth; vec4 controls; };
layout(std430, binding = 0) readonly buffer Cases { Case cases[]; };
layout(std430, binding = 1) writeonly buffer Results { vec4 results[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= cases.length()) return;
    Case c = cases[i];
    UsdGenRtxcrHairInteractionSurface s;
    s.tangent = vec3(1, 0, 0);
    s.shadingNormal = vec3(0, 0, 1);
    vec3 L = c.lightWidth.xyz;
    vec3 V = c.viewIor.xyz;
    if (c.controls.z > 0.5) {
        s.tangent = s.tangent.zxy;
        s.shadingNormal = s.shadingNormal.zxy;
        L = L.zxy;
        V = V.zxy;
    }
    UsdGenRtxcrHairMaterialInteractionBcsdf m;
    m.roughness = c.lightWidth.w;
    m.primaryRoughness = m.roughness;
    m.ior = c.viewIor.w;
    m.absorptionCoefficient = UsdGenRtxcrAbsorptionCoefficientFromColor(
        c.colorAzimuth.rgb, c.colorAzimuth.w);
    m.cuticleAngle = radians(c.controls.x);
    m.diffuseReflectionTint = c.colorAzimuth.rgb;
    m.diffuseReflectionWeight = c.controls.y;
    m.reflectionGain = 1.0;
    m.transmissionGain = 1.0;
    vec3 specular, diffuse;
    UsdGenRtxcrHairFarFieldBcsdfEval(s, m, L, V, specular, diffuse);
    // Independent 1D integral of M over sin(thetaI) should be one.
    float longitudinal = UsdGenRtxcrFarFieldLongitudinal(
        asin(clamp(V.x, -1.0, 1.0)), L.x,
        sqrt(max(1.0 - L.x * L.x, 0.0)), m.roughness);
    results[i] = vec4(specular + diffuse, longitudinal);
}
"""
    # A compile/link failure must fail, never skip like an unavailable context.
    program = compileProgram(compileShader(shader, GL.GL_COMPUTE_SHADER))
    buffers = GL.glGenBuffers(2)

    def evaluate(cases):
        data = np.ascontiguousarray(cases, dtype=np.float32)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffers[0])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, buffers[0])
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffers[1])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(data) * 16, None, GL.GL_STREAM_READ)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, buffers[1])
        GL.glUseProgram(program)
        GL.glDispatchCompute((len(data) + 63) // 64, 1, 1)
        GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        result = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, len(data) * 16)
        return np.frombuffer(result, dtype=np.float32).reshape(-1, 4).copy()

    rng = np.random.default_rng(2718)
    cases = np.zeros((1024, 4, 4), dtype=np.float32)
    for slot in (0, 1):
        directions = rng.normal(size=(len(cases), 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        cases[:, slot, :3] = directions
        # Exactly axial rays exercise the frame and TT grazing fallbacks.
        cases[:8, slot, :3] = np.tile([[1, 0, 0], [-1, 0, 0]], (4, 1))
    cases[:, 0, 3] = rng.choice([0.01, 0.06, 0.35, 1.0], len(cases))
    cases[:, 1, 3] = rng.choice([1.0001, 1.55, 2.2, 4.0], len(cases))
    cases[:, 2, :3] = rng.uniform(0.001, 1.0, (len(cases), 3))
    cases[:, 2, 3] = rng.choice([0, 0.35, 1], len(cases))
    cases[:, 3, 0] = rng.choice([0, 2, 30], len(cases))
    original = evaluate(cases)
    assert np.isfinite(original).all() and original.min() >= 0, "nonfinite/negative scattering"

    rotated = cases.copy()
    rotated[:, 3, 2] = 1
    # The fourth channel is a separate, x-axis-only normalization probe.
    np.testing.assert_allclose(evaluate(rotated)[:, :3], original[:, :3],
                               rtol=2e-4, atol=2e-5)
    dark = cases.copy()
    dark[:, 2, :3] *= 0.1
    assert np.all(evaluate(dark)[:, :3] <= original[:, :3] + 2e-5), "absorption adds energy"

    # Diffuse response has a closed-form cylinder integral, independent of IOR.
    diffuse = cases.copy()
    diffuse[:, 3, 1] = 1
    v = diffuse[:, 1, 1:3].astype(float)
    l = diffuse[:, 0, 1:3].astype(float)
    cos_i = np.linalg.norm(l, axis=1)
    cos_o = np.linalg.norm(v, axis=1)
    valid = (cos_i > 1e-4) & (cos_o > 1e-4)
    cos_phi = np.clip(np.sum(v * l, axis=1)[valid] / (cos_i * cos_o)[valid], -1, 1)
    phi = np.arccos(cos_phi)
    expected = diffuse[valid, 2, :3] * (
        cos_i[valid] * ((math.pi - phi) * cos_phi + np.sin(phi)) / (4 * math.pi))[:, None]
    np.testing.assert_allclose(evaluate(diffuse)[valid, :3], expected, rtol=5e-4, atol=2e-5)

    # Integrate M in solid angle's elevation coordinate, d(sin thetaI).
    # No extra cos(thetaI) is applied to the CRCF in the lighting adapter.
    nodes, weights = np.polynomial.legendre.leggauss(512)
    integral_cases = np.repeat(cases[8:9], len(nodes), axis=0)
    integral_cases[:, 0, :3] = np.column_stack((nodes, np.sqrt(1 - nodes * nodes), nodes * 0))
    for width in (0.06, 0.35, 1.0):
        integral_cases[:, 0, 3] = width
        for sin_view in (0.0, 0.9, 1.0):
            integral_cases[:, 1, :3] = [sin_view, math.sqrt(1 - sin_view * sin_view), 0]
            integral = float(evaluate(integral_cases)[:, 3] @ weights)
            assert abs(integral - 1) < 0.006, (width, sin_view, integral)
    GL.glDeleteBuffers(2, buffers)
    GL.glDeleteProgram(program)
    del ctx
    print("RTXCR GLSL: PASS (1024 edge/random cases, rotations, absorption, diffuse, longitudinal normalization)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
