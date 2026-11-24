// usdGenPomade — K1/K2/K3 kernels (P2) + the P0 tessellation kernel.
//
// One thread per ray / chord sample / query point. The device math spells the
// CPU twins (pomadeScalp.cpp, pomadeGraph.cpp, pomadeRegion.cpp) with the same
// operation order, so parity is bit-exact on arithmetic-only paths (the TN-6
// rule; transcendentals take tolerance, and these kernels use none).
#include "usdGenPomade/pomadeKernels.h"

#include <cfloat>
#include <cstdio>
#include <atomic>
#include <utility>

#include <math_constants.h>  // CUDART_INF_F (cf. curveTileBounds.cu)

namespace usdGenPomade {

namespace {

__global__ void _PomadeTessellateKernel(PomadeTubeShape shape,
                                       float const *centerX,
                                       float const *centerY,
                                       float const *centerZ,
                                       float *positions,
                                       float *normals)
{
    int const v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= PomadeTubeVertexCount(shape)) {
        return;
    }
    PomadeTessellatedVertex const tv =
        PomadeTessellateVertex(shape, v, centerX, centerY, centerZ);
    positions[size_t(v) * 3 + 0] = tv.px;
    positions[size_t(v) * 3 + 1] = tv.py;
    positions[size_t(v) * 3 + 2] = tv.pz;
    normals[size_t(v) * 3 + 0] = tv.nx;
    normals[size_t(v) * 3 + 1] = tv.ny;
    normals[size_t(v) * 3 + 2] = tv.nz;
}

__device__ float _Dot3(float const a[3], float const b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Same watertight dominant-axis shear test as the CPU twin.  Rounded double
// products stay separate: FMA contraction would break the exact sign reversal
// on a shared fan edge and reintroduce the seam crack.
__device__ double _EdgeDet(double ax, double ay, double bx, double by)
{
    return __dsub_rn(__dmul_rn(ax, by), __dmul_rn(ay, bx));
}

__device__ float _RayTriangle(float const origin[3], float const dir[3],
                              float const v0[3], float const v1[3],
                              float const v2[3])
{
    int kz = 0;
    if (fabsf(dir[1]) > fabsf(dir[kz])) {
        kz = 1;
    }
    if (fabsf(dir[2]) > fabsf(dir[kz])) {
        kz = 2;
    }
    if (dir[kz] == 0.0f) {
        return CUDART_INF_F;
    }
    int kx = (kz + 1) % 3;
    int ky = (kx + 1) % 3;
    if (dir[kz] < 0.0f) {
        int const swap = kx;
        kx = ky;
        ky = swap;
    }
    double const sx = __ddiv_rn(double(dir[kx]), double(dir[kz]));
    double const sy = __ddiv_rn(double(dir[ky]), double(dir[kz]));
    double const sz = __ddiv_rn(1.0, double(dir[kz]));
    double a[3], b[3], c[3];
    auto shear = [&](float const p[3], double out[3]) {
        double const x = __dsub_rn(double(p[kx]), double(origin[kx]));
        double const y = __dsub_rn(double(p[ky]), double(origin[ky]));
        double const z = __dsub_rn(double(p[kz]), double(origin[kz]));
        out[0] = __dsub_rn(x, __dmul_rn(sx, z));
        out[1] = __dsub_rn(y, __dmul_rn(sy, z));
        out[2] = __dmul_rn(z, sz);
    };
    shear(v0, a);
    shear(v1, b);
    shear(v2, c);
    double const u = _EdgeDet(c[0], c[1], b[0], b[1]);
    double const v = _EdgeDet(a[0], a[1], c[0], c[1]);
    double const w = _EdgeDet(b[0], b[1], a[0], a[1]);
    bool const anyNegative = u < 0.0 || v < 0.0 || w < 0.0;
    bool const anyPositive = u > 0.0 || v > 0.0 || w > 0.0;
    if (anyNegative && anyPositive) {
        return CUDART_INF_F;
    }
    double const det = __dadd_rn(__dadd_rn(u, v), w);
    if (det == 0.0) {
        return CUDART_INF_F;
    }
    double const t = __ddiv_rn(
        __dadd_rn(__dadd_rn(__dmul_rn(u, a[2]), __dmul_rn(v, b[2])),
                  __dmul_rn(w, c[2])), det);
    return isfinite(t) && t >= 0.0 ? float(t) : CUDART_INF_F;
}

__device__ bool _RayAabb(float const origin[3], float const inv[3], float tMax,
                         PomadeDeviceBvhNode const &node)
{
    float t0 = 0.0f;
    float t1 = tMax;
    float const mn[3] = {node.minX, node.minY, node.minZ};
    float const mx[3] = {node.maxX, node.maxY, node.maxZ};
    for (int a = 0; a < 3; ++a) {
        float lo = (mn[a] - origin[a]) * inv[a];
        float hi = (mx[a] - origin[a]) * inv[a];
        if (lo > hi) {
            float const tmp = lo;
            lo = hi;
            hi = tmp;
        }
        t0 = fmaxf(t0, lo);
        t1 = fminf(t1, hi);
        if (t0 > t1) {
            return false;
        }
    }
    return true;
}

// Same closest-point-on-triangle as the CPU twin (Ericson §5.1.5).
__device__ float _ClosestOnTriangle(float const p[3], float const v0[3],
                                    float const v1[3], float const v2[3],
                                    float out[3])
{
    float ab[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    float ac[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    float ap[3] = {p[0] - v0[0], p[1] - v0[1], p[2] - v0[2]};
    float const d1 = _Dot3(ab, ap);
    float const d2 = _Dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        out[0] = v0[0];
        out[1] = v0[1];
        out[2] = v0[2];
    } else {
        float bp[3] = {p[0] - v1[0], p[1] - v1[1], p[2] - v1[2]};
        float const d3 = _Dot3(ab, bp);
        float const d4 = _Dot3(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) {
            out[0] = v1[0];
            out[1] = v1[1];
            out[2] = v1[2];
        } else {
            float const vc = d1 * d4 - d3 * d2;
            float cp[3] = {p[0] - v2[0], p[1] - v2[1], p[2] - v2[2]};
            float const d5 = _Dot3(ab, cp);
            float const d6 = _Dot3(ac, cp);
            if (d6 >= 0.0f && d5 <= d6) {
                out[0] = v2[0];
                out[1] = v2[1];
                out[2] = v2[2];
            } else if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                float const w = d1 / (d1 - d3);
                out[0] = v0[0] + w * ab[0];
                out[1] = v0[1] + w * ab[1];
                out[2] = v0[2] + w * ab[2];
            } else {
                float const vb = d5 * d2 - d1 * d6;
                if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                    float const w = d2 / (d2 - d6);
                    out[0] = v0[0] + w * ac[0];
                    out[1] = v0[1] + w * ac[1];
                    out[2] = v0[2] + w * ac[2];
                } else {
                    float const va = d3 * d6 - d5 * d4;
                    if (va <= 0.0f && (d4 - d3) >= 0.0f &&
                        (d5 - d6) >= 0.0f) {
                        float const w =
                            (d4 - d3) / ((d4 - d3) + (d5 - d6));
                        out[0] = v1[0] + w * (v2[0] - v1[0]);
                        out[1] = v1[1] + w * (v2[1] - v1[1]);
                        out[2] = v1[2] + w * (v2[2] - v1[2]);
                    } else {
                        float const denom = 1.0f / (va + vb + vc);
                        float const v = vb * denom;
                        float const w = vc * denom;
                        out[0] = v0[0] + ab[0] * v + ac[0] * w;
                        out[1] = v0[1] + ab[1] * v + ac[1] * w;
                        out[2] = v0[2] + ab[2] * v + ac[2] * w;
                    }
                }
            }
        }
    }
    float dx = p[0] - out[0];
    float dy = p[1] - out[1];
    float dz = p[2] - out[2];
    return dx * dx + dy * dy + dz * dz;
}

__global__ void _RaycastKernel(float const *points,
                               int const *faceCounts,
                               int const *faceIndices,
                               int const *faceOffsets,
                               float const *faceNormals,
                               PomadeDeviceBvhNode const *nodes,
                               int const *order,
                               int root,
                               float const *origins,
                               float const *dirs,
                               PomadeDeviceHit *hits,
                               int rayCount)
{
    int const r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rayCount) {
        return;
    }
    PomadeDeviceHit hit;
    hit.faceId = -1;
    float const *org = origins + size_t(r) * 3;
    float const *dd = dirs + size_t(r) * 3;
    float const dl = sqrtf(_Dot3(dd, dd));
    if (!(dl > 0.0f)) {
        hits[r] = hit;
        return;
    }
    float o[3] = {org[0], org[1], org[2]};
    float d[3] = {dd[0] / dl, dd[1] / dl, dd[2] / dl};
    float inv[3];
    for (int a = 0; a < 3; ++a) {
        inv[a] = d[a] != 0.0f ? 1.0f / d[a]
                              : CUDART_INF_F * (d[a] >= 0.0f ? 1.0f : -1.0f);
    }
    float bestT = CUDART_INF_F;
    int bestFace = -1;
    int stack[64];
    int top = 0;
    stack[top++] = root;
    while (top > 0) {
        int const node = stack[--top];
        PomadeDeviceBvhNode const nd = nodes[node];
        if (!_RayAabb(o, inv, bestT, nd)) {
            continue;
        }
        if (nd.left < 0) {
            for (int i = nd.faceBegin; i < nd.faceEnd; ++i) {
                int const f = order[i];
                int const nv = faceCounts[f];
                int const off = faceOffsets[f];
                float const *v0 =
                    points + size_t(faceIndices[off]) * 3;
                for (int k = 1; k + 1 < nv; ++k) {
                    float const *v1 =
                        points + size_t(faceIndices[off + k]) * 3;
                    float const *v2 =
                        points + size_t(faceIndices[off + k + 1]) * 3;
                    float const t = _RayTriangle(o, d, v0, v1, v2);
                    if (t < bestT) {
                        bestT = t;
                        bestFace = f;
                    }
                }
            }
        } else {
            if (top + 2 > 64) {
                continue;
            }
            stack[top++] = nd.left;
            stack[top++] = nd.right;
        }
    }
    if (bestFace >= 0) {
        hit.faceId = bestFace;
        hit.t = bestT;
        hit.px = o[0] + d[0] * bestT;
        hit.py = o[1] + d[1] * bestT;
        hit.pz = o[2] + d[2] * bestT;
        hit.nx = faceNormals[size_t(bestFace) * 3 + 0];
        hit.ny = faceNormals[size_t(bestFace) * 3 + 1];
        hit.nz = faceNormals[size_t(bestFace) * 3 + 2];
    }
    hits[r] = hit;
}

__global__ void _EdgeTraceKernel(float const *points,
                                 int const *faceCounts,
                                 int const *faceIndices,
                                 int const *faceOffsets,
                                 int faceCount,
                                 float ax,
                                 float ay,
                                 float az,
                                 float bx,
                                 float by,
                                 float bz,
                                 float *outPos,
                                 int *outFaces,
                                 int sampleCount)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sampleCount) {
        return;
    }
    float const t =
        sampleCount == 1 ? 0.0f : float(i) / float(sampleCount - 1);
    float const q[3] = {ax + (bx - ax) * t, ay + (by - ay) * t,
                        az + (bz - az) * t};
    float bestD2 = CUDART_INF_F;
    int bestFace = -1;
    float best[3] = {q[0], q[1], q[2]};
    float cand[3];
    for (int f = 0; f < faceCount; ++f) {
        int const nv = faceCounts[f];
        int const off = faceOffsets[f];
        float const *v0 = points + size_t(faceIndices[off]) * 3;
        for (int k = 1; k + 1 < nv; ++k) {
            float const *v1 = points + size_t(faceIndices[off + k]) * 3;
            float const *v2 =
                points + size_t(faceIndices[off + k + 1]) * 3;
            float const d2 = _ClosestOnTriangle(q, v0, v1, v2, cand);
            if (d2 < bestD2) {
                bestD2 = d2;
                bestFace = f;
                best[0] = cand[0];
                best[1] = cand[1];
                best[2] = cand[2];
            }
        }
    }
    if (i == 0) {
        best[0] = ax;
        best[1] = ay;
        best[2] = az;
    } else if (i == sampleCount - 1) {
        best[0] = bx;
        best[1] = by;
        best[2] = bz;
    }
    outPos[size_t(i) * 3 + 0] = best[0];
    outPos[size_t(i) * 3 + 1] = best[1];
    outPos[size_t(i) * 3 + 2] = best[2];
    outFaces[i] = bestFace;
}

__device__ bool _PointInRegion(float const p[3],
                               float const *loopPts,
                               int const *loopBegin,
                               int const *loopCount,
                               float const *planeN,
                               float const *planeP,
                               float const *basisU,
                               float const *basisV,
                               int region)
{
    float const *n = planeN + size_t(region) * 3;
    float const *pp = planeP + size_t(region) * 3;
    float const *u = basisU + size_t(region) * 3;
    float const *v = basisV + size_t(region) * 3;
    float d[3] = {p[0] - pp[0], p[1] - pp[1], p[2] - pp[2]};
    float const along = _Dot3(d, n);
    float dq[3] = {d[0] - along * n[0], d[1] - along * n[1],
                   d[2] - along * n[2]};
    float const qx = _Dot3(dq, u);
    float const qy = _Dot3(dq, v);
    int const begin = loopBegin[region];
    int const count = loopCount[region];
    bool inside = false;
    for (int i = 0, j = count - 1; i < count; j = i++) {
        float const *si = loopPts + size_t(begin + i) * 3;
        float const *sj = loopPts + size_t(begin + j) * 3;
        float di[3] = {si[0] - pp[0], si[1] - pp[1], si[2] - pp[2]};
        float dj[3] = {sj[0] - pp[0], sj[1] - pp[1], sj[2] - pp[2]};
        float const xi = _Dot3(di, u);
        float const yi = _Dot3(di, v);
        float const xj = _Dot3(dj, u);
        float const yj = _Dot3(dj, v);
        float const ex = xi - xj;
        float const ey = yi - yj;
        float const len2 = ex * ex + ey * ey;
        if (len2 > 0.0f) {
            float t = ((qx - xj) * ex + (qy - yj) * ey) / len2;
            t = fminf(fmaxf(t, 0.0f), 1.0f);
            float const bx = xj + ex * t - qx;
            float const by = yj + ey * t - qy;
            if (bx * bx + by * by <= 1e-12f * (1.0f + len2)) {
                return true;
            }
        }
        if ((yi > qy) != (yj > qy)) {
            float const xt = xj + (qy - yj) * (xi - xj) / (yi - yj);
            if (qx <= xt) {
                inside = !inside;
            }
        }
    }
    return inside;
}

__global__ void _ClassifyKernel(float const *positions,
                                int pointCount,
                                float const *loopPts,
                                int const *loopBegin,
                                int const *loopCount,
                                float const *planeN,
                                float const *planeP,
                                float const *basisU,
                                float const *basisV,
                                int const *interpIds,
                                int regionCount,
                                int *outIds)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pointCount) {
        return;
    }
    float const *p = positions + size_t(i) * 3;
    int best = -1;
    for (int r = 0; r < regionCount; ++r) {
        if (_PointInRegion(p, loopPts, loopBegin, loopCount, planeN, planeP,
                           basisU, basisV, r)) {
            int const id = interpIds[r];
            best = (best < 0) ? id : (id < best ? id : best);
        }
    }
    outIds[i] = best;
}

__global__ void _SmoothnessKernel(float const *cx, float const *cy,
                                  float const *cz, int n, float *out)
{
    int const i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    if (i == 0 || i + 1 == n) {
        out[i] = 0.0f;
        return;
    }
    float const p0[3] = {cx[i - 1], cy[i - 1], cz[i - 1]};
    float const p1[3] = {cx[i], cy[i], cz[i]};
    float const p2[3] = {cx[i + 1], cy[i + 1], cz[i + 1]};
    out[i] = PomadeKinkScore(p0, p1, p2);
}

__global__ void _IntersectPairKernel(PomadeDeviceRootChart const *charts,
                                     int const *ids, int const *parents, int n,
                                     int *pairHits, int *broadHits)
{
    int const p = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n * n) {
        return;
    }
    int const i = p / n, j = p % n;
    if (i >= j || PomadePairRelated(ids, parents, n, i, j)) {
        pairHits[p] = 0;
        broadHits[p] = 0;
        return;
    }
    PomadeDeviceRootChart const &a = charts[i];
    PomadeDeviceRootChart const &b = charts[j];
    if (a.vertCount < 3 || a.vertCount > kPomadeCheckRingMax ||
        b.vertCount < 3 || b.vertCount > kPomadeCheckRingMax) {
        pairHits[p] = 0;
        broadHits[p] = 0;
        return;
    }
    float minA[3], maxA[3], minB[3], maxB[3];
    PomadeRingAabb3(a.verts, a.vertCount, minA, maxA);
    PomadeRingAabb3(b.verts, b.vertCount, minB, maxB);
    bool const broad = PomadeAabbsOverlap3(minA, maxA, minB, maxB);
    broadHits[p] = broad ? 1 : 0;
    pairHits[p] = broad ? PomadeRootPairNarrowHit(
                              a.center, a.nrm, a.bin, a.verts, a.vertCount,
                              b.verts, b.vertCount)
                        : 0;
}

__global__ void _IntersectFlagKernel(int const *pairHits, int n, int *flags)
{
    int const t = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= n) {
        return;
    }
    int hit = 0;
    for (int j = t + 1; j < n; ++j) {
        hit |= pairHits[t * n + j];
    }
    for (int i = 0; i < t; ++i) {
        hit |= pairHits[i * n + t];
    }
    flags[t] = hit ? 1 : 0;
}

}  // namespace

bool PomadeLaunchTessellate(PomadeTubeShape const &shape,
                           float const *deviceCenterX,
                           float const *deviceCenterY,
                           float const *deviceCenterZ,
                           float *devicePositions,
                           float *deviceNormals,
                           cudaStream_t stream,
                           char *errBuf,
                           size_t errBufLen)
{
    auto fail = [&](char const *what, cudaError_t status) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "%s: %s", what,
                          cudaGetErrorString(status));
        }
        return false;
    };
    if (!deviceCenterX || !deviceCenterY || !deviceCenterZ ||
        !devicePositions || !deviceNormals) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer");
        }
        return false;
    }
    int const vertexCount = PomadeTubeVertexCount(shape);
    if (vertexCount <= 0) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "empty tube shape");
        }
        return false;
    }
    int const block = 256;
    int const grid = (vertexCount + block - 1) / block;
    _PomadeTessellateKernel<<<grid, block, 0, stream>>>(
        shape, deviceCenterX, deviceCenterY, deviceCenterZ,
        devicePositions, deviceNormals);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        return fail("tessellate launch", launch);
    }
    return true;
}

bool PomadeLaunchRaycastBatch(float const *devicePoints,
                             int const *deviceFaceCounts,
                             int const *deviceFaceIndices,
                             int const *deviceFaceOffsets,
                             float const *deviceFaceNormals,
                             PomadeDeviceBvhNode const *deviceNodes,
                             int const *deviceOrder,
                             int faceCount,
                             int nodeCount,
                             int root,
                             float const *deviceOrigins,
                             float const *deviceDirs,
                             PomadeDeviceHit *deviceHits,
                             int rayCount,
                             cudaStream_t stream,
                             char *errBuf,
                             size_t errBufLen)
{
    if (!devicePoints || !deviceFaceCounts || !deviceFaceIndices ||
        !deviceFaceOffsets || !deviceFaceNormals || !deviceNodes ||
        !deviceOrder || !deviceOrigins || !deviceDirs || !deviceHits ||
        faceCount <= 0 || nodeCount <= 0 || root < 0 || rayCount <= 0) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    int const block = 128;
    int const grid = (rayCount + block - 1) / block;
    _RaycastKernel<<<grid, block, 0, stream>>>(
        devicePoints, deviceFaceCounts, deviceFaceIndices, deviceFaceOffsets,
        deviceFaceNormals, deviceNodes, deviceOrder, root, deviceOrigins,
        deviceDirs, deviceHits, rayCount);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "raycast launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchEdgeTrace(float const *devicePoints,
                          int const *deviceFaceCounts,
                          int const *deviceFaceIndices,
                          int const *deviceFaceOffsets,
                          int faceCount,
                          float const a[3],
                          float const b[3],
                          float *deviceOutPos,
                          int *deviceOutFaces,
                          int sampleCount,
                          cudaStream_t stream,
                          char *errBuf,
                          size_t errBufLen)
{
    if (!devicePoints || !deviceFaceCounts || !deviceFaceIndices ||
        !deviceFaceOffsets || !a || !b || !deviceOutPos || !deviceOutFaces ||
        faceCount <= 0 || sampleCount <= 0) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    int const block = 128;
    int const grid = (sampleCount + block - 1) / block;
    _EdgeTraceKernel<<<grid, block, 0, stream>>>(
        devicePoints, deviceFaceCounts, deviceFaceIndices, deviceFaceOffsets,
        faceCount, a[0], a[1], a[2], b[0], b[1], b[2], deviceOutPos,
        deviceOutFaces, sampleCount);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "edge trace launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchClassifyPoints(float const *devicePositions,
                               int pointCount,
                               float const *deviceLoopPts,
                               int const *deviceLoopBegin,
                               int const *deviceLoopCount,
                               float const *devicePlaneN,
                               float const *devicePlaneP,
                               float const *deviceBasisU,
                               float const *deviceBasisV,
                               int const *deviceInterpIds,
                               int regionCount,
                               int *deviceOutIds,
                               cudaStream_t stream,
                               char *errBuf,
                               size_t errBufLen)
{
    if (!devicePositions || !deviceLoopPts || !deviceLoopBegin ||
        !deviceLoopCount || !devicePlaneN || !devicePlaneP || !deviceBasisU ||
        !deviceBasisV || !deviceInterpIds || !deviceOutIds || pointCount <= 0 ||
        regionCount <= 0) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    int const block = 256;
    int const grid = (pointCount + block - 1) / block;
    _ClassifyKernel<<<grid, block, 0, stream>>>(
        devicePositions, pointCount, deviceLoopPts, deviceLoopBegin,
        deviceLoopCount, devicePlaneN, devicePlaneP, deviceBasisU, deviceBasisV,
        deviceInterpIds, regionCount, deviceOutIds);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "classify launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

namespace {

// -- P3 kernels (plan/17 K4/K5/K8-K11) ----------------------------------------
// All spell through the shared pomadeTube.h inlines (PomadeReflectNormal,
// PomadeEvalCenter, PomadeNlerpFrame, PomadeHermite, PomadeHash01,
// PomadeProjectPoint), so the CPU twins match by construction.

__global__ void _CenterFramesKernel(float const *cx, float const *cy,
                                    float const *cz, int nCv,
                                    PomadeFrame *frames)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    float tPrev[3];
    PomadeCenterTangent(cx, cy, cz, nCv, 0, tPrev);
    float n[3];
    PomadePerp3(tPrev, n);
    for (int i = 0; i < nCv; ++i) {
        float t[3];
        PomadeCenterTangent(cx, cy, cz, nCv, i, t);
        if (i > 0) {
            float p0[3] = {cx[i - 1], cy[i - 1], cz[i - 1]};
            float p1[3] = {cx[i], cy[i], cz[i]};
            PomadeReflectNormal(p0, p1, tPrev, t, n);
            tPrev[0] = t[0];
            tPrev[1] = t[1];
            tPrev[2] = t[2];
        }
        float b[3];
        PomadeCross3(t, n, b);
        PomadeFrame f;
        f.tx = t[0];
        f.ty = t[1];
        f.tz = t[2];
        f.nx = n[0];
        f.ny = n[1];
        f.nz = n[2];
        f.bx = b[0];
        f.by = b[1];
        f.bz = b[2];
        frames[i] = f;
    }
}

__global__ void _TubeTessellateKernel(
    float const *cx, float const *cy, float const *cz, int nCv,
    PomadeFrame const *frames, float const *secT, float const *secU,
    float const *secV, float const *secScale, float const *secTwist, int nSec,
    int ringVerts, int nRings, int segmentsPerSpan, float *positions,
    float *normals, float *ringT, float const *rootOffsets)
{
    int const v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nRings * ringVerts) {
        return;
    }
    int const r = v / ringVerts;
    int const s = v % ringVerts;
    int const k = min(r / segmentsPerSpan, nSec - 2);
    int const localRow = r - k * segmentsPerSpan;
    float const sample = PomadeDisplaySpanFraction(
        localRow, segmentsPerSpan, k, nSec - 1);
    float const t = sample == 1.0f ? secT[k + 1]
        : secT[k] + (secT[k + 1] - secT[k]) * sample;
    bool const rootHold = segmentsPerSpan >= 3 && k == 0 && localRow == 1;
    bool const tipHold = segmentsPerSpan >= 3 && k == nSec - 2 &&
                         localRow == segmentsPerSpan - 1;
    float const f = rootHold ? 0.0f : (tipHold ? 1.0f : sample);
    if (s == 0) {
        ringT[r] = t;
    }
    int const kP = k > 0 ? k - 1 : k;
    int const kN = k + 2 < nSec ? k + 2 : k + 1;
    // Central differences inside, one-sided at the ends (matches the CPU
    // twin exactly, including the explicit end holding rows).
    bool const first = (k == 0);
    bool const last = (k + 1 == nSec - 1);
    float const m0sc = first ? secScale[k + 1] - secScale[k]
                             : 0.5f * (secScale[k + 1] - secScale[kP]);
    float const m1sc = last ? secScale[k + 1] - secScale[k]
                            : 0.5f * (secScale[kN] - secScale[k]);
    float const m0tw = first ? secTwist[k + 1] - secTwist[k]
                             : 0.5f * (secTwist[k + 1] - secTwist[kP]);
    float const m1tw = last ? secTwist[k + 1] - secTwist[k]
                            : 0.5f * (secTwist[kN] - secTwist[k]);
    float sc = PomadeHermite(secScale[k], secScale[k + 1], m0sc, m1sc, f);
    float tw = PomadeHermite(secTwist[k], secTwist[k + 1], m0tw, m1tw, f);
    if (!(sc > 1e-6f)) {
        sc = 1e-6f;
    }
    float const ct = cosf(tw), st = sinf(tw);
    float cp[3];
    PomadeEvalCenter(cx, cy, cz, nCv, t, cp);
    PomadeFrame fr;
    PomadeNlerpFrame(frames, nCv, t, &fr);
    float nA[3] = {fr.nx, fr.ny, fr.nz};
    float bA[3] = {fr.bx, fr.by, fr.bz};
    size_t const oU = size_t(k) * size_t(ringVerts) + size_t(s);
    size_t const oU1 = size_t(k + 1) * size_t(ringVerts) + size_t(s);
    size_t const oUP = size_t(kP) * size_t(ringVerts) + size_t(s);
    size_t const oUN = size_t(kN) * size_t(ringVerts) + size_t(s);
    float const m0u =
        first ? secU[oU1] - secU[oU] : 0.5f * (secU[oU1] - secU[oUP]);
    float const m1u =
        last ? secU[oU1] - secU[oU] : 0.5f * (secU[oUN] - secU[oU]);
    float const m0v =
        first ? secV[oU1] - secV[oU] : 0.5f * (secV[oU1] - secV[oUP]);
    float const m1v =
        last ? secV[oU1] - secV[oU] : 0.5f * (secV[oUN] - secV[oU]);
    float uu = PomadeHermite(secU[oU], secU[oU1], m0u, m1u, f) * sc;
    float vv = PomadeHermite(secV[oU], secV[oU1], m0v, m1v, f) * sc;
    float const ru = uu * ct - vv * st;
    float const rvv = uu * st + vv * ct;
    size_t const o = size_t(v) * 3;
    positions[o + 0] = cp[0] + nA[0] * ru + bA[0] * rvv;
    positions[o + 1] = cp[1] + nA[1] * ru + bA[1] * rvv;
    positions[o + 2] = cp[2] + nA[2] * ru + bA[2] * rvv;
    if (rootOffsets) {
        float const weight = rootHold ? 1.0f :
            PomadeRootAttachmentWeight(t - secT[0], secT[1] - secT[0]);
        for (int axis = 0; axis < 3; ++axis)
            positions[o + axis] += weight * rootOffsets[size_t(s) * 3 + axis];
    }
    float nn[3] = {nA[0] * ru + bA[0] * rvv, nA[1] * ru + bA[1] * rvv,
                   nA[2] * ru + bA[2] * rvv};
    float const nl = PomadeLen3(nn);
    if (nl > 1e-12f) {
        normals[o + 0] = nn[0] / nl;
        normals[o + 1] = nn[1] / nl;
        normals[o + 2] = nn[2] / nl;
    } else {
        normals[o + 0] = fr.tx;
        normals[o + 1] = fr.ty;
        normals[o + 2] = fr.tz;
    }
}

__device__ bool _RootFarEnough(PomadeDeviceRoot const *roots, int begin,
                               int end, float x, float y, float minDist2)
{
    for (int i = begin; i < end; ++i) {
        float const dx = x - roots[i].ru, dy = y - roots[i].rv;
        if (dx * dx + dy * dy < minDist2) {
            return false;
        }
    }
    return true;
}

__global__ void _RootSampleDiscKernel(int tubeId, int seed, float radius,
                                      float centerX, float centerY,
                                      float centerZ, PomadeFrame rootFrame,
                                      int count, PomadeDeviceRoot *roots,
                                      int frozen)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    if (count <= 0) {
        return;
    }
    float const twoPi = 6.28318530717958647692f;
    float const minDist =
        count > 1 ? 0.9f * sqrtf(3.14159265f / float(count)) : 0.0f;
    float const minDist2 = minDist * minDist;
    uint64_t const key =
        (uint64_t(uint32_t(tubeId)) << 32) | uint64_t(uint32_t(seed));
    int const budget = count * 40 + 64;
    int accepted = frozen;
    for (int cand = 0; cand < budget && accepted < count; ++cand) {
        float const u1 =
            PomadeHash01(uint64_t(cand) * 2 + key, kPomadeSaltRoot);
        float const u2 = PomadeHash01(
            uint64_t(cand) * 2 + 1 + (key ^ 0x85ebca6bu), kPomadeSaltRoot);
        float const rr = sqrtf(u1);
        float const th = twoPi * u2;
        float const ru = rr * cosf(th), rv = rr * sinf(th);
        if (minDist2 > 0.0f &&
            !_RootFarEnough(roots, 0, accepted, ru, rv, minDist2)) {
            continue;
        }
        PomadeDeviceRoot root;
        root.faceId = -1;
        root.u = ru;
        root.v = rv;
        root.ru = ru;
        root.rv = rv;
        root.px = centerX + (rootFrame.nx * ru + rootFrame.bx * rv) * radius;
        root.py = centerY + (rootFrame.ny * ru + rootFrame.by * rv) * radius;
        root.pz = centerZ + (rootFrame.nz * ru + rootFrame.bz * rv) * radius;
        roots[accepted] = root;
        ++accepted;
    }
    for (int i = accepted; i < count; ++i) {
        float const u1 =
            PomadeHash01(uint64_t(budget + i) * 2 + key, kPomadeSaltRoot);
        float const u2 = PomadeHash01(
            uint64_t(budget + i) * 2 + 1 + (key ^ 0x85ebca6bu),
            kPomadeSaltRoot);
        float const rr = sqrtf(u1);
        float const th = twoPi * u2;
        PomadeDeviceRoot root;
        root.faceId = -1;
        root.ru = rr * cosf(th);
        root.rv = rr * sinf(th);
        root.u = root.ru;
        root.v = root.rv;
        root.px =
            centerX + (rootFrame.nx * root.ru + rootFrame.bx * root.rv) *
                          radius;
        root.py =
            centerY + (rootFrame.ny * root.ru + rootFrame.by * root.rv) *
                          radius;
        root.pz =
            centerZ + (rootFrame.nz * root.ru + rootFrame.bz * root.rv) *
                          radius;
        roots[i] = root;
    }
}

__global__ void _RootSampleMeshKernel(
    float const *points, int const *faceCounts, int const *faceIndices,
    int const *faceOffsets, int faceCount, int const *regionFaces,
    int regionFaceCount, float density, int tubeId, int seed, float rootCX,
    float rootCY, float rootCZ, PomadeFrame rootFrame, float rootRadius,
    PomadeDeviceRoot *roots, int maxRoots, int frozen, int *outCount)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    uint64_t const key =
        (uint64_t(uint32_t(tubeId)) << 32) | uint64_t(uint32_t(seed));
    float nA[3] = {rootFrame.nx, rootFrame.ny, rootFrame.nz};
    float bA[3] = {rootFrame.bx, rootFrame.by, rootFrame.bz};
    float rootC[3] = {rootCX, rootCY, rootCZ};
    int fresh = 0;
    for (int rf = 0; rf < regionFaceCount; ++rf) {
        int const f = regionFaces[rf];
        if (f < 0 || f >= faceCount) {
            continue;
        }
        int const nv = faceCounts[f];
        int const off = faceOffsets[f];
        if (nv < 3) {
            continue;
        }
        float const *v0 = points + size_t(faceIndices[off]) * 3;
        float area = 0.0f;
        for (int k = 1; k + 1 < nv; ++k) {
            float const *v1 = points + size_t(faceIndices[off + k]) * 3;
            float const *v2 = points + size_t(faceIndices[off + k + 1]) * 3;
            float e1[3], e2[3], cr[3];
            PomadeSub3(v1, v0, e1);
            PomadeSub3(v2, v0, e2);
            PomadeCross3(e1, e2, cr);
            area += 0.5f * PomadeLen3(cr);
        }
        float const expected = density * area;
        int take = int(expected);
        float const frac = expected - float(take);
        if (PomadeHash01(key ^ (uint64_t(f) * 0x9E3779B1u), kPomadeSaltRoot) <
            frac) {
            ++take;
        }
        float const minDist =
            take > 1 ? 0.9f * sqrtf(3.14159265f / float(take)) *
                           sqrtf(area / 3.14159265f) / rootRadius
                     : 0.0f;
        float const minDist2 = minDist * minDist;
        int const budget = take * 40 + 16;
        int const faceBase = frozen + fresh;
        int accepted = 0;
        for (int cand = 0; cand < budget && accepted < take; ++cand) {
            uint64_t const ck = key ^ (uint64_t(f) * 0x9E3779B1u) ^
                                (uint64_t(cand) * 0x85EBCA77u);
            float const ba = PomadeHash01(ck, kPomadeSaltRoot);
            float const bb = PomadeHash01(ck ^ 0xC2B2AE35u, kPomadeSaltRoot);
            // Barycentric emit (same op order as the CPU twin).
            float const bc = 1.0f - ba - bb;
            int tri = int(ba * float(nv - 2));
            tri = tri < 0 ? 0 : (tri > nv - 3 ? nv - 3 : tri);
            float const *t1 = points + size_t(faceIndices[off + tri]) * 3;
            float const *t2 = points + size_t(faceIndices[off + tri + 1]) * 3;
            float const *t3 = points + size_t(faceIndices[off + tri + 2]) * 3;
            float const w1 = fabsf(bb), w2 = fabsf(bc);
            float const wsum = fabsf(ba) + w1 + w2;
            float const i0 = wsum > 0.0f ? fabsf(ba) / wsum : 1.0f / 3.0f;
            float const i1 = wsum > 0.0f ? w1 / wsum : 1.0f / 3.0f;
            float const i2 = wsum > 0.0f ? w2 / wsum : 1.0f / 3.0f;
            float p[3] = {t1[0] * i0 + t2[0] * i1 + t3[0] * i2,
                          t1[1] * i0 + t2[1] * i1 + t3[1] * i2,
                          t1[2] * i0 + t2[2] * i1 + t3[2] * i2};
            float d[3];
            PomadeSub3(p, rootC, d);
            float ru = PomadeDot3(d, nA) / rootRadius;
            float rv = PomadeDot3(d, bA) / rootRadius;
            float const rl = sqrtf(ru * ru + rv * rv);
            if (rl > 1.0f) {
                ru /= rl;
                rv /= rl;
            }
            if (minDist2 > 0.0f &&
                !_RootFarEnough(roots, faceBase, faceBase + accepted, ru, rv,
                                minDist2)) {
                continue;
            }
            if (faceBase + accepted >= maxRoots) {
                break;
            }
            PomadeDeviceRoot root;
            root.faceId = f;
            root.u = i1;
            root.v = i2;
            root.px = p[0];
            root.py = p[1];
            root.pz = p[2];
            root.ru = ru;
            root.rv = rv;
            roots[faceBase + accepted] = root;
            ++accepted;
        }
        for (int i = accepted; i < take; ++i) {
            if (faceBase + i >= maxRoots) {
                break;
            }
            uint64_t const ck = key ^ (uint64_t(f) * 0x9E3779B1u) ^
                                (uint64_t(budget + i) * 0x85EBCA77u);
            float const ba = PomadeHash01(ck, kPomadeSaltRoot);
            float const bb = PomadeHash01(ck ^ 0xC2B2AE35u, kPomadeSaltRoot);
            float const bc = 1.0f - ba - bb;
            int tri = int(ba * float(nv - 2));
            tri = tri < 0 ? 0 : (tri > nv - 3 ? nv - 3 : tri);
            float const *t1 = points + size_t(faceIndices[off + tri]) * 3;
            float const *t2 = points + size_t(faceIndices[off + tri + 1]) * 3;
            float const *t3 = points + size_t(faceIndices[off + tri + 2]) * 3;
            float const w1 = fabsf(bb), w2 = fabsf(bc);
            float const wsum = fabsf(ba) + w1 + w2;
            float const i0 = wsum > 0.0f ? fabsf(ba) / wsum : 1.0f / 3.0f;
            float const i1 = wsum > 0.0f ? w1 / wsum : 1.0f / 3.0f;
            float const i2 = wsum > 0.0f ? w2 / wsum : 1.0f / 3.0f;
            float p[3] = {t1[0] * i0 + t2[0] * i1 + t3[0] * i2,
                          t1[1] * i0 + t2[1] * i1 + t3[1] * i2,
                          t1[2] * i0 + t2[2] * i1 + t3[2] * i2};
            float d[3];
            PomadeSub3(p, rootC, d);
            float ru = PomadeDot3(d, nA) / rootRadius;
            float rv = PomadeDot3(d, bA) / rootRadius;
            float const rl = sqrtf(ru * ru + rv * rv);
            if (rl > 1.0f) {
                ru /= rl;
                rv /= rl;
            }
            PomadeDeviceRoot root;
            root.faceId = f;
            root.u = i1;
            root.v = i2;
            root.px = p[0];
            root.py = p[1];
            root.pz = p[2];
            root.ru = ru;
            root.rv = rv;
            roots[faceBase + i] = root;
        }
        fresh += take;
        if (frozen + fresh >= maxRoots) {
            fresh = maxRoots - frozen;
            break;
        }
    }
    *outCount = frozen + fresh;
}

__global__ void _GuideFillKernel(
    float const *cx, float const *cy, float const *cz, int nCv,
    PomadeFrame const *frames, float const *secT, float const *secU,
    float const *secV, float const *secScale, float const *secTwist,
    int nSec, int ringVerts, PomadeDeviceRoot const *roots,
    PomadeGuideMaterialBinding const *bindings, int guideCount,
    float edgeBias, float const *profilePairs, int profilePairCount,
    float t0, float span, float *out, float *lengths, int cvCount)
{
    int const g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= guideCount) {
        return;
    }
    PomadeDeviceRoot const root = roots[g];
    float const rootRadius = bindings[size_t(g) * size_t(cvCount)].edgeRadius;
    float length = PomadeEvalLengthProfile(profilePairs, profilePairCount,
                                          rootRadius);
    if (!(length > 0.0f)) {
        length = 0.0f;
    }
    if (length > 1.0f) {
        length = 1.0f;
    }
    lengths[g] = length;
    for (int c = 0; c < cvCount; ++c) {
        float const s = float(c) / float(cvCount - 1);
        float const t = t0 + span * s * length;
        size_t const o = (size_t(g) * size_t(cvCount) + size_t(c)) * 3;
        if (c == 0) {
            out[o + 0] = root.px;
            out[o + 1] = root.py;
            out[o + 2] = root.pz;
            continue;
        }
        float cp[3];
        PomadeEvalCenter(cx, cy, cz, nCv, t, cp);
        PomadeFrame fr;
        PomadeNlerpFrame(frames, nCv, t, &fr);
        int k = 0;
        while (k + 1 < nSec - 1 && secT[k + 1] < t) {
            ++k;
        }
        float f = (secT[k + 1] > secT[k]) ? (t - secT[k]) /
                                                (secT[k + 1] - secT[k])
                                          : 0.0f;
        f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
        int const kp = k > 0 ? k - 1 : k;
        int const kn = k + 2 < nSec ? k + 2 : k + 1;
        bool const first = k == 0;
        bool const last = k + 1 == nSec - 1;
        float const m0sc = first ? secScale[k + 1] - secScale[k]
                                 : 0.5f * (secScale[k + 1] - secScale[kp]);
        float const m1sc = last ? secScale[k + 1] - secScale[k]
                                : 0.5f * (secScale[kn] - secScale[k]);
        float const m0tw = first ? secTwist[k + 1] - secTwist[k]
                                 : 0.5f * (secTwist[k + 1] - secTwist[kp]);
        float const m1tw = last ? secTwist[k + 1] - secTwist[k]
                                : 0.5f * (secTwist[kn] - secTwist[k]);
        float sc = PomadeHermite(secScale[k], secScale[k + 1], m0sc, m1sc, f);
        if (!(sc > 1e-6f)) sc = 1e-6f;
        float const tw = PomadeHermite(secTwist[k], secTwist[k + 1], m0tw,
                                      m1tw, f);
        float const ct = cosf(tw), st = sinf(tw);
        PomadeGuideMaterialBinding const binding =
            bindings[size_t(g) * size_t(cvCount) + size_t(c)];
        int const slots[3] = {binding.slot0, binding.slot1, binding.slot2};
        float const weights[3] = {binding.w0, binding.w1, binding.w2};
        float u = 0.0f, v = 0.0f;
        for (int q = 0; q < 3; ++q) {
            int const slot = slots[q];
            if (slot < 0 || slot >= ringVerts) return;
            int const a = k * ringVerts + slot;
            int const b = (k + 1) * ringVerts + slot;
            int const p = kp * ringVerts + slot;
            int const n = kn * ringVerts + slot;
            float const m0u = first ? secU[b] - secU[a]
                                    : 0.5f * (secU[b] - secU[p]);
            float const m1u = last ? secU[b] - secU[a]
                                   : 0.5f * (secU[n] - secU[a]);
            float const m0v = first ? secV[b] - secV[a]
                                    : 0.5f * (secV[b] - secV[p]);
            float const m1v = last ? secV[b] - secV[a]
                                   : 0.5f * (secV[n] - secV[a]);
            u += weights[q] * PomadeHermite(secU[a], secU[b], m0u, m1u, f);
            v += weights[q] * PomadeHermite(secV[a], secV[b], m0v, m1v, f);
        }
        u *= sc; v *= sc;
        float const ru = u * ct - v * st;
        float const rv = u * st + v * ct;
        out[o + 0] = cp[0] + fr.nx * ru + fr.bx * rv;
        out[o + 1] = cp[1] + fr.ny * ru + fr.by * rv;
        out[o + 2] = cp[2] + fr.nz * ru + fr.bz * rv;
        float const remain = 1.0f - s * length;
        out[o + 0] += binding.rootDx * remain;
        out[o + 1] += binding.rootDy * remain;
        out[o + 2] += binding.rootDz * remain;
    }
}

__global__ void _GuideResampleKernel(float const *points, int const *offsets,
                                     int const *counts, int guideCount,
                                     int cvCount, float rd0, float rd1,
                                     float rd2, float rd3, float rd4,
                                     float rd5, float rd6, float rd7,
                                     float rd8, float *out, int *outCounts,
                                     double *outFrames)
{
    int const g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= guideCount) {
        return;
    }
    int const n = counts[g];
    if (n < 2 || n > 64) {
        outCounts[g] = -1;
        return;
    }
    float const *src = points + size_t(offsets[g]) * 3;
    float cum[64];
    cum[0] = 0.0f;
    for (int i = 1; i < n; ++i) {
        float dx = src[size_t(i) * 3 + 0] - src[size_t(i - 1) * 3 + 0];
        float dy = src[size_t(i) * 3 + 1] - src[size_t(i - 1) * 3 + 1];
        float dz = src[size_t(i) * 3 + 2] - src[size_t(i - 1) * 3 + 2];
        cum[i] = cum[i - 1] + sqrtf(dx * dx + dy * dy + dz * dz);
    }
    float const total = cum[n - 1];
    for (int c = 0; c < cvCount; ++c) {
        float const s = float(c) / float(cvCount - 1) * total;
        int seg = 0;
        while (seg + 1 < n - 1 && cum[seg + 1] < s) {
            ++seg;
        }
        float const c0 = cum[seg], c1 = cum[seg + 1];
        float f = (c1 > c0) ? (s - c0) / (c1 - c0) : 0.0f;
        f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
        size_t const o = (size_t(g) * size_t(cvCount) + size_t(c)) * 3;
        out[o + 0] = src[size_t(seg) * 3 + 0] +
                     (src[size_t(seg + 1) * 3 + 0] - src[size_t(seg) * 3 + 0]) *
                         f;
        out[o + 1] = src[size_t(seg) * 3 + 1] +
                     (src[size_t(seg + 1) * 3 + 1] - src[size_t(seg) * 3 + 1]) *
                         f;
        out[o + 2] = src[size_t(seg) * 3 + 2] +
                     (src[size_t(seg + 1) * 3 + 2] - src[size_t(seg) * 3 + 2]) *
                         f;
    }
    outCounts[g] = cvCount;
    size_t const f = size_t(g) * 16;
    outFrames[f + 0] = rd0;
    outFrames[f + 1] = rd1;
    outFrames[f + 2] = rd2;
    outFrames[f + 3] = 0.0;
    outFrames[f + 4] = rd3;
    outFrames[f + 5] = rd4;
    outFrames[f + 6] = rd5;
    outFrames[f + 7] = 0.0;
    outFrames[f + 8] = rd6;
    outFrames[f + 9] = rd7;
    outFrames[f + 10] = rd8;
    outFrames[f + 11] = 0.0;
    outFrames[f + 12] = src[0];
    outFrames[f + 13] = src[1];
    outFrames[f + 14] = src[2];
    outFrames[f + 15] = 1.0;
}

struct _PickViewProj {
    float m[16];
};

__device__ PomadeDevicePickBest _PickBetter(PomadeDevicePickBest a,
                                           PomadeDevicePickBest b)
{
    if (a.hit && !b.hit) {
        return a;
    }
    if (b.hit && !a.hit) {
        return b;
    }
    if (!a.hit) {
        return a;
    }
    if (b.distPx < a.distPx) {
        return b;
    }
    if (b.distPx > a.distPx) {
        return a;
    }
    return b.depth < a.depth ? b : a;
}

__global__ void _PickBlockKernel(float const *positions, int candidateCount,
                                 _PickViewProj vp, int w, int h, float x,
                                 float y, float radiusPx,
                                 PomadeDevicePickBest *blockBest)
{
    __shared__ PomadeDevicePickBest shared[256];
    int const tid = threadIdx.x;
    int const i = blockIdx.x * blockDim.x + tid;
    PomadeDevicePickBest mine;
    mine.index = -1;
    mine.hit = 0;
    if (i < candidateCount) {
        float const *p = positions + size_t(i) * 3;
        float px, py, ndcZ;
        if (PomadeProjectPoint(p, vp.m, w, h, &px, &py, &ndcZ)) {
            float const dx = px - x, dy = py - y;
            float const d2 = dx * dx + dy * dy;
            if (d2 <= radiusPx * radiusPx) {
                mine.index = i;
                mine.distPx = sqrtf(d2);
                mine.depth = ndcZ;
                mine.hit = 1;
            }
        }
    }
    shared[tid] = mine;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = _PickBetter(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        blockBest[blockIdx.x] = shared[0];
    }
}

__global__ void _SelectMaskKernel(float const *positions,
                                  int candidateCount, _PickViewProj vp,
                                  int w, int h, float x0, float y0, float x1,
                                  float y1, float const *polygonXY,
                                  int polygonCount, unsigned char *mask)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= candidateCount) {
        return;
    }
    unsigned char hit = 0;
    float px, py, ndcZ;
    if (PomadeProjectPoint(positions + size_t(i) * 3, vp.m, w, h, &px, &py,
                          &ndcZ)) {
        hit = (polygonCount >= 3 && polygonXY)
                  ? (PomadePointInPolygon(px, py, polygonXY, polygonCount)
                         ? 1
                         : 0)
                  : (PomadePointInRect(px, py, x0, y0, x1, y1) ? 1 : 0);
    }
    mask[i] = hit;
}

__global__ void _PickFinalizeKernel(PomadeDevicePickBest const *blockBest,
                                    int blockCount, PomadeDevicePickBest *best)
{
    __shared__ PomadeDevicePickBest shared[256];
    int const tid = threadIdx.x;
    PomadeDevicePickBest mine;
    mine.index = -1;
    mine.hit = 0;
    if (tid < blockCount) {
        mine = blockBest[tid];
    }
    shared[tid] = mine;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = _PickBetter(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        best[0] = shared[0];
    }
}

}  // namespace

bool PomadeLaunchCenterFrames(float const *deviceCenterX,
                             float const *deviceCenterY,
                             float const *deviceCenterZ, int nCv,
                             PomadeFrame *deviceFrames, cudaStream_t stream,
                             char *errBuf, size_t errBufLen)
{
    if (!deviceCenterX || !deviceCenterY || !deviceCenterZ || !deviceFrames ||
        nCv < 2) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    _CenterFramesKernel<<<1, 1, 0, stream>>>(deviceCenterX, deviceCenterY,
                                             deviceCenterZ, nCv, deviceFrames);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "center frames launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchTubeTessellate(
    float const *deviceCenterX, float const *deviceCenterY,
    float const *deviceCenterZ, int nCv, PomadeFrame const *deviceFrames,
    float const *deviceSectionT, float const *deviceSectionU,
    float const *deviceSectionV, float const *deviceSectionScale,
    float const *deviceSectionTwist, int nSec, int ringVerts,
    int segmentsPerSpan, float *devicePositions, float *deviceNormals,
    float *deviceRingT, cudaStream_t stream, char *errBuf, size_t errBufLen,
    float const *deviceRootOffsets)
{
    if (!deviceCenterX || !deviceCenterY || !deviceCenterZ || !deviceFrames ||
        !deviceSectionT || !deviceSectionU || !deviceSectionV ||
        !deviceSectionScale || !deviceSectionTwist || !devicePositions ||
        !deviceNormals || !deviceRingT || nCv < 2 || nSec < 2 ||
        ringVerts < 3 || ringVerts > 32 || segmentsPerSpan < 1) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    int const nRings = (nSec - 1) * segmentsPerSpan + 1;
    int const vertexCount = nRings * ringVerts;
    int const block = 256;
    int const grid = (vertexCount + block - 1) / block;
    _TubeTessellateKernel<<<grid, block, 0, stream>>>(
        deviceCenterX, deviceCenterY, deviceCenterZ, nCv, deviceFrames,
        deviceSectionT, deviceSectionU, deviceSectionV, deviceSectionScale,
        deviceSectionTwist, nSec, ringVerts, nRings, segmentsPerSpan, devicePositions,
        deviceNormals, deviceRingT, deviceRootOffsets);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "tube tessellate launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchRootSampleDisc(int tubeId, int seed, float radius,
                               float centerX, float centerY, float centerZ,
                               PomadeFrame const &rootFrame, int count,
                               PomadeDeviceRoot *deviceRoots, int frozen,
                               cudaStream_t stream, char *errBuf,
                               size_t errBufLen)
{
    if (!deviceRoots || count < 0 || frozen < 0 || frozen > count ||
        !(radius > 0.0f)) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    if (count == 0) {
        return true;
    }
    _RootSampleDiscKernel<<<1, 1, 0, stream>>>(tubeId, seed, radius, centerX,
                                               centerY, centerZ, rootFrame,
                                               count, deviceRoots, frozen);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "root sample launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchRootSampleMesh(
    float const *devicePoints, int const *deviceFaceCounts,
    int const *deviceFaceIndices, int const *deviceFaceOffsets, int faceCount,
    int const *deviceRegionFaces, int regionFaceCount, float density,
    int tubeId, int seed, float rootCenterX, float rootCenterY,
    float rootCenterZ, PomadeFrame const &rootFrame, float rootRadius,
    PomadeDeviceRoot *deviceRoots, int maxRoots, int frozen,
    int *deviceOutCount, cudaStream_t stream, char *errBuf, size_t errBufLen)
{
    if (!devicePoints || !deviceFaceCounts || !deviceFaceIndices ||
        !deviceFaceOffsets || !deviceRegionFaces || !deviceRoots ||
        !deviceOutCount || faceCount <= 0 || regionFaceCount <= 0 ||
        !(density >= 0.0f) || frozen < 0 || frozen >= maxRoots ||
        !(rootRadius > 0.0f)) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    _RootSampleMeshKernel<<<1, 1, 0, stream>>>(
        devicePoints, deviceFaceCounts, deviceFaceIndices, deviceFaceOffsets,
        faceCount, deviceRegionFaces, regionFaceCount, density, tubeId, seed,
        rootCenterX, rootCenterY, rootCenterZ, rootFrame, rootRadius,
        deviceRoots, maxRoots, frozen, deviceOutCount);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "root sample launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchGuideFill(
    float const *deviceCenterX, float const *deviceCenterY,
    float const *deviceCenterZ, int nCv, PomadeFrame const *deviceFrames,
    float const *deviceSectionT, float const *deviceSectionU,
    float const *deviceSectionV, float const *deviceSectionScale,
    float const *deviceSectionTwist, int nSec, int ringVerts,
    PomadeDeviceRoot const *deviceRoots,
    PomadeGuideMaterialBinding const *deviceBindings, int guideCount,
    float edgeBias, float const *deviceProfilePairs, int profilePairCount,
    float *deviceOut, float *deviceLengths, int cvCount, cudaStream_t stream,
    char *errBuf, size_t errBufLen)
{
    if (!deviceCenterX || !deviceCenterY || !deviceCenterZ || !deviceFrames ||
        !deviceSectionT || !deviceSectionU || !deviceSectionV ||
        !deviceSectionScale || !deviceSectionTwist || !deviceRoots ||
        !deviceBindings || !deviceOut || !deviceLengths || nCv < 2 ||
        nSec < 2 || ringVerts < 3 || ringVerts > 32 || guideCount < 0 ||
        cvCount < 2 || cvCount > 64 || edgeBias < -1.0f || edgeBias > 1.0f ||
        profilePairCount < 0 || (profilePairCount > 0 && !deviceProfilePairs)) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    if (guideCount == 0) {
        return true;
    }
    float t0 = 0.0f, t1 = 1.0f;
    if (cudaMemcpyAsync(&t0, deviceSectionT, sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&t1, deviceSectionT + nSec - 1, sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "section range readback failed");
        }
        return false;
    }
    float const span = t1 > t0 ? t1 - t0 : 1.0f;
    int const block = 128;
    int const grid = (guideCount + block - 1) / block;
    _GuideFillKernel<<<grid, block, 0, stream>>>(
        deviceCenterX, deviceCenterY, deviceCenterZ, nCv, deviceFrames,
        deviceSectionT, deviceSectionU, deviceSectionV, deviceSectionScale,
        deviceSectionTwist, nSec, ringVerts, deviceRoots, deviceBindings,
        guideCount, edgeBias, deviceProfilePairs, profilePairCount, t0, span,
        deviceOut, deviceLengths, cvCount);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "guide fill launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchGuideResample(float const *devicePoints,
                              int const *deviceOffsets,
                              int const *deviceCounts, int guideCount,
                              int cvCount, float const rootDirs[9],
                              float *deviceOut, int *deviceOutCounts,
                              double *deviceOutFrames, cudaStream_t stream,
                              char *errBuf, size_t errBufLen)
{
    if (!devicePoints || !deviceOffsets || !deviceCounts || !rootDirs ||
        !deviceOut || !deviceOutCounts || !deviceOutFrames || guideCount < 0 ||
        cvCount < 2 || cvCount > 64) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    if (guideCount == 0) {
        return true;
    }
    int const block = 128;
    int const grid = (guideCount + block - 1) / block;
    _GuideResampleKernel<<<grid, block, 0, stream>>>(
        devicePoints, deviceOffsets, deviceCounts, guideCount, cvCount,
        rootDirs[0], rootDirs[1], rootDirs[2], rootDirs[3], rootDirs[4],
        rootDirs[5], rootDirs[6], rootDirs[7], rootDirs[8], deviceOut,
        deviceOutCounts, deviceOutFrames);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "guide resample launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchPickReduce(float const *devicePositions, int candidateCount,
                           float const viewProj[16], int w, int h, float x,
                           float y, float radiusPx,
                           PomadeDevicePickBest *deviceBest,
                           PomadeDevicePickBest *deviceScratch,
                           cudaStream_t stream, char *errBuf,
                           size_t errBufLen)
{
    if (!devicePositions || !viewProj || !deviceBest || !deviceScratch ||
        candidateCount <= 0 || w <= 0 || h <= 0 || !(radiusPx >= 0.0f)) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    _PickViewProj vp;
    for (int i = 0; i < 16; ++i) {
        vp.m[i] = viewProj[i];
    }
    int const block = 256;
    int const grid = (candidateCount + block - 1) / block;
    _PickBlockKernel<<<grid, block, 0, stream>>>(
        devicePositions, candidateCount, vp, w, h, x, y, radiusPx,
        deviceScratch);
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "pick block launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    // The finalize is one block of 256 over the block bests; larger grids
    // finalize in slices (still one scalar readback at the end).
    int const slices = (grid + block - 1) / block;
    for (int s = 0; s < slices; ++s) {
        int const off = s * block;
        int const n = grid - off < block ? grid - off : block;
        _PickFinalizeKernel<<<1, block, 0, stream>>>(deviceScratch + off, n,
                                                     deviceBest + s);
        launch = cudaGetLastError();
        if (launch != cudaSuccess) {
            if (errBuf && errBufLen > 0) {
                std::snprintf(errBuf, errBufLen, "pick finalize launch: %s",
                              cudaGetErrorString(launch));
            }
            return false;
        }
    }
    if (slices > 1) {
        // Fold the slice bests into slot 0 (one more single-block pass).
        _PickFinalizeKernel<<<1, block, 0, stream>>>(deviceBest, slices,
                                                     deviceScratch);
        launch = cudaGetLastError();
        if (launch != cudaSuccess) {
            if (errBuf && errBufLen > 0) {
                std::snprintf(errBuf, errBufLen, "pick fold launch: %s",
                              cudaGetErrorString(launch));
            }
            return false;
        }
        if (cudaMemcpyAsync(deviceBest, deviceScratch,
                            sizeof(PomadeDevicePickBest),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            if (errBuf && errBufLen > 0) {
                std::snprintf(errBuf, errBufLen, "pick fold copy failed");
            }
            return false;
        }
    }
    return true;
}

bool PomadeLaunchSelectMask(float const *devicePositions, int candidateCount,
                           float const viewProj[16], int w, int h, float x0,
                           float y0, float x1, float y1,
                           float const *devicePolygonXY, int polygonCount,
                           unsigned char *deviceMask, cudaStream_t stream,
                           char *errBuf, size_t errBufLen)
{
    if (!devicePositions || !viewProj || !deviceMask || candidateCount <= 0 ||
        w <= 0 || h <= 0) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    if (polygonCount > 0 && (!devicePolygonXY || polygonCount < 3)) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "lasso needs >= 3 points");
        }
        return false;
    }
    _PickViewProj vp;
    for (int i = 0; i < 16; ++i) {
        vp.m[i] = viewProj[i];
    }
    int const block = 256;
    int const grid = (candidateCount + block - 1) / block;
    _SelectMaskKernel<<<grid, block, 0, stream>>>(
        devicePositions, candidateCount, vp, w, h, x0, y0, x1, y1,
        devicePolygonXY, polygonCount, deviceMask);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "select mask launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchSmoothnessScores(float const *deviceCenterX,
                                 float const *deviceCenterY,
                                 float const *deviceCenterZ, int nCv,
                                 float *deviceScores, cudaStream_t stream,
                                 char *errBuf, size_t errBufLen)
{
    if (!deviceCenterX || !deviceCenterY || !deviceCenterZ ||
        !deviceScores || nCv <= 0) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    int const block = 256;
    int const grid = (nCv + block - 1) / block;
    _SmoothnessKernel<<<grid, block, 0, stream>>>(
        deviceCenterX, deviceCenterY, deviceCenterZ, nCv, deviceScores);
    cudaError_t const launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "smoothness launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

bool PomadeLaunchTubeIntersect(PomadeDeviceRootChart const *deviceCharts,
                              int const *deviceTubeIds,
                              int const *deviceParentIds, int tubeCount,
                              int *devicePairHits, int *deviceBroadHits,
                              int *deviceFlags, cudaStream_t stream,
                              char *errBuf, size_t errBufLen)
{
    if (!deviceCharts || !deviceTubeIds || !deviceParentIds ||
        !devicePairHits || !deviceBroadHits || !deviceFlags ||
        tubeCount < 0) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "null device pointer or range");
        }
        return false;
    }
    if (tubeCount == 0) {
        return true;
    }
    int const block = 256;
    int const pairs = tubeCount * tubeCount;
    int const pairGrid = (pairs + block - 1) / block;
    _IntersectPairKernel<<<pairGrid, block, 0, stream>>>(
        deviceCharts, deviceTubeIds, deviceParentIds, tubeCount,
        devicePairHits, deviceBroadHits);
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "intersect pair launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    int const flagGrid = (tubeCount + block - 1) / block;
    _IntersectFlagKernel<<<flagGrid, block, 0, stream>>>(
        devicePairHits, tubeCount, deviceFlags);
    launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        if (errBuf && errBufLen > 0) {
            std::snprintf(errBuf, errBufLen, "intersect flag launch: %s",
                          cudaGetErrorString(launch));
        }
        return false;
    }
    return true;
}

// -- K6/K7/K14 hierarchy device lanes (plan/17 §4.1) --------------------------
//
// Device mirrors of the file-static helpers in pomadeHierarchy.cpp, in the
// same operation order and with the same float/double split, over
// fixed-capacity block storage instead of std::vector. Every structural
// check the CPU twins make (ValidateTube, PomadeValidateSubdivide, the K6
// layout contract) runs on the host while packing, so the kernels report
// only the runtime failures: a degenerate root ring, an empty edge side,
// an empty sub-region, a degenerate union ring, a degenerate rescale.

namespace {

// One tube inside a packed batch. Centers take nCv slots from centerBegin,
// sections nSec slots from sectionBegin, rings nSec * ringVerts slots from
// ringBegin (section-major).
struct PomadeHTube {
    int centerBegin;
    int nCv;
    int sectionBegin;
    int nSec;
    int ringBegin;
    int ringVerts;
};

// Read-only view of one packed batch.
struct PomadeHSpan {
    float const *cx;
    float const *cy;
    float const *cz;
    float const *secT;
    float const *secScale;
    float const *secTwist;
    float const *ringU;
    float const *ringV;
};

// Writable view of one packed batch.
struct PomadeHOut {
    float *cx;
    float *cy;
    float *cz;
    float *secT;
    float *secScale;
    float *secTwist;
    float *ringU;
    float *ringV;
};

// Runtime failures the kernels report; the host maps them to the CPU twins'
// messages. Structural failures never reach the device.
enum PomadeHStatus {
    kPomadeHOk = 0,
    kPomadeHDegenerateRoot = 1,
    kPomadeHEdgeEmpty = 2,
    kPomadeHEmptySubRegion = 3,
    kPomadeHDegenerateUnion = 4,
    kPomadeHRescale = 5,
};

struct PomadeHPt {
    float u;
    float v;
};

// Placed ring coords: the K5 spelling (scale, then twist rotation).
__device__ void _HPlaceRing(float const *u, float const *v, int n, float scale,
                            float twist, PomadeHPt *out)
{
    float const sc = scale > 1e-6f ? scale : 1e-6f;
    float const ct = cosf(twist), st = sinf(twist);
    for (int i = 0; i < n; ++i) {
        float const uu = u[i] * sc, vv = v[i] * sc;
        out[i].u = uu * ct - vv * st;
        out[i].v = uu * st + vv * ct;
    }
}

__device__ float _HMeanRadius(PomadeHPt const *pts, int n)
{
    if (n <= 0) {
        return 0.0f;
    }
    double su = 0.0, sv = 0.0;
    for (int i = 0; i < n; ++i) {
        su += pts[i].u;
        sv += pts[i].v;
    }
    double const cu = su / double(n), cv = sv / double(n);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        double const du = double(pts[i].u) - cu, dv = double(pts[i].v) - cv;
        acc += sqrt(du * du + dv * dv);
    }
    return float(acc / double(n));
}

__device__ void _HRingMean(PomadeHPt const *pts, int n, double *meanU,
                           double *meanV)
{
    if (n <= 0) {
        *meanU = 0.0;
        *meanV = 0.0;
        return;
    }
    double su = 0.0, sv = 0.0;
    for (int i = 0; i < n; ++i) {
        su += pts[i].u;
        sv += pts[i].v;
    }
    *meanU = su / double(n);
    *meanV = sv / double(n);
}

// Section ring interpolated at t: the InterpSectionAt spelling (Hermite per
// slot with the K5 bracketing and end rules).
__device__ void _HInterpSectionAt(PomadeHSpan const &a, PomadeHTube const &tb,
                                  float t, PomadeHPt *placed, float *radius)
{
    int const nSec = tb.nSec;
    int const rv = tb.ringVerts;
    float const *T = a.secT + tb.sectionBegin;
    float const *S = a.secScale + tb.sectionBegin;
    float const *W = a.secTwist + tb.sectionBegin;
    float const *U = a.ringU + tb.ringBegin;
    float const *V = a.ringV + tb.ringBegin;
    int k = 0;
    while (k + 1 < nSec - 1 && T[k + 1] < t) {
        ++k;
    }
    int const kP = k > 0 ? k - 1 : k;
    int const kN = k + 2 < nSec ? k + 2 : k + 1;
    float const dt = T[k + 1] > T[k] ? T[k + 1] - T[k] : 1.0f;
    float f = (t - T[k]) / dt;
    f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
    bool const first = (k == 0), last = (k + 1 == nSec - 1);
    float const m0sc = first ? S[k + 1] - S[k] : 0.5f * (S[k + 1] - S[kP]);
    float const m1sc = last ? S[k + 1] - S[k] : 0.5f * (S[kN] - S[k]);
    float const m0tw = first ? W[k + 1] - W[k] : 0.5f * (W[k + 1] - W[kP]);
    float const m1tw = last ? W[k + 1] - W[k] : 0.5f * (W[kN] - W[k]);
    float sc = PomadeHermite(S[k], S[k + 1], m0sc, m1sc, f);
    float const tw = PomadeHermite(W[k], W[k + 1], m0tw, m1tw, f);
    if (!(sc > 1e-6f)) {
        sc = 1e-6f;
    }
    float const ct = cosf(tw), st = sinf(tw);
    for (int s = 0; s < rv; ++s) {
        float const au = U[k * rv + s], bu = U[(k + 1) * rv + s];
        float const pu = U[kP * rv + s], nu = U[kN * rv + s];
        float const av = V[k * rv + s], bv = V[(k + 1) * rv + s];
        float const pv = V[kP * rv + s], nv = V[kN * rv + s];
        float const uu =
            PomadeHermite(au, bu, first ? bu - au : 0.5f * (bu - pu),
                         last ? bu - au : 0.5f * (nu - au), f) *
            sc;
        float const vv =
            PomadeHermite(av, bv, first ? bv - av : 0.5f * (bv - pv),
                         last ? bv - av : 0.5f * (nv - av), f) *
            sc;
        placed[s].u = uu * ct - vv * st;
        placed[s].v = uu * st + vv * ct;
    }
    if (radius) {
        *radius = _HMeanRadius(placed, rv);
    }
}

// Deterministic k-means over pts: the KMeans spelling (k-means++ init from
// the Pomade hash stream keyed by key, Lloyd's to convergence, empty-cluster
// rescue to the farthest point).
__device__ void _HKMeans(PomadeHPt const *pts, int n, int k,
                         unsigned long long key, int *assign,
                         PomadeHPt *centroids, double *best)
{
    for (int i = 0; i < n; ++i) {
        assign[i] = 0;
    }
    unsigned long long stream = key;
    centroids[0] =
        pts[(unsigned)(PomadeHash01(stream++, kPomadeSaltRoot) * float(n)) %
            (unsigned)n];
    for (int i = 0; i < n; ++i) {
        best[i] = 1.7976931348623157e308;  // numeric_limits<double>::max()
    }
    for (int c = 1; c < k; ++c) {
        for (int i = 0; i < n; ++i) {
            double const du = double(pts[i].u) - double(centroids[c - 1].u);
            double const dv = double(pts[i].v) - double(centroids[c - 1].v);
            double const d2 = du * du + dv * dv;
            if (d2 < best[i]) {
                best[i] = d2;
            }
        }
        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            total += best[i];
        }
        int pick = 0;
        if (total > 0.0) {
            double const r =
                double(PomadeHash01(stream++, kPomadeSaltRoot)) * total;
            double acc = 0.0;
            for (int i = 0; i < n; ++i) {
                acc += best[i];
                if (acc >= r) {
                    pick = i;
                    break;
                }
            }
        } else {
            pick = int(PomadeHash01(stream++, kPomadeSaltRoot) * float(n)) % n;
        }
        centroids[c] = pts[pick];
    }
    for (int it = 0; it < 64; ++it) {  // kKMeansIters
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            int bc = 0;
            double bd = 1.7976931348623157e308;
            for (int c = 0; c < k; ++c) {
                double const du = double(pts[i].u) - double(centroids[c].u);
                double const dv = double(pts[i].v) - double(centroids[c].v);
                double const d2 = du * du + dv * dv;
                if (d2 < bd) {  // strict: lowest index wins ties
                    bd = d2;
                    bc = c;
                }
            }
            if (assign[i] != bc) {
                assign[i] = bc;
                changed = true;
            }
        }
        for (int c = 0; c < k; ++c) {
            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                if (assign[i] == c) {
                    ++cnt;
                }
            }
            if (cnt == 0) {
                int fi = 0;
                double fd = -1.0;
                for (int i = 0; i < n; ++i) {
                    int const oc = assign[i];
                    double const du =
                        double(pts[i].u) - double(centroids[oc].u);
                    double const dv =
                        double(pts[i].v) - double(centroids[oc].v);
                    double const d2 = du * du + dv * dv;
                    if (d2 > fd) {
                        fd = d2;
                        fi = i;
                    }
                }
                assign[fi] = c;
                changed = true;
            }
        }
        for (int c = 0; c < k; ++c) {
            int cnt = 0;
            double au = 0.0, av = 0.0;
            for (int i = 0; i < n; ++i) {
                if (assign[i] == c) {
                    ++cnt;
                    au += pts[i].u;
                    av += pts[i].v;
                }
            }
            if (cnt > 0) {
                centroids[c].u = float(au / double(cnt));
                centroids[c].v = float(av / double(cnt));
            }
        }
        if (!changed) {
            break;
        }
    }
}

// Sutherland-Hodgman clip of poly to a*u + b*v + c >= 0 (keepPositive) or
// <= 0. Writes `out`, returns the vertex count (at most n + 1).
__device__ int _HClipHalfPlane(PomadeHPt const *poly, int n, double a, double b,
                               double c, bool keepPositive, PomadeHPt *out)
{
    if (n == 0) {
        return 0;
    }
    int m = 0;
    for (int i = 0; i < n; ++i) {
        PomadeHPt const P = poly[i];
        PomadeHPt const Q = poly[(i + 1) % n];
        double const sp = a * double(P.u) + b * double(P.v) + c;
        double const sq = a * double(Q.u) + b * double(Q.v) + c;
        bool const inP = keepPositive ? sp >= 0.0 : sp <= 0.0;
        bool const inQ = keepPositive ? sq >= 0.0 : sq <= 0.0;
        if (inQ) {
            if (!inP) {
                double const f = (sp != sq) ? sp / (sp - sq) : 0.0;
                out[m].u =
                    float(double(P.u) + (double(Q.u) - double(P.u)) * f);
                out[m].v =
                    float(double(P.v) + (double(Q.v) - double(P.v)) * f);
                ++m;
            }
            out[m++] = Q;
        } else if (inP) {
            double const f = (sp != sq) ? sp / (sp - sq) : 0.0;
            out[m].u = float(double(P.u) + (double(Q.u) - double(P.u)) * f);
            out[m].v = float(double(P.v) + (double(Q.v) - double(P.v)) * f);
            ++m;
        }
    }
    return m;
}

// Clip poly to the Voronoi cell of cent[c]; the result lands in `out` and
// `scratch` is the ping-pong buffer. Both hold at least n + k vertices.
__device__ int _HClipToCell(PomadeHPt const *poly, int n, PomadeHPt const *cent,
                            int k, int c, PomadeHPt *out, PomadeHPt *scratch)
{
    int cnt = n;
    for (int i = 0; i < n; ++i) {
        out[i] = poly[i];
    }
    for (int o = 0; o < k; ++o) {
        if (o == c) {
            continue;
        }
        double const ax = double(cent[o].u) - double(cent[c].u);
        double const ay = double(cent[o].v) - double(cent[c].v);
        double const rhs = (double(cent[o].u) * double(cent[o].u) +
                            double(cent[o].v) * double(cent[o].v)) -
                           (double(cent[c].u) * double(cent[c].u) +
                            double(cent[c].v) * double(cent[c].v));
        cnt = _HClipHalfPlane(out, cnt, -2.0 * ax, -2.0 * ay, rhs, true,
                              scratch);
        for (int i = 0; i < cnt; ++i) {
            out[i] = scratch[i];
        }
        if (cnt == 0) {
            break;
        }
    }
    return cnt;
}

__device__ double _HPolyArea(PomadeHPt const *poly, int n)
{
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        PomadeHPt const P = poly[i];
        PomadeHPt const Q = poly[(i + 1) % n];
        acc += double(P.u) * double(Q.v) - double(Q.u) * double(P.v);
    }
    return 0.5 * acc;
}

// Subdivide edges longer than maxLen until the budget is reached (inserted
// points stay on the polygon). `poly` holds at least `budget` vertices.
// tieRel > 0 splits the lowest-index edge within that relative margin of
// the longest (the twin's K14 tie rule, see SubdivideLongEdges).
__device__ void _HSubdivideLongEdges(PomadeHPt *poly, int *n, double maxLen,
                                     int budget, double tieRel = 0.0)
{
    for (;;) {
        int const m = *n;
        if (m >= budget) {
            return;
        }
        int bi = -1;
        double bl = maxLen;
        for (int i = 0; i < m; ++i) {
            PomadeHPt const P = poly[i];
            PomadeHPt const Q = poly[(i + 1) % m];
            double const du = double(Q.u) - double(P.u);
            double const dv = double(Q.v) - double(P.v);
            double const len = sqrt(du * du + dv * dv);
            if (len > bl) {
                bl = len;
                bi = i;
            }
        }
        if (bi < 0) {
            return;
        }
        if (tieRel > 0.0) {
            double const tied = fmax(maxLen, bl * (1.0 - tieRel));
            for (int i = 0; i < bi; ++i) {
                PomadeHPt const P = poly[i];
                PomadeHPt const Q = poly[(i + 1) % m];
                double const du = double(Q.u) - double(P.u);
                double const dv = double(Q.v) - double(P.v);
                if (sqrt(du * du + dv * dv) > tied) {
                    bi = i;
                    break;
                }
            }
        }
        PomadeHPt const P = poly[bi];
        PomadeHPt const Q = poly[(bi + 1) % m];
        PomadeHPt mid;
        mid.u = float((double(P.u) + double(Q.u)) * 0.5);
        mid.v = float((double(P.v) + double(Q.v)) * 0.5);
        for (int i = m; i > bi + 1; --i) {
            poly[i] = poly[i - 1];
        }
        poly[bi + 1] = mid;
        *n = m + 1;
    }
}

// Twin of DropCoincidentVertices: drop consecutive (cyclic) clip vertices
// closer than tol, a merged pair keeping the original parent corner. The
// twin tracks corners by provenance; here a corner is a vertex bitwise
// equal to one of the placed parent ring's `corners` (a clip copies its
// inside vertices verbatim, and a fresh intersection never lands on one
// but at an endpoint, where it is that corner).
__device__ void _HDropCoincident(PomadeHPt *poly, int *n,
                                 PomadeHPt const *corners, int cornerCount,
                                 double tol, PomadeHPt *scratch)
{
    int const m = *n;
    if (m < 4 || !(tol > 0.0)) {
        return;
    }
    double const tol2 = tol * tol;
    auto close = [tol2](PomadeHPt const &a, PomadeHPt const &b) {
        double const du = double(a.u) - double(b.u);
        double const dv = double(a.v) - double(b.v);
        return du * du + dv * dv < tol2;
    };
    auto corner = [&](PomadeHPt const &a) {
        for (int j = 0; j < cornerCount; ++j) {
            if (corners[j].u == a.u && corners[j].v == a.v) {
                return true;
            }
        }
        return false;
    };
    int k = 0;
    for (int i = 0; i < m; ++i) {
        PomadeHPt const p = poly[i];
        if (k > 0 && close(scratch[k - 1], p)) {
            if (!corner(scratch[k - 1]) && corner(p)) {
                scratch[k - 1] = p;
            }
            continue;
        }
        scratch[k++] = p;
    }
    while (k > 3 && close(scratch[k - 1], scratch[0])) {
        if (!corner(scratch[0]) && corner(scratch[k - 1])) {
            scratch[0] = scratch[k - 1];
        }
        --k;
    }
    if (k >= 3) {
        for (int i = 0; i < k; ++i) {
            poly[i] = scratch[i];
        }
        *n = k;
    }
}

// Twin of AlignRingToReference: whether `poly` (n points) must reverse to
// take the reference ring's winding, and the cyclic shift that then best
// matches it slot for slot (least summed squared distance about each
// ring's own mean, normalised by its mean radius; near-ties keep the
// smallest shift). Emit slot i as poly[idx], idx = (i + shift) % n, taken
// from the end when reversed.
__device__ void _HAlignToReference(PomadeHPt const *poly, int n,
                                   float const *refU, float const *refV,
                                   bool *reversed, int *shift)
{
    *reversed = false;
    *shift = 0;
    if (n < 3) {
        return;
    }
    double area = 0.0, refArea = 0.0;
    for (int i = 0; i < n; ++i) {
        int const j = (i + 1) % n;
        area += double(poly[i].u) * double(poly[j].v) -
                double(poly[j].u) * double(poly[i].v);
        refArea += double(refU[i]) * double(refV[j]) -
                   double(refU[j]) * double(refV[i]);
    }
    area *= 0.5;
    refArea *= 0.5;
    bool const rev =
        (area > 0.0 && refArea < 0.0) || (area < 0.0 && refArea > 0.0);
    double pu[kPomadeDeviceMaxRing], pv[kPomadeDeviceMaxRing];
    double ru[kPomadeDeviceMaxRing], rv[kPomadeDeviceMaxRing];
    for (int i = 0; i < n; ++i) {
        PomadeHPt const p = poly[rev ? n - 1 - i : i];
        pu[i] = double(p.u);
        pv[i] = double(p.v);
        ru[i] = double(refU[i]);
        rv[i] = double(refV[i]);
    }
    auto normalise = [n](double *u, double *v) {
        double mu = 0.0, mv = 0.0;
        for (int i = 0; i < n; ++i) {
            mu += u[i];
            mv += v[i];
        }
        mu /= double(n);
        mv /= double(n);
        double radius = 0.0;
        for (int i = 0; i < n; ++i) {
            double const du = u[i] - mu;
            double const dv = v[i] - mv;
            radius += sqrt(du * du + dv * dv);
        }
        radius /= double(n);
        double const inv = radius > 0.0 ? 1.0 / radius : 1.0;
        for (int i = 0; i < n; ++i) {
            u[i] = (u[i] - mu) * inv;
            v[i] = (v[i] - mv) * inv;
        }
    };
    normalise(pu, pv);
    normalise(ru, rv);
    double costs[kPomadeDeviceMaxRing];
    double best = 1e300;
    for (int k = 0; k < n; ++k) {
        double cost = 0.0;
        for (int i = 0; i < n; ++i) {
            int const j = (i + k) % n;
            double const du = pu[j] - ru[i];
            double const dv = pv[j] - rv[i];
            cost += du * du + dv * dv;
        }
        costs[k] = cost;
        best = fmin(best, cost);
    }
    double const tied = best + 1e-4 * double(n);
    int k = 0;
    while (costs[k] > tied) {
        ++k;
    }
    *reversed = rev;
    *shift = k;
}

// Arc-length resample of a closed polygon to `count` points. The running
// cumulative sum reproduces the twin's precomputed cum[] term for term.
__device__ void _HResamplePoly(PomadeHPt const *poly, int n, int count,
                               PomadeHPt *out)
{
    if (n == 0 || count <= 0) {
        return;
    }
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        PomadeHPt const P = poly[i];
        PomadeHPt const Q = poly[(i + 1) % n];
        double const du = double(Q.u) - double(P.u);
        double const dv = double(Q.v) - double(P.v);
        total += sqrt(du * du + dv * dv);
    }
    if (!(total > 0.0)) {
        for (int i = 0; i < count; ++i) {
            out[i] = poly[0];
        }
        return;
    }
    int seg = 0;
    double cumSeg = 0.0;
    double segLen;
    {
        PomadeHPt const P = poly[0];
        PomadeHPt const Q = poly[1 % n];
        double const du = double(Q.u) - double(P.u);
        double const dv = double(Q.v) - double(P.v);
        segLen = sqrt(du * du + dv * dv);
    }
    for (int i = 0; i < count; ++i) {
        double const s = total * double(i) / double(count);
        while (seg + 1 < n && cumSeg + segLen < s) {
            cumSeg += segLen;
            ++seg;
            PomadeHPt const P = poly[seg % n];
            PomadeHPt const Q = poly[(seg + 1) % n];
            double const du = double(Q.u) - double(P.u);
            double const dv = double(Q.v) - double(P.v);
            segLen = sqrt(du * du + dv * dv);
        }
        double const s0 = cumSeg, s1 = cumSeg + segLen;
        double const f = (s1 > s0) ? (s - s0) / (s1 - s0) : 0.0;
        PomadeHPt const P = poly[seg % n];
        PomadeHPt const Q = poly[(seg + 1) % n];
        out[i].u = float(double(P.u) + (double(Q.u) - double(P.u)) * f);
        out[i].v = float(double(P.v) + (double(Q.v) - double(P.v)) * f);
    }
}

// Piecewise-linear arc-length resample of a center curve to outN points.
__device__ void _HResampleCenter(float const *cx, float const *cy,
                                 float const *cz, int n, int outN, float *ox,
                                 float *oy, float *oz, double *cum)
{
    cum[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        double const dx = double(cx[i]) - double(cx[i - 1]);
        double const dy = double(cy[i]) - double(cy[i - 1]);
        double const dz = double(cz[i]) - double(cz[i - 1]);
        cum[i] = cum[i - 1] + sqrt(dx * dx + dy * dy + dz * dz);
    }
    double const total = cum[n - 1];
    int seg = 0;
    for (int i = 0; i < outN; ++i) {
        double const s = outN > 1 ? total * double(i) / double(outN - 1) : 0.0;
        while (seg + 1 < n - 1 && cum[seg + 1] < s) {
            ++seg;
        }
        double const s0 = cum[seg], s1 = cum[seg + 1];
        double const f = (s1 > s0) ? (s - s0) / (s1 - s0) : 0.0;
        ox[i] = float(double(cx[seg]) +
                      (double(cx[seg + 1]) - double(cx[seg])) * f);
        oy[i] = float(double(cy[seg]) +
                      (double(cy[seg + 1]) - double(cy[seg])) * f);
        oz[i] = float(double(cz[seg]) +
                      (double(cz[seg + 1]) - double(cz[seg])) * f);
    }
}

__device__ float _HCenterArcLength(float const *cx, float const *cy,
                                   float const *cz, int nCv)
{
    double acc = 0.0;
    for (int i = 1; i < nCv; ++i) {
        double const dx = double(cx[i]) - double(cx[i - 1]);
        double const dy = double(cy[i]) - double(cy[i - 1]);
        double const dz = double(cz[i]) - double(cz[i - 1]);
        acc += sqrt(dx * dx + dy * dy + dz * dz);
    }
    return float(acc);
}

// Incremental K4 walk: the PomadeCenterFramesCpu loop body, one CV per call,
// so a caller that consumes frames in index order stores no frame array.
struct PomadeHRmf {
    float tPrev[3];
    float n[3];
    int i;
};

__device__ void _HRmfBegin(float const *cx, float const *cy, float const *cz,
                           int nCv, PomadeHRmf *w)
{
    PomadeCenterTangent(cx, cy, cz, nCv, 0, w->tPrev);
    PomadePerp3(w->tPrev, w->n);
    w->i = 0;
}

__device__ void _HRmfNext(float const *cx, float const *cy, float const *cz,
                          int nCv, PomadeHRmf *w, PomadeFrame *out)
{
    int const i = w->i;
    float t[3];
    PomadeCenterTangent(cx, cy, cz, nCv, i, t);
    if (i > 0) {
        float const p0[3] = {cx[i - 1], cy[i - 1], cz[i - 1]};
        float const p1[3] = {cx[i], cy[i], cz[i]};
        PomadeReflectNormal(p0, p1, w->tPrev, t, w->n);
        w->tPrev[0] = t[0];
        w->tPrev[1] = t[1];
        w->tPrev[2] = t[2];
    }
    float b[3];
    PomadeCross3(t, w->n, b);
    out->tx = t[0];
    out->ty = t[1];
    out->tz = t[2];
    out->nx = w->n[0];
    out->ny = w->n[1];
    out->nz = w->n[2];
    out->bx = b[0];
    out->by = b[1];
    out->bz = b[2];
    w->i = i + 1;
}

__device__ void _HCenterFrames(float const *cx, float const *cy,
                               float const *cz, int nCv, PomadeFrame *frames)
{
    PomadeHRmf w;
    _HRmfBegin(cx, cy, cz, nCv, &w);
    for (int i = 0; i < nCv; ++i) {
        _HRmfNext(cx, cy, cz, nCv, &w, frames + i);
    }
}

// PomadeRescaleCenterLength in place: uniform scaling about the root.
__device__ bool _HRescaleCenterLength(float *cx, float *cy, float *cz,
                                      int nCv, float targetLen)
{
    if (!(targetLen > 0.0f)) {
        return false;
    }
    double total = 0.0;
    for (int i = 1; i < nCv; ++i) {
        double const dx = double(cx[i]) - double(cx[i - 1]);
        double const dy = double(cy[i]) - double(cy[i - 1]);
        double const dz = double(cz[i]) - double(cz[i - 1]);
        total += sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (!(total > 0.0)) {
        return false;
    }
    if (float(total) == targetLen) {
        return true;  // already there: the twin copies bit-exactly
    }
    double const k = double(targetLen) / total;
    float const x0 = cx[0], y0 = cy[0], z0 = cz[0];
    for (int i = 0; i < nCv; ++i) {
        cx[i] = float(double(x0) + (double(cx[i]) - double(x0)) * k);
        cy[i] = float(double(y0) + (double(cy[i]) - double(y0)) * k);
        cz[i] = float(double(z0) + (double(cz[i]) - double(z0)) * k);
    }
    return true;
}

constexpr int kHClipCap = kPomadeDeviceMaxRing + kPomadeDeviceMaxChildren + 8;
// The twin's kSplitTieRel: K14's densify/pad edge-length tie margin.
constexpr double kHSplitTieRel = 1e-4;
constexpr int kHCloudCap = kPomadeDeviceMaxRing * kPomadeDeviceMaxChildren;

// -- K14: subdivide -----------------------------------------------------------
//
// One block per parent, one lane per child. Lane 0 partitions the root ring
// into shared memory; every lane then derives its own child exactly as
// PomadeSubdivideTubeCpu does, two passes over the sections (the first sizes
// the child ring, the second emits it) so no lane holds every clip.

__global__ void _HSubdivideKernel(PomadeHSpan in, PomadeHTube const *parents,
                                  PomadeFrame const *parentFrames,
                                  PomadeSubdivideDesc const *params,
                                  int const *parentTubeIds, int parentCount,
                                  PomadeHOut out, PomadeHTube *outTubes,
                                  int outCvStride, int outSecStride,
                                  int outRingStride, int *status,
                                  int *statusAux)
{
    __shared__ PomadeHPt s_root[kPomadeDeviceMaxRing];
    __shared__ int s_assign[kPomadeDeviceMaxRing];
    __shared__ double s_best[kPomadeDeviceMaxRing];
    __shared__ PomadeHPt s_cent[kPomadeDeviceMaxChildren];
    __shared__ double s_rootMeanU;
    __shared__ double s_rootMeanV;
    __shared__ double s_spacing;
    __shared__ float s_rootRadius;
    __shared__ int s_err;
    __shared__ PomadeHPt s_a[kPomadeDeviceMaxChildren][kHClipCap];
    __shared__ PomadeHPt s_b[kPomadeDeviceMaxChildren][kHClipCap];
    __shared__ PomadeHPt s_c[kPomadeDeviceMaxChildren][kHClipCap];

    int const p = blockIdx.x;
    if (p >= parentCount) {
        return;
    }
    int const lane = threadIdx.x;
    PomadeHTube const parent = parents[p];
    PomadeSubdivideDesc const par = params[p];
    int const nCv = parent.nCv;
    int const nSec = parent.nSec;
    int const prv = parent.ringVerts;

    if (lane == 0) {
        s_err = kPomadeHOk;
        _HPlaceRing(in.ringU + parent.ringBegin, in.ringV + parent.ringBegin,
                    prv, in.secScale[parent.sectionBegin],
                    in.secTwist[parent.sectionBegin], s_root);
        _HRingMean(s_root, prv, &s_rootMeanU, &s_rootMeanV);
        s_rootRadius = _HMeanRadius(s_root, prv);
        if (!(s_rootRadius > 0.0f)) {
            s_err = kPomadeHDegenerateRoot;
        } else if (par.splitMode == PomadeSplit_KMeans) {
            unsigned long long const key =
                ((unsigned long long)(unsigned)parentTubeIds[p] << 32) ^
                (unsigned long long)(unsigned)par.seed ^ 0x484B3134ull;
            _HKMeans(s_root, prv, par.count, key, s_assign, s_cent, s_best);
        } else {
            for (int i = 0; i < prv; ++i) {
                double const s = double(par.edgeA) * double(s_root[i].u) +
                                 double(par.edgeB) * double(s_root[i].v) +
                                 double(par.edgeC);
                s_assign[i] = (s >= 0.0) ? 1 : 0;
            }
            for (int c = 0; c < 2; ++c) {
                double su = 0.0, sv = 0.0;
                int cnt = 0;
                for (int i = 0; i < prv; ++i) {
                    if (s_assign[i] == c) {
                        su += s_root[i].u;
                        sv += s_root[i].v;
                        ++cnt;
                    }
                }
                if (cnt == 0) {
                    s_err = kPomadeHEdgeEmpty;
                } else {
                    s_cent[c].u = float(su / double(cnt));
                    s_cent[c].v = float(sv / double(cnt));
                }
            }
        }
        if (s_err == kPomadeHOk) {
            double spacing = 0.0;
            for (int i = 0; i < prv; ++i) {
                PomadeHPt const P = s_root[i];
                PomadeHPt const Q = s_root[(i + 1) % prv];
                double const du = double(Q.u) - double(P.u);
                double const dv = double(Q.v) - double(P.v);
                spacing += sqrt(du * du + dv * dv);
            }
            spacing /= double(prv);
            s_spacing = spacing;
            if (!(spacing > 0.0)) {
                s_err = kPomadeHDegenerateRoot;
            }
        }
        if (s_err != kPomadeHOk) {
            for (int c = 0; c < kPomadeDeviceMaxChildren; ++c) {
                status[p * kPomadeDeviceMaxChildren + c] = s_err;
                statusAux[p * kPomadeDeviceMaxChildren + c] = 0;
            }
        }
    }
    __syncthreads();
    if (s_err != kPomadeHOk || lane >= par.count) {
        return;
    }
    int const c = lane;
    int const oBase = p * kPomadeDeviceMaxChildren + c;
    float const rootRadius = s_rootRadius;
    PomadeHPt *bufA = s_a[c];
    PomadeHPt *bufB = s_b[c];
    PomadeHPt *bufC = s_c[c];

    // Centers: parent center offset by the centroid, scaled by the
    // local-to-root radius ratio (§2.3 step 2).
    float *ocx = out.cx + size_t(oBase) * size_t(outCvStride);
    float *ocy = out.cy + size_t(oBase) * size_t(outCvStride);
    float *ocz = out.cz + size_t(oBase) * size_t(outCvStride);
    for (int i = 0; i < nCv; ++i) {
        float const ti = nCv > 1 ? float(i) / float(nCv - 1) : 0.0f;
        float radiusAt = 0.0f;
        _HInterpSectionAt(in, parent, ti, bufC, &radiusAt);
        float const ratio = rootRadius > 0.0f ? radiusAt / rootRadius : 1.0f;
        PomadeFrame const fr = parentFrames[parent.centerBegin + i];
        float const ox = s_cent[c].u * ratio;
        float const oy = s_cent[c].v * ratio;
        ocx[i] = in.cx[parent.centerBegin + i] + fr.nx * ox + fr.bx * oy;
        ocy[i] = in.cy[parent.centerBegin + i] + fr.ny * ox + fr.by * oy;
        ocz[i] = in.cz[parent.centerBegin + i] + fr.nz * ox + fr.bz * oy;
    }

    int target = 0;
    int failSection = -1;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            if (target > kPomadeDeviceMaxRing) {
                target = kPomadeDeviceMaxRing;  // kMaxRingVerts
            }
        }
        for (int s = 0; s < nSec; ++s) {
            _HPlaceRing(in.ringU + parent.ringBegin + s * prv,
                        in.ringV + parent.ringBegin + s * prv, prv,
                        in.secScale[parent.sectionBegin + s],
                        in.secTwist[parent.sectionBegin + s], bufA);
            double secMeanU = 0.0, secMeanV = 0.0;
            _HRingMean(bufA, prv, &secMeanU, &secMeanV);
            float const secRadius = _HMeanRadius(bufA, prv);
            bool const sameChart =
                (secMeanU == s_rootMeanU && secMeanV == s_rootMeanV &&
                 secRadius == rootRadius);
            int cnt = 0;
            if (par.splitMode == PomadeSplit_KMeans) {
                if (sameChart || !(secRadius > 0.0f)) {
                    cnt = _HClipToCell(bufA, prv, s_cent, par.count, c, bufB,
                                       bufC);
                } else {
                    double const q = double(secRadius) / double(rootRadius);
                    PomadeHPt carried[kPomadeDeviceMaxChildren];
                    for (int o = 0; o < par.count; ++o) {
                        carried[o].u = float(
                            secMeanU + (double(s_cent[o].u) - s_rootMeanU) * q);
                        carried[o].v = float(
                            secMeanV + (double(s_cent[o].v) - s_rootMeanV) * q);
                    }
                    cnt = _HClipToCell(bufA, prv, carried, par.count, c, bufB,
                                       bufC);
                }
            } else {
                double const a = double(par.edgeA);
                double const b = double(par.edgeB);
                double cut = double(par.edgeC);
                if (!sameChart && secRadius > 0.0f) {
                    double const q = double(secRadius) / double(rootRadius);
                    double const rootC =
                        a * s_rootMeanU + b * s_rootMeanV + cut;
                    cut = q * rootC - (a * secMeanU + b * secMeanV);
                }
                cnt = _HClipHalfPlane(bufA, prv, a, b, cut, c == 1, bufB);
            }
            double const clippedArea = fabs(_HPolyArea(bufB, cnt));
            double const emptyTol =
                1e-12 * double(rootRadius) * double(rootRadius);
            if (cnt < 3 || clippedArea <= emptyTol) {
                failSection = s;
                break;
            }
            // bufA still holds the placed parent ring (the corners).
            _HDropCoincident(bufB, &cnt, bufA, prv, 1e-4 * s_spacing, bufC);
            _HSubdivideLongEdges(bufB, &cnt, 2.0 * s_spacing,
                                 kPomadeDeviceMaxRing, kHSplitTieRel);
            if (pass == 0) {
                if (cnt > target) {
                    target = cnt;
                }
                continue;
            }
            // Recenter on the child center (radius ratio at this t).
            float radiusAt = 0.0f;
            _HInterpSectionAt(in, parent, in.secT[parent.sectionBegin + s],
                              bufC, &radiusAt);
            float const ratio =
                rootRadius > 0.0f ? radiusAt / rootRadius : 1.0f;
            float const ou = s_cent[c].u * ratio;
            float const ov = s_cent[c].v * ratio;
            for (int i = 0; i < cnt; ++i) {
                bufB[i].u -= ou;
                bufB[i].v -= ov;
            }
            // Fit to `target`: resample down, exact-length pad up.
            PomadeHPt const *emit = bufB;
            if (cnt > target) {
                _HResamplePoly(bufB, cnt, target, bufC);
                emit = bufC;
            } else {
                for (int i = 0; i < cnt; ++i) {
                    bufA[i] = bufB[i];
                }
                int padded = cnt;
                _HSubdivideLongEdges(bufA, &padded, 0.0, target,
                                     kHSplitTieRel);
                if (padded != target) {
                    _HResamplePoly(bufB, cnt, target, bufC);
                    emit = bufC;
                } else {
                    emit = bufA;
                }
            }
            size_t const ringBase =
                size_t(oBase) * size_t(outRingStride) + size_t(s * target);
            // Slot alignment to the section already emitted before this
            // one (section 0 is the anchor), as the twin does.
            bool reversed = false;
            int shift = 0;
            if (s > 0) {
                _HAlignToReference(emit, target,
                                   out.ringU + ringBase - size_t(target),
                                   out.ringV + ringBase - size_t(target),
                                   &reversed, &shift);
            }
            for (int i = 0; i < target; ++i) {
                int const k = (i + shift) % target;
                PomadeHPt const e = emit[reversed ? target - 1 - k : k];
                out.ringU[ringBase + size_t(i)] = e.u;
                out.ringV[ringBase + size_t(i)] = e.v;
            }
            size_t const secBase = size_t(oBase) * size_t(outSecStride);
            out.secT[secBase + size_t(s)] = in.secT[parent.sectionBegin + s];
            out.secScale[secBase + size_t(s)] = 1.0f;
            out.secTwist[secBase + size_t(s)] = 0.0f;
        }
        if (failSection >= 0) {
            break;
        }
    }
    if (failSection >= 0) {
        status[oBase] = kPomadeHEmptySubRegion;
        statusAux[oBase] = failSection;
        return;
    }
    PomadeHTube ot;
    ot.centerBegin = oBase * outCvStride;
    ot.nCv = nCv;
    ot.sectionBegin = oBase * outSecStride;
    ot.nSec = nSec;
    ot.ringBegin = oBase * outRingStride;
    ot.ringVerts = target;
    outTubes[oBase] = ot;
    status[oBase] = kPomadeHOk;
    statusAux[oBase] = 0;
}

// -- K7: parent average -------------------------------------------------------
//
// Two launches. The first averages the arc-length-resampled child centers
// (one block per group, one lane per child) and walks the parent K4 frames;
// the second refits one section per block against the union of the child
// rings in the parent plane.

__global__ void _HParentCenterKernel(PomadeHSpan in, PomadeHTube const *tubes,
                                     int const *groupBegin,
                                     int const *groupCount, int groups,
                                     float *outCx, float *outCy, float *outCz,
                                     PomadeFrame *outFrames, int outCvStride)
{
    __shared__ float s_rx[kPomadeDeviceMaxChildren][kPomadeDeviceMaxCv];
    __shared__ float s_ry[kPomadeDeviceMaxChildren][kPomadeDeviceMaxCv];
    __shared__ float s_rz[kPomadeDeviceMaxChildren][kPomadeDeviceMaxCv];
    __shared__ double s_cum[kPomadeDeviceMaxChildren][kPomadeDeviceMaxCv];

    int const g = blockIdx.x;
    if (g >= groups) {
        return;
    }
    int const lane = threadIdx.x;
    int const begin = groupBegin[g];
    int const count = groupCount[g];
    int const nCv = tubes[begin].nCv;
    if (lane < count) {
        PomadeHTube const ch = tubes[begin + lane];
        _HResampleCenter(in.cx + ch.centerBegin, in.cy + ch.centerBegin,
                         in.cz + ch.centerBegin, ch.nCv, nCv, s_rx[lane],
                         s_ry[lane], s_rz[lane], s_cum[lane]);
    }
    __syncthreads();
    if (lane != 0) {
        return;
    }
    float *cx = outCx + size_t(g) * size_t(outCvStride);
    float *cy = outCy + size_t(g) * size_t(outCvStride);
    float *cz = outCz + size_t(g) * size_t(outCvStride);
    double const inv = 1.0 / double(count);
    for (int i = 0; i < nCv; ++i) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int c = 0; c < count; ++c) {
            ax += s_rx[c][i];
            ay += s_ry[c][i];
            az += s_rz[c][i];
        }
        cx[i] = float(ax * inv);
        cy[i] = float(ay * inv);
        cz[i] = float(az * inv);
    }
    _HCenterFrames(cx, cy, cz, nCv,
                   outFrames + size_t(g) * size_t(outCvStride));
}

// Stable bottom-up merge sort over the cloud indices, ordered by the twin's
// comparator (angle, then squared radius). A tie is two identical points,
// which the dedup below collapses either way.
__device__ void _HSortCloud(int n, double const *ang, double const *rad,
                            short *idx, short *tmp)
{
    for (int i = 0; i < n; ++i) {
        idx[i] = short(i);
    }
    for (int width = 1; width < n; width *= 2) {
        for (int lo = 0; lo < n; lo += 2 * width) {
            int const mid = lo + width < n ? lo + width : n;
            int const hi = lo + 2 * width < n ? lo + 2 * width : n;
            int a = lo, b = mid, o = lo;
            while (a < mid && b < hi) {
                int const ia = idx[a], ib = idx[b];
                bool const takeB = (ang[ib] != ang[ia]) ? (ang[ib] < ang[ia])
                                                        : (rad[ib] < rad[ia]);
                tmp[o++] = takeB ? short(ib) : short(ia);
                if (takeB) {
                    ++b;
                } else {
                    ++a;
                }
            }
            while (a < mid) {
                tmp[o++] = idx[a++];
            }
            while (b < hi) {
                tmp[o++] = idx[b++];
            }
        }
        for (int i = 0; i < n; ++i) {
            idx[i] = tmp[i];
        }
    }
}

__global__ void _HParentSectionKernel(
    PomadeHSpan in, PomadeHTube const *tubes, int const *groupBegin,
    int const *groupCount, int const *groupSectionBegin, int groups,
    int totalSections, float const *parentCx, float const *parentCy,
    float const *parentCz, PomadeFrame const *parentFrames, PomadeHOut out,
    int outCvStride, int outSecStride, int outRingStride, int *status)
{
    __shared__ PomadeHPt s_cloud[kHCloudCap];
    __shared__ PomadeHPt s_ring[kHCloudCap];
    __shared__ double s_ang[kHCloudCap];
    __shared__ double s_rad[kHCloudCap];
    __shared__ short s_idx[kHCloudCap];
    __shared__ short s_tmp[kHCloudCap];
    // PomadeFrame carries default member initializers, which a __shared__
    // variable may not, so the frame stores are raw aligned bytes.
    __shared__ __align__(16) unsigned char s_cfRaw[kPomadeDeviceMaxChildren *
                                                   kPomadeDeviceMaxCv *
                                                   sizeof(PomadeFrame)];
    __shared__ __align__(16) unsigned char s_pfRaw[sizeof(PomadeFrame)];
    __shared__ int s_off[kPomadeDeviceMaxChildren];
    __shared__ float s_pc[3];
    __shared__ int s_total;
    PomadeFrame *const s_cf = reinterpret_cast<PomadeFrame *>(s_cfRaw);
    PomadeFrame *const s_pf = reinterpret_cast<PomadeFrame *>(s_pfRaw);

    int const item = blockIdx.x;
    if (item >= totalSections) {
        return;
    }
    // Locate (group, section): groupSectionBegin is the prefix sum of each
    // group's section count.
    int g = 0;
    while (g + 1 < groups && groupSectionBegin[g + 1] <= item) {
        ++g;
    }
    int const s = item - groupSectionBegin[g];
    int const lane = threadIdx.x;
    int const begin = groupBegin[g];
    int const count = groupCount[g];
    int const nCv = tubes[begin].nCv;
    int const rv = tubes[begin].ringVerts;
    float const t = in.secT[tubes[begin].sectionBegin + s];

    if (lane == 0) {
        float const *pcx = parentCx + size_t(g) * size_t(outCvStride);
        float const *pcy = parentCy + size_t(g) * size_t(outCvStride);
        float const *pcz = parentCz + size_t(g) * size_t(outCvStride);
        float cp[3];
        PomadeEvalCenter(pcx, pcy, pcz, nCv, t, cp);
        s_pc[0] = cp[0];
        s_pc[1] = cp[1];
        s_pc[2] = cp[2];
        PomadeNlerpFrame(parentFrames + size_t(g) * size_t(outCvStride), nCv, t,
                        s_pf);
        int off = 0;
        for (int c = 0; c < count; ++c) {
            s_off[c] = off;
            off += tubes[begin + c].ringVerts;
        }
        s_total = off;
    }
    __syncthreads();
    if (lane < count) {
        PomadeHTube const ch = tubes[begin + lane];
        _HCenterFrames(in.cx + ch.centerBegin, in.cy + ch.centerBegin,
                       in.cz + ch.centerBegin, ch.nCv,
                       s_cf + size_t(lane) * size_t(kPomadeDeviceMaxCv));
        PomadeHPt *ring = s_cloud + s_off[lane];
        _HInterpSectionAt(in, ch, t, ring, nullptr);
        float cc[3];
        PomadeEvalCenter(in.cx + ch.centerBegin, in.cy + ch.centerBegin,
                        in.cz + ch.centerBegin, ch.nCv, t, cc);
        PomadeFrame cf;
        PomadeNlerpFrame(s_cf + size_t(lane) * size_t(kPomadeDeviceMaxCv),
                        ch.nCv, t, &cf);
        for (int i = 0; i < ch.ringVerts; ++i) {
            float const pu = ring[i].u, pv = ring[i].v;
            float const wx = cc[0] + cf.nx * pu + cf.bx * pv;
            float const wy = cc[1] + cf.ny * pu + cf.by * pv;
            float const wz = cc[2] + cf.nz * pu + cf.bz * pv;
            float const dx = wx - s_pc[0], dy = wy - s_pc[1],
                        dz = wz - s_pc[2];
            ring[i].u = dx * s_pf->nx + dy * s_pf->ny + dz * s_pf->nz;
            ring[i].v = dx * s_pf->bx + dy * s_pf->by + dz * s_pf->bz;
        }
    }
    __syncthreads();
    if (lane != 0) {
        return;
    }
    int const n = s_total;
    double su = 0.0, sv = 0.0;
    for (int i = 0; i < n; ++i) {
        su += s_cloud[i].u;
        sv += s_cloud[i].v;
    }
    double const cu = su / double(n);
    double const cv = sv / double(n);
    for (int i = 0; i < n; ++i) {
        double const du = double(s_cloud[i].u) - cu;
        double const dv = double(s_cloud[i].v) - cv;
        s_ang[i] = atan2(dv, du);
        s_rad[i] = du * du + dv * dv;
    }
    _HSortCloud(n, s_ang, s_rad, s_idx, s_tmp);
    int m = 0;
    for (int i = 0; i < n; ++i) {
        PomadeHPt const p = s_cloud[s_idx[i]];
        if (m == 0 || p.u != s_ring[m - 1].u || p.v != s_ring[m - 1].v) {
            s_ring[m++] = p;
        }
    }
    if (m > 1 && s_ring[0].u == s_ring[m - 1].u &&
        s_ring[0].v == s_ring[m - 1].v) {
        --m;
    }
    if (m < 3) {
        status[item] = kPomadeHDegenerateUnion;
        return;
    }
    if (m < rv) {
        _HSubdivideLongEdges(s_ring, &m, 0.0, rv);
    } else if (m > rv) {
        for (int i = 0; i < rv; ++i) {
            int const idx =
                int((static_cast<long long>(i) * static_cast<long long>(m)) / static_cast<long long>(rv)) % m;
            s_cloud[i] = s_ring[idx];
        }
        for (int i = 0; i < rv; ++i) {
            s_ring[i] = s_cloud[i];
        }
        m = rv;
    }
    size_t const secBase = size_t(g) * size_t(outSecStride);
    out.secT[secBase + size_t(s)] = t;
    out.secScale[secBase + size_t(s)] = 1.0f;
    out.secTwist[secBase + size_t(s)] = 0.0f;
    size_t const ringBase =
        size_t(g) * size_t(outRingStride) + size_t(s) * size_t(rv);
    for (int i = 0; i < rv; ++i) {
        out.ringU[ringBase + size_t(i)] = s_ring[i].u;
        out.ringV[ringBase + size_t(i)] = s_ring[i].v;
    }
    status[item] = kPomadeHOk;
}

// -- K6: hierarchical sculpt, the apply half ----------------------------------
//
// One thread per child. The output actual starts as the new derived shape
// and the stored deltas re-apply in the new K4 frames (or ride rigidly
// under lockChildren), exactly as PomadeHierarchicalSculptApplyCpu does.

__global__ void _HSculptApplyKernel(
    PomadeHSpan dIn, PomadeHTube const *dTubes, PomadeHSpan aIn,
    PomadeHTube const *aTubes, PomadeHSpan oIn, PomadeHTube const *oTubes,
    PomadeHSpan sIn, PomadeHTube const *sTubes, unsigned char const *flags,
    int itemCount, PomadeHOut out, PomadeHOut outDelta, int outCvStride,
    int outSecStride, int outRingStride, int *status)
{
    extern __shared__ PomadeHPt s_scratch[];
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= itemCount) {
        return;
    }
    int const lane = threadIdx.x;
    PomadeHPt *poly = s_scratch + size_t(lane) * size_t(2 * kPomadeDeviceMaxRing);
    PomadeHPt *rs = poly + kPomadeDeviceMaxRing;

    PomadeHTube const D = dTubes[i];
    PomadeHTube const A = aTubes[i];
    PomadeHTube const O = oTubes[i];
    PomadeHTube const S = sTubes[i];
    int const nCv = D.nCv;
    int const nSec = D.nSec;
    int const drv = D.ringVerts;
    bool const lockChildren = (flags[i] & 1u) != 0u;
    bool const preserveLength = (flags[i] & 2u) != 0u;

    float *ocx = out.cx + size_t(i) * size_t(outCvStride);
    float *ocy = out.cy + size_t(i) * size_t(outCvStride);
    float *ocz = out.cz + size_t(i) * size_t(outCvStride);
    size_t const secBase = size_t(i) * size_t(outSecStride);
    size_t const ringBase = size_t(i) * size_t(outRingStride);
    for (int j = 0; j < nCv; ++j) {
        ocx[j] = dIn.cx[D.centerBegin + j];
        ocy[j] = dIn.cy[D.centerBegin + j];
        ocz[j] = dIn.cz[D.centerBegin + j];
    }
    for (int s = 0; s < nSec; ++s) {
        out.secT[secBase + size_t(s)] = dIn.secT[D.sectionBegin + s];
        out.secScale[secBase + size_t(s)] = dIn.secScale[D.sectionBegin + s];
        out.secTwist[secBase + size_t(s)] = dIn.secTwist[D.sectionBegin + s];
        for (int k = 0; k < drv; ++k) {
            out.ringU[ringBase + size_t(s * drv + k)] =
                dIn.ringU[D.ringBegin + s * drv + k];
            out.ringV[ringBase + size_t(s * drv + k)] =
                dIn.ringV[D.ringBegin + s * drv + k];
        }
    }

    if (lockChildren) {
        for (int j = 0; j < nCv; ++j) {
            ocx[j] += aIn.cx[A.centerBegin + j] - oIn.cx[O.centerBegin + j];
            ocy[j] += aIn.cy[A.centerBegin + j] - oIn.cy[O.centerBegin + j];
            ocz[j] += aIn.cz[A.centerBegin + j] - oIn.cz[O.centerBegin + j];
        }
        int const arv = A.ringVerts;
        for (int s = 0; s < nSec; ++s) {
            for (int k = 0; k < arv; ++k) {
                poly[k].u = aIn.ringU[A.ringBegin + s * arv + k] -
                            oIn.ringU[O.ringBegin + s * arv + k];
                poly[k].v = aIn.ringV[A.ringBegin + s * arv + k] -
                            oIn.ringV[O.ringBegin + s * arv + k];
            }
            if (arv == drv) {
                for (int k = 0; k < drv; ++k) {
                    rs[k] = poly[k];
                }
            } else {
                _HResamplePoly(poly, arv, drv, rs);
            }
            for (int k = 0; k < drv; ++k) {
                out.ringU[ringBase + size_t(s * drv + k)] += rs[k].u;
                out.ringV[ringBase + size_t(s * drv + k)] += rs[k].v;
            }
            out.secScale[secBase + size_t(s)] +=
                aIn.secScale[A.sectionBegin + s] -
                oIn.secScale[O.sectionBegin + s];
            out.secTwist[secBase + size_t(s)] +=
                aIn.secTwist[A.sectionBegin + s] -
                oIn.secTwist[O.sectionBegin + s];
        }
        status[i] = kPomadeHOk;
        return;
    }

    float const *dcx = dIn.cx + D.centerBegin;
    float const *dcy = dIn.cy + D.centerBegin;
    float const *dcz = dIn.cz + D.centerBegin;
    {
        PomadeHRmf w;
        _HRmfBegin(dcx, dcy, dcz, nCv, &w);
        for (int j = 0; j < nCv; ++j) {
            PomadeFrame fr;
            _HRmfNext(dcx, dcy, dcz, nCv, &w, &fr);
            float const du = sIn.cx[S.centerBegin + j];
            float const dv = sIn.cy[S.centerBegin + j];
            float const dw = sIn.cz[S.centerBegin + j];
            ocx[j] += fr.nx * du + fr.bx * dv + fr.tx * dw;
            ocy[j] += fr.ny * du + fr.by * dv + fr.ty * dw;
            ocz[j] += fr.nz * du + fr.bz * dv + fr.tz * dw;
        }
    }
    int const srv = S.ringVerts;
    for (int s = 0; s < nSec; ++s) {
        for (int k = 0; k < srv; ++k) {
            poly[k].u = sIn.ringU[S.ringBegin + s * srv + k];
            poly[k].v = sIn.ringV[S.ringBegin + s * srv + k];
        }
        if (srv == drv) {
            for (int k = 0; k < drv; ++k) {
                rs[k] = poly[k];
            }
        } else {
            _HResamplePoly(poly, srv, drv, rs);
        }
        for (int k = 0; k < drv; ++k) {
            out.ringU[ringBase + size_t(s * drv + k)] += rs[k].u;
            out.ringV[ringBase + size_t(s * drv + k)] += rs[k].v;
        }
        out.secScale[secBase + size_t(s)] +=
            sIn.secScale[S.sectionBegin + s];
        out.secTwist[secBase + size_t(s)] +=
            sIn.secTwist[S.sectionBegin + s];
    }
    if (preserveLength) {
        float const oldLen = _HCenterArcLength(aIn.cx + A.centerBegin,
                                               aIn.cy + A.centerBegin,
                                               aIn.cz + A.centerBegin, nCv);
        if (!_HRescaleCenterLength(ocx, ocy, ocz, nCv, oldLen)) {
            status[i] = kPomadeHRescale;
            return;
        }
    }
    // Deltas of the new actual against the new derived, in the new frames.
    float *ddu = outDelta.cx + size_t(i) * size_t(outCvStride);
    float *ddv = outDelta.cy + size_t(i) * size_t(outCvStride);
    float *ddw = outDelta.cz + size_t(i) * size_t(outCvStride);
    {
        PomadeHRmf w;
        _HRmfBegin(dcx, dcy, dcz, nCv, &w);
        for (int j = 0; j < nCv; ++j) {
            PomadeFrame fr;
            _HRmfNext(dcx, dcy, dcz, nCv, &w, &fr);
            float const wx = ocx[j] - dcx[j];
            float const wy = ocy[j] - dcy[j];
            float const wz = ocz[j] - dcz[j];
            ddu[j] = wx * fr.nx + wy * fr.ny + wz * fr.nz;
            ddv[j] = wx * fr.bx + wy * fr.by + wz * fr.bz;
            ddw[j] = wx * fr.tx + wy * fr.ty + wz * fr.tz;
        }
    }
    for (int s = 0; s < nSec; ++s) {
        outDelta.secScale[secBase + size_t(s)] =
            out.secScale[secBase + size_t(s)] -
            dIn.secScale[D.sectionBegin + s];
        outDelta.secTwist[secBase + size_t(s)] =
            out.secTwist[secBase + size_t(s)] -
            dIn.secTwist[D.sectionBegin + s];
        for (int k = 0; k < drv; ++k) {
            outDelta.ringU[ringBase + size_t(s * drv + k)] =
                out.ringU[ringBase + size_t(s * drv + k)] -
                dIn.ringU[D.ringBegin + s * drv + k];
            outDelta.ringV[ringBase + size_t(s * drv + k)] =
                out.ringV[ringBase + size_t(s * drv + k)] -
                dIn.ringV[D.ringBegin + s * drv + k];
        }
    }
    status[i] = kPomadeHOk;
}

// -- host packing and dispatch ------------------------------------------------

bool _HFail(std::string *err, char const *msg)
{
    if (err) {
        *err = msg;
    }
    return false;
}

struct PomadeHPack {
    std::vector<PomadeHTube> tubes;
    std::vector<float> cx, cy, cz;
    std::vector<float> secT, secScale, secTwist;
    std::vector<float> ringU, ringV;
};

bool _HasNonIdentityFrameReference(PomadeTubeDesc const &tube)
{
    static float const identity[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; ++i) {
        if (tube.frameReference[size_t(i)] != identity[i]) {
            return true;
        }
    }
    return false;
}

// ValidateTube from pomadeHierarchy.cpp (same messages) plus the device caps.
bool _HValidate(PomadeTubeDesc const &t, std::string *err)
{
    int const nCv = int(t.centerX.size());
    int const nSec = int(t.sections.size());
    if (nCv < 2 || t.centerY.size() != size_t(nCv) ||
        t.centerZ.size() != size_t(nCv)) {
        return _HFail(err, "hierarchy: bad center CVs");
    }
    if (nSec < 2 || t.ringVerts < 3 || t.ringVerts > 256) {
        return _HFail(err, "hierarchy: bad sections or ringVerts");
    }
    for (auto const &s : t.sections) {
        if (int(s.u.size()) != t.ringVerts ||
            int(s.v.size()) != t.ringVerts) {
            return _HFail(err, "hierarchy: section ring size mismatch");
        }
    }
    for (int i = 1; i < nSec; ++i) {
        if (!(t.sections[size_t(i)].t >= t.sections[size_t(i - 1)].t)) {
            return _HFail(err, "hierarchy: section t must ascend");
        }
    }
    if (nCv > kPomadeDeviceMaxCv || nSec > kPomadeDeviceMaxSec ||
        t.ringVerts > kPomadeDeviceMaxRing) {
        return _HFail(err, "pomade device lane: tube exceeds the device caps");
    }
    return true;
}

void _HAppend(std::vector<float> const &cx, std::vector<float> const &cy,
              std::vector<float> const &cz,
              std::vector<PomadeTubeSection> const &sections, int ringVerts,
              PomadeHPack *p)
{
    PomadeHTube t;
    t.centerBegin = int(p->cx.size());
    t.nCv = int(cx.size());
    t.sectionBegin = int(p->secT.size());
    t.nSec = int(sections.size());
    t.ringBegin = int(p->ringU.size());
    t.ringVerts = ringVerts;
    p->cx.insert(p->cx.end(), cx.begin(), cx.end());
    p->cy.insert(p->cy.end(), cy.begin(), cy.end());
    p->cz.insert(p->cz.end(), cz.begin(), cz.end());
    for (auto const &s : sections) {
        p->secT.push_back(s.t);
        p->secScale.push_back(s.scale);
        p->secTwist.push_back(s.twist);
        p->ringU.insert(p->ringU.end(), s.u.begin(), s.u.end());
        p->ringV.insert(p->ringV.end(), s.v.begin(), s.v.end());
    }
    p->tubes.push_back(t);
}

void _HAppendTube(PomadeTubeDesc const &d, PomadeHPack *p)
{
    _HAppend(d.centerX, d.centerY, d.centerZ, d.sections, d.ringVerts, p);
}

void _HAppendDeltas(PomadeShapeDeltas const &d, PomadeHPack *p)
{
    int const rv = d.sections.empty() ? 0 : int(d.sections.front().u.size());
    _HAppend(d.centerDu, d.centerDv, d.centerDw, d.sections, rv, p);
}

// Grow-only device arena shared by the three hierarchy lanes. A lane is one
// launch per move, and a cudaMalloc/cudaFree pair costs more than the
// kernels do at production sizes, so the slabs are cut once and reused: a
// call bumps through them and resets on the way out. Slabs are never freed
// -- a cudaFree from a static destructor runs after the CUDA context may
// already be gone, which is the DLL-teardown hang this codebase avoids
// elsewhere.
struct PomadeHArena {
    std::vector<std::pair<void *, size_t>> slabs;
    size_t slabIndex = 0;
    size_t used = 0;

    void Reset()
    {
        slabIndex = 0;
        used = 0;
    }

    void *Bump(size_t bytes)
    {
        if (bytes == 0) {
            return nullptr;
        }
        bytes = (bytes + 255u) & ~size_t(255u);
        while (slabIndex < slabs.size()) {
            if (used + bytes <= slabs[slabIndex].second) {
                void *p = static_cast<char *>(slabs[slabIndex].first) + used;
                used += bytes;
                return p;
            }
            ++slabIndex;
            used = 0;
        }
        size_t const cap = bytes > (size_t(1) << 20) ? bytes
                                                     : (size_t(1) << 20);
        void *p = nullptr;
        if (cudaMalloc(&p, cap) != cudaSuccess) {
            slabIndex = slabs.size();
            return nullptr;
        }
        slabs.push_back(std::make_pair(p, cap));
        slabIndex = slabs.size() - 1;
        used = bytes;
        return p;
    }
};

// One kernel context per device: the arena plus an ownership word. A call
// claims its device's context with one CAS and never waits for it; a call
// that finds it claimed fails like an allocation failure, and the model's
// lanes take their bit-exact CPU twin. Contexts are published by CAS on
// first use and deliberately leaked (see above).
struct PomadeKernelContext {
    std::atomic<uint32_t> owned{0};
    PomadeHArena arena;
};

constexpr int kPomadeKernelDevices = 64;
std::atomic<PomadeKernelContext *> g_kernelContexts[kPomadeKernelDevices];

PomadeKernelContext *_ClaimKernelContext()
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
        device >= kPomadeKernelDevices) {
        return nullptr;
    }
    std::atomic<PomadeKernelContext *> &cell = g_kernelContexts[device];
    PomadeKernelContext *ctx = cell.load(std::memory_order_acquire);
    if (!ctx) {
        PomadeKernelContext *fresh = new PomadeKernelContext();
        if (cell.compare_exchange_strong(ctx, fresh,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
            ctx = fresh;
        } else {
            delete fresh;
        }
    }
    uint32_t idle = 0;
    if (!ctx->owned.compare_exchange_strong(idle, 1u,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
        return nullptr;
    }
    return ctx;
}

// Device allocations for one call, handed back to the arena on return.
struct PomadeHScratch {
    PomadeKernelContext *ctx;
    PomadeHArena *arena;
    bool ok = true;

    PomadeHScratch()
        : ctx(_ClaimKernelContext()), arena(ctx ? &ctx->arena : nullptr)
    {
        ok = arena != nullptr;
        if (arena) {
            arena->Reset();
        }
    }

    ~PomadeHScratch()
    {
        if (ctx) {
            arena->Reset();
            ctx->owned.store(0u, std::memory_order_release);
        }
    }
    PomadeHScratch(PomadeHScratch const &) = delete;
    PomadeHScratch &operator=(PomadeHScratch const &) = delete;

    template <class T>
    T *Alloc(size_t n)
    {
        if (!ok || n == 0) {
            return nullptr;
        }
        void *p = arena->Bump(n * sizeof(T));
        if (!p) {
            ok = false;
            return nullptr;
        }
        return static_cast<T *>(p);
    }

    template <class T>
    T *Upload(std::vector<T> const &v, cudaStream_t stream)
    {
        T *p = Alloc<T>(v.size());
        if (!p) {
            return nullptr;
        }
        if (cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        return p;
    }

    bool UploadSpan(PomadeHPack const &pack, cudaStream_t stream,
                    PomadeHSpan *span, PomadeHTube **tubes)
    {
        *tubes = Upload(pack.tubes, stream);
        span->cx = Upload(pack.cx, stream);
        span->cy = Upload(pack.cy, stream);
        span->cz = Upload(pack.cz, stream);
        span->secT = Upload(pack.secT, stream);
        span->secScale = Upload(pack.secScale, stream);
        span->secTwist = Upload(pack.secTwist, stream);
        span->ringU = Upload(pack.ringU, stream);
        span->ringV = Upload(pack.ringV, stream);
        return ok;
    }

    bool AllocOut(size_t cvTotal, size_t secTotal, size_t ringTotal,
                  PomadeHOut *out)
    {
        out->cx = Alloc<float>(cvTotal);
        out->cy = Alloc<float>(cvTotal);
        out->cz = Alloc<float>(cvTotal);
        out->secT = Alloc<float>(secTotal);
        out->secScale = Alloc<float>(secTotal);
        out->secTwist = Alloc<float>(secTotal);
        out->ringU = Alloc<float>(ringTotal);
        out->ringV = Alloc<float>(ringTotal);
        return ok;
    }
};

template <class T>
bool _HDownload(std::vector<T> *host, T const *device, size_t n,
                cudaStream_t stream)
{
    host->resize(n);
    if (n == 0) {
        return true;
    }
    return cudaMemcpyAsync(host->data(), device, n * sizeof(T),
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

std::string _HStatusMessage(int code, int aux, int childIndex)
{
    switch (code) {
    case kPomadeHDegenerateRoot:
        return "PomadeSubdivideTubeCpu: degenerate root ring";
    case kPomadeHEdgeEmpty:
        return "PomadeSubdivideTubeCpu: edge leaves a side empty";
    case kPomadeHEmptySubRegion:
        return "PomadeSubdivideTubeCpu: empty sub-region c=" +
               std::to_string(childIndex) + " s=" + std::to_string(aux) +
               " (raise the ring density or change the seed)";
    case kPomadeHDegenerateUnion:
        return "PomadeParentAverageCpu: degenerate union ring";
    case kPomadeHRescale:
        return "PomadeRescaleCenterLength: degenerate curve";
    default:
        break;
    }
    return "pomade device lane: unknown kernel status " + std::to_string(code);
}

// Read one tube out of a downloaded output pack.
void _HReadTube(std::vector<float> const &cx, std::vector<float> const &cy,
                std::vector<float> const &cz, std::vector<float> const &secT,
                std::vector<float> const &secScale,
                std::vector<float> const &secTwist,
                std::vector<float> const &ringU,
                std::vector<float> const &ringV, size_t cvBase, int nCv,
                size_t secBase, int nSec, size_t ringBase, int ringVerts,
                PomadeTubeDesc *outTube)
{
    outTube->centerX.assign(cx.begin() + cvBase, cx.begin() + cvBase + nCv);
    outTube->centerY.assign(cy.begin() + cvBase, cy.begin() + cvBase + nCv);
    outTube->centerZ.assign(cz.begin() + cvBase, cz.begin() + cvBase + nCv);
    outTube->ringVerts = ringVerts;
    outTube->sections.resize(size_t(nSec));
    for (int s = 0; s < nSec; ++s) {
        PomadeTubeSection &sec = outTube->sections[size_t(s)];
        sec.t = secT[secBase + size_t(s)];
        sec.scale = secScale[secBase + size_t(s)];
        sec.twist = secTwist[secBase + size_t(s)];
        size_t const rb = ringBase + size_t(s) * size_t(ringVerts);
        sec.u.assign(ringU.begin() + rb, ringU.begin() + rb + ringVerts);
        sec.v.assign(ringV.begin() + rb, ringV.begin() + rb + ringVerts);
    }
}

}  // namespace

bool PomadeSubdivideTubesDevice(
    PomadeTubeDesc const *parents,
    std::vector<PomadeFrame> const *parentFrames,
    PomadeSubdivideDesc const *params, int parentCount,
    std::vector<std::vector<PomadeTubeDesc>> *children, cudaStream_t stream,
    std::string *err)
{
    if (!children || (parentCount > 0 && (!parents || !parentFrames ||
                                          !params))) {
        return _HFail(err, "PomadeSubdivideTubesDevice: null argument");
    }
    children->clear();
    if (parentCount <= 0) {
        return true;
    }
    PomadeHPack pack;
    std::vector<PomadeFrame> frames;
    std::vector<PomadeSubdivideDesc> pars;
    std::vector<int> tubeIds;
    int maxCv = 0, maxSec = 0;
    for (int i = 0; i < parentCount; ++i) {
        if (!PomadeValidateSubdivide(params[i], err) ||
            !_HValidate(parents[i], err)) {
            return false;
        }
        if (parentFrames[i].size() != parents[i].centerX.size()) {
            return _HFail(err,
                          "PomadeSubdivideTubeCpu: frames match center CVs");
        }
        _HAppendTube(parents[i], &pack);
        frames.insert(frames.end(), parentFrames[i].begin(),
                      parentFrames[i].end());
        pars.push_back(params[i]);
        tubeIds.push_back(parents[i].tubeId);
        maxCv = std::max(maxCv, int(parents[i].centerX.size()));
        maxSec = std::max(maxSec, int(parents[i].sections.size()));
    }
    int const slots = parentCount * kPomadeDeviceMaxChildren;
    int const outCvStride = maxCv;
    int const outSecStride = maxSec;
    int const outRingStride = maxSec * kPomadeDeviceMaxRing;

    PomadeHScratch scr;
    PomadeHSpan span = {};
    PomadeHTube *dTubes = nullptr;
    PomadeHOut out = {};
    if (!scr.UploadSpan(pack, stream, &span, &dTubes)) {
        return _HFail(err, "pomade device lane: upload failed");
    }
    PomadeFrame *dFrames = scr.Upload(frames, stream);
    PomadeSubdivideDesc *dParams = scr.Upload(pars, stream);
    int *dTubeIds = scr.Upload(tubeIds, stream);
    int *dStatus = scr.Alloc<int>(size_t(slots));
    int *dAux = scr.Alloc<int>(size_t(slots));
    if (!scr.AllocOut(size_t(slots) * size_t(outCvStride),
                      size_t(slots) * size_t(outSecStride),
                      size_t(slots) * size_t(outRingStride), &out) ||
        !dFrames || !dParams || !dTubeIds || !dStatus || !dAux) {
        return _HFail(err, "pomade device lane: device allocation failed");
    }
    PomadeHTube *dOutTubes = scr.Alloc<PomadeHTube>(size_t(slots));
    if (!dOutTubes ||
        cudaMemsetAsync(dStatus, 0, size_t(slots) * sizeof(int), stream) !=
            cudaSuccess) {
        return _HFail(err, "pomade device lane: device allocation failed");
    }
    _HSubdivideKernel<<<parentCount, kPomadeDeviceMaxChildren, 0, stream>>>(
        span, dTubes, dFrames, dParams, dTubeIds, parentCount, out, dOutTubes,
        outCvStride, outSecStride, outRingStride, dStatus, dAux);
    if (cudaGetLastError() != cudaSuccess) {
        return _HFail(err, "pomade device lane: K14 launch failed");
    }
    std::vector<PomadeHTube> hTubes;
    std::vector<int> hStatus, hAux;
    std::vector<float> hcx, hcy, hcz, hsT, hsS, hsW, hrU, hrV;
    bool ok = _HDownload(&hTubes, dOutTubes, size_t(slots), stream) &&
              _HDownload(&hStatus, dStatus, size_t(slots), stream) &&
              _HDownload(&hAux, dAux, size_t(slots), stream) &&
              _HDownload(&hcx, out.cx, size_t(slots) * size_t(outCvStride),
                         stream) &&
              _HDownload(&hcy, out.cy, size_t(slots) * size_t(outCvStride),
                         stream) &&
              _HDownload(&hcz, out.cz, size_t(slots) * size_t(outCvStride),
                         stream) &&
              _HDownload(&hsT, out.secT, size_t(slots) * size_t(outSecStride),
                         stream) &&
              _HDownload(&hsS, out.secScale,
                         size_t(slots) * size_t(outSecStride), stream) &&
              _HDownload(&hsW, out.secTwist,
                         size_t(slots) * size_t(outSecStride), stream) &&
              _HDownload(&hrU, out.ringU,
                         size_t(slots) * size_t(outRingStride), stream) &&
              _HDownload(&hrV, out.ringV,
                         size_t(slots) * size_t(outRingStride), stream);
    if (!ok || cudaStreamSynchronize(stream) != cudaSuccess) {
        return _HFail(err, "pomade device lane: K14 readback failed");
    }
    children->resize(size_t(parentCount));
    for (int i = 0; i < parentCount; ++i) {
        (*children)[size_t(i)].resize(size_t(params[i].count));
        for (int c = 0; c < params[i].count; ++c) {
            int const slot = i * kPomadeDeviceMaxChildren + c;
            if (hStatus[size_t(slot)] != kPomadeHOk) {
                if (err) {
                    *err = _HStatusMessage(hStatus[size_t(slot)],
                                           hAux[size_t(slot)], c);
                }
                children->clear();
                return false;
            }
            PomadeHTube const &ot = hTubes[size_t(slot)];
            PomadeTubeDesc &child = (*children)[size_t(i)][size_t(c)];
            _HReadTube(hcx, hcy, hcz, hsT, hsS, hsW, hrU, hrV,
                       size_t(ot.centerBegin), ot.nCv,
                       size_t(ot.sectionBegin), ot.nSec,
                       size_t(ot.ringBegin), ot.ringVerts, &child);
            child.tubeId = parents[i].tubeId * 16 + 1 + c;
            child.regionId = parents[i].regionId;
            child.level = parents[i].level + 1;
            child.parentTubeId = parents[i].tubeId;
            child.childIndex = c;
            child.rootFramePinned = parents[i].rootFramePinned;
            child.rootFrame = parents[i].rootFrame;
            child.frameReference = parents[i].frameReference;
        }
    }
    return true;
}

bool PomadeParentAverageDevice(std::vector<PomadeTubeDesc> const *groups,
                              int groupCount,
                              std::vector<PomadeTubeDesc> *parentsOut,
                              cudaStream_t stream, std::string *err)
{
    if (!parentsOut || (groupCount > 0 && !groups)) {
        return _HFail(err, "PomadeParentAverageDevice: null argument");
    }
    parentsOut->clear();
    if (groupCount <= 0) {
        return true;
    }
    // K7's packed device frame walk predates authored root planes and the
    // descriptor material-frame reference. Keep either on the
    // descriptor-aware twin until the pack carries both charts.
    bool pinned = false;
    bool materialFrame = false;
    for (int g = 0; g < groupCount; ++g) {
        for (auto const &child : groups[g]) {
            pinned = pinned || child.rootFramePinned;
            materialFrame = materialFrame ||
                            _HasNonIdentityFrameReference(child);
        }
    }
    if (pinned || materialFrame) {
        std::vector<PomadeTubeDesc> result(size_t(groupCount), PomadeTubeDesc{});
        for (int g = 0; g < groupCount; ++g) {
            if (!PomadeParentAverageCpu(groups[g], &result[size_t(g)], err)) {
                return false;
            }
        }
        parentsOut->swap(result);
        return true;
    }
    PomadeHPack pack;
    std::vector<int> begins, counts, sectionBegins;
    int maxCv = 0, maxSec = 0, totalSections = 0;
    for (int g = 0; g < groupCount; ++g) {
        std::vector<PomadeTubeDesc> const &grp = groups[g];
        if (grp.empty()) {
            return _HFail(err, "PomadeParentAverageCpu: no children");
        }
        if (int(grp.size()) > kPomadeDeviceMaxChildren) {
            return _HFail(err,
                          "pomade device lane: group exceeds the device caps");
        }
        for (auto const &ch : grp) {
            if (!_HValidate(ch, err)) {
                return false;
            }
        }
        begins.push_back(int(pack.tubes.size()));
        counts.push_back(int(grp.size()));
        sectionBegins.push_back(totalSections);
        totalSections += int(grp[0].sections.size());
        maxCv = std::max(maxCv, int(grp[0].centerX.size()));
        maxSec = std::max(maxSec, int(grp[0].sections.size()));
        for (auto const &ch : grp) {
            _HAppendTube(ch, &pack);
        }
    }
    int const outCvStride = maxCv;
    int const outSecStride = maxSec;
    int const outRingStride = maxSec * kPomadeDeviceMaxRing;

    PomadeHScratch scr;
    PomadeHSpan span = {};
    PomadeHTube *dTubes = nullptr;
    PomadeHOut out = {};
    if (!scr.UploadSpan(pack, stream, &span, &dTubes)) {
        return _HFail(err, "pomade device lane: upload failed");
    }
    int *dBegin = scr.Upload(begins, stream);
    int *dCount = scr.Upload(counts, stream);
    int *dSecBegin = scr.Upload(sectionBegins, stream);
    float *dPcx = scr.Alloc<float>(size_t(groupCount) * size_t(outCvStride));
    float *dPcy = scr.Alloc<float>(size_t(groupCount) * size_t(outCvStride));
    float *dPcz = scr.Alloc<float>(size_t(groupCount) * size_t(outCvStride));
    PomadeFrame *dPf =
        scr.Alloc<PomadeFrame>(size_t(groupCount) * size_t(outCvStride));
    int *dStatus = scr.Alloc<int>(size_t(totalSections));
    if (!scr.AllocOut(size_t(groupCount) * size_t(outCvStride),
                      size_t(groupCount) * size_t(outSecStride),
                      size_t(groupCount) * size_t(outRingStride), &out) ||
        !dBegin || !dCount || !dSecBegin || !dPcx || !dPcy || !dPcz || !dPf ||
        !dStatus) {
        return _HFail(err, "pomade device lane: device allocation failed");
    }
    _HParentCenterKernel<<<groupCount, kPomadeDeviceMaxChildren, 0, stream>>>(
        span, dTubes, dBegin, dCount, groupCount, dPcx, dPcy, dPcz, dPf,
        outCvStride);
    if (cudaGetLastError() != cudaSuccess) {
        return _HFail(err, "pomade device lane: K7 center launch failed");
    }
    _HParentSectionKernel<<<totalSections, kPomadeDeviceMaxChildren, 0,
                            stream>>>(
        span, dTubes, dBegin, dCount, dSecBegin, groupCount, totalSections,
        dPcx, dPcy, dPcz, dPf, out, outCvStride, outSecStride, outRingStride,
        dStatus);
    if (cudaGetLastError() != cudaSuccess) {
        return _HFail(err, "pomade device lane: K7 section launch failed");
    }
    std::vector<int> hStatus;
    std::vector<float> hcx, hcy, hcz, hsT, hsS, hsW, hrU, hrV;
    size_t const cvTotal = size_t(groupCount) * size_t(outCvStride);
    size_t const secTotal = size_t(groupCount) * size_t(outSecStride);
    size_t const ringTotal = size_t(groupCount) * size_t(outRingStride);
    bool ok = _HDownload(&hStatus, dStatus, size_t(totalSections), stream) &&
              _HDownload(&hcx, dPcx, cvTotal, stream) &&
              _HDownload(&hcy, dPcy, cvTotal, stream) &&
              _HDownload(&hcz, dPcz, cvTotal, stream) &&
              _HDownload(&hsT, out.secT, secTotal, stream) &&
              _HDownload(&hsS, out.secScale, secTotal, stream) &&
              _HDownload(&hsW, out.secTwist, secTotal, stream) &&
              _HDownload(&hrU, out.ringU, ringTotal, stream) &&
              _HDownload(&hrV, out.ringV, ringTotal, stream);
    if (!ok || cudaStreamSynchronize(stream) != cudaSuccess) {
        return _HFail(err, "pomade device lane: K7 readback failed");
    }
    for (int i = 0; i < totalSections; ++i) {
        if (hStatus[size_t(i)] != kPomadeHOk) {
            return _HFail(err, _HStatusMessage(hStatus[size_t(i)], 0, 0)
                                   .c_str());
        }
    }
    parentsOut->resize(size_t(groupCount));
    for (int g = 0; g < groupCount; ++g) {
        std::vector<PomadeTubeDesc> const &grp = groups[g];
        PomadeTubeDesc &parent = (*parentsOut)[size_t(g)];
        _HReadTube(hcx, hcy, hcz, hsT, hsS, hsW, hrU, hrV,
                   size_t(g) * size_t(outCvStride),
                   int(grp[0].centerX.size()),
                   size_t(g) * size_t(outSecStride),
                   int(grp[0].sections.size()),
                   size_t(g) * size_t(outRingStride), grp[0].ringVerts,
                   &parent);
        // Output identity: the twin's rule (copies of one tube keep its
        // identity, siblings yield their parent, anything else is a
        // transient the caller names).
        bool sameTube = true;
        for (auto const &ch : grp) {
            sameTube = sameTube && ch.tubeId == grp[0].tubeId;
        }
        bool sameParent = grp[0].parentTubeId >= 0;
        for (auto const &ch : grp) {
            if (ch.parentTubeId != grp[0].parentTubeId) {
                sameParent = false;
            }
        }
        if (sameTube) {
            parent.tubeId = grp[0].tubeId;
            parent.regionId = grp[0].regionId;
            parent.level = grp[0].level;
            parent.parentTubeId = grp[0].parentTubeId;
            parent.childIndex = grp[0].childIndex;
        } else if (sameParent) {
            parent.tubeId = grp[0].parentTubeId;
            parent.regionId = grp[0].regionId;
            parent.level = grp[0].level > 1 ? grp[0].level - 1 : 1;
            parent.parentTubeId = -1;
            parent.childIndex = -1;
        } else {
            int minLevel = grp[0].level;
            for (auto const &ch : grp) {
                minLevel = std::min(minLevel, ch.level);
            }
            parent.tubeId = -1;
            parent.regionId = grp[0].regionId;
            parent.level = minLevel > 1 ? minLevel - 1 : 1;
            parent.parentTubeId = -1;
            parent.childIndex = -1;
        }
    }
    return true;
}

bool PomadeHierarchicalSculptApplyDevice(
    PomadeSculptApplyItem const *items, int itemCount,
    std::vector<PomadeTubeDesc> *outActual,
    std::vector<PomadeShapeDeltas> *outStored, cudaStream_t stream,
    std::string *err)
{
    if (!outActual || !outStored || (itemCount > 0 && !items)) {
        return _HFail(err, "PomadeHierarchicalSculptApplyDevice: null argument");
    }
    outActual->clear();
    outStored->clear();
    if (itemCount <= 0) {
        return true;
    }
    bool pinned = false;
    bool materialFrame = false;
    for (int i = 0; i < itemCount; ++i) {
        PomadeSculptApplyItem const &item = items[i];
        if (!item.derivedNew || !item.oldActual || !item.oldDerived ||
            !item.oldStored) {
            return _HFail(err, "PomadeHierarchicalSculptApplyDevice: null item");
        }
        pinned = pinned || item.derivedNew->rootFramePinned ||
                 item.oldActual->rootFramePinned ||
                 item.oldDerived->rootFramePinned;
        materialFrame = materialFrame ||
            _HasNonIdentityFrameReference(*item.derivedNew) ||
            _HasNonIdentityFrameReference(*item.oldActual) ||
            _HasNonIdentityFrameReference(*item.oldDerived);
    }
    // The old GPU sculpt pack has neither authored root-frame nor material
    // reference data. The host twin consumes the descriptor-aware chart.
    if (pinned || materialFrame) {
        std::vector<PomadeTubeDesc> actual(size_t(itemCount), PomadeTubeDesc{});
        std::vector<PomadeShapeDeltas> stored(size_t(itemCount), PomadeShapeDeltas{});
        for (int i = 0; i < itemCount; ++i) {
            PomadeSculptApplyItem const &item = items[i];
            if (!PomadeHierarchicalSculptApplyCpu(
                    *item.derivedNew, *item.oldActual, *item.oldDerived,
                    *item.oldStored, item.lockChildren, item.preserveLength,
                    &actual[size_t(i)], &stored[size_t(i)], err)) {
                return false;
            }
        }
        outActual->swap(actual);
        outStored->swap(stored);
        return true;
    }
    PomadeHPack dPack, aPack, oPack, sPack;
    std::vector<unsigned char> flags;
    int maxCv = 0, maxSec = 0;
    for (int i = 0; i < itemCount; ++i) {
        PomadeSculptApplyItem const &it = items[i];
        if (!it.derivedNew || !it.oldActual || !it.oldDerived ||
            !it.oldStored) {
            return _HFail(err,
                          "PomadeHierarchicalSculptApplyDevice: null item");
        }
        if (!_HValidate(*it.derivedNew, err) ||
            !_HValidate(*it.oldActual, err) ||
            !_HValidate(*it.oldDerived, err)) {
            return false;
        }
        // The twin's layout contract, checked here so the kernel is pure.
        if (it.derivedNew->centerX.size() != it.oldActual->centerX.size() ||
            it.derivedNew->sections.size() != it.oldActual->sections.size()) {
            return _HFail(err,
                          "PomadeHierarchicalSculptCpu: center/section counts "
                          "changed under the sculpt (re-subdivide)");
        }
        for (size_t s = 0; s < it.derivedNew->sections.size(); ++s) {
            if (it.derivedNew->sections[s].t != it.oldActual->sections[s].t) {
                return _HFail(err,
                              "PomadeHierarchicalSculptCpu: section t drifted "
                              "(re-subdivide)");
            }
        }
        if (it.derivedNew->ringVerts != it.oldActual->ringVerts &&
            !it.oldActual->inheritedBoundaryBindings.empty()) {
            return _HFail(err,
                          "PomadeHierarchicalSculptCpu: boundary material "
                          "slots need an explicit topology remap");
        }
        if (it.lockChildren) {
            if (it.oldActual->centerX.size() !=
                    it.oldDerived->centerX.size() ||
                it.oldActual->sections.size() !=
                    it.oldDerived->sections.size()) {
                return _HFail(err,
                              "PomadeHierarchicalSculptCpu: stale child layout");
            }
            for (size_t s = 0; s < it.derivedNew->sections.size(); ++s) {
                if (it.oldActual->sections[s].u.size() !=
                    it.oldDerived->sections[s].u.size()) {
                    return _HFail(err,
                                  "PomadeHierarchicalSculptCpu: stale child "
                                  "section layout");
                }
            }
        } else {
            if (it.oldStored->centerDu.size() !=
                    it.derivedNew->centerX.size() ||
                it.oldStored->sections.size() !=
                    it.derivedNew->sections.size()) {
                return _HFail(err,
                              "PomadeHierarchicalSculptCpu: stored layout "
                              "drift");
            }
            // The kernel indexes the stored rings at a uniform stride.
            int const srv = int(it.oldStored->sections.front().u.size());
            for (auto const &ds : it.oldStored->sections) {
                if (int(ds.u.size()) != srv || int(ds.v.size()) != srv) {
                    return _HFail(err,
                                  "PomadeHierarchicalSculptCpu: stored layout "
                                  "drift");
                }
            }
            if (srv < 3 || srv > kPomadeDeviceMaxRing) {
                return _HFail(
                    err, "pomade device lane: tube exceeds the device caps");
            }
        }
        _HAppendTube(*it.derivedNew, &dPack);
        _HAppendTube(*it.oldActual, &aPack);
        _HAppendTube(*it.oldDerived, &oPack);
        _HAppendDeltas(*it.oldStored, &sPack);
        flags.push_back((unsigned char)((it.lockChildren ? 1u : 0u) |
                                        (it.preserveLength ? 2u : 0u)));
        maxCv = std::max(maxCv, int(it.derivedNew->centerX.size()));
        maxSec = std::max(maxSec, int(it.derivedNew->sections.size()));
    }
    int const outCvStride = maxCv;
    int const outSecStride = maxSec;
    int const outRingStride = maxSec * kPomadeDeviceMaxRing;

    PomadeHScratch scr;
    PomadeHSpan dSpan = {}, aSpan = {}, oSpan = {}, sSpan = {};
    PomadeHTube *dT = nullptr, *aT = nullptr, *oT = nullptr, *sT = nullptr;
    PomadeHOut out = {}, outDelta = {};
    size_t const cvTotal = size_t(itemCount) * size_t(outCvStride);
    size_t const secTotal = size_t(itemCount) * size_t(outSecStride);
    size_t const ringTotal = size_t(itemCount) * size_t(outRingStride);
    if (!scr.UploadSpan(dPack, stream, &dSpan, &dT) ||
        !scr.UploadSpan(aPack, stream, &aSpan, &aT) ||
        !scr.UploadSpan(oPack, stream, &oSpan, &oT) ||
        !scr.UploadSpan(sPack, stream, &sSpan, &sT)) {
        return _HFail(err, "pomade device lane: upload failed");
    }
    unsigned char *dFlags = scr.Upload(flags, stream);
    int *dStatus = scr.Alloc<int>(size_t(itemCount));
    if (!scr.AllocOut(cvTotal, secTotal, ringTotal, &out) ||
        !scr.AllocOut(cvTotal, secTotal, ringTotal, &outDelta) || !dFlags ||
        !dStatus) {
        return _HFail(err, "pomade device lane: device allocation failed");
    }
    int const block = 16;
    int const grid = (itemCount + block - 1) / block;
    size_t const shared =
        size_t(block) * size_t(2 * kPomadeDeviceMaxRing) * sizeof(PomadeHPt);
    _HSculptApplyKernel<<<grid, block, shared, stream>>>(
        dSpan, dT, aSpan, aT, oSpan, oT, sSpan, sT, dFlags, itemCount, out,
        outDelta, outCvStride, outSecStride, outRingStride, dStatus);
    if (cudaGetLastError() != cudaSuccess) {
        return _HFail(err, "pomade device lane: K6 launch failed");
    }
    std::vector<int> hStatus;
    std::vector<float> hcx, hcy, hcz, hsT, hsS, hsW, hrU, hrV;
    std::vector<float> hdu, hdv, hdw, hds, hdt, hdrU, hdrV;
    bool ok = _HDownload(&hStatus, dStatus, size_t(itemCount), stream) &&
              _HDownload(&hcx, out.cx, cvTotal, stream) &&
              _HDownload(&hcy, out.cy, cvTotal, stream) &&
              _HDownload(&hcz, out.cz, cvTotal, stream) &&
              _HDownload(&hsT, out.secT, secTotal, stream) &&
              _HDownload(&hsS, out.secScale, secTotal, stream) &&
              _HDownload(&hsW, out.secTwist, secTotal, stream) &&
              _HDownload(&hrU, out.ringU, ringTotal, stream) &&
              _HDownload(&hrV, out.ringV, ringTotal, stream) &&
              _HDownload(&hdu, outDelta.cx, cvTotal, stream) &&
              _HDownload(&hdv, outDelta.cy, cvTotal, stream) &&
              _HDownload(&hdw, outDelta.cz, cvTotal, stream) &&
              _HDownload(&hds, outDelta.secScale, secTotal, stream) &&
              _HDownload(&hdt, outDelta.secTwist, secTotal, stream) &&
              _HDownload(&hdrU, outDelta.ringU, ringTotal, stream) &&
              _HDownload(&hdrV, outDelta.ringV, ringTotal, stream);
    if (!ok || cudaStreamSynchronize(stream) != cudaSuccess) {
        return _HFail(err, "pomade device lane: K6 readback failed");
    }
    outActual->resize(size_t(itemCount));
    outStored->resize(size_t(itemCount));
    for (int i = 0; i < itemCount; ++i) {
        if (hStatus[size_t(i)] != kPomadeHOk) {
            if (err) {
                *err = _HStatusMessage(hStatus[size_t(i)], 0, 0);
            }
            outActual->clear();
            outStored->clear();
            return false;
        }
        PomadeTubeDesc const &D = *items[i].derivedNew;
        PomadeTubeDesc &act = (*outActual)[size_t(i)];
        _HReadTube(hcx, hcy, hcz, hsT, hsS, hsW, hrU, hrV,
                   size_t(i) * size_t(outCvStride), int(D.centerX.size()),
                   size_t(i) * size_t(outSecStride), int(D.sections.size()),
                   size_t(i) * size_t(outRingStride), D.ringVerts, &act);
        act.tubeId = D.tubeId;
        act.regionId = D.regionId;
        act.level = D.level;
        act.parentTubeId = D.parentTubeId;
        act.childIndex = D.childIndex;
        act.rootFramePinned = D.rootFramePinned;
        act.rootFrame = D.rootFrame;
        act.frameReference = D.frameReference;
        if (D.ringVerts == items[i].oldActual->ringVerts) {
            act.inheritedBoundaryBindings =
                items[i].oldActual->inheritedBoundaryBindings;
        }
        PomadeShapeDeltas &st = (*outStored)[size_t(i)];
        if (items[i].lockChildren) {
            st = *items[i].oldStored;
            continue;
        }
        // Exact-zero sculpt stays exact-zero: derivation drift is not
        // authoring (the twin's rule, scanned on the host).
        bool sculpted = false;
        PomadeShapeDeltas const &old = *items[i].oldStored;
        for (float v : old.centerDu) {
            sculpted = sculpted || (v != 0.0f);
        }
        for (float v : old.centerDv) {
            sculpted = sculpted || (v != 0.0f);
        }
        for (float v : old.centerDw) {
            sculpted = sculpted || (v != 0.0f);
        }
        for (auto const &s : old.sections) {
            sculpted = sculpted || (s.scale != 0.0f) || (s.twist != 0.0f);
            for (float v : s.u) {
                sculpted = sculpted || (v != 0.0f);
            }
            for (float v : s.v) {
                sculpted = sculpted || (v != 0.0f);
            }
        }
        if (!sculpted) {
            PomadeClearDeltas(D, &st);
            continue;
        }
        size_t const cvBase = size_t(i) * size_t(outCvStride);
        size_t const ringBase = size_t(i) * size_t(outRingStride);
        int const nCv = int(D.centerX.size());
        st.centerDu.assign(hdu.begin() + cvBase, hdu.begin() + cvBase + nCv);
        st.centerDv.assign(hdv.begin() + cvBase, hdv.begin() + cvBase + nCv);
        st.centerDw.assign(hdw.begin() + cvBase, hdw.begin() + cvBase + nCv);
        st.sections.resize(D.sections.size());
        for (size_t s = 0; s < D.sections.size(); ++s) {
            PomadeTubeSection &ds = st.sections[s];
            ds.t = D.sections[s].t;
            ds.scale = hds[size_t(i) * size_t(outSecStride) + s];
            ds.twist = hdt[size_t(i) * size_t(outSecStride) + s];
            size_t const rb = ringBase + s * size_t(D.ringVerts);
            ds.u.assign(hdrU.begin() + rb, hdrU.begin() + rb + D.ringVerts);
            ds.v.assign(hdrV.begin() + rb, hdrV.begin() + rb + D.ringVerts);
        }
    }
    return true;
}

// -- Host-facing wrappers for the K12/K13 production lanes --------------------
//
// The launches above take device pointers; these two carry the CPU twins'
// signatures so a production call site is a one-line choice between the
// twin and the lane. Both kernels are bit-exact against their twins
// (testUsdGenPomadeKernels), which is why they are safe to run in
// production: a check that flags a tube must flag it identically whichever
// lane ran. The K8/K9/K10 lanes deliberately have no wrapper here, because
// they agree with their twins only to a tolerance and the committer and
// hydrate compare guides bit for bit (pomadeModel.h, PomadeGenerateGuides).

// pomadeTube.h and pomadeCheck.h hide their host twins behind
// #ifndef __CUDA_ARCH__, and nvcc parses host code in the device pass as
// well, so the two the K12 wrapper needs are re-declared here.
bool PomadeTubeFramesCpu(PomadeTubeDesc const &tube,
                        std::vector<PomadeFrame> *frames, std::string *err);
bool PomadeRootChartCpu(PomadeTubeDesc const &tube,
                       std::vector<PomadeFrame> const &frames, float center[3],
                       float nrm[3], float bin[3],
                       std::vector<float> *worldRing, std::string *err);

bool PomadeSmoothnessScoreDevice(float const *cx, float const *cy,
                                float const *cz, int nCv, float *out,
                                cudaStream_t stream, std::string *err)
{
    if (!cx || !cy || !cz || !out) {
        return _HFail(err, "PomadeSmoothnessScoreCpu: null input or output");
    }
    if (nCv < 1) {
        return _HFail(err, "PomadeSmoothnessScoreCpu: need >= 1 CV");
    }
    PomadeHScratch scr;
    float *dcx = scr.Alloc<float>(size_t(nCv));
    float *dcy = scr.Alloc<float>(size_t(nCv));
    float *dcz = scr.Alloc<float>(size_t(nCv));
    float *dOut = scr.Alloc<float>(size_t(nCv));
    if (!scr.ok || !dcx || !dcy || !dcz || !dOut) {
        return _HFail(err, "pomade device lane: device allocation failed");
    }
    size_t const bytes = size_t(nCv) * sizeof(float);
    if (cudaMemcpyAsync(dcx, cx, bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(dcy, cy, bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(dcz, cz, bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
        return _HFail(err, "pomade device lane: K13 upload failed");
    }
    char buf[256] = {0};
    if (!PomadeLaunchSmoothnessScores(dcx, dcy, dcz, nCv, dOut, stream, buf,
                                     sizeof(buf))) {
        return _HFail(err, buf[0] ? buf : "pomade device lane: K13 launch");
    }
    if (cudaMemcpyAsync(out, dOut, bytes, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return _HFail(err, "pomade device lane: K13 readback failed");
    }
    return true;
}

bool PomadeTubeIntersectDevice(PomadeTubeDesc const *tubes, int tubeCount,
                              int *outFlags, cudaStream_t stream,
                              std::string *err)
{
    if (!tubes || !outFlags) {
        return _HFail(err, "PomadeTubeIntersectCpu: null input or output");
    }
    if (tubeCount <= 0) {
        return true;
    }
    // Charts come from the same derivation the twin uses, so the two lanes
    // start from identical bytes.
    std::vector<PomadeDeviceRootChart> charts(static_cast<size_t>(tubeCount));
    std::vector<int> ids(static_cast<size_t>(tubeCount));
    std::vector<int> parents(static_cast<size_t>(tubeCount));
    for (int i = 0; i < tubeCount; ++i) {
        PomadeTubeDesc const &tube = tubes[size_t(i)];
        if (tube.ringVerts < 3 || tube.ringVerts > kPomadeCheckRingMax) {
            return _HFail(err,
                          "pomade device lane: tube exceeds the device caps");
        }
        std::vector<PomadeFrame> frames;
        std::vector<float> ring;
        PomadeDeviceRootChart &chart = charts[size_t(i)];
        if (!PomadeTubeFramesCpu(tube, &frames, err) ||
            !PomadeRootChartCpu(tube, frames, chart.center, chart.nrm,
                               chart.bin, &ring, err)) {
            return false;
        }
        chart.vertCount = int(ring.size() / 3);
        if (chart.vertCount < 3 || chart.vertCount > kPomadeCheckRingMax) {
            return _HFail(err,
                          "pomade device lane: tube exceeds the device caps");
        }
        for (size_t k = 0; k < ring.size(); ++k) {
            chart.verts[k] = ring[k];
        }
        ids[size_t(i)] = tube.tubeId;
        parents[size_t(i)] = tube.parentTubeId;
    }
    PomadeHScratch scr;
    PomadeDeviceRootChart *dCharts = scr.Upload(charts, stream);
    int *dIds = scr.Upload(ids, stream);
    int *dParents = scr.Upload(parents, stream);
    size_t const pairs = size_t(tubeCount) * size_t(tubeCount);
    int *dPair = scr.Alloc<int>(pairs);
    int *dBroad = scr.Alloc<int>(pairs);
    int *dFlags = scr.Alloc<int>(size_t(tubeCount));
    if (!scr.ok || !dCharts || !dIds || !dParents || !dPair || !dBroad ||
        !dFlags) {
        return _HFail(err, "pomade device lane: device allocation failed");
    }
    char buf[256] = {0};
    if (!PomadeLaunchTubeIntersect(dCharts, dIds, dParents, tubeCount, dPair,
                                  dBroad, dFlags, stream, buf, sizeof(buf))) {
        return _HFail(err, buf[0] ? buf : "pomade device lane: K12 launch");
    }
    if (cudaMemcpyAsync(outFlags, dFlags, size_t(tubeCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return _HFail(err, "pomade device lane: K12 readback failed");
    }
    return true;
}

}  // namespace usdGenPomade
