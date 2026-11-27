// stormTestUtils.h — shared harness for the T2 Storm decision-gate tests
// (S-8/S-9 family). EGL headless context via libs/hosts/usdGenHydra/primitives/curves/shaders/test/eglctx.h,
// FrameRecorder renders, Hio PNG round-trip with a native-format pixel loader
// (a conversion Read() into a Float32Vec4 StorageSpec silently yields an
// unfilled buffer on 26.08 — decode the native format instead; verified).
#ifndef USDGEN_TESTS_STORMTESTUTILS_H
#define USDGEN_TESTS_STORMTESTUTILS_H

#include "utils/pxrInterop.h"
#include "pxr/pxr.h"
#include "pxr/base/plug/registry.h"
#include "pxr/base/tf/registryManager.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/imaging/hio/image.h"
#include "pxr/imaging/hio/types.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/camera.h"
#include "pxr/usdImaging/usdAppUtils/frameRecorder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <filesystem>
#include <string>
#include <vector>

#include "eglctx.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace stormtest {

inline bool initGl(const char* testname) {
    PlugRegistry::GetInstance();
    TfRegistryManager::GetInstance();
    if (!eglctx::MakeHeadlessGLContext()) {
        std::printf("SKIP: %s needs a headless EGL device context\n",
                    testname);
        return false;
    }
    return true;
}

// Renders `usda`'s /World/Cam at `complexity` into `outPng` (px x px).
// Returns FrameRecorder's ok flag; wall ms of the recorded frame in *ms.
inline bool renderStage(const std::string& usda, const std::string& outPng,
                        int px, float complexity, double* ms = nullptr) {
    UsdStageRefPtr stage = UsdStage::Open(usda);
    if (!stage) {
        std::fprintf(stderr, "stormtest: cannot open %s\n", usda.c_str());
        return false;
    }
    UsdGeomCamera cam(stage->GetPrimAtPath(SdfPath("/World/Cam")));
    if (!cam) {
        std::fprintf(stderr, "stormtest: no /World/Cam in %s\n",
                     usda.c_str());
        return false;
    }
    UsdAppUtilsFrameRecorder rec(TfToken(), /*gpuEnabled*/ true);
    rec.SetRendererPlugin(TfToken("HdStormRendererPlugin"));
    rec.SetImageWidth(px);
    rec.SetColorCorrectionMode(TfToken("sRGB"));
    rec.SetCameraLightEnabled(true);
    rec.SetComplexity(complexity);
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = rec.Record(stage, cam, UsdTimeCode::EarliestTime(), outPng);
    if (ms) {
        *ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
    }
    return ok;
}

// One engine + stage for many frames: keeps Storm's tessellation caches
// warm so per-frame timings measure rendering, not first-frame bakes.
class StormSession {
  public:
    bool open(const std::string& usda, int px) {
        stage_ = UsdStage::Open(usda);
        if (!stage_) return false;
        cam_ = UsdGeomCamera(stage_->GetPrimAtPath(SdfPath("/World/Cam")));
        if (!cam_) return false;
        rec_ = std::make_unique<UsdAppUtilsFrameRecorder>(TfToken(), true);
        rec_->SetRendererPlugin(TfToken("HdStormRendererPlugin"));
        rec_->SetImageWidth(px);
        rec_->SetColorCorrectionMode(TfToken("sRGB"));
        rec_->SetCameraLightEnabled(true);
        return true;
    }
    bool frame(float complexity, const std::string& outPng,
               double* ms = nullptr) {
        rec_->SetComplexity(complexity);
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok =
            rec_->Record(stage_, cam_, UsdTimeCode::EarliestTime(), outPng);
        if (ms) {
            *ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        }
        return ok;
    }

  private:
    UsdStageRefPtr stage_;
    UsdGeomCamera cam_;
    std::unique_ptr<UsdAppUtilsFrameRecorder> rec_;
};

inline float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t em = h & 0x7FFFu;
    uint32_t out;
    if (em < 0x0400u) {
        out = sign;  // zero / flushed subnormal
    } else if ((em & 0x7C00u) == 0x7C00u) {
        out = sign | 0x7F800000u | (static_cast<uint32_t>(em & 0x3FFu) << 13);
    } else {
        out = sign | (((em >> 10) - 15u + 127u) << 23) |
              (static_cast<uint32_t>(em & 0x3FFu) << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof f);
    return f;
}

// Loads a FrameRecorder-written PNG into RGBA floats in [0,1].
inline bool loadRgba(const std::string& path, int* w, int* h,
                     VtArray<float>* rgba) {
    auto img = HioImage::OpenForReading(path);
    if (!img) {
        std::fprintf(stderr, "loadRgba: cannot open %s\n", path.c_str());
        return false;
    }
    const int iw = img->GetWidth(), ih = img->GetHeight();
    const HioFormat fmt = img->GetFormat();
    *w = iw;
    *h = ih;
    const bool isHalf = (fmt == HioFormatFloat16Vec4 ||
                         fmt == HioFormatFloat16Vec3);
    const bool isF32 = (fmt == HioFormatFloat32Vec4 ||
                        fmt == HioFormatFloat32Vec3);
    int nCh = 4;
    switch (fmt) {
        case HioFormatUNorm8Vec4:
        case HioFormatUNorm8Vec4srgb: nCh = 4; break;
        case HioFormatUNorm8Vec3: nCh = 3; break;
        case HioFormatFloat16Vec4: nCh = 4; break;
        case HioFormatFloat16Vec3: nCh = 3; break;
        case HioFormatFloat32Vec4: nCh = 4; break;
        case HioFormatFloat32Vec3: nCh = 3; break;
        default:
            std::fprintf(stderr, "loadRgba: unhandled HioFormat %d (%s)\n",
                         static_cast<int>(fmt), path.c_str());
            return false;
    }
    const size_t pixels = static_cast<size_t>(iw) * ih;
    const size_t elt = isHalf ? 2 : (isF32 ? 4 : 1);
    std::vector<uint8_t> raw(pixels * nCh * elt);
    HioImage::StorageSpec spec;
    spec.width = iw;
    spec.height = ih;
    spec.depth = 1;
    spec.format = fmt;
    spec.flipped = false;
    spec.data = raw.data();
    if (!img->Read(spec)) {
        std::fprintf(stderr, "loadRgba: Read failed %s\n", path.c_str());
        return false;
    }
    rgba->clear();
    rgba->reserve(pixels * 4);
    for (size_t i = 0; i < pixels; ++i) {
        const uint8_t* p = raw.data() + i * nCh * elt;
        float c[4] = {0, 0, 0, 1};
        for (int ch = 0; ch < nCh; ++ch) {
            float v = 0.f;
            if (isHalf) {
                uint16_t u;
                std::memcpy(&u, p + ch * 2, sizeof u);
                v = halfToFloat(u);
            } else if (isF32) {
                std::memcpy(&v, p + ch * 4, sizeof v);
            } else {
                v = p[ch] / 255.0f;
            }
            c[ch] = v;
        }
        rgba->insert(rgba->end(), c, c + 4);
    }
    return true;
}

// Max-per-channel-abs RGB difference stats between two equal-size loads.
inline void diffStats(const VtArray<float>& a, const VtArray<float>& b,
                      double* meanAbsOut, double* frac8Out) {
    double mean = 0.0;
    size_t over = 0;
    const size_t n = std::min(a.size(), b.size()) / 4;
    for (size_t i = 0; i < n; ++i) {
        double best = 0.0;
        for (int c = 0; c < 3; ++c) {
            best = std::max(best,
                            std::fabs(static_cast<double>(a[i * 4 + c]) -
                                      static_cast<double>(b[i * 4 + c])));
        }
        mean += best;
        if (best * 255.0 > 8.0) ++over;
    }
    *meanAbsOut = n ? mean / n : 1.0;
    *frac8Out = n ? static_cast<double>(over) / n : 1.0;
}

// Fraction of pixels brighter than 1/255 luma (coverage proxy on black bg).
inline double coverage(const VtArray<float>& px) {
    size_t lit = 0;
    const size_t n = px.size() / 4;
    for (size_t i = 0; i < n; ++i) {
        const float* p = &px[i * 4];
        if (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2] > 1.0f / 255.0f) {
            ++lit;
        }
    }
    return n ? static_cast<double>(lit) / n : 0.0;
}

}  // namespace stormtest

PXR_NAMESPACE_CLOSE_SCOPE

#endif  // USDGEN_TESTS_STORMTESTUTILS_H
