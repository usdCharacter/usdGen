// Vendored verbatim from docs/prework/probes/PW-egl/eglctx.h (known-good
// PW-2 EGL harness on the measurement host); kept in sync manually.
// Headless GL context via EGL_EXT_platform_device. No EGL headers on this host,
// so entry points are declared by hand and resolved via dlopen/eglGetProcAddress.
#pragma once
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
namespace eglctx {
typedef void* EGLDisplay; typedef void* EGLContext; typedef void* EGLSurface;
typedef void* EGLDeviceEXT; typedef void* EGLConfig; typedef unsigned int EGLenum;
typedef int EGLint; typedef unsigned int EGLBoolean;
static const EGLint kNone = 0x3038;
static const EGLenum kPlatformDevice = 0x313F;
static const EGLenum kOpenGLAPI = 0x30A2;
typedef void* (*PFNEGLGETPROC)(const char*);
// profileBit: 0x1 = core, 0x2 = compatibility
inline bool MakeHeadlessGLContext(int major = 4, int minor = 5, int profileBit = 0x2)
{
    void* egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!egl) { fprintf(stderr, "eglctx: dlopen libEGL failed\n"); return false; }
    auto gpa = (PFNEGLGETPROC)dlsym(egl, "eglGetProcAddress");
    auto R = [&](const char* n) -> void* {
        void* p = dlsym(egl, n); if (!p && gpa) p = gpa(n); return p; };
    auto eglQueryDevicesEXT = (EGLBoolean(*)(EGLint, EGLDeviceEXT*, EGLint*))R("eglQueryDevicesEXT");
    auto eglGetPlatformDisplayEXT = (EGLDisplay(*)(EGLenum, void*, const EGLint*))R("eglGetPlatformDisplayEXT");
    auto eglInitialize = (EGLBoolean(*)(EGLDisplay, EGLint*, EGLint*))R("eglInitialize");
    auto eglBindAPI = (EGLBoolean(*)(EGLenum))R("eglBindAPI");
    auto eglCreateContext = (EGLContext(*)(EGLDisplay, EGLConfig, EGLContext, const EGLint*))R("eglCreateContext");
    auto eglMakeCurrent = (EGLBoolean(*)(EGLDisplay, EGLSurface, EGLSurface, EGLContext))R("eglMakeCurrent");
    auto eglChooseConfig = (EGLBoolean(*)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*))R("eglChooseConfig");
    auto eglCreatePbufferSurface = (EGLSurface(*)(EGLDisplay, EGLConfig, const EGLint*))R("eglCreatePbufferSurface");
    auto eglGetError = (EGLint(*)())R("eglGetError");
    if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT) {
        fprintf(stderr, "eglctx: EGL_EXT_platform_device unavailable\n"); return false; }
    EGLDeviceEXT devs[16]; EGLint n = 0;
    if (!eglQueryDevicesEXT(16, devs, &n) || n <= 0) {
        fprintf(stderr, "eglctx: no EGL devices\n"); return false; }
    for (int i = 0; i < n; ++i) {
        EGLDisplay d = eglGetPlatformDisplayEXT(kPlatformDevice, devs[i], nullptr);
        if (!d) continue;
        EGLint ma = 0, mi = 0;
        if (!eglInitialize(d, &ma, &mi)) continue;
        if (!eglBindAPI(kOpenGLAPI)) continue;
        // Pick a config with a real (pbuffer) default framebuffer: a
        // surfaceless context leaves the default FBO incomplete, which makes
        // HgiGL_ScopedStateHolder's glGet queries on the draw buffers raise
        // GL_INVALID_OPERATION and the final blit/resolve fail.
        const EGLint cfgAttrs[] = {
            0x3033 /*EGL_SURFACE_TYPE*/,   0x0001 /*EGL_PBUFFER_BIT*/,
            0x3040 /*EGL_RENDERABLE_TYPE*/,0x0008 /*EGL_OPENGL_BIT*/,
            0x3024 /*EGL_RED_SIZE*/,   8,
            0x3023 /*EGL_GREEN_SIZE*/, 8,
            0x3022 /*EGL_BLUE_SIZE*/,  8,
            0x3021 /*EGL_ALPHA_SIZE*/, 8,
            0x3025 /*EGL_DEPTH_SIZE*/, 24,
            kNone };
        EGLConfig cfg = (EGLConfig)0; EGLint nc = 0;
        if (!eglChooseConfig || !eglChooseConfig(d, cfgAttrs, &cfg, 1, &nc) || nc < 1) {
            fprintf(stderr, "eglctx: no pbuffer config on device %d\n", i); continue; }
        const EGLint pbAttrs[] = { 0x3057 /*EGL_WIDTH*/, 64,
                                   0x3056 /*EGL_HEIGHT*/, 64, kNone };
        EGLSurface surf = eglCreatePbufferSurface(d, cfg, pbAttrs);
        if (!surf) { fprintf(stderr, "eglctx: pbuffer failed 0x%x\n", eglGetError()); continue; }
        // Compatibility profile: HdSt/HgiGL still issues a few legacy enums
        // (e.g. GL_POINT_SMOOTH / GL_LINE_SMOOTH state save-restore) that raise
        // GL_INVALID_ENUM in a strict core profile.
        EGLint attrs[] = { 0x3098, major, 0x30FB, minor,
                           0x30FD /*PROFILE_MASK*/, profileBit, kNone };
        EGLContext c = eglCreateContext(d, cfg, (EGLContext)0, attrs);
        if (!c) { fprintf(stderr, "eglctx: createContext failed 0x%x\n", eglGetError()); continue; }
        if (!eglMakeCurrent(d, surf, surf, c)) {
            fprintf(stderr, "eglctx: makeCurrent failed 0x%x\n", eglGetError()); continue; }
        fprintf(stderr, "eglctx: GL context current on EGL device %d (EGL %d.%d)\n", i, ma, mi);
        return true;
    }
    fprintf(stderr, "eglctx: could not make a context current\n");
    return false;
}
} // namespace eglctx
