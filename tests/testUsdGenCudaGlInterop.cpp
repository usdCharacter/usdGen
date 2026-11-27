// Real Storm buffer allocation/computation and GL rasterization. Only the
// final image is read back; generated geometry never crosses the host.
#include "utils/pxrInterop.h"
#include "usdGenImaging/cudaGlComputation.h"
#include "usdGen/executionPipeline.h"
#include "cudaGlFixture.h"
#include "eglctx.h"
#include "pxr/base/tf/errorMark.h"
#include "pxr/imaging/garch/glApi.h"
#include "pxr/imaging/hd/bufferSpec.h"
#include "pxr/imaging/hd/driver.h"
#include "pxr/imaging/hdSt/bufferArrayRange.h"
#include "pxr/imaging/hdSt/bufferResource.h"
#include "pxr/imaging/hdSt/resourceRegistry.h"
#include "pxr/imaging/hdSt/renderDelegate.h"
#include "pxr/imaging/hgiGL/buffer.h"
#include "pxr/imaging/hgiGL/hgi.h"
#include <array>
#include <cstdio>
#include <memory>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using usdGenImaging::UsdGenCudaGlComputation;
using usdGenTest::MakeCudaGlFixture;
using Semantic = usdGen::UsdGenDeviceChannelSemantic;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (false)

namespace {
constexpr std::array<Semantic,8> channels{Semantic::Points, Semantic::RestPoints,
    Semantic::Widths, Semantic::HairT, Semantic::ElementOffsets,
    Semantic::StableIds, Semantic::RootPrim, Semantic::RootUV};
struct Buffers {
    std::array<HdBufferArrayRangeSharedPtr,8> ranges;
    std::array<std::shared_ptr<UsdGenCudaGlComputation>,8> copies;
};
Buffers Queue(HdStResourceRegistry& registry,
              std::shared_ptr<const usdGen::UsdGenDeviceGeneration> generation,
              bool interleaved = false, bool rejectLastChannel = false) {
    Buffers result;
    for (size_t i = 0; i < channels.size(); ++i) {
        auto copy = std::make_shared<UsdGenCudaGlComputation>(generation, channels[i], TfToken("data"));
        HdBufferSpecVector specs;
        copy->GetBufferSpecs(&specs);
        // Test-only admission failure after other channels may have copied.
        if (rejectLastChannel && i + 1 == channels.size()) {
            specs.back().tupleType = HdTupleType{HdTypeFloatVec3, 1};
        }
        // Same role/spec permits aggregation. Later copies must preserve
        // the earlier range in the same backing GL allocation.
        if (interleaved) {
            specs.insert(specs.begin(), HdBufferSpec(TfToken("aaaPadding"), {HdTypeUInt32Vec4,1}));
            result.ranges[i] = registry.AllocateShaderStorageBufferArrayRange(
                TfToken("usdGenInteropTest"), specs, HdBufferArrayUsageHintBitsStorage);
        } else {
            result.ranges[i] = registry.AllocateNonUniformBufferArrayRange(
                TfToken("usdGenInteropTest"), specs, HdBufferArrayUsageHintBitsStorage);
        }
        registry.AddComputation(result.ranges[i], copy, HdStComputeQueueZero);
        result.copies[i] = std::move(copy);
    }
    return result;
}
GLuint Shader(GLenum type, char const* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint okay = 0; glGetShaderiv(shader, GL_COMPILE_STATUS, &okay);
    if (!okay) { char log[4096]; glGetShaderInfoLog(shader, sizeof(log), nullptr, log); std::fprintf(stderr, "%s\n", log); return 0; }
    return shader;
}
GLuint Program() {
    char const* vertex = R"GLSL(#version 450 core
layout(std430,binding=0) readonly buffer P { float p[]; };
layout(std430,binding=1) readonly buffer R { float r[]; };
layout(std430,binding=2) readonly buffer W { float w[]; };
layout(std430,binding=3) readonly buffer T { float t[]; };
layout(std430,binding=4) readonly buffer O { uint o[]; };
layout(std430,binding=5) readonly buffer I { uvec2 ids[]; };
layout(std430,binding=6) readonly buffer F { int f[]; };
layout(std430,binding=7) readonly buffer U { float uv[]; };
uniform int base[8];
uniform int stride[8];
uniform float expectedWidth;
out float hairWidth;
flat out int valid;
bool near(float a,float b) { return abs(a-b)<0.00001; }
void main() {
    bool ok = o[base[4]]==0u && o[base[4]+stride[4]]==2u && o[base[4]+2*stride[4]]==5u;
    ok = ok && all(equal(ids[base[5]],uvec2(0x0000005bu,0x12345678u)))
            && all(equal(ids[base[5]+stride[5]],uvec2(0x00000025u,0xabcdef01u)));
    ok = ok && f[base[6]]==4 && f[base[6]+stride[6]]==8;
    ok = ok && near(uv[base[7]],.1) && near(uv[base[7]+1],.2)
            && near(uv[base[7]+stride[7]],.3) && near(uv[base[7]+stride[7]+1],.4);
    for(int j=0;j<5;++j) {
        float x = j<2 ? -.5 : .5;
        float y = j==0 || j==2 ? -.6 : j==3 ? 0.0 : .6;
        float h = j==0 || j==2 ? 0.0 : j==3 ? .5 : 1.0;
        ok = ok && near(p[base[0]+j*stride[0]],x) && near(p[base[0]+j*stride[0]+1],y)
                && near(p[base[0]+j*stride[0]+2],0.0) && near(t[base[3]+j*stride[3]],h)
                && near(w[base[2]+j*stride[2]],expectedWidth);
        for(int k=0;k<3;++k) ok = ok && near(r[base[1]+j*stride[1]+k],p[base[0]+j*stride[0]+k]);
    }
    // Curve endpoints come from GPU topology, not a host offsets mirror.
    int curve=gl_VertexID/2;
    uint index = gl_VertexID%2==0 ? o[base[4]+curve*stride[4]] : o[base[4]+(curve+1)*stride[4]]-1u;
    // Clamp only to contain a broken test input; validation still turns red.
    index=min(index,4u);
    int v=base[0]+int(index)*stride[0];
    gl_Position=vec4(p[v],p[v+1],p[v+2],1.0);
    hairWidth=w[base[2]+int(index)*stride[2]];
    valid=ok?1:0;
})GLSL";
    char const* geometry = R"GLSL(#version 450 core
layout(lines) in;
layout(triangle_strip,max_vertices=4) out;
in float hairWidth[];
flat in int valid[];
flat out int good;
void main() {
    good=valid[0]&valid[1];
    for(int end=0;end<2;++end) for(int side=-1;side<=1;side+=2) {
        gl_Position=gl_in[end].gl_Position+vec4(float(side)*hairWidth[end]*.5,0,0,0);
        EmitVertex();
    }
    EndPrimitive();
})GLSL";
    char const* fragment = R"GLSL(#version 450 core
flat in int good;
out vec4 color;
void main() { color=good==1?vec4(0,1,0,1):vec4(1,0,0,1); }
)GLSL";
    GLuint v=Shader(GL_VERTEX_SHADER,vertex), g=Shader(GL_GEOMETRY_SHADER,geometry), f=Shader(GL_FRAGMENT_SHADER,fragment);
    if (!v || !g || !f) return 0;
    GLuint program=glCreateProgram();
    glAttachShader(program,v); glAttachShader(program,g); glAttachShader(program,f); glLinkProgram(program);
    glDeleteShader(v); glDeleteShader(g); glDeleteShader(f);
    GLint okay=0; glGetProgramiv(program,GL_LINK_STATUS,&okay);
    if (!okay) { char log[4096]; glGetProgramInfoLog(program,sizeof(log),nullptr,log); std::fprintf(stderr,"%s\n",log); return 0; }
    return program;
}
int Draw(Buffers const& buffers, GLuint program, float width) {
    std::array<GLint,8> offsets{};
    std::array<GLint,8> strides{};
    for(size_t i=0;i<channels.size();++i) {
        auto bar=std::dynamic_pointer_cast<HdStBufferArrayRange>(buffers.ranges[i]);
        if (!bar) return -1;
        auto resource=bar->GetResource(TfToken("data"));
        auto* buffer=resource ? dynamic_cast<HgiGLBuffer*>(resource->GetHandle().Get()) : nullptr;
        if (!buffer) return -1;
        // Bind the entire allocation (BAR byte offsets need not meet SSBO
        // binding alignment). Shader indexing applies the range's offset.
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,static_cast<GLuint>(i),buffer->GetBufferId());
        offsets[i]=(bar->GetByteOffset(TfToken("data"))+resource->GetOffset())/(i==5?8:4);
        strides[i]=resource->GetStride()/(i==5?8:4);
    }
    glUseProgram(program);
    glUniform1iv(glGetUniformLocation(program,"base[0]"),8,offsets.data());
    glUniform1iv(glGetUniformLocation(program,"stride[0]"),8,strides.data());
    glUniform1f(glGetUniformLocation(program,"expectedWidth"),width);
    glViewport(0,0,64,64); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);
    glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES,0,4);
    std::array<unsigned char,64*64*4> pixels{};
    glReadPixels(0,0,64,64,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    if(glGetError()!=GL_NO_ERROR) return -1;
    int green=0, red=0;
    for(size_t i=0;i<pixels.size();i+=4) { if(pixels[i]>128) ++red; if(pixels[i+1]>128) ++green; }
    std::printf("width %.3f: green=%d red=%d\n",width,green,red);
    return red ? -1 : green;
}

}

int main() {
    TfErrorMark errors;
    if(!eglctx::MakeHeadlessGLContext()) return 77;
    GarchGLApiLoad();
    HgiGL hgi;
    // Let the SDK construct its own registry. Its public header has a
    // MaterialX-conditional member but the SDK does not export that feature
    // macro to consumers, so allocating sizeof(HdStResourceRegistry) here
    // would assume an ABI/layout which may differ from the shared library.
    HdStRenderDelegate delegate;
    HdDriver driver{HgiTokens->renderDriver, VtValue(static_cast<Hgi*>(&hgi))};
    delegate.SetDrivers({&driver});
    auto registryOwner = std::dynamic_pointer_cast<HdStResourceRegistry>(delegate.GetResourceRegistry());
    CHECK(registryOwner);
    auto& registry = *registryOwner;
    auto first=MakeCudaGlFixture(.1f), second=MakeCudaGlFixture(.2f);
    CHECK(first && second);
    std::weak_ptr<const usdGen::UsdGenDeviceOwner> retained=first->Owner();
    auto a=Queue(registry,first), b=Queue(registry,second);
    auto c=Queue(registry,MakeCudaGlFixture(.3f),true);
    first.reset(); second.reset();
    CHECK(!retained.expired());
    registry.Commit();
    for(auto const* set : {&a,&b,&c}) for(auto const& copy:set->copies) {
        if(!copy->Succeeded()) std::fprintf(stderr,"interop: %s\n",copy->Error().c_str());
        CHECK(copy->Succeeded());
    }
    GLuint program=Program(), vao=0;
    CHECK(program);
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    int thin=Draw(a,program,.1f), thick=Draw(b,program,.2f);
    CHECK(thin>100 && thick>thin);
    CHECK(Draw(c,program,.3f)>thick);
    auto interleaved=std::dynamic_pointer_cast<HdStBufferArrayRange>(c.ranges[0]);
    CHECK(interleaved->GetResource(TfToken("data"))->GetOffset()>0);
    CHECK(interleaved->GetResource(TfToken("data"))->GetStride()>12);
    auto aPoints=std::dynamic_pointer_cast<HdStBufferArrayRange>(a.ranges[0]);
    auto bPoints=std::dynamic_pointer_cast<HdStBufferArrayRange>(b.ranges[0]);
    CHECK(aPoints->GetResource(TfToken("data"))->GetHandle()==bPoints->GetResource(TfToken("data"))->GetHandle());
    CHECK(aPoints->GetByteOffset(TfToken("data"))!=bPoints->GetByteOffset(TfToken("data")));
    CHECK(Draw(a,program,.1f)==thin); // no adjacent-range corruption
    {
        UsdGenCudaGlComputation missing({},Semantic::Points,TfToken("data"));
        HdBufferSpecVector specs;
        missing.GetBufferSpecs(&specs);
        CHECK(specs.empty() && missing.GetNumOutputElements()==0);
        missing.Execute(a.ranges[0],&registry);
        CHECK(!missing.Succeeded() && !missing.Error().empty());
        auto validationSource=MakeCudaGlFixture(.8f);
        CHECK(validationSource);
        UsdGenCudaGlComputation unsupported(validationSource,Semantic::Generic,TfToken("data"));
        unsupported.Execute(a.ranges[0],&registry);
        CHECK(!unsupported.Succeeded());
        UsdGenCudaGlComputation mismatch(validationSource,Semantic::Widths,TfToken("data"));
        mismatch.Execute(a.ranges[0],&registry); // widths cannot overwrite points
        CHECK(!mismatch.Succeeded() && !mismatch.Error().empty());
        mismatch.Execute({},&registry);
        CHECK(!mismatch.Succeeded());
        usdGen::UsdGenExecutionRuntime runtime(2);
        usdGen::UsdGenExecutionPipeline pipeline(runtime);
        pipeline.InvokeOwner([&] { mismatch.Execute(a.ranges[2],&registry); });
        CHECK(!mismatch.Succeeded() && mismatch.Error().find("external graphics")!=std::string::npos);
        pipeline.Shutdown();
        CHECK(Draw(a,program,.1f)==thin); // rejected work did not alter destination
    }
    for(auto& copy:a.copies) copy.reset();
    for(auto& copy:b.copies) copy.reset();
    for(auto& copy:c.copies) copy.reset();
    CHECK(retained.expired());
    CHECK(Draw(a,program,.1f)==thin); // GL allocation owns its D2D result
    glBindVertexArray(0); glDeleteVertexArrays(1,&vao); glUseProgram(0); glDeleteProgram(program);
    CHECK(errors.IsClean());
    std::puts("testUsdGenCudaGlInterop: PASS (Storm BAR -> GL draw; not full BasisCurves publication)");
    return 0;
}
