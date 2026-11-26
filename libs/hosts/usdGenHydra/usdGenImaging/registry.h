// usdGen imaging registry: process-global, keyed per render instance.
// M0: minimal. M1 adds session (stage, sessionId) keyed generation store.
#ifndef USDGEN_IMAGING_REGISTRY_H
#define USDGEN_IMAGING_REGISTRY_H

#include <string>

namespace usdGenImaging {

// Opaque handle reserved for the M1 (stage, sessionId) session model.
class UsdGenImagingRegistry {
public:
    static UsdGenImagingRegistry& GetInstance();

    // Records that a scene index chain was attached for a render instance.
    void RegisterRenderInstance(const std::string& renderInstanceId);

private:
    UsdGenImagingRegistry() = default;
};

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_REGISTRY_H
