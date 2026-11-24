// Engine-neutral automatic C3 rest-surface bindings.  This helper contains
// no scene-index, USD-stage, or device state: callers own the policy for when
// a successful candidate replaces authored C3 bindings.
#ifndef USDGEN_SURFACE_ROOT_BINDINGS_H
#define USDGEN_SURFACE_ROOT_BINDINGS_H

#include "usdGen/graphDesc.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

struct UsdGenSurfaceRootBindingResult
{
    // Fresh, owned parent-mesh bindings.  An unresolved root has rootPrim=-1
    // and rootUV=(0,0), never a plausible binding to face zero.
    VtIntArray rootPrim;
    VtVec2fArray rootUV;

    // `valid[i]` is one exactly when the candidate search produced a binding.
    // `unresolved[i]` distinguishes a finite root for which no candidate
    // surface patch could be solved from a structural input failure, which
    // returns false and leaves the entire result untouched.
    VtArray<uint8_t> valid;
    VtArray<uint8_t> unresolved;
    uint32_t unresolvedCount = 0;
};

/// Immutable spatial index over one parent surface's rest snapshot.  It is
/// safe to retain across captures and to call Bind concurrently: every query
/// allocates its result and k-nearest scratch locally.  `Matches` deliberately
/// compares the full spatial snapshot rather than trusting a generation bump.
class UsdGenRestSurfaceBindingCache final
{
public:
    static std::shared_ptr<const UsdGenRestSurfaceBindingCache> Create(
        UsdGenSurfaceDesc const &surface, std::string *error = nullptr);

    UsdGenRestSurfaceBindingCache(UsdGenRestSurfaceBindingCache const &) = default;
    UsdGenRestSurfaceBindingCache &operator=(UsdGenRestSurfaceBindingCache const &) = default;
    ~UsdGenRestSurfaceBindingCache();

    bool Matches(UsdGenSurfaceDesc const &surface) const;
    bool Bind(VtVec3fArray const &sourceRestRoots,
              GfMatrix4d const &sourceWorldMatrix,
              UsdGenSurfaceRootBindingResult *out,
              std::string *error = nullptr) const;
    size_t BytesOwned() const noexcept;

private:
    struct Impl;
    explicit UsdGenRestSurfaceBindingCache(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> _impl;
};

/// Compatibility one-shot wrapper around UsdGenRestSurfaceBindingCache::Create
/// and Bind. Computes parent-mesh skinprim/skinprimuv bindings for the supplied
/// source-local rest roots (one root per curve). `sourceWorldMatrix` converts
/// those roots into the surface's post-flattened local rest space; both
/// transforms must be finite affine, and the surface transform invertible.
/// The surface must have an authored or default-time rest snapshot;
/// current-frame fallback is rejected. A nanoflann k=8 face-centroid query
/// narrows each root's exact patch tests.
/// Triangles and n-gon fan triangles use an exact closest-point projection;
/// quads use a bounded deterministic bilinear-patch closest solve so the
/// resulting (u,v) retains the C3 bilinear convention. Exact-distance ties
/// are broken by lower parent face then fan only within that selected k=8
/// candidate set; the centroid cut-off is intentionally not a global tie
/// search across every mesh face.
///
/// Topology, provenance, non-finite input, and allocation failures return
/// false and do not replace `out`.  Per-root geometric degeneracy or a bounded
/// quad solve with no converged candidate succeeds with an explicit unresolved
/// entry, allowing a loader to apply its rebind/drop policy transactionally.
bool UsdGenBuildRestSurfaceRootBindings(
    UsdGenSurfaceDesc const &surface,
    VtVec3fArray const &sourceRestRoots,
    GfMatrix4d const &sourceWorldMatrix,
    UsdGenSurfaceRootBindingResult *out,
    std::string *error = nullptr);

} // namespace usdGen

#endif // USDGEN_SURFACE_ROOT_BINDINGS_H
