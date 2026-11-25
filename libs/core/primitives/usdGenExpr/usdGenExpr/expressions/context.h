#ifndef USDGEN_EXPRESSIONS_CONTEXT_H
#define USDGEN_EXPRESSIONS_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace usdGen::expr {

enum class Domain : uint8_t { None = 0, Groom = 1, Primitive = 2, Point = 4,
                              All = 7 };
constexpr Domain operator|(Domain a, Domain b) {
    return Domain(uint8_t(a) | uint8_t(b));
}
constexpr bool HasDomain(Domain mask, Domain d) {
    return (uint8_t(mask) & uint8_t(d)) != 0;
}

enum class ScalarType : uint8_t { Invalid, Bool, Int32, UInt32, Int64, UInt64,
                                  Float16, Float32, Float64 };

// Non-owning, trivially-copyable payload view. Shape is explicit: arrays are
// not collapsed into point samples and matrices retain row-major dimensions.
struct ValueView {
    const void *data = nullptr;
    uint32_t elementCount = 0; // array length; 1 for scalar/vector/matrix
    uint32_t components = 1;
    uint32_t rows = 1;
    uint32_t columns = 1;
    uint32_t strideBytes = 0;
    ScalarType scalar = ScalarType::Invalid;
    bool isArray = false;
};

struct ValueShape {
    ScalarType scalar = ScalarType::Invalid;
    uint32_t elementCount = 1;
    uint32_t components = 1;
    uint32_t rows = 1;
    uint32_t columns = 1;
    bool isArray = false;
};

// $patchId, $Cs and $As were declared here through M1 but no field builder ever
// wrote them, so every expression that named one poisoned. usdGen has no patch
// table and the engine's curve buffer carries no surface colour/opacity at
// evaluation time, so they are gone rather than silently unavailable; see
// plan/07-look-maps-expressions.md.
// Q and QDist exist only inside a geoSampler() element expression (the query
// position and its distance to the element); their registry entries carry no
// domain, so an ordinary expression cannot name them.
enum class Variable : uint16_t {
    Invalid, Value, Frame, Time, Index, Count, Seed, DescId,
    PrimIndex, PrimCount, IdLo, IdHi, Id, U, V, FaceId,
    P, PRef, RootP, RootPRef, N, NRef, DPdu, DPdv, DPduRef, DPdvRef,
    T, PointIndex, PointCount, CLength, CWidth, Q, QDist, CountVariables
};

struct VariableInfo {
    const char *name;
    Variable id;
    ScalarType scalar;
    uint32_t components;
    Domain domains;
    // One line of prose. Every entry carries one; the editor's variable browser
    // and plan/07's table are generated from this field, so the language has a
    // single description of itself.
    const char *doc;
};

class Registry {
public:
    static const Registry &Get();
    const VariableInfo *Find(const char *name) const noexcept;
    const VariableInfo *Find(Variable id) const noexcept;
    /// `samplerElement` admits the variables that exist only inside a
    /// geoSampler() element expression ($Q, $Qdist).
    bool Validate(const char *name, Domain domain, std::string *diagnostic = nullptr,
                  bool samplerElement = false) const;
    /// True for the variables only a geoSampler() element expression may read.
    static bool IsSamplerVariable(Variable id) noexcept
    {
        return id == Variable::Q || id == Variable::QDist;
    }
    // Enumeration, for authoring tools that present the variable table to a
    // user (the usdview SeExpr editor's variable browser). At() returns
    // nullptr past the end so a caller can walk without a second call.
    size_t Count() const noexcept;
    const VariableInfo *At(size_t index) const noexcept;
};

// Spellings shared by the tool surfaces. ScalarTypeName returns the lowercase
// C-ish name ("float32", "int32", ...) and "invalid" for ScalarType::Invalid,
// which is how the polymorphic $value is recorded. DomainName takes ONE domain
// bit and returns "groom" | "primitive" | "point" (or "" for anything else) --
// the spellings used by the usdGen:evaluation customData.
const char *ScalarTypeName(ScalarType type) noexcept;
const char *DomainName(Domain domain) noexcept;

struct Context {
    double frame = 0.0;
    double time = 0.0;
    uint32_t index = 0;
    uint32_t count = 1;
    int32_t seed = 0;
    uint32_t descId = 0;
    Domain domain = Domain::Groom;
};

uint32_t DescriptionId(const char *descriptionPath) noexcept;
bool ValidateValueShape(ValueView const &value, ValueShape const &expected,
                        std::string *diagnostic = nullptr);

} // namespace usdGen::expr
#endif
