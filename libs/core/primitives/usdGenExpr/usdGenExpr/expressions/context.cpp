#include "usdGenExpr/expressions/context.h"

#include <cstring>

namespace usdGen::expr {
namespace {
constexpr Domain kCurve = Domain::Primitive | Domain::Point;
constexpr VariableInfo kVars[] = {
 // $value is polymorphic: its concrete type/shape belongs to the consumer.
 {"$value",Variable::Value,ScalarType::Invalid,0,Domain::All,
  "The consumer's own authored literal, in the consumer's own type."},
 {"$frame",Variable::Frame,ScalarType::Float64,1,Domain::All,"Stage time code."},
 {"$time",Variable::Time,ScalarType::Float64,1,Domain::All,"Stage time in seconds."},
 {"$index",Variable::Index,ScalarType::Float64,1,Domain::All,
  "Element index in the evaluation domain."},
 {"$count",Variable::Count,ScalarType::Float64,1,Domain::All,
  "Element count in the evaluation domain."},
 {"$seed",Variable::Seed,ScalarType::Int32,1,Domain::All,
  "The consuming operator's usdGen:seed; also seeds rand()."},
 {"$descId",Variable::DescId,ScalarType::Float64,1,Domain::All,
  "Stable id of the description being cooked."},
 {"$primIndex",Variable::PrimIndex,ScalarType::Float64,1,kCurve,"Strand index."},
 {"$primCount",Variable::PrimCount,ScalarType::Float64,1,kCurve,"Strand count."},
 {"$idLo",Variable::IdLo,ScalarType::UInt32,1,kCurve,
  "Low 32 bits of the stable strand id."},
 {"$idHi",Variable::IdHi,ScalarType::UInt32,1,kCurve,
  "High 32 bits of the stable strand id."},
 {"$id",Variable::Id,ScalarType::Float64,1,kCurve,
  "Stable strand id; exact only through 2^53, use $idLo/$idHi for the full 64 bits."},
 {"$u",Variable::U,ScalarType::Float64,1,kCurve,"Surface u at the strand root."},
 {"$v",Variable::V,ScalarType::Float64,1,kCurve,"Surface v at the strand root."},
 {"$faceId",Variable::FaceId,ScalarType::Int32,1,kCurve,
  "Index of the surface face the strand roots on."},
 {"$P",Variable::P,ScalarType::Float32,3,kCurve,"Current position."},
 {"$Pref",Variable::PRef,ScalarType::Float32,3,kCurve,
  "Rest position; reads as the current position when the source has no rest channel."},
 {"$rootP",Variable::RootP,ScalarType::Float32,3,kCurve,
  "Current position of the strand root."},
 {"$rootPref",Variable::RootPRef,ScalarType::Float32,3,kCurve,
  "Rest position of the strand root."},
 {"$N",Variable::N,ScalarType::Float32,3,kCurve,
  "Surface normal of the strand's rest root frame."},
 {"$Nref",Variable::NRef,ScalarType::Float32,3,kCurve,
  "Rest surface normal at the root (usdGen keeps one rest root frame, so this equals $N)."},
 {"$dPdu",Variable::DPdu,ScalarType::Float32,3,kCurve,
  "Surface tangent along u of the strand's rest root frame."},
 {"$dPdv",Variable::DPdv,ScalarType::Float32,3,kCurve,
  "Surface bitangent along v of the strand's rest root frame."},
 {"$dPduref",Variable::DPduRef,ScalarType::Float32,3,kCurve,
  "Rest surface tangent along u at the root (equals $dPdu)."},
 {"$dPdvref",Variable::DPdvRef,ScalarType::Float32,3,kCurve,
  "Rest surface bitangent along v at the root (equals $dPdv)."},
 {"$t",Variable::T,ScalarType::Float64,1,kCurve,
  "Root-to-tip parameter: 0 at the root, 1 at the tip; 0 at primitive rate."},
 {"$pointIndex",Variable::PointIndex,ScalarType::Float64,1,Domain::Point,
  "CV index within the strand."},
 {"$pointCount",Variable::PointCount,ScalarType::Float64,1,Domain::Point,
  "CV count of the strand."},
 {"$cLength",Variable::CLength,ScalarType::Float32,1,kCurve,"Strand arc length."},
 {"$cWidth",Variable::CWidth,ScalarType::Float32,1,kCurve,"Strand width at this sample."},
 // No domain: only a geoSampler() element expression reads these.
 {"$Q",Variable::Q,ScalarType::Float64,3,Domain::None,
  "geoSampler() element expressions only: the position the sampler was queried at."},
 {"$Qdist",Variable::QDist,ScalarType::Float64,1,Domain::None,
  "geoSampler() element expressions only: distance from the element's $P to $Q."},
};
constexpr size_t kVarCount = sizeof(kVars) / sizeof(kVars[0]);
uint32_t Bytes(ScalarType t) { switch(t) { case ScalarType::Bool: return 1; case ScalarType::Int32: case ScalarType::UInt32: case ScalarType::Float32: return 4; case ScalarType::Int64: case ScalarType::UInt64: case ScalarType::Float64: return 8; case ScalarType::Float16: return 2; default: return 0; } }
}

namespace {
constexpr Registry kRegistry{};  // stateless: the tables above are constant
}
const Registry &Registry::Get() { return kRegistry; }
const VariableInfo *Registry::Find(const char *name) const noexcept {
    if (!name) return nullptr; for (auto const &v : kVars) if (std::strcmp(v.name,name)==0) return &v; return nullptr;
}
const VariableInfo *Registry::Find(Variable id) const noexcept {
    for (auto const &v : kVars) if (v.id==id) return &v; return nullptr;
}
bool Registry::Validate(const char *name, Domain domain, std::string *d, bool samplerElement) const {
    auto v=Find(name); if (!v) { if(d)*d=std::string("unknown expression variable ")+ (name?name:"<null>"); return false; }
    if (domain != Domain::Groom && domain != Domain::Primitive && domain != Domain::Point) { if(d)*d="invalid expression evaluation domain"; return false; }
    if (IsSamplerVariable(v->id)) {
        if (samplerElement) return true;
        if(d)*d=std::string("variable ")+v->name+" is only available inside a geoSampler() element expression";
        return false;
    }
    if (!HasDomain(v->domains,domain)) { if(d)*d=std::string("variable ")+v->name+" unavailable at requested evaluation domain"; return false; } return true;
}
size_t Registry::Count() const noexcept { return kVarCount; }
const VariableInfo *Registry::At(size_t i) const noexcept { return i < kVarCount ? &kVars[i] : nullptr; }

const char *ScalarTypeName(ScalarType t) noexcept {
    switch (t) {
    case ScalarType::Bool: return "bool";
    case ScalarType::Int32: return "int32";
    case ScalarType::UInt32: return "uint32";
    case ScalarType::Int64: return "int64";
    case ScalarType::UInt64: return "uint64";
    case ScalarType::Float16: return "float16";
    case ScalarType::Float32: return "float32";
    case ScalarType::Float64: return "float64";
    default: return "invalid";
    }
}
const char *DomainName(Domain d) noexcept {
    switch (d) {
    case Domain::Groom: return "groom";
    case Domain::Primitive: return "primitive";
    case Domain::Point: return "point";
    default: return "";
    }
}
uint32_t DescriptionId(const char *p) noexcept { uint32_t h=2166136261u; if(!p)return h; for(;*p;++p){h^=uint8_t(*p);h*=16777619u;} return h; }
bool ValidateValueShape(ValueView const &v, ValueShape const &e, std::string *d) {
    if(v.scalar!=e.scalar || v.isArray!=e.isArray || v.elementCount!=e.elementCount || v.components!=e.components || v.rows!=e.rows || v.columns!=e.columns) { if(d)*d="expression value shape/type does not match destination"; return false; }
    const uint64_t bytes = uint64_t(Bytes(v.scalar)) * v.components;
    if(!v.data || Bytes(v.scalar)==0 || bytes > UINT32_MAX || (v.strideBytes && v.strideBytes < bytes)) { if(d)*d="expression value buffer is invalid"; return false; } return true;
}
}
