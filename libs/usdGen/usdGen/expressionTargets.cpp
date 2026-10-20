#include "usdGen/expressionTargets.h"

#include <set>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

using Type = expr::ScalarType;
using Domain = expr::Domain;

bool Reject(std::vector<std::string> *errors, std::string const &message)
{
    if (errors) errors->push_back(message);
    return false;
}

/// scalar/native-type/shape agreement. `vector` marks the one two-component
/// destination (Length's `length:random`).
bool ShapeOk(UsdGenExpressionBinding const &binding, Type scalar, char const *nativeType,
             unsigned components)
{
    auto const &shape = binding.destinationShape;
    return !shape.isArray && shape.elementCount == 1 && shape.rows == 1 &&
        shape.columns == 1 && shape.components == components && shape.scalar == scalar &&
        binding.nativeType == TfToken(nativeType);
}

bool ValidateLength(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "length:value", "minRemainingLength", "cullThreshold", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled", vector = name == "length:random";
        if ((!boolean && !vector && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : vector ? "float2" : "float", vector ? 2u : 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed Length expression " + name);
            continue;
        }
        if ((boolean && binding.domain != Domain::Groom) ||
            ((name == "cullThreshold" || vector) && binding.domain == Domain::Point))
            ok = Reject(errors, "Length enabled requires groom; cullThreshold/random require groom/primitive");
    }
    return ok;
}

bool ValidateWidth(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{"width", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "replace" || name == "enabled";
        if ((!boolean && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported/incorrectly typed Width expression target " + name);
            continue;
        }
        if (name == "enabled" && binding.domain != Domain::Groom)
            ok = Reject(errors, "Width enabled must evaluate at groom granularity");
    }
    return ok;
}

bool ValidateNoise(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "noise:magnitude", "noise:frequency", "noise:correlation",
        "noise:lacunarity", "noise:gain", "preserveLength", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled" || name == "cumulative";
        const bool integer = name == "noise:octaves" || name == "noise:seed";
        if ((!boolean && !integer && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : integer ? Type::Int32 : Type::Float32,
                     boolean ? "bool" : integer ? "int" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed Noise expression " + name);
            continue;
        }
        if ((name == "enabled" && binding.domain != Domain::Groom) ||
            (name == "cumulative" && binding.domain == Domain::Point) ||
            (name == "noise:seed" && binding.domain == Domain::Point))
            ok = Reject(errors, "Noise expression domain is unsupported for " + name);
    }
    return ok;
}

bool ValidateDeform(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled" || name == "lockRoots";
        const bool integer = name == "rbfSamples";
        if ((!boolean && !integer && name != "mask") || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : integer ? Type::Int32 : Type::Float32,
                     boolean ? "bool" : integer ? "int" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed RBF expression " + name);
            continue;
        }
        if (((name == "enabled" || integer) && binding.domain != Domain::Groom) ||
            (name == "lockRoots" && binding.domain == Domain::Point))
            ok = Reject(errors, "RBF structural/enabled controls require groom; root locking requires groom/primitive");
    }
    return ok;
}

/// Clump resolves at capture; its amount may vary per CV, its membership map
/// is per strand.
bool ValidateClump(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "clump:amount", "clump:map", "preserveLength", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        if ((!boolean && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed Clump expression " + name);
            continue;
        }
        if ((boolean && binding.domain != Domain::Groom) ||
            (name == "clump:map" && binding.domain == Domain::Point))
            ok = Reject(errors, "Clump enabled requires groom; clump:map requires groom/primitive");
    }
    return ok;
}

/// GuideInterpolate reads its region per strand root.
bool ValidateGuideInterpolate(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        if ((!boolean && name != "region") || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed GuideInterpolate expression " + name);
            continue;
        }
        if ((boolean && binding.domain != Domain::Groom) ||
            (name == "region" && binding.domain == Domain::Point))
            ok = Reject(errors, "GuideInterpolate enabled requires groom; region requires groom/primitive");
    }
    return ok;
}

bool ValidateCurveSource(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool resample = name == "resampleTo" && ShapeOk(binding, Type::Int32, "int", 1u);
        const bool useRest = name == "useRest" && ShapeOk(binding, Type::Bool, "bool", 1u);
        if ((!resample && !useRest) || !seen.insert(name).second ||
            binding.domain != Domain::Groom)
            ok = Reject(errors, "CurveSource only supports groom native resampleTo/useRest expressions");
    }
    return ok;
}

/// Grow is a generator, but its per-strand controls (length, lift, the
/// lengthRandom range and the directionVector) are sampled per curve root at
/// capture, so a connected primitive-domain expression combs the fade stubble
/// differently from the top hair. Point-domain values would be sampled at the
/// root CV only, which silently ignores per-CV variation; they are rejected
/// so the authoring fails closed instead of partially applying.
bool ValidateGrow(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{"length", "lift", "azimuth", "azimuthRandom"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        const bool integer = name == "segments";
        const bool vec2 = name == "lengthRandom";
        const bool vec3 = name == "directionVector";
        bool shape = false;
        if (boolean) shape = ShapeOk(binding, Type::Bool, "bool", 1u);
        else if (integer) shape = ShapeOk(binding, Type::Int32, "int", 1u);
        else if (vec2) shape = ShapeOk(binding, Type::Float32, "float2", 2u);
        else if (vec3) shape = ShapeOk(binding, Type::Float32, "vector3f", 3u);
        else if (floats.count(name)) shape = ShapeOk(binding, Type::Float32, "float", 1u);
        if (!shape || !seen.insert(name).second) {
            ok = Reject(errors, "unsupported or incorrectly typed Grow expression " + name);
            continue;
        }
        if ((boolean || integer) && binding.domain != Domain::Groom)
            ok = Reject(errors, "Grow enabled/segments require groom evaluation");
        else if (!boolean && !integer && binding.domain == Domain::Point)
            ok = Reject(errors, "Grow " + name + " requires groom/primitive evaluation");
    }
    return ok;
}

/// Curl coils each strand around its tangent frame. Radius and the mask are
/// sampled per CV; frequency, phase and phaseRandom are sampled per curve
/// root. The branch toggles (taper, clockwise) are groom-wide.
bool ValidateCurl(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "radius", "frequency", "phase", "phaseRandom", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled" || name == "taper" || name == "clockwise";
        if ((!boolean && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed Curl expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "Curl enabled/taper/clockwise require groom evaluation");
    }
    return ok;
}

/// Bend rotates downstream CVs about each segment point. The angle is sampled
/// per CV (the incremental bend rate); the angleRandom range and the axis are
/// sampled per curve root.
bool ValidateBend(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{"angle", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        const bool vec2 = name == "angleRandom";
        const bool vec3 = name == "axis";
        bool shape = false;
        if (boolean) shape = ShapeOk(binding, Type::Bool, "bool", 1u);
        else if (vec2) shape = ShapeOk(binding, Type::Float32, "float2", 2u);
        else if (vec3) shape = ShapeOk(binding, Type::Float32, "vector3f", 3u);
        else if (floats.count(name)) shape = ShapeOk(binding, Type::Float32, "float", 1u);
        if (!shape || !seen.insert(name).second) {
            ok = Reject(errors, "unsupported or incorrectly typed Bend expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "Bend enabled requires groom evaluation");
        else if ((vec2 || vec3) && binding.domain == Domain::Point)
            ok = Reject(errors, "Bend " + name + " requires groom/primitive evaluation");
    }
    return ok;
}

/// Wave displaces along the root tangent/normal by sine fields. Amplitudes
/// and the mask are sampled per CV; frequencies are sampled per curve root.
/// Whole-strand stylers (Braid, Twist, Trim, Frizz, TieClump): the result is
/// computed at capture over every strand, so only the per-CV usdGen:mask
/// envelope (and the groom-level enabled toggle) is connectable.
bool ValidateWholeStrand(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        if ((!boolean && name != "mask") || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed " + node.type.GetString() +
                        " expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, node.type.GetString() + " enabled requires groom evaluation");
    }
    return ok;
}

bool ValidateWave(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "frequencyU", "frequencyN", "amplitudeU", "amplitudeN", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        if ((!boolean && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed Wave expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "Wave enabled requires groom evaluation");
    }
    return ok;
}

/// Part emits partId from a parting curve set. Radius and strength are sampled
/// per curve root; the mask envelopes the separating displacement per CV.
bool ValidatePart(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "part:radius", "part:strength", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        if ((!boolean && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed Part expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "Part enabled requires groom evaluation");
        else if (!boolean && name != "mask" && binding.domain == Domain::Point)
            ok = Reject(errors, "Part " + name + " requires groom/primitive evaluation");
    }
    return ok;
}

/// Direction combs toward a target: direction/amount/lift are sampled per
/// curve root; the mask is sampled per CV; the tilt/around/follow shaping
/// and enabled are groom-wide.
bool ValidateDirection(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> groomFloats{
        "tiltU", "tiltV", "tiltN", "aroundN", "followSkinContour"};
    static const std::set<std::string> rootFloats{"amount", "lift", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        const bool vec3 = name == "direction";
        bool shape = false;
        if (boolean) shape = ShapeOk(binding, Type::Bool, "bool", 1u);
        else if (vec3) shape = ShapeOk(binding, Type::Float32, "vector3f", 3u);
        else if (groomFloats.count(name) || rootFloats.count(name))
            shape = ShapeOk(binding, Type::Float32, "float", 1u);
        if (!shape || !seen.insert(name).second) {
            ok = Reject(errors, "unsupported or incorrectly typed Direction expression " + name);
            continue;
        }
        if ((boolean || groomFloats.count(name)) && binding.domain != Domain::Groom)
            ok = Reject(errors, "Direction " + name + " requires groom evaluation");
        else if ((vec3 || name == "amount" || name == "lift") &&
                 binding.domain == Domain::Point)
            ok = Reject(errors, "Direction " + name + " requires groom/primitive evaluation");
    }
    return ok;
}

/// Smooth relaxes strands: strength, searchRadius and numNeighbors are
/// sampled per curve root; the mask is sampled per CV; iterations, lockRoot
/// and enabled are groom-wide.
bool ValidateSmooth(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> rootFloats{"strength", "searchRadius", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled" || name == "lockRoot";
        const bool integer = name == "iterations" || name == "numNeighbors";
        bool shape = false;
        if (boolean) shape = ShapeOk(binding, Type::Bool, "bool", 1u);
        else if (integer) shape = ShapeOk(binding, Type::Int32, "int", 1u);
        else if (rootFloats.count(name))
            shape = ShapeOk(binding, Type::Float32, "float", 1u);
        if (!shape || !seen.insert(name).second) {
            ok = Reject(errors, "unsupported or incorrectly typed Smooth expression " + name);
            continue;
        }
        if ((boolean || name == "iterations") && binding.domain != Domain::Groom)
            ok = Reject(errors, "Smooth " + name + " requires groom evaluation");
        else if (name != "mask" && !boolean && name != "iterations" &&
                 binding.domain == Domain::Point)
            ok = Reject(errors, "Smooth " + name + " requires groom/primitive evaluation");
    }
    return ok;
}

/// Resample repartitions to cvCount: the count and the restore toggle are
/// groom-wide; the mask is sampled at each output CV's nearest input CV.
bool ValidateResample(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled" || name == "restoreSegmentLengths";
        const bool integer = name == "cvCount";
        bool shape = false;
        if (boolean) shape = ShapeOk(binding, Type::Bool, "bool", 1u);
        else if (integer) shape = ShapeOk(binding, Type::Int32, "int", 1u);
        else if (name == "mask") shape = ShapeOk(binding, Type::Float32, "float", 1u);
        if (!shape || !seen.insert(name).second) {
            ok = Reject(errors, "unsupported or incorrectly typed Resample expression " + name);
            continue;
        }
        if ((boolean || integer) && binding.domain != Domain::Groom)
            ok = Reject(errors, "Resample " + name + " requires groom evaluation");
    }
    return ok;
}

/// Scale multiplies offsets about each root: scale and the scaleRandom range
/// are sampled per curve root; the mask is sampled per CV; enabled and
/// widthToo are groom-wide.
bool ValidateScale(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled" || name == "widthToo";
        const bool vec2 = name == "scaleRandom";
        bool shape = false;
        if (boolean) shape = ShapeOk(binding, Type::Bool, "bool", 1u);
        else if (vec2) shape = ShapeOk(binding, Type::Float32, "float2", 2u);
        else if (name == "scale" || name == "mask")
            shape = ShapeOk(binding, Type::Float32, "float", 1u);
        if (!shape || !seen.insert(name).second) {
            ok = Reject(errors, "unsupported or incorrectly typed Scale expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "Scale " + name + " requires groom evaluation");
        else if ((vec2 || name == "scale") && binding.domain == Domain::Point)
            ok = Reject(errors, "Scale " + name + " requires groom/primitive evaluation");
    }
    return ok;
}

/// Straighten blends toward the root-tip line: both straightness controls
/// and the mask are sampled per CV; enabled is groom-wide.
bool ValidateStraighten(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "tangentStraightness", "normalStraightness", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        if ((!boolean && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed Straighten expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "Straighten enabled requires groom evaluation");
    }
    return ok;
}

/// Displace offsets along the root normal: amount/base/scale/offset/mask are
/// sampled per CV; the displace:map sample arrives per curve root (mirror of
/// clump:map); enabled is groom-wide.
bool ValidateDisplace(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "displace:amount", "displace:base", "displace:scale",
        "displace:offset", "displace:map", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        if ((!boolean && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed Displace expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "Displace enabled requires groom evaluation");
        else if (name == "displace:map" && binding.domain == Domain::Point)
            ok = Reject(errors, "Displace displace:map requires groom/primitive evaluation");
    }
    return ok;
}

/// SculptLayer applies hand deltas keyed by stable curveId. Weight and the
/// mask scale the deltas; weight is groom-wide, the mask is sampled per CV.
bool ValidateSculptLayer(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{"sculpt:weight", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        if ((!boolean && !floats.count(name)) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed SculptLayer expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "SculptLayer enabled requires groom evaluation");
        else if (name == "sculpt:weight" && binding.domain != Domain::Groom)
            ok = Reject(errors, "SculptLayer sculpt:weight requires groom evaluation");
    }
    return ok;
}

/// Wind deflects strands along its direction: a constant term plus a
/// time-varying gust field. Direction and the strengths are sampled per
/// curve root; the stiffness baseline is groom-wide; the mask per CV.
bool ValidateWind(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{
        "constStrength", "gustStrength", "stiffness", "mask"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        const bool vec3 = name == "direction";
        bool shape = false;
        if (boolean) shape = ShapeOk(binding, Type::Bool, "bool", 1u);
        else if (vec3) shape = ShapeOk(binding, Type::Float32, "vector3f", 3u);
        else if (floats.count(name)) shape = ShapeOk(binding, Type::Float32, "float", 1u);
        if (!shape || !seen.insert(name).second) {
            ok = Reject(errors, "unsupported or incorrectly typed Wind expression " + name);
            continue;
        }
        if ((boolean || name == "stiffness") && binding.domain != Domain::Groom)
            ok = Reject(errors, "Wind enabled/stiffness require groom evaluation");
        else if (name != "mask" && binding.domain == Domain::Point)
            ok = Reject(errors, "Wind " + name + " requires groom/primitive evaluation");
    }
    return ok;
}

/// ExprOp runs one authored expression as a styler; the source itself is a
/// plain string control, so only the mask and enabled accept connections.
bool ValidateExprOp(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        const bool mask = name == "mask";
        if ((!boolean && !mask) || !seen.insert(name).second ||
            !ShapeOk(binding, boolean ? Type::Bool : Type::Float32,
                     boolean ? "bool" : "float", 1u)) {
            ok = Reject(errors, "unsupported or incorrectly typed ExprOp expression " + name);
            continue;
        }
        if (boolean && binding.domain != Domain::Groom)
            ok = Reject(errors, "ExprOp enabled requires groom evaluation");
    }
    return ok;
}

/// Collide pushes penetrating CVs out of the collider meshes. Offset and
/// push amount are sampled per curve root; the iteration count is
/// groom-wide; the mask is sampled per CV.
bool ValidateCollide(UsdGenNodeDesc const &node, std::vector<std::string> *errors)
{
    static const std::set<std::string> floats{"offset", "pushAmount", "mask",
                                              "clearance:root", "clearance:tip"};
    std::set<std::string> seen;
    bool ok = true;
    for (auto const &binding : node.expressionBindings) {
        const auto name = UsdGenCanonicalParamName(binding.destination).GetString();
        const bool boolean = name == "enabled";
        const bool integer = name == "iterations" || name == "relax:iterations";
        const bool clearance = name == "clearance:root" || name == "clearance:tip";
        bool shape = false;
        if (boolean) shape = ShapeOk(binding, Type::Bool, "bool", 1u);
        else if (integer) shape = ShapeOk(binding, Type::Int32, "int", 1u);
        else if (floats.count(name)) shape = ShapeOk(binding, Type::Float32, "float", 1u);
        if (!shape || !seen.insert(name).second) {
            ok = Reject(errors, "unsupported or incorrectly typed Collide expression " + name);
            continue;
        }
        if ((boolean || integer || clearance) && binding.domain != Domain::Groom)
            ok = Reject(errors, "Collide enabled/iterations/clearance require groom evaluation");
        else if (name != "mask" && binding.domain == Domain::Point)
            ok = Reject(errors, "Collide " + name + " requires groom/primitive evaluation");
    }
    return ok;
}
} // namespace

TfToken UsdGenCanonicalParamName(TfToken const &name)
{
    std::string value = name.GetString();
    if (value.compare(0, 7, "usdGen:") == 0) value.erase(0, 7);
    return TfToken(value);
}

bool UsdGenValidateExpressionTargets(UsdGenNodeDesc const &node,
                                     std::vector<std::string> *errors)
{
    if (node.expressionBindings.empty()) return true;
    if (node.type == TfToken("UsdGenWidth")) return ValidateWidth(node, errors);
    if (node.type == TfToken("UsdGenLength")) return ValidateLength(node, errors);
    if (node.type == TfToken("UsdGenNoise")) return ValidateNoise(node, errors);
    if (node.type == TfToken("UsdGenDeform")) return ValidateDeform(node, errors);
    if (node.type == TfToken("UsdGenCurveSource")) return ValidateCurveSource(node, errors);
    if (node.type == TfToken("UsdGenClump")) return ValidateClump(node, errors);
    if (node.type == TfToken("UsdGenGuideInterpolate"))
        return ValidateGuideInterpolate(node, errors);
    if (node.type == TfToken("UsdGenGrow")) return ValidateGrow(node, errors);
    if (node.type == TfToken("UsdGenCurl")) return ValidateCurl(node, errors);
    if (node.type == TfToken("UsdGenBend")) return ValidateBend(node, errors);
    if (node.type == TfToken("UsdGenWave")) return ValidateWave(node, errors);
    if (node.type == TfToken("UsdGenPart")) return ValidatePart(node, errors);
    if (node.type == TfToken("UsdGenDirection")) return ValidateDirection(node, errors);
    if (node.type == TfToken("UsdGenSmooth")) return ValidateSmooth(node, errors);
    if (node.type == TfToken("UsdGenResample")) return ValidateResample(node, errors);
    if (node.type == TfToken("UsdGenScale")) return ValidateScale(node, errors);
    if (node.type == TfToken("UsdGenStraighten")) return ValidateStraighten(node, errors);
    if (node.type == TfToken("UsdGenDisplace")) return ValidateDisplace(node, errors);
    if (node.type == TfToken("UsdGenSculptLayer")) return ValidateSculptLayer(node, errors);
    if (node.type == TfToken("UsdGenWind")) return ValidateWind(node, errors);
    if (node.type == TfToken("UsdGenExprOp")) return ValidateExprOp(node, errors);
    if (node.type == TfToken("UsdGenCollide")) return ValidateCollide(node, errors);
    if (node.type == TfToken("UsdGenBraid") || node.type == TfToken("UsdGenTwist") ||
        node.type == TfToken("UsdGenTrim") || node.type == TfToken("UsdGenFrizz") ||
        node.type == TfToken("UsdGenTieClump"))
        return ValidateWholeStrand(node, errors);
    // Scatter owns its topology and consumes no connected control: it would
    // silently ignore one on both lanes, which is worse than a diagnostic.
    // WidthBlend blends two proved widths and owns no parameter evaluation;
    // ReferenceSource accepts no parameters at all.
    return Reject(errors, "operator " + node.type.GetString() + " at " +
        node.path.GetString() + " does not accept connected (expression) parameters");
}

} // namespace usdGen
