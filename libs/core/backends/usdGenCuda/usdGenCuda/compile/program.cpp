// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Key functions of the program interfaces: one typeinfo, in usdGenCuda; and
// the lookup of the program a graph runs.
#include "usdGen/api/token.h"
#include "usdGenCuda/compile/program.h"
#include "usdGenCuda/backendId.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/ext/backendDef.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/ext/typeRegistry.h"

namespace usdGen::cuda {

Job::~Job() = default;

UsdGenRef<const UsdGenPublication> Job::Report() const noexcept
{
    return {};
}

ProgramState::~ProgramState() = default;

ProgramDef const *ProgramOf(UsdGenTypeId type) noexcept
{
    return static_cast<ProgramDef const *>(
        UsdGenFindDeviceProgram(UsdGenCudaBackendId(), type));
}

ProgramDef const *ProgramFor(UsdGenGraphSpec const &spec) noexcept
{
    UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
    UsdGenTypeId const fallback = types.DefaultPrimitive();
    UsdGenTypeId type = fallback;
    for (UsdGenOpNode const &node : spec.nodes) {
        if (node.key != spec.terminal) continue;
        if (node.opId != UsdGenOpId::Invalid) {
            UsdGenToken const &name = UsdGenOpRegistry::Get().Record(node.opId).OutputType();
            if (!name.IsEmpty()) type = types.Find(name);
        }
        break;
    }
    ProgramDef const *program = ProgramOf(type);
    return program || type == fallback ? program : ProgramOf(fallback);
}

}  // namespace usdGen::cuda
