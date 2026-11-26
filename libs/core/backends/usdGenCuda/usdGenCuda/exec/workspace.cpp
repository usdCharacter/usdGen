// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCuda/exec/workspace.h"
#include "usdGen/registries/op.h"

#include <cuda_runtime_api.h>

#include <string>
#include <utility>

namespace usdGen::cuda {
namespace {

void Fail(UsdGenDiagnostics *diagnostics, std::string message)
{
    if (diagnostics) diagnostics->Error("CUDA: " + std::move(message));
}

std::string StatusText(cudaError_t status)
{
    return std::string(cudaGetErrorName(status)) + " (" +
        std::to_string(static_cast<int>(status)) + ": " + cudaGetErrorString(status) + ")";
}

}  // namespace

std::unique_ptr<Workspace> Workspace::Create(int device, UsdGenDiagnostics *diagnostics)
{
    if (device < -1) {
        Fail(diagnostics, "cannot capture CUDA workspace device: invalid device " +
                          std::to_string(device));
        return {};
    }
    if (device == -1) {
        cudaError_t const status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
            Fail(diagnostics, "cannot capture CUDA workspace device: cudaGetDevice returned " +
                              StatusText(status));
            return {};
        }
    }
    return std::unique_ptr<Workspace>(new Workspace(device));
}

Workspace::~Workspace()
{
    // Later slots may borrow earlier ones; release in reverse.
    for (size_t i = kPrograms; i-- > 0;) _slots[i].state.reset();
}

bool Workspace::IsPoisoned() const noexcept
{
    if (_lost) return true;
    for (Slot const &slot : _slots)
        if (slot.state && slot.state->Poisoned()) return true;
    return false;
}

void Workspace::MarkContextLost() noexcept
{
    _lost = true;
    for (Slot &slot : _slots)
        if (slot.state) slot.state->MarkContextLost();
}

ProgramState *Workspace::State(ProgramDef const &program, UsdGenDiagnostics *diagnostics)
{
    for (Slot &slot : _slots)
        if (slot.program == &program) return slot.state.get();
    if (!program.createState) {
        Fail(diagnostics, "device program has no workspace state");
        return nullptr;
    }
    for (Slot &slot : _slots) {
        if (slot.program) continue;
        std::unique_ptr<ProgramState> state = program.createState(*this, diagnostics);
        if (!state) return nullptr;
        if (_lost) state->MarkContextLost();
        slot.program = &program;
        slot.state = std::move(state);
        return slot.state.get();
    }
    Fail(diagnostics, "too many device programs on one workspace");
    return nullptr;
}

}  // namespace usdGen::cuda
