// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// One device's execution workspace of a CUDA provider: the device, the
// context-loss mark and the state of each program that cooks on it. A
// provider replaces it when the cook's device or graph description changes;
// program state lives and dies with it.
#pragma once
#include "usdGenCuda/export.h"
#include "usdGenCuda/compile/program.h"

#include <cstddef>
#include <memory>

namespace usdGen {
struct UsdGenDiagnostics;
}  // namespace usdGen

namespace usdGen::cuda {

class USDGEN_CUDA_API Workspace {
public:
    /// device -1: the calling thread's current device. Null with a
    /// diagnostic when it cannot be captured.
    static std::unique_ptr<Workspace> Create(int device, UsdGenDiagnostics *);
    ~Workspace();
    Workspace(Workspace const &) = delete;
    Workspace &operator=(Workspace const &) = delete;

    int DeviceIndex() const noexcept { return _device; }
    /// Context lost, or a program state left unprovable.
    bool IsPoisoned() const noexcept;
    /// An externally reported context loss: later cooks fail closed.
    void MarkContextLost() noexcept;
    /// The state of 'program', made on first use. Null with a diagnostic
    /// when it cannot be made or every slot is taken.
    ProgramState *State(ProgramDef const &program, UsdGenDiagnostics *);

private:
    explicit Workspace(int device) noexcept : _device(device) {}
    static constexpr size_t kPrograms = 4;
    struct Slot {
        ProgramDef const *program = nullptr;
        std::unique_ptr<ProgramState> state;
    };
    Slot _slots[kPrograms];
    int _device = -1;
    bool _lost = false;
};

}  // namespace usdGen::cuda
