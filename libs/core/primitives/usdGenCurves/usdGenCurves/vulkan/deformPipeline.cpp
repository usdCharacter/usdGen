// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/deformPipeline.h"
#include "usdGenCurves/vulkan/deformEvaluateCpu.h"
#include "usdGenVulkan/vulkan/queueTimeline.h"
#include "usdGen/epoch.h"
#include "usdGen/scratch.h"
#include "usdGen/taskScheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace usdGen::vulkan {

char const* DeformSemanticName(DeformSemantic s) noexcept {
    switch (s) {
        case DeformSemantic::Ok:            return "Ok";
        case DeformSemantic::BadOffsets:    return "BadOffsets";
        case DeformSemantic::NonFinite:     return "NonFinite";
        case DeformSemantic::BadValue:      return "BadValue";
        case DeformSemantic::RankDeficient: return "RankDeficient";
        case DeformSemantic::SolverError:   return "SolverError";
    }
    return "Invalid";
}

namespace {

// Workgroup widths, each matched to its shader's local_size_x and each at
// its measured best (spin-observed GPU times, stable to +-2us): the fp64
// eval loop runs ~80us faster per 1M-CV pose at 128-wide groups (13.25
// -> 13.17ms; 64 and 32 are no faster), and the memory-bound apply is
// ~4us faster at 256 (0.167 -> 0.163ms), so each dispatch keeps its own
// width.
constexpr uint32_t kEvalLocalSize = 128;
constexpr uint32_t kApplyLocalSize = 256;
constexpr uint32_t kEvalUboBytes = 64;
constexpr uint32_t kApplyUboBytes = 80;
// Eval set: bindings 0,1,2,3,5 = 5 storage + binding 4 = 1 uniform.
constexpr uint32_t kEvalStorage = 5;
// Apply set: bindings 0..8 = 9 storage + binding 9 = 1 uniform.
constexpr uint32_t kApplyStorage = 9;
// Cached-R evaluate: the first P CVs run through the cached
// radius-cubed rows (P*n doubles); points past the prefix run the direct
// shader with a base offset. P is sized from the pool budget at funding
// time (AffordableCachePrefix), not fixed: the 512MiB bench budget funds
// ~613k CVs at n=100 (~468MiB of R plus a ~7MiB proof next to the ~36MiB
// of pose scratch and inputs). Past the hetero balance point funding
// more prefix wastes the host (HeteroBalanceCap holds the active prefix
// while the funding stays full), so each CV moved onto the streaming
// prefix (~3.2ns/CV) off the fp64-sqrt suffix (~13.1ns/CV) saves ~9.9ns
// only up to the balance. Budgets below the floor run the direct shader
// with no added work (pure arithmetic: no vkAllocateMemory probes on the
// miss path).
constexpr uint32_t kEvalCacheMinPrefixCVs = 4096;
constexpr uint32_t kEvalCacheMaxPrefixCVs = 715827882; // INT32_MAX/3: 3*active fits the int32 verify word count
constexpr size_t kEvalCacheFundingMargin = size_t{1} << 20;
constexpr uint64_t kEvalCacheFundingAlign = 65536;
constexpr uint32_t kVerifyLocalSize = 256;
constexpr uint32_t kFillLocalSize = 256;
constexpr uint32_t kCachedLocalSize = 128;
constexpr uint32_t kVerifyUboBytes = 16;
constexpr uint32_t kFillUboBytes = 48;
constexpr uint32_t kCachedUboBytes = 64;
constexpr uint32_t kFillArgsBytes = 12;
// Verify/fill set: bindings 0,1,2 = 3 storage + binding 3 = 1 uniform.
// The cached set reuses the eval layout (binding 2 carries R instead of
// the samples; same types), so it needs no new layout.
constexpr uint32_t kVerifyFillStorage = 3;
// Miss backoff: two consecutive refills mean the CVs (or rest) churn
// every pose, so the next poses run direct until one probe re-arms the
// cache. Churning inputs then cost ~+1% instead of +15%.
constexpr int kEvalCacheBypassMisses = 2;
constexpr int kEvalCacheBypassPoses = 32;

std::atomic<bool> disableEvalCache{false};
std::atomic<uint64_t> evalCacheHits{0}, evalCacheMisses{0};
std::atomic<uint64_t> evalCacheBypassed{0}, evalCacheUnfunded{0};
std::atomic<uint32_t> evalCacheFundedPrefix{0};
// Heterogeneous suffix: the past-prefix suffix evaluates on the host
// (deformEvaluateCpu.h, bitwise the direct shader) while the proof submit
// runs the cached prefix on the device. 10 workers fill this host's fast
// cores (8: 3.3ms, 10: 2.9ms); 2048-CV grains keep the pool's claimed
// chunks coarse. The fixed overhead covers the extra copy submit + fence
// waits + pool dispatch. The seed throughputs are the measured GB10
// figures (see the prefix comment above); the policy replaces them with
// per-GPU measurements after the first poses. The cached seed prices the
// burdened prefix proof (submit + verify + cached eval + waits, median
// 4.6-4.8ns/CV at 613k CVs over three quiet runs), not the 3.2ns marginal
// the balance cap divided by before: the marginal seed froze the sticky
// split ~15% past the leg crossing and left the pose proof-bound by half
// a millisecond. The fixed seed is the median blocking copy wait (~0.2ms)
// plus map/dispatch. Live EMAs converge to the same crossing (fresh caps
// 531-542k), so seeds and policy agree; the direct seed is untouched.
constexpr int kHeteroWorkers = 10;
constexpr uint32_t kHeteroGrainsize = 2048;
constexpr uint32_t kHeteroCalibCvs = 8192;
constexpr double kHeteroFixedOverheadNs = 200000.0;
constexpr double kHeteroSeedGpuDirectNsPerCv = 13.1;
constexpr double kHeteroSeedCachedNsPerCv = 4.7;
constexpr double kHeteroEmaAlpha = 0.3;
constexpr int kHeteroProbeBase = 64;
constexpr int kHeteroProbeMax = 1024;
// Balance-cap margin: the capped split must beat the full prefix by this
// factor before the cap engages (mirrors the first-probe margin below).
constexpr double kHeteroBalanceMargin = 0.9;
std::atomic<int> forceHeteroSuffix{0};
std::atomic<uint64_t> heteroRuns{0};

uint32_t Groups(uint32_t n, uint32_t size) { return (n + size - 1) / size; }

// The synchronous Begin's proof wait bound, matching the retired fence
// timeout: a proof that has not signaled by then fails the pose.
constexpr uint64_t kProofTimeoutNs = 10000000000ull;

// Scratch pooling: a pose allocates 8 buffers and frees them at candidate
// destruction, and vkAllocateMemory/vkFreeMemory cost ~70/80us (median) each
// on the qualified driver, so ~1.2ms of every pose is allocation lifecycle.
// All eight are reusable across poses: the proof/eval status words, the
// eval/apply UBOs, the coefficient/sample uploads, the warped staging
// buffer, and outPoints are all GPU-idle once the candidate proves (the
// apply fence signals after the last read) and none is read on the host
// afterwards, except outPoints through a live Output.
// outPoints rejoins its own pool at State destruction rather than at Poll:
// it is exposed through Candidate::output() and the caller may read it
// until the Candidate dies, so Poll-time return would hand a live buffer
// to the next pose. The coefficient/sample buffers
// are allocated at their max (sampleCount is rejected above 100) so a pooled
// set fits any pose; the warped buffer must match the pose's point count
// exactly. Only proven-idle sets rejoin: a set from a
// failed or unproven submission stays with its owner for quarantine or
// destruction, never pooled.
constexpr int kMaxRbfSamples = 100;
constexpr VkDeviceSize kMaxCoefBytes = VkDeviceSize(kMaxRbfSamples + 4) * 24u;
constexpr VkDeviceSize kMaxSamplesBytes = VkDeviceSize(kMaxRbfSamples) * 24u;
constexpr size_t kMaxIdleScratchSets = 2;
constexpr size_t kMaxIdleOutPoints = 2;

// Domain codes (deformApply.comp constants).
enum : uint32_t {
    kDomainGroom = 1u,
    kDomainPrimitive = 2u,
    kDomainPoint = 4u,
};

struct EvalUbo {
    int32_t n_m[2];
    int32_t counts[2];
    double cx, cy, cz, invScale, scale, _pad2;
};
static_assert(sizeof(EvalUbo) == 64);

// Cached-R UBOs (see the .comp ABI notes). VerifyUbo: words, n.
// FillUbo: n, active prefix CVs, P, then the center/invScale the fill
// consumes. CachedUbo: n, m, active count, P, then the eval tail.
struct VerifyUbo {
    int32_t words_n[4];
};
static_assert(sizeof(VerifyUbo) == 16);
struct FillUbo {
    int32_t shape[4];
    double cx, cy, cz, invScale;
};
static_assert(sizeof(FillUbo) == 48);
static_assert(offsetof(FillUbo, cx) == 16);
struct CachedUbo {
    int32_t nmcs[4];
    double cx, cy, cz, invScale, scale, _pad;
};
static_assert(sizeof(CachedUbo) == 64);
static_assert(offsetof(CachedUbo, cx) == 16);

struct ApplyUbo {
    uint32_t counts[3];       // uvec3: 12B in std140, offset 0
    float groomEnvelope;      // offset 12
    float maskF[4];          // vec4: 16B in std140, offset 16
    float enabledF[4];       // vec4: offset 32
    float lockF[4];          // vec4: offset 48
    uint32_t fieldCounts[3]; // uvec3: 12B in std140, offset 64
    uint32_t strideCandidate; // offset 76; total 80, 16B-aligned block
};
static_assert(sizeof(ApplyUbo) == 80);
static_assert(offsetof(ApplyUbo, groomEnvelope) == 12);
static_assert(offsetof(ApplyUbo, maskF) == 16);
static_assert(offsetof(ApplyUbo, enabledF) == 32);
static_assert(offsetof(ApplyUbo, lockF) == 48);
static_assert(offsetof(ApplyUbo, fieldCounts) == 64);

} // namespace

void TestDisableDeformEvalCache(bool disable) noexcept {
    disableEvalCache.store(disable, std::memory_order_release);
}
uint64_t DeformEvalCacheHitsForTesting() noexcept {
    return evalCacheHits.load(std::memory_order_acquire);
}
uint64_t DeformEvalCacheMissesForTesting() noexcept {
    return evalCacheMisses.load(std::memory_order_acquire);
}
uint64_t DeformEvalCacheBypassedForTesting() noexcept {
    return evalCacheBypassed.load(std::memory_order_acquire);
}
uint64_t DeformEvalCacheUnfundedForTesting() noexcept {
    return evalCacheUnfunded.load(std::memory_order_acquire);
}
uint32_t DeformEvalCacheFundedPrefixForTesting() noexcept {
    return evalCacheFundedPrefix.load(std::memory_order_acquire);
}
void TestForceDeformHeteroSuffix(int force) noexcept {
    forceHeteroSuffix.store(force, std::memory_order_relaxed);
}
uint64_t DeformHeteroSuffixRunsForTesting() noexcept {
    return heteroRuns.load(std::memory_order_acquire);
}

struct DeformPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule evalShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout evalLayout = VK_NULL_HANDLE;
    VkPipelineLayout evalPipelineLayout = VK_NULL_HANDLE;
    VkPipeline evalPipeline = VK_NULL_HANDLE;
    VkShaderModule applyShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout applyLayout = VK_NULL_HANDLE;
    VkPipelineLayout applyPipelineLayout = VK_NULL_HANDLE;
    VkPipeline applyPipeline = VK_NULL_HANDLE;
    // Cached-R evaluate (null without cache SPIR-V). The verify and fill
    // shaders share one set layout (3 storage + 1 uniform at the same
    // bindings); the cached shader reuses the eval set layout and the
    // eval pipeline layout (binding 2 carries R instead of the samples).
    VkShaderModule verifyShader = VK_NULL_HANDLE;
    VkShaderModule fillShader = VK_NULL_HANDLE;
    VkShaderModule cachedShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout verifyFillLayout = VK_NULL_HANDLE;
    VkPipelineLayout verifyFillPipelineLayout = VK_NULL_HANDLE;
    VkPipeline verifyPipeline = VK_NULL_HANDLE;
    VkPipeline fillPipeline = VK_NULL_HANDLE;
    VkPipeline cachedPipeline = VK_NULL_HANDLE;
    // The rest-only factorization survives poses that keep the same rest
    // samples and smoothing; a pose then pays only the right-hand side and
    // the triangular solves. Begin may run on several threads: the entry
    // is immutable and published by atomic pointer; a miss builds a fresh
    // one and publishes it (concurrent misses each build the same bytes).
    struct FactorEntry final : UsdGenRefCounted {
        std::vector<float> rest;
        double smoothing = 0.0;
        int n = 0;
        RbfFactorization factor;
    };
    UsdGenAtomicRef<FactorEntry> factorCache;
    // Pose-invariant R cache. R[j*P+i] is the radius-cubed kernel value
    // for CV i and sample j: it depends only on the CVs, the rest
    // samples, and the rest-derived center/scale, so poses that keep
    // them evaluate the cached prefix through a streaming FMA pass and
    // only the past-prefix suffix through the sqrt loop. The verify
    // shader checks the prefix CV bytes on the device (and snapshots
    // them into `proof`), self-scheduling the fill through the indirect
    // args; the rest and n are host BeginInfo, compared by the owner of
    // the proof state (a mismatch forces the refill instead of
    // speculating). The cache lives in ProofState below.
    struct EvalCache {
        std::shared_ptr<ChargedBuffer> r;     // R rows, rowsFunded*rowsAlloc doubles
        std::shared_ptr<ChargedBuffer> proof; // prefix CV snapshot, rowsFunded float3s
        std::vector<float> rest;              // rest proof (host)
        int n = 0;                            // n at the last fill
        int rowsAlloc = 0;                    // R rows allocated
        uint32_t rowsFunded = 0;              // R/proof capacity (CVs), sized from the budget
        uint32_t filled = 0;                  // valid R/proof prefix (CVs)
        uint32_t capActive = 0;               // sticky balance cap (CVs), 0 = none
        uint32_t capPoints = 0;               // point count the sticky cap was computed for
        int missStreak = 0;                   // consecutive refills
        int bypassLeft = 0;                   // direct poses left in backoff
        bool everFilled = false;
    };
    // Heterogeneous suffix execution: the adaptive policy for running the
    // past-prefix suffix on the host. The device copies the suffix CVs
    // into the idle warped suffix (byte staging, no extra buffer) while the
    // host evaluates the suffix (bitwise the direct shader) during the
    // prefix proof; host-visible coherent inputs map directly instead (no
    // staging copy). The suffix runs on the process task scheduler
    // (UsdGenParallelFor), which exists before any backend registers, so
    // there is no pipeline-owned pool to create or keep warm. The policy
    // lives in ProofState below, so only its owner updates it.
    struct HeteroPolicy {
        double cpuNsPerCv = 0.0;   // host suffix throughput (EMA), 0 = uncalibrated
        double gpuNsPerCv = 0.0;   // device suffix throughput, 0 = seed
        double prefixProofNs = 0.0;// prefix-only proof fence wait (EMA)
        uint32_t prefixProofCvs = 0; // active CVs that proof measured
        bool armed = false;        // hetero engaged
        bool everProbed = false;   // a hetero pose has run (estimates are live)
        int probeIn = 0;           // eligible GPU poses until the next probe
        int probeInterval = kHeteroProbeBase;
    };
    // Everything the cached proof phase reads and writes. Begin may run on
    // several threads: a pose takes the state by atomic exchange for its
    // decision and proof phase and puts it back after; a pose that finds
    // it taken runs direct (bitwise the same) instead of waiting.
    struct ProofState {
        EvalCache evalCache;
        HeteroPolicy hetero;
    };
    std::atomic<ProofState*> proofState{new ProofState};
    // Owns the proof state from Take until Put (or scope exit).
    class ProofHold {
    public:
        explicit ProofHold(Native* native) noexcept : native_(native) {}
        ~ProofHold() { Put(); }
        ProofHold(ProofHold const&) = delete;
        ProofHold& operator=(ProofHold const&) = delete;
        ProofState* Take() noexcept {
            if (!state_) state_ = native_->proofState.exchange(nullptr, std::memory_order_acq_rel);
            return state_;
        }
        void Put() noexcept {
            if (state_) native_->proofState.store(state_, std::memory_order_release);
            state_ = nullptr;
        }
        // Hands the held state to a staged proof (Candidate::State::proof)
        // without putting it back; Adopt takes it again.
        ProofState* Release() noexcept { ProofState* p = state_; state_ = nullptr; return p; }
        void Adopt(ProofState* p) noexcept { Put(); state_ = p; }
        ProofState* operator->() const noexcept { return state_; }
        explicit operator bool() const noexcept { return state_ != nullptr; }
    private:
        Native* native_;
        ProofState* state_ = nullptr;
    };
    // The hetero arming decision (see the definition below). Caller owns
    // the proof state.
    static bool HeteroWants(HeteroPolicy& hp, uint32_t suffixCvs,
                            uint32_t prefixCvs);
    // The hetero balance cap (see the definition below): the active prefix
    // past which the host idles while the device runs, or UINT32_MAX for
    // no cap. Caller owns the proof state.
    static uint32_t HeteroBalanceCap(HeteroPolicy& hp, uint32_t points);
    // Idle per-pose scratch sets (everything but outPoints): allocation
    // lifecycle costs ~150us per buffer per pose, so proven-idle sets are
    // checked out by the next Begin instead of allocating. Guarded: Begin
    // may run on several threads. At most kMaxIdleScratchSets are kept;
    // surplus sets are destroyed on release. Members destroy after the body
    // with context (declared first) still alive, so no explicit clear.
    // Lock-free: each pool slot has a state word; checkout and return claim
    // a slot by CAS and skip a busy one, so nobody ever waits.
    struct ScratchSet {
        std::shared_ptr<ChargedBuffer> warped;
        // The warped memory type the set was allocated with: hetero poses
        // need host-visible (the host reads the staged CVs and writes the
        // suffix), GPU poses prefer device-local. Matched at checkout.
        bool warpedHostVisible = false;
        std::shared_ptr<ChargedBuffer> status;
        std::shared_ptr<ChargedBuffer> evalStatus;
        std::shared_ptr<ChargedBuffer> coefBuf;
        std::shared_ptr<ChargedBuffer> samplesBuf;
        std::shared_ptr<ChargedBuffer> evalUbo;
        std::shared_ptr<ChargedBuffer> applyUbo;
        // Cached-R per-pose scratch (null on direct-only pipelines): the
        // three cache UBOs plus the per-pose indirect fill args. The args
        // are per-pose (not cache-owned) so concurrent poses never share
        // a pre-write.
        std::shared_ptr<ChargedBuffer> verifyUbo;
        std::shared_ptr<ChargedBuffer> fillUbo;
        std::shared_ptr<ChargedBuffer> cachedUbo;
        std::shared_ptr<ChargedBuffer> fillArgs;
    };
    enum : uint32_t { kSlotEmpty = 0, kSlotFull = 1, kSlotBusy = 2 };
    template <class T>
    struct IdleSlot {
        std::atomic<uint32_t> state{kSlotEmpty};
        T value;
        bool Claim(uint32_t from) noexcept {
            return state.compare_exchange_strong(from, kSlotBusy, std::memory_order_acquire,
                                                 std::memory_order_relaxed);
        }
        void Release(uint32_t to) noexcept { state.store(to, std::memory_order_release); }
    };
    // Parks a value in an empty slot; false when every slot is taken.
    template <class T, size_t N>
    static bool ParkIdle(IdleSlot<T> (&slots)[N], T&& value) noexcept {
        for (auto& slot : slots) {
            if (!slot.Claim(kSlotEmpty)) continue;
            slot.value = std::move(value);
            slot.Release(kSlotFull);
            return true;
        }
        return false;
    }
    IdleSlot<ScratchSet> scratchIdle[kMaxIdleScratchSets];
    // Idle outPoints buffers, exact-size matched at checkout: allocation
    // lifecycle costs ~150us per pose, and the buffer is provably idle at
    // State destruction (see ReturnOutPoints). The caller may hold
    // Output::points past the Candidate's death, so a pooled buffer is
    // reused only while the pool is its sole owner; anything still shared
    // stays idle and the pose allocates fresh. Capped.
    IdleSlot<std::shared_ptr<ChargedBuffer>> outPointsIdle[kMaxIdleOutPoints];
    ~Native() {
        delete proofState.load(std::memory_order_acquire);
        if (!context) return;
        auto d = context->device();
        if (verifyPipeline) vkDestroyPipeline(d, verifyPipeline, nullptr);
        if (fillPipeline) vkDestroyPipeline(d, fillPipeline, nullptr);
        if (cachedPipeline) vkDestroyPipeline(d, cachedPipeline, nullptr);
        if (verifyFillPipelineLayout)
            vkDestroyPipelineLayout(d, verifyFillPipelineLayout, nullptr);
        if (verifyFillLayout) vkDestroyDescriptorSetLayout(d, verifyFillLayout, nullptr);
        if (verifyShader) vkDestroyShaderModule(d, verifyShader, nullptr);
        if (fillShader) vkDestroyShaderModule(d, fillShader, nullptr);
        if (cachedShader) vkDestroyShaderModule(d, cachedShader, nullptr);
        if (evalPipeline) vkDestroyPipeline(d, evalPipeline, nullptr);
        if (evalPipelineLayout) vkDestroyPipelineLayout(d, evalPipelineLayout, nullptr);
        if (evalLayout) vkDestroyDescriptorSetLayout(d, evalLayout, nullptr);
        if (evalShader) vkDestroyShaderModule(d, evalShader, nullptr);
        if (applyPipeline) vkDestroyPipeline(d, applyPipeline, nullptr);
        if (applyPipelineLayout) vkDestroyPipelineLayout(d, applyPipelineLayout, nullptr);
        if (applyLayout) vkDestroyDescriptorSetLayout(d, applyLayout, nullptr);
        if (applyShader) vkDestroyShaderModule(d, applyShader, nullptr);
    }
};

struct DeformPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    BeginInfo inputs;
    std::shared_ptr<ChargedBuffer> outPoints;
    std::shared_ptr<ChargedBuffer> warped;
    std::shared_ptr<ChargedBuffer> status;
    std::shared_ptr<ChargedBuffer> evalStatus;
    std::shared_ptr<ChargedBuffer> coefBuf;
    std::shared_ptr<ChargedBuffer> samplesBuf;
    std::shared_ptr<ChargedBuffer> evalUbo;
    std::shared_ptr<ChargedBuffer> applyUbo;
    std::shared_ptr<ChargedBuffer> verifyUbo;
    std::shared_ptr<ChargedBuffer> fillUbo;
    std::shared_ptr<ChargedBuffer> cachedUbo;
    std::shared_ptr<ChargedBuffer> fillArgs;
    // The warped memory type (see ScratchSet): recorded at checkout-time
    // so the returned set re-pools under the same tag.
    bool warpedHostVisible = false;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    // Proof: the evaluate proof is in flight (Advance/Poll move it on);
    // Apply: the apply is in flight (applyValue); Done: proved (output or
    // a rejection semantic); Failed: failure holds the error.
    enum class Stage : uint8_t { Proof, Apply, Done, Failed };
    Stage stage = Stage::Done;
    VkResult failure = VK_SUCCESS;
    uint64_t applyValue = 0;  // the apply submit's timeline value
    // Everything the proof stage needs after the proof submit returns.
    struct ProofStage {
        Native::ProofState* owned = nullptr;  // held from Launch until observed
        BeforeSubmit beforeSubmit;
        VkCommandBuffer applyCmd = VK_NULL_HANDLE;
        uint64_t copyValue = 0, proofValue = 0;
        std::chrono::steady_clock::time_point submitted;
        double samplesNorm[3 * kMaxRbfSamples] = {};
        double coef[3 * (kMaxRbfSamples + 4)] = {};
        double center[3] = {};
        double invScale = 1.0, scale = 1.0;
        int n = 0, m = 0;
        uint32_t cacheActive = 0;
        VkDeviceSize pointsBytes = 0, outBytes = 0;
        bool useCache = false, forceFill = false, heteroEligible = false;
        bool useHetero = false, stageCopy = false, suffixDone = false;
        uint32_t heteroFlag = 0;
        double heteroCpuNs = 0.0;
    } proof;
    // The timeline value the proof stage is blocked on.
    uint64_t ProofWaitValue() const noexcept {
        return proof.useHetero && proof.stageCopy && !proof.suffixDone ? proof.copyValue
                                                                        : proof.proofValue;
    }
    // Returns a proof state the stage still holds (failure paths).
    void PutProof() noexcept {
        if (proof.owned) native->proofState.store(proof.owned, std::memory_order_release);
        proof.owned = nullptr;
    }
    bool pending = false, lost = false, proved = false;
    uint32_t semantic = UINT32_MAX;
    uint32_t pointCount = 0;
    // Exactly-once guard for the scratch release below: Poll returns the
    // set early (so the next Begin reuses it while this candidate is still
    // alive), and the destructor returns it when Poll never ran. Atomic:
    // two threads may Poll the same candidate concurrently.
    std::atomic<bool> scratchReturned{false};
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    // Moves the scratch set back to the pool when it is complete and
    // pool-sized; anything else (the empty-topology early-out, a failure
    // path's partial set) is destroyed with the state. The caller proves
    // GPU idleness: Poll after the apply fence signaled, or the destructor
    // when nothing is pending. Never throws.
    void ReturnScratch() noexcept {
        Native::ScratchSet set;
        set.warped = std::move(warped);
        set.warpedHostVisible = warpedHostVisible;
        set.status = std::move(status);
        set.evalStatus = std::move(evalStatus);
        set.coefBuf = std::move(coefBuf);
        set.samplesBuf = std::move(samplesBuf);
        set.evalUbo = std::move(evalUbo);
        set.applyUbo = std::move(applyUbo);
        set.verifyUbo = std::move(verifyUbo);
        set.fillUbo = std::move(fillUbo);
        set.cachedUbo = std::move(cachedUbo);
        set.fillArgs = std::move(fillArgs);
        if (!set.warped || !set.status || !set.evalStatus || !set.coefBuf ||
            !set.samplesBuf || !set.evalUbo || !set.applyUbo)
            return;
        // Direct-only pipelines never allocate the cache scratch; cached
        // pipelines pool only complete sets.
        bool const wantCache = native->cachedPipeline != VK_NULL_HANDLE;
        if (wantCache && (!set.verifyUbo || !set.fillUbo || !set.cachedUbo || !set.fillArgs))
            return;
        // Pool-sized: the fixed buffers fit every pose at or above their
        // nominal size (descriptors range the actual bytes); the warped
        // buffer is matched exactly at checkout.
        if (set.status->sizeBytes() < 4 || set.evalStatus->sizeBytes() < 4 ||
            set.evalUbo->sizeBytes() < kEvalUboBytes ||
            set.applyUbo->sizeBytes() < kApplyUboBytes ||
            set.coefBuf->sizeBytes() < kMaxCoefBytes ||
            set.samplesBuf->sizeBytes() < kMaxSamplesBytes)
            return;
        if (wantCache &&
            (set.verifyUbo->sizeBytes() < kVerifyUboBytes ||
             set.fillUbo->sizeBytes() < kFillUboBytes ||
             set.cachedUbo->sizeBytes() < kCachedUboBytes ||
             set.fillArgs->sizeBytes() < kFillArgsBytes))
            return;
        // A full pool destroys the surplus set here.
        (void)Native::ParkIdle(native->scratchIdle, std::move(set));
    }
    // Returns outPoints to the idle pool. Only the destructor calls this:
    // between Poll and destruction the caller may still read output(), so
    // Poll-time return would hand a live buffer to the next pose. The
    // caller requires !pending && !lost (GPU-idle, like ReturnScratch:
    // the apply fence proved the write complete, or nothing submitted);
    // Outputs the caller kept past destruction stay shared, and checkout
    // only reuses sole-owned buffers. Never throws.
    void ReturnOutPoints() noexcept {
        if (!outPoints) return;
        (void)Native::ParkIdle(native->outPointsIdle, std::move(outPoints));
    }
    ~State() {
        auto d = native->context->device();
        // A candidate destroyed without Poll still owns a GPU-idle set when
        // nothing is pending (rejected before submit, or the proof wait
        // passed and the apply never submitted): rejoin it, else a pending
        // or lost set stays here for quarantine/destruction, never pooled.
        if (!scratchReturned.exchange(true, std::memory_order_acq_rel) &&
            !pending && !lost)
            ReturnScratch();
        // outPoints rejoins its pool at destruction, not at Poll (Output
        // readers may outlive the proof). Exactly once: this runs once.
        if (!pending && !lost)
            ReturnOutPoints();
        // Only reached with nothing pending (else the Candidate quarantines
        // instead), so a proof state still held here is idle.
        PutProof();
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

DeformPipeline::DeformPipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
DeformPipeline::~DeformPipeline() = default;
std::shared_ptr<DeviceContext> const& DeformPipeline::context() const noexcept {
    return native_->context;
}

// ============================ Create ============================

std::shared_ptr<DeformPipeline> DeformPipeline::Create(
    std::shared_ptr<DeviceContext> context,
    std::vector<uint32_t> const& evaluateSpirv,
    std::vector<uint32_t> const& applySpirv,
    VkResult* result,
    DeformEvalCacheSpirv const& cacheSpirv) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto valid = [](std::vector<uint32_t> const& c) {
        return c.size() >= 5 && c.front() == 0x07230203u;
    };
    if (!context || !valid(evaluateSpirv) || !valid(applySpirv)) return {};
    bool const wantCache = !cacheSpirv.verify.empty() || !cacheSpirv.fill.empty() ||
        !cacheSpirv.cached.empty();
    // All three cache shaders or none: a partial set is a wiring bug,
    // failed loudly rather than silently direct.
    if (wantCache && (!valid(cacheSpirv.verify) || !valid(cacheSpirv.fill) ||
                      !valid(cacheSpirv.cached)))
        return {};
    if (!context->shaderFloat64Enabled() || !context->computeTimeline()) return {};
    // The hetero suffix runs on the process task scheduler: bring it up with
    // the pipeline (backend registration), not inside the first pose.
    (void)UsdGenTaskScheduler::Process();
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < kApplyLocalSize ||
        physical.limits.maxComputeWorkGroupSize[0] < kApplyLocalSize ||
        physical.limits.maxPerStageDescriptorStorageBuffers < kApplyStorage ||
        physical.limits.maxDescriptorSetStorageBuffers < kApplyStorage) return {};
    try {
        auto n = std::make_shared<Native>();
        n->context = std::move(context);
        auto d = n->context->device();
        auto mkModule = [&](std::vector<uint32_t> const& code, VkShaderModule* out) {
            VkShaderModuleCreateInfo sm{};
            sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            sm.codeSize = code.size() * sizeof(uint32_t);
            sm.pCode = code.data();
            return vkCreateShaderModule(d, &sm, nullptr, out);
        };
        VkResult r = mkModule(evaluateSpirv, &n->evalShader);
        if (r == VK_SUCCESS) r = mkModule(applySpirv, &n->applyShader);
        if (wantCache && r == VK_SUCCESS) r = mkModule(cacheSpirv.verify, &n->verifyShader);
        if (wantCache && r == VK_SUCCESS) r = mkModule(cacheSpirv.fill, &n->fillShader);
        if (wantCache && r == VK_SUCCESS) r = mkModule(cacheSpirv.cached, &n->cachedShader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        {
            // Eval layout: 5 storage (0,1,2,3,5) + 1 uniform (4).
            uint32_t const total = kEvalStorage + 1;
            VkDescriptorSetLayoutBinding bindings[total]{};
            uint32_t const storageBindings[] = {0, 1, 2, 3, 5};
            for (uint32_t i = 0; i < total; ++i) {
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                if (i == 4) {
                    bindings[i].binding = 4;
                    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                } else {
                    bindings[i].binding = storageBindings[i < 4 ? i : i - 1];
                    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                }
            }
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount = total; ds.pBindings = bindings;
            r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->evalLayout);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        {
            // Apply layout: 9 storage (0..8) + 1 uniform (9).
            uint32_t const total = kApplyStorage + 1;
            VkDescriptorSetLayoutBinding bindings[total]{};
            for (uint32_t i = 0; i < total; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                if (i == kApplyStorage)
                    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                else
                    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            }
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount = total; ds.pBindings = bindings;
            r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->applyLayout);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        if (wantCache) {
            // Verify/fill layout: 3 storage (0,1,2) + 1 uniform (3),
            // shared by both shaders.
            uint32_t const total = kVerifyFillStorage + 1;
            VkDescriptorSetLayoutBinding bindings[total]{};
            for (uint32_t i = 0; i < total; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                bindings[i].descriptorType =
                    (i == kVerifyFillStorage) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                              : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            }
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount = total; ds.pBindings = bindings;
            r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->verifyFillLayout);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        {
            VkPipelineLayoutCreateInfo pl{};
            pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl.setLayoutCount = 1; pl.pSetLayouts = &n->evalLayout;
            r = vkCreatePipelineLayout(d, &pl, nullptr, &n->evalPipelineLayout);
            if (r == VK_SUCCESS) {
                pl.pSetLayouts = &n->applyLayout;
                r = vkCreatePipelineLayout(d, &pl, nullptr, &n->applyPipelineLayout);
            }
            if (wantCache && r == VK_SUCCESS) {
                pl.pSetLayouts = &n->verifyFillLayout;
                r = vkCreatePipelineLayout(d, &pl, nullptr, &n->verifyFillPipelineLayout);
            }
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        auto mkPipeline = [&](VkShaderModule module, VkPipelineLayout layout, VkPipeline* out) {
            VkComputePipelineCreateInfo cp{};
            cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp.layout = layout;
            cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cp.stage.module = module; cp.stage.pName = "main";
            return vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, out);
        };
        r = mkPipeline(n->evalShader, n->evalPipelineLayout, &n->evalPipeline);
        if (r == VK_SUCCESS)
            r = mkPipeline(n->applyShader, n->applyPipelineLayout, &n->applyPipeline);
        // The cached shader reuses the eval pipeline layout (same
        // binding types at the same numbers).
        if (wantCache && r == VK_SUCCESS)
            r = mkPipeline(n->verifyShader, n->verifyFillPipelineLayout, &n->verifyPipeline);
        if (wantCache && r == VK_SUCCESS)
            r = mkPipeline(n->fillShader, n->verifyFillPipelineLayout, &n->fillPipeline);
        if (wantCache && r == VK_SUCCESS)
            r = mkPipeline(n->cachedShader, n->evalPipelineLayout, &n->cachedPipeline);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto pipeline = std::shared_ptr<DeformPipeline>(new DeformPipeline(std::move(n)));
        finish(VK_SUCCESS);
        return pipeline;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

// ============================ Candidate ============================

DeformPipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
DeformPipeline::Candidate::~Candidate() {
    auto& s = *state_;
    if (s.stage == State::Stage::Proof && s.pending && !s.lost) {
        // Dropped mid-proof: the proof still reads the cache buffers the
        // proof state owns. Destruction may block (never the cook path):
        // wait it out, then the proof state rejoins without a commit (the
        // next pose re-verifies or refills).
        auto* timeline = s.native->context->computeTimeline();
        if (timeline->Wait(s.proof.proofValue, kProofTimeoutNs) == VK_SUCCESS) {
            s.pending = false;
            s.PutProof();
        }
    }
    if (s.pending) Quarantine();
}

void DeformPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}

bool DeformPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}

DeformPipeline::Candidate::Output DeformPipeline::Candidate::output() const noexcept {
    if (!state_ || !state_->proved || state_->semantic != 0) return {};
    return Output{state_->pointCount ? state_->outPoints : nullptr};
}

std::shared_ptr<const ChargedBuffer> DeformPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputs.points : nullptr;
}

std::shared_ptr<DeviceContext> DeformPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

uint32_t DeformPipeline::Candidate::pointCount() const noexcept {
    return state_ ? state_->pointCount : 0;
}

uint64_t DeformPipeline::Candidate::ProofValue() const noexcept {
    if (!state_ || state_->lost || state_->stage != State::Stage::Proof) return 0;
    return state_->ProofWaitValue();
}

VkResult DeformPipeline::Candidate::Poll(DeformSemantic* semantic) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (s.stage == State::Stage::Proof || s.stage == State::Stage::Failed) {
        VkResult r = Advance(semantic);
        if (r != VK_SUCCESS) return r;
    }
    if (!s.proved) {
        auto d = s.native->context->device();
        VkResult r = s.native->context->computeTimeline()->Check(s.applyValue);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        void* data = nullptr;
        r = vkMapMemory(d, s.status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) return r;
        std::memcpy(&s.semantic, data, 4);
        vkUnmapMemory(d, s.status->memory());
        s.pending = false;
        s.proved = true;
        s.stage = State::Stage::Done;
        // The apply signaled, so the scratch set is GPU-idle and no
        // host reader remains (Poll copied the only scalar): rejoin the
        // pool now, while this candidate is still alive, so the next pose's
        // Begin reuses it. Exactly once; concurrent Polls share the guard.
        if (!s.scratchReturned.exchange(true, std::memory_order_acq_rel))
            s.ReturnScratch();
    }
    if (semantic) *semantic = static_cast<DeformSemantic>(s.semantic);
    return VK_SUCCESS;
}

// ============================ Begin ============================

namespace {

// Affordable R-cache capacity (CVs) for n samples given this many free
// pool bytes: both buffers' 64KB-aligned charges must fit under the free
// bytes minus the funding margin. 0 means the budget cannot fund the
// floor, so the pose runs direct. Pure arithmetic on a snapshot
// (usedBytes is one coherent atomic; the margin plus the halving retry
// at the funding site cover races and wider-than-assumed driver
// alignments), so an unfunded pose costs no vkAllocateMemory. The caller
// clamps the capacity to the pose's point count: the floor gates the
// budget, not small poses (a 12-point pose on a big budget funds 12).
// maxRowBytes caps the whole-bound R rows (and proof snapshot) to the
// device single-descriptor range as well as the pool budget: bindings
// carry the full buffer, so a budget-funded prefix must also fit
// maxStorageBufferRange (128MiB on lavapipe and the spec minimum).
uint32_t AffordableCachePrefix(size_t freeBytes, int n, uint64_t maxRowBytes) {
    if (freeBytes <= kEvalCacheFundingMargin) return 0;
    size_t const free = freeBytes - kEvalCacheFundingMargin;
    auto aligned = [](uint64_t bytes) {
        return (bytes + kEvalCacheFundingAlign - 1) / kEvalCacheFundingAlign *
            kEvalCacheFundingAlign;
    };
    uint64_t p = free / (uint64_t(n) * 8u + 12u);
    if (p > kEvalCacheMaxPrefixCVs) p = kEvalCacheMaxPrefixCVs;
    uint64_t const perRow = uint64_t(n) * 8u;
    uint64_t rowCap = perRow ? maxRowBytes / perRow : 0;
    uint64_t const proofCap = maxRowBytes / 12u;
    if (rowCap > proofCap) rowCap = proofCap;
    if (p > rowCap) p = rowCap;
    while (p >= kEvalCacheMinPrefixCVs) {
        uint64_t const charged =
            aligned(p * 12u) + aligned(p * uint64_t(n) * 8u);
        if (charged <= free) break;
        p = (p > 64) ? p - 64 : 0;
    }
    if (p < kEvalCacheMinPrefixCVs) return 0;
    return uint32_t(p);
}

// Runs the past-prefix suffix [begin, end) on the process task scheduler
// (at most kHeteroWorkers threads, the caller included).
// `cvs` and `warped` are the mapped warped buffer base (the device staged
// the suffix CV bytes into warped[begin, end) first); the range writes its
// own warped triples and ORs the non-finite flag. Every query is
// independent, so any partition is bitwise the direct shader. Grains of
// kHeteroGrainsize CVs keep the claimed-chunk traffic negligible. Returns
// false only on a programming error (nulls, bad n/m), which the caller
// quarantines: a silent partial suffix must never reach apply.
bool RunHeteroSuffix(DeformEvalCpuParams const& p,
                         float const* cvs, float* warped,
                         uint32_t begin, uint32_t end, uint32_t* flag)
{
    if (begin >= end) return true;
    struct Payload {
        DeformEvalCpuParams const* p;
        float const* cvs;
        float* warped;
        uint32_t begin, end;
        std::atomic<uint32_t>* bad;
        std::atomic<bool>* ok;
    };
    std::atomic<uint32_t> bad{0};
    std::atomic<bool> ok{true};
    Payload payload{&p, cvs, warped, begin, end, &bad, &ok};
    size_t const grains =
        (size_t(end - begin) + kHeteroGrainsize - 1) / kHeteroGrainsize;
    UsdGenParallelFor(grains,
                     [](size_t g, void* v) {
                         auto* pl = static_cast<Payload*>(v);
                         size_t const lo =
                             size_t(pl->begin) + g * kHeteroGrainsize;
                         size_t const hi =
                             std::min(lo + kHeteroGrainsize, size_t(pl->end));
                         uint32_t f = 0;
                         if (!DeformEvaluateCpu(*pl->p, pl->cvs + lo * 3,
                                     pl->warped + lo * 3, hi - lo, &f))
                             pl->ok->store(false, std::memory_order_relaxed);
                         if (f) pl->bad->fetch_or(f, std::memory_order_relaxed);
                     },
                     &payload, size_t(kHeteroWorkers));
    if (bad.load(std::memory_order_relaxed)) *flag |= 1u;
    return ok.load(std::memory_order_relaxed);
}

// One-time host throughput calibration over synthetic normal-range CVs
// (timing only; values never escape). Runs on the calling thread ahead of
// the first eligible pose's tentative decision.
double CalibrateHeteroCpu(double const* samplesNorm, double const* coef,
                           int n, int m, double const center[3],
                           double invScale, double scale)
{
    UsdGenScratchScope scratch;
    float* const qs = scratch.Array<float>(size_t(kHeteroCalibCvs) * 3);
    float* const ds = scratch.Array<float>(size_t(kHeteroCalibCvs) * 3);
    uint64_t s = 0x243f6a8885a308d3ull;
    for (size_t i = 0; i < size_t(kHeteroCalibCvs) * 3; ++i) {
        float& v = qs[i];
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v = float(double(s >> 11) * 0x1p-53 * 4.0 - 2.0);
    }
    DeformEvalCpuParams p;
    p.samples = samplesNorm;
    p.coef = coef;
    p.n = n;
    p.m = m;
    p.cx = center[0];
    p.cy = center[1];
    p.cz = center[2];
    p.invScale = invScale;
    p.scale = scale;
    uint32_t flag = 0;
    auto t0 = std::chrono::steady_clock::now();
    DeformEvaluateCpu(p, qs, ds, kHeteroCalibCvs, &flag);
    double ns = std::chrono::duration<double, std::nano>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    double per = ns / double(kHeteroCalibCvs);
    return per > 0.0 ? per : 0.0;
}

} // namespace

bool DeformPipeline::Native::HeteroWants(HeteroPolicy& hp,
                                         uint32_t suffixCvs,
                                         uint32_t prefixCvs)
{
    int const force = forceHeteroSuffix.load(std::memory_order_relaxed);
    if (force > 0) return true;
    if (force < 0) return false;
    if (hp.cpuNsPerCv <= 0.0) return false;
    double const cpuEst = double(suffixCvs) * hp.cpuNsPerCv /
            double(kHeteroWorkers) +
        kHeteroFixedOverheadNs;
    double const gpuSuffix = hp.gpuNsPerCv > 0.0
        ? double(suffixCvs) * hp.gpuNsPerCv
        : double(suffixCvs) * kHeteroSeedGpuDirectNsPerCv;
    double const prefix = hp.prefixProofNs > 0.0 ? hp.prefixProofNs
                                                 : double(prefixCvs) * kHeteroSeedCachedNsPerCv;
    if (hp.armed) return cpuEst < gpuSuffix + prefix;
    if (hp.probeIn > 0) return false;
    // First probe is margin-gated (fresh calibration and seeds); re-probes
    // run blind, since a stale estimate must never veto the measurement
    // that would refresh it.
    if (!hp.everProbed) return cpuEst < (gpuSuffix + prefix) * 0.9;
    return true;
}

uint32_t DeformPipeline::Native::HeteroBalanceCap(HeteroPolicy& hp,
                                                  uint32_t points)
{
    // Past-balance funding wastes the host: the 2048MiB bench funds the
    // whole 1M prefix and runs 3.99ms GPU-only while the 512MiB split
    // overlaps the host suffix with the prefix proof at 2.90ms. Both legs
    // are linear in the split, so the optimum is their crossing; the
    // margin absorbs the linearized prefix rate (the proof EMA carries
    // fixed submit/wait costs the per-CV rate spreads). Tiny poses and
    // tiny suffixes never cap: the fixed overhead owns the comparison
    // there and the minimum keeps hetero in its tested shapes.
    if (hp.cpuNsPerCv <= 0.0 || points == 0) return UINT32_MAX;
    double const prefixRate = (hp.prefixProofNs > 0.0 && hp.prefixProofCvs > 0)
        ? hp.prefixProofNs / double(hp.prefixProofCvs)
        : kHeteroSeedCachedNsPerCv;
    if (!(prefixRate > 0.0)) return UINT32_MAX;
    double const cpuRate = hp.cpuNsPerCv / double(kHeteroWorkers);
    double const fullGpu = double(points) * prefixRate;
    double const active = (double(points) * cpuRate + kHeteroFixedOverheadNs) /
        (prefixRate + cpuRate);
    if (!(active > 0.0) || active >= double(points)) return UINT32_MAX;
    uint32_t const cap = uint32_t(active);
    if (cap < kEvalCacheMinPrefixCVs) return UINT32_MAX;
    if (points - cap < kEvalCacheMinPrefixCVs) return UINT32_MAX;
    double const leg = std::max(double(cap) * prefixRate,
        double(points - cap) * cpuRate + kHeteroFixedOverheadNs);
    if (!(leg < fullGpu * kHeteroBalanceMargin)) return UINT32_MAX;
    return cap;
}

std::unique_ptr<DeformPipeline::Candidate> DeformPipeline::Launch(
    BeginInfo info, VkResult* result, DeformSemantic* semantic, BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    auto reject = [&](DeformSemantic code) {
        if (semantic) *semantic = code;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        return std::unique_ptr<Candidate>{};
    };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (semantic) *semantic = DeformSemantic::Ok;
    auto context = info.points ? info.points->context()
                   : (info.curveOffsets ? info.curveOffsets->context() : nullptr);
    if (!context) context = native_->context;
    if (!context || context != native_->context) return {};
    if (!context->computeTimeline()) return {};

    uint32_t const curves = info.curveCount, points = info.pointCount;
    int const n = info.sampleCount;
    int const m = n + 4;

    // Sample count range.
    if (n < 4 || n > 100) return reject(DeformSemantic::BadValue);

    // Sample sizes.
    if ((int)info.restSamples.size() != 3 * n || (int)info.posedSamples.size() != 3 * n)
        return reject(DeformSemantic::BadValue);

    // Finite check.
    for (float v : info.restSamples)
        if (!std::isfinite(v)) return reject(DeformSemantic::NonFinite);
    for (float v : info.posedSamples)
        if (!std::isfinite(v)) return reject(DeformSemantic::NonFinite);

    // Groom envelope.
    if (!std::isfinite(info.groomEnvelope) || info.groomEnvelope < 0.0f)
        return reject(DeformSemantic::BadValue);

    auto check = [&](std::shared_ptr<ChargedBuffer const> const& b, VkDeviceSize bytes) {
        return b && b->context() == context && !b->unproven() && b->buffer() &&
               (b->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && b->sizeBytes() >= bytes;
    };

    // Geometry shape.
    if (curves == 0 && points != 0) return reject(DeformSemantic::BadOffsets);
    if (curves > 0) {
        if (!check(info.points, VkDeviceSize(points) * 12u) ||
            !check(info.curveOffsets, VkDeviceSize(curves + 1u) * 4u) ||
            !check(info.rootTargets, VkDeviceSize(curves) * 12u))
            return reject(DeformSemantic::BadOffsets);
    } else {
        if (info.points || info.curveOffsets || info.rootTargets)
            return reject(DeformSemantic::BadOffsets);
    }

    // Field validation.
    auto fieldCount = [&](uint32_t domain) -> uint32_t {
        switch (domain) {
            case kDomainGroom:       return 1u;
            case kDomainPrimitive:   return curves;
            case kDomainPoint:       return points;
            default: return 0;
        }
    };
    auto vScalar = [&](DeformPipeline::ScalarField const& f) {
        if (f.domain != kDomainGroom && f.domain != kDomainPrimitive && f.domain != kDomainPoint)
            return false;
        if (!f.data) {
            if (f.count != 0) return false;
            if (f.domain != kDomainGroom && fieldCount(f.domain) != 0) return false;
            if (!std::isfinite(f.literal)) return false;
            if (f.domain == kDomainGroom) {
                if (f.literal < 0.0f || f.literal > 1.0f) return false;
            }
            return true;
        }
        uint32_t expected = fieldCount(f.domain);
        return f.count == expected && check(f.data, VkDeviceSize(expected) * 4u);
    };
    auto vBool = [&](DeformPipeline::BoolField const& f, bool groomOnly, bool allowPrimitive) {
        if (f.domain != kDomainGroom && f.domain != kDomainPrimitive && f.domain != kDomainPoint)
            return false;
        if (groomOnly && f.domain != kDomainGroom) return false;
        if (!allowPrimitive && f.domain == kDomainPrimitive) return false;
        if (f.domain == kDomainPoint) return false;
        if (!f.data) {
            if (f.domain == kDomainGroom && f.count == 0) return true;
            if (f.domain == kDomainPrimitive && f.count == 0 && curves == 0) return true;
            return false;
        }
        uint32_t expected = fieldCount(f.domain);
        return f.count == expected && check(f.data, VkDeviceSize(expected) * 4u);
    };
    if (!vScalar(info.mask)) return reject(DeformSemantic::BadValue);
    if (!vBool(info.enabled, true, false)) return reject(DeformSemantic::BadValue);
    if (!vBool(info.lockRoots, false, true)) return reject(DeformSemantic::BadValue);

    // Device-resident topology is validated by the apply shader; only its
    // scalar diagnostic is read back before exposing any output.

    // CPU solve, with the factorization cached while rest and smoothing
    // are unchanged (a bitwise hit replays exactly what a rebind would
    // compute, including bind-phase failures).
    RbfState rstate;
    {
        float3 const* rest = reinterpret_cast<float3 const*>(info.restSamples.data());
        float3 const* posed = reinterpret_cast<float3 const*>(info.posedSamples.data());
        auto cache = native_->factorCache.Load();
        size_t const restBytes = info.restSamples.size() * sizeof(float);
        bool const hit = cache && cache->n == n &&
            cache->smoothing == info.smoothing &&
            cache->rest.size() == info.restSamples.size() &&
            std::memcmp(cache->rest.data(), info.restSamples.data(), restBytes) == 0;
        if (!hit) {
            auto fresh = UsdGenMakeRef<Native::FactorEntry>();
            fresh->rest = info.restSamples;
            fresh->n = n;
            fresh->smoothing = info.smoothing;
            BindRbf(rest, n, info.smoothing, fresh->factor);
            cache = fresh;
            native_->factorCache.Store(std::move(fresh));
        }
        if (cache->factor.status != RbfStatus::Code::Ok)
            rstate.status = cache->factor.status;
        else
            SolveRbfPosed(cache->factor, rest, posed, rstate);
    }
    double const invScale = 1.0 / rstate.scale;
    if (rstate.status != RbfStatus::Code::Ok) {
        DeformSemantic sem;
        switch (rstate.status) {
            case RbfStatus::Code::RankDeficient: sem = DeformSemantic::RankDeficient; break;
            case RbfStatus::Code::NonFinite:     sem = DeformSemantic::NonFinite; break;
            default:                             sem = DeformSemantic::SolverError; break;
        }
        return reject(sem);
    }

    // Host-centered samples, once per pose: the shader's samples[] input
    // (double(float) - center), shared by the samplesBuf upload and the
    // hetero suffix so both consume identical doubles. Pure host math on
    // already-validated inputs, hoisted before scratch funding so the
    // hetero calibration below can use it.
    double samplesNorm[3 * kMaxRbfSamples];
    for (int j = 0; j < n; ++j) {
        samplesNorm[size_t(3 * j)] =
            double(info.restSamples[size_t(3 * j)]) - rstate.center[0];
        samplesNorm[size_t(3 * j + 1)] =
            double(info.restSamples[size_t(3 * j + 1)]) - rstate.center[1];
        samplesNorm[size_t(3 * j + 2)] =
            double(info.restSamples[size_t(3 * j + 2)]) - rstate.center[2];
    }

    // Tentative hetero decision (finalized after the cache decision): the
    // warped memory type funds with the scratch, so the policy answers
    // from the funded-prefix estimate. Steady poses estimate exactly
    // (cacheActive equals min(points, rowsFunded) absent regrow and the
    // balance cap below); pose 0 has no prefix. The proof state is taken
    // here and kept through the proof phase (or put back once the pose
    // runs direct); a pose that finds it taken runs direct.
    bool const heteroStatic = native_->cachedPipeline != VK_NULL_HANDLE &&
        !disableEvalCache.load(std::memory_order_relaxed) &&
        info.points &&
        (info.points->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    bool wantHeteroTent = false;
    Native::ProofHold proof(native_.get());
    if (native_->cachedPipeline != VK_NULL_HANDLE &&
        !disableEvalCache.load(std::memory_order_relaxed))
        (void)proof.Take();
    if (proof) {
        uint32_t tentPrefix = proof->evalCache.rowsFunded;
        auto& hp = proof->hetero;
        // Balance-aware estimate: the final decision caps a past-balance
        // prefix, so the suffix here — and the warped typing plus the
        // calibration below — must see the capped split, not the funded
        // one. Uncapped budgets estimate exactly as before.
        if (heteroStatic) {
            uint32_t const tentCap = Native::HeteroBalanceCap(hp, points);
            if (tentCap < tentPrefix) tentPrefix = tentCap;
        }
        uint32_t const tentSuffix = points > tentPrefix ? points - tentPrefix : 0;
        if (heteroStatic && tentSuffix > 0 && hp.cpuNsPerCv <= 0.0) {
            hp.cpuNsPerCv = CalibrateHeteroCpu(samplesNorm, rstate.coef,
                n, m, rstate.center, invScale, rstate.scale);
        }
        wantHeteroTent = heteroStatic && tentSuffix > 0 &&
            Native::HeteroWants(hp, tentSuffix, std::min(points, tentPrefix));
    }

    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->inputs = info;
        s->pointCount = points;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        auto d = context->device();
        VkResult r;
        VkDeviceSize const outBytes = points ? VkDeviceSize(points) * 12u : 4u;
        // Evicts a funded R cache to make room for pose scratch. Only the
        // proof state's owner evicts, so no in-flight evaluate references
        // the buffers (every pose's evaluate completes before it puts the
        // state back), and the cache buffers are never MarkSubmitted, so
        // dropping them frees the storage and returns the charge. False
        // when nothing was funded or another pose owns the state.
        auto evictCache = [&](Native::ProofState& state) -> bool {
            auto& cache = state.evalCache;
            if (!cache.r && !cache.proof) return false;
            cache.r.reset();
            cache.proof.reset();
            cache.rest.clear();
            cache.n = 0;
            cache.rowsAlloc = 0;
            cache.rowsFunded = 0;
            cache.filled = 0;
            cache.missStreak = 0;
            cache.bypassLeft = 0;
            cache.everFilled = false;
            evalCacheUnfunded.fetch_add(1, std::memory_order_relaxed);
            evalCacheFundedPrefix.store(0, std::memory_order_relaxed);
            return true;
        };
        // A pose that runs direct because another pose held the state takes
        // it now if that pose has put it back meanwhile, evicts, and returns
        // it at once (it still runs direct). While the state is held, the
        // holder's evaluate may read the cache, so nothing is evicted and the
        // pose fails on the budget: it never waits for the holder's proof.
        auto evictCacheForRoom = [&]() -> bool {
            if (proof) return evictCache(*proof.operator->());
            Native::ProofHold late(native_.get());
            return late.Take() && evictCache(*late.operator->());
        };
        // Pose-scratch funding with one eviction retry: a failed charge
        // first drops the R cache (if any) and retries, so a pose the
        // cache starved runs direct instead of failing.
        auto createScratch = [&](std::shared_ptr<ChargedBuffer>& slot,
                                 VkBufferCreateInfo const& bi,
                                 VkMemoryPropertyFlags props,
                                 UsdGenExecutionResourceKind kind) {
            slot = ChargedBuffer::Create(context, bi, props, kind, &r);
            if (!slot && evictCacheForRoom())
                slot = ChargedBuffer::Create(context, bi, props, kind, &r);
            return bool(slot);
        };

        // Check out a proven-idle scratch set whose warped buffer matches
        // this pose's point count; anything missing is allocated fresh
        // below (at pool sizes, so it rejoins the pool in turn). The
        // empty-topology pose needs no scratch and skips the checkout.
        s->warpedHostVisible = wantHeteroTent;
        if (curves != 0) {
            for (auto& slot : native_->scratchIdle) {
                if (!slot.Claim(Native::kSlotFull)) continue;  // empty or busy: skip
                auto& set = slot.value;
                if (!(set.warped && set.warped->sizeBytes() == outBytes &&
                      set.warpedHostVisible == wantHeteroTent)) {
                    slot.Release(Native::kSlotFull);
                    continue;
                }
                s->warped = std::move(set.warped);
                s->status = std::move(set.status);
                s->evalStatus = std::move(set.evalStatus);
                s->coefBuf = std::move(set.coefBuf);
                s->samplesBuf = std::move(set.samplesBuf);
                s->evalUbo = std::move(set.evalUbo);
                s->applyUbo = std::move(set.applyUbo);
                s->verifyUbo = std::move(set.verifyUbo);
                s->fillUbo = std::move(set.fillUbo);
                s->cachedUbo = std::move(set.cachedUbo);
                s->fillArgs = std::move(set.fillArgs);
                set = Native::ScratchSet{};
                slot.Release(Native::kSlotEmpty);
                break;
            }
            // outPoints: exact-size reuse, but only while the pool is the
            // sole owner — a caller-held Output keeps its buffer shared,
            // and then the pose allocates fresh below. A buffer the pool
            // owns alone has no caller reader: new Outputs only come from
            // live states, never from the pool.
            for (auto& slot : native_->outPointsIdle) {
                if (!slot.Claim(Native::kSlotFull)) continue;
                if (slot.value->sizeBytes() == outBytes && slot.value.use_count() == 1) {
                    s->outPoints = std::move(slot.value);
                    slot.Release(Native::kSlotEmpty);
                    break;
                }
                slot.Release(Native::kSlotFull);
            }
        }

        // outPoints: DEVICE_LOCAL.
        {
            if (!s->outPoints) {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = outBytes;
                bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (!createScratch(s->outPoints, bi,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        UsdGenExecutionResourceKind::Active)) { finish(r); return {}; }
            }
        }
        // status: 4B host-visible.
        {
            if (!s->status) {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = 4;
                bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (!createScratch(s->status, bi,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        UsdGenExecutionResourceKind::Scratch)) { finish(r); return {}; }
            }
            void* data = nullptr;
            r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            std::memset(data, 0, 4);
            vkUnmapMemory(d, s->status->memory());
        }
        // evalStatus: 4B host-visible.
        {
            if (!s->evalStatus) {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = 4;
                bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (!createScratch(s->evalStatus, bi,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        UsdGenExecutionResourceKind::Scratch)) { finish(r); return {}; }
            }
            void* data = nullptr;
            r = vkMapMemory(d, s->evalStatus->memory(), 0, 4, 0, &data);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            std::memset(data, 0, 4);
            vkUnmapMemory(d, s->evalStatus->memory());
        }

        // Empty topology: early-out.
        if (curves == 0) {
            s->proved = true;
            s->semantic = 0;
            finish(VK_SUCCESS);
            return candidate;
        }

        // Pose scratch funds before the cache decision below, so a funded
        // cache can never starve the current pose: whatever the estimate
        // sizes the prefix to, this pose's scratch is already charged.
        // warped: hetero poses take host-visible (the host reads the staged
        // suffix CVs and writes the suffix back through the mapping); GPU
        // poses keep device-local. TRANSFER_DST on both (the hetero copy
        // target); usage bits alone never move the pool or the speed.
        {
            if (!s->warped) {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = outBytes;
                bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                VkMemoryPropertyFlags const warpedProps = wantHeteroTent
                    ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                    : VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (!createScratch(s->warped, bi, warpedProps,
                        UsdGenExecutionResourceKind::Active)) { finish(r); return {}; }
            }
        }
        // coefBuf: HOST_VISIBLE, 3*m doubles. Fresh buffers are allocated
        // at the max so the set rejoins the pool at any sample count; the
        // upload and the descriptor range stay at this pose's actual bytes.
        VkDeviceSize const coefBytes = VkDeviceSize(m) * 24u;
        {
            if (!s->coefBuf) {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = std::max(coefBytes, kMaxCoefBytes);
                bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (!createScratch(s->coefBuf, bi,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        UsdGenExecutionResourceKind::Scratch)) { finish(r); return {}; }
            }
            void* data = nullptr;
            r = vkMapMemory(d, s->coefBuf->memory(), 0, coefBytes, 0, &data);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            std::memcpy(data, rstate.coef, coefBytes);
            vkUnmapMemory(d, s->coefBuf->memory());
        }
        // samplesBuf: HOST_VISIBLE, 3*n doubles, centered on the host.
        // The evaluate shader used to reload every float sample and subtract
        // the center for every CV; the stored doubles are bitwise what that
        // computed (double(float) - center). The * invScale stays in the
        // shader deliberately: the driver FMA-contracts x - sn*invScale, and
        // keeping the expression shape keeps that contraction (and the bits)
        // identical.
        VkDeviceSize const samplesBytes = VkDeviceSize(n) * 24u;
        {
            if (!s->samplesBuf) {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = std::max(samplesBytes, kMaxSamplesBytes);
                bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (!createScratch(s->samplesBuf, bi,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        UsdGenExecutionResourceKind::Scratch)) { finish(r); return {}; }
            }
            void* data = nullptr;
            r = vkMapMemory(d, s->samplesBuf->memory(), 0, samplesBytes, 0, &data);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            std::memcpy(data, samplesNorm, size_t(samplesBytes));
            vkUnmapMemory(d, s->samplesBuf->memory());
        }
        // evalUbo: HOST_VISIBLE, 64B. Funded here with the rest of the
        // pose scratch; filled after the cache decision below (the base
        // word depends on it).
        {
            if (!s->evalUbo) {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = kEvalUboBytes;
                bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (!createScratch(s->evalUbo, bi,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        UsdGenExecutionResourceKind::Scratch)) { finish(r); return {}; }
            }
        }
        // applyUbo: HOST_VISIBLE, 80B.
        {
            if (!s->applyUbo) {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = kApplyUboBytes;
                bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (!createScratch(s->applyUbo, bi,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        UsdGenExecutionResourceKind::Scratch)) { finish(r); return {}; }
            }
            ApplyUbo a = {};
            a.counts[0] = curves; a.counts[1] = points; a.counts[2] = n;
            a.groomEnvelope = info.groomEnvelope;
            a.maskF[0] = info.mask.literal;
            a.maskF[1] = info.mask.data ? 1.0f : 0.0f;
            a.maskF[2] = float(info.mask.domain);
            a.enabledF[0] = float(info.enabled.literal);
            a.enabledF[1] = info.enabled.data ? 1.0f : 0.0f;
            a.enabledF[2] = float(info.enabled.domain);
            a.lockF[0] = float(info.lockRoots.literal);
            a.lockF[1] = info.lockRoots.data ? 1.0f : 0.0f;
            a.lockF[2] = float(info.lockRoots.domain);
            a.fieldCounts[0] = info.mask.count;
            a.fieldCounts[1] = info.enabled.count;
            a.fieldCounts[2] = info.lockRoots.count;
            // Uniform-stride candidate for the apply shader's point->curve
            // map (verified against the offsets at every use; 0 disables
            // it). Pure scalar math on already-known counts: no producer
            // trust, no scan. The empty-topology pose returns before this
            // fill, so curves is nonzero; the guard is belt and braces.
            a.strideCandidate =
                curves > 0 && points % curves == 0 ? points / curves : 0;
            void* data = nullptr;
            r = vkMapMemory(d, s->applyUbo->memory(), 0, kApplyUboBytes, 0, &data);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            std::memcpy(data, &a, sizeof(a));
            vkUnmapMemory(d, s->applyUbo->memory());
        }
        // Cache per-pose scratch: the three cache UBOs plus the indirect
        // fill args. Allocated on cache-capable pipelines even for direct
        // poses (so pooled sets stay complete); filled below only when
        // this pose runs cached.
        bool const cacheCapable = native_->cachedPipeline != VK_NULL_HANDLE;
        auto allocUbo = [&](std::shared_ptr<ChargedBuffer>& slot, uint32_t bytes,
                            VkBufferUsageFlags usage) {
            if (slot || !cacheCapable) return true;
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes;
            bi.usage = usage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            return createScratch(slot, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch);
        };
        if (!allocUbo(s->verifyUbo, kVerifyUboBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
            !allocUbo(s->fillUbo, kFillUboBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
            !allocUbo(s->cachedUbo, kCachedUboBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
            !allocUbo(s->fillArgs, kFillArgsBytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)) {
            finish(r);
            return {};
        }

        // Cached-R decision. The whole cached proof phase below (uploads,
        // descriptors, recording, submit, wait, bookkeeping) owns the proof
        // state, so a force commit is atomic with its fill; early returns
        // put it back via the hold's destructor. The direct path only
        // uses it for the bypass countdown.
        bool useCache = false;
        bool forceFill = false;
        uint32_t cacheActive = 0;
        uint32_t fundedPrefix = 0;
        if (proof) {
            auto& cache = proof->evalCache;
            if (cache.bypassLeft > 0) {
                cache.bypassLeft--;
                evalCacheBypassed.fetch_add(1, std::memory_order_relaxed);
                proof.Put();
            } else {
                // Single-descriptor range cap for the whole-bound R rows
                // (and proof snapshot): queried per funding decision, so
                // steady-state poses pay no driver call.
                auto maxRowBytes = [&]() {
                    VkPhysicalDeviceProperties props{};
                    vkGetPhysicalDeviceProperties(context->physicalDevice(),
                                                  &props);
                    return uint64_t(props.limits.maxStorageBufferRange);
                };
                auto poolFree = [&]() {
                    auto snap = context->resources()->Snapshot();
                    return snap.usableBytes > snap.usedBytes
                        ? snap.usableBytes - snap.usedBytes : size_t(0);
                };
                // Funds fresh R rows plus the proof snapshot at the given
                // prefix, halving down to the floor while the charges miss
                // (a race, or a wider-than-assumed driver alignment). All
                // or nothing: a partial pair is destroyed, never kept.
                // Returns the funded prefix, or 0. Below-floor targets
                // (small poses clamped to their point count) get one
                // attempt: a race there retries next pose, not smaller.
                auto fundFresh = [&](uint32_t p) -> uint32_t {
                    uint32_t const lo = std::min(p, kEvalCacheMinPrefixCVs);
                    while (p >= lo && p != 0) {
                        VkBufferCreateInfo bi{};
                        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                        bi.size = VkDeviceSize(p) * VkDeviceSize(n) * 8u;
                        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                        VkResult cr = VK_SUCCESS;
                        auto rBuf = ChargedBuffer::Create(context, bi,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            UsdGenExecutionResourceKind::Cache, &cr);
                        if (rBuf) {
                            bi.size = VkDeviceSize(p) * 12u;
                            auto proofBuf = ChargedBuffer::Create(context, bi,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                UsdGenExecutionResourceKind::Cache, &cr);
                            if (proofBuf) {
                                cache.r = std::move(rBuf);
                                cache.proof = std::move(proofBuf);
                                cache.rowsAlloc = n;
                                cache.rowsFunded = p;
                                cache.filled = 0;
                                cache.everFilled = false;
                                evalCacheFundedPrefix.store(p, std::memory_order_relaxed);
                                return p;
                            }
                        }
                        p /= 2;
                    }
                    return 0;
                };
                // First cached pose funds the prefix from the pool budget;
                // later poses reuse it. Any failure runs this pose direct
                // and retries next pose.
                bool funded = cache.proof && cache.r;
                if (!funded) {
                    uint32_t const cap =
                        AffordableCachePrefix(poolFree(), n, maxRowBytes());
                    uint32_t const p = std::min(points, cap);
                    funded = p != 0 && fundFresh(p) != 0;
                }
                // R grows to a high-water n; shrinks never reallocate. A
                // failed grow keeps the old rows and runs direct.
                if (funded && n > cache.rowsAlloc &&
                    uint64_t(cache.rowsFunded) * uint64_t(n) * 8u > maxRowBytes()) {
                    // Grown samples would push the whole-bound R rows past
                    // the single-descriptor range: run this pose direct and
                    // keep the old cache for smaller-n poses.
                    funded = false;
                }
                if (funded && n > cache.rowsAlloc) {
                    VkBufferCreateInfo bi{};
                    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    bi.size = VkDeviceSize(cache.rowsFunded) *
                        VkDeviceSize(n) * 8u;
                    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    VkResult cr = VK_SUCCESS;
                    auto grown = ChargedBuffer::Create(context, bi,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        UsdGenExecutionResourceKind::Cache, &cr);
                    if (grown) {
                        cache.r = std::move(grown);
                        cache.rowsAlloc = n;
                    }
                    funded = n <= cache.rowsAlloc;
                }
                if (funded) {
                    uint32_t const want = std::min(points, kEvalCacheMaxPrefixCVs);
                    if (want > cache.rowsFunded) {
                        // Points outgrew the funded prefix (or a race
                        // underfunded it): regrow while the budget admits
                        // more. The old rows stay live until the swap, so
                        // the estimate excludes their charge; on failure
                        // the pose keeps the old prefix with a bigger
                        // suffix (still cached, still correct).
                        size_t const freeExcl = poolFree() +
                            cache.proof->allocationBytes() + cache.r->allocationBytes();
                        uint32_t const target = std::min(
                            want, AffordableCachePrefix(freeExcl, n, maxRowBytes()));
                        if (target > cache.rowsFunded) {
                            VkBufferCreateInfo bi{};
                            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                            bi.size = VkDeviceSize(target) * VkDeviceSize(n) * 8u;
                            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                            VkResult cr = VK_SUCCESS;
                            auto grownR = ChargedBuffer::Create(context, bi,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                UsdGenExecutionResourceKind::Cache, &cr);
                            std::shared_ptr<ChargedBuffer> grownProof;
                            if (grownR) {
                                bi.size = VkDeviceSize(target) * 12u;
                                grownProof = ChargedBuffer::Create(context, bi,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    UsdGenExecutionResourceKind::Cache, &cr);
                            }
                            if (grownR && grownProof) {
                                cache.r = std::move(grownR);
                                cache.proof = std::move(grownProof);
                                cache.rowsAlloc = n;
                                cache.rowsFunded = target;
                                cache.filled = 0;
                                evalCacheFundedPrefix.store(target,
                                    std::memory_order_relaxed);
                            }
                        }
                    }
                    cacheActive = std::min(points, cache.rowsFunded);
                    fundedPrefix = cache.rowsFunded;
                    size_t const restBytes = info.restSamples.size() * sizeof(float);
                    bool const restMatch = cache.everFilled &&
                        cache.rest.size() == info.restSamples.size() &&
                        std::memcmp(cache.rest.data(), info.restSamples.data(),
                                   restBytes) == 0;
                    // Hetero balance cap: past the balance point the extra
                    // prefix costs more device time than the host suffix it
                    // removes, so an engaging hetero pose runs the capped
                    // split. Funding stays full: a disengaging pose uncaps
                    // for free (forceFill below refills the regrown range),
                    // and forceFill sees the active it fills. The warped
                    // guard keeps a concurrent policy flip correct: a pose
                    // the tentative call typed device-local runs the full
                    // prefix on the device, never a capped prefix the host
                    // cannot map. The cap is sticky across steady poses: the
                    // live estimates drift, and a drifting cap would regrow
                    // the active prefix past the filled rows every pose. A
                    // model rejection clears it only once hetero agrees the
                    // GPU wins, so margin noise never flickers the split.
                    if (heteroStatic) {
                        auto& hp = proof->hetero;
                        uint32_t const freshCap =
                            Native::HeteroBalanceCap(hp, points);
                        bool const keysChanged = !restMatch || n != cache.n ||
                            cache.filled == 0 || points != cache.capPoints;
                        if (keysChanged || cache.capActive == 0) {
                            if (freshCap != UINT32_MAX) {
                                cache.capActive = freshCap;
                                cache.capPoints = points;
                            } else if (cache.capActive == 0 ||
                                       points <= cache.capActive ||
                                       !Native::HeteroWants(hp,
                                           points - cache.capActive,
                                           cache.capActive)) {
                                cache.capActive = 0;
                            }
                        }
                        uint32_t const cap = cache.capActive;
                        if (cap != 0 && cap < cacheActive && s->warpedHostVisible &&
                            Native::HeteroWants(hp, points - cap, cap))
                            cacheActive = cap;
                    }
                    forceFill = !cache.everFilled || !restMatch ||
                        n != cache.n || cacheActive > cache.filled;
                    useCache = true;
                    // Stays owned through the proof phase below.
                } else {
                    evalCacheUnfunded.fetch_add(1, std::memory_order_relaxed);
                    proof.Put();
                }
            }
        }

        // Final hetero decision: the tentative call reserved a host-visible
        // warped; the exact suffix confirms. A tentative hetero that lands
        // here GPU (concurrent policy flip, regrow surprise) runs the GPU
        // suffix on its host-visible warped: correct, transiently typed.
        bool const heteroEligible =
            heteroStatic && useCache && points > cacheActive;
        bool useHetero = heteroEligible && wantHeteroTent;
        if (useHetero)
            useHetero = Native::HeteroWants(proof->hetero,
                points - cacheActive, cacheActive);
        if (useHetero)
            heteroRuns.fetch_add(1, std::memory_order_relaxed);
        // Suffix input staging: the host suffix needs the suffix CV bytes.
        // A device-local points buffer stages them through cmds[2] into the
        // idle warped suffix (the copy batch below); a
        // host-visible coherent points buffer maps directly instead,
        // skipping the copy batch, its queue slot ahead of
        // the proof, and the copy wait. info.points is never
        // device-written (an input; check() rejects unproven buffers), so
        // mapped reads see exactly the bytes the copy would stage, and
        // they race nothing (the proof only reads points too).
        bool const pointsDirect = useHetero && info.points &&
            (info.points->memoryPropertyFlags() &
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        bool const stageCopy = useHetero && !pointsDirect;

        // evalUbo fill (see the funding above): on the cached path this
        // UBO feeds only the past-prefix suffix dispatch (base=active);
        // the direct dispatch keeps base=0.
        {
            EvalUbo e;
            e.n_m[0] = n; e.n_m[1] = m;
            e.counts[0] = points;
            e.counts[1] = useCache ? int32_t(cacheActive) : 0;
            e.cx = rstate.center[0]; e.cy = rstate.center[1]; e.cz = rstate.center[2];
            e.invScale = invScale;
            e.scale = rstate.scale;
            e._pad2 = 0.0;
            void* data = nullptr;
            r = vkMapMemory(d, s->evalUbo->memory(), 0, kEvalUboBytes, 0, &data);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            std::memcpy(data, &e, sizeof(e));
            vkUnmapMemory(d, s->evalUbo->memory());
        }
        if (useCache) {
            // The verify/fill/cached UBO contents plus the indirect args
            // pre-write: (0,1,1) speculates (a verify match leaves a no-op
            // fill), (fillGroups,n,1) forces the refill.
            VerifyUbo v;
            v.words_n[0] = int32_t(3u * cacheActive);
            v.words_n[1] = n;
            v.words_n[2] = 0; v.words_n[3] = 0;
            FillUbo f;
            f.shape[0] = n; f.shape[1] = int32_t(cacheActive);
            f.shape[2] = int32_t(fundedPrefix); f.shape[3] = 0;
            f.cx = rstate.center[0]; f.cy = rstate.center[1]; f.cz = rstate.center[2];
            f.invScale = invScale;
            CachedUbo c;
            c.nmcs[0] = n; c.nmcs[1] = m;
            c.nmcs[2] = int32_t(cacheActive); c.nmcs[3] = int32_t(fundedPrefix);
            c.cx = rstate.center[0]; c.cy = rstate.center[1]; c.cz = rstate.center[2];
            c.invScale = invScale;
            c.scale = rstate.scale;
            c._pad = 0.0;
            uint32_t args[3];
            // The divisor is the fill width, matching the verify shader's
            // mismatch write exactly (same formula, same values).
            args[0] = forceFill ? (cacheActive + kFillLocalSize - 1u) / kFillLocalSize : 0u;
            args[1] = forceFill ? uint32_t(n) : 1u;
            args[2] = 1u;
            struct Fill { std::shared_ptr<ChargedBuffer> const* slot; void const* data; uint32_t bytes; };
            Fill const fills[] = {
                {&s->verifyUbo, &v, kVerifyUboBytes},
                {&s->fillUbo, &f, kFillUboBytes},
                {&s->cachedUbo, &c, kCachedUboBytes},
                {&s->fillArgs, &args, kFillArgsBytes},
            };
            for (auto const& fill : fills) {
                void* data = nullptr;
                r = vkMapMemory(d, (*fill.slot)->memory(), 0, fill.bytes, 0, &data);
                if (r != VK_SUCCESS) { finish(r); return {}; }
                std::memcpy(data, fill.data, fill.bytes);
                vkUnmapMemory(d, (*fill.slot)->memory());
            }
        }

        // Descriptor pool + sets. The suffix dispatch keys off the active
        // prefix, not the funded one: they agree absent the balance cap,
        // and a capped pose that lands here GPU (a concurrent policy flip
        // between the cap and the hetero decision) must still evaluate
        // its suffix on the device.
        bool const hasSuffix = useCache && points > cacheActive && !useHetero;
        VkDescriptorPoolSize sizes[2] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, useCache
                 ? kVerifyFillStorage + kVerifyFillStorage + kEvalStorage +
                       (hasSuffix ? kEvalStorage : 0u) + kApplyStorage
                 : kEvalStorage + kApplyStorage},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, useCache
                 ? (hasSuffix ? 5u : 4u)
                 : 2u},
        };
        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = useCache ? (hasSuffix ? 5u : 4u) : 2u;
        dp.poolSizeCount = 2; dp.pPoolSizes = sizes;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet* set) {
            VkDescriptorSetAllocateInfo da{};
            da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            da.descriptorPool = s->descriptors;
            da.descriptorSetCount = 1; da.pSetLayouts = &layout;
            return vkAllocateDescriptorSets(d, &da, set);
        };
        VkDescriptorSet evalSet = VK_NULL_HANDLE, applySet = VK_NULL_HANDLE;
        VkDescriptorSet verifySet = VK_NULL_HANDLE, fillSet = VK_NULL_HANDLE;
        VkDescriptorSet cachedSet = VK_NULL_HANDLE, suffixSet = VK_NULL_HANDLE;
        if (useCache) {
            r = allocSet(native_->verifyFillLayout, &verifySet);
            if (r == VK_SUCCESS) r = allocSet(native_->verifyFillLayout, &fillSet);
            if (r == VK_SUCCESS) r = allocSet(native_->evalLayout, &cachedSet);
            if (r == VK_SUCCESS && hasSuffix) r = allocSet(native_->evalLayout, &suffixSet);
            if (r == VK_SUCCESS) r = allocSet(native_->applyLayout, &applySet);
        } else {
            r = allocSet(native_->evalLayout, &evalSet);
            if (r == VK_SUCCESS) r = allocSet(native_->applyLayout, &applySet);
        }
        if (r != VK_SUCCESS) { finish(r); return {}; }

        // Fallback planes for null field data.
        auto applyPlane = [&](std::shared_ptr<ChargedBuffer const> const& b, VkDeviceSize bytes)
            -> VkDescriptorBufferInfo {
            if (b && bytes) return {b->buffer(), 0, bytes};
            return {s->status->buffer(), 0, 4u};
        };

        VkDeviceSize const pointsBytes = points ? VkDeviceSize(points) * 12u : 4u;
        VkDeviceSize const offsetsBytes = VkDeviceSize(curves + 1u) * 4u;
        VkDeviceSize const targetsBytes = VkDeviceSize(curves) * 12u;

        // Eval set: b0=cvs, b1=warped, b2=samples, b3=coef, b4=evalUbo, b5=evalStatus.
        auto writeEvalLayoutSet = [&](VkDescriptorSet set, VkDescriptorBufferInfo rOrSamples,
                                      VkDescriptorBufferInfo ubo) {
            VkDescriptorBufferInfo eInfos[kEvalStorage + 1] = {
                applyPlane(info.points, pointsBytes),
                {s->warped->buffer(), 0, outBytes},
                rOrSamples,
                {s->coefBuf->buffer(), 0, coefBytes},
                ubo,
                {s->evalStatus->buffer(), 0, 4u},
            };
            VkWriteDescriptorSet eWrites[kEvalStorage + 1]{};
            uint32_t const eBindings[kEvalStorage + 1] = {0, 1, 2, 3, 4, 5};
            for (uint32_t i = 0; i < kEvalStorage + 1; ++i) {
                eWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                eWrites[i].dstSet = set; eWrites[i].dstBinding = eBindings[i];
                eWrites[i].descriptorCount = 1;
                eWrites[i].descriptorType = (i == 4) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                eWrites[i].pBufferInfo = &eInfos[i];
            }
            vkUpdateDescriptorSets(d, kEvalStorage + 1, eWrites, 0, nullptr);
        };
        auto writeVerifyFillSet = [&](VkDescriptorSet set, VkDescriptorBufferInfo b0,
                                      VkDescriptorBufferInfo b1, VkDescriptorBufferInfo b2,
                                      VkDescriptorBufferInfo ubo) {
            VkDescriptorBufferInfo infos[kVerifyFillStorage + 1] = {b0, b1, b2, ubo};
            VkWriteDescriptorSet writes[kVerifyFillStorage + 1]{};
            for (uint32_t i = 0; i < kVerifyFillStorage + 1; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = set; writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType =
                    (i == kVerifyFillStorage) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                              : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &infos[i];
            }
            vkUpdateDescriptorSets(d, kVerifyFillStorage + 1, writes, 0, nullptr);
        };
        if (useCache) {
            auto const& cache = proof->evalCache;
            // Verify set: b0=cvs, b1=proof, b2=fillArgs, b3=verifyUbo.
            writeVerifyFillSet(verifySet, applyPlane(info.points, pointsBytes),
                               {cache.proof->buffer(), 0, cache.proof->sizeBytes()},
                               {s->fillArgs->buffer(), 0, kFillArgsBytes},
                               {s->verifyUbo->buffer(), 0, kVerifyUboBytes});
            // Fill set: b0=cvs, b1=R, b2=samples, b3=fillUbo.
            writeVerifyFillSet(fillSet, applyPlane(info.points, pointsBytes),
                               {cache.r->buffer(), 0, cache.r->sizeBytes()},
                               {s->samplesBuf->buffer(), 0, samplesBytes},
                               {s->fillUbo->buffer(), 0, kFillUboBytes});
            // Cached set: b0=cvs, b1=warped, b2=R, b3=coef, b4=cachedUbo, b5=evalStatus.
            writeEvalLayoutSet(cachedSet, {cache.r->buffer(), 0, cache.r->sizeBytes()},
                               {s->cachedUbo->buffer(), 0, kCachedUboBytes});
            if (hasSuffix) {
                // Suffix set: the direct eval over [P,points).
                writeEvalLayoutSet(suffixSet,
                                   {s->samplesBuf->buffer(), 0, samplesBytes},
                                   {s->evalUbo->buffer(), 0, kEvalUboBytes});
            }
        } else {
            writeEvalLayoutSet(evalSet, {s->samplesBuf->buffer(), 0, samplesBytes},
                               {s->evalUbo->buffer(), 0, kEvalUboBytes});
        }

        // Apply set: b0=src, b1=warped, b2=offsets, b3=targets, b4=outPoints,
        //            b5=maskData, b6=enabledData, b7=lockData, b8=status, b9=applyUbo.
        VkDescriptorBufferInfo aInfos[kApplyStorage + 1] = {
            applyPlane(info.points, pointsBytes),
            {s->warped->buffer(), 0, outBytes},
            applyPlane(info.curveOffsets, offsetsBytes),
            applyPlane(info.rootTargets, targetsBytes),
            {s->outPoints->buffer(), 0, outBytes},
            applyPlane(info.mask.data,
                info.mask.count ? VkDeviceSize(info.mask.count) * 4u : 4u),
            applyPlane(info.enabled.data,
                info.enabled.count ? VkDeviceSize(info.enabled.count) * 4u : 4u),
            applyPlane(info.lockRoots.data,
                info.lockRoots.count ? VkDeviceSize(info.lockRoots.count) * 4u : 4u),
            {s->status->buffer(), 0, 4u},
            {s->applyUbo->buffer(), 0, kApplyUboBytes},
        };
        VkWriteDescriptorSet aWrites[kApplyStorage + 1]{};
        for (uint32_t i = 0; i < kApplyStorage + 1; ++i) {
            aWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            aWrites[i].dstSet = applySet; aWrites[i].dstBinding = i;
            aWrites[i].descriptorCount = 1;
            aWrites[i].descriptorType = (i == kApplyStorage) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            aWrites[i].pBufferInfo = &aInfos[i];
        }
        vkUpdateDescriptorSets(d, kApplyStorage + 1, aWrites, 0, nullptr);

        // Command pool + buffers.
        VkCommandPoolCreateInfo pc{};
        pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferAllocateInfo ca{};
        ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = s->commands; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = stageCopy ? 3u : 2u;
        VkCommandBuffer cmds[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
        r = vkAllocateCommandBuffers(d, &ca, cmds);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        // cmd0: evaluate (proof).
        r = vkBeginCommandBuffer(cmds[0], &begin);
        if (r == VK_SUCCESS) {
            VkMemoryBarrier before{};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmds[0],
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            if (useCache) {
                // Verify the prefix CVs (snapshotting them into the
                // proof), self-schedule the refill through the indirect
                // args, then evaluate the prefix through R and the suffix
                // directly.
                vkCmdBindPipeline(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->verifyPipeline);
                vkCmdBindDescriptorSets(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->verifyFillPipelineLayout, 0, 1, &verifySet, 0, nullptr);
                vkCmdDispatch(cmds[0], Groups(3u * cacheActive, kVerifyLocalSize), 1, 1);
                VkMemoryBarrier verifyToFill{};
                verifyToFill.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                verifyToFill.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                verifyToFill.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
                vkCmdPipelineBarrier(cmds[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &verifyToFill,
                    0, nullptr, 0, nullptr);
                vkCmdBindPipeline(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->fillPipeline);
                vkCmdBindDescriptorSets(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->verifyFillPipelineLayout, 0, 1, &fillSet, 0, nullptr);
                vkCmdDispatchIndirect(cmds[0], s->fillArgs->buffer(), 0);
                VkMemoryBarrier fillToCached{};
                fillToCached.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                fillToCached.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                fillToCached.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmds[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillToCached,
                    0, nullptr, 0, nullptr);
                vkCmdBindPipeline(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->cachedPipeline);
                vkCmdBindDescriptorSets(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->evalPipelineLayout, 0, 1, &cachedSet, 0, nullptr);
                vkCmdDispatch(cmds[0], Groups(cacheActive, kCachedLocalSize), 1, 1);
                if (hasSuffix) {
                    vkCmdBindPipeline(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                        native_->evalPipeline);
                    vkCmdBindDescriptorSets(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                        native_->evalPipelineLayout, 0, 1, &suffixSet, 0, nullptr);
                    vkCmdDispatch(cmds[0],
                        Groups(points - cacheActive, kEvalLocalSize), 1, 1);
                }
            } else {
                vkCmdBindPipeline(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->evalPipeline);
                vkCmdBindDescriptorSets(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->evalPipelineLayout, 0, 1, &evalSet, 0, nullptr);
                vkCmdDispatch(cmds[0], Groups(points, kEvalLocalSize), 1, 1);
            }
            VkMemoryBarrier after{};
            after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                  VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmds[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
            r = vkEndCommandBuffer(cmds[0]);
        }
        if (r != VK_SUCCESS) { finish(r); return {}; }
        // cmd1 is recorded before the eval proof runs, not after it: the
        // recording touches only host-side handles and counts (applySet,
        // the pipelines, curves/points), so it needs nothing the eval
        // produces, and the driver record calls then overlap the 13ms eval
        // instead of serializing behind the proof wait. A failed proof
        // discards the recording with the command pool; the submit below
        // still gates on the proof and beforeSubmit.
        // No zero-fill: every apply thread writes its point or reports a
        // status first (each early return calls StatusErr), so status 0
        // implies every point was written, and any other status discards
        // the output; the fill is never read.
        r = vkBeginCommandBuffer(cmds[1], &begin);
        if (r == VK_SUCCESS) {
            VkMemoryBarrier before{};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmds[1],
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(cmds[1], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->applyPipeline);
            vkCmdBindDescriptorSets(cmds[1], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->applyPipelineLayout, 0, 1, &applySet, 0, nullptr);
            // Per-point apply: one thread per point (plus curve-span
            // validation for thread i < curves), so the dispatch covers
            // whichever domain is larger.
            vkCmdDispatch(cmds[1], Groups(std::max(curves, points), kApplyLocalSize), 1, 1);
            VkMemoryBarrier after{};
            after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                  VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmds[1], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
            r = vkEndCommandBuffer(cmds[1]);
        }
        if (r != VK_SUCCESS) { finish(r); return {}; }
        // cmds[2] (staged-copy hetero only): stage the suffix CV bytes
        // into the idle warped suffix for the host. Submitted before the
        // proof so its timeline value delivers the CVs while the prefix still
        // runs; the proof's barrier orders it against the cached writes
        // (disjoint ranges). Direct-points hetero maps the inputs
        // instead and records no copy.
        if (stageCopy) {
            r = vkBeginCommandBuffer(cmds[2], &begin);
            if (r == VK_SUCCESS) {
                VkMemoryBarrier copyBefore{};
                copyBefore.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                copyBefore.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                copyBefore.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmds[2],
                    VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &copyBefore,
                    0, nullptr, 0, nullptr);
                VkBufferCopy region{};
                region.srcOffset = VkDeviceSize(cacheActive) * 12u;
                region.dstOffset = VkDeviceSize(cacheActive) * 12u;
                region.size = VkDeviceSize(points - cacheActive) * 12u;
                vkCmdCopyBuffer(cmds[2], info.points->buffer(),
                    s->warped->buffer(), 1, &region);
                // Post-copy availability: the host suffix maps and reads
                // these bytes right after the copy signals. The signal alone
                // only orders queue completion; without this barrier the
                // transfer writes' host visibility lags the signal by a
                // timing window (wider under unified-memory pressure),
                // and any fast-start reader (spin polling, a stagger-free
                // pool) samples stale patches. The barrier forces the
                // flush inside command execution, ahead of the signal.
                VkMemoryBarrier copyDone{};
                copyDone.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                copyDone.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                copyDone.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(cmds[2],
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &copyDone,
                    0, nullptr, 0, nullptr);
                r = vkEndCommandBuffer(cmds[2]);
            }
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        // Proof submit on the queue timeline: the staged suffix copy (when
        // any) and the evaluate proof go out as consecutive batches, so the
        // copy's value signals first and the host suffix can start while
        // the prefix still runs. Nothing waits here: Advance (or Poll)
        // checks the values, and Begin waits on them for its synchronous
        // contract.
        VulkanQueueTimeline* const timeline = context->computeTimeline();
        auto& ps = s->proof;
        VkCommandBuffer const proofCmds[2] = {cmds[2], cmds[0]};
        s->pending = true; // proof owns the same allocations as the final submit
        uint64_t const last = stageCopy ? timeline->Submit(proofCmds, 2u, &r)
                                        : timeline->Submit(&cmds[0], 1u, &r);
        if (!last) { candidate->Quarantine(); finish(r); return {}; }
        ps.submitted = std::chrono::steady_clock::now();
        ps.proofValue = last;
        ps.copyValue = stageCopy ? last - 1u : 0u;
        ps.applyCmd = cmds[1];
        ps.beforeSubmit = std::move(beforeSubmit);
        std::memcpy(ps.samplesNorm, samplesNorm, sizeof(double) * size_t(3 * n));
        std::memcpy(ps.coef, rstate.coef, sizeof(ps.coef));
        ps.center[0] = rstate.center[0];
        ps.center[1] = rstate.center[1];
        ps.center[2] = rstate.center[2];
        ps.invScale = invScale;
        ps.scale = rstate.scale;
        ps.n = n;
        ps.m = m;
        ps.cacheActive = cacheActive;
        ps.pointsBytes = pointsBytes;
        ps.outBytes = outBytes;
        ps.useCache = useCache;
        ps.forceFill = forceFill;
        ps.heteroEligible = heteroEligible;
        ps.useHetero = useHetero;
        ps.stageCopy = stageCopy;
        // The proof state rides with the stage until the proof is observed.
        ps.owned = proof.Release();
        s->stage = Candidate::State::Stage::Proof;
        finish(VK_SUCCESS);
        return candidate;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}


// ============================ Proof stage ============================

VkResult DeformPipeline::Candidate::Advance(DeformSemantic* semantic) {
    auto& s = *state_;
    using Stage = State::Stage;
    if (semantic) *semantic = DeformSemantic::Ok;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (s.stage == Stage::Failed) return s.failure;
    if (s.stage != Stage::Proof) {
        if (semantic && s.proved) *semantic = static_cast<DeformSemantic>(s.semantic);
        return VK_SUCCESS;
    }
    auto& ps = s.proof;
    Native* const native = s.native.get();
    auto const& context = native->context;
    VkDevice const d = context->device();
    VulkanQueueTimeline* const timeline = context->computeTimeline();
    auto fail = [&](VkResult r) {
        s.stage = Stage::Failed;
        s.failure = r;
        return r;
    };
    uint32_t const points = s.pointCount;
    VkResult r = VK_SUCCESS;
    if (ps.useHetero && !ps.suffixDone) {
        // The staged CVs (or the host-visible inputs, mapped directly), then
        // the suffix on the shared scheduler while the prefix proof runs. A
        // failed map runs the quarantine path like any proof failure: no
        // output escapes.
        void* pointsMap = nullptr;
        if (ps.stageCopy) {
            r = timeline->Check(ps.copyValue);
            if (r == VK_NOT_READY) return r;
        } else {
            r = vkMapMemory(d, s.inputs.points->memory(), 0, ps.pointsBytes, 0, &pointsMap);
        }
        if (r == VK_SUCCESS) {
            void* warpMap = nullptr;
            r = vkMapMemory(d, s.warped->memory(), 0, ps.outBytes, 0, &warpMap);
            if (r == VK_SUCCESS) {
                DeformEvalCpuParams xp;
                xp.samples = ps.samplesNorm;
                xp.coef = ps.coef;
                xp.n = ps.n; xp.m = ps.m;
                xp.cx = ps.center[0];
                xp.cy = ps.center[1];
                xp.cz = ps.center[2];
                xp.invScale = ps.invScale;
                xp.scale = ps.scale;
                // Staged copies read back out of the warped suffix; direct
                // points read the inputs in place (same bytes, same
                // absolute suffix range).
                float const* suffixCvs = ps.stageCopy
                    ? static_cast<float const*>(warpMap)
                    : static_cast<float const*>(pointsMap);
                auto tCpu0 = std::chrono::steady_clock::now();
                bool suffixOk = RunHeteroSuffix(xp, suffixCvs, static_cast<float*>(warpMap),
                                                ps.cacheActive, points, &ps.heteroFlag);
                ps.heteroCpuNs = std::chrono::duration<double, std::nano>(
                    std::chrono::steady_clock::now() - tCpu0).count();
                vkUnmapMemory(d, s.warped->memory());
                if (!suffixOk) r = VK_ERROR_UNKNOWN;
            }
            if (pointsMap) vkUnmapMemory(d, s.inputs.points->memory());
        }
        ps.suffixDone = true;
        if (r != VK_SUCCESS) {
            s.PutProof();
            Quarantine();
            return fail(r);
        }
    }
    r = timeline->Check(ps.proofValue);
    if (r == VK_NOT_READY) return r;
    // Submit-to-observed: the GPU prefix (plus the GPU suffix on direct
    // poses) regardless of host overlap. Drives the hetero policy.
    double const proofNs = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - ps.submitted).count();
    // The proof is observed: the proof state is this stage's again, and the
    // hold puts it back on every exit below.
    Native::ProofHold proof(native);
    proof.Adopt(ps.owned);
    ps.owned = nullptr;
    if (r != VK_SUCCESS) { Quarantine(); return fail(r); }
    s.pending = false;
    uint32_t st = 0;
    {
        void* data = nullptr;
        r = vkMapMemory(d, s.evalStatus->memory(), 0, 4, 0, &data);
        if (r == VK_SUCCESS) {
            std::memcpy(&st, data, 4);
            vkUnmapMemory(d, s.evalStatus->memory());
        }
    }
    st |= ps.heteroFlag;
    // The indirect args double as the miss signal: nonzero x means the
    // refill ran (forced or by mismatch). A free 12B read on the
    // already-synchronized proof path.
    uint32_t argsX = 0;
    if (ps.useCache && r == VK_SUCCESS) {
        void* data = nullptr;
        r = vkMapMemory(d, s.fillArgs->memory(), 0, kFillArgsBytes, 0, &data);
        if (r == VK_SUCCESS) {
            std::memcpy(&argsX, data, 4);
            vkUnmapMemory(d, s.fillArgs->memory());
        }
    }
    if (r != VK_SUCCESS) return fail(r);
    if (ps.useCache) {
        // Still owning the proof state: commit the force (R/proof now
        // hold this pose's fill) and account the hit/miss. The commit is
        // correct even for a NonFinite pose (the fill ran regardless;
        // NaN inputs fill NaN words the cached evaluator flags on).
        auto& cache = proof->evalCache;
        uint32_t const cacheActive = ps.cacheActive;
        if (ps.forceFill) {
            cache.rest = s.inputs.restSamples;
            cache.n = ps.n;
            cache.filled = cacheActive;
            cache.everFilled = true;
        }
        if (argsX != 0) {
            evalCacheMisses.fetch_add(1, std::memory_order_relaxed);
            if (++cache.missStreak >= kEvalCacheBypassMisses) {
                cache.bypassLeft = kEvalCacheBypassPoses;
                cache.missStreak = 0;
            }
        } else {
            evalCacheHits.fetch_add(1, std::memory_order_relaxed);
            cache.missStreak = 0;
        }
        if (ps.heteroEligible) {
            // Hetero policy, still owning the proof state. Hit-only
            // proof EMAs: fills
            // pollute the proof wait, while the CPU suffix measures
            // clean every hetero pose. A GPU hit against a known prefix
            // derives the device suffix rate; until then the seed
            // decides, and the first hetero pose corrects it.
            auto& hp = proof->hetero;
            bool const hit = (argsX == 0);
            uint32_t const suffixCvs = points - cacheActive;
            auto ema = [](double oldV, double newV) {
                return oldV <= 0.0 ? newV
                    : oldV + kHeteroEmaAlpha * (newV - oldV);
            };
            if (ps.useHetero) {
                if (suffixCvs > 0)
                    hp.cpuNsPerCv = ema(hp.cpuNsPerCv, ps.heteroCpuNs *
                        double(kHeteroWorkers) / double(suffixCvs));
                hp.everProbed = true;
                if (hit) {
                    hp.prefixProofNs = ema(hp.prefixProofNs, proofNs);
                    hp.prefixProofCvs = cacheActive;
                }
                double const cpuEst = double(suffixCvs) * hp.cpuNsPerCv /
                        double(kHeteroWorkers) +
                    kHeteroFixedOverheadNs;
                double const gpuSuffix = hp.gpuNsPerCv > 0.0
                    ? double(suffixCvs) * hp.gpuNsPerCv
                    : double(suffixCvs) * kHeteroSeedGpuDirectNsPerCv;
                if (cpuEst < gpuSuffix + hp.prefixProofNs) {
                    hp.armed = true;
                    hp.probeInterval = kHeteroProbeBase;
                } else {
                    hp.armed = false;
                    hp.probeIn = hp.probeInterval;
                    if (hp.probeInterval < kHeteroProbeMax)
                        hp.probeInterval *= 2;
                }
            } else {
                if (hp.probeIn > 0) --hp.probeIn;
                if (hit && hp.prefixProofNs > 0.0 && suffixCvs > 0) {
                    double const gpuSuffix = proofNs - hp.prefixProofNs;
                    if (gpuSuffix > 0.0)
                        hp.gpuNsPerCv = ema(hp.gpuNsPerCv,
                            gpuSuffix / double(suffixCvs));
                }
            }
        }
        // End of the cached proof phase; the apply submit below (and
        // its beforeSubmit hook) runs without the proof state, as before.
        proof.Put();
    }
    if (st != 0) {
        // A bad evaluation: rejected before any apply work is scheduled.
        s.stage = Stage::Done;
        s.proved = true;
        s.semantic = static_cast<uint32_t>(DeformSemantic::NonFinite);
        if (semantic) *semantic = DeformSemantic::NonFinite;
        return VK_SUCCESS;
    }
    // cmd1 was recorded before the eval submit (see Launch); the submit
    // below still gates on the proof and beforeSubmit.
    if (ps.beforeSubmit) {
        bool admitted = false;
        try { admitted = ps.beforeSubmit(); }
        catch (...) { return fail(VK_ERROR_UNKNOWN); }
        if (!admitted) return fail(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    s.pending = true;
    s.applyValue = timeline->Submit(&ps.applyCmd, 1u, &r);
    if (!s.applyValue) { Quarantine(); return fail(r); }
    s.stage = Stage::Apply;
    return VK_SUCCESS;
}

std::unique_ptr<DeformPipeline::Candidate> DeformPipeline::Begin(
    BeginInfo info, VkResult* result, DeformSemantic* semantic, BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    auto candidate = Launch(std::move(info), result, semantic, std::move(beforeSubmit));
    if (!candidate || candidate->state_->stage != Candidate::State::Stage::Proof)
        return candidate;
    // The synchronous contract: wait the proof on the timeline (a kernel
    // wait, no spin) and advance until the apply is submitted.
    VulkanQueueTimeline* const timeline = native_->context->computeTimeline();
    for (;;) {
        DeformSemantic sem = DeformSemantic::Ok;
        VkResult r = candidate->Advance(&sem);
        if (r == VK_NOT_READY) {
            r = timeline->Wait(candidate->state_->ProofWaitValue(), kProofTimeoutNs);
            if (r == VK_SUCCESS) continue;
            candidate->state_->PutProof();
            candidate->Quarantine();
            finish(r);
            return {};
        }
        if (r != VK_SUCCESS) { finish(r); return {}; }
        if (sem != DeformSemantic::Ok) {
            if (semantic) *semantic = sem;
            finish(VK_ERROR_INITIALIZATION_FAILED);
            return {};
        }
        finish(VK_SUCCESS);
        return candidate;
    }
}

} // namespace usdGen::vulkan
