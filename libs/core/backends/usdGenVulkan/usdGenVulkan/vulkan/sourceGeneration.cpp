// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "sourceGeneration.h"
#include "sourceValidation.h"
#include <new>

#include <cmath>
#include <cstring>
#include <limits>

namespace usdGen::vulkan {
struct VulkanSourceUpload::Storage {
    struct Plane {
        UsdGenDeviceChannelMetadata meta;
        // Staging is retained/fenced with its paired immutable device-local
        // output; neither allocation can be freed after an unproven submit.
        bool privateFrame = false;
        std::shared_ptr<ChargedBuffer> staging;
        std::shared_ptr<ChargedBuffer> buffer;
    };
    std::shared_ptr<DeviceContext> context;
    std::vector<Plane> planes;
    std::vector<UsdGenChunkDesc> chunks;
    UsdGenDeviceGeometryMetadata geometry;
    uint64_t topology = 0, value = 0;
    uint32_t curves = 0, points = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::shared_ptr<const void> submission;
    SourceGenerationStatus status = SourceGenerationStatus::InvalidInput;
    bool submitted = false, quarantined = false;
    ~Storage() {
        if (quarantined) return;
        if (pool) vkDestroyCommandPool(context->device(), pool, nullptr);
        if (fence) vkDestroyFence(context->device(), fence, nullptr);
    }
};

VulkanSourceUpload::VulkanSourceUpload(std::shared_ptr<Storage> value) : storage_(std::move(value)) {}
VulkanSourceUpload::~VulkanSourceUpload() { if (storage_ && storage_->submitted) Quarantine(); }

static bool Add(VulkanSourceUpload::Storage& s, UsdGenDeviceChannelMetadata meta,
                void const* bytes, size_t count, std::string* why, VkResult* result,
                bool privateFrame = false) {
    if (!count) { s.planes.push_back({std::move(meta), privateFrame, {}, {}}); return true; }
    if (!bytes || count > std::numeric_limits<VkDeviceSize>::max()) {
        if (why) *why = "oversized source plane";
        if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
        return false;
    }
    VkBufferCreateInfo stagingInfo{}; stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; stagingInfo.size = count;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto staging = ChargedBuffer::Create(s.context, stagingInfo,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch, result);
    if (!staging) return false;
    void* mapped = nullptr;
    VkResult r = vkMapMemory(s.context->device(), staging->memory(), 0, count, 0, &mapped);
    if (r != VK_SUCCESS) { if (result) *result = r; return false; }
    std::memcpy(mapped, bytes, count); vkUnmapMemory(s.context->device(), staging->memory());
    VkBufferCreateInfo outputInfo{}; outputInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; outputInfo.size = count;
    outputInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    outputInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = ChargedBuffer::Create(s.context, outputInfo,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UsdGenExecutionResourceKind::Pinned, result);
    if (!buffer) return false;
    s.planes.push_back({std::move(meta), privateFrame, std::move(staging), std::move(buffer)}); return true;
}

std::shared_ptr<const VulkanPreparedSource> VulkanPreparedSource::Prepare(
    VulkanSourcePrepareInfo in, std::string* why) {
    try {
        auto prepared = std::shared_ptr<VulkanPreparedSource>(new VulkanPreparedSource);
        if (!PackSource(in, &prepared->planes_, &prepared->geometry_, why)) return {};
        prepared->chunks_ = std::move(in.source.chunks);
        prepared->topology_ = in.source.topologyVersion;
        prepared->value_ = in.source.valueVersion;
        prepared->curves_ = in.source.totalCurves;
        prepared->points_ = in.source.totalCvs;
        return prepared;
    } catch (std::bad_alloc const&) {
        if (why) *why = "source preparation allocation failed";
        return {};
    }
}

std::shared_ptr<VulkanSourceUpload> VulkanSourceUpload::Create(
    VulkanSourceGenerationCreateInfo in, VkResult* result, std::string* why) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!in.context) { if (why) *why = "missing source capture context"; return {}; }
    VulkanSourcePrepareInfo prepare{std::move(in.source), std::move(in.geometry), std::move(in.additionalNamed)};
    auto prepared = VulkanPreparedSource::Prepare(std::move(prepare), why);
    if (!prepared) return {};
    return Create(std::move(in.context), std::move(prepared), result, why);
}

std::shared_ptr<VulkanSourceUpload> VulkanSourceUpload::Create(std::shared_ptr<DeviceContext> context,
    std::shared_ptr<const VulkanPreparedSource> prepared, VkResult* result, std::string* why) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context || !prepared) { if (why) *why = "missing prepared source upload input"; return {}; }
    try {
        auto s = std::make_shared<Storage>();
        s->context = std::move(context);
        s->geometry = prepared->geometry();
        s->chunks = prepared->chunks();
        s->topology = prepared->topologyVersion();
        s->value = prepared->valueVersion();
        s->curves = prepared->curveCount();
        s->points = prepared->pointCount();
        s->planes.reserve(prepared->planes().size());
        for (auto const& plane : prepared->planes()) {
            if (!Add(*s, plane.metadata, plane.bytes.data(), plane.bytes.size(), why, result))
                return {};
            s->planes.back().privateFrame = plane.privateFrame;
        }
        VkCommandPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.queueFamilyIndex = s->context->computeQueueFamily();
        auto r = vkCreateCommandPool(s->context->device(), &pool, nullptr, &s->pool);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkCommandBufferAllocateInfo command{};
        command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command.commandPool = s->pool;
        command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command.commandBufferCount = 1;
        r = vkAllocateCommandBuffers(s->context->device(), &command, &s->command);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(s->context->device(), &fence, nullptr, &s->fence);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        s->status = SourceGenerationStatus::Created;
        auto upload = std::shared_ptr<VulkanSourceUpload>(new VulkanSourceUpload(std::move(s)));
        if (result) *result = VK_SUCCESS;
        return upload;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}
SourceGenerationStatus VulkanSourceUpload::Submit() noexcept {
    if (!storage_ || storage_->status != SourceGenerationStatus::Created) return SourceGenerationStatus::InvalidInput;
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(storage_->command, &begin) != VK_SUCCESS) { storage_->status = SourceGenerationStatus::QueueRejected; return storage_->status; }
    for (auto const& plane : storage_->planes) {
        if (!plane.buffer) continue;
        VkBufferCopy copy{}; copy.size = plane.buffer->sizeBytes();
        vkCmdCopyBuffer(storage_->command, plane.staging->buffer(), plane.buffer->buffer(), 1, &copy);
    }
    VkMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(storage_->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(storage_->command) != VK_SUCCESS) { storage_->status = SourceGenerationStatus::QueueRejected; return storage_->status; }
    // Enter pending state before arming any native owner: every later failure
    // is conservatively quarantined even when vkQueueSubmit itself rejects.
    storage_->submission = storage_; storage_->submitted = true;
    storage_->status = SourceGenerationStatus::Submitted;
    for (auto const& plane : storage_->planes) {
        if (!plane.buffer) continue;
        if (plane.staging->MarkSubmitted(storage_->fence, storage_->submission) != VK_SUCCESS ||
            plane.buffer->MarkSubmitted(storage_->fence, storage_->submission) != VK_SUCCESS) {
            Quarantine(); return SourceGenerationStatus::LostProof;
        }
    }
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &storage_->command;
    if (vkQueueSubmit(storage_->context->computeQueue(), 1, &submit, storage_->fence) != VK_SUCCESS) {
        Quarantine(); return SourceGenerationStatus::LostProof;
    }
    return storage_->status;
}
SourceGenerationStatus VulkanSourceUpload::Poll() noexcept {
    if (!storage_) return SourceGenerationStatus::InvalidInput;
    if (storage_->status == SourceGenerationStatus::Ready) return SourceGenerationStatus::Ready;
    if (storage_->status != SourceGenerationStatus::Submitted) return SourceGenerationStatus::InvalidInput;
    VkResult r = vkGetFenceStatus(storage_->context->device(), storage_->fence);
    if (r == VK_NOT_READY) return SourceGenerationStatus::NotReady;
    if (r != VK_SUCCESS) { Quarantine(); return SourceGenerationStatus::LostProof; }
    // Keep every staging owner until all proof checks finish: quarantine must
    // still be able to retain a complete packet if any later owner fails.
    for (auto const& plane : storage_->planes) {
        if (!plane.buffer) continue;
        if (plane.staging->PollComplete() != VK_SUCCESS || plane.buffer->PollComplete() != VK_SUCCESS) {
            Quarantine(); return SourceGenerationStatus::LostProof;
        }
    }
    for (auto& plane : storage_->planes) plane.staging.reset();
    storage_->submitted = false;
    storage_->submission.reset();
    storage_->status = SourceGenerationStatus::Ready;
    return storage_->status;
}
void VulkanSourceUpload::Quarantine() noexcept {
    if (!storage_ || storage_->status != SourceGenerationStatus::Submitted ||
        storage_->quarantined) return;
    storage_->quarantined = true;
    for (auto const& plane : storage_->planes) {
        if (!plane.buffer) continue;
        plane.staging->Quarantine();
        plane.buffer->Quarantine();
    }
    storage_->submission.reset();
    storage_->status = SourceGenerationStatus::LostProof;
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceUpload::TakeReady() noexcept {
    if (!storage_ || storage_->status != SourceGenerationStatus::Ready) return {};
    try {
        return std::shared_ptr<const VulkanSourceGeneration>(new VulkanSourceGeneration(storage_));
    } catch (std::bad_alloc const&) {
        return {};
    }
}

VulkanSourceGeneration::VulkanSourceGeneration(
    std::shared_ptr<const VulkanSourceUpload::Storage> storage)
    : storage_(std::move(storage)) {
    planes_.reserve(storage_->planes.size());
    for (auto const& plane : storage_->planes) {
        PlaneView view{plane.meta, plane.buffer ? plane.buffer->buffer() : VK_NULL_HANDLE,
                       plane.buffer ? plane.buffer->sizeBytes() : 0};
        if (plane.privateFrame) sourceFrames_.push_back(std::move(view));
        else planes_.push_back(std::move(view));
    }
}

uint64_t VulkanSourceGeneration::topologyVersion() const noexcept { return base_ ? childGeometry_.topologyVersion : storage_->topology; }
VulkanSourceGeneration::VulkanSourceGeneration(
    std::shared_ptr<const VulkanSourceGeneration> base, uint64_t value)
    : base_(std::move(base)), valueVersion_(value), childGeometry_(base_->geometry()),
      planes_(base_->planes()) {
    childGeometry_.valueVersion = value;
}
std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithValueVersion(
    std::shared_ptr<const VulkanSourceGeneration> const& base, uint64_t value) {
    if (!base || value <= base->valueVersion()) return {};
    try {
        return std::shared_ptr<const VulkanSourceGeneration>(new VulkanSourceGeneration(base, value));
    } catch (...) { return {}; }
}
std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithTopologyVersion(
    std::shared_ptr<const VulkanSourceGeneration> const& base, uint64_t value) {
    if (!base || value <= base->valueVersion()) return {};
    try {
        auto child = std::shared_ptr<VulkanSourceGeneration>(new VulkanSourceGeneration(base, value));
        child->childGeometry_.topologyVersion = value;
        child->nonWidthIdentity_ = std::make_shared<uint8_t>(0);
        return child;
    } catch (...) { return {}; }
}
uint64_t VulkanSourceGeneration::valueVersion() const noexcept { return base_ ? valueVersion_ : storage_->value; }
std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithDeformAlias(
    std::shared_ptr<const VulkanSourceGeneration> const& base, uint64_t value) {
    if (!base || value <= base->valueVersion()) return {};
    try {
        auto child = std::shared_ptr<VulkanSourceGeneration>(new VulkanSourceGeneration(base, value));
        child->childGeometry_.alreadyDeformed = true;
        child->nonWidthIdentity_ = std::make_shared<uint8_t>(0);
        return child;
    } catch (...) { return {}; }
}
uint32_t VulkanSourceGeneration::curveCount() const noexcept { return compacted_ ? compactCurves_ : (base_ ? base_->curveCount() : storage_->curves); }
uint32_t VulkanSourceGeneration::pointCount() const noexcept { return compacted_ ? compactPoints_ : (base_ ? base_->pointCount() : storage_->points); }
size_t VulkanSourceGeneration::ExclusiveRetainedBytes() const noexcept {
    if (compacted_) {
        size_t bytes = 0;
        for (size_t i = 0; i != compactOwners_.size(); ++i) {
            auto const& owner = compactOwners_[i].second;
            if (!owner) continue;
            bool duplicate = false;
            for (size_t j = 0; j != i; ++j)
                duplicate = duplicate || compactOwners_[j].second.get() == owner.get();
            if (!duplicate) {
                auto const inheritedPublic = base_->PlaneOwner(compactOwners_[i].first);
                auto const inheritedPrivate = base_->SourceFrameOwner(compactOwners_[i].first);
                if (owner != inheritedPublic && owner != inheritedPrivate)
                    bytes += size_t(owner->allocationBytes());
            }
        }
        return bytes;
    }
    if (base_) return (widthOverride_ ? size_t(widthOverride_->allocationBytes()) : 0) +
        (pointsOverride_ ? size_t(pointsOverride_->allocationBytes()) : 0);
    size_t bytes = 0;
    for (auto const& plane : storage_->planes)
        if (plane.buffer) bytes += size_t(plane.buffer->allocationBytes());
    return bytes;
}
size_t VulkanSourceGeneration::InclusiveRetainedBytes() const noexcept {
    return ExclusiveRetainedBytes() + (base_ ? base_->InclusiveRetainedBytes() : 0);
}
std::shared_ptr<DeviceContext> VulkanSourceGeneration::context() const noexcept { return base_ ? base_->context() : storage_->context; }
UsdGenDeviceGeometryMetadata const& VulkanSourceGeneration::geometry() const noexcept {
    return (base_ || compacted_) ? childGeometry_ : storage_->geometry;
}
std::vector<UsdGenChunkDesc> const& VulkanSourceGeneration::chunks() const noexcept {
    return compacted_ ? compactChunks_ : (base_ ? base_->chunks() : storage_->chunks);
}
std::vector<VulkanSourceGeneration::PlaneView> const& VulkanSourceGeneration::planes() const noexcept {
    return planes_;
}
std::vector<VulkanSourceGeneration::PlaneView> const& VulkanSourceGeneration::sourceFrames() const noexcept {
    return compacted_ ? compactFrames_ : (base_ ? base_->sourceFrames() : sourceFrames_);
}
std::shared_ptr<const ChargedBuffer> VulkanSourceGeneration::PlaneOwner(std::string const& name) const noexcept {
    if (base_) {
        if (compacted_) {
            for (auto const& plane : planes_)
                if (plane.metadata.name == name)
                    for (auto const& item : compactOwners_)
                        if (item.first == name) return item.second;
            return {};
        }
        if (name == "width" && widthOverride_) return widthOverride_;
        if (name == "points" && pointsOverride_) return pointsOverride_;
        return base_->PlaneOwner(name);
    }
    for (auto const& plane : storage_->planes)
        if (!plane.privateFrame && plane.meta.name == name) return plane.buffer;
    return {};
}
std::shared_ptr<const ChargedBuffer> VulkanSourceGeneration::SourceFrameOwner(std::string const& name) const noexcept {
    if (compacted_) {
        for (auto const& plane : compactFrames_)
            if (plane.metadata.name == name)
                for (auto const& item : compactOwners_)
                    if (item.first == name) return item.second;
        return {};
    }
    if (base_) return base_->SourceFrameOwner(name);
    if (!compacted_ && !base_)
        for (auto const& plane : sourceFrames_)
            if (plane.metadata.name == name) {
                for (auto const& item : storage_->planes)
                    if (item.privateFrame && item.meta.name == name) return item.buffer;
            }
    return {};
}
void const* VulkanSourceGeneration::NonWidthIdentity() const noexcept {
    if (nonWidthIdentity_) return nonWidthIdentity_.get();
    if (base_) return base_->NonWidthIdentity();
    return storage_.get();
}
VulkanSourceGeneration::VulkanSourceGeneration(std::shared_ptr<const VulkanSourceGeneration> base,
    std::shared_ptr<const ChargedBuffer> points, uint64_t value, bool)
    : base_(std::move(base)), pointsOverride_(std::move(points)), valueVersion_(value) {
    // This marker is deliberately independent of pointsOverride_: an empty
    // point-changing child still severs joins with its points ancestor.
    nonWidthIdentity_ = std::make_shared<uint8_t>(0);
    childGeometry_ = base_->geometry(); childGeometry_.valueVersion = value;
    // Point edits make captured bounds stale.  They remain range-valid but are
    // never exposed as conservative bounds until a later producer recomputes them.
    for (auto& tile : childGeometry_.tiles) tile.boundsValid = false;
    // Length scaling changes the sampled points, but it does not apply scalp
    // motion.  Preserve the captured deformation provenance from the base.
    childGeometry_.alreadyDeformed = base_->geometry().alreadyDeformed;
    planes_ = base_->planes();
    bool replaced = false;
    for (auto& plane : planes_) if (plane.metadata.name == "points") {
        plane.buffer = pointsOverride_ ? pointsOverride_->buffer() : VK_NULL_HANDLE;
        plane.bytes = pointsOverride_ ? pointsOverride_->sizeBytes() : 0;
        replaced = true;
    }
    if (!replaced && !base_->pointCount()) planes_.push_back({{"points", UsdGenDeviceValueType::Float32x3,
        UsdGenDeviceDomain::Point, 0, 3, 12, true, UsdGenDeviceChannelSemantic::Points}, VK_NULL_HANDLE, 0});
}
VulkanSourceGeneration::VulkanSourceGeneration(std::shared_ptr<const VulkanSourceGeneration> base,
    std::shared_ptr<const ChargedBuffer> width, uint64_t value)
    : base_(std::move(base)), widthOverride_(std::move(width)), valueVersion_(value) {
    childGeometry_ = base_->geometry(); childGeometry_.valueVersion = value;
    planes_ = base_->planes();
    bool replaced = false;
    for (auto& plane : planes_) if (plane.metadata.name == "width") {
        plane.buffer = widthOverride_ ? widthOverride_->buffer() : VK_NULL_HANDLE;
        plane.bytes = widthOverride_ ? widthOverride_->sizeBytes() : 0;
        replaced = true;
    }
    if (!replaced && !base_->pointCount()) planes_.push_back({{"width", UsdGenDeviceValueType::Float32,
        UsdGenDeviceDomain::Point, 0, 1, 4, true, UsdGenDeviceChannelSemantic::Widths}, VK_NULL_HANDLE, 0});
}
std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithWidth(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    WidthPipeline::Candidate const& candidate, uint64_t value, std::string* why) {
    if (!base || value <= base->valueVersion() || !candidate.succeeded() ||
        candidate.count() != base->pointCount() || candidate.context() != base->context() ||
        candidate.inputOwner() != base->PlaneOwner("width")) {
        if (why) *why = "invalid width COW proof";
        return {};
    }
    auto width = candidate.output();
    if (base->pointCount()) {
        if (!width || width == base->PlaneOwner("width") || width->context() != base->context() ||
            width->sizeBytes() != VkDeviceSize(base->pointCount()) * sizeof(float) ||
            !(width->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || !(width->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
            if (why) *why = "invalid width COW output";
            return {};
        }
    } else if (width) { if (why) *why="empty width COW has output"; return {}; }
    try { return std::shared_ptr<const VulkanSourceGeneration>(new VulkanSourceGeneration(base, std::move(width), value)); }
    catch (std::bad_alloc const&) { if (why) *why="width COW allocation failed"; return {}; }
}
std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithExpressionWidth(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    ExprVkWidthApplyPipeline::Candidate const& candidate, uint64_t value, std::string* why) {
    auto const width = candidate.output();
    if (!base || value <= base->valueVersion() || !candidate.succeeded() ||
        candidate.context() != base->context() ||
        candidate.inputOwner() != base->PlaneOwner("width") ||
        candidate.pointCount() != base->pointCount() || candidate.curveCount() != base->curveCount() ||
        (base->pointCount() && (!width || width == base->PlaneOwner("width") ||
            width->sizeBytes() != VkDeviceSize(base->pointCount()) * sizeof(float) ||
            !(width->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
            !(width->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT))) ||
        (!base->pointCount() && width)) {
        if (why) *why = "invalid expression width COW proof";
        return {};
    }
    try { return std::shared_ptr<const VulkanSourceGeneration>(new VulkanSourceGeneration(base, width, value)); }
    catch (std::bad_alloc const&) { if (why) *why = "expression width COW allocation failed"; return {}; }
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithGrown(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    CurveGrowPipeline::Candidate const& grow, ResampleVkPipeline::Candidate const& named,
    uint32_t target, uint64_t value, std::string* why) {
    if (!base || !grow.succeeded() || grow.context() != base->context() ||
        grow.inputOwner() != base->PlaneOwner("points") ||
        grow.curveCount() != base->curveCount() ||
        uint64_t(grow.pointCount()) != uint64_t(base->curveCount()) * target ||
        !named.succeeded() || named.inputOwner() != base || named.target() != target ||
        named.context() != base->context() || value <= base->valueVersion()) {
        if (why) *why = "invalid Grow geometry proof";
        return {};
    }
    try {
        auto child = std::shared_ptr<VulkanSourceGeneration>(
            new VulkanSourceGeneration(base, named, target, value));
        child->childGeometry_.curveTopology = {UsdGenDeviceCurveType::Cubic,
                                               UsdGenDeviceCurveBasis::BSpline,
                                               UsdGenDeviceCurveWrap::Pinned};
        if (!base->curveCount()) return child;
        using T = UsdGenDeviceValueType; using D = UsdGenDeviceDomain;
        using S = UsdGenDeviceChannelSemantic;
        auto replace = [&](char const* name, T type, D domain, uint32_t arity, S semantic,
                           std::shared_ptr<const ChargedBuffer> owner, bool privateFrame = false) {
            auto& views = privateFrame ? child->compactFrames_ : child->planes_;
            uint64_t count = domain == D::Point ? grow.pointCount() :
                domain == D::Topology ? uint64_t(grow.curveCount()) + 1 : grow.curveCount();
            PlaneView view{{name, type, domain, count, arity,
                           (type == T::UInt64 ? 8u : 4u) * arity, true, semantic},
                           owner ? owner->buffer() : VK_NULL_HANDLE,
                           owner ? owner->sizeBytes() : 0};
            bool found = false;
            for (auto& item : views) if (item.metadata.name == name) { item = view; found = true; }
            if (!found) views.push_back(std::move(view));
            found = false;
            for (auto& item : child->compactOwners_) if (item.first == name) {
                item.second = owner; found = true;
            }
            if (!found) child->compactOwners_.emplace_back(name, std::move(owner));
        };
        replace("points", T::Float32x3, D::Point, 3, S::Points, grow.output());
        replace("rest", T::Float32x3, D::Point, 3, S::RestPoints, grow.rest());
        replace("width", T::Float32, D::Point, 1, S::Widths, grow.widths());
        replace("hairT", T::Float32, D::Point, 1, S::HairT, grow.hairT());
        replace("curveOffsets", T::UInt32, D::Topology, 1, S::CurveOffsets, grow.offsets());
        replace("stableIds", T::UInt64, D::Primitive, 1, S::StableIds, grow.ids());
        replace("rootPrim", T::Int32, D::Primitive, 1, S::RootPrim, grow.rootPrim());
        replace("rootUV", T::Float32x2, D::Primitive, 2, S::RootUV, grow.rootUV());
        replace("sourceRootT", T::Float32x3, D::Primitive, 3, S::Generic, grow.rootT(), true);
        replace("sourceRootB", T::Float32x3, D::Primitive, 3, S::Generic, grow.rootB(), true);
        replace("sourceRootN", T::Float32x3, D::Primitive, 3, S::Generic, grow.rootN(), true);
        return child;
    } catch (std::bad_alloc const&) { if (why) *why = "Grow generation allocation failed"; return {}; }
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithPoints(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    LengthScalePipeline::Candidate const& candidate, uint64_t value, std::string* why) {
    if (!base || value <= base->valueVersion() || !candidate.succeeded() ||
        candidate.count() != base->pointCount() || candidate.curveCount() != base->curveCount() ||
        candidate.context() != base->context() || candidate.inputOwner() != base->PlaneOwner("points") ||
        candidate.offsetsOwner() != base->PlaneOwner("curveOffsets") ||
        (candidate.usesHairT() && candidate.hairTOwner() != base->PlaneOwner("hairT")) ||
        (candidate.usesStableIds() && candidate.stableIdsOwner() != base->PlaneOwner("stableIds"))) {
        if (why) *why = "invalid points COW proof";
        return {};
    }
    auto points = candidate.output();
    if (base->pointCount()) {
        if (!points || points == base->PlaneOwner("points") || points->context() != base->context() ||
            points->sizeBytes() != VkDeviceSize(base->pointCount()) * 3 * sizeof(float) ||
            !(points->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
            !(points->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
            if (why) *why = "invalid points COW output";
            return {};
        }
    } else if (points) { if (why) *why = "empty points COW has output"; return {}; }
    try { return std::shared_ptr<const VulkanSourceGeneration>(new VulkanSourceGeneration(base, std::move(points), value, true)); }
    catch (std::bad_alloc const&) { if (why) *why = "points COW allocation failed"; return {}; }
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithNoise(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    NoisePipeline::Candidate const& candidate, uint64_t value, std::string* why) {
    // Noise preserves topology: only point count and context must match.
    if (!base || value <= base->valueVersion() || !candidate.succeeded() ||
        candidate.pointCount() != base->pointCount() ||
        candidate.context() != base->context()) {
        if (why) *why = "invalid noise COW proof";
        return {};
    }
    auto points = candidate.output().points;
    if (base->pointCount()) {
        if (!points || points->context() != base->context() ||
            points->sizeBytes() != VkDeviceSize(base->pointCount()) * 3 * sizeof(float) ||
            !(points->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
            !(points->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
            if (why) *why = "invalid noise COW output";
            return {};
        }
    } else if (points) { if (why) *why = "empty noise COW has output"; return {}; }
    try { return std::shared_ptr<const VulkanSourceGeneration>(new VulkanSourceGeneration(base, std::move(points), value, true)); }
    catch (std::bad_alloc const&) { if (why) *why = "noise COW allocation failed"; return {}; }
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithDeform(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    DeformPipeline::Candidate const& candidate, uint64_t value, std::string* why) {
    // Deform preserves topology and must prove the exact immutable point input.
    if (!base || value <= base->valueVersion() || !candidate.succeeded() ||
        candidate.pointCount() != base->pointCount() ||
        candidate.context() != base->context() ||
        candidate.inputOwner() != base->PlaneOwner("points")) {
        if (why) *why = "invalid deform COW proof";
        return {};
    }
    auto points = candidate.output().points;
    if (base->pointCount()) {
        if (!points || points->context() != base->context() ||
            points->sizeBytes() != VkDeviceSize(base->pointCount()) * 3 * sizeof(float) ||
            !(points->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
            !(points->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
            if (why) *why = "invalid deform COW output";
            return {};
        }
    } else if (points) { if (why) *why = "empty deform COW has output"; return {}; }
    try {
        auto child = std::shared_ptr<VulkanSourceGeneration>(
            new VulkanSourceGeneration(base, std::move(points), value, true));
        child->childGeometry_.alreadyDeformed = true;
        return child;
    } catch (std::bad_alloc const&) { if (why) *why = "deform COW allocation failed"; return {}; }
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithDeform(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    RbfVkDeformPipeline::Candidate const& candidate, uint64_t value, std::string* why) {
    // Deform preserves topology and must prove the exact immutable point input.
    if (!base || value <= base->valueVersion() || !candidate.succeeded() ||
        candidate.pointCount() != base->pointCount() ||
        candidate.context() != base->context() ||
        candidate.inputOwner() != base->PlaneOwner("points")) {
        if (why) *why = "invalid deform COW proof";
        return {};
    }
    auto points = candidate.output().points;
    if (base->pointCount()) {
        if (!points || points->context() != base->context() ||
            points->sizeBytes() != VkDeviceSize(base->pointCount()) * 3 * sizeof(float) ||
            !(points->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
            !(points->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
            if (why) *why = "invalid deform COW output";
            return {};
        }
    } else if (points) { if (why) *why = "empty deform COW has output"; return {}; }
    try {
        auto child = std::shared_ptr<VulkanSourceGeneration>(
            new VulkanSourceGeneration(base, std::move(points), value, true));
        child->childGeometry_.alreadyDeformed = true;
        return child;
    } catch (std::bad_alloc const&) { if (why) *why = "deform COW allocation failed"; return {}; }
}

VulkanSourceGeneration::VulkanSourceGeneration(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    LengthCompactionPipeline::Candidate const& candidate, uint64_t value)
    : base_(base), compacted_(true), valueVersion_(value) {
    compactCurves_ = candidate.curveCount();
    compactPoints_ = candidate.pointCount();
    compactChunks_ = candidate.chunks();
    compactTiles_ = candidate.tiles();
    childGeometry_ = base->geometry();
    childGeometry_.topologyVersion = value;
    childGeometry_.valueVersion = value;
    childGeometry_.curveCount = compactCurves_;
    childGeometry_.pointCount = compactPoints_;
    childGeometry_.tiles = compactTiles_;
    for (auto& tile : childGeometry_.tiles) tile.boundsValid = false;
    childGeometry_.alreadyDeformed = base->geometry().alreadyDeformed;
    nonWidthIdentity_ = std::make_shared<uint8_t>(0);
    planes_.reserve(candidate.outputs().size());
    for (auto const& output : candidate.outputs()) {
        compactOwners_.emplace_back(output.metadata.name, output.owner);
        PlaneView view{output.metadata, output.owner ? output.owner->buffer() : VK_NULL_HANDLE,
                       output.owner ? output.bytes : 0};
        if (output.privateFrame) compactFrames_.push_back(std::move(view));
        else planes_.push_back(std::move(view));
    }
}

VulkanSourceGeneration::VulkanSourceGeneration(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    ResampleVkPipeline::Candidate const& candidate, uint32_t, uint64_t value)
    : base_(base), compacted_(true), valueVersion_(value) {
    // Same full-packet COW shape as the compaction child: every plane is
    // replaced (Groom planes alias the input owners).
    compactCurves_ = candidate.curveCount();
    compactPoints_ = candidate.pointCount();
    compactChunks_ = candidate.chunks();
    compactTiles_ = candidate.tiles();
    childGeometry_ = base->geometry();
    childGeometry_.topologyVersion = value;
    childGeometry_.valueVersion = value;
    childGeometry_.curveCount = compactCurves_;
    childGeometry_.pointCount = compactPoints_;
    childGeometry_.tiles = compactTiles_;
    for (auto& tile : childGeometry_.tiles) tile.boundsValid = false;
    childGeometry_.alreadyDeformed = base->geometry().alreadyDeformed;
    nonWidthIdentity_ = std::make_shared<uint8_t>(0);
    planes_.reserve(candidate.outputs().size());
    for (auto const& output : candidate.outputs()) {
        compactOwners_.emplace_back(output.metadata.name, output.owner);
        PlaneView view{output.metadata, output.owner ? output.owner->buffer() : VK_NULL_HANDLE,
                       output.owner ? output.bytes : 0};
        if (output.privateFrame) compactFrames_.push_back(std::move(view));
        else planes_.push_back(std::move(view));
    }
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithCompacted(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    LengthCompactionPipeline::Candidate const& candidate, uint64_t value,
    std::string* why, bool emptyGrow) {
    if (!base || !candidate.succeeded() || candidate.inputOwner() != base ||
        candidate.context() != base->context() || value <= base->valueVersion() ||
        candidate.curveCount() > base->curveCount() || candidate.pointCount() > base->pointCount() ||
        (emptyGrow && (candidate.curveCount() || candidate.pointCount()))) {
        if (why) *why = "invalid compacted COW proof";
        return {};
    }
    try {
        auto child = std::shared_ptr<VulkanSourceGeneration>(new VulkanSourceGeneration(base, candidate, value));
        // Disabled Grow lowers to a native empty compaction but retains the
        // generated topology contract instead of the predecessor's basis.
        if (emptyGrow)
            child->childGeometry_.curveTopology = {UsdGenDeviceCurveType::Cubic,
                                                   UsdGenDeviceCurveBasis::BSpline,
                                                   UsdGenDeviceCurveWrap::Pinned};
        return child;
    } catch (std::bad_alloc const&) {
        if (why) *why = "compacted COW allocation failed";
        return {};
    }
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithResampled(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    ResampleVkPipeline::Candidate const& candidate, uint32_t target, uint64_t value,
    std::string* reason) {
    uint64_t const want = uint64_t(base ? base->curveCount() : 0) * target;
    if (!base || target < 2 || want > UINT32_MAX || !candidate.succeeded() || candidate.inputOwner() != base ||
        candidate.context() != base->context() || value <= base->valueVersion() ||
        candidate.target() != target || candidate.curveCount() != base->curveCount() ||
        candidate.pointCount() != uint32_t(want)) {
        if (reason) *reason = "invalid resampled COW proof";
        return {};
    }
    try {
        return std::shared_ptr<const VulkanSourceGeneration>(new VulkanSourceGeneration(base, candidate, target, value));
    } catch (std::bad_alloc const&) {
        if (reason) *reason = "resampled COW allocation failed";
        return {};
    }
}

VulkanSourceGeneration::VulkanSourceGeneration(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    std::vector<UsdGenDeviceTileMetadata> tiles)
    : base_(base), valueVersion_(base->valueVersion()) {
    // Metadata-only alias: no plane or frame is replaced, so joins keep the
    // base non-width identity and versions are inherited unchanged.
    childGeometry_ = base_->geometry();
    childGeometry_.tiles = std::move(tiles);
    planes_ = base_->planes();
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithTiles(
    std::shared_ptr<const VulkanSourceGeneration> const& base,
    std::vector<UsdGenDeviceTileMetadata> tiles, std::string* why) {
    if (!base) { if (why) *why = "tile attach requires a base generation"; return {}; }
    uint64_t const curves = base->curveCount(), points = base->pointCount();
    uint64_t curveEnd = 0, pointEnd = 0, previousId = 0;
    bool first = true;
    for (auto const& tile : tiles) {
        if (!first && tile.tile <= previousId) {
            if (why) *why = "tile IDs must be strictly increasing";
            return {};
        }
        first = false;
        previousId = tile.tile;
        if (tile.curveCount > tile.pointCount ||
            (tile.curveCount == 0 && tile.pointCount != 0) ||
            tile.firstPoint > std::numeric_limits<uint32_t>::max() ||
            tile.pointCount > std::numeric_limits<uint32_t>::max() - tile.firstPoint ||
            tile.firstCurve != curveEnd || tile.firstPoint != pointEnd ||
            tile.curveCount > curves - curveEnd || tile.pointCount > points - pointEnd ||
            tile.curveCount > std::numeric_limits<uint64_t>::max() - curveEnd ||
            tile.pointCount > std::numeric_limits<uint64_t>::max() - pointEnd) {
            if (why) *why = "tile ranges must be contiguous and in bounds";
            return {};
        }
        if (!tile.boundsValid) { if (why) *why = "tile attach requires finalized bounds"; return {}; }
        for (size_t axis = 0; axis != 3; ++axis) {
            float const lo = tile.extentMin[axis], hi = tile.extentMax[axis];
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) {
                if (why) *why = "tile attach requires conservative bounds";
                return {};
            }
        }
        curveEnd += tile.curveCount;
        pointEnd += tile.pointCount;
    }
    if (tiles.empty()) {
        if (curves != 0 || points != 0) {
            if (why) *why = "non-empty generation requires tile metadata";
            return {};
        }
    } else if (curveEnd != curves || pointEnd != points) {
        if (why) *why = "tile metadata does not cover the generation";
        return {};
    }
    try {
        return std::shared_ptr<const VulkanSourceGeneration>(
            new VulkanSourceGeneration(base, std::move(tiles)));
    } catch (std::bad_alloc const&) {
        if (why) *why = "tile attach allocation failed";
        return {};
    }
}

std::shared_ptr<const VulkanSourceGeneration> VulkanSourceGeneration::WithWidthBlend(
    std::shared_ptr<const VulkanSourceGeneration> const& left,
    std::shared_ptr<const VulkanSourceGeneration> const& right,
    WidthBlendPipeline::Candidate const& candidate, uint64_t value,
    std::string* why, NonWidthComparePipeline::Candidate const* nonWidthProof) {
    bool const sameBundle = left && right && left->NonWidthIdentity() == right->NonWidthIdentity();
    bool const provedJoin = nonWidthProof && nonWidthProof->succeeded() &&
        nonWidthProof->leftOwner() == left && nonWidthProof->rightOwner() == right;
    if (!left || !right || (!sameBundle && !provedJoin) ||
        value <= left->valueVersion() || value <= right->valueVersion() ||
        !candidate.succeeded() || candidate.count() != left->pointCount() ||
        left->pointCount() != right->pointCount() || candidate.context() != left->context() ||
        left->context() != right->context() || candidate.leftOwner() != left->PlaneOwner("width") ||
        candidate.rightOwner() != right->PlaneOwner("width")) {
        if (why) *why = "invalid width blend COW proof";
        return {};
    }
    auto width = candidate.output();
    if (left->pointCount()) {
        if (!width || width == left->PlaneOwner("width") || width == right->PlaneOwner("width") ||
            width->context() != left->context() ||
            width->sizeBytes() != VkDeviceSize(left->pointCount()) * sizeof(float) ||
            !(width->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
            !(width->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
            if (why) *why = "invalid width blend COW output";
            return {};
        }
    } else if (width) { if (why) *why = "empty width blend has output"; return {}; }
    try { return std::shared_ptr<const VulkanSourceGeneration>(
        new VulkanSourceGeneration(left, std::move(width), value)); }
    catch (std::bad_alloc const&) { if (why) *why = "width blend COW allocation failed"; return {}; }
}
} // namespace usdGen::vulkan
