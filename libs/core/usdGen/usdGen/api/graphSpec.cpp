// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/writeSet.h"
#include "usdGen/registries/typeRegistry.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>

namespace usdGen {

UsdGenInputDesc const *UsdGenFindInputDesc(UsdGenTypeId kind) noexcept
{
    return kind == UsdGenTypeId::Invalid ? nullptr : UsdGenTypeRegistry::Get().Def(kind).input;
}

UsdGenTypeId UsdGenInputKindWithRole(UsdGenInputRole role) noexcept
{
    UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
    for (uint32_t i = 0; i < types.Count(); ++i) {
        UsdGenInputDesc const *input = types.Def(UsdGenTypeId(i)).input;
        if (input && input->role == role) return UsdGenTypeId(i);
    }
    return UsdGenTypeId::Invalid;
}

UsdGenNodeKey UsdGenNodeKeyOfText(std::string const &text) noexcept
{
    uint64_t h = 0xcbf29ce484222325ULL;
    for (char c : text) {
        h ^= uint64_t(static_cast<unsigned char>(c));
        h *= 0x100000001b3ULL;
    }
    return UsdGenNodeKey{h ? h : 1};
}

namespace {

UsdGenText const kNoLabel;
UsdGenToken const kNoName;

UsdGenNodeLabel const *FindLabel(std::vector<UsdGenNodeLabel> const &labels, UsdGenNodeKey key)
{
    auto const it = std::lower_bound(labels.begin(), labels.end(), key,
        [](UsdGenNodeLabel const &l, UsdGenNodeKey k) { return l.key < k; });
    return it != labels.end() && it->key == key ? &*it : nullptr;
}

void AppendLabel(std::vector<UsdGenNodeLabel> const &labels, UsdGenNodeKey key,
                 std::string *out, int depth)
{
    UsdGenNodeLabel const *label = FindLabel(labels, key);
    if (!label) return;
    if (label->parent.IsValid() && depth < 64) {
        AppendLabel(labels, label->parent, out, depth + 1);
        out->push_back('/');
    }
    out->append(label->text.View().data(), label->text.View().size());
}

}  // namespace

std::string UsdGenGraphSpec::Label(UsdGenNodeKey key) const
{
    std::string text;
    if (key.IsValid()) AppendLabel(labels, key, &text, 0);
    return text;
}

UsdGenNodeKey UsdGenGraphSpec::DeriveKey(UsdGenNodeKey parent, UsdGenToken const &child)
{
    std::string text = Label(parent);
    text += '/';
    text += child.GetString();
    UsdGenNodeKey const key = UsdGenNodeKeyOfText(text);
    auto const it = std::lower_bound(labels.begin(), labels.end(), key,
        [](UsdGenNodeLabel const &l, UsdGenNodeKey k) { return l.key < k; });
    if (it == labels.end() || it->key != key)
        labels.insert(it, UsdGenNodeLabel{key, parent, UsdGenText(child.View()), child});
    return key;
}

UsdGenText const &UsdGenGraphSpec::LabelText(UsdGenNodeKey key) const noexcept
{
    UsdGenNodeLabel const *label = key.IsValid() ? FindLabel(labels, key) : nullptr;
    return label && !label->parent.IsValid() ? label->text : kNoLabel;
}

UsdGenToken const &UsdGenGraphSpec::DerivedName(UsdGenNodeKey key) const noexcept
{
    UsdGenNodeLabel const *label = key.IsValid() ? FindLabel(labels, key) : nullptr;
    return label && label->parent.IsValid() ? label->name : kNoName;
}

namespace {

template <class Records>
auto KindRun(Records &inputs, UsdGenTypeId kind) noexcept
{
    auto const lo = std::lower_bound(inputs.begin(), inputs.end(), kind,
        [](UsdGenInputRecord const &r, UsdGenTypeId k) { return r.kind < k; });
    auto hi = lo;
    while (hi != inputs.end() && hi->kind == kind) ++hi;
    return std::make_pair(lo, hi);
}

}  // namespace

UsdGenSpan<const UsdGenInputRecord> UsdGenGraphSpec::Inputs(UsdGenTypeId kind) const noexcept
{
    if (kind == UsdGenTypeId::Invalid) return {};
    auto const run = KindRun(inputs, kind);
    return {inputs.data() + (run.first - inputs.begin()), size_t(run.second - run.first)};
}

UsdGenSpan<UsdGenInputRecord> UsdGenGraphSpec::MutableInputs(UsdGenTypeId kind) noexcept
{
    if (kind == UsdGenTypeId::Invalid) return {};
    auto const run = KindRun(inputs, kind);
    return {inputs.data() + (run.first - inputs.begin()), size_t(run.second - run.first)};
}

UsdGenInputRecord const *UsdGenGraphSpec::FindInput(UsdGenTypeId kind, UsdGenNodeKey key) const noexcept
{
    for (UsdGenInputRecord const &record : Inputs(kind))
        if (record.key == key) return &record;
    return nullptr;
}

void UsdGenGraphSpec::SortInputs()
{
    std::stable_sort(inputs.begin(), inputs.end(),
        [](UsdGenInputRecord const &a, UsdGenInputRecord const &b) { return a.kind < b.kind; });
}

UsdGenInputRecordWriter::UsdGenInputRecordWriter(UsdGenTypeId kind) : _kind(kind)
{
    if (kind == UsdGenTypeId::Invalid) return;
    UsdGenTypeDef const &def = UsdGenTypeRegistry::Get().Def(kind);
    _layout = def.layout;
    _desc = def.input;
    if (!Valid()) return;
    _staged.resize(_layout->channels.size());
    _sources.resize(_layout->channels.size());
    _values.resize(_desc->attributes.size());
    for (size_t i = 0; i < _values.size(); ++i) _values[i].name = _desc->attributes[i];
}

void UsdGenInputRecordWriter::_Stage(UsdGenSlot slot, void const *data, size_t bytes)
{
    if (slot >= _staged.size()) return;
    _staged[slot] = Staged{bytes ? data : nullptr, bytes, nullptr, kUsdGenNoSlot};
}

void UsdGenInputRecordWriter::Share(UsdGenSlot slot, UsdGenElementSet const &from, UsdGenSlot fromSlot)
{
    if (slot >= _staged.size()) return;
    _staged[slot] = from.Has(fromSlot) ? Staged{nullptr, 0, &from, fromSlot} : Staged{};
}

void UsdGenInputRecordWriter::Value(size_t index, UsdGenValue value)
{
    if (index < _values.size() && (!_desc || index < _desc->attributes.size()))
        _values[index].value = std::move(value);
}

void UsdGenInputRecordWriter::Extra(UsdGenParamValue value)
{
    if (Valid()) _values.push_back(std::move(value));
}

UsdGenInputRecord UsdGenInputRecordWriter::Finish(UsdGenNodeKey key, UsdGenNodeKey owner,
                                                  UsdGenText const &label, uint64_t generation)
{
    UsdGenInputRecord record;
    record.key = key;
    record.owner = owner;
    record.kind = _kind;
    record.label = label;
    record.generation = generation;
    if (!Valid()) return record;
    // A cached column whose source is the same held buffer is shared.
    auto cacheId = [&](UsdGenSlot s) {
        return (_cacheKey.value * 0x9E3779B97F4A7C15ULL) ^ (uint64_t(_kind) << 16) ^ uint64_t(s);
    };
    std::vector<Staged> sources;   // the host buffers, before sharing
    if (_cache) {
        sources = _staged;
        for (UsdGenSlot s = 0; s < _staged.size(); ++s) {
            Staged &st = _staged[s];
            if (!st.data || _sources[s].IsEmpty()) continue;
            auto const it = _cache->_entries.find(cacheId(s));
            if (it == _cache->_entries.end()) continue;
            UsdGenInputColumnCache::Entry const &e = it->second;
            if (e.key == _cacheKey && e.kind == _kind && e.slot == s && e.data == st.data &&
                e.bytes == st.bytes && e.set && e.set->Has(s))
                st = Staged{nullptr, 0, e.set.Get(), s};
        }
    }
    UsdGenRef<const UsdGenChannelTable> table = UsdGenChannelTable::Make(*_layout, {});
    UsdGenWriteSet w = UsdGenWriteSet::Fresh(table, {});
    std::vector<UsdGenSlot> written;
    for (UsdGenSlot s = 0; s < _staged.size(); ++s) {
        Staged const &st = _staged[s];
        uint32_t const scalarBytes = UsdGenScalarBytes((*table)[s].scalar);
        if (st.from) {
            UsdGenSpan<const std::byte> const column = st.from->Column<std::byte>(st.fromSlot);
            w.SetValueCount(s, uint32_t(column.size() / scalarBytes));
            w.Share(s, *st.from, st.fromSlot);
        } else if (st.data) {
            w.SetValueCount(s, uint32_t(st.bytes / scalarBytes));
            written.push_back(s);
        }
    }
    w.Plan(UsdGenSpan<const UsdGenSlot>(written.data(), written.size()));
    for (UsdGenSlot s : written) {
        UsdGenSpan<std::byte> out = w.Write<std::byte>(s);
        if (out.size() >= _staged[s].bytes) std::memcpy(out.data(), _staged[s].data, _staged[s].bytes);
    }
    record.data = std::move(w).Seal();
    record.data->Export();
    if (_cache) {
        for (UsdGenSlot s = 0; s < _sources.size(); ++s) {
            if (_sources[s].IsEmpty() || !record.data->Has(s)) continue;
            UsdGenInputColumnCache::Entry &e = _cache->_entries[cacheId(s)];
            e.key = _cacheKey;
            e.kind = _kind;
            e.slot = s;
            e.data = sources[s].data;
            e.bytes = sources[s].bytes;
            e.source = std::move(_sources[s]);
            e.set = record.data;
            e.used = _cache->_epoch;
        }
    }
    record.values = std::move(_values);
    return record;
}

UsdGenInputColumnCache::UsdGenInputColumnCache() = default;
UsdGenInputColumnCache::~UsdGenInputColumnCache() = default;

void UsdGenInputColumnCache::Advance(uint32_t age)
{
    ++_epoch;
    for (auto it = _entries.begin(); it != _entries.end();)
        it = _epoch - it->second.used > age ? _entries.erase(it) : std::next(it);
}

void UsdGenInputColumnCache::Clear() noexcept
{
    _entries.clear();
}

bool UsdGenGraphSpec::LabelLess(UsdGenNodeKey a, UsdGenNodeKey b) const
{
    if (a == b) return false;
    UsdGenNodeLabel const *la = a.IsValid() ? FindLabel(labels, a) : nullptr;
    UsdGenNodeLabel const *lb = b.IsValid() ? FindLabel(labels, b) : nullptr;
    // Host labels compare without allocating; derived ones build their text.
    if ((!la || !la->parent.IsValid()) && (!lb || !lb->parent.IsValid())) {
        char const *ta = la ? la->text.c_str() : "";
        char const *tb = lb ? lb->text.c_str() : "";
        return std::strcmp(ta, tb) < 0;
    }
    return Label(a) < Label(b);
}

}  // namespace usdGen
