// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/graphSpec.h"

#include <algorithm>
#include <cstring>

namespace usdGen {

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

TfToken const kNoLabel;

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
    out->append(label->text.GetString());
}

}  // namespace

std::string UsdGenGraphSpec::Label(UsdGenNodeKey key) const
{
    std::string text;
    if (key.IsValid()) AppendLabel(labels, key, &text, 0);
    return text;
}

UsdGenNodeKey UsdGenGraphSpec::DeriveKey(UsdGenNodeKey parent, TfToken const &child)
{
    std::string text = Label(parent);
    text += '/';
    text += child.GetString();
    UsdGenNodeKey const key = UsdGenNodeKeyOfText(text);
    auto const it = std::lower_bound(labels.begin(), labels.end(), key,
        [](UsdGenNodeLabel const &l, UsdGenNodeKey k) { return l.key < k; });
    if (it == labels.end() || it->key != key)
        labels.insert(it, UsdGenNodeLabel{key, parent, child});
    return key;
}

TfToken const &UsdGenGraphSpec::LabelToken(UsdGenNodeKey key) const noexcept
{
    UsdGenNodeLabel const *label = key.IsValid() ? FindLabel(labels, key) : nullptr;
    return label && !label->parent.IsValid() ? label->text : kNoLabel;
}

TfToken const &UsdGenGraphSpec::DerivedName(UsdGenNodeKey key) const noexcept
{
    UsdGenNodeLabel const *label = key.IsValid() ? FindLabel(labels, key) : nullptr;
    return label && label->parent.IsValid() ? label->text : kNoLabel;
}

bool UsdGenGraphSpec::LabelLess(UsdGenNodeKey a, UsdGenNodeKey b) const
{
    if (a == b) return false;
    UsdGenNodeLabel const *la = a.IsValid() ? FindLabel(labels, a) : nullptr;
    UsdGenNodeLabel const *lb = b.IsValid() ? FindLabel(labels, b) : nullptr;
    // Host labels compare without allocating; derived ones build their text.
    if ((!la || !la->parent.IsValid()) && (!lb || !lb->parent.IsValid())) {
        char const *ta = la ? la->text.GetText() : "";
        char const *tb = lb ? lb->text.GetText() : "";
        return std::strcmp(ta, tb) < 0;
    }
    return Label(a) < Label(b);
}

}  // namespace usdGen
