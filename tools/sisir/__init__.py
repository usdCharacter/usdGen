# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""Sisir groom import for usdGen (``sir2usd``).

    from tools.sisir import convert, apply_preset_to_stage
    convert("Autumn.sir", "out/Autumn.usda")
    apply_preset_to_stage("out/Autumn.usda", "curly.json", "/World/Groom/Set_1")

Card-grown hair sets are authored through ``cards.py`` (same folder) when it is present.
Run Python with ``-S`` (see README.md).
"""

from .sir2usd import (  # noqa: F401
    __version__,
    ConvertOptions,
    Converter,
    ObjMesh,
    SirFormatError,
    SisirScene,
    apply_hair_preset,
    apply_preset_to_stage,
    clamp_hair_settings,
    convert,
    decode_vertex_mask,
    derive_ops,
    map_hair_material,
    read_hair_preset,
    resolve_hair_settings,
    settings_for,
)

__all__ = [
    "__version__", "ConvertOptions", "Converter", "ObjMesh", "SirFormatError", "SisirScene",
    "apply_hair_preset", "apply_preset_to_stage", "clamp_hair_settings", "convert", "decode_vertex_mask",
    "derive_ops", "map_hair_material", "read_hair_preset", "resolve_hair_settings", "settings_for",
]
