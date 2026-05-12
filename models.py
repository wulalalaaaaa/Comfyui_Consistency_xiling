from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CharacterIdentity:
    name: str
    trigger_word: str
    visual_tags: List[str]
    face_tags: List[str]
    hair_tags: List[str]
    outfit_tags: List[str]
    body_tags: List[str]
    accessory_tags: List[str]
    color_palette: List[str]
    module_constraints: Dict[str, Any]
    reference_embeddings: Dict[str, Any]
    masks: Dict[str, Any]


@dataclass
class StyleIdentity:
    style_name: str
    style_tags: List[str]
    color_palette: List[str]
    line_features: Dict[str, Any]
    shading_features: Dict[str, Any]
    texture_features: Dict[str, Any]
    reference_embeddings: Dict[str, Any]
    leakage_control: Dict[str, Any]

