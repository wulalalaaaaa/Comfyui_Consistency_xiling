from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .core_utils import to_dict


def _limited_tags(tags: List[str], preset: str) -> List[str]:
    if not isinstance(tags, list):
        return []
    cleaned = [str(t).strip() for t in tags if isinstance(t, str) and str(t).strip()]
    if preset == "soft":
        return cleaned[:3]
    if preset == "medium":
        return cleaned[:6]
    return cleaned[:10]


def _module_phrase(module_name: str, tags: List[str], weight: float) -> str:
    if not tags:
        return ""
    readable = {
        "face": "face identity",
        "hair": "hair design",
        "outfit": "outfit design",
        "body": "body silhouette",
        "accessory": "accessory details",
    }.get(module_name, module_name)
    tag_text = ", ".join(tags)
    w = max(0.0, float(weight))
    return f"({readable}, {tag_text}:{w:.2f})"


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        t = str(item).strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _build_global_character_anchor(
    module_tag_map: Dict[str, List[str]],
    preset: str,
) -> str:
    readable = {
        "face": "face identity",
        "hair": "hair silhouette",
        "outfit": "outfit motif",
        "body": "body silhouette",
        "accessory": "accessory signature",
    }
    parts: List[str] = []
    for module_name in ("face", "hair", "outfit", "body", "accessory"):
        tags = module_tag_map.get(module_name, [])
        if not tags:
            continue
        head = ", ".join(tags[:2])
        parts.append(f"{readable.get(module_name, module_name)} ({head})")
    if not parts:
        return ""
    anchor_weight = {"soft": 1.05, "medium": 1.20, "hard": 1.35}.get(preset, 1.20)
    return f"(same character identity across {'; '.join(parts)}:{anchor_weight:.2f})"


class CharacterConstraintPromptComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {"default": "", "multiline": True}),
                "character_identity": ("CHARACTER_IDENTITY",),
                "enable_face_constraint": ("BOOLEAN", {"default": True}),
                "enable_hair_constraint": ("BOOLEAN", {"default": True}),
                "enable_outfit_constraint": ("BOOLEAN", {"default": True}),
                "enable_body_constraint": ("BOOLEAN", {"default": True}),
                "enable_accessory_constraint": ("BOOLEAN", {"default": False}),
                "face_weight": ("FLOAT", {"default": 1.30, "min": 0.0, "max": 2.5, "step": 0.01}),
                "hair_weight": ("FLOAT", {"default": 1.20, "min": 0.0, "max": 2.5, "step": 0.01}),
                "outfit_weight": ("FLOAT", {"default": 1.30, "min": 0.0, "max": 2.5, "step": 0.01}),
                "body_weight": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 2.5, "step": 0.01}),
                "accessory_weight": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.5, "step": 0.01}),
                "constraint_strength_preset": (["soft", "medium", "hard"],),
                "enable_global_character_anchor": ("BOOLEAN", {"default": True}),
                "include_trigger_word": ("BOOLEAN", {"default": True}),
                "include_hair_color_hint": ("BOOLEAN", {"default": True}),
                "include_hair_hex_hint": ("BOOLEAN", {"default": False}),
                "join_style": (["append", "prepend"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("composed_prompt", "constraint_fragment")
    FUNCTION = "compose"
    CATEGORY = "xiling_anime_Consistency/Prompt"

    def compose(
        self,
        base_prompt: str,
        character_identity: Dict[str, Any],
        enable_face_constraint: bool,
        enable_hair_constraint: bool,
        enable_outfit_constraint: bool,
        enable_body_constraint: bool,
        enable_accessory_constraint: bool,
        face_weight: float,
        hair_weight: float,
        outfit_weight: float,
        body_weight: float,
        accessory_weight: float,
        constraint_strength_preset: str,
        enable_global_character_anchor: bool,
        include_trigger_word: bool,
        include_hair_color_hint: bool,
        include_hair_hex_hint: bool,
        join_style: str,
    ):
        identity = to_dict(character_identity)
        module_constraints = to_dict(identity.get("module_constraints", {}))

        ui_switches: List[Tuple[str, bool, float]] = [
            ("face", bool(enable_face_constraint), float(face_weight)),
            ("hair", bool(enable_hair_constraint), float(hair_weight)),
            ("outfit", bool(enable_outfit_constraint), float(outfit_weight)),
            ("body", bool(enable_body_constraint), float(body_weight)),
            ("accessory", bool(enable_accessory_constraint), float(accessory_weight)),
        ]

        fragments: List[str] = []
        module_tag_map: Dict[str, List[str]] = {}
        base = str(base_prompt or "").strip()
        base_lower = base.lower()
        if bool(include_trigger_word):
            trigger = str(identity.get("trigger_word", "")).strip()
            if trigger and trigger.lower() not in base_lower:
                fragments.append(trigger)

        for module_name, ui_enabled, ui_weight in ui_switches:
            mod = to_dict(module_constraints.get(module_name, {}))
            if not (bool(mod.get("enabled", True)) and ui_enabled):
                continue
            tags = mod.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = _limited_tags(tags, constraint_strength_preset)
            final_weight = max(0.0, min(2.5, ui_weight))
            phrase = _module_phrase(module_name, tags, final_weight)
            if phrase:
                fragments.append(phrase)
                module_tag_map[module_name] = tags

        if bool(enable_global_character_anchor):
            anchor_phrase = _build_global_character_anchor(module_tag_map, constraint_strength_preset)
            if anchor_phrase:
                fragments.insert(0, anchor_phrase)

        backend_report = to_dict(to_dict(identity.get("reference_embeddings", {})).get("backend_report", {}))
        hair_profile = to_dict(backend_report.get("hair_color_profile", {}))
        hair_family = str(hair_profile.get("primary_family", "")).strip()
        hair_hex = str(hair_profile.get("primary_hex", "")).strip()
        if bool(include_hair_color_hint) and hair_family:
            readable_family = hair_family.replace("-", " ")
            fragments.append(f"(hair color family: {readable_family}:1.25)")
        if bool(include_hair_hex_hint) and hair_hex:
            fragments.append(f"(hair color {hair_hex}:1.15)")

        fragments = _dedupe_keep_order(fragments)
        constraint_fragment = ", ".join([f for f in fragments if f.strip()])
        if not constraint_fragment:
            return base, ""

        if not base:
            return constraint_fragment, constraint_fragment

        if join_style == "prepend":
            composed = f"{constraint_fragment}, {base}"
        else:
            composed = f"{base}, {constraint_fragment}"
        return composed, constraint_fragment
