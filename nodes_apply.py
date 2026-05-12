from __future__ import annotations

import json
import math
from typing import List

import torch

from .color_features import extract_hair_color_profile
from .core_utils import append_note_to_conditioning, clamp01, to_dict


def _lab_from_hair_profile(profile: dict):
    dominant = profile.get("dominant_colors", [])
    if not isinstance(dominant, list) or not dominant:
        return None
    first = to_dict(dominant[0])
    lab = to_dict(first.get("lab", {}))
    if all(k in lab for k in ("l", "a", "b")):
        try:
            return float(lab["l"]), float(lab["a"]), float(lab["b"])
        except Exception:
            return None
    return None


def _delta_e76(lab1, lab2) -> float:
    dl = float(lab1[0]) - float(lab2[0])
    da = float(lab1[1]) - float(lab2[1])
    db = float(lab1[2]) - float(lab2[2])
    return float(math.sqrt(dl * dl + da * da + db * db))


class ApplyCharacterConsistency:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "character_identity": ("CHARACTER_IDENTITY",),
                "identity_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_freedom": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "outfit_strength": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_refine_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt_fidelity": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_face_constraint": ("BOOLEAN", {"default": True}),
                "enable_hair_constraint": ("BOOLEAN", {"default": True}),
                "enable_outfit_constraint": ("BOOLEAN", {"default": True}),
                "enable_body_constraint": ("BOOLEAN", {"default": True}),
                "enable_accessory_constraint": ("BOOLEAN", {"default": False}),
                "face_strength": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hair_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "body_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "accessory_strength": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_hair_color_deltae_constraint": ("BOOLEAN", {"default": True}),
                "hair_color_deltae_threshold": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "hair_color_deltae_softness": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 50.0, "step": 0.1}),
            }
            ,
            "optional": {
                "current_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "adapter_state", "debug_report_json")
    FUNCTION = "apply"
    CATEGORY = "AnimeConsistency/Apply"

    def apply(
        self,
        conditioning,
        character_identity,
        identity_strength: float,
        pose_freedom: float,
        outfit_strength: float,
        detail_refine_strength: float,
        prompt_fidelity: float,
        enable_face_constraint: bool,
        enable_hair_constraint: bool,
        enable_outfit_constraint: bool,
        enable_body_constraint: bool,
        enable_accessory_constraint: bool,
        face_strength: float,
        hair_strength: float,
        body_strength: float,
        accessory_strength: float,
        enable_hair_color_deltae_constraint: bool,
        hair_color_deltae_threshold: float,
        hair_color_deltae_softness: float,
        current_image: torch.Tensor = None,
    ):
        module_defaults = to_dict(character_identity).get("module_constraints", {})
        module_switches = {
            "face": bool(enable_face_constraint),
            "hair": bool(enable_hair_constraint),
            "outfit": bool(enable_outfit_constraint),
            "body": bool(enable_body_constraint),
            "accessory": bool(enable_accessory_constraint),
        }
        module_strength_inputs = {
            "face": clamp01(face_strength),
            "hair": clamp01(hair_strength),
            "outfit": clamp01(outfit_strength),
            "body": clamp01(body_strength),
            "accessory": clamp01(accessory_strength),
        }

        module_constraints = {}
        enabled_modules: List[str] = []
        for module_name, enabled_from_ui in module_switches.items():
            base = to_dict(module_defaults.get(module_name))
            enabled = bool(base.get("enabled", True)) and enabled_from_ui
            strength = module_strength_inputs[module_name] if enabled else 0.0
            tags = base.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            module_constraints[module_name] = {
                "enabled": enabled,
                "strength": strength,
                "tags": tags,
                "status": base.get("status", "pending_backend"),
            }
            if enabled:
                enabled_modules.append(module_name)

        backend_report = to_dict(
            to_dict(character_identity).get("reference_embeddings", {})
        ).get("backend_report", {})
        backend_report = to_dict(backend_report)
        target_hair_profile = to_dict(backend_report.get("hair_color_profile", {}))
        target_lab = _lab_from_hair_profile(target_hair_profile)
        current_hair_profile = {}
        current_lab = None
        if current_image is not None:
            current_hair_profile = extract_hair_color_profile(current_image, top_k=4)
            current_lab = _lab_from_hair_profile(current_hair_profile)

        deltae_value = None
        if target_lab is not None and current_lab is not None:
            deltae_value = _delta_e76(target_lab, current_lab)

        threshold = max(0.0, float(hair_color_deltae_threshold))
        softness = max(0.1, float(hair_color_deltae_softness))
        pass_flag = None
        penalty = 0.0
        if deltae_value is not None:
            pass_flag = deltae_value <= threshold
            over = max(0.0, deltae_value - threshold)
            penalty = over / (over + softness) if over > 0.0 else 0.0

        hair_color_constraint = {
            "enabled": bool(enable_hair_color_deltae_constraint),
            "deltae_threshold": round(threshold, 4),
            "deltae_softness": round(softness, 4),
            "target_lab": {
                "l": round(target_lab[0], 4),
                "a": round(target_lab[1], 4),
                "b": round(target_lab[2], 4),
            } if target_lab is not None else None,
            "current_lab": {
                "l": round(current_lab[0], 4),
                "a": round(current_lab[1], 4),
                "b": round(current_lab[2], 4),
            } if current_lab is not None else None,
            "deltae": round(deltae_value, 4) if deltae_value is not None else None,
            "pass": pass_flag,
            "penalty": round(penalty, 6),
            "status": "ok" if deltae_value is not None else "missing_target_or_current_image",
        }

        if bool(enable_hair_color_deltae_constraint):
            hair_mod = to_dict(module_constraints.get("hair", {}))
            base_strength = float(hair_mod.get("strength", 0.0))
            adjusted_strength = clamp01(base_strength * (1.0 + penalty * 0.35))
            hair_mod["strength_base"] = round(base_strength, 6)
            hair_mod["strength"] = round(adjusted_strength, 6)
            hair_mod["color_deltae"] = hair_color_constraint
            module_constraints["hair"] = hair_mod

        debug_report = {
            "mode": "character_lock_debug",
            "hair_color_constraint": hair_color_constraint,
            "target_hair_profile": target_hair_profile,
            "current_hair_profile": current_hair_profile,
        }

        note = {
            "mode": "character_lock",
            "character_identity": character_identity,
            "identity_strength": clamp01(identity_strength),
            "pose_freedom": clamp01(pose_freedom),
            "outfit_strength": clamp01(outfit_strength),
            "detail_refine_strength": clamp01(detail_refine_strength),
            "prompt_fidelity": clamp01(prompt_fidelity),
            "module_constraints": module_constraints,
            "enabled_modules": enabled_modules,
            "hair_color_constraint": hair_color_constraint,
            "debug_report": debug_report,
        }
        out = append_note_to_conditioning(conditioning, note)
        enabled_text = "+".join(enabled_modules) if enabled_modules else "none"
        adapter_state = f"character_backend_placeholder:{enabled_text}"
        if deltae_value is not None:
            adapter_state += f":hair_deltae={round(deltae_value, 3)}"
        return out, adapter_state, json.dumps(debug_report, ensure_ascii=False)


class ApplyStyleConsistency:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_identity": ("STYLE_IDENTITY",),
                "style_strength": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01}),
                "content_leakage_prevention": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "line_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shading_strength": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "adapter_state")
    FUNCTION = "apply"
    CATEGORY = "AnimeConsistency/Apply"

    def apply(
        self,
        conditioning,
        style_identity,
        style_strength: float,
        content_leakage_prevention: float,
        color_strength: float,
        line_strength: float,
        shading_strength: float,
    ):
        note = {
            "mode": "style_lock",
            "style_identity": style_identity,
            "style_strength": clamp01(style_strength),
            "content_leakage_prevention": clamp01(content_leakage_prevention),
            "color_strength": clamp01(color_strength),
            "line_strength": clamp01(line_strength),
            "shading_strength": clamp01(shading_strength),
        }
        out = append_note_to_conditioning(conditioning, note)
        return out, "style_backend_placeholder"


class CharacterStyleController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "character_identity": ("CHARACTER_IDENTITY",),
                "style_identity": ("STYLE_IDENTITY",),
                "character_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "style_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "conflict_policy": (["character_first", "style_first", "balanced"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "adapter_state")
    FUNCTION = "apply"
    CATEGORY = "AnimeConsistency/Apply"

    def apply(
        self,
        conditioning,
        character_identity,
        style_identity,
        character_strength: float,
        style_strength: float,
        conflict_policy: str,
    ):
        char_s = clamp01(character_strength)
        sty_s = clamp01(style_strength)
        if conflict_policy == "character_first":
            char_s = min(1.0, char_s + 0.10)
            sty_s = max(0.0, sty_s - 0.10)
        elif conflict_policy == "style_first":
            char_s = max(0.0, char_s - 0.10)
            sty_s = min(1.0, sty_s + 0.10)

        note = {
            "mode": "character_style_lock",
            "character_identity": character_identity,
            "style_identity": style_identity,
            "character_strength": char_s,
            "style_strength": sty_s,
            "conflict_policy": conflict_policy,
        }
        out = append_note_to_conditioning(conditioning, note)
        return out, f"combo_backend_placeholder:{conflict_policy}"
