from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

import torch

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


def _safe_tags(tags, limit: int = 2) -> List[str]:
    if not isinstance(tags, list):
        return []
    out: List[str] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _build_character_anchor(module_constraints: dict) -> dict:
    module_readable = {
        "face": "face identity",
        "hair": "hair shape",
        "outfit": "outfit motif",
        "body": "body silhouette",
        "accessory": "accessory signature",
    }
    enabled_modules: List[str] = []
    anchor_parts: List[str] = []
    tag_snapshot = {}

    for module_name in ("face", "hair", "outfit", "body", "accessory"):
        mod = to_dict(module_constraints.get(module_name, {}))
        if not bool(mod.get("enabled", False)):
            continue
        enabled_modules.append(module_name)
        tags = _safe_tags(mod.get("tags", []), limit=2)
        if tags:
            tag_snapshot[module_name] = tags
            anchor_parts.append(f"{module_readable.get(module_name, module_name)} ({', '.join(tags)})")
        else:
            anchor_parts.append(module_readable.get(module_name, module_name))

    if not enabled_modules:
        return {
            "enabled": False,
            "summary": "no character modules enabled",
            "enabled_modules": [],
            "tag_snapshot": {},
        }

    return {
        "enabled": True,
        "summary": "same character identity across " + "; ".join(anchor_parts),
        "enabled_modules": enabled_modules,
        "tag_snapshot": tag_snapshot,
    }


def _rebalance_module_strengths(
    module_constraints: dict,
    identity_strength: float,
    pose_freedom: float,
    prompt_fidelity: float,
) -> dict:
    # Keep "whole character" consistency by avoiding over-focus on one module.
    role_weight = {
        "face": 1.05,
        "hair": 0.90,
        "outfit": 1.05,
        "body": 1.00,
        "accessory": 0.85,
    }
    id_s = clamp01(identity_strength)
    pose_s = clamp01(pose_freedom)
    prompt_s = clamp01(prompt_fidelity)
    global_lock = clamp01((0.55 + 0.45 * id_s) * (0.70 + 0.30 * (1.0 - pose_s)))
    prompt_relax = 1.0 - 0.25 * prompt_s

    out = {}
    for module_name in ("face", "hair", "outfit", "body", "accessory"):
        mod = to_dict(module_constraints.get(module_name, {}))
        if not bool(mod.get("enabled", False)):
            mod["strength"] = 0.0
            mod["strength_base"] = float(mod.get("strength", 0.0))
            mod["strength_role_weight"] = float(role_weight.get(module_name, 1.0))
            mod["strength_global_lock"] = round(global_lock, 6)
            mod["strength_prompt_relax"] = round(prompt_relax, 6)
            out[module_name] = mod
            continue

        base = clamp01(float(mod.get("strength", 0.0)))
        mod["strength_base"] = round(base, 6)
        mod["strength_role_weight"] = float(role_weight.get(module_name, 1.0))
        mod["strength_global_lock"] = round(global_lock, 6)
        mod["strength_prompt_relax"] = round(prompt_relax, 6)
        blended = base * role_weight.get(module_name, 1.0) * global_lock * prompt_relax + 0.15 * id_s
        mod["strength"] = round(clamp01(blended), 6)
        out[module_name] = mod
    return out


def _clean_tags(tags: Any, limit: int = 6) -> List[str]:
    if not isinstance(tags, list):
        return []
    out: List[str] = []
    seen = set()
    for t in tags:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _build_constraint_prompt_from_identity(
    character_identity: Dict[str, Any],
    module_constraints: Dict[str, Any],
    include_hair_hint: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    identity = to_dict(character_identity)
    backend_report = to_dict(to_dict(identity.get("reference_embeddings", {})).get("backend_report", {}))
    hair_profile = to_dict(backend_report.get("hair_color_profile", {}))
    hair_family = str(hair_profile.get("primary_family", "")).strip()
    eye_profile = to_dict(backend_report.get("eye_color_profile", {}))
    eye_family = str(eye_profile.get("primary_family", "")).strip()
    outfit_profile = to_dict(backend_report.get("outfit_color_profile", {}))
    outfit_family = str(outfit_profile.get("primary_family", "")).strip()

    module_readable = {
        "face": "face identity",
        "hair": "hair shape and color",
        "outfit": "outfit design and palette",
        "body": "body silhouette",
        "accessory": "accessory signature",
    }
    module_weight_default = {
        "face": 1.35,
        "hair": 1.30,
        "outfit": 1.30,
        "body": 1.20,
        "accessory": 1.10,
    }

    fragments: List[str] = []
    module_tag_counts: Dict[str, int] = {}
    trigger = str(identity.get("trigger_word", "")).strip()
    if trigger:
        fragments.append(trigger)

    visual_tags = _clean_tags(identity.get("visual_tags", []), limit=6)
    if visual_tags:
        fragments.append(", ".join(visual_tags))

    fallback_module_tags = {
        "face": _clean_tags(identity.get("face_tags", []), limit=5),
        "hair": _clean_tags(identity.get("hair_tags", []), limit=5),
        "outfit": _clean_tags(identity.get("outfit_tags", []), limit=5),
        "body": _clean_tags(identity.get("body_tags", []), limit=5),
        "accessory": _clean_tags(identity.get("accessory_tags", []), limit=5),
    }
    active_modules: List[str] = []
    for module_name in ("face", "hair", "outfit", "body", "accessory"):
        mod = to_dict(module_constraints.get(module_name, {}))
        if not bool(mod.get("enabled", False)):
            continue
        active_modules.append(module_name)
        tags = _clean_tags(mod.get("tags", []), limit=5)
        if not tags:
            tags = fallback_module_tags[module_name]
        module_tag_counts[module_name] = len(tags)
        if tags:
            w = float(module_weight_default.get(module_name, 1.15))
            fragments.append(f"({module_readable[module_name]}, {', '.join(tags)}:{w:.2f})")

    if active_modules:
        fragments.append(
            "(same character identity, same hairstyle, same eye color, same outfit colors and motifs:1.35)"
        )
        fragments.append(
            "(avoid identity drift, avoid hairstyle drift, avoid outfit color drift:1.30)"
        )
    if include_hair_hint and hair_family:
        fragments.append(f"(hair color family {hair_family.replace('-', ' ')}:1.20)")
    if eye_family:
        fragments.append(f"(eye color family {eye_family.replace('-', ' ')}:1.25)")
    if outfit_family:
        fragments.append(f"(outfit main color family {outfit_family.replace('-', ' ')}:1.20)")

    # Deduplicate while keeping order.
    cleaned: List[str] = []
    seen = set()
    for x in fragments:
        s = str(x).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        cleaned.append(s)

    summary = {
        "active_modules": active_modules,
        "module_tag_counts": module_tag_counts,
        "has_trigger": bool(trigger),
        "has_visual_tags": bool(visual_tags),
    }
    return ", ".join(cleaned), summary


def _encode_prompt_with_clip(clip, text: str):
    if clip is None:
        return None, "clip_missing"
    prompt = str(text or "").strip()
    if not prompt:
        return None, "empty_constraint_prompt"
    try:
        tokens = clip.tokenize(prompt)
        encoded = clip.encode_from_tokens(tokens, return_pooled=True)
        if isinstance(encoded, tuple) and len(encoded) >= 2:
            cond, pooled = encoded[0], encoded[1]
        else:
            cond, pooled = encoded, None
        if not torch.is_tensor(cond):
            return None, "clip_encode_invalid_cond"
        meta = {}
        if torch.is_tensor(pooled):
            meta["pooled_output"] = pooled
        return [[cond, meta]], "ok"
    except Exception as e:
        return None, f"clip_encode_error:{type(e).__name__}"


def _fuse_conditioning(
    base_conditioning,
    extra_conditioning,
    fusion_strength: float,
    fusion_mode: str = "append",
    soft_lock_boost: float = 1.0,
    append_repeat: int = 1,
):
    w = clamp01(fusion_strength)
    boost = max(0.5, min(4.0, float(soft_lock_boost)))
    repeat = max(1, min(8, int(append_repeat)))
    if w <= 0.0:
        return base_conditioning, {"status": "skip_strength_zero", "fused_items": 0}
    if not isinstance(base_conditioning, list) or not base_conditioning:
        return base_conditioning, {"status": "skip_base_empty", "fused_items": 0}
    if not isinstance(extra_conditioning, list) or not extra_conditioning:
        return base_conditioning, {"status": "skip_extra_empty", "fused_items": 0}

    extra_item = extra_conditioning[0]
    if not (isinstance(extra_item, (list, tuple)) and len(extra_item) >= 2):
        return base_conditioning, {"status": "skip_extra_format", "fused_items": 0}

    extra_cond = extra_item[0]
    extra_meta = to_dict(extra_item[1])
    if not torch.is_tensor(extra_cond):
        return base_conditioning, {"status": "skip_extra_not_tensor", "fused_items": 0}

    if fusion_mode == "append":
        extra_meta_scaled = copy.deepcopy(extra_meta)
        scaled_strength = w * boost
        if torch.is_tensor(extra_meta_scaled.get("pooled_output")):
            extra_meta_scaled["pooled_output"] = extra_meta_scaled["pooled_output"] * scaled_strength
        extra_meta_scaled["strength"] = round(float(scaled_strength), 4)
        appended_cond = extra_cond * scaled_strength
        out = copy.deepcopy(base_conditioning)
        for _ in range(repeat):
            out.append([appended_cond, copy.deepcopy(extra_meta_scaled)])
        return out, {
            "status": "ok_append_boost",
            "fused_items": repeat,
            "fusion_strength": round(w, 4),
            "soft_lock_boost": round(boost, 4),
            "append_repeat": repeat,
        }

    out = []
    fused_items = 0
    # Blend mode: strengthen replacement ratio and extra conditioning gain.
    w_eff = clamp01(w * min(2.0, boost))
    extra_gain = max(1.0, min(2.5, boost))
    for item in base_conditioning:
        if not (isinstance(item, (list, tuple)) and len(item) >= 2 and torch.is_tensor(item[0])):
            out.append(item)
            continue
        cond = item[0]
        meta = copy.deepcopy(item[1]) if isinstance(item[1], dict) else {}
        if cond.shape != extra_cond.shape:
            out.append(item)
            continue

        fused_cond = cond * (1.0 - w_eff) + extra_cond * (w_eff * extra_gain)
        pooled_a = meta.get("pooled_output")
        pooled_b = extra_meta.get("pooled_output")
        if torch.is_tensor(pooled_a) and torch.is_tensor(pooled_b) and pooled_a.shape == pooled_b.shape:
            meta["pooled_output"] = pooled_a * (1.0 - w_eff) + pooled_b * (w_eff * extra_gain)
        meta["strength"] = round(float(max(w_eff, w_eff * extra_gain)), 4)

        out.append([fused_cond, meta])
        fused_items += 1

    status = "ok_blend_boost" if fused_items > 0 else "skip_shape_mismatch_or_non_tensor"
    return out, {
        "status": status,
        "fused_items": fused_items,
        "fusion_strength": round(w, 4),
        "soft_lock_boost": round(boost, 4),
        "blend_effective_strength": round(w_eff, 4),
    }


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
                "enable_hair_color_deltae_constraint": ("BOOLEAN", {"default": False}),
                "hair_color_deltae_threshold": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "hair_color_deltae_softness": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "enable_direct_constraint_fusion": ("BOOLEAN", {"default": True}),
                "constraint_fusion_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "constraint_fusion_mode": (["blend", "append"],),
                "soft_lock_boost": ("FLOAT", {"default": 1.8, "min": 0.5, "max": 4.0, "step": 0.05}),
                "soft_lock_repeat": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "include_hair_color_hint_in_fusion": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip": ("CLIP",),
                "constraint_prompt_override": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "adapter_state", "debug_report_json")
    FUNCTION = "apply"
    CATEGORY = "xiling_anime_Consistency/Apply"

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
        enable_direct_constraint_fusion: bool,
        constraint_fusion_strength: float,
        constraint_fusion_mode: str,
        soft_lock_boost: float,
        soft_lock_repeat: int,
        include_hair_color_hint_in_fusion: bool,
        clip=None,
        constraint_prompt_override: str = "",
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
        threshold = max(0.0, float(hair_color_deltae_threshold))
        softness = max(0.1, float(hair_color_deltae_softness))

        hair_color_constraint = {
            "enabled": bool(enable_hair_color_deltae_constraint),
            "deltae_threshold": round(threshold, 4),
            "deltae_softness": round(softness, 4),
            "target_lab": {
                "l": round(target_lab[0], 4),
                "a": round(target_lab[1], 4),
                "b": round(target_lab[2], 4),
            } if target_lab is not None else None,
            "current_lab": None,
            "deltae": None,
            "pass": None,
            "penalty": 0.0,
            "status": "deferred_to_debug_viewer",
        }

        if bool(enable_hair_color_deltae_constraint):
            hair_mod = to_dict(module_constraints.get("hair", {}))
            base_strength = float(hair_mod.get("strength", 0.0))
            hair_mod["strength_base"] = round(base_strength, 6)
            hair_mod["strength"] = round(base_strength, 6)
            hair_mod["color_deltae"] = hair_color_constraint
            module_constraints["hair"] = hair_mod

        module_constraints = _rebalance_module_strengths(
            module_constraints=module_constraints,
            identity_strength=identity_strength,
            pose_freedom=pose_freedom,
            prompt_fidelity=prompt_fidelity,
        )
        character_anchor = _build_character_anchor(module_constraints)

        debug_report = {
            "mode": "character_lock_debug_config",
            "character_anchor": character_anchor,
            "hair_color_constraint": hair_color_constraint,
            "target_hair_profile": target_hair_profile,
            "current_hair_profile": {},
        }

        constraint_prompt, prompt_summary = _build_constraint_prompt_from_identity(
            to_dict(character_identity),
            module_constraints,
            include_hair_hint=bool(include_hair_color_hint_in_fusion),
        )
        override_text = str(constraint_prompt_override or "").strip()
        if override_text:
            constraint_prompt = override_text

        fusion_report = {
            "enabled": bool(enable_direct_constraint_fusion),
            "used_override": bool(override_text),
            "constraint_prompt": constraint_prompt,
            "prompt_summary": prompt_summary,
            "status": "skipped",
            "fused_items": 0,
            "soft_lock_boost": round(float(soft_lock_boost), 4),
            "soft_lock_repeat": int(soft_lock_repeat),
        }
        out = conditioning
        if bool(enable_direct_constraint_fusion):
            extra_conditioning, encode_status = _encode_prompt_with_clip(clip, constraint_prompt)
            fusion_report["encode_status"] = encode_status
            if encode_status == "ok":
                out, fusion_meta = _fuse_conditioning(
                    out,
                    extra_conditioning,
                    float(constraint_fusion_strength),
                    fusion_mode=str(constraint_fusion_mode or "append"),
                    soft_lock_boost=float(soft_lock_boost),
                    append_repeat=int(soft_lock_repeat),
                )
                fusion_report.update(fusion_meta)
            else:
                fusion_report["status"] = "skipped_encode_failed"

        debug_report["direct_fusion"] = fusion_report

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
            "character_anchor": character_anchor,
            "hair_color_constraint": hair_color_constraint,
            "constraint_prompt_for_fusion": constraint_prompt,
            "direct_fusion": fusion_report,
            "debug_report": debug_report,
        }
        out = append_note_to_conditioning(out, note)
        enabled_text = "+".join(enabled_modules) if enabled_modules else "none"
        return out, f"character_constraint_fusion:{enabled_text}", json.dumps(debug_report, ensure_ascii=False)


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
    CATEGORY = "xiling_anime_Consistency/Apply"

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
    CATEGORY = "xiling_anime_Consistency/Apply"

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
