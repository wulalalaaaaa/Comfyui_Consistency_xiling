from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List

import torch

from .backends import run_character_backend
from .core_utils import merge_tags, palette_from_image, parse_csv_tags, to_dict
from .models import CharacterIdentity


class CharacterReferenceAnalyzer:
    @staticmethod
    def _hair_family_to_prompt(family_label: str) -> str:
        t = str(family_label or "").strip().lower()
        if not t:
            return ""
        # expected like "mid-vivid-violet"
        parts = [p for p in t.split("-") if p]
        if len(parts) >= 3:
            tone, chroma, base = parts[0], parts[1], parts[2]
            return f"{chroma} {base} hair, {tone} tone"
        if len(parts) == 1:
            return f"{parts[0]} hair"
        return " ".join(parts) + " hair"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "character_name": ("STRING", {"default": "my_character", "multiline": False}),
                "trigger_word": ("STRING", {"default": "char_tok", "multiline": False}),
                "use_multi_reference": ("BOOLEAN", {"default": False}),
                "remove_background": ("BOOLEAN", {"default": False}),
                "split_face_hair_outfit": ("BOOLEAN", {"default": True}),
                "extra_tags_csv": ("STRING", {"default": "", "multiline": False}),
                "use_face_module": ("BOOLEAN", {"default": True}),
                "use_hair_module": ("BOOLEAN", {"default": True}),
                "use_outfit_module": ("BOOLEAN", {"default": True}),
                "use_body_module": ("BOOLEAN", {"default": True}),
                "use_accessory_module": ("BOOLEAN", {"default": False}),
                "face_tags_csv": ("STRING", {"default": "", "multiline": False}),
                "hair_tags_csv": ("STRING", {"default": "", "multiline": False}),
                "outfit_tags_csv": ("STRING", {"default": "", "multiline": False}),
                "body_tags_csv": ("STRING", {"default": "", "multiline": False}),
                "accessory_tags_csv": ("STRING", {"default": "", "multiline": False}),
                "backend_mode": (["placeholder", "local_vision", "remote_api"],),
                "run_backend": ("BOOLEAN", {"default": False}),
                "local_backend_model_name": ("STRING", {"default": "vit_base_patch16_224", "multiline": False}),
                "local_backend_use_pretrained": ("BOOLEAN", {"default": False}),
                "local_backend_checkpoint_path": ("STRING", {"default": "", "multiline": False}),
                "remote_api_url": ("STRING", {"default": "", "multiline": False}),
                "remote_api_key": ("STRING", {"default": "", "multiline": False}),
                "remote_api_model": ("STRING", {"default": "gpt-4.1-mini", "multiline": False}),
                "remote_api_timeout_s": ("INT", {"default": 30, "min": 5, "max": 120, "step": 1}),
                "remote_output_schema_json": (
                    "STRING",
                    {
                        "default": '{"global_tags":[],"module_tags":{"face":[],"hair":[],"outfit":[],"body":[],"accessory":[]},"notes":"","confidence":"low"}',
                        "multiline": True,
                    },
                ),
                "remote_extra_headers_json": ("STRING", {"default": "{}", "multiline": False}),
            }
        }

    RETURN_TYPES = ("CHARACTER_IDENTITY", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "character_identity",
        "character_prompt",
        "face_crop_hint",
        "hair_crop_hint",
        "outfit_crop_hint",
        "palette_csv",
        "backend_report_json",
    )
    FUNCTION = "analyze"
    CATEGORY = "AnimeConsistency/Character"

    def analyze(
        self,
        reference_image: torch.Tensor,
        character_name: str,
        trigger_word: str,
        use_multi_reference: bool,
        remove_background: bool,
        split_face_hair_outfit: bool,
        extra_tags_csv: str,
        use_face_module: bool,
        use_hair_module: bool,
        use_outfit_module: bool,
        use_body_module: bool,
        use_accessory_module: bool,
        face_tags_csv: str,
        hair_tags_csv: str,
        outfit_tags_csv: str,
        body_tags_csv: str,
        accessory_tags_csv: str,
        backend_mode: str,
        run_backend: bool,
        local_backend_model_name: str,
        local_backend_use_pretrained: bool,
        local_backend_checkpoint_path: str,
        remote_api_url: str,
        remote_api_key: str,
        remote_api_model: str,
        remote_api_timeout_s: int,
        remote_output_schema_json: str,
        remote_extra_headers_json: str,
    ):
        palette = palette_from_image(reference_image, top_k=6)
        extra_tags = parse_csv_tags(extra_tags_csv)
        face_tags = parse_csv_tags(face_tags_csv)
        hair_tags = parse_csv_tags(hair_tags_csv)
        outfit_tags = parse_csv_tags(outfit_tags_csv)
        body_tags = parse_csv_tags(body_tags_csv)
        accessory_tags = parse_csv_tags(accessory_tags_csv)

        module_constraints: Dict[str, Dict[str, Any]] = {
            "face": {"enabled": bool(use_face_module), "strength": 0.80, "tags": face_tags, "status": "pending_backend"},
            "hair": {"enabled": bool(use_hair_module), "strength": 0.75, "tags": hair_tags, "status": "pending_backend"},
            "outfit": {"enabled": bool(use_outfit_module), "strength": 0.75, "tags": outfit_tags, "status": "pending_backend"},
            "body": {"enabled": bool(use_body_module), "strength": 0.65, "tags": body_tags, "status": "pending_backend"},
            "accessory": {"enabled": bool(use_accessory_module), "strength": 0.60, "tags": accessory_tags, "status": "pending_backend"},
        }

        backend_report = run_character_backend(
            backend_mode=backend_mode if run_backend else "placeholder",
            reference_image=reference_image,
            character_name=character_name,
            trigger_word=trigger_word,
            module_constraints=module_constraints,
            palette=palette,
            local_model_name=local_backend_model_name,
            local_use_pretrained=local_backend_use_pretrained,
            local_checkpoint_path=local_backend_checkpoint_path,
            remote_api_url=remote_api_url,
            remote_api_key=remote_api_key,
            remote_api_model=remote_api_model,
            remote_timeout_s=remote_api_timeout_s,
            remote_output_schema_json=remote_output_schema_json,
            remote_extra_headers_json=remote_extra_headers_json,
        )

        report_module_tags = to_dict(backend_report.get("module_tags", {}))
        for module_name in ("face", "hair", "outfit", "body", "accessory"):
            mod = to_dict(module_constraints.get(module_name))
            tags = mod.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            inferred_tags = report_module_tags.get(module_name, [])
            if not isinstance(inferred_tags, list):
                inferred_tags = []
            inferred_tags = [str(t) for t in inferred_tags if isinstance(t, str)]
            mod["tags"] = merge_tags(tags, inferred_tags)
            mod["inferred_tags"] = inferred_tags
            module_constraints[module_name] = mod

        identity = CharacterIdentity(
            name=character_name.strip() or "my_character",
            trigger_word=trigger_word.strip() or "char_tok",
            visual_tags=extra_tags,
            face_tags=face_tags,
            hair_tags=hair_tags,
            outfit_tags=outfit_tags,
            body_tags=body_tags,
            accessory_tags=accessory_tags,
            color_palette=palette,
            module_constraints=module_constraints,
            reference_embeddings={
                "backend": backend_mode if run_backend else "placeholder",
                "use_multi_reference": bool(use_multi_reference),
                "remove_background": bool(remove_background),
                "split_face_hair_outfit": bool(split_face_hair_outfit),
                "backend_report": backend_report,
            },
            masks={},
        )

        prompt_parts: List[str] = [identity.trigger_word, "anime character", "2d illustration"]
        prompt_parts.extend(extra_tags)
        for module_name in ("face", "hair", "outfit", "body", "accessory"):
            mod = module_constraints[module_name]
            if mod.get("enabled"):
                prompt_parts.extend(mod.get("tags", []))

        # Add human-readable hair color hint derived from backend profile.
        hair_profile = to_dict(backend_report.get("hair_color_profile", {}))
        hair_family = str(hair_profile.get("primary_family", "")).strip()
        hair_hex = str(hair_profile.get("primary_hex", "")).strip()
        hair_prompt = self._hair_family_to_prompt(hair_family)
        if hair_prompt:
            prompt_parts.append(hair_prompt)
        if hair_hex:
            prompt_parts.append(f"hair color {hair_hex}")
        character_prompt = ", ".join(prompt_parts)

        face_crop_hint = "auto_face_crop_pending" if use_face_module else "face_module_disabled"
        hair_crop_hint = "auto_hair_crop_pending" if use_hair_module else "hair_module_disabled"
        outfit_crop_hint = "auto_outfit_crop_pending" if use_outfit_module else "outfit_module_disabled"

        return (
            asdict(identity),
            character_prompt,
            face_crop_hint,
            hair_crop_hint,
            outfit_crop_hint,
            ",".join(palette),
            json.dumps(backend_report, ensure_ascii=False),
        )
