from __future__ import annotations

from dataclasses import asdict

import torch

from .core_utils import palette_from_image, parse_csv_tags
from .models import StyleIdentity


class StyleReferenceAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_reference_image": ("IMAGE",),
                "style_name": ("STRING", {"default": "target_style", "multiline": False}),
                "style_region_mode": (["full_image", "background_only", "character_only", "manual_mask"],),
                "avoid_content_leakage": ("BOOLEAN", {"default": True}),
                "style_tags_csv": ("STRING", {"default": "anime style, clean lineart", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STYLE_IDENTITY", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("style_identity", "style_prompt", "palette_csv", "line_hint", "texture_hint")
    FUNCTION = "analyze"
    CATEGORY = "xiling_anime_Consistency/Style"

    def analyze(
        self,
        style_reference_image: torch.Tensor,
        style_name: str,
        style_region_mode: str,
        avoid_content_leakage: bool,
        style_tags_csv: str,
    ):
        palette = palette_from_image(style_reference_image, top_k=6)
        style_tags = parse_csv_tags(style_tags_csv)

        identity = StyleIdentity(
            style_name=style_name.strip() or "target_style",
            style_tags=style_tags,
            color_palette=palette,
            line_features={"line_hint": "lineart_thickness_pending"},
            shading_features={"shading_hint": "shading_type_pending"},
            texture_features={"texture_hint": "texture_detail_pending"},
            reference_embeddings={"backend": "placeholder", "style_region_mode": style_region_mode},
            leakage_control={"avoid_content_leakage": bool(avoid_content_leakage), "method": "placeholder"},
        )

        style_prompt = ", ".join(style_tags) if style_tags else "anime style"
        return (
            asdict(identity),
            style_prompt,
            ",".join(palette),
            "lineart_thickness_pending",
            "texture_detail_pending",
        )
