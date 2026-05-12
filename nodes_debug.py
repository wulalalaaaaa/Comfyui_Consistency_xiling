from __future__ import annotations

import json
from typing import Any, Dict

from .core_utils import to_dict


def _parse_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    s = text.strip()
    if not s:
        return {}
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


class ConsistencyDebugViewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend_report_json": ("STRING", {"default": "{}", "multiline": True}),
                "debug_report_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("summary_text", "backend_status", "hair_deltae_status")
    FUNCTION = "summarize"
    CATEGORY = "AnimeConsistency/Debug"

    def summarize(self, backend_report_json: str, debug_report_json: str):
        backend = _parse_json(backend_report_json)
        debug = _parse_json(debug_report_json)

        backend_status = str(backend.get("status", "unknown"))
        backend_mode = str(backend.get("backend_mode", "unknown"))

        hair_profile = to_dict(backend.get("hair_color_profile", {}))
        primary_hex = str(hair_profile.get("primary_hex", ""))
        primary_family = str(hair_profile.get("primary_family", ""))

        color_constraint = to_dict(debug.get("hair_color_constraint", {}))
        deltae = color_constraint.get("deltae", None)
        threshold = color_constraint.get("deltae_threshold", None)
        pass_flag = color_constraint.get("pass", None)
        constraint_status = str(color_constraint.get("status", "unknown"))

        if isinstance(deltae, (int, float)) and isinstance(threshold, (int, float)):
            hair_deltae_status = f"deltaE={round(float(deltae), 4)} / threshold={round(float(threshold), 4)} / pass={pass_flag}"
        else:
            hair_deltae_status = f"deltaE=unavailable / status={constraint_status}"

        summary_lines = [
            f"backend_mode={backend_mode}",
            f"backend_status={backend_status}",
            f"hair_primary_hex={primary_hex}",
            f"hair_primary_family={primary_family}",
            hair_deltae_status,
        ]
        return "\n".join(summary_lines), backend_status, hair_deltae_status

