from __future__ import annotations

import json
import hashlib
import math
import os
import re
from datetime import datetime
from typing import Any, Dict

import torch

from .color_features import extract_hair_color_profile
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


def _resolve_output_dir() -> str:
    try:
        import folder_paths  # type: ignore

        out = folder_paths.get_output_directory()
        if isinstance(out, str) and out:
            return out
    except Exception:
        pass
    return os.path.join(os.getcwd(), "output")


def _safe_token(text: str, fallback: str = "na") -> str:
    t = str(text or "").strip().lower()
    if not t:
        return fallback
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", t)
    t = t.strip("._-")
    return t or fallback


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return "{}"


def _image_signature_sha256(image: torch.Tensor) -> str:
    if image is None or image.numel() == 0:
        return "empty_image"
    sample = image
    if sample.dim() == 4 and int(sample.shape[0]) > 0:
        sample = sample[0]
    if sample.dim() != 3 or int(sample.shape[-1]) != 3:
        return "invalid_image_shape"

    q = (sample.detach().to("cpu").clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(int(x) for x in q.shape)).encode("utf-8"))
    try:
        h.update(q.numpy().tobytes())
    except Exception:
        return "image_hash_failed"
    return h.hexdigest()


def _resolve_cache_dir() -> str:
    return os.path.join(_resolve_output_dir(), "anime_consistency", "debug_cache")


def _cache_path(cache_key: str) -> str:
    key = _safe_token(cache_key, "cache")
    return os.path.join(_resolve_cache_dir(), f"{key}.json")


def _load_cache(cache_key: str) -> Dict[str, Any]:
    path = _cache_path(cache_key)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _save_cache(cache_key: str, payload: Dict[str, Any]) -> str:
    os.makedirs(_resolve_cache_dir(), exist_ok=True)
    path = _cache_path(cache_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


class ConsistencyDebugViewer:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_image": ("IMAGE",),
                "hair_color_deltae_threshold": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "hair_color_deltae_softness": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "use_cache": ("BOOLEAN", {"default": True}),
                "save_to_file": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "anime_consistency/debug_eval", "multiline": False}),
                "run_tag": ("STRING", {"default": "test", "multiline": False}),
            },
            "optional": {
                "backend_report_json": ("STRING", {"default": "{}", "multiline": True, "forceInput": True}),
                "debug_report_json": ("STRING", {"default": "{}", "multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "summary_text",
        "backend_status",
        "hair_deltae_status",
        "fusion_status",
        "debug_report_json",
        "saved_path",
    )
    FUNCTION = "summarize"
    CATEGORY = "xiling_anime_Consistency/Debug"

    @classmethod
    def IS_CHANGED(
        cls,
        current_image: torch.Tensor,
        hair_color_deltae_threshold: float,
        hair_color_deltae_softness: float,
        use_cache: bool,
        save_to_file: bool,
        filename_prefix: str,
        run_tag: str,
        backend_report_json: str = "{}",
        debug_report_json: str = "{}",
    ):
        return float("NaN")

    def summarize(
        self,
        current_image: torch.Tensor,
        hair_color_deltae_threshold: float,
        hair_color_deltae_softness: float,
        use_cache: bool,
        save_to_file: bool,
        filename_prefix: str,
        run_tag: str,
        backend_report_json: str = "{}",
        debug_report_json: str = "{}",
    ):
        backend = _parse_json(backend_report_json)
        debug_cfg = _parse_json(debug_report_json)
        direct_fusion = to_dict(debug_cfg.get("direct_fusion", {}))
        fusion_status = str(direct_fusion.get("status", "missing"))
        fusion_encode_status = str(direct_fusion.get("encode_status", "missing"))
        fusion_fused_items = int(direct_fusion.get("fused_items", 0)) if str(direct_fusion.get("fused_items", "")).strip() else 0

        backend_status = str(backend.get("status", "unknown"))
        backend_mode = str(backend.get("backend_mode", "unknown"))

        hair_profile = to_dict(backend.get("hair_color_profile", {}))
        primary_hex = str(hair_profile.get("primary_hex", ""))
        primary_family = str(hair_profile.get("primary_family", ""))

        cfg_constraint = to_dict(debug_cfg.get("hair_color_constraint", {}))
        threshold = float(cfg_constraint.get("deltae_threshold", hair_color_deltae_threshold))
        softness = float(cfg_constraint.get("deltae_softness", hair_color_deltae_softness))
        threshold = max(0.0, float(threshold))
        softness = max(0.1, float(softness))

        def lab_from_profile(profile: Dict[str, Any]):
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

        target_lab = lab_from_profile(hair_profile)
        image_sha = _image_signature_sha256(current_image)
        cache_key_source = {
            "image_sha256": image_sha,
            "threshold": round(float(threshold), 4),
            "softness": round(float(softness), 4),
            "backend": backend,
            "debug_cfg": debug_cfg,
        }
        cache_key = hashlib.sha256(_stable_json(cache_key_source).encode("utf-8")).hexdigest()
        cache_status = "disabled"

        if bool(use_cache):
            cached = _load_cache(cache_key)
            if cached:
                summary_text = str(cached.get("summary_text", ""))
                backend_status_cached = str(cached.get("backend_status", backend_status))
                hair_deltae_status_cached = str(cached.get("hair_deltae_status", "deltaE=unavailable / status=cache_corrupt"))
                fusion_status_cached = str(cached.get("fusion_status", fusion_status))
                debug_report_cached = to_dict(cached.get("debug_report", {}))
                if summary_text:
                    if "cache_status=" not in summary_text:
                        summary_text = summary_text + "\ncache_status=hit"
                    else:
                        summary_text = re.sub(r"cache_status=[^\n]*", "cache_status=hit", summary_text)
                else:
                    summary_text = "cache_status=hit"
                debug_report_cached["cache"] = {
                    "status": "hit",
                    "key": cache_key,
                    "image_sha256": image_sha,
                }
                return {
                    "ui": {
                        "summary_text": [summary_text],
                        "backend_status": [backend_status_cached],
                        "hair_deltae_status": [hair_deltae_status_cached],
                        "fusion_status": [fusion_status_cached],
                        "saved_path": [str(cached.get("saved_path", ""))],
                    },
                    "result": (
                        summary_text,
                        backend_status_cached,
                        hair_deltae_status_cached,
                        fusion_status_cached,
                        json.dumps(debug_report_cached, ensure_ascii=False),
                        str(cached.get("saved_path", "")),
                    ),
                }
            cache_status = "miss"

        current_hair_profile = extract_hair_color_profile(current_image, top_k=4)
        current_lab = lab_from_profile(current_hair_profile)

        deltae = None
        pass_flag = None
        penalty = 0.0
        constraint_status = "missing_target_or_current_image"
        if target_lab is not None and current_lab is not None:
            dl = float(target_lab[0]) - float(current_lab[0])
            da = float(target_lab[1]) - float(current_lab[1])
            db = float(target_lab[2]) - float(current_lab[2])
            deltae = float(math.sqrt(dl * dl + da * da + db * db))
            pass_flag = deltae <= threshold
            over = max(0.0, deltae - threshold)
            penalty = over / (over + max(0.1, softness)) if over > 0.0 else 0.0
            constraint_status = "ok"

        if isinstance(deltae, (int, float)) and isinstance(threshold, (int, float)):
            hair_deltae_status = f"deltaE={round(float(deltae), 4)} / threshold={round(float(threshold), 4)} / pass={pass_flag}"
        else:
            hair_deltae_status = f"deltaE=unavailable / status={constraint_status}"

        summary_lines = [
            f"backend_mode={backend_mode}",
            f"backend_status={backend_status}",
            f"fusion_status={fusion_status}",
            f"fusion_encode_status={fusion_encode_status}",
            f"fusion_fused_items={fusion_fused_items}",
            f"hair_primary_hex={primary_hex}",
            f"hair_primary_family={primary_family}",
            hair_deltae_status,
            f"cache_status={cache_status}",
        ]
        debug_report = {
            "mode": "character_lock_debug_eval",
            "hair_color_constraint": {
                "enabled": True,
                "deltae_threshold": round(float(threshold), 4),
                "deltae_softness": round(float(softness), 4),
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
                "deltae": round(float(deltae), 4) if deltae is not None else None,
                "pass": pass_flag,
                "penalty": round(float(penalty), 6),
                "status": constraint_status,
            },
            "target_hair_profile": hair_profile,
            "current_hair_profile": current_hair_profile,
            "input_debug_config": debug_cfg,
            "direct_fusion": direct_fusion,
            "cache": {
                "status": cache_status,
                "key": cache_key if bool(use_cache) else "",
                "image_sha256": image_sha,
            },
        }
        summary_text = "\n".join(summary_lines)
        debug_report_json_out = json.dumps(debug_report, ensure_ascii=False)
        saved_path = ""

        if bool(save_to_file):
            output_dir = _resolve_output_dir()
            safe_prefix = str(filename_prefix or "anime_consistency/debug_eval").replace("\\", "/").strip("/")
            if not safe_prefix:
                safe_prefix = "anime_consistency/debug_eval"
            subdir = os.path.dirname(safe_prefix)
            stem = os.path.basename(safe_prefix) or "debug_eval"

            final_dir = os.path.join(output_dir, subdir) if subdir else output_dir
            os.makedirs(final_dir, exist_ok=True)

            mode_token = _safe_token(backend_mode, "mode")
            status_token = _safe_token(constraint_status, "status")
            tag_token = _safe_token(run_tag, "run")
            de_token = "na"
            if isinstance(deltae, (int, float)):
                de_token = f"{float(deltae):.2f}".replace(".", "p")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{stem}_{tag_token}_{mode_token}_{status_token}_de{de_token}_{ts}.json"
            path = os.path.join(final_dir, filename)

            payload = {
                "summary_text": summary_text,
                "backend_status": backend_status,
                "hair_deltae_status": hair_deltae_status,
                "fusion_status": fusion_status,
                "fusion_encode_status": fusion_encode_status,
                "fusion_fused_items": fusion_fused_items,
                "debug_report": debug_report,
                "backend_report": backend,
                "input_debug_config": debug_cfg,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            saved_path = path
            print(f"[ConsistencyDebugViewer] Saved debug report: {path}")

        if bool(use_cache):
            try:
                _save_cache(
                    cache_key,
                    {
                        "summary_text": summary_text,
                        "backend_status": backend_status,
                        "hair_deltae_status": hair_deltae_status,
                        "fusion_status": fusion_status,
                        "debug_report": debug_report,
                        "saved_path": saved_path,
                    },
                )
            except Exception:
                pass

        return {
            "ui": {
                "summary_text": [summary_text],
                "backend_status": [backend_status],
                "hair_deltae_status": [hair_deltae_status],
                "fusion_status": [fusion_status],
                "saved_path": [saved_path],
            },
            "result": (summary_text, backend_status, hair_deltae_status, fusion_status, debug_report_json_out, saved_path),
        }
