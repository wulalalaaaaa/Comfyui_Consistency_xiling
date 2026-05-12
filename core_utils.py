from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import torch


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_csv_tags(csv_text: str) -> List[str]:
    if not csv_text:
        return []
    return [t.strip() for t in csv_text.split(",") if t.strip()]


def to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def merge_tags(primary: List[str], secondary: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for src in (primary, secondary):
        for tag in src:
            if not isinstance(tag, str):
                continue
            t = tag.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def ensure_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        t = item.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    s = text.strip()
    if not s:
        return {}
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    snippet = s[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def palette_from_image(image: torch.Tensor, top_k: int = 5) -> List[str]:
    if image is None or image.numel() == 0:
        return []
    sample = image[0]
    if sample.dim() != 3 or sample.shape[-1] != 3:
        return []

    pixels = sample.reshape(-1, 3).clamp(0.0, 1.0)
    if pixels.shape[0] == 0:
        return []

    step = max(1, pixels.shape[0] // 50_000)
    pixels = pixels[::step]

    q = (pixels * 15.0).round().to(torch.int64).clamp(0, 15)
    bins = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    uniq, counts = torch.unique(bins, return_counts=True)
    order = torch.argsort(counts, descending=True)

    palette = []
    for idx in order[:top_k]:
        b = int(uniq[idx].item())
        r = (b // 256) & 15
        g = (b // 16) & 15
        bl = b & 15
        rr = int(round(r / 15.0 * 255))
        gg = int(round(g / 15.0 * 255))
        bb = int(round(bl / 15.0 * 255))
        palette.append(f"#{rr:02X}{gg:02X}{bb:02X}")
    return palette


def deepcopy_conditioning(conditioning: List[Any]) -> List[Any]:
    return copy.deepcopy(conditioning)


def append_note_to_conditioning(conditioning: List[Any], note: Dict[str, Any]) -> List[Any]:
    out = deepcopy_conditioning(conditioning)
    for i, item in enumerate(out):
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict):
            meta = copy.deepcopy(item[1])
            meta.setdefault("anime_consistency", {})
            meta["anime_consistency"].update(note)
            out[i] = [item[0], meta]
    return out

