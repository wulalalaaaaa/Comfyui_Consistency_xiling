from __future__ import annotations

import colorsys
from typing import Any, Dict, List

import torch

from .core_utils import clamp01


def rgb01_to_hex(rgb: torch.Tensor) -> str:
    if rgb.numel() != 3:
        return "#000000"
    r = int(round(clamp01(float(rgb[0].item())) * 255))
    g = int(round(clamp01(float(rgb[1].item())) * 255))
    b = int(round(clamp01(float(rgb[2].item())) * 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb01_to_hsv(rgb: torch.Tensor) -> Dict[str, float]:
    r = clamp01(float(rgb[0].item()))
    g = clamp01(float(rgb[1].item()))
    b = clamp01(float(rgb[2].item()))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return {"h": round(h * 360.0, 2), "s": round(s, 4), "v": round(v, 4)}


def _srgb_channel_to_linear(v: float) -> float:
    vv = clamp01(v)
    if vv <= 0.04045:
        return vv / 12.92
    return ((vv + 0.055) / 1.055) ** 2.4


def rgb01_to_lab(rgb: torch.Tensor) -> Dict[str, float]:
    r = _srgb_channel_to_linear(float(rgb[0].item()))
    g = _srgb_channel_to_linear(float(rgb[1].item()))
    b = _srgb_channel_to_linear(float(rgb[2].item()))

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    xr = x / 0.95047
    yr = y / 1.00000
    zr = z / 1.08883

    def f(t: float) -> float:
        if t > 0.008856:
            return t ** (1.0 / 3.0)
        return 7.787 * t + (16.0 / 116.0)

    fx = f(xr)
    fy = f(yr)
    fz = f(zr)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return {"l": round(l, 3), "a": round(a, 3), "b": round(bb, 3)}


def color_family_label(hsv: Dict[str, float]) -> str:
    h = float(hsv.get("h", 0.0)) % 360.0
    s = float(hsv.get("s", 0.0))
    v = float(hsv.get("v", 0.0))
    if v < 0.12:
        base = "near-black"
    elif s < 0.10:
        if v > 0.85:
            base = "near-white"
        elif v > 0.45:
            base = "neutral-gray"
        else:
            base = "charcoal-gray"
    elif h < 15 or h >= 345:
        base = "red"
    elif h < 35:
        base = "orange"
    elif h < 60:
        base = "yellow"
    elif h < 95:
        base = "yellow-green"
    elif h < 150:
        base = "green"
    elif h < 190:
        base = "cyan"
    elif h < 235:
        base = "blue"
    elif h < 275:
        base = "violet"
    elif h < 320:
        base = "magenta"
    else:
        base = "rose"

    tone = "dark" if v < 0.35 else ("light" if v > 0.78 else "mid")
    chroma = "muted" if s < 0.35 else ("vivid" if s > 0.70 else "normal")
    return f"{tone}-{chroma}-{base}"


def build_color_record(rgb: torch.Tensor, weight: float) -> Dict[str, Any]:
    hsv = rgb01_to_hsv(rgb)
    return {
        "hex": rgb01_to_hex(rgb),
        "rgb": [round(clamp01(float(c.item())), 4) for c in rgb],
        "hsv": hsv,
        "lab": rgb01_to_lab(rgb),
        "weight": round(clamp01(weight), 4),
        "family_label": color_family_label(hsv),
    }


def dominant_color_records(pixels: torch.Tensor, top_k: int = 4) -> List[Dict[str, Any]]:
    if pixels is None or pixels.numel() == 0:
        return []
    px = pixels.reshape(-1, 3).clamp(0.0, 1.0)
    if px.shape[0] == 0:
        return []

    step = max(1, px.shape[0] // 80_000)
    px = px[::step]
    q = (px * 31.0).round().to(torch.int64).clamp(0, 31)
    bins = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    uniq, counts = torch.unique(bins, return_counts=True)
    order = torch.argsort(counts, descending=True)
    total = max(1, int(counts.sum().item()))

    out: List[Dict[str, Any]] = []
    for idx in order[:top_k]:
        bin_id = uniq[idx]
        count = int(counts[idx].item())
        mask = bins == bin_id
        if int(mask.sum().item()) == 0:
            continue
        mean_rgb = px[mask].mean(dim=0)
        out.append(build_color_record(mean_rgb, float(count) / float(total)))
    return out


def region_pixels(sample: torch.Tensor, y0: float, y1: float, x0: float, x1: float) -> torch.Tensor:
    h = int(sample.shape[0])
    w = int(sample.shape[1])
    y_start = max(0, min(h - 1, int(round(h * y0))))
    y_end = max(y_start + 1, min(h, int(round(h * y1))))
    x_start = max(0, min(w - 1, int(round(w * x0))))
    x_end = max(x_start + 1, min(w, int(round(w * x1))))
    return sample[y_start:y_end, x_start:x_end, :].reshape(-1, 3).clamp(0.0, 1.0)


def filter_chromatic_pixels(pixels: torch.Tensor, min_s: float = 0.14, min_v: float = 0.10) -> torch.Tensor:
    if pixels is None or pixels.numel() == 0:
        return pixels
    maxc, _ = pixels.max(dim=1)
    minc, _ = pixels.min(dim=1)
    delta = maxc - minc
    s = delta / (maxc + 1e-8)
    v = maxc
    mask = (s >= float(min_s)) & (v >= float(min_v))
    if int(mask.sum().item()) >= 128:
        return pixels[mask]
    return pixels


def extract_hair_color_profile(reference_image: torch.Tensor, top_k: int = 4) -> Dict[str, Any]:
    if reference_image is None or reference_image.numel() == 0:
        return {"status": "empty_image", "dominant_colors": []}
    sample = reference_image[0]
    if sample.dim() != 3 or sample.shape[-1] != 3:
        return {"status": "invalid_image_shape", "dominant_colors": []}

    roi = region_pixels(sample, y0=0.02, y1=0.52, x0=0.08, x1=0.92)
    roi_filtered = filter_chromatic_pixels(roi, min_s=0.10, min_v=0.08)
    dominant = dominant_color_records(roi_filtered, top_k=max(1, int(top_k)))
    primary = dominant[0] if dominant else {}
    return {
        "status": "ok" if dominant else "low_signal",
        "roi": {"y0": 0.02, "y1": 0.52, "x0": 0.08, "x1": 0.92},
        "pixel_count": int(roi_filtered.shape[0]) if roi_filtered is not None else 0,
        "dominant_colors": dominant,
        "primary_hex": primary.get("hex", ""),
        "primary_family": primary.get("family_label", ""),
    }

