from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import torch

from .color_features import dominant_color_records, extract_hair_color_profile, filter_chromatic_pixels, region_pixels
from .core_utils import (
    ensure_str_list,
    extract_json_object,
    merge_tags,
    palette_from_image,
    to_dict,
)


def normalize_backend_schema(parsed_content: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    normalized: Dict[str, Any] = {}

    global_tags = parsed_content.get("global_tags", [])
    if not isinstance(global_tags, list):
        errors.append("global_tags must be a list of strings")
        global_tags = []
    normalized["global_tags"] = ensure_str_list(global_tags)

    module_tags_in = parsed_content.get("module_tags", {})
    if not isinstance(module_tags_in, dict):
        errors.append("module_tags must be an object")
        module_tags_in = {}

    module_tags_out: Dict[str, List[str]] = {}
    for module_name in ("face", "hair", "outfit", "body", "accessory"):
        value = module_tags_in.get(module_name, [])
        if not isinstance(value, list):
            errors.append(f"module_tags.{module_name} must be a list of strings")
            value = []
        module_tags_out[module_name] = ensure_str_list(value)
    normalized["module_tags"] = module_tags_out

    notes = parsed_content.get("notes", "")
    if not isinstance(notes, str):
        errors.append("notes must be a string")
        notes = str(notes)
    normalized["notes"] = notes

    confidence = parsed_content.get("confidence", "unknown")
    if not isinstance(confidence, str):
        errors.append("confidence must be a string")
        confidence = str(confidence)
    normalized["confidence"] = confidence
    return normalized, errors


def build_remote_endpoint_candidates(api_url: str) -> List[Tuple[str, str]]:
    u = (api_url or "").strip().rstrip("/")
    if not u:
        return []
    if u.endswith("/v1/responses"):
        return [("responses", u)]
    if u.endswith("/v1/chat/completions"):
        return [("chat_completions", u)]
    if u.endswith("/v1"):
        return [("responses", f"{u}/responses"), ("chat_completions", f"{u}/chat/completions")]
    return [("responses", f"{u}/v1/responses"), ("chat_completions", f"{u}/v1/chat/completions")]


def build_remote_payload(api_kind: str, model: str, user_prompt: str, image_data_url: str) -> Dict[str, Any]:
    if api_kind == "responses":
        return {
            "model": model,
            "temperature": 0.2,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a character-consistency vision analyzer. Return JSON only."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            ],
        }
    return {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a character-consistency vision analyzer. Return JSON only."},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": image_data_url}}]},
        ],
    }


def extract_response_content_text(api_kind: str, raw_data: Dict[str, Any], raw_text: str) -> str:
    if api_kind == "chat_completions":
        choices = raw_data.get("choices", []) if isinstance(raw_data, dict) else []
        if isinstance(choices, list) and choices:
            msg = to_dict(to_dict(choices[0]).get("message", {}))
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    txt = to_dict(part).get("text", "")
                    if isinstance(txt, str):
                        parts.append(txt)
                return "\n".join(parts)
        return raw_text

    if isinstance(raw_data, dict):
        output = raw_data.get("output", [])
        if isinstance(output, list):
            text_parts: List[str] = []
            for item in output:
                content_list = to_dict(item).get("content", [])
                if not isinstance(content_list, list):
                    continue
                for c in content_list:
                    txt = to_dict(c).get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        text_parts.append(txt)
            if text_parts:
                return "\n".join(text_parts)
        output_text = raw_data.get("output_text", "")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
    return raw_text


def image_to_data_url(image: torch.Tensor) -> Tuple[str, str]:
    try:
        from PIL import Image
    except Exception as e:
        return "", f"pil_unavailable:{str(e)}"
    if image is None or image.numel() == 0:
        return "", "empty_image"
    sample = image[0]
    if sample.dim() != 3 or sample.shape[-1] != 3:
        return "", "invalid_image_shape"
    try:
        sample_uint8 = sample.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
        pil_image = Image.fromarray(sample_uint8, mode="RGB")
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}", ""
    except Exception as e:
        return "", f"encode_failed:{str(e)}"


def try_vit_embedding(reference_image: torch.Tensor, model_name: str, use_pretrained: bool) -> Dict[str, Any]:
    if reference_image is None or reference_image.numel() == 0:
        return {"status": "skipped", "reason": "empty_image"}
    sample = reference_image[0]
    if sample.dim() != 3 or sample.shape[-1] != 3:
        return {"status": "skipped", "reason": "invalid_image_shape"}

    try:
        import timm  # type: ignore

        model = timm.create_model(model_name.strip() or "vit_base_patch16_224", pretrained=bool(use_pretrained), num_classes=0)
        model.eval()
        x = sample.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feat = model(x)
        feat = feat.reshape(-1).detach().cpu()
        return {
            "status": "ok",
            "provider": "timm",
            "embedding_dim": int(feat.numel()),
            "embedding_preview": [round(float(v), 6) for v in feat[:8]],
            "pretrained": bool(use_pretrained),
            "note": "embedding scaffold; semantic quality depends on selected weights",
        }
    except Exception as e:
        return {"status": "unavailable", "provider": "timm", "error": str(e)}


def try_vit_embedding_with_checkpoint(
    reference_image: torch.Tensor,
    model_name: str,
    use_pretrained: bool,
    checkpoint_path: str,
) -> Dict[str, Any]:
    ckpt = str(checkpoint_path or "").strip()
    if not ckpt:
        return try_vit_embedding(reference_image, model_name, use_pretrained)

    if reference_image is None or reference_image.numel() == 0:
        return {"status": "skipped", "reason": "empty_image"}
    sample = reference_image[0]
    if sample.dim() != 3 or sample.shape[-1] != 3:
        return {"status": "skipped", "reason": "invalid_image_shape"}

    try:
        import os
        import timm  # type: ignore

        exists = os.path.exists(ckpt)
        if not exists:
            return {"status": "unavailable", "provider": "timm", "error": f"checkpoint_not_found:{ckpt}"}

        model = timm.create_model(
            model_name.strip() or "vit_base_patch16_224",
            pretrained=bool(use_pretrained),
            checkpoint_path=ckpt,
            num_classes=0,
        )
        model.eval()
        x = sample.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feat = model(x)
        feat = feat.reshape(-1).detach().cpu()
        return {
            "status": "ok",
            "provider": "timm",
            "embedding_dim": int(feat.numel()),
            "embedding_preview": [round(float(v), 6) for v in feat[:8]],
            "pretrained": bool(use_pretrained),
            "checkpoint_path": ckpt,
            "checkpoint_loaded": True,
            "note": "embedding scaffold with manual checkpoint_path",
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "provider": "timm",
            "checkpoint_path": ckpt,
            "checkpoint_loaded": False,
            "error": str(e),
        }


def run_local_vision_backend(
    reference_image: torch.Tensor,
    character_name: str,
    module_constraints: Dict[str, Any],
    model_name: str,
    use_pretrained: bool,
    checkpoint_path: str,
) -> Dict[str, Any]:
    def _extract_region_profile(y0: float, y1: float, x0: float, x1: float, top_k: int = 3) -> Dict[str, Any]:
        if reference_image is None or reference_image.numel() == 0:
            return {"status": "empty_image", "dominant_colors": []}
        sample = reference_image[0]
        if sample.dim() != 3 or sample.shape[-1] != 3:
            return {"status": "invalid_image_shape", "dominant_colors": []}
        roi = region_pixels(sample, y0=y0, y1=y1, x0=x0, x1=x1)
        roi_filtered = filter_chromatic_pixels(roi, min_s=0.08, min_v=0.08)
        dominant = dominant_color_records(roi_filtered, top_k=max(1, int(top_k)))
        primary = dominant[0] if dominant else {}
        return {
            "status": "ok" if dominant else "low_signal",
            "roi": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
            "pixel_count": int(roi_filtered.shape[0]) if roi_filtered is not None else 0,
            "dominant_colors": dominant,
            "primary_hex": str(primary.get("hex", "")),
            "primary_family": str(primary.get("family_label", "")),
        }

    palette = palette_from_image(reference_image, top_k=6)
    inferred_global_tags = []
    if palette:
        inferred_global_tags.append("color-stable-design")
    if character_name.strip():
        inferred_global_tags.append(f"character:{character_name.strip()}")
    inferred_global_tags.append("same-character-identity")

    vit_report = try_vit_embedding_with_checkpoint(
        reference_image,
        model_name=model_name,
        use_pretrained=use_pretrained,
        checkpoint_path=checkpoint_path,
    )
    hair_color_profile = extract_hair_color_profile(reference_image, top_k=4)
    eye_color_profile = _extract_region_profile(y0=0.24, y1=0.48, x0=0.28, x1=0.72, top_k=3)
    outfit_color_profile = _extract_region_profile(y0=0.48, y1=0.96, x0=0.12, x1=0.88, top_k=4)

    module_tags: Dict[str, List[str]] = {}
    for module_name in ("face", "hair", "outfit", "body", "accessory"):
        mod = to_dict(module_constraints.get(module_name))
        base_tags = mod.get("tags", [])
        if not isinstance(base_tags, list):
            base_tags = []
        inferred: List[str] = []
        if module_name == "face":
            inferred.extend(["same face identity", "same eye shape"])
            eye_hex = str(eye_color_profile.get("primary_hex", "")).strip()
            eye_family = str(eye_color_profile.get("primary_family", "")).strip()
            if eye_family:
                inferred.append(f"eye_color_family:{eye_family}")
            if eye_hex:
                inferred.append(f"eye_color_hex:{eye_hex}")
        if module_name == "hair":
            inferred.append("same hairstyle")
            primary_hex = str(hair_color_profile.get("primary_hex", "")).strip()
            primary_family = str(hair_color_profile.get("primary_family", "")).strip()
            if primary_hex:
                inferred.append(f"hair_color_hex:{primary_hex}")
            if primary_family:
                inferred.append(f"hair_color_family:{primary_family}")
        if module_name == "outfit":
            inferred.extend(["same outfit design", "same outfit color palette"])
            outfit_hex = str(outfit_color_profile.get("primary_hex", "")).strip()
            outfit_family = str(outfit_color_profile.get("primary_family", "")).strip()
            if outfit_family:
                inferred.append(f"outfit_color_family:{outfit_family}")
            if outfit_hex:
                inferred.append(f"outfit_color_hex:{outfit_hex}")
        if module_name == "body":
            inferred.append("same body silhouette")
        if module_name == "accessory":
            inferred.append("same accessories")
        module_tags[module_name] = merge_tags(base_tags, inferred)

    return {
        "backend_mode": "local_vision",
        "status": "ok",
        "model_name": model_name.strip() or "vit_base_patch16_224",
        "summary": "local backend produced structured color profile + optional vit embedding",
        "global_tags": inferred_global_tags,
        "module_tags": module_tags,
        "hair_color_profile": hair_color_profile,
        "eye_color_profile": eye_color_profile,
        "outfit_color_profile": outfit_color_profile,
        "vit_embedding": vit_report,
        "confidence": "medium" if hair_color_profile.get("status") == "ok" else "low",
    }


def run_remote_api_backend(
    reference_image: torch.Tensor,
    character_name: str,
    trigger_word: str,
    module_constraints: Dict[str, Any],
    palette: List[str],
    api_url: str,
    api_key: str,
    api_model: str,
    timeout_s: int,
    output_schema_json: str,
    extra_headers_json: str,
) -> Dict[str, Any]:
    url = (api_url or "").strip()
    if not url:
        return {"backend_mode": "remote_api", "status": "error", "error_type": "config_error", "error": "remote_api_url is empty"}

    image_data_url, image_error = image_to_data_url(reference_image)
    if not image_data_url:
        return {"backend_mode": "remote_api", "status": "error", "error_type": "image_error", "error": image_error or "image_encode_failed"}

    schema_hint = output_schema_json.strip() or '{"global_tags":[],"module_tags":{"face":[],"hair":[],"outfit":[],"body":[],"accessory":[]},"notes":"","confidence":"low"}'
    user_prompt = (
        "Analyze this anime-style character reference and output STRICT JSON only.\n"
        f"Character name: {character_name}\n"
        f"Trigger word: {trigger_word}\n"
        f"Palette hint: {palette}\n"
        f"Output schema hint: {schema_hint}"
    )

    model_name = (api_model or "gpt-4.1-mini").strip()
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    extra_headers = extract_json_object(extra_headers_json)
    for k, v in extra_headers.items():
        if isinstance(k, str) and isinstance(v, str):
            headers[k] = v

    candidates = build_remote_endpoint_candidates(url)
    if not candidates:
        return {"backend_mode": "remote_api", "status": "error", "error_type": "config_error", "error": "unable to build remote endpoint candidates"}

    errors: List[Dict[str, Any]] = []
    timeout = max(5, int(timeout_s))
    for api_kind, endpoint_url in candidates:
        payload = build_remote_payload(api_kind, model_name, user_prompt, image_data_url)
        request_body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url=endpoint_url, data=request_body, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")
                status_code = int(getattr(resp, "status", 200))
        except urlerror.HTTPError as e:
            err_text = ""
            try:
                err_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_text = str(e)
            errors.append({"api_kind": api_kind, "endpoint_url": endpoint_url, "error_type": "http_error", "http_status": int(e.code), "error": err_text[:500]})
            continue
        except Exception as e:
            errors.append({"api_kind": api_kind, "endpoint_url": endpoint_url, "error_type": "request_error", "error": str(e)})
            continue

        raw_data = extract_json_object(raw_text)
        content_text = extract_response_content_text(api_kind, raw_data, raw_text)
        parsed_content = extract_json_object(content_text) or extract_json_object(raw_text)
        normalized_content, schema_errors = normalize_backend_schema(parsed_content)
        parsed_module_tags = to_dict(normalized_content.get("module_tags", {}))

        module_tags: Dict[str, List[str]] = {}
        for module_name in ("face", "hair", "outfit", "body", "accessory"):
            base = to_dict(module_constraints.get(module_name))
            base_tags = base.get("tags", [])
            if not isinstance(base_tags, list):
                base_tags = []
            remote_tags = parsed_module_tags.get(module_name, [])
            if not isinstance(remote_tags, list):
                remote_tags = []
            module_tags[module_name] = merge_tags(base_tags, remote_tags)

        return {
            "backend_mode": "remote_api",
            "status": "ok",
            "api_kind": api_kind,
            "endpoint_url": endpoint_url,
            "http_status": status_code,
            "model_name": model_name,
            "global_tags": ensure_str_list(normalized_content.get("global_tags", [])),
            "module_tags": module_tags,
            "notes": normalized_content.get("notes", ""),
            "confidence": normalized_content.get("confidence", "unknown"),
            "schema_valid": len(schema_errors) == 0,
            "schema_errors": schema_errors,
            "raw_response_excerpt": raw_text[:1000],
        }

    return {
        "backend_mode": "remote_api",
        "status": "error",
        "error_type": "all_endpoints_failed",
        "tried_endpoints": [endpoint for _, endpoint in candidates],
        "errors": errors,
    }


def run_character_backend(
    backend_mode: str,
    reference_image: torch.Tensor,
    character_name: str,
    trigger_word: str,
    module_constraints: Dict[str, Any],
    palette: List[str],
    local_model_name: str,
    local_use_pretrained: bool,
    local_checkpoint_path: str,
    remote_api_url: str,
    remote_api_key: str,
    remote_api_model: str,
    remote_timeout_s: int,
    remote_output_schema_json: str,
    remote_extra_headers_json: str,
) -> Dict[str, Any]:
    mode = (backend_mode or "placeholder").strip()
    if mode == "local_vision":
        return run_local_vision_backend(
            reference_image,
            character_name,
            module_constraints,
            local_model_name,
            local_use_pretrained,
            local_checkpoint_path,
        )
    if mode == "remote_api":
        return run_remote_api_backend(
            reference_image=reference_image,
            character_name=character_name,
            trigger_word=trigger_word,
            module_constraints=module_constraints,
            palette=palette,
            api_url=remote_api_url,
            api_key=remote_api_key,
            api_model=remote_api_model,
            timeout_s=remote_timeout_s,
            output_schema_json=remote_output_schema_json,
            extra_headers_json=remote_extra_headers_json,
        )
    return {"backend_mode": "placeholder", "status": "skipped", "summary": "backend execution disabled or placeholder mode selected"}
