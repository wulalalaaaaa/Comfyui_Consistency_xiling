from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List

from .core_utils import extract_json_object

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _safe_token(text: str, fallback: str = "na") -> str:
    t = str(text or "").strip().lower()
    if not t:
        return fallback
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", t)
    t = t.strip("._-")
    return t or fallback


def _q(text: str) -> str:
    return '"' + str(text).replace('"', '\\"') + '"'


def _iter_images_from_path(source_path: str) -> List[str]:
    p = os.path.abspath(os.path.expanduser(str(source_path or "").strip()))
    if os.path.isfile(p):
        ext = os.path.splitext(p)[1].lower()
        return [p] if ext in _IMAGE_EXTS else []
    if not os.path.isdir(p):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(p)):
        fp = os.path.join(p, name)
        if not os.path.isfile(fp):
            continue
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS:
            out.append(fp)
    return out


def _read_caption_for_image(image_path: str) -> str:
    cap_path = os.path.splitext(image_path)[0] + ".txt"
    if not os.path.isfile(cap_path):
        return ""
    try:
        with open(cap_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _ensure_trigger(caption: str, trigger_word: str) -> str:
    t = str(trigger_word or "").strip()
    c = str(caption or "").strip()
    if not t:
        return c
    if t.lower() in c.lower():
        return c
    if c:
        return f"{t}, {c}"
    return t


class LoRAQuickDatasetPreparer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image_path": ("STRING", {"default": "", "multiline": False}),
                "prepared_dataset_dir": ("STRING", {"default": "", "multiline": False}),
                "trigger_word": ("STRING", {"default": "char_tok", "multiline": False}),
                "caption_template": ("STRING", {"default": "char_tok, 1girl, portrait", "multiline": False}),
                "duplicate_count": ("INT", {"default": 24, "min": 1, "max": 500, "step": 1}),
                "force_trigger_word": ("BOOLEAN", {"default": True}),
                "preserve_existing_caption": ("BOOLEAN", {"default": True}),
                "strict_single_image": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "quick", "multiline": False}),
                "overwrite_existing_files": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("prepared_dataset_dir", "source_image_count", "prepared_image_count", "generated_caption_count", "report_json")
    FUNCTION = "prepare"
    CATEGORY = "AnimeConsistency/LoRAFast"

    def prepare(
        self,
        source_image_path: str,
        prepared_dataset_dir: str,
        trigger_word: str,
        caption_template: str,
        duplicate_count: int,
        force_trigger_word: bool,
        preserve_existing_caption: bool,
        strict_single_image: bool,
        filename_prefix: str,
        overwrite_existing_files: bool,
    ):
        src_images = _iter_images_from_path(source_image_path)
        dst_dir = os.path.abspath(os.path.expanduser(str(prepared_dataset_dir or "").strip()))
        dup = max(1, int(duplicate_count))
        prefix = _safe_token(filename_prefix, "quick")
        report: Dict[str, Any] = {
            "status": "ok",
            "source_image_path": os.path.abspath(os.path.expanduser(str(source_image_path or "").strip())),
            "prepared_dataset_dir": dst_dir,
            "duplicate_count": dup,
            "strict_single_image": bool(strict_single_image),
        }

        if not src_images:
            report["status"] = "no_source_images"
            return dst_dir, 0, 0, 0, json.dumps(report, ensure_ascii=False)

        if bool(strict_single_image) and len(src_images) != 1:
            report["status"] = "strict_single_image_violation"
            report["source_image_count"] = len(src_images)
            return dst_dir, len(src_images), 0, 0, json.dumps(report, ensure_ascii=False)

        os.makedirs(dst_dir, exist_ok=True)

        prepared_image_count = 0
        generated_caption_count = 0
        skipped_existing: List[str] = []

        for src_idx, src_img in enumerate(src_images):
            base_stem = _safe_token(os.path.splitext(os.path.basename(src_img))[0], f"img{src_idx:03d}")
            ext = os.path.splitext(src_img)[1].lower()

            base_caption = ""
            if bool(preserve_existing_caption):
                base_caption = _read_caption_for_image(src_img)
            if not base_caption:
                base_caption = str(caption_template or "").strip()
            if bool(force_trigger_word):
                base_caption = _ensure_trigger(base_caption, trigger_word)

            for dup_idx in range(dup):
                out_stem = f"{prefix}_{src_idx:03d}_{dup_idx:04d}_{base_stem}"
                out_img = os.path.join(dst_dir, f"{out_stem}{ext}")
                out_cap = os.path.join(dst_dir, f"{out_stem}.txt")

                if (not bool(overwrite_existing_files)) and (os.path.exists(out_img) or os.path.exists(out_cap)):
                    skipped_existing.append(out_stem)
                    continue

                shutil.copy2(src_img, out_img)
                with open(out_cap, "w", encoding="utf-8") as f:
                    f.write(base_caption)
                prepared_image_count += 1
                generated_caption_count += 1

        report["source_image_count"] = len(src_images)
        report["prepared_image_count"] = prepared_image_count
        report["generated_caption_count"] = generated_caption_count
        report["skipped_existing_count"] = len(skipped_existing)
        report["skipped_existing_preview"] = skipped_existing[:20]
        report["recommended_next"] = "Use LoRA Quick Config Builder with speed_preset=single_image_fastest for 1 image."

        return dst_dir, len(src_images), prepared_image_count, generated_caption_count, json.dumps(report, ensure_ascii=False)


class LoRAQuickConfigBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "", "multiline": False}),
                "prepared_dataset_dir": ("STRING", {"default": "", "multiline": False}),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "trigger_word": ("STRING", {"default": "char_tok", "multiline": False}),
                "speed_preset": (["single_image_fastest", "few_image_fast", "few_image_balanced"],),
                "trainer_script_mode": (["auto", "train_network.py", "sdxl_train_network.py"],),
                "save_config_to_file": ("BOOLEAN", {"default": True}),
                "config_filename_prefix": ("STRING", {"default": "lora_quick_config", "multiline": False}),
                "extra_args_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("config_json", "launch_command", "config_path", "summary")
    FUNCTION = "build"
    CATEGORY = "AnimeConsistency/LoRAFast"

    def build(
        self,
        base_model_path: str,
        prepared_dataset_dir: str,
        output_dir: str,
        trigger_word: str,
        speed_preset: str,
        trainer_script_mode: str,
        save_config_to_file: bool,
        config_filename_prefix: str,
        extra_args_json: str,
    ):
        base_model = os.path.abspath(os.path.expanduser(str(base_model_path or "").strip()))
        data_dir = os.path.abspath(os.path.expanduser(str(prepared_dataset_dir or "").strip()))
        out_dir = os.path.abspath(os.path.expanduser(str(output_dir or "").strip()))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_prefix = _safe_token(config_filename_prefix, "lora_quick_config")

        preset_table: Dict[str, Dict[str, Any]] = {
            "single_image_fastest": {
                "resolution": 768,
                "network_dim": 8,
                "network_alpha": 8,
                "unet_lr": 2e-4,
                "text_encoder_lr": 0.0,
                "batch_size": 1,
                "max_train_steps": 180,
                "save_every_n_steps": 60,
                "train_unet": True,
                "train_text_encoder": False,
                "mixed_precision": "fp16",
                "notes": "Fastest mode for 1 image. Overfit risk is high; use for emergency style lock only.",
            },
            "few_image_fast": {
                "resolution": 768,
                "network_dim": 16,
                "network_alpha": 16,
                "unet_lr": 1.5e-4,
                "text_encoder_lr": 0.0,
                "batch_size": 1,
                "max_train_steps": 350,
                "save_every_n_steps": 100,
                "train_unet": True,
                "train_text_encoder": False,
                "mixed_precision": "fp16",
                "notes": "Fast mode for 2-8 images. Better generalization than single-image preset.",
            },
            "few_image_balanced": {
                "resolution": 1024,
                "network_dim": 16,
                "network_alpha": 16,
                "unet_lr": 1e-4,
                "text_encoder_lr": 5e-6,
                "batch_size": 1,
                "max_train_steps": 700,
                "save_every_n_steps": 150,
                "train_unet": True,
                "train_text_encoder": True,
                "mixed_precision": "fp16",
                "notes": "Balanced mode for 4-20 images. More stable identity and color retention.",
            },
        }
        preset = preset_table.get(speed_preset, preset_table["few_image_fast"])

        extra_args = extract_json_object(extra_args_json)
        if not isinstance(extra_args, dict):
            extra_args = {}

        default_fast_extra = {
            "xformers": True,
            "gradient_checkpointing": True,
            "cache_latents": True,
            "save_model_as": "safetensors",
        }
        for k, v in default_fast_extra.items():
            if k not in extra_args:
                extra_args[k] = v

        script_name = str(trainer_script_mode or "").strip()
        if script_name not in ("train_network.py", "sdxl_train_network.py"):
            model_hint = base_model.lower()
            script_name = "sdxl_train_network.py" if any(k in model_hint for k in ("sdxl", "noobxl", "xl")) else "train_network.py"

        cfg: Dict[str, Any] = {
            "schema_version": "1.0",
            "trainer": "kohya_ss",
            "mode": "quick_lora",
            "metadata": {
                "created_at": ts,
                "trigger_word": str(trigger_word or "").strip(),
                "speed_preset": speed_preset,
                "notes": preset["notes"],
            },
            "paths": {
                "pretrained_model_name_or_path": base_model,
                "train_data_dir": data_dir,
                "output_dir": out_dir,
            },
            "network": {
                "network_module": "networks.lora",
                "network_dim": int(preset["network_dim"]),
                "network_alpha": int(preset["network_alpha"]),
                "train_unet": bool(preset["train_unet"]),
                "train_text_encoder": bool(preset["train_text_encoder"]),
            },
            "optimization": {
                "optimizer_type": "AdamW8bit",
                "lr_scheduler": "cosine",
                "unet_lr": float(preset["unet_lr"]),
                "text_encoder_lr": float(preset["text_encoder_lr"]),
                "train_batch_size": int(preset["batch_size"]),
                "max_train_steps": int(preset["max_train_steps"]),
                "save_every_n_steps": int(preset["save_every_n_steps"]),
                "seed": 42,
                "mixed_precision": str(preset["mixed_precision"]),
                "resolution": f"{int(preset['resolution'])},{int(preset['resolution'])}",
            },
            "extra_args": extra_args,
        }

        cmd_parts: List[str] = [
            f"accelerate launch {script_name}",
            f"--pretrained_model_name_or_path {_q(base_model)}",
            f"--train_data_dir {_q(data_dir)}",
            f"--output_dir {_q(out_dir)}",
            "--network_module networks.lora",
            f"--network_dim {int(preset['network_dim'])}",
            f"--network_alpha {int(preset['network_alpha'])}",
            "--optimizer_type AdamW8bit",
            "--lr_scheduler cosine",
            f"--unet_lr {float(preset['unet_lr'])}",
            f"--text_encoder_lr {float(preset['text_encoder_lr'])}",
            f"--train_batch_size {int(preset['batch_size'])}",
            f"--max_train_steps {int(preset['max_train_steps'])}",
            f"--save_every_n_steps {int(preset['save_every_n_steps'])}",
            "--seed 42",
            f"--mixed_precision {_q(str(preset['mixed_precision']))}",
            f"--resolution {_q(str(int(preset['resolution'])) + ',' + str(int(preset['resolution'])))}",
        ]

        if bool(preset["train_unet"]) and not bool(preset["train_text_encoder"]):
            cmd_parts.append("--network_train_unet_only")
        if bool(preset["train_text_encoder"]) and not bool(preset["train_unet"]):
            cmd_parts.append("--network_train_text_encoder_only")

        for key, value in extra_args.items():
            k = str(key).strip()
            if not k:
                continue
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{k}")
            elif value is None:
                continue
            else:
                cmd_parts.append(f"--{k} {_q(value)}")

        config_path = ""
        if bool(save_config_to_file):
            os.makedirs(out_dir, exist_ok=True)
            config_path = os.path.join(out_dir, f"{config_prefix}_{ts}.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            cmd_parts.append(f"--config_file {_q(config_path)}")

        launch_command = " ".join(cmd_parts)
        config_json = json.dumps(cfg, ensure_ascii=False, indent=2)
        summary = (
            f"quick_preset={speed_preset} | script={script_name} | "
            f"steps={int(preset['max_train_steps'])} | res={int(preset['resolution'])} | batch={int(preset['batch_size'])}"
        )
        return config_json, launch_command, config_path, summary
