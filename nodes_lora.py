from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .core_utils import extract_json_object

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


def _iter_image_files(directory: str) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(directory):
        return files
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in _IMAGE_EXTS:
            files.append(path)
    return files


def _read_image_size(path: str) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        return None
    return None


def _q(text: str) -> str:
    return '"' + str(text).replace('"', '\\"') + '"'


def _load_json_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _collect_checkpoints(output_dir: str, recursive: bool = True) -> List[Dict[str, Any]]:
    ckpt_exts = {".safetensors", ".ckpt", ".pt", ".bin"}
    out: List[Dict[str, Any]] = []
    if not output_dir or not os.path.isdir(output_dir):
        return out

    def maybe_add(path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in ckpt_exts:
            return
        try:
            stat = os.stat(path)
        except Exception:
            return
        name = os.path.basename(path)
        step = None
        step_match = re.search(r"(?:^|[^0-9])(\d{3,})(?:[^0-9]|$)", name)
        if step_match:
            try:
                step = int(step_match.group(1))
            except Exception:
                step = None
        out.append(
            {
                "path": path,
                "name": name,
                "size_bytes": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "step": step,
                "is_best_named": ("best" in name.lower()),
            }
        )

    if recursive:
        for root, _, files in os.walk(output_dir):
            for name in files:
                maybe_add(os.path.join(root, name))
    else:
        for name in os.listdir(output_dir):
            maybe_add(os.path.join(output_dir, name))

    out.sort(key=lambda x: float(x.get("mtime", 0.0)), reverse=True)
    return out


def _pick_best_latest(ckpt_infos: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not ckpt_infos:
        return "", ""
    latest_path = str(ckpt_infos[0].get("path", ""))

    named_best = [x for x in ckpt_infos if bool(x.get("is_best_named"))]
    if named_best:
        named_best.sort(key=lambda x: float(x.get("mtime", 0.0)), reverse=True)
        return str(named_best[0].get("path", "")), latest_path

    with_step = [x for x in ckpt_infos if isinstance(x.get("step"), int)]
    if with_step:
        with_step.sort(
            key=lambda x: (
                int(x.get("step", -1)),
                float(x.get("mtime", 0.0)),
            ),
            reverse=True,
        )
        return str(with_step[0].get("path", "")), latest_path

    return latest_path, latest_path


class LoRADatasetInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": "", "multiline": False}),
                "min_width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "min_height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "min_image_count": ("INT", {"default": 10, "min": 1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image_count", "min_resolution_found", "invalid_file_count", "passed", "report_json")
    FUNCTION = "inspect"
    CATEGORY = "AnimeConsistency/LoRA"

    def inspect(self, dataset_dir: str, min_width: int, min_height: int, min_image_count: int):
        data_dir = os.path.abspath(os.path.expanduser(str(dataset_dir or "").strip()))
        min_w_req = max(1, int(min_width))
        min_h_req = max(1, int(min_height))
        min_count_req = max(1, int(min_image_count))

        report: Dict[str, Any] = {
            "status": "ok",
            "dataset_dir": data_dir,
            "requirements": {
                "min_width": min_w_req,
                "min_height": min_h_req,
                "min_image_count": min_count_req,
            },
        }

        if not os.path.isdir(data_dir):
            report["status"] = "dataset_dir_not_found"
            report["passed"] = False
            report["image_count"] = 0
            report["invalid_files"] = []
            report["too_small_files"] = []
            report["unreadable_files"] = []
            return 0, "0x0", 0, False, json.dumps(report, ensure_ascii=False)

        image_files = _iter_image_files(data_dir)
        unreadable_files: List[str] = []
        too_small_files: List[str] = []
        width_list: List[int] = []
        height_list: List[int] = []

        for path in image_files:
            size = _read_image_size(path)
            if size is None:
                unreadable_files.append(path)
                continue
            w, h = size
            width_list.append(w)
            height_list.append(h)
            if w < min_w_req or h < min_h_req:
                too_small_files.append(path)

        image_count = len(image_files)
        invalid_file_count = len(unreadable_files) + len(too_small_files)
        if width_list and height_list:
            min_resolution_found = f"{min(width_list)}x{min(height_list)}"
        else:
            min_resolution_found = "0x0"

        passed = image_count >= min_count_req and invalid_file_count == 0
        if image_count < min_count_req:
            report["status"] = "image_count_below_threshold"
        elif invalid_file_count > 0:
            report["status"] = "dataset_has_invalid_files"
        report["passed"] = bool(passed)
        report["image_count"] = image_count
        report["resolution_stats"] = {
            "min_width_found": min(width_list) if width_list else 0,
            "max_width_found": max(width_list) if width_list else 0,
            "min_height_found": min(height_list) if height_list else 0,
            "max_height_found": max(height_list) if height_list else 0,
        }
        report["too_small_files"] = too_small_files
        report["unreadable_files"] = unreadable_files
        report["invalid_files"] = too_small_files + unreadable_files

        return image_count, min_resolution_found, invalid_file_count, passed, json.dumps(report, ensure_ascii=False)


class LoRACaptionValidator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": ("STRING", {"default": "", "multiline": False}),
                "caption_dir": ("STRING", {"default": "", "multiline": False}),
                "trigger_word": ("STRING", {"default": "char_tok", "multiline": False}),
                "min_caption_chars": ("INT", {"default": 5, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image_count", "missing_caption_count", "trigger_coverage", "passed", "report_json")
    FUNCTION = "validate"
    CATEGORY = "AnimeConsistency/LoRA"

    def validate(self, image_dir: str, caption_dir: str, trigger_word: str, min_caption_chars: int):
        resolved_image_dir = os.path.abspath(os.path.expanduser(str(image_dir or "").strip()))
        resolved_caption_dir = os.path.abspath(
            os.path.expanduser(str(caption_dir or "").strip() or str(image_dir or "").strip())
        )
        trigger = str(trigger_word or "").strip().lower()
        min_chars = max(0, int(min_caption_chars))

        report: Dict[str, Any] = {
            "status": "ok",
            "image_dir": resolved_image_dir,
            "caption_dir": resolved_caption_dir,
            "trigger_word": trigger_word,
            "min_caption_chars": min_chars,
        }

        if not os.path.isdir(resolved_image_dir):
            report["status"] = "image_dir_not_found"
            report["passed"] = False
            report["image_count"] = 0
            report["missing_captions"] = []
            return 0, 0, 0.0, False, json.dumps(report, ensure_ascii=False)

        if not os.path.isdir(resolved_caption_dir):
            report["status"] = "caption_dir_not_found"
            report["passed"] = False
            report["image_count"] = 0
            report["missing_captions"] = []
            return 0, 0, 0.0, False, json.dumps(report, ensure_ascii=False)

        image_files = _iter_image_files(resolved_image_dir)
        missing_captions: List[str] = []
        short_captions: List[str] = []
        existing_caption_count = 0
        trigger_hit_count = 0
        caption_lengths: List[int] = []

        for img_path in image_files:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            cap_path = os.path.join(resolved_caption_dir, f"{stem}.txt")
            if not os.path.isfile(cap_path):
                missing_captions.append(img_path)
                continue

            try:
                with open(cap_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except Exception:
                missing_captions.append(img_path)
                continue

            existing_caption_count += 1
            caption_lengths.append(len(text))
            if len(text) < min_chars:
                short_captions.append(cap_path)
            if trigger and trigger in text.lower():
                trigger_hit_count += 1

        image_count = len(image_files)
        missing_caption_count = len(missing_captions)
        trigger_coverage = (trigger_hit_count / existing_caption_count) if existing_caption_count > 0 else 0.0
        passed = missing_caption_count == 0 and len(short_captions) == 0 and (
            True if not trigger else trigger_coverage >= 0.8
        )

        if missing_caption_count > 0:
            report["status"] = "missing_captions"
        elif len(short_captions) > 0:
            report["status"] = "captions_too_short"
        elif trigger and trigger_coverage < 0.8:
            report["status"] = "low_trigger_coverage"

        report["passed"] = bool(passed)
        report["image_count"] = image_count
        report["missing_caption_count"] = missing_caption_count
        report["existing_caption_count"] = existing_caption_count
        report["trigger_coverage"] = round(float(trigger_coverage), 6)
        report["trigger_hit_count"] = trigger_hit_count
        report["caption_length_stats"] = {
            "min": min(caption_lengths) if caption_lengths else 0,
            "max": max(caption_lengths) if caption_lengths else 0,
            "avg": (sum(caption_lengths) / len(caption_lengths)) if caption_lengths else 0.0,
        }
        report["missing_captions"] = missing_captions
        report["short_captions"] = short_captions

        return image_count, missing_caption_count, float(trigger_coverage), passed, json.dumps(report, ensure_ascii=False)


class LoRATrainConfigBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "", "multiline": False}),
                "dataset_dir": ("STRING", {"default": "", "multiline": False}),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "trigger_word": ("STRING", {"default": "char_tok", "multiline": False}),
                "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "network_dim": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "network_alpha": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "unet_lr": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 1.0, "step": 1e-6}),
                "text_encoder_lr": ("FLOAT", {"default": 5e-6, "min": 0.0, "max": 1.0, "step": 1e-7}),
                "optimizer": (["AdamW8bit", "AdamW", "PagedAdamW8bit", "Adafactor", "Lion"],),
                "lr_scheduler": (["cosine", "constant", "constant_with_warmup", "linear"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "max_train_steps": ("INT", {"default": 1600, "min": 1, "max": 2000000, "step": 1}),
                "save_every_n_steps": ("INT", {"default": 200, "min": 1, "max": 100000, "step": 1}),
                "seed": ("INT", {"default": 42, "min": -9223372036854775807, "max": 9223372036854775807, "step": 1}),
                "train_unet": ("BOOLEAN", {"default": True}),
                "train_text_encoder": ("BOOLEAN", {"default": True}),
                "mixed_precision": (["fp16", "bf16", "no"],),
                "save_config_to_file": ("BOOLEAN", {"default": True}),
                "config_filename_prefix": ("STRING", {"default": "lora_train_config", "multiline": False}),
                "extra_args_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("config_json", "launch_command", "config_path", "summary")
    FUNCTION = "build"
    CATEGORY = "AnimeConsistency/LoRA"

    def build(
        self,
        base_model_path: str,
        dataset_dir: str,
        output_dir: str,
        trigger_word: str,
        resolution: int,
        network_dim: int,
        network_alpha: int,
        unet_lr: float,
        text_encoder_lr: float,
        optimizer: str,
        lr_scheduler: str,
        batch_size: int,
        max_train_steps: int,
        save_every_n_steps: int,
        seed: int,
        train_unet: bool,
        train_text_encoder: bool,
        mixed_precision: str,
        save_config_to_file: bool,
        config_filename_prefix: str,
        extra_args_json: str,
    ):
        base_model = os.path.abspath(os.path.expanduser(str(base_model_path or "").strip()))
        data_dir = os.path.abspath(os.path.expanduser(str(dataset_dir or "").strip()))
        user_output_dir = str(output_dir or "").strip()
        if user_output_dir:
            out_dir = os.path.abspath(os.path.expanduser(user_output_dir))
        else:
            out_dir = os.path.join(_resolve_output_dir(), "anime_consistency", "lora_train_runs")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_prefix = _safe_token(config_filename_prefix, "lora_train_config")
        config_path = ""

        extra_args = extract_json_object(extra_args_json)
        if not isinstance(extra_args, dict):
            extra_args = {}

        sdxl_lora_config: Dict[str, Any] = {
            "schema_version": "1.0",
            "trainer": "kohya_ss",
            "mode": "traditional_lora_stable",
            "metadata": {
                "created_at": ts,
                "trigger_word": str(trigger_word or "").strip(),
                "notes": "Build first, run later. Stable defaults for 10-30 images.",
            },
            "paths": {
                "pretrained_model_name_or_path": base_model,
                "train_data_dir": data_dir,
                "output_dir": out_dir,
            },
            "network": {
                "network_module": "networks.lora",
                "network_dim": int(network_dim),
                "network_alpha": int(network_alpha),
                "train_unet": bool(train_unet),
                "train_text_encoder": bool(train_text_encoder),
            },
            "optimization": {
                "optimizer_type": optimizer,
                "lr_scheduler": lr_scheduler,
                "unet_lr": float(unet_lr),
                "text_encoder_lr": float(text_encoder_lr),
                "train_batch_size": int(batch_size),
                "max_train_steps": int(max_train_steps),
                "save_every_n_steps": int(save_every_n_steps),
                "seed": int(seed),
                "mixed_precision": mixed_precision,
                "resolution": f"{int(resolution)},{int(resolution)}",
            },
            "extra_args": extra_args,
        }

        cmd_parts: List[str] = [
            "accelerate launch train_network.py",
            f"--pretrained_model_name_or_path {_q(base_model)}",
            f"--train_data_dir {_q(data_dir)}",
            f"--output_dir {_q(out_dir)}",
            "--network_module networks.lora",
            f"--network_dim {int(network_dim)}",
            f"--network_alpha {int(network_alpha)}",
            f"--optimizer_type {_q(optimizer)}",
            f"--lr_scheduler {_q(lr_scheduler)}",
            f"--unet_lr {float(unet_lr)}",
            f"--text_encoder_lr {float(text_encoder_lr)}",
            f"--train_batch_size {int(batch_size)}",
            f"--max_train_steps {int(max_train_steps)}",
            f"--save_every_n_steps {int(save_every_n_steps)}",
            f"--seed {int(seed)}",
            f"--mixed_precision {_q(mixed_precision)}",
            f"--resolution {_q(f'{int(resolution)},{int(resolution)}')}",
        ]

        if train_unet and not train_text_encoder:
            cmd_parts.append("--network_train_unet_only")
        if train_text_encoder and not train_unet:
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

        if bool(save_config_to_file):
            os.makedirs(out_dir, exist_ok=True)
            config_filename = f"{config_prefix}_{ts}.json"
            config_path = os.path.join(out_dir, config_filename)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(sdxl_lora_config, f, ensure_ascii=False, indent=2)
            cmd_parts.append(f"--config_file {_q(config_path)}")

        launch_command = " ".join(cmd_parts)
        config_json = json.dumps(sdxl_lora_config, ensure_ascii=False, indent=2)

        summary_parts = [
            "LoRA config ready",
            f"optimizer={optimizer}",
            f"steps={int(max_train_steps)}",
            f"batch={int(batch_size)}",
            f"resolution={int(resolution)}",
            f"saved={'yes' if bool(save_config_to_file) else 'no'}",
        ]
        summary = " | ".join(summary_parts)

        return config_json, launch_command, config_path, summary


class LoRATrainLauncher:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_json": ("STRING", {"default": "{}", "multiline": True}),
                "config_path": ("STRING", {"default": "", "multiline": False}),
                "launch_command": ("STRING", {"default": "", "multiline": True}),
                "execute": ("BOOLEAN", {"default": False}),
                "working_dir": ("STRING", {"default": "", "multiline": False}),
                "log_dir": ("STRING", {"default": "", "multiline": False}),
                "log_filename_prefix": ("STRING", {"default": "lora_train", "multiline": False}),
                "timeout_s": ("INT", {"default": 3600, "min": 10, "max": 604800, "step": 1}),
                "recursive_scan_ckpt": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "launch_command", "train_log_path", "latest_checkpoint", "launcher_report_json")
    FUNCTION = "launch"
    CATEGORY = "AnimeConsistency/LoRA"

    def launch(
        self,
        config_json: str,
        config_path: str,
        launch_command: str,
        execute: bool,
        working_dir: str,
        log_dir: str,
        log_filename_prefix: str,
        timeout_s: int,
        recursive_scan_ckpt: bool,
    ):
        parsed_cfg = extract_json_object(config_json)
        resolved_cfg_path = os.path.abspath(os.path.expanduser(str(config_path or "").strip())) if config_path else ""
        if not parsed_cfg and resolved_cfg_path:
            parsed_cfg = _load_json_file(resolved_cfg_path)

        paths_cfg = parsed_cfg.get("paths", {}) if isinstance(parsed_cfg, dict) else {}
        if not isinstance(paths_cfg, dict):
            paths_cfg = {}
        output_dir = os.path.abspath(
            os.path.expanduser(
                str(paths_cfg.get("output_dir", "")).strip() or os.path.join(_resolve_output_dir(), "anime_consistency", "lora_train_runs")
            )
        )

        resolved_workdir = (
            os.path.abspath(os.path.expanduser(str(working_dir or "").strip()))
            if str(working_dir or "").strip()
            else os.getcwd()
        )
        resolved_log_dir = (
            os.path.abspath(os.path.expanduser(str(log_dir or "").strip()))
            if str(log_dir or "").strip()
            else os.path.join(output_dir, "logs")
        )

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(resolved_log_dir, exist_ok=True)

        cmd = str(launch_command or "").strip()
        temp_cfg_path = ""
        if not cmd:
            if resolved_cfg_path and os.path.isfile(resolved_cfg_path):
                temp_cfg_path = resolved_cfg_path
            elif parsed_cfg:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_cfg_path = os.path.join(output_dir, f"lora_runtime_config_{ts}.json")
                with open(temp_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(parsed_cfg, f, ensure_ascii=False, indent=2)
            if temp_cfg_path:
                cmd = f"accelerate launch train_network.py --config_file {_q(temp_cfg_path)}"

        if not cmd:
            report = {
                "status": "no_command",
                "reason": "No launch_command and no valid config found.",
                "config_path": resolved_cfg_path,
                "temp_config_path": temp_cfg_path,
            }
            return "no_command", "", "", "", json.dumps(report, ensure_ascii=False)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_prefix = _safe_token(log_filename_prefix, "lora_train")
        train_log_path = os.path.join(resolved_log_dir, f"{log_prefix}_{ts}.log")

        report: Dict[str, Any] = {
            "status": "ready",
            "execute": bool(execute),
            "working_dir": resolved_workdir,
            "output_dir": output_dir,
            "log_dir": resolved_log_dir,
            "config_path": resolved_cfg_path,
            "temp_config_path": temp_cfg_path,
            "launch_command": cmd,
        }

        if not bool(execute):
            ckpt_infos = _collect_checkpoints(output_dir, recursive=bool(recursive_scan_ckpt))
            _, latest_ckpt = _pick_best_latest(ckpt_infos)
            report["status"] = "preview_only"
            report["latest_checkpoint"] = latest_ckpt
            report["checkpoint_count"] = len(ckpt_infos)
            return "preview_only", cmd, "", latest_ckpt, json.dumps(report, ensure_ascii=False)

        exit_code = None
        status = "completed"
        try:
            with open(train_log_path, "w", encoding="utf-8", errors="replace") as logf:
                logf.write(f"[LoRATrainLauncher] start={datetime.now().isoformat()}\n")
                logf.write(f"[LoRATrainLauncher] workdir={resolved_workdir}\n")
                logf.write(f"[LoRATrainLauncher] command={cmd}\n\n")
                logf.flush()

                completed = subprocess.run(
                    cmd,
                    cwd=resolved_workdir,
                    shell=True,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=max(10, int(timeout_s)),
                    check=False,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                exit_code = int(completed.returncode)
                status = "completed" if exit_code == 0 else "failed"
                logf.write(f"\n[LoRATrainLauncher] end={datetime.now().isoformat()}\n")
                logf.write(f"[LoRATrainLauncher] return_code={exit_code}\n")
        except subprocess.TimeoutExpired:
            status = "timeout"
            try:
                with open(train_log_path, "a", encoding="utf-8", errors="replace") as logf:
                    logf.write(f"\n[LoRATrainLauncher] timeout after {int(timeout_s)} seconds\n")
            except Exception:
                pass
        except Exception as e:
            status = "error"
            try:
                with open(train_log_path, "a", encoding="utf-8", errors="replace") as logf:
                    logf.write(f"\n[LoRATrainLauncher] exception: {repr(e)}\n")
            except Exception:
                pass

        ckpt_infos = _collect_checkpoints(output_dir, recursive=bool(recursive_scan_ckpt))
        _, latest_ckpt = _pick_best_latest(ckpt_infos)

        report["status"] = status
        report["return_code"] = exit_code
        report["train_log_path"] = train_log_path
        report["latest_checkpoint"] = latest_ckpt
        report["checkpoint_count"] = len(ckpt_infos)

        return status, cmd, train_log_path, latest_ckpt, json.dumps(report, ensure_ascii=False)


class LoRAArtifactIndexer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "recursive_scan": ("BOOLEAN", {"default": True}),
                "max_list_items": ("INT", {"default": 30, "min": 1, "max": 5000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("best_checkpoint", "latest_checkpoint", "checkpoint_count", "summary", "report_json")
    FUNCTION = "index"
    CATEGORY = "AnimeConsistency/LoRA"

    def index(self, output_dir: str, recursive_scan: bool, max_list_items: int):
        resolved_output_dir = os.path.abspath(os.path.expanduser(str(output_dir or "").strip()))
        report: Dict[str, Any] = {
            "status": "ok",
            "output_dir": resolved_output_dir,
            "recursive_scan": bool(recursive_scan),
        }
        if not os.path.isdir(resolved_output_dir):
            report["status"] = "output_dir_not_found"
            report["checkpoint_count"] = 0
            report["best_checkpoint"] = ""
            report["latest_checkpoint"] = ""
            return "", "", 0, "No output directory found.", json.dumps(report, ensure_ascii=False)

        ckpt_infos = _collect_checkpoints(resolved_output_dir, recursive=bool(recursive_scan))
        best_checkpoint, latest_checkpoint = _pick_best_latest(ckpt_infos)
        checkpoint_count = len(ckpt_infos)

        cap = max(1, int(max_list_items))
        listed = ckpt_infos[:cap]
        report["checkpoint_count"] = checkpoint_count
        report["best_checkpoint"] = best_checkpoint
        report["latest_checkpoint"] = latest_checkpoint
        report["checkpoints"] = listed

        latest_name = os.path.basename(latest_checkpoint) if latest_checkpoint else "none"
        best_name = os.path.basename(best_checkpoint) if best_checkpoint else "none"
        summary = f"checkpoints={checkpoint_count} | best={best_name} | latest={latest_name}"

        return best_checkpoint, latest_checkpoint, checkpoint_count, summary, json.dumps(report, ensure_ascii=False)
