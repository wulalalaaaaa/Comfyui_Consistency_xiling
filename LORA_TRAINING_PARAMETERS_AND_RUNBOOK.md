# LoRA 训练参数与一遍跑通文档（Traditional v1）

最后更新：2026-05-14

## 1. 适用范围

- 目标：在 ComfyUI 里用本仓库 LoRA 节点完成一次可复现训练。
- 路线：`LoRA Dataset Inspector -> LoRA Caption Validator -> LoRA Train Config Builder -> LoRA Train Launcher -> LoRA Artifact Indexer`
- 训练后端：`sd-scripts`（例如 `E:\draw\sd-scripts`）。

## 2. 数据与标注要求

- 每张图片一个同名 `.txt`，UTF-8 编码。
- caption 推荐逗号分隔，例如：`xiling, silver hair, blue eyes, school uniform, upper body`
- 触发词（例如 `xiling`）建议每张都出现。

## 3. 16GB 显存推荐参数（SDXL/NoobXL）

### 3.1 Dataset Inspector

- `dataset_dir`: 你的训练图目录
- `min_width`: `768`（保守）或 `1024`（更严格）
- `min_height`: `768`（保守）或 `1024`（更严格）
- `min_image_count`: `20`

通过标准：`passed=true`

### 3.2 Caption Validator

- `image_dir`: 训练图目录
- `caption_dir`: caption 目录（同目录就填同一路径）
- `trigger_word`: 你的触发词（如 `xiling`）
- `min_caption_chars`: `8`

通过标准：
- `missing_caption_count=0`
- `trigger_coverage` 尽量接近 `1.0`
- `passed=true`

### 3.3 Train Config Builder（首轮试训）

- `resolution`: `1024`（OOM 时改 `768`）
- `network_dim`: `16`
- `network_alpha`: `16`
- `unet_lr`: `1e-4`
- `text_encoder_lr`: `5e-6`（显存紧张可配合只训 UNet）
- `optimizer`: `AdamW8bit`
- `lr_scheduler`: `cosine`
- `batch_size`: `1`（16GB 推荐先从 1 开始）
- `max_train_steps`: `500`（试训）
- `save_every_n_steps`: `200`
- `train_unet`: `true`
- `train_text_encoder`: `true`（OOM 时改 `false`）
- `mixed_precision`: `fp16`
- `save_config_to_file`: `true`
- `extra_args_json` 推荐：
```json
{
  "xformers": true,
  "gradient_checkpointing": true,
  "cache_latents": true,
  "save_model_as": "safetensors"
}
```

### 3.4 Train Launcher

- `working_dir`: `E:\draw\sd-scripts`
- `command_mode`: `rebuild_from_config`
- `trainer_script_mode`: `sdxl_train_network.py`（SDXL）
- `trainer_script_path`: 留空（默认从 `working_dir` 拼）
- `execute`: 先 `false` 预览，再改 `true` 开训
- `timeout_s`: `21600`

说明：
- 预览阶段可先看 `launch_command` 是否是你预期脚本。
- 真跑时检查 `train_log_path` 输出。

### 3.5 Artifact Indexer

- `output_dir`: 与 Builder 一致
- `recursive_scan`: `true`
- `max_list_items`: `50`

查看：
- `best_checkpoint`
- `latest_checkpoint`
- `checkpoint_count`

## 4. 一遍跑通步骤（最小实践）

1. 运行 `LoRA Dataset Inspector`，确认 `passed=true`。
2. 运行 `LoRA Caption Validator`，确认 `passed=true`。
3. 运行 `LoRA Train Config Builder`，得到 `config_json/config_path/launch_command`。
4. 在 `LoRA Train Launcher` 设 `execute=false` 先预览。
5. 确认命令无误后把 `execute=true`，启动训练。
6. 训练结束后运行 `LoRA Artifact Indexer`，定位最新和最佳模型。

## 5. 常见问题

- 问：不连 Inspector/Validator 到 Launcher 能训吗？  
  答：能，它们是校验节点，不是强制门控。

- 问：为什么找不到 `train_network.py`？  
  答：它不在本仓库，属于 `sd-scripts` 外部训练仓库。

- 问：16GB 爆显存怎么办？  
  答：按顺序降负载：`batch_size=1` -> `resolution=768` -> `train_text_encoder=false`。

