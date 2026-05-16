# LoRA 快速训练节点方案（Fast v1）

最后更新：2026-05-14

## 1. 目标

- 在传统稳定流程之外，提供一套“快速出结果”路线。
- 支持：
  - 单图最快模式（1 图也能训）
  - 少图快速模式（2-8 图）
  - 少图平衡模式（4-20 图）

## 2. 风险声明

- 单图训练极易过拟合，仅适合“先出可用原型”。
- 一图模型泛化通常很差，建议后续补图转传统流程复训。

## 3. 新节点

### 3.1 `LoRA Quick Dataset Preparer [Fast v1]`

作用：
- 接收单图或小图集，自动复制扩增成可训练目录。
- 自动生成同名 caption（可保留原 caption，缺失时用模板补齐）。

关键参数：
- `source_image_path`: 图片文件或目录
- `prepared_dataset_dir`: 输出训练集目录
- `trigger_word`: 触发词
- `caption_template`: 缺失 caption 时使用
- `duplicate_count`: 每张图复制次数（单图建议 16-32）
- `strict_single_image`: 开启后要求输入必须是 1 张

### 3.2 `LoRA Quick Config Builder [Fast v1]`

作用：
- 一键生成快训配置与命令。
- 内置 3 档预设并自动加速参数（xformers、gradient checkpointing、cache_latents）。

预设说明：
- `single_image_fastest`
  - 1 图应急，默认约 180 steps，风险最高
- `few_image_fast`
  - 2-8 图快速出图，默认约 350 steps
- `few_image_balanced`
  - 4-20 图平衡质量，默认约 700 steps

## 4. 推荐最小链路（单图）

1. `LoRA Quick Dataset Preparer`
   - `strict_single_image=true`
   - `duplicate_count=24`
2. `LoRA Quick Config Builder`
   - `speed_preset=single_image_fastest`
   - SDXL 底模建议 `trainer_script_mode=sdxl_train_network.py`
3. `LoRA Train Launcher [Traditional v1]`
   - `command_mode=rebuild_from_config`
   - `execute=false` 先预览，再 `true` 启动训练
4. `LoRA Artifact Indexer [Traditional v1]`
   - 查看 `latest_checkpoint`

## 5. 什么时候从快训切回传统流程

- 当你确认角色方向对了，建议切回传统流程并做：
  - 数据补充到 10-30 图
  - 降低学习率
  - 增加训练步数与 checkpoint 对比

