# LoRA 训练节点开发文档（稳定传统路线 10-30 图）

最后更新：2026-05-12

## 1. 目标与边界

- 目标：在 Comfy 节点体系内提供“稳定可复现”的角色 LoRA 训练工作流（10-30 张图）。
- 边界：本阶段只做训练链路节点，不改现有推理节点逻辑。
- 原则：先配置可解释、可校验、可复现，再做一键训练。

## 2. 适用路线（首版）

- 路线：传统 LoRA 训练（非 1 图极速方案）。
- 数据规模：10-30 张同角色图，建议 15+ 张。
- 输出：可直接用于推理的 LoRA 权重文件 + 训练元数据日志。

## 3. 建议节点拆分

1. `LoRA Dataset Inspector`
- 输入：数据目录、最小分辨率、最小图数阈值
- 输出：图像数量、尺寸统计、异常文件清单、是否通过校验

2. `LoRA Caption Validator`
- 输入：图像目录、caption 目录（可同目录）
- 输出：缺失 caption 列表、长度统计、触发词覆盖率

3. `LoRA Train Config Builder`
- 输入：底模路径、输出目录、训练超参（rank/alpha/lr/batch/steps 等）
- 输出：训练配置 JSON/TOML、可执行命令字符串

4. `LoRA Train Launcher`（后续）
- 输入：配置文件路径、执行开关
- 输出：训练日志路径、最新 checkpoint 路径

5. `LoRA Artifact Indexer`
- 输入：输出目录
- 输出：最佳 checkpoint、最近 checkpoint、训练摘要

## 4. 首版稳定默认参数（建议）

以下是“先稳定再提效”的默认值（供 SDXL/NoobXL 类底模参考）：

- `network_dim`: 16
- `network_alpha`: 16
- `unet_lr`: 1e-4
- `text_encoder_lr`: 5e-6（或先 0，仅训 UNet）
- `optimizer`: AdamW8bit
- `lr_scheduler`: cosine
- `batch_size`: 1~2（按显存）
- `resolution`: 1024（或 768 作为保守值）
- `max_train_steps`: 1200~2500（按数据量）
- `save_every_n_steps`: 200

## 5. 数据规范（建议）

- 每张图都有对应 caption（同名 `.txt`）。
- caption 含固定触发词（例如 `char_tok`）。
- 场景、构图、姿态有变化，但角色核心特征稳定。
- 发色任务重点：caption 中显式标注发色关键词。

## 6. 与当前项目节点的接口约定

- 训练产物 LoRA 文件路径应可直接接入推理链路中的 LoRA 加载节点。
- 保留 `CharacterReferenceAnalyzer` 与 `CharacterConstraintPromptComposer` 用于推理期约束。
- 训练完成后可用现有 `ConsistencyDebugViewer` 做发色偏差验证（deltaE）。

## 7. 失败模式与防护

1. 数据太少或重复高：过拟合，泛化差。
2. caption 过弱：学不到稳定触发。
3. 学习率过高：细节漂移、色彩崩坏。
4. steps 过长：角色死板、背景污染。

防护建议：
- 先短步数试训（300-500 steps）观察趋势，再加训。
- 保留每阶段 checkpoint，便于回退。

## 8. 下一次对话可直接使用的开场模板

1. `请读取 LORA_NODE_DEVELOPMENT.md，并从 LoRA Dataset Inspector 节点开始实现。`
2. `先完成数据校验与配置文件生成，不要先做自动开训。`
3. `每一步给出可在 Comfy 复现的最小测试方法。`

