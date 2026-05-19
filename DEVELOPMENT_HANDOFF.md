# ComfyUI Consistency 节点开发交接文档

最后更新：2026-05-12  
项目目录：`E:\Project_python\xiling_anime_Consistency`

## 1. 项目目标（当前定义）

面向二次元/插画角色的 2D 图像生成，构建可控的一致性节点体系，分离两类能力：

1. 角色特征一致性（Character Consistency）
2. 画风一致性（Style Consistency）

并支持组合模式：

3. Character + Style

明确不以“真人脸身份保持”作为重点方向。

---

## 2. 当前已完成内容（可运行骨架）

已完成基础框架与接口定义，包含以下文件：

- `__init__.py`
- `anime_consistency_nodes.py`
- `README.md`

已通过基础语法检查（`python -m py_compile`）。

当前节点可在 ComfyUI 分类 `AnimeConsistency/*` 下显示：

1. `Character Reference Analyzer`
2. `Style Reference Analyzer`
3. `Apply Character Consistency`
4. `Apply Style Consistency`
5. `Character + Style Controller`

---

## 3. 节点模式与产品语义

### Mode A: Character Lock

对应节点：`Apply Character Consistency`  
目标：锁定角色身份相关要素（发型、服装、配饰、主色、体型符号等）。

### Mode B: Style Lock

对应节点：`Apply Style Consistency`  
目标：锁定画风（线稿、阴影、色彩、质感），尽量避免角色内容泄漏。

### Mode C: Character + Style

对应节点：`Character + Style Controller`  
目标：在角色一致性与画风一致性间做联合控制，并提供冲突策略。

---

## 4. 当前实现边界（重要）

当前版本是“框架版”，不是论文复现版：

1. Analyzer 节点已输出结构化 identity（字典）与基础提示信息。
2. Apply 节点当前通过 conditioning 元数据附加控制信息（占位逻辑）。
3. 尚未完成真实后端注入（如 IPAdapter/LoRA/DiT adapter 的实际推理控制）。
4. 所有 `*_backend_placeholder` 返回值都表示“后端待接入”。

换言之：  
“接口和线路图已经搭好，算法内核还需分阶段替换”。

---

## 5. 当前代码结构说明（anime_consistency_nodes.py）

### 5.1 数据结构

- `CharacterIdentity`（角色身份结构）
- `StyleIdentity`（画风身份结构）

### 5.2 工具函数

- `_palette_from_image(...)`：从参考图提取简易主色调（占位版）。
- `_append_note_to_conditioning(...)`：向 conditioning 元数据注入一致性控制信息。

### 5.3 节点职责

1. `CharacterReferenceAnalyzer`
   - 输入：角色参考图、角色名、触发词、开关参数
   - 输出：`CHARACTER_IDENTITY`、角色提示、裁剪占位提示、调色板

2. `StyleReferenceAnalyzer`
   - 输入：风格参考图、region 模式、防泄漏开关、风格 tags
   - 输出：`STYLE_IDENTITY`、风格提示、调色板、线稿/纹理提示

3. `ApplyCharacterConsistency`
   - 核心参数：`identity_strength`、`pose_freedom`、`outfit_strength`、`detail_refine_strength`、`prompt_fidelity`

4. `ApplyStyleConsistency`
   - 核心参数：`style_strength`、`content_leakage_prevention`、`color_strength`、`line_strength`、`shading_strength`

5. `CharacterStyleController`
   - 核心参数：`character_strength`、`style_strength`、`conflict_policy`
   - `conflict_policy`：`character_first | style_first | balanced`

---

## 6. ComfyUI 测试建议（每次迭代后执行）

1. 将目录放入 `ComfyUI/custom_nodes/`。
2. 重启 ComfyUI。
3. 搜索 `AnimeConsistency`，确认 5 节点出现。
4. 做最小链路测试：
   - `CLIP Text Encode -> Apply Character Consistency -> KSampler`
   - `CLIP Text Encode -> Apply Style Consistency -> KSampler`
5. 若报错，优先检查：
   - 自定义类型是否注册（`CHARACTER_IDENTITY`, `STYLE_IDENTITY`）
   - conditioning 元数据结构兼容性
   - ComfyUI 版本差异导致的节点输入输出签名变化

---

## 7. 后续开发路线（按优先级）

## P0（先可用）

1. 引入后端抽象层（backend interface）
   - `CharacterBackend`
   - `StyleBackend`
   - `ComboBackend`
2. 将 placeholder 切换为可配置后端选择（先保留默认占位）。

## P1（角色一致性）

1. 落地 `identity_strength vs pose_freedom` 调度（CoDi 思路）。
2. 增加多 reference 的特征融合策略。
3. 逐步细化 face/hair/outfit/accessory 子权重（CharaConsist 思路）。

## P2（画风一致性）

1. 引入 style 特征缓存（中间特征路径）。
2. 增加内容泄漏抑制机制（Only-Style 思路）。
3. 扩展 style_region 处理（full/bg/character/mask）。

## P3（组合策略）

1. 在 attention 或特征融合层实现冲突策略真正生效。
2. 支持 `character_first/style_first/balanced` 的可解释行为输出。

## P4（训练与复用）

1. 角色 LoRA 与风格 LoRA 分训流程。
2. 增加节点级训练辅助（数据清洗、标签拆分、训练配置模板）。

---

## 8. 与论文方向的对应关系（用于后续实现）

1. AnimeDiff：二次元角色定制与角色属性绑定
2. CharaConsist：细粒度角色一致性控制
3. CoDi：角色一致性与姿态自由的平衡
4. Style Diversity：一致风格的中间特征保持
5. Only-Style：风格迁移中的内容泄漏控制

当前代码仅完成“承载这些方法的工程外壳”。

---

## 9. 下一次对话快速恢复模板（建议直接复制）

可在新对话中直接输入：

1. `项目路径是 E:\Project_python\xiling_anime_Consistency，请先读取 DEVELOPMENT_HANDOFF.md 与 anime_consistency_nodes.py。`
2. `当前阶段从 P0/P1 开始，实现真实 backend 接口与 Character Lock 的第一版。`
3. `每改完一步给出可在 ComfyUI 复现的最小测试工作流。`

---

## 10. 风险与注意事项

1. 不要将“角色一致性”误简化为“脸一致性”。
2. 风格提取必须防止角色内容泄漏。
3. 组合模式必须允许角色与风格冲突时的显式策略选择。
4. 先稳定接口，再逐步替换算法，避免一次性大改导致不可测。
