# ComfyNode Anime Consistency Scaffold

This repository currently provides a framework-first ComfyUI custom node pack for:

- Character Lock
- Style Lock
- Character + Style

The current implementation is intentionally lightweight and does **not** claim paper-level quality yet.
It is designed so we can iteratively replace placeholder logic with methods inspired by:

- AnimeDiff (anime character customization)
- CharaConsist / CoDi (character consistency vs pose freedom)
- Only-Style / Style-Diversity (style consistency with leakage control)

## Files

- `__init__.py`: ComfyUI node export entry.
- `anime_consistency_nodes.py`: compatibility entrypoint (imports registry).
- `registry.py`: node class mappings and display names.
- `nodes_character.py`: character analyzer node (includes backend controls).
- `nodes_style.py`: style analyzer node.
- `nodes_apply.py`: apply/control nodes.
- `nodes_debug.py`: debug viewer node for JSON reports and direct file save.
- `nodes_prompt.py`: strong constraint prompt composer with per-module bool switches.
- `backends.py`: local vision + remote API backend logic.
- `color_features.py`: structured color analysis (hair profile).
- `core_utils.py`: shared helpers.
- `models.py`: dataclass schemas.
- `LORA_NODE_DEVELOPMENT.md`: standalone LoRA training-node development brief.

## Node List

- `Character Reference Analyzer`
- `Style Reference Analyzer`
- `Apply Character Consistency`
- `Apply Style Consistency`
- `Character + Style Controller`

## Current Behavior

- Analyzer nodes output structured identity dictionaries and basic hints.
- Apply nodes attach consistency metadata into conditioning payload metadata.
- No heavy adapter injection is performed yet (placeholder backend state is returned).

### Character Modular Constraints (v1)

`Character Reference Analyzer` now supports decoupled module extraction inputs:

- face
- hair
- outfit
- body
- accessory

Each module can be enabled/disabled and assigned tags independently, then
`Apply Character Consistency` can selectively enable module constraints and tune
per-module strength before metadata is injected into conditioning.

### Character Recognition Backend (v1)

`Character Reference Analyzer` supports two early-stage backend interfaces:

- `local_vision`: local model route (placeholder for ViT-like recognizers)
- `remote_api`: OpenAI-compatible chat-completions style API endpoint

When enabled, backend output is returned as `backend_report_json` and also stored
inside `character_identity.reference_embeddings.backend_report`.

Remote API v1.1 notes:

- `remote_api_url` can be:
  - base URL (example: `https://api.openai.com`)
  - v1 base (example: `https://api.openai.com/v1`)
  - full endpoint (`.../v1/responses` or `.../v1/chat/completions`)
- node will auto-try `responses` first, then `chat/completions` when possible
- backend report includes:
  - `api_kind`, `endpoint_url`, `schema_valid`, `schema_errors`
  - useful for quick debugging in ComfyUI without reading logs

Local vision backend v1.1 notes:

- `local_vision` now returns structured hair color data in `backend_report_json.hair_color_profile`:
  - `dominant_colors[]` with `hex`, `rgb`, `hsv`, `lab`, `weight`, `family_label`
  - `primary_hex`, `primary_family`
- this avoids relying only on coarse color words (for example "purple")
- optional `vit_embedding` is included as a scaffold signal; for true semantic recognition,
  replace with trained/loaded vision weights
- `local_backend_use_pretrained` can be enabled to try loading pretrained ViT weights
  (depends on local environment/model availability)
- `local_backend_checkpoint_path` supports manual local checkpoint loading
  (for timm models via `checkpoint_path`)
  - this is optional; keep empty if you do not want to load a local checkpoint file

## Strong Constraint Prompt (No File Required)

Use `Character Constraint Prompt Composer [Modular v2]`:

1. input `character_identity` from `Character Reference Analyzer`
2. set bool switches per module:
   - `enable_face_constraint`
   - `enable_hair_constraint`
   - `enable_outfit_constraint`
   - `enable_body_constraint`
   - `enable_accessory_constraint`
3. output `composed_prompt` to `CLIPTextEncode.text`

This path does not require local checkpoint files unless you choose to set
`local_backend_checkpoint_path` for local vision embedding.

## Troubleshooting (ComfyUI)

If you cannot see `run_backend` / `backend_mode` in `Character Reference Analyzer`:

1. Ensure this exact folder is the one under `ComfyUI/custom_nodes/`.
2. In ComfyUI manager, use Reload Custom Nodes (or fully restart ComfyUI process).
3. Search the node name with version suffix:
   - `Character Reference Analyzer [Modular v2]`
4. Check ComfyUI startup log for import errors in this node package.

## Debug Workflow (Hair deltaE)

1. `Character Reference Analyzer [Modular v2]`
   - enable `run_backend=true`
   - use `backend_mode=local_vision`
2. `Apply Character Consistency [Modular v2]`
   - connect `character_identity`
   - tune `hair_color_deltae_threshold`
3. `Consistency Debug Viewer [Modular v2]`
   - input `backend_report_json` from analyzer
   - input `debug_report_json` from apply node (config)
   - input `current_image` from `VAEDecode`
   - set `save_to_file=true`, `filename_prefix`, `run_tag`
   - read `summary_text` / `hair_deltae_status` / output `debug_report_json` / `saved_path`

Important:
- Do not connect `VAEDecode` image back into `Apply Character Consistency`.
- Image-based deltaE evaluation now lives in `Consistency Debug Viewer`, avoiding graph cycles.

## Suggested Incremental Roadmap

1. Character branch MVP:
   - add reference feature extraction backend hook
   - implement identity-strength vs pose-freedom scheduling
2. Style branch MVP:
   - add style feature cache extraction
   - add explicit content-leakage penalty/control
3. Combo branch:
   - implement conflict policy at feature/attention routing layer
4. Backend abstraction:
   - IPAdapter backend
   - LoRA backend
   - future DiT/Flux backend

## Installation

Place this folder under ComfyUI custom nodes directory, e.g.:

`ComfyUI/custom_nodes/ComfyNode`

Then restart ComfyUI and search category:

`AnimeConsistency/*`
