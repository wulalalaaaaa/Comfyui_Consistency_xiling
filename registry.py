from __future__ import annotations

from .nodes_apply import ApplyCharacterConsistency, ApplyStyleConsistency, CharacterStyleController
from .nodes_character import CharacterReferenceAnalyzer
from .nodes_debug import ConsistencyDebugViewer
from .nodes_lora import (
    LoRAArtifactIndexer,
    LoRACaptionValidator,
    LoRADatasetInspector,
    LoRATrainConfigBuilder,
    LoRATrainLauncher,
)
from .nodes_prompt import CharacterConstraintPromptComposer
from .nodes_style import StyleReferenceAnalyzer


NODE_CLASS_MAPPINGS = {
    "CharacterReferenceAnalyzer": CharacterReferenceAnalyzer,
    "StyleReferenceAnalyzer": StyleReferenceAnalyzer,
    "ApplyCharacterConsistency": ApplyCharacterConsistency,
    "ApplyStyleConsistency": ApplyStyleConsistency,
    "CharacterStyleController": CharacterStyleController,
    "ConsistencyDebugViewer": ConsistencyDebugViewer,
    "CharacterConstraintPromptComposer": CharacterConstraintPromptComposer,
    "LoRADatasetInspector": LoRADatasetInspector,
    "LoRACaptionValidator": LoRACaptionValidator,
    "LoRATrainConfigBuilder": LoRATrainConfigBuilder,
    "LoRATrainLauncher": LoRATrainLauncher,
    "LoRAArtifactIndexer": LoRAArtifactIndexer,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterReferenceAnalyzer": "Character Reference Analyzer [Modular v2]",
    "StyleReferenceAnalyzer": "Style Reference Analyzer [Modular v2]",
    "ApplyCharacterConsistency": "Apply Character Consistency [Modular v2]",
    "ApplyStyleConsistency": "Apply Style Consistency [Modular v2]",
    "CharacterStyleController": "Character + Style Controller [Modular v2]",
    "ConsistencyDebugViewer": "Consistency Debug Viewer [Modular v2]",
    "CharacterConstraintPromptComposer": "Character Constraint Prompt Composer [Modular v2]",
    "LoRADatasetInspector": "LoRA Dataset Inspector [Traditional v1]",
    "LoRACaptionValidator": "LoRA Caption Validator [Traditional v1]",
    "LoRATrainConfigBuilder": "LoRA Train Config Builder [Traditional v1]",
    "LoRATrainLauncher": "LoRA Train Launcher [Traditional v1]",
    "LoRAArtifactIndexer": "LoRA Artifact Indexer [Traditional v1]",
}
