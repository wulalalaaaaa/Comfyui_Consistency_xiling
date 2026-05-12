from __future__ import annotations

from .nodes_apply import ApplyCharacterConsistency, ApplyStyleConsistency, CharacterStyleController
from .nodes_character import CharacterReferenceAnalyzer
from .nodes_debug import ConsistencyDebugViewer
from .nodes_style import StyleReferenceAnalyzer


NODE_CLASS_MAPPINGS = {
    "CharacterReferenceAnalyzer": CharacterReferenceAnalyzer,
    "StyleReferenceAnalyzer": StyleReferenceAnalyzer,
    "ApplyCharacterConsistency": ApplyCharacterConsistency,
    "ApplyStyleConsistency": ApplyStyleConsistency,
    "CharacterStyleController": CharacterStyleController,
    "ConsistencyDebugViewer": ConsistencyDebugViewer,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterReferenceAnalyzer": "Character Reference Analyzer [Modular v2]",
    "StyleReferenceAnalyzer": "Style Reference Analyzer [Modular v2]",
    "ApplyCharacterConsistency": "Apply Character Consistency [Modular v2]",
    "ApplyStyleConsistency": "Apply Style Consistency [Modular v2]",
    "CharacterStyleController": "Character + Style Controller [Modular v2]",
    "ConsistencyDebugViewer": "Consistency Debug Viewer [Modular v2]",
}
