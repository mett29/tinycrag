"""Julia-1 navigation of PageIndex document trees, compared with PageIndex chat.

``PageIndexDocuments`` builds and stores PageIndex trees and runs PageIndex's own
chat. ``JuliaTreeNavigator`` walks a tree top-down with Julia-1, and
``OpenAIAnswerer`` or ``GeminiAnswerer`` answers from the selected sections.
Importing this package does not load PageIndex, PyTorch, or model weights.
"""

from .generation import GeminiAnswerer, OpenAIAnswerer
from .indexing import PageIndexAnswer, PageIndexDocuments, ToolCall
from .models import Passage
from .retrieval import DecisionEngine, JuliaRetriever
from .tree import JuliaTreeNavigator, NodeDecision, TreeNode, TreeRetrievalResult

__all__ = [
    "DecisionEngine",
    "GeminiAnswerer",
    "JuliaRetriever",
    "JuliaTreeNavigator",
    "NodeDecision",
    "OpenAIAnswerer",
    "PageIndexAnswer",
    "PageIndexDocuments",
    "Passage",
    "ToolCall",
    "TreeNode",
    "TreeRetrievalResult",
]
