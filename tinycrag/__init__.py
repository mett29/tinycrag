"""Reusable PDF retrieval with LiteParse OCR and Julia-1 decision classification.

Use ``RetrievalPipeline`` for a complete folder-to-results workflow, or compose
``PdfLoader``, ``WordChunker``, and ``JuliaRetriever`` independently. Importing
this package does not load OCR dependencies, PyTorch, or model weights.
"""

from .chunking import WordChunker
from .generation import GeminiAnswerer
from .models import (
    Chunk,
    ChunkClassification,
    PdfCorpus,
    RetrievalResult,
    RetrievalTimings,
)
from .parsing import PdfLoader
from .pipeline import RetrievalPipeline
from .retrieval import DecisionEngine, JuliaRetriever

__all__ = [
    "Chunk",
    "ChunkClassification",
    "DecisionEngine",
    "GeminiAnswerer",
    "JuliaRetriever",
    "PdfCorpus",
    "PdfLoader",
    "RetrievalPipeline",
    "RetrievalResult",
    "RetrievalTimings",
    "WordChunker",
]
