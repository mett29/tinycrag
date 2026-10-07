"""Typed data objects exchanged by the PDF retrieval pipeline."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A page-local text fragment with its original document location.

    Attributes:
        source: PDF path relative to the input folder.
        page: One-based page number in the PDF.
        chunk_index: One-based chunk number within the PDF, across all pages.
        text: Extracted text with whitespace normalized by the chunker.
    """

    source: str
    page: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class PdfCorpus:
    """The chunks extracted from one source folder.

    Attributes:
        source: Absolute path of the input folder.
        pdf_count: Number of PDFs parsed, including those without text.
        chunk_size_words: Maximum words per chunk used during extraction.
        chunks: All extracted chunks in document and page order.
    """

    source: Path
    pdf_count: int
    chunk_size_words: int
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class ChunkClassification:
    """Julia's decision and the threshold-based selection for one chunk.

    Attributes:
        chunk: The original text and document location.
        choice: Winning model label, either ``relevant`` or ``irrelevant``.
        relevance_score: Softmax probability assigned to ``relevant``.
        selected: Whether the relevance score meets the requested threshold.
            This can differ from the winning choice at thresholds other than 0.5.
    """

    chunk: Chunk
    choice: str
    relevance_score: float
    selected: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping with flattened chunk metadata."""
        return {
            **asdict(self.chunk),
            "choice": self.choice,
            "relevance_score": self.relevance_score,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class RetrievalTimings:
    """Elapsed wall-clock seconds for one retrieval run.

    Attributes:
        parse_and_chunk: PDF discovery, parsing, OCR, and chunking time.
        model_load: Time spent preparing the model; near zero when already loaded.
        classification: Time spent scoring all chunks against the query.
    """

    parse_and_chunk: float
    model_load: float
    classification: float


@dataclass(frozen=True)
class RetrievalResult:
    """A complete retrieval report containing both selected and rejected chunks.

    Attributes:
        query: User query passed to the classifier.
        corpus: Extracted PDF corpus and its chunking configuration.
        threshold: Inclusive minimum relevance score for selecting a chunk.
        classifications: Decisions for every chunk in source order.
        timings: Elapsed seconds for each pipeline stage.
    """

    query: str
    corpus: PdfCorpus
    threshold: float
    classifications: tuple[ChunkClassification, ...]
    timings: RetrievalTimings

    @property
    def matches(self) -> tuple[ChunkClassification, ...]:
        """Return all selected chunks sorted by decreasing relevance score.

        Equal scores preserve source order. No top-k cutoff is applied.
        """
        return tuple(
            sorted(
                (item for item in self.classifications if item.selected),
                key=lambda item: item.relevance_score,
                reverse=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return the CLI report schema using only JSON-compatible values.

        Chunk locations are flattened, paths are rendered as strings, and
        timings are rounded to milliseconds. Model probabilities remain unrounded.
        """
        matches = self.matches
        return {
            "query": self.query,
            "source": str(self.corpus.source),
            "chunk_size_words": self.corpus.chunk_size_words,
            "threshold": self.threshold,
            "pdf_count": self.corpus.pdf_count,
            "chunk_count": len(self.classifications),
            "match_count": len(matches),
            "timings_seconds": {
                name: round(value, 3) for name, value in asdict(self.timings).items()
            },
            "matches": [item.to_dict() for item in matches],
            "classifications": [item.to_dict() for item in self.classifications],
        }
