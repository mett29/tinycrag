"""Composition of PDF extraction, chunking, and Julia-1 retrieval."""

from pathlib import Path
from time import perf_counter

from .models import RetrievalResult, RetrievalTimings
from .parsing import PdfLoader
from .retrieval import JuliaRetriever


class RetrievalPipeline:
    """Run folder-to-chunk retrieval with reusable, independently usable components.

    Reusing this pipeline keeps Julia's model resident. PDF parsing is performed
    on each call; use ``PdfLoader.load`` and ``JuliaRetriever.classify`` separately
    to reuse an extracted corpus for multiple queries.
    """

    def __init__(
        self,
        loader: PdfLoader | None = None,
        retriever: JuliaRetriever | None = None,
    ) -> None:
        """Compose the loader and retriever.

        Args:
            loader: PDF loader. Defaults to English OCR and 300-word chunks.
            retriever: Classifier. Defaults to the local ``Julia-1`` folder on CPU.
        """
        self.loader = loader if loader is not None else PdfLoader()
        self.retriever = retriever if retriever is not None else JuliaRetriever()

    def retrieve(
        self,
        source: str | Path,
        query: str,
        *,
        threshold: float = 0.5,
    ) -> RetrievalResult:
        """Extract all PDF chunks and classify their relevance to the query.

        Args:
            source: Input folder, searched recursively for PDFs.
            query: Nonempty user question or search request.
            threshold: Inclusive minimum probability for selecting a chunk.

        Returns:
            A typed report exposing ranked ``matches``, all ``classifications``,
            the extracted corpus, and stage timings. ``to_dict`` serializes it.

        Raises:
            ValueError: If arguments are invalid, no PDFs/text are found, or the
                model path is missing.
            RuntimeError: If parsing is incomplete or classification fails.
            Exception: If model loading or a dependency fails.
        """
        self.retriever.validate_request(query, threshold)
        started = perf_counter()
        corpus = self.loader.load(source)
        parsed_at = perf_counter()
        self.retriever.load()
        loaded_at = perf_counter()
        classifications = self.retriever.classify(
            corpus.chunks, query, threshold=threshold
        )
        finished = perf_counter()
        return RetrievalResult(
            query=query,
            corpus=corpus,
            threshold=threshold,
            classifications=classifications,
            timings=RetrievalTimings(
                parse_and_chunk=parsed_at - started,
                model_load=loaded_at - parsed_at,
                classification=finished - loaded_at,
            ),
        )
