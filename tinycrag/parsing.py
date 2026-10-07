"""Recursive PDF discovery and LiteParse extraction with page provenance."""

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from .chunking import WordChunker
from .models import Chunk, PdfCorpus

if TYPE_CHECKING:
    from liteparse import LiteParse

logger = logging.getLogger(__name__)


class PdfLoader:
    """Extract and chunk all PDFs in a folder with local OCR enabled.

    A loader can be reused across folders. Each ``load`` call creates and closes
    its own LiteParse instance unless a parser was supplied by the caller. Empty
    pages are skipped; failed or incomplete parses stop the operation.
    """

    def __init__(
        self,
        chunker: WordChunker | None = None,
        *,
        ocr_language: str = "eng",
        parser: LiteParse | None = None,
    ) -> None:
        """Configure text extraction and chunking.

        Args:
            chunker: Word chunker to use. Defaults to 300 words per chunk.
            ocr_language: Tesseract language code, used when creating LiteParse.
            parser: Optional existing LiteParse instance. The caller owns its
                configuration and lifecycle; this loader never closes it.
        """
        self.chunker = chunker if chunker is not None else WordChunker()
        self.ocr_language = ocr_language
        self._parser = parser

    @staticmethod
    def discover(source: str | Path) -> list[Path]:
        """Find PDFs recursively, including uppercase file extensions.

        Args:
            source: Folder path. Home-directory notation is expanded.

        Returns:
            Absolute PDF paths sorted by path for deterministic document order.

        Raises:
            ValueError: If the source is not a directory or contains no PDFs.
        """
        folder = Path(source).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(
                f"Source folder does not exist or is not a directory: {folder}"
            )
        pdfs = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        if not pdfs:
            raise ValueError(f"No PDFs found in {folder}")
        return pdfs

    def load(self, source: str | Path) -> PdfCorpus:
        """Parse and chunk every PDF, keeping each chunk within one page.

        Args:
            source: Folder to search recursively for PDFs.

        Returns:
            A corpus with relative source paths and one-based page/chunk numbers.

        Raises:
            ValueError: If the folder has no PDFs or no text can be extracted.
            RuntimeError: If a PDF fails to parse or any pages are omitted.

        Notes:
            PDFs containing no text emit a logging warning. Other PDFs are still
            processed. OCR and parser resources are released even if parsing fails.
        """
        folder = Path(source).expanduser().resolve()
        pdfs = self.discover(folder)
        if self._parser is None:
            from liteparse import LiteParse

            context = LiteParse(
                ocr_enabled=True,
                ocr_language=self.ocr_language,
                output_format="text",
                quiet=True,
            )
        else:
            context = nullcontext(self._parser)

        chunks = []
        with context as parser:
            for pdf in pdfs:
                chunks.extend(self._parse_pdf(pdf, folder, parser))
        if not chunks:
            raise ValueError("No text extracted from the PDFs (including OCR)")
        return PdfCorpus(folder, len(pdfs), self.chunker.chunk_size, tuple(chunks))

    def _parse_pdf(self, pdf: Path, folder: Path, parser: LiteParse) -> list[Chunk]:
        """Parse a single PDF, failing explicitly rather than skipping pages."""
        source = str(pdf.relative_to(folder))
        logger.info("Parsing %s", source)
        try:
            result = parser.parse(pdf)
        except Exception as exc:
            raise RuntimeError(f"Could not parse {pdf}: {exc}") from exc
        if result.page_errors or len(result.pages) != result.total_pages:
            raise RuntimeError(f"Incomplete parse of {pdf}; refusing to skip pages")
        chunks = []
        for page in result.pages:
            for text in self.chunker.split(page.text):
                chunks.append(Chunk(source, page.page_num, len(chunks) + 1, text))
        if not chunks:
            logger.warning("No text extracted from %s", pdf)
        return chunks
