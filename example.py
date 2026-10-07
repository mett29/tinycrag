"""Example CLI for retrieving PDF chunks with the tinycrag library."""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from tinycrag import (
    GeminiAnswerer,
    JuliaRetriever,
    PdfLoader,
    RetrievalPipeline,
    WordChunker,
)


def positive_int(value: str) -> int:
    """Parse an argparse value as a strictly positive integer."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def probability(value: str) -> float:
    """Parse an argparse value as a finite probability between 0 and 1."""
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def nonempty(value: str) -> str:
    """Reject empty or whitespace-only argparse values."""
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the example CLI's argument parser without importing model dependencies."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Folder to search recursively for PDFs",
    )
    cli.add_argument(
        "--chunk-size",
        type=positive_int,
        default=300,
        help="Maximum words per chunk (default: 300)",
    )
    cli.add_argument(
        "--query",
        type=nonempty,
        required=True,
        help="Query used to classify every chunk",
    )
    cli.add_argument(
        "--threshold",
        type=probability,
        default=0.5,
        help="Minimum relevance score (default: 0.5)",
    )
    cli.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "Julia-1",
        help="Local Julia-1 checkpoint folder",
    )
    cli.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code (default: eng)",
    )
    cli.add_argument(
        "--answer",
        action="store_true",
        help="Print only Gemini's answer and the chunks sent to it",
    )
    cli.add_argument(
        "--gemini-api",
        action="store_true",
        help="Use the Gemini API (GEMINI_API_KEY) instead of Vertex AI",
    )
    cli.add_argument(
        "--gemini-model",
        default="gemini-3.5-flash-lite",
        help="Gemini model used with --answer (default: gemini-3.5-flash-lite)",
    )
    cli.add_argument(
        "--max-chunks",
        type=positive_int,
        default=None,
        help="Send at most this many top-ranked chunks to Gemini (default: all)",
    )
    return cli


def main(argv: list[str] | None = None) -> int:
    """Run the example CLI and write one JSON report to stdout.

    Args:
        argv: Arguments without the executable name. ``None`` uses ``sys.argv``.

    Returns:
        Zero on success or one after a pipeline error, which is printed to stderr.
        Argument parsing errors raise ``SystemExit`` with argparse's usual code 2.

    Notes:
        This entry point configures logging to stderr. Library components only
        emit log records and leave logging configuration to the host application.
    """
    args = build_parser().parse_args(argv)
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        pipeline = RetrievalPipeline(
            loader=PdfLoader(
                WordChunker(args.chunk_size), ocr_language=args.ocr_language
            ),
            retriever=JuliaRetriever(args.model),
        )
        result = pipeline.retrieve(args.source, args.query, threshold=args.threshold)
        if args.answer:
            answerer = GeminiAnswerer(args.gemini_model, enterprise=not args.gemini_api)
            used = result.matches[: args.max_chunks]
            report = {
                "answer": answerer.answer(result, max_chunks=args.max_chunks),
                "chunks": [item.to_dict() for item in used],
            }
        else:
            report = result.to_dict()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
