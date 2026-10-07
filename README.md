# tinycrag (tiny Classifier for RAG)

A small Python library in `tinycrag/`, with an example CLI in `example.py`. It parses PDFs with [LiteParse](https://github.com/run-llama/liteparse/blob/main/packages/python/README.md), splits their text, and independently classifies **every chunk** for relevance to a query using the local [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) decision model. It retrieves text; it does not generate an answer.

This project is inspired by [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a decision model from TypeSafe. Jev takes a shared piece of context and a typed question (pick one option, rate on a scale, or answer yes/no) and returns probabilities rather than generated text. I wanted to see how that style of judgment fits retrieval-augmented generation (RAG). Instead of embedding documents and ranking them by vector similarity, each chunk is treated as the context and the query as a yes/no question: "is this passage relevant?" The model returns a probability, and a threshold decides what to keep.

## Setup

Python 3.14+ and [uv](https://docs.astral.sh/uv/) are required by this project. The `Julia-1/` directory must contain both the model's Python package and the actual checkpoint weights (not Git LFS pointers). The existing local checkout is used as an editable uv workspace dependency.

For a fresh checkout, download the model first:

```bash
uv run --no-project --with huggingface-hub python -c "from huggingface_hub import snapshot_download; snapshot_download('SupersonicLabs/Julia-1', local_dir='Julia-1')"
uv sync
```

If `Julia-1/` is already present, just run `uv sync`. This installs the project as an editable package, so `from tinycrag import ...` works from other scripts in the environment. It also installs the Ruff development dependency. LiteParse enables local OCR for pages that need it; no parsing API key is required. OCR may download language data on first use, so that run needs network access. Julia inference runs locally on CPU by default.

## Run

```bash
uv run python example.py \
  --source ./sources \
  --chunk-size 300 \
  --query "How does multi-head attention work?" > results.json
```

`--source` searches recursively for PDFs, including uppercase `.PDF` extensions. `--chunk-size` is a maximum number of **whitespace-separated words**, not characters or model tokens; its default is 300. Chunks have no overlap, stay within a page, and preserve document order. The last chunk on a page may be smaller. Whitespace is normalized, and there is no semantic splitting, embedding index, or top-k limit.

Optional flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--threshold` | `0.5` | Keep every chunk whose relevance score is at least this value. Lower it to retrieve more chunks. |
| `--model` | `Julia-1/` next to `example.py` | Local checkpoint directory. |
| `--ocr-language` | `eng` | Tesseract language code for OCR, e.g. `ita`. |

Progress goes to stderr; stdout contains one JSON report:

- `matches`: all selected chunks, sorted by decreasing relevance score.
- `classifications`: every chunk in source order, including rejected chunks for inspection.
- Each chunk includes its relative source path, one-based page and per-document chunk index, text, model `choice`, `relevance_score`, and threshold-based `selected` flag.
- Counts and separate parsing, model loading, and classification timings.

The score is Julia's softmax probability for the `relevant` choice, not a calibrated measure of retrieval quality. `selected` follows your threshold and may differ from the model's winning `choice` when the threshold changes. The model is loaded once per CLI run and reused for all chunks. Parsing and classification are repeated on each CLI run; there is no persistent cache.

Strict encoding is enabled with the original 8,192-token context and 512-token question/options budget. Oversized inputs fail explicitly rather than silently dropping text; reduce the chunk size or shorten the query if needed. Parsing errors also fail the run rather than returning incomplete retrieval results. Blank pages create no chunks; PDFs without extractable text produce a warning, and a folder with no extracted text fails.

## Python API

All public classes are available from `tinycrag`. The components expose typed results and comprehensive docstrings describing arguments, return values, resource ownership, and failure behavior. Importing the package does not load LiteParse, PyTorch, or model weights.

```python
from tinycrag import JuliaRetriever, PdfLoader, RetrievalPipeline, WordChunker

pipeline = RetrievalPipeline(
    loader=PdfLoader(WordChunker(chunk_size=300), ocr_language="eng"),
    retriever=JuliaRetriever(model_path="Julia-1", device="cpu"),
)

result = pipeline.retrieve(
    source="sources",
    query="How does multi-head attention work?",
    threshold=0.5,
)

for match in result.matches:
    print(match.chunk.source, match.chunk.page, match.relevance_score)
    print(match.chunk.text)

report = result.to_dict()  # Same JSON-compatible schema as the example CLI.
```

Reusing the pipeline keeps Julia's engine resident across calls. `retrieve` parses PDFs on each call. To extract a corpus once and classify it against multiple queries, use the components separately:

```python
from tinycrag import JuliaRetriever, PdfLoader, WordChunker

corpus = PdfLoader(WordChunker(300)).load("sources")
retriever = JuliaRetriever("Julia-1")

for query in ["How does multi-head attention work?", "What optimizer was used?"]:
    classifications = retriever.classify(corpus.chunks, query, threshold=0.5)
    selected = [item for item in classifications if item.selected]
    print(query, len(selected))
```

`classify` returns all decisions in input order. `RetrievalResult.matches` returns every selected decision in descending score order. No top-k cutoff is applied. `ChunkClassification.chunk` retains the text, relative source path, page, and chunk index.

| Class | Responsibility |
| --- | --- |
| `WordChunker` | Split text into fixed-size, non-overlapping word chunks. |
| `PdfLoader` | Discover PDFs, run LiteParse/OCR, and return a `PdfCorpus`. |
| `JuliaRetriever` | Lazily load Julia once and classify supplied chunks. |
| `RetrievalPipeline` | Compose extraction and classification into a `RetrievalResult`. |
| `Chunk`, `PdfCorpus`, `ChunkClassification` | Immutable data objects for text and decisions. |
| `RetrievalResult`, `RetrievalTimings` | Ranked matches, all classifications, serialization, and stage timings. |

The library's default model path is `Julia-1/` relative to the working directory. Pass an explicit path when using it from another location. `JuliaRetriever.load()` eagerly prepares and returns the engine; subsequent calls return the same engine without loading it again. Calling `classify` with an empty iterable returns an empty tuple without loading weights.

For custom integration or testing, `PdfLoader(parser=...)` accepts an existing LiteParse instance, and `JuliaRetriever(engine=...)` accepts an object implementing the exported `DecisionEngine` protocol. The caller owns injected dependencies and their configuration; the loader does not close an injected parser. Internally created parsers are closed on success and failure. Library classes emit standard Python logging records; `example.py` configures progress logging to stderr.

## Development

Ruff checks include imports, common Python errors, and Google-style docstrings for the public library and CLI. The downloaded `Julia-1/` dependency is excluded from project linting.

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
```

To apply formatting, run `uv run ruff format .`. The offline tests cover chunk boundaries, PDF discovery and parser ownership, model reuse, threshold selection, report serialization, and CLI behavior without loading real weights.
