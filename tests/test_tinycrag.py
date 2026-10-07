import argparse
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from example import main, probability
from tinycrag import (
    Chunk,
    JuliaRetriever,
    PdfLoader,
    RetrievalPipeline,
    WordChunker,
)


def parsed_document(*texts):
    return SimpleNamespace(
        total_pages=len(texts),
        page_errors=[],
        pages=[
            SimpleNamespace(page_num=i, text=text) for i, text in enumerate(texts, 1)
        ],
    )


def prediction(score):
    return {
        "answers": {
            "relevance": {
                "choice": "relevant" if score >= 0.5 else "irrelevant",
                "probabilities": {"relevant": score, "irrelevant": 1 - score},
            }
        }
    }


class ChunkerTests(unittest.TestCase):
    def test_every_word_is_kept_once_including_tail(self):
        chunker = WordChunker(2)
        self.assertEqual(
            chunker.split("one  two\nthree\tfour five"),
            ["one two", "three four", "five"],
        )
        self.assertEqual(chunker.split(" \n"), [])

    def test_rejects_invalid_chunk_sizes(self):
        for size in [0, -1, 1.5, True, "2"]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                WordChunker(size)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name).resolve()
        self.pdf = self.source / "a.pdf"
        self.pdf.touch()
        self.parser = Mock()
        self.parser.parse.return_value = parsed_document("one two three")
        self.loader = PdfLoader(WordChunker(2), parser=self.parser)

    def test_recursive_discovery_and_uppercase_extension(self):
        nested = self.source / "nested"
        nested.mkdir()
        pdf = nested / "b.PDF"
        pdf.touch()
        (self.source / "ignored.txt").touch()
        self.assertEqual(PdfLoader.discover(self.source), [self.pdf, pdf])
        with self.assertRaisesRegex(ValueError, "not a directory"):
            PdfLoader.discover(self.pdf)

    def test_page_boundaries_blank_pages_and_document_indices(self):
        nested = self.source / "nested"
        nested.mkdir()
        (nested / "b.pdf").touch()
        self.parser.parse.side_effect = [
            parsed_document("a b c", "", "d e"),
            parsed_document("f"),
        ]
        corpus = self.loader.load(self.source)
        self.assertEqual(corpus.pdf_count, 2)
        self.assertEqual(corpus.source, self.source)
        self.assertEqual(corpus.chunk_size_words, 2)
        self.assertEqual(
            corpus.chunks,
            (
                Chunk("a.pdf", 1, 1, "a b"),
                Chunk("a.pdf", 1, 2, "c"),
                Chunk("a.pdf", 3, 3, "d e"),
                Chunk("nested/b.pdf", 1, 1, "f"),
            ),
        )
        self.parser.close.assert_not_called()

    def test_rejects_incomplete_parse(self):
        self.parser.parse.return_value = SimpleNamespace(
            total_pages=2, page_errors=[], pages=[]
        )
        with self.assertRaisesRegex(RuntimeError, "Incomplete parse"):
            self.loader.load(self.source)

    def test_rejects_page_errors(self):
        self.parser.parse.return_value.page_errors = ["broken page"]
        with self.assertRaisesRegex(RuntimeError, "Incomplete parse"):
            self.loader.load(self.source)

    def test_parse_failure_identifies_document(self):
        self.parser.parse.side_effect = ValueError("corrupt PDF")
        with self.assertRaisesRegex(RuntimeError, "a.pdf: corrupt PDF"):
            self.loader.load(self.source)

    def test_rejects_empty_extraction_and_empty_folder(self):
        self.parser.parse.return_value = parsed_document("")
        with (
            self.assertLogs("tinycrag.parsing", level="WARNING"),
            self.assertRaisesRegex(ValueError, "No text extracted"),
        ):
            self.loader.load(self.source)
        with (
            tempfile.TemporaryDirectory() as empty,
            self.assertRaisesRegex(ValueError, "No PDFs"),
        ):
            self.loader.load(empty)

    def test_owned_parser_closes_after_parse_failure(self):
        parser = Mock()
        parser.parse.side_effect = ValueError("corrupt PDF")
        context = Mock(
            __enter__=Mock(return_value=parser), __exit__=Mock(return_value=False)
        )
        factory = Mock(return_value=context)
        module = SimpleNamespace(LiteParse=factory)
        with (
            patch.dict(sys.modules, {"liteparse": module}),
            self.assertRaises(RuntimeError),
        ):
            PdfLoader(ocr_language="ita").load(self.source)
        factory.assert_called_once_with(
            ocr_enabled=True, ocr_language="ita", output_format="text", quiet=True
        )
        context.__exit__.assert_called_once()


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = tuple(
            Chunk("a.pdf", 1, i, text) for i, text in enumerate(["a", "b", "c"], 1)
        )
        self.engine = Mock()
        self.engine.predict.side_effect = [prediction(s) for s in [0.9, 0.4, 0.3]]
        self.retriever = JuliaRetriever(engine=self.engine)

    def test_all_chunks_are_scored_and_threshold_is_inclusive(self):
        results = self.retriever.classify(
            iter(self.chunks), "test query", threshold=0.4
        )
        self.assertEqual(self.engine.predict.call_count, 3)
        self.assertEqual([r.selected for r in results], [True, True, False])
        self.assertEqual([r.chunk for r in results], list(self.chunks))
        self.assertEqual(results[1].choice, "irrelevant")
        for chunk, call in zip(
            self.chunks, self.engine.predict.call_args_list, strict=True
        ):
            self.assertIn("test query", call.kwargs["state"])
            self.assertTrue(call.kwargs["state"].endswith(chunk.text))

    def test_model_is_loaded_only_once_across_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Mock()
            engine.predict.return_value = prediction(0.8)
            factory = Mock(return_value=engine)
            with patch.dict(
                sys.modules, {"julia": SimpleNamespace(load_model=factory)}
            ):
                retriever = JuliaRetriever(folder)
                factory.assert_not_called()
                retriever.classify(self.chunks, "first query")
                retriever.classify(self.chunks, "second query")
                self.assertIs(retriever.load(), engine)
            factory.assert_called_once_with(
                Path(folder).resolve(),
                device="cpu",
                strict_encoding=True,
                max_length=8192,
                head_length=512,
            )
            self.assertEqual(engine.predict.call_count, 6)

    def test_empty_chunks_do_not_load_model(self):
        retriever = JuliaRetriever("/nonexistent-model")
        self.assertEqual(retriever.classify([], "query"), ())
        with self.assertRaisesRegex(ValueError, "checkpoint folder"):
            retriever.classify(self.chunks, "query")

    def test_validation_happens_before_inference(self):
        for query, threshold in [
            (" ", 0.5),
            ("query", float("nan")),
            ("query", float("inf")),
            ("query", -0.1),
            ("query", 1.1),
            ("query", "0.5"),
        ]:
            with (
                self.subTest(query=query, threshold=threshold),
                self.assertRaises(ValueError),
            ):
                self.retriever.classify(self.chunks, query, threshold=threshold)
        self.engine.predict.assert_not_called()

    def test_encoding_failure_identifies_chunk(self):
        self.engine.predict.side_effect = ValueError("lossless context budget exceeded")
        with self.assertRaisesRegex(
            RuntimeError, "a.pdf, page 3, chunk 7.*reduce the chunk size"
        ):
            self.retriever.classify([Chunk("a.pdf", 3, 7, "text")], "query")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name)
        (self.source / "a.pdf").touch()
        self.parser = Mock()
        self.parser.parse.return_value = parsed_document("first second third")
        self.engine = Mock()
        self.engine.predict.side_effect = [prediction(s) for s in [0.7, 0.2, 0.9]]
        self.pipeline = RetrievalPipeline(
            PdfLoader(WordChunker(1), parser=self.parser),
            JuliaRetriever(engine=self.engine),
        )

    def test_complete_report_preserves_schema_order_and_precise_scores(self):
        result = self.pipeline.retrieve(self.source, "query")
        self.assertEqual([r.chunk.text for r in result.matches], ["third", "first"])
        self.assertEqual(
            [r.chunk.text for r in result.classifications], ["first", "second", "third"]
        )
        report = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(report["chunk_count"], 3)
        self.assertEqual(report["match_count"], 2)
        self.assertEqual(report["pdf_count"], 1)
        self.assertEqual(report["matches"][0]["relevance_score"], 0.9)
        self.assertEqual(report["matches"][0]["source"], "a.pdf")
        self.assertEqual(report["matches"][0]["chunk_index"], 3)
        self.assertEqual(
            set(report["timings_seconds"]),
            {"parse_and_chunk", "model_load", "classification"},
        )

    def test_invalid_request_does_not_parse_pdfs(self):
        with self.assertRaises(ValueError):
            self.pipeline.retrieve(self.source, " ")
        self.parser.parse.assert_not_called()

    def test_example_emits_library_report(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("example.RetrievalPipeline", return_value=self.pipeline),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = main(
                ["--source", str(self.source), "--chunk-size", "1", "--query", "query"]
            )
        self.assertEqual(status, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["match_count"], 2)
        self.assertEqual(report["query"], "query")


class CliTests(unittest.TestCase):
    def test_invalid_inputs_and_probabilities(self):
        for value in ["nan", "inf", "-0.1", "1.1"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                probability(value)
        for flags in [["--chunk-size", "0"], ["--query", " "]]:
            with (
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(["--source", ".", "--query", "query", *flags])
            self.assertEqual(raised.exception.code, 2)
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            tempfile.TemporaryDirectory() as folder,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            self.assertEqual(main(["--source", folder, "--query", "query"]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("No PDFs found", stderr.getvalue())

    def test_package_import_does_not_import_ocr_or_model_dependencies(self):
        script = (
            "import sys; import tinycrag; "
            "assert all(name not in sys.modules for name in ['liteparse', 'julia', 'torch'])"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            cwd=tempfile.gettempdir(),
        )


if __name__ == "__main__":
    unittest.main()
