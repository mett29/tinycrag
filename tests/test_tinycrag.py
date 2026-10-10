import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tinycrag import (
    JuliaRetriever,
    JuliaTreeNavigator,
    OpenAIAnswerer,
    Passage,
    TreeNode,
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


class RetrieverTests(unittest.TestCase):
    def test_score_returns_choice_and_probability(self):
        engine = Mock()
        engine.predict.return_value = prediction(0.3)
        choice, score = JuliaRetriever(engine=engine).score("query", "evidence")
        self.assertEqual((choice, score), ("irrelevant", 0.3))
        state = engine.predict.call_args.kwargs["state"]
        self.assertEqual(state, "User query:\nquery\n\nText section:\nevidence")

    def test_model_is_loaded_only_once(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Mock()
            engine.predict.return_value = prediction(0.8)
            factory = Mock(return_value=engine)
            with patch.dict(
                sys.modules, {"julia": SimpleNamespace(load_model=factory)}
            ):
                retriever = JuliaRetriever(folder)
                factory.assert_not_called()
                retriever.score("first query", "a")
                retriever.score("second query", "b")
                self.assertIs(retriever.load(), engine)
            factory.assert_called_once_with(
                Path(folder).resolve(),
                device="cpu",
                strict_encoding=True,
                max_length=8192,
                head_length=512,
            )

    def test_missing_checkpoint_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "checkpoint folder"):
            JuliaRetriever("/nonexistent-model").score("query", "text")


class OpenAIAnswererTests(unittest.TestCase):
    def test_passages_are_tagged_in_the_prompt(self):
        answerer = OpenAIAnswerer("gpt-6-luna")
        answerer._client = Mock()
        answerer._client.responses.create.return_value.output_text = (
            "It works [a.pdf:0001]"
        )
        answer = answerer.answer("q?", [Passage("a.pdf:0001", "Intro, page 1", "text")])
        self.assertEqual(answer, "It works [a.pdf:0001]")
        kwargs = answerer._client.responses.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-6-luna")
        self.assertEqual(
            kwargs["input"],
            "Context:\n[a.pdf:0001] (Intro, page 1)\ntext\n\nQuestion: q?",
        )

    def test_no_passages_skips_the_model(self):
        answerer = OpenAIAnswerer()
        answerer._client = Mock()
        self.assertIn("No relevant passages", answerer.answer("q?", []))
        answerer._client.responses.create.assert_not_called()


class TreeNavigatorTests(unittest.TestCase):
    def setUp(self):
        self.tree = TreeNode.from_pageindex(
            {
                "title": "Model",
                "node_id": "0001",
                "start_index": 2,
                "end_index": 5,
                "summary": "model summary",
                "text": "model intro",
                "nodes": [
                    {
                        "title": "Attention",
                        "node_id": "0002",
                        "start_index": 3,
                        "end_index": 3,
                        "summary": "attention summary",
                        "text": "attention text",
                    },
                    {
                        "title": "Training",
                        "node_id": "0003",
                        "start_index": 4,
                        "end_index": 5,
                        "summary": "training summary",
                        "text": "",
                        "nodes": [
                            {
                                "title": "Optimizer",
                                "node_id": "0004",
                                "start_index": 5,
                                "end_index": 5,
                                "text": "adam",
                            }
                        ],
                    },
                ],
            },
            "paper.pdf",
        )
        self.other = TreeNode.from_pageindex(
            {"title": "Results", "node_id": "0005", "start_index": 6, "text": "bleu"},
            "paper.pdf",
        )

    def navigate(self, scores):
        engine = Mock()
        engine.predict.side_effect = [prediction(s) for s in scores]
        navigator = JuliaTreeNavigator(JuliaRetriever(engine=engine))
        result = navigator.navigate((self.tree, self.other), "attention?")
        return result, engine

    def test_rejected_nodes_prune_their_subtree(self):
        # Level 0: Model, Results. Level 1: Attention, Training.
        result, engine = self.navigate([0.9, 0.2, 0.8, 0.1])
        self.assertEqual(
            [d.node.node_id for d in result.decisions], ["0001", "0005", "0002", "0003"]
        )
        self.assertEqual(engine.predict.call_count, 4)
        state = engine.predict.call_args_list[0].kwargs["state"]
        self.assertIn(
            "Title: Model\nLocation: pages 2-5\nSummary: model summary", state
        )
        self.assertIn("Text section:", state)

    def test_selected_own_text_becomes_passages_in_document_order(self):
        result, _ = self.navigate([0.9, 0.7, 0.3, 0.8, 0.95])
        passages = result.passages()
        self.assertEqual(
            [p.tag for p in passages],
            ["paper.pdf:0001", "paper.pdf:0004", "paper.pdf:0005"],
        )
        self.assertEqual(passages[1].location, "Optimizer, page 5")
        self.assertEqual(passages[1].text, "adam")

    def test_node_without_summary_is_judged_on_its_text(self):
        engine = Mock()
        engine.predict.return_value = prediction(0.1)
        JuliaTreeNavigator(JuliaRetriever(engine=engine)).navigate((self.other,), "q")
        self.assertIn("Summary: bleu", engine.predict.call_args.kwargs["state"])


if __name__ == "__main__":
    unittest.main()
